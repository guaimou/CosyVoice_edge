from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import onnxruntime
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
OUTPUT_DIR = PROJECT_ROOT / "output"
UPSTREAM_ROOT = THIRD_PARTY_DIR / "CosyVoice_edge"
DEFAULT_MODEL_DIR = PRETRAINED_DIR / "CosyVoice-300M"
DEFAULT_PROMPT_WAV = PROJECT_ROOT / "assets" / "zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
DEFAULT_TEXT = "你好，这是一次Flow Decoder输入导出检查。"
DEFAULT_OUT_DIR = OUTPUT_DIR / "flow_decoder_case"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.mask import make_pad_mask


_ORIGINAL_INFERENCE_SESSION = onnxruntime.InferenceSession


def cpu_inference_session(path, sess_options=None, providers=None, provider_options=None, **kwargs):
    return _ORIGINAL_INFERENCE_SESSION(
        path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
        provider_options=provider_options,
        **kwargs,
    )


def describe_array(name: str, value: np.ndarray) -> str:
    return f"{name}\tshape={list(value.shape)}\tdtype={value.dtype}"


def ensure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_llm_tokens(model, model_input: dict) -> torch.Tensor:
    llm = model.model.llm
    device = model.model.device
    with model.model.llm_context, torch.cuda.amp.autocast(model.model.fp16 is True and hasattr(llm, "vllm") is False):
        token_generator = llm.inference(
            text=model_input["text"].to(device),
            text_len=torch.tensor([model_input["text"].shape[1]], dtype=torch.int32).to(device),
            prompt_text=model_input["prompt_text"].to(device),
            prompt_text_len=torch.tensor([model_input["prompt_text"].shape[1]], dtype=torch.int32).to(device),
            prompt_speech_token=model_input["llm_prompt_speech_token"].to(device),
            prompt_speech_token_len=torch.tensor([model_input["llm_prompt_speech_token"].shape[1]], dtype=torch.int32).to(device),
            embedding=model_input["llm_embedding"].to(device),
            uuid="flow-decoder-dump",
        )
        tokens = [token for token in token_generator]
    if not tokens:
        raise RuntimeError("LLM returned no speech tokens")
    return torch.tensor(tokens, dtype=torch.int32, device=device).unsqueeze(0)


def build_estimator_inputs(model, model_input: dict, generated_token: torch.Tensor) -> tuple[dict, dict]:
    flow = model.model.flow
    device = model.model.device

    embedding = model_input["flow_embedding"].to(device)
    embedding = F.normalize(embedding, dim=1)
    spks = flow.spk_embed_affine_layer(embedding)

    prompt_token = model_input["flow_prompt_speech_token"].to(device)
    prompt_feat = model_input["prompt_speech_feat"].to(device)
    prompt_token_len = prompt_token.shape[1]
    token_len = generated_token.shape[1]

    token = torch.concat([prompt_token, generated_token], dim=1)
    token_len_tensor = torch.tensor([prompt_token_len + token_len], dtype=torch.int32, device=device)
    token_mask = (~make_pad_mask(token_len_tensor)).unsqueeze(-1).to(spks)
    token = flow.input_embedding(torch.clamp(token, min=0)) * token_mask

    h, _ = flow.encoder(token, token_len_tensor)
    h = flow.encoder_proj(h)

    mel_len1 = prompt_feat.shape[1]
    mel_len2 = int(token_len / flow.input_frame_rate * 22050 / 256)
    h, _ = flow.length_regulator.inference(h[:, :prompt_token_len], h[:, prompt_token_len:], mel_len1, mel_len2, flow.input_frame_rate)

    cond = torch.zeros([1, mel_len1 + mel_len2, flow.output_size], device=device, dtype=h.dtype)
    cond[:, :mel_len1] = prompt_feat
    cond = cond.transpose(1, 2)

    total_mel_len = mel_len1 + mel_len2
    mask = (~make_pad_mask(torch.tensor([total_mel_len], device=device))).to(h).unsqueeze(1)
    mu = h.transpose(1, 2).contiguous()

    decoder = flow.decoder
    x = torch.randn_like(mu)
    t_span = torch.linspace(0, 1, 10 + 1, device=mu.device, dtype=mu.dtype)
    if decoder.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t = t_span[0].unsqueeze(0)

    estimator_inputs = {
        "x": x,
        "mask": mask,
        "mu": mu,
        "t": t,
        "spks": spks,
        "cond": cond,
    }
    metadata = {
        "generated_token_length": int(token_len),
        "prompt_token_length": int(prompt_token_len),
        "prompt_mel_length": int(mel_len1),
        "generated_mel_length": int(mel_len2),
        "total_mel_length": int(total_mel_len),
        "expected_shapes": {key: list(value.shape) for key, value in estimator_inputs.items()},
    }
    return estimator_inputs, metadata


def save_bundle(out_dir: Path, arrays: dict, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(out_dir / f"{name}.npy", value)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--prompt-wav", default=str(DEFAULT_PROMPT_WAV))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    prompt_wav = Path(args.prompt_wav)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    if not prompt_wav.is_absolute():
        prompt_wav = (PROJECT_ROOT / prompt_wav).resolve()
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()

    if not prompt_wav.is_file():
        raise FileNotFoundError(f"prompt wav not found: {prompt_wav}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    ensure_seed(args.seed)
    onnxruntime.InferenceSession = cpu_inference_session
    try:
        model = CosyVoice(str(model_dir))
    finally:
        onnxruntime.InferenceSession = _ORIGINAL_INFERENCE_SESSION

    normalized_prompt = model.frontend.text_normalize(args.prompt_text, split=False, text_frontend=True)
    normalized_segments = model.frontend.text_normalize(args.text, split=True, text_frontend=True)
    first_segment = normalized_segments[0]
    model_input = model.frontend.frontend_zero_shot(first_segment, normalized_prompt, str(prompt_wav), model.sample_rate, "")
    generated_token = sample_llm_tokens(model, model_input)
    estimator_inputs, metadata = build_estimator_inputs(model, model_input, generated_token)

    arrays = {name: value.detach().cpu().numpy() for name, value in estimator_inputs.items()}
    metadata.update(
        {
            "text": args.text,
            "prompt_text": args.prompt_text,
            "prompt_wav": str(prompt_wav),
            "model_dir": str(model_dir),
            "selected_segment": first_segment,
            "estimator_target": "flow.decoder.estimator.fp32.onnx",
            "seed": args.seed,
        }
    )
    save_bundle(out_dir, arrays, metadata)

    print(f"model_dir={model_dir}")
    print(f"prompt_wav={prompt_wav}")
    print(f"out_dir={out_dir}")
    print(f"selected_segment={first_segment}")
    for name in ["x", "mask", "mu", "t", "spks", "cond"]:
        print(describe_array(name, arrays[name]))
    print(f"metadata={out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
