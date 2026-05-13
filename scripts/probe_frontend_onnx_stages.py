from pathlib import Path
import argparse
import sys

import numpy as np
import torchaudio.compliance.kaldi as kaldi
import whisper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
UPSTREAM_ROOT = THIRD_PARTY_DIR / "CosyVoice_edge"
DEFAULT_MODEL_DIR = PRETRAINED_DIR / "CosyVoice-300M"
DEFAULT_PROMPT_WAV = PROJECT_ROOT / "assets" / "zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
DEFAULT_TEXT = "你好，这是一次前端ONNX阶段检查。"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav


def describe_array(name: str, value) -> str:
    shape = list(value.shape)
    dtype = getattr(value, "dtype", type(value).__name__)
    return f"{name}\tshape={shape}\tdtype={dtype}"


def describe_tensor(name: str, value) -> str:
    shape = list(value.shape)
    dtype = getattr(value, "dtype", type(value).__name__)
    device = getattr(value, "device", "n/a")
    return f"{name}\tshape={shape}\tdtype={dtype}\tdevice={device}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--prompt-wav", default=str(DEFAULT_PROMPT_WAV))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    prompt_wav = Path(args.prompt_wav)
    model_dir = Path(args.model_dir)
    if not prompt_wav.is_absolute():
        prompt_wav = (PROJECT_ROOT / prompt_wav).resolve()
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()

    if not prompt_wav.is_file():
        raise FileNotFoundError(f"prompt wav not found: {prompt_wav}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    model = CosyVoice(str(model_dir))
    frontend = model.frontend

    speech_16k = load_wav(str(prompt_wav), 16000)
    campplus_feat = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    campplus_feat = campplus_feat - campplus_feat.mean(dim=0, keepdim=True)
    campplus_input = campplus_feat.unsqueeze(0).cpu().numpy()
    campplus_output = frontend.campplus_session.run(None, {frontend.campplus_session.get_inputs()[0].name: campplus_input})[0]

    speech_token_feat = whisper.log_mel_spectrogram(speech_16k, n_mels=128)
    speech_token_inputs = {
        frontend.speech_tokenizer_session.get_inputs()[0].name: speech_token_feat.detach().cpu().numpy(),
        frontend.speech_tokenizer_session.get_inputs()[1].name: np.array([speech_token_feat.shape[2]], dtype=np.int32),
    }
    speech_token_output = frontend.speech_tokenizer_session.run(None, speech_token_inputs)[0]

    normalized_prompt = frontend.text_normalize(args.prompt_text, split=False, text_frontend=True)
    normalized_segments = frontend.text_normalize(args.text, split=True, text_frontend=True)
    first_segment = normalized_segments[0]
    model_input = frontend.frontend_zero_shot(first_segment, normalized_prompt, str(prompt_wav), model.sample_rate, '')

    print(f"model_dir={model_dir}")
    print(f"prompt_wav={prompt_wav}")
    print(f"campplus_model={model_dir / 'campplus.onnx'}")
    print(f"speech_tokenizer_model={model_dir / 'speech_tokenizer_v1.onnx'}")

    print("campplus_stage_start")
    print(describe_array("campplus_input", campplus_input))
    print(describe_array("campplus_output", campplus_output))
    print(describe_tensor("flow_embedding", model_input["flow_embedding"]))
    print(describe_tensor("llm_embedding", model_input["llm_embedding"]))
    print("campplus_stage_end")

    print("speech_tokenizer_stage_start")
    print(describe_tensor("speech_token_feat", speech_token_feat))
    print(describe_array("speech_tokenizer_output", speech_token_output))
    print(describe_tensor("flow_prompt_speech_token", model_input["flow_prompt_speech_token"]))
    print(describe_tensor("llm_prompt_speech_token", model_input["llm_prompt_speech_token"]))
    print("speech_tokenizer_stage_end")

    print("frontend_alignment_start")
    print(describe_tensor("prompt_speech_feat", model_input["prompt_speech_feat"]))
    print(describe_tensor("prompt_text", model_input["prompt_text"]))
    print(describe_tensor("text", model_input["text"]))
    print("frontend_alignment_end")


if __name__ == "__main__":
    main()
