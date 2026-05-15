"""Dump real HiFT inputs/outputs from a live CosyVoice zero-shot case.

Captures speech_feat, source, and audio from one hift.inference() call
so the pre-ISTFT decode ONNX can be validated with real data.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
OUTPUT_DIR = PROJECT_ROOT / "output"
UPSTREAM_ROOT = THIRD_PARTY_DIR / "CosyVoice_edge"
DEFAULT_MODEL_DIR = PRETRAINED_DIR / "CosyVoice-300M"
DEFAULT_CASE_DIR = OUTPUT_DIR / "hift_real_case"
DEFAULT_ONNX = OUTPUT_DIR / "hift_decode_pre_istft_dynamic.onnx"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from cosyvoice.cli.cosyvoice import CosyVoice


def capture_hift_inputs(model: CosyVoice, prompt_text: str, prompt_wav: str) -> dict:
    """Run one zero-shot inference and capture the first hift call's inputs."""
    captured = {}

    original_inference = model.model.hift.inference

    def patched_inference(self, speech_feat, cache_source=torch.zeros(1, 1, 0)):
        captured["speech_feat"] = speech_feat.detach().cpu().clone()
        captured["cache_source"] = cache_source.detach().cpu().clone()
        audio, source = original_inference(speech_feat, cache_source)
        captured["audio"] = audio.detach().cpu().clone()
        captured["source"] = source.detach().cpu().clone()
        return audio, source

    model.model.hift.inference = patched_inference.__get__(model.model.hift)

    for result in model.inference_zero_shot("你好，测试。", prompt_text, prompt_wav):
        captured["tts_speech"] = result["tts_speech"].detach().cpu().clone()
        break

    return captured


def compare_decode_onnx(onnx_path: Path, case_dir: Path) -> dict:
    """Compare pre-ISTFT decode ONNX against PyTorch reference."""
    speech_feat = torch.from_numpy(np.load(case_dir / "speech_feat.npy"))
    source = torch.from_numpy(np.load(case_dir / "source.npy"))

    # Load the model to get the same weights
    with (DEFAULT_MODEL_DIR / "cosyvoice.yaml").open("r", encoding="utf-8") as f:
        from hyperpyyaml import load_hyperpyyaml
        configs = load_hyperpyyaml(f)
    hift = configs["hift"]
    hift.load_state_dict({
        k.replace("generator.", ""): v
        for k, v in torch.load(DEFAULT_MODEL_DIR / "hift.pt", map_location="cpu", weights_only=True).items()
    }, strict=True)
    hift.eval()

    # PyTorch reference: compute magnitude/phase from decode (stopping before ISTFT)
    with torch.inference_mode():
        s_stft_real, s_stft_imag = hift._stft(source.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        # Run the pre-ISTFT part manually (same logic as DecodePreIStftWrapper)
        class _Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_pre = hift.conv_pre
                self.ups = hift.ups
                self.reflection_pad = hift.reflection_pad
                self.source_downs = hift.source_downs
                self.source_resblocks = hift.source_resblocks
                self.resblocks = hift.resblocks
                self.conv_post = hift.conv_post
                self.lrelu_slope = hift.lrelu_slope
                self.n_fft = hift.istft_params["n_fft"]
                self.num_upsamples = hift.num_upsamples
                self.num_kernels = hift.num_kernels

            def forward(self, speech_feat, s_stft):
                x = self.conv_pre(speech_feat)
                for i in range(self.num_upsamples):
                    x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
                    x = self.ups[i](x)
                    if i == self.num_upsamples - 1:
                        x = self.reflection_pad(x)
                    si = self.source_downs[i](s_stft)
                    si = self.source_resblocks[i](si)
                    x = x + si
                    xs = None
                    for j in range(self.num_kernels):
                        if xs is None:
                            xs = self.resblocks[i * self.num_kernels + j](x)
                        else:
                            xs += self.resblocks[i * self.num_kernels + j](x)
                    x = xs / self.num_kernels
                x = torch.nn.functional.leaky_relu(x)
                x = self.conv_post(x)
                magnitude = torch.exp(x[:, : self.n_fft // 2 + 1, :])
                phase = torch.sin(x[:, self.n_fft // 2 + 1 :, :])
                return magnitude, phase

        wrapper = _Wrapper()
        ref_mag, ref_phase = wrapper(speech_feat, s_stft)

    # ORT
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "speech_feat": speech_feat.numpy().astype(np.float32),
        "s_stft": s_stft.numpy().astype(np.float32),
    }
    ort_mag, ort_phase = session.run(["magnitude", "phase"], ort_inputs)

    results = {}
    for name, ref, ort_val in [("magnitude", ref_mag, ort_mag), ("phase", ref_phase, ort_phase)]:
        diff = np.abs(ref.numpy() - ort_val)
        results[name] = {
            "shape": list(ref.shape),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "rmse": float(np.sqrt(np.mean((ref.numpy() - ort_val) ** 2))),
            "ref_min": float(ref.min()),
            "ref_max": float(ref.max()),
        }
        print(f"{name}: shape={list(ref.shape)} max_abs_diff={diff.max():.6f} mean_abs_diff={diff.mean():.6f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--case-dir", default=str(DEFAULT_CASE_DIR))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。")
    parser.add_argument("--prompt-wav", default=str(PROJECT_ROOT / "assets" / "zero_shot_prompt.wav"))
    parser.add_argument("--compare-only", action="store_true", help="Skip capture, only run ONNX comparison")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.is_absolute():
        case_dir = (PROJECT_ROOT / case_dir).resolve()
    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = (PROJECT_ROOT / onnx_path).resolve()

    if not args.compare_only:
        model_dir = Path(args.model_dir)
        prompt_wav = Path(args.prompt_wav)

        print("loading CosyVoice model...")
        model = CosyVoice(str(model_dir))

        print("capturing hift inputs from zero-shot inference...")
        captured = capture_hift_inputs(model, args.prompt_text, str(prompt_wav))

        case_dir.mkdir(parents=True, exist_ok=True)
        for name in ["speech_feat", "source", "audio", "cache_source"]:
            if name in captured:
                np.save(case_dir / f"{name}.npy", captured[name].numpy())
                print(f"  saved {name}: shape={list(captured[name].shape)}")

        metadata = {
            "speech_feat_shape": list(captured["speech_feat"].shape),
            "source_shape": list(captured["source"].shape),
            "audio_shape": list(captured["audio"].shape),
            "audio_duration_s": float(captured["audio"].shape[-1] / 22050),
        }
        (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"case saved to {case_dir}")

    # Compare ONNX
    print(f"\ncomparing ONNX {onnx_path}...")
    results = compare_decode_onnx(onnx_path, case_dir)
    (case_dir / "decode_pre_istft_onnx_compare.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"comparison saved to {case_dir / 'decode_pre_istft_onnx_compare.json'}")


if __name__ == "__main__":
    main()
