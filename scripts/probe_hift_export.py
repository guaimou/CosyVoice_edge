"""Export/probe script for the HiFT vocoder used by CosyVoice.

Goal: identify the smallest stable export boundary for DLC conversion.
This script supports three probe targets:

1. f0_predictor  : speech_feat[B,80,T] -> f0[B,T]
2. decode_pre_istft: (speech_feat[B,80,T], source[B,1,T*256]) -> (magnitude, phase)
3. decode        : (speech_feat[B,80,T], source[B,1,T*256]) -> audio[B,N]
4. inference     : speech_feat[B,80,T] -> (audio[B,N], source[B,1,T*256])

The recommended order is:
- Start with f0_predictor
- Then try decode
- Only then attempt full inference
"""

from pathlib import Path
import argparse
import sys

import torch
import onnx
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
UPSTREAM_ROOT = THIRD_PARTY_DIR / "CosyVoice_edge"
DEFAULT_MODEL_DIR = PRETRAINED_DIR / "CosyVoice-300M"
OUTPUT_DIR = PROJECT_ROOT / "output"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from hyperpyyaml import load_hyperpyyaml


class F0PredictorWrapper(torch.nn.Module):
    def __init__(self, f0_predictor: torch.nn.Module):
        super().__init__()
        self.f0_predictor = f0_predictor

    def forward(self, speech_feat: torch.Tensor) -> torch.Tensor:
        return self.f0_predictor(speech_feat)


class DecodeWrapper(torch.nn.Module):
    def __init__(self, hift: torch.nn.Module):
        super().__init__()
        self.hift = hift

    def forward(self, speech_feat: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self.hift.decode(x=speech_feat, s=source)


class DecodePreIStftWrapper(torch.nn.Module):
    """Export the pre-ISTFT part of decode: conv stack -> magnitude/phase.

    Takes pre-computed STFT of the source signal as input (to avoid
    torch.stft which is not ONNX-exportable). The caller runs the STFT
    on the host side and provides the result.
    """

    def __init__(self, hift: torch.nn.Module):
        super().__init__()
        self.hift = hift
        self.num_upsamples = hift.num_upsamples
        self.num_kernels = hift.num_kernels
        self.conv_pre = hift.conv_pre
        self.ups = hift.ups
        self.reflection_pad = hift.reflection_pad
        self.source_downs = hift.source_downs
        self.source_resblocks = hift.source_resblocks
        self.resblocks = hift.resblocks
        self.conv_post = hift.conv_post
        self.lrelu_slope = hift.lrelu_slope
        self.n_fft = hift.istft_params["n_fft"]

    def forward(self, speech_feat: torch.Tensor, s_stft: torch.Tensor):
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


class InferenceWrapper(torch.nn.Module):
    def __init__(self, hift: torch.nn.Module):
        super().__init__()
        self.hift = hift

    def forward(self, speech_feat: torch.Tensor):
        audio, source = self.hift.inference(speech_feat=speech_feat)
        return audio, source


def load_hift(model_dir: Path):
    with (model_dir / "cosyvoice.yaml").open("r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f)
    hift = configs["hift"]
    hift_state_dict = {
        k.replace("generator.", ""): v
        for k, v in torch.load(model_dir / "hift.pt", map_location="cpu", weights_only=True).items()
    }
    hift.load_state_dict(hift_state_dict, strict=True)
    hift.eval()
    return hift


def export_and_check(module: torch.nn.Module, inputs: tuple, input_names: list[str], output_names: list[str], onnx_path: Path, dynamic_axes=None):
    module.eval()
    with torch.inference_mode():
        ref = module(*inputs)

    # Use legacy exporter for better compatibility on this repo/toolchain.
    torch.onnx.export(
        module,
        inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        dynamo=False,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, inputs)}
    ort_outputs = session.run(output_names, ort_inputs)

    if not isinstance(ref, tuple):
        ref = (ref,)

    print(f"exported={onnx_path}")
    for name, ref_tensor, ort_tensor in zip(output_names, ref, ort_outputs):
        ref_np = ref_tensor.detach().cpu().numpy()
        diff = abs(ref_np - ort_tensor)
        print(
            f"{name}: shape={list(ref_np.shape)} max_abs_diff={float(diff.max())} "
            f"mean_abs_diff={float(diff.mean())}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--target", choices=["f0_predictor", "decode_pre_istft", "decode", "inference"], required=True)
    parser.add_argument("--frames", type=int, default=100, help="mel frames T")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()

    hift = load_hift(model_dir)
    T = args.frames

    if args.target == "f0_predictor":
        module = F0PredictorWrapper(hift.f0_predictor)
        speech_feat = torch.randn(1, 80, T, dtype=torch.float32)
        dynamic_axes = {
            "speech_feat": {0: "batch", 2: "mel_frames"},
            "f0": {0: "batch", 1: "time"},
        }
        # Fixed-size for DLC
        onnx_fixed = OUTPUT_DIR / f"hift_f0_predictor_T{T}.onnx"
        export_and_check(module, (speech_feat,), ["speech_feat"], ["f0"], onnx_fixed)
        # Dynamic for general use
        onnx_dyn = OUTPUT_DIR / "hift_f0_predictor_dynamic.onnx"
        export_and_check(module, (speech_feat,), ["speech_feat"], ["f0"],
                         onnx_dyn, dynamic_axes=dynamic_axes)
        return

    if args.target == "decode_pre_istft":
        module = DecodePreIStftWrapper(hift)
        speech_feat = torch.randn(1, 80, T, dtype=torch.float32)
        source_len = T * 256
        source = torch.randn(1, 1, source_len, dtype=torch.float32)

        # Pre-compute source STFT (host-side, not in ONNX graph)
        s_stft_real, s_stft_imag = hift._stft(source.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        # Try dynamo-based export first (handles complex ops better)
        onnx_path = OUTPUT_DIR / "hift_decode_pre_istft_dynamic.onnx"
        onnx_path_fixed = OUTPUT_DIR / f"hift_decode_pre_istft_T{T}.onnx"

        # Dynamic axes for variable-length input
        dynamic_axes = {
            "speech_feat": {0: "batch", 2: "mel_frames"},
            "s_stft": {0: "batch", 2: "stft_frames"},
            "magnitude": {0: "batch", 2: "audio_frames"},
            "phase": {0: "batch", 2: "audio_frames"},
        }

        try:
            # Try new dynamo export
            export_and_check(module, (speech_feat, s_stft),
                             ["speech_feat", "s_stft"], ["magnitude", "phase"],
                             onnx_path, dynamic_axes=dynamic_axes)
        except Exception as e:
            print(f"Legacy export failed: {e}")

        # Also export fixed-size for DLC conversion
        export_and_check(module, (speech_feat, s_stft),
                         ["speech_feat", "s_stft"], ["magnitude", "phase"],
                         onnx_path_fixed)
        return

    if args.target == "decode":
        module = DecodeWrapper(hift)
        speech_feat = torch.randn(1, 80, T, dtype=torch.float32)
        source_len = T * 256
        source = torch.randn(1, 1, source_len, dtype=torch.float32)
        onnx_path = OUTPUT_DIR / f"hift_decode_T{T}.onnx"
        export_and_check(module, (speech_feat, source), ["speech_feat", "source"], ["audio"], onnx_path)
        return

    if args.target == "inference":
        module = InferenceWrapper(hift)
        speech_feat = torch.randn(1, 80, T, dtype=torch.float32)
        onnx_path = OUTPUT_DIR / f"hift_inference_T{T}.onnx"
        export_and_check(module, (speech_feat,), ["speech_feat"], ["audio", "source"], onnx_path)
        return


if __name__ == "__main__":
    main()
