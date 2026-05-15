"""Export HiFT decode pre-ISTFT with explicit padding for SNPE compatibility.

SNPE 2.29 cannot handle dilated Conv1d with implicit padding. This script
replaces all dilated Conv1d layers with Pad1d + Conv1d(padding=0) wrappers,
then exports the ONNX model for DLC conversion.
"""

from pathlib import Path
import argparse
import sys
import copy

import torch
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


def get_explicit_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size - 1) * dilation // 2


class ExplicitPadConv1d(torch.nn.Module):
    """Conv1d with explicit reflection padding, zero padding on the conv itself."""

    def __init__(self, orig_conv: torch.nn.Conv1d):
        super().__init__()
        self.in_channels = orig_conv.in_channels
        self.out_channels = orig_conv.out_channels
        self.kernel_size = orig_conv.kernel_size[0]
        self.stride = orig_conv.stride[0]
        self.dilation = orig_conv.dilation[0]
        self.pad_amount = get_explicit_padding(self.kernel_size, self.dilation)

        self.conv = torch.nn.Conv1d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, dilation=self.dilation,
            padding=0, bias=orig_conv.bias is not None,
        )
        self.conv.weight.data = orig_conv.weight.data.clone()
        if orig_conv.bias is not None:
            self.conv.bias.data = orig_conv.bias.data.clone()

    def forward(self, x):
        if self.pad_amount > 0:
            x = torch.nn.functional.pad(x, (self.pad_amount, self.pad_amount), mode="reflect")
        return self.conv(x)


def patch_resblock_with_explicit_pad(resblock) -> None:
    for conv_list_name in ["convs1", "convs2"]:
        conv_list = getattr(resblock, conv_list_name)
        for i, conv in enumerate(conv_list):
            if isinstance(conv, torch.nn.Conv1d) and (conv.dilation[0] > 1 or conv.padding[0] > 0):
                new_conv = ExplicitPadConv1d(conv)
                conv_list[i] = new_conv


def patch_decoder_with_explicit_pad(hift) -> None:
    for resblock in hift.resblocks:
        patch_resblock_with_explicit_pad(resblock)
    for resblock in hift.source_resblocks:
        patch_resblock_with_explicit_pad(resblock)


class DecodePreIStftExplicitPad(torch.nn.Module):
    def __init__(self, hift):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()

    with (model_dir / "cosyvoice.yaml").open("r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f)
    hift = configs["hift"]
    hift.load_state_dict({
        k.replace("generator.", ""): v
        for k, v in torch.load(model_dir / "hift.pt", map_location="cpu", weights_only=True).items()
    }, strict=True)
    hift.eval()

    # Patch everything with explicit padding
    print("patching ResBlocks with explicit padding...")
    patch_decoder_with_explicit_pad(hift)

    T = args.frames
    module = DecodePreIStftExplicitPad(hift)
    speech_feat = torch.randn(1, 80, T, dtype=torch.float32)
    source_len = T * 256
    source = torch.randn(1, 1, source_len, dtype=torch.float32)
    s_stft_real, s_stft_imag = hift._stft(source.squeeze(1))
    s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

    # PyTorch reference
    module.eval()
    with torch.inference_mode():
        ref_mag, ref_phase = module(speech_feat, s_stft)

    # ONNX export
    dynamic_axes = {
        "speech_feat": {0: "batch", 2: "mel_frames"},
        "s_stft": {0: "batch", 2: "stft_frames"},
        "magnitude": {0: "batch", 2: "audio_frames"},
        "phase": {0: "batch", 2: "audio_frames"},
    }
    onnx_path = OUTPUT_DIR / "hift_decode_pre_istft_explicit_pad.onnx"
    torch.onnx.export(
        module, (speech_feat, s_stft), str(onnx_path),
        input_names=["speech_feat", "s_stft"],
        output_names=["magnitude", "phase"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        dynamo=False,
    )
    print(f"exported={onnx_path}")

    # ORT validation
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_mag, ort_phase = session.run(
        ["magnitude", "phase"],
        {"speech_feat": speech_feat.numpy(), "s_stft": s_stft.numpy()},
    )
    for name, ref, ort_val in [("magnitude", ref_mag, ort_mag), ("phase", ref_phase, ort_phase)]:
        diff = abs(ref.numpy() - ort_val)
        print(f"{name}: max={diff.max():.6f} mean={diff.mean():.8f}")


if __name__ == "__main__":
    main()
