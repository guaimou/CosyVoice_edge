"""CosyVoice TTS inference on SC171v3 board with SNPE/QNN DLC accelerator.

Replaces the PyTorch flow.decoder.estimator with fiboaisdk InferenceSession
using the pre-converted DLC. Other components (frontend, LLM, HiFiGAN) use
PyTorch/ONNX Runtime as available on the board.

Usage (on board):
  python3 infer_tts_board.py --text "你好，这是板端验证。" --out output.wav
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add cosyvoice source
COSYVOICE_PATH = os.path.join(PROJECT_ROOT, "cosyvoice_src")
if COSYVOICE_PATH not in sys.path:
    sys.path.insert(0, COSYVOICE_PATH)

from cosyvoice.cli.cosyvoice import CosyVoice

# Import board SDK
from fiboaisdk.api_aisdk_py import api_infer_py


class BoardEstimator:
    """SNPE/QNN DLC estimator running via fiboaisdk on Qualcomm board."""

    def __init__(self, dlc_path: str, framework: str = "SNPE",
                 runtime: str = "CPU", profile_level: int = 5):
        self.dlc_path = dlc_path
        self.call_count = 0
        self.target_seq_len = 500  # DLC fixed seq_len
        self.session = None

        # Create inference session (don't init yet — init per sequence length)
        self.params = api_infer_py.InferParams(
            dlc_path,
            "QUALCOMM",
            framework,
            runtime,
            "ERROR",       # log_level
            profile_level,  # BURST mode
        )
        self.api = api_infer_py.InferAPI()

    def initialize(self):
        """Initialize the SNPE/QNN session."""
        print(f"  [BoardEstimator] Initializing: {self.dlc_path}")
        result = self.api.Init(self.params)
        print(f"  [BoardEstimator] Init result: {result}")
        return result

    def release(self):
        """Release SNPE/QNN resources."""
        if self.api:
            self.api.Release()

    def __call__(self, x, mask, mu, t, spks, cond, streaming=False):
        self.call_count += 1
        if self.call_count == 1:
            self.initialize()

        orig_seq_len = x.size(2)

        # Resize to target seq_len
        if orig_seq_len != self.target_seq_len:
            def resize(tensor):
                if tensor.ndim == 3 and tensor.size(2) == orig_seq_len:
                    if orig_seq_len < self.target_seq_len:
                        pad = torch.zeros(tensor.size(0), tensor.size(1),
                                          self.target_seq_len - orig_seq_len,
                                          device=tensor.device, dtype=tensor.dtype)
                        return torch.cat([tensor, pad], dim=2)
                    else:
                        return tensor[:, :, :self.target_seq_len]
                return tensor

            x = resize(x)
            mask = resize(mask)
            mu = resize(mu)
            cond = resize(cond)

        # Convert to numpy with NFC layout for 3D inputs
        def to_nfc(t, batch, channels):
            """NCF [b,c,t] -> NFC [b,t,c] float32 numpy"""
            return t.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 1).copy()

        input_feed = {
            "x": to_nfc(x, 2, 80),
            "mask": mask.detach().cpu().numpy().astype(np.float32),
            "mu": to_nfc(mu, 2, 80),
            "t": t.detach().cpu().numpy().astype(np.float32),
            "spks": spks.detach().cpu().numpy().astype(np.float32),
            "cond": to_nfc(cond, 2, 80),
        }

        # Flatten numpy arrays to lists (required by fiboaisdk API)
        input_lists = {k: v.flatten().tolist() for k, v in input_feed.items()}

        # Execute inference
        ret = self.api.Execute_float(input_lists)
        if ret != 0:
            raise RuntimeError(f"SNPE Execute_float failed with code {ret}")
        result = self.api.FetchOutputs_float(["estimator_out"])

        # Parse output: SNPE produces NFC [2, seq_len, 80], convert to NCF
        out_data = np.array(result["estimator_out"], dtype=np.float32)
        out_data = out_data.reshape(2, self.target_seq_len, 80).transpose(0, 2, 1).copy()

        # Adjust to original length
        output = torch.from_numpy(out_data).to(x.device).to(x.dtype)
        if output.size(2) > orig_seq_len:
            output = output[:, :, :orig_seq_len]
        elif output.size(2) < orig_seq_len:
            pad = torch.zeros(output.size(0), output.size(1),
                              orig_seq_len - output.size(2),
                              device=output.device, dtype=output.dtype)
            output = torch.cat([output, pad], dim=2)

        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。")
    parser.add_argument("--prompt-wav", default=os.path.join(PROJECT_ROOT, "assets", "zero_shot_prompt.wav"))
    parser.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "pretrained", "CosyVoice-300M"))
    parser.add_argument("--dlc", default=os.path.join(PROJECT_ROOT, "dlc", "flow.decoder.estimator.fp32.dlc"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--framework", default="SNPE", choices=["SNPE", "QNN"])
    parser.add_argument("--runtime", default="CPU", choices=["CPU", "GPU", "DSP", "NPU"])
    parser.add_argument("--profile-level", type=int, default=5,
                        help="0=BALANCED, 1=HIGH_PERFORMANCE, 2=POWER_SAVER, 5=BURST")
    args = parser.parse_args()

    print(f"text={args.text}")
    print(f"out={args.out}")
    print(f"prompt_wav={args.prompt_wav}")
    print(f"model_dir={args.model_dir}")
    print(f"dlc={args.dlc}")
    print(f"framework={args.framework}, runtime={args.runtime}")

    # Load CosyVoice model
    print(f"loading_model={args.model_dir}")
    start = time.time()
    model = CosyVoice(args.model_dir)
    print(f"model_loaded_in={time.time() - start:.1f}s")

    # Replace estimator with board DLC
    print("replacing estimator with board DLC...")
    board_estimator = BoardEstimator(
        args.dlc,
        framework=args.framework,
        runtime=args.runtime,
        profile_level=args.profile_level,
    )

    # Patch forward_estimator
    decoder = model.model.flow.decoder
    original_forward = decoder.forward_estimator

    def snpe_forward_estimator(x, mask, mu, t, spks, cond, streaming=False):
        return board_estimator(x, mask, mu, t, spks, cond, streaming=streaming)

    decoder.forward_estimator = snpe_forward_estimator
    print("estimator replaced")

    # Run inference
    print(f"running inference: {args.text}")
    start_infer = time.time()
    generated = False

    try:
        for index, result in enumerate(
            model.inference_zero_shot(args.text, args.prompt_text, args.prompt_wav, speed=args.speed)
        ):
            output_path = args.out
            if index > 0:
                base, ext = os.path.splitext(output_path)
                output_path = f"{base}_{index}{ext}"

            samples = result["tts_speech"].squeeze().numpy()
            sf.write(output_path, samples, model.sample_rate)
            duration = result["tts_speech"].shape[1] / model.sample_rate
            total_time = time.time() - start_infer
            print(f"saved={output_path} duration={duration:.2f}s "
                  f"infer_time={total_time:.1f}s calls={board_estimator.call_count}")
            generated = True
    finally:
        board_estimator.release()

    if not generated:
        raise RuntimeError("CosyVoice returned no audio segments")


if __name__ == "__main__":
    main()
