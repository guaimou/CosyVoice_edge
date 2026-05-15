"""CosyVoice TTS inference with SNPE DLC replacing the flow decoder estimator.

This script hooks into the CosyVoice pipeline and redirects flow.decoder.estimator
calls through SNPE DLC running in Docker, producing audio for quality verification.

Usage:
  .venv\Scripts\python.exe scripts\run_infer_snpe.py --text "你好，这是一次SNPE验证。" --out output\snpe_test.wav
"""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
OUTPUT_DIR = PROJECT_ROOT / "output"
UPSTREAM_ROOT = THIRD_PARTY_DIR / "CosyVoice_edge"
DEFAULT_MODEL_DIR = PRETRAINED_DIR / "CosyVoice-300M"
DEFAULT_PROMPT_WAV = PROJECT_ROOT / "assets" / "zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
DEFAULT_DLC = OUTPUT_DIR / "flow.decoder.estimator.fp32.dlc"
DEFAULT_CONTAINER = "my_work"
DEFAULT_SNPE_PROJECT = "/project/cosyvoice_snpe_snpe"
DEFAULT_SNPE_ROOT = "/opt/2.29.0.241129"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from cosyvoice.cli.cosyvoice import CosyVoice


class SnpeEstimatorWrapper:
    """Wraps SNPE DLC inference to match the PyTorch estimator interface."""

    def __init__(self, dlc_path: Path, container: str, snpe_project: str, snpe_root: str):
        self.dlc_path = dlc_path
        self.container = container
        self.snpe_project = snpe_project
        self.snpe_root = snpe_root
        self.temp_dir = OUTPUT_DIR / "snpe_infer_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0

    def _ensure_dlc_in_container(self) -> str:
        """Copy DLC to container if needed, return container path."""
        container_dlc = f"{self.snpe_project}/{self.dlc_path.name}"
        return container_dlc

    def _write_raw(self, name: str, tensor: torch.Tensor) -> Path:
        path = self.temp_dir / f"{name}.raw"
        # SNPE expects NFC input: transpose NCF -> NFC for x/mu/cond
        tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensor_np.tofile(path)
        return path

    def _prepare_snpe_inputs(self, x, mask, mu, t, spks, cond) -> Path:
        """Write raw tensors and input_list.txt for SNPE. Returns path to input_list."""
        raw_dir = self.temp_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Convert NCF -> NFC for 3D tensors (SNPE DLC expects NFC)
        def to_nfc(t: torch.Tensor) -> np.ndarray:
            """NCF [b,c,t] -> NFC [b,t,c]"""
            return t.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 1).copy()

        for name, tensor, is_3d in [
            ("x", x, True), ("mask", mask, True),
            ("mu", mu, True), ("t", t, False),
            ("spks", spks, False), ("cond", cond, True),
        ]:
            if is_3d and tensor.ndim == 3:
                data = to_nfc(tensor)
            else:
                data = tensor.detach().cpu().numpy().astype(np.float32)
            data.tofile(raw_dir / f"{name}.raw")

        # Write input_list with container paths
        container_raw = f"{self.snpe_project}/snpe_infer_temp/raw"
        input_list = self.temp_dir / "input_list.txt"
        lines = [
            f"x:={container_raw}/x.raw",
            f"mask:={container_raw}/mask.raw",
            f"mu:={container_raw}/mu.raw",
            f"t:={container_raw}/t.raw",
            f"spks:={container_raw}/spks.raw",
            f"cond:={container_raw}/cond.raw",
        ]
        input_list.write_text(" ".join(lines) + "\n", encoding="ascii", newline="\n")
        return input_list

    def _run_snpe(self, input_list: Path) -> np.ndarray:
        """Run SNPE in Docker and return the output tensor as numpy array."""
        import os as _os
        container_dlc = f"{self.snpe_project}/{self.dlc_path.name}"
        container_temp = f"{self.snpe_project}/snpe_infer_temp"
        container_output = f"{container_temp}/output"
        seq_len = self._last_seq_len

        # Ensure container temp dir exists
        subprocess.run(
            f'docker exec {self.container} mkdir -p {container_temp}/raw {container_output}',
            check=True, shell=True, capture_output=True,
        )

        # Copy each raw file individually (Windows paths for docker cp)
        for raw_file in (self.temp_dir / "raw").glob("*.raw"):
            src = str(raw_file.resolve())
            filename = raw_file.name
            subprocess.run(
                f'docker cp "{src}" {self.container}:{container_temp}/raw/{filename}',
                check=True, shell=True, capture_output=True,
            )

        # Copy input_list
        src_list = str(input_list.resolve())
        subprocess.run(
            f'docker cp "{src_list}" {self.container}:{container_temp}/input_list.txt',
            check=True, shell=True, capture_output=True,
        )

        # Execute SNPE
        result = subprocess.run(
            'docker exec ' + self.container + ' bash -c "'
            'export SNPE_ROOT=' + self.snpe_root + ' && '
            'export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH && '
            'rm -rf ' + container_output + ' && mkdir -p ' + container_output + ' && '
            '$SNPE_ROOT/bin/x86_64-linux-clang/snpe-net-run '
            '--container ' + container_dlc + ' '
            '--input_list ' + container_temp + '/input_list.txt '
            '--output_dir ' + container_output + ' '
            '--runtime_order cpu --userbuffer_float"',
            check=True, shell=True, capture_output=True, text=True,
        )
        if result.stdout:
            print(f"    SNPE: {result.stdout.strip()[:150]}")
        if result.stderr and 'ERROR' in result.stderr:
            print(f"    SNPE ERR: {result.stderr.strip()[:200]}")

        # Copy output back
        host_output_dir = self.temp_dir / "output"
        host_output_dir.mkdir(parents=True, exist_ok=True)
        dst_out = str(host_output_dir.resolve() / "estimator_out.raw")
        subprocess.run(
            f'docker cp {self.container}:{container_output}/Result_0/estimator_out.raw "{dst_out}"',
            check=True, shell=True, capture_output=True,
        )

        # Load output (SNPE DLC produces NFC output [2, seq_len, 80], convert to NCF)
        raw = np.fromfile(host_output_dir / "estimator_out.raw", dtype=np.float32)
        out = raw.reshape(2, seq_len, 80).transpose(0, 2, 1).copy()  # NFC -> NCF
        return out

    def __call__(self, x, mask, mu, t, spks, cond, streaming=False):
        """Mimics torch.nn.Module __call__ for the estimator."""
        self.call_count += 1
        orig_seq_len = x.size(2)
        target_seq_len = getattr(SnpeEstimatorWrapper, 'TARGET_SEQ_LEN', 500)

        print(f"  [SNPE estimator call #{self.call_count}] x={list(x.shape)} t={t.tolist()}")

        # Pad or truncate to target_seq_len
        if orig_seq_len != target_seq_len:
            def resize(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.ndim == 3 and tensor.size(2) == orig_seq_len:
                    if orig_seq_len < target_seq_len:
                        pad = torch.zeros(tensor.size(0), tensor.size(1), target_seq_len - orig_seq_len,
                                          device=tensor.device, dtype=tensor.dtype)
                        return torch.cat([tensor, pad], dim=2)
                    else:
                        return tensor[:, :, :target_seq_len]
                return tensor

            x = resize(x)
            mask = resize(mask)
            mu = resize(mu)
            cond = resize(cond)

        self._last_seq_len = target_seq_len

        # Run SNPE
        self._prepare_snpe_inputs(x, mask, mu, t, spks, cond)
        output_np = self._run_snpe(self.temp_dir / "input_list.txt")

        # Unpad or truncate output to match original length
        output = torch.from_numpy(output_np).to(x.device).to(x.dtype)
        if output.size(2) > orig_seq_len:
            output = output[:, :, :orig_seq_len]
        elif output.size(2) < orig_seq_len:
            pad = torch.zeros(output.size(0), output.size(1), orig_seq_len - output.size(2),
                              device=output.device, dtype=output.dtype)
            output = torch.cat([output, pad], dim=2)
        # output size now matches orig_seq_len

        return output


def patch_estimator(model: CosyVoice, dlc_path: Path, container: str, snpe_project: str, snpe_root: str) -> SnpeEstimatorWrapper:
    """Replace flow.decoder.forward_estimator to route through SNPE DLC."""
    wrapper = SnpeEstimatorWrapper(dlc_path, container, snpe_project, snpe_root)

    decoder = model.model.flow.decoder

    def snpe_forward_estimator(x, mask, mu, t, spks, cond, streaming=False):
        return wrapper(x, mask, mu, t, spks, cond, streaming=streaming)

    decoder.forward_estimator = snpe_forward_estimator

    return wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--prompt-wav", default=str(DEFAULT_PROMPT_WAV))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--dlc", default=str(DEFAULT_DLC))
    parser.add_argument("--dlc-seq-len", type=int, default=500, help="Target seq_len of the DLC (default 500)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--snpe-project", default=DEFAULT_SNPE_PROJECT)
    parser.add_argument("--snpe-root", default=DEFAULT_SNPE_ROOT)
    args = parser.parse_args()

    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_wav = Path(args.prompt_wav)
    model_dir = Path(args.model_dir)
    dlc_path = Path(args.dlc)
    if not prompt_wav.is_absolute():
        prompt_wav = (PROJECT_ROOT / prompt_wav).resolve()
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()
    if not dlc_path.is_absolute():
        dlc_path = (PROJECT_ROOT / dlc_path).resolve()

    print(f"text={args.text}")
    print(f"out={output_path}")
    print(f"prompt_text={args.prompt_text}")
    print(f"prompt_wav={prompt_wav}")
    print(f"model_dir={model_dir}")
    print(f"dlc={dlc_path}")

    # Load model
    print(f"loading_model={model_dir}")
    start = time.time()
    model = CosyVoice(str(model_dir))
    print(f"model_loaded_in={time.time() - start:.1f}s")

    # Patch estimator with SNPE DLC
    print("patching estimator with SNPE DLC wrapper...")
    # Set the target seq_len for the DLC
    SnpeEstimatorWrapper.TARGET_SEQ_LEN = args.dlc_seq_len
    wrapper = patch_estimator(model, dlc_path, args.container, args.snpe_project, args.snpe_root)
    print("estimator patched")

    # Run inference
    print(f"running inference: {args.text}")
    generated = False
    for index, result in enumerate(
        model.inference_zero_shot(args.text, args.prompt_text, str(prompt_wav), speed=args.speed)
    ):
        current_output = output_path if index == 0 else output_path.with_name(f"{output_path.stem}_{index}{output_path.suffix}")
        samples = result["tts_speech"].squeeze().numpy()
        sf.write(current_output, samples, model.sample_rate)
        duration = result["tts_speech"].shape[1] / model.sample_rate
        print(f"saved={current_output} duration={duration:.2f}s SNPE_calls={wrapper.call_count}")
        generated = True

    if not generated:
        raise RuntimeError("CosyVoice returned no audio segments")


if __name__ == "__main__":
    main()
