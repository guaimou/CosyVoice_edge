"""CosyVoice TTS on SC171v3 with full SNPE DLC pipeline.

Replaces:
- flow.decoder.estimator → SNPE DLC
- hift.f0_predictor   → SNPE DLC
- hift.decode (pre-ISTFT) → SNPE DLC

Remaining in PyTorch:
- Source generator (m_source + f0_upsamp)
- ISTFT reconstruction

Usage:
  python3 infer_tts_board_v2.py --text "你好" --out /tmp/test.wav --runtime DSP
"""

import os
import sys
import time
import argparse

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COSYVOICE_PATH = os.path.join(PROJECT_ROOT, "cosyvoice_src")
MATCHA_PATH = os.path.join(PROJECT_ROOT, "cosyvoice_src")

for p in [PROJECT_ROOT, COSYVOICE_PATH, MATCHA_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cosyvoice.cli.cosyvoice import CosyVoice
from fiboaisdk.api_aisdk_py import api_infer_py


class Dlcsession:
    """Wrapper for a single SNPE/QNN DLC inference session."""

    def __init__(self, dlc_path, framework="SNPE", runtime="CPU", profile=5):
        self.dlc_path = dlc_path
        self.params = api_infer_py.InferParams(dlc_path, "QUALCOMM", framework, runtime, "ERROR", profile)
        self.api = api_infer_py.InferAPI()
        self._initialized = False

    def init(self):
        if not self._initialized:
            ret = self.api.Init(self.params)
            if ret != 0:
                raise RuntimeError(f"DLC init failed: {self.dlc_path} code={ret}")
            self._initialized = True

    def run(self, input_feed):
        """Execute DLC. Session must already be initialized via init()."""
        input_lists = {k: v.astype(np.float32).flatten().tolist() for k, v in input_feed.items()}
        ret = self.api.Execute_float(input_lists)
        if ret != 0:
            raise RuntimeError(f"DLC execute failed: {self.dlc_path} code={ret}")
        return self.api

    def fetch(self, output_names):
        result = self.api.FetchOutputs_float(output_names)
        return {k: np.array(v, dtype=np.float32) for k, v in result.items()}

    def release(self):
        if self._initialized:
            self.api.Release()
            self._initialized = False


class VocoderDLC:
    """HiFT vocoder using SNPE DLCs for f0_predictor and decode pre-ISTFT."""

    def __init__(self, hift_pytorch, dlc_dir, runtime="CPU", use_int8=True):
        self.hift = hift_pytorch
        f0_name = "hift_f0_predictor_int8.dlc" if use_int8 else "hift_f0_predictor.dlc"
        dec_name = "hift_decode_pre_istft_int8.dlc" if use_int8 else "hift_decode_pre_istft.dlc"
        self.f0_session = Dlcsession(os.path.join(dlc_dir, f0_name), runtime=runtime)
        self.decode_session = Dlcsession(os.path.join(dlc_dir, dec_name), runtime=runtime)
        self.TARGET_T = 500
        self.TARGET_STFT = 12801
        # Pre-initialize all sessions once
        self.f0_session.init()
        self.decode_session.init()

    def __call__(self, speech_feat):
        """speech_feat: [1, 80, T] torch tensor → audio: [1, N] torch tensor"""
        T = speech_feat.shape[2]

        # 1. F0 prediction via DLC
        mel_ncf = speech_feat.detach().cpu().numpy().astype(np.float32)
        mel_padded = np.zeros((1, 80, self.TARGET_T), dtype=np.float32)
        mel_padded[:, :, :T] = mel_ncf

        f0_raw = self.f0_session.run({"speech_feat": mel_padded})
        f0_out = self.f0_session.fetch(["f0"])
        f0 = torch.from_numpy(f0_out["f0"].reshape(1, self.TARGET_T)[:, :T])

        # 2. Source generation (PyTorch)
        s = self.hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.hift.m_source(s)
        s = s.transpose(1, 2)  # [1, 1, audio_len]

        # 3. Compute source STFT (PyTorch)
        with torch.no_grad():
            s_stft_real, s_stft_imag = self.hift._stft(s.squeeze(1))
            s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)  # [1, 18, stft_frames]
        stft_frames = s_stft.shape[2]

        # 4. Decode via DLC
        mel_padded2 = np.zeros((1, 80, self.TARGET_T), dtype=np.float32)
        mel_padded2[:, :, :T] = mel_ncf
        s_stft_ncf = s_stft.detach().cpu().numpy().astype(np.float32)
        use_frames = min(stft_frames, self.TARGET_STFT)
        if stft_frames > self.TARGET_STFT:
            print(f"    [vocoder] truncating stft: {stft_frames} -> {self.TARGET_STFT}")
            s_stft_ncf = s_stft_ncf[:, :, :self.TARGET_STFT]
        s_stft_padded = np.zeros((1, 18, self.TARGET_STFT), dtype=np.float32)
        s_stft_padded[:, :, :use_frames] = s_stft_ncf[:, :, :use_frames]

        self.decode_session.run({"speech_feat": mel_padded2, "s_stft": s_stft_padded})
        dec_out = self.decode_session.fetch(["magnitude", "phase"])
        mag = torch.from_numpy(dec_out["magnitude"].reshape(1, 9, self.TARGET_STFT)[:, :, :use_frames])
        phase = torch.from_numpy(dec_out["phase"].reshape(1, 9, self.TARGET_STFT)[:, :, :use_frames])

        # 5. ISTFT (PyTorch)
        mag = torch.clip(mag, max=1e2)
        real = mag * torch.cos(phase)
        img = mag * torch.sin(phase)
        audio = torch.istft(torch.complex(real, img),
                            self.hift.istft_params["n_fft"],
                            self.hift.istft_params["hop_len"],
                            self.hift.istft_params["n_fft"],
                            window=self.hift.stft_window)
        audio = torch.clamp(audio, -self.hift.audio_limit, self.hift.audio_limit)
        return audio.unsqueeze(0)

    def release(self):
        self.f0_session.release()
        self.decode_session.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。")
    parser.add_argument("--prompt-wav", default=os.path.join(PROJECT_ROOT, "assets", "zero_shot_prompt.wav"))
    parser.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "pretrained", "CosyVoice-300M"))
    parser.add_argument("--dlc-dir", default=os.path.join(PROJECT_ROOT, "dlc"))
    parser.add_argument("--runtime", default="CPU", choices=["CPU", "GPU", "DSP"])
    parser.add_argument("--vocoder-runtime", default=None, choices=["CPU", "GPU", "DSP"],
                        help="Vocoder backend (default: same as --runtime). Use DSP for INT8 vocoder DLCs.")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    vocoder_runtime = args.vocoder_runtime or args.runtime

    print(f"text={args.text}")
    print(f"out={args.out}")
    print(f"runtime(estimator)={args.runtime}, runtime(vocoder)={vocoder_runtime}")
    print(f"model_dir={args.model_dir}")
    print(f"dlc_dir={args.dlc_dir}")

    # Load CosyVoice
    print("loading CosyVoice model...")
    t0 = time.time()
    model = CosyVoice(args.model_dir)
    print(f"model loaded in {time.time()-t0:.1f}s")

    # Flow estimator: INT8 only works on DSP, FP32 on CPU
    est_int8 = os.path.join(args.dlc_dir, "flow.decoder.estimator_int8.dlc")
    est_fp32 = os.path.join(args.dlc_dir, "flow.decoder.estimator.fp32.dlc")
    if args.runtime == "DSP" and os.path.exists(est_int8):
        est_dlc = est_int8
    else:
        est_dlc = est_fp32
        if args.runtime != "CPU":
            print("Note: flow estimator INT8 needs DSP, using FP32 on CPU")
    print(f"initializing flow estimator DLC ({os.path.basename(est_dlc)})...")
    estimator_dlc = Dlcsession(est_dlc, runtime="CPU" if est_dlc == est_fp32 else args.runtime)
    estimator_dlc.init()

    decoder = model.model.flow.decoder

    class EstimatorWrapper:
        def __init__(self, session, target_seq_len=500):
            self.session = session
            self.target = target_seq_len
            self.call_count = 0

        def __call__(self, x, mask, mu, t, spks, cond, streaming=False):
            self.call_count += 1
            orig_len = x.size(2)

            def pad3d(tensor):
                if tensor.ndim == 3 and tensor.size(2) == orig_len:
                    if orig_len < self.target:
                        p = torch.zeros(tensor.size(0), tensor.size(1), self.target - orig_len)
                        return torch.cat([tensor, p], dim=2)
                    else:
                        return tensor[:, :, :self.target]
                return tensor

            x_p = pad3d(x); mask_p = pad3d(mask); mu_p = pad3d(mu); cond_p = pad3d(cond)

            def to_nfc(t):
                return t.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 1).copy()

            inp = {
                "x": to_nfc(x_p), "mask": mask_p.detach().cpu().numpy().astype(np.float32),
                "mu": to_nfc(mu_p), "t": t.detach().cpu().numpy().astype(np.float32),
                "spks": spks.detach().cpu().numpy().astype(np.float32),
                "cond": to_nfc(cond_p),
            }
            self.session.run(inp)
            out = self.session.fetch(["estimator_out"])
            out_data = out["estimator_out"].reshape(2, self.target, 80).transpose(0, 2, 1)
            result = torch.from_numpy(out_data).to(x.device).to(x.dtype)
            if result.size(2) > orig_len:
                result = result[:, :, :orig_len]
            elif result.size(2) < orig_len:
                p = torch.zeros(result.size(0), result.size(1), orig_len - result.size(2))
                result = torch.cat([result, p], dim=2)
            return result

    est_wrapper = EstimatorWrapper(estimator_dlc)
    decoder.forward_estimator = lambda x, mask, mu, t, spks, cond, streaming=False: est_wrapper(x, mask, mu, t, spks, cond, streaming=streaming)
    print("flow estimator DLC ready")

    # Replace HiFT vocoder with DLC chain
    print("initializing vocoder DLCs...")
    vocoder = VocoderDLC(model.model.hift, args.dlc_dir, runtime=vocoder_runtime)
    original_inference = model.model.hift.inference

    def snpe_inference(self, speech_feat, cache_source=torch.zeros(1, 1, 0)):
        audio = vocoder(speech_feat)
        # Match original interface: return audio, source
        return audio, torch.zeros(1, 1, 0)

    model.model.hift.inference = snpe_inference.__get__(model.model.hift)
    print("vocoder DLCs ready")

    # Run inference
    print(f"running: {args.text}")
    t0 = time.time()
    generated = False
    try:
        for idx, result in enumerate(
            model.inference_zero_shot(args.text, args.prompt_text, args.prompt_wav, speed=args.speed)
        ):
            out_path = args.out if idx == 0 else args.out.replace(".wav", f"_{idx}.wav")
            samples = result["tts_speech"].squeeze().numpy()
            sf.write(out_path, samples, model.sample_rate)
            dur = len(samples) / model.sample_rate
            print(f"saved={out_path} dur={dur:.1f}s time={time.time()-t0:.1f}s "
                  f"estimator_calls={est_wrapper.call_count}")
            generated = True
    finally:
        estimator_dlc.release()
        vocoder.release()

    if not generated:
        raise RuntimeError("no audio generated")


if __name__ == "__main__":
    main()
