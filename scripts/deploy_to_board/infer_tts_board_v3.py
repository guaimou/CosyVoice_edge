"""CosyVoice TTS on SC171v3 - Optimized: flow estimator INT8 DSP, rest CPU.

Single DSP session for the bottleneck (flow estimator), everything else on CPU.
Session caching: Init once, reuse across all 10 ODE steps.

Usage:
  python3 infer_tts_board_v3.py --text '你好' --out /tmp/test.wav
"""

import os, sys, time, argparse
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for p in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "cosyvoice_src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cosyvoice.cli.cosyvoice import CosyVoice
from fiboaisdk.api_aisdk_py import api_infer_py


class Dlcsession:
    def __init__(self, dlc_path, framework="SNPE", runtime="CPU", profile=5):
        self.dlc_path = dlc_path
        self.params = api_infer_py.InferParams(dlc_path, "QUALCOMM", framework, runtime, "ERROR", profile)
        self.api = api_infer_py.InferAPI()
        self._init = False

    def init(self):
        if not self._init:
            ret = self.api.Init(self.params)
            if ret != 0:
                raise RuntimeError(f"DLC init failed: {self.dlc_path} code={ret}")
            self._init = True

    def run(self, input_feed):
        inp = {k: v.astype(np.float32).flatten().tolist() for k, v in input_feed.items()}
        ret = self.api.Execute_float(inp)
        if ret != 0:
            raise RuntimeError(f"DLC exec failed: {self.dlc_path} code={ret}")

    def fetch(self, names):
        result = self.api.FetchOutputs_float(names)
        return {k: np.array(v, dtype=np.float32) for k, v in result.items()}

    def release(self):
        if self._init:
            self.api.Release()
            self._init = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。")
    parser.add_argument("--prompt-wav", default=os.path.join(PROJECT_ROOT, "assets", "zero_shot_prompt.wav"))
    parser.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "pretrained", "CosyVoice-300M"))
    parser.add_argument("--dlc-dir", default=os.path.join(PROJECT_ROOT, "dlc"))
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    print(f"text={args.text}")
    print(f"out={args.out}")
    print(f"model_dir={args.model_dir}")

    dlc_dir = args.dlc_dir

    # === Step 1: Load CosyVoice FIRST (LLM needs 4.5GB, DSP needs remaining) ===
    print("loading CosyVoice model...")
    t0 = time.time()
    model = CosyVoice(args.model_dir)
    print(f"  model loaded ({time.time()-t0:.1f}s)")

    # === Step 2: Init flow estimator INT8 on DSP (after LLM, fits in remaining ~1GB) ===
    print("initializing flow estimator INT8 DSP...")
    est_dlc = os.path.join(dlc_dir, "flow.decoder.estimator_int8.dlc")
    if not os.path.exists(est_dlc):
        est_dlc = os.path.join(dlc_dir, "flow.decoder.estimator.fp32.dlc")
        print("  INT8 DLC not found, using FP32 CPU")
        runtime = "CPU"
    else:
        runtime = "DSP"
    print(f"  DLC: {os.path.basename(est_dlc)}, runtime: {runtime}")
    t0 = time.time()
    est_session = Dlcsession(est_dlc, runtime=runtime)
    est_session.init()
    print(f"  estimator ready ({time.time()-t0:.1f}s)")

    # === Step 3: Patch flow estimator ===
    print("patching flow estimator...")
    decoder = model.model.flow.decoder

    class EstWrapper:
        def __init__(self, session, target=500):
            self.session = session
            self.target = target
            self.calls = 0
        def __call__(self, x, mask, mu, t, spks, cond, streaming=False):
            self.calls += 1
            ol = x.size(2)
            def pad3d(ten):
                if ten.ndim == 3 and ten.size(2) == ol:
                    if ol < self.target:
                        p = torch.zeros(ten.size(0), ten.size(1), self.target - ol)
                        return torch.cat([ten, p], dim=2)
                    else:
                        return ten[:, :, :self.target]
                return ten
            xp, mp, mup, cp = pad3d(x), pad3d(mask), pad3d(mu), pad3d(cond)
            def nfc(t): return t.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 1).copy()
            inp = {"x": nfc(xp), "mask": mp.detach().cpu().numpy().astype(np.float32),
                   "mu": nfc(mup), "t": t.detach().cpu().numpy().astype(np.float32),
                   "spks": spks.detach().cpu().numpy().astype(np.float32),
                   "cond": nfc(cp)}
            self.session.run(inp)
            out = self.session.fetch(["estimator_out"])
            od = out["estimator_out"].reshape(2, self.target, 80).transpose(0, 2, 1)
            res = torch.from_numpy(od).to(x.device).to(x.dtype)
            if res.size(2) > ol: res = res[:, :, :ol]
            elif res.size(2) < ol:
                p = torch.zeros(res.size(0), res.size(1), ol - res.size(2))
                res = torch.cat([res, p], dim=2)
            return res

    est_w = EstWrapper(est_session)
    decoder.forward_estimator = lambda x, mask, mu, t, spks, cond, streaming=False: est_w(x, mask, mu, t, spks, cond, streaming=streaming)
    print("  estimator patched")

    # === Step 4: Run TTS ===
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
            total_t = time.time() - t0
            print(f"saved={out_path} dur={dur:.1f}s total={total_t:.1f}s estimator_calls={est_w.calls}")
            generated = True
    finally:
        est_session.release()

    if not generated:
        raise RuntimeError("no audio generated")


if __name__ == "__main__":
    main()
