"""Test: Init DSP, Release, Load LLM, Re-Init DSP, Infer."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosyvoice_src"))
from fiboaisdk.api_aisdk_py import api_infer_py

dlc = "dlc/flow.decoder.estimator_int8.dlc"

# 1. Warm DSP
print("=== Phase 1: Warm DSP ===")
t0 = time.time()
p = api_infer_py.InferParams(dlc, "QUALCOMM", "SNPE", "DSP", "ERROR", 5)
api = api_infer_py.InferAPI()
print("Init1:", api.Init(p), "%.1fs" % (time.time()-t0))
api.Release()
print("Released")

# 2. Load LLM
print("=== Phase 2: Load LLM ===")
t0 = time.time()
from cosyvoice.cli.cosyvoice import CosyVoice
model = CosyVoice("pretrained/CosyVoice-300M")
print("LLM loaded: %.1fs" % (time.time()-t0))

# 3. Re-Init DSP
print("=== Phase 3: Re-Init DSP ===")
t0 = time.time()
p2 = api_infer_py.InferParams(dlc, "QUALCOMM", "SNPE", "DSP", "ERROR", 5)
api2 = api_infer_py.InferAPI()
print("Init2:", api2.Init(p2), "%.1fs" % (time.time()-t0))

# 4. Infer
print("=== Phase 4: Infer ===")
inp = {}
for k in ["x", "mu", "cond"]:
    inp[k] = np.random.randn(2, 500, 80).astype(np.float32).flatten().tolist()
inp["mask"] = np.ones((2, 1, 500), dtype=np.float32).flatten().tolist()
inp["t"] = np.zeros(2, dtype=np.float32).flatten().tolist()
inp["spks"] = np.zeros((2, 80), dtype=np.float32).flatten().tolist()
t0 = time.time()
api2.Execute_float(inp)
dt = time.time() - t0
out = api2.FetchOutputs_float(["estimator_out"])
shape = str(np.array(out["estimator_out"]).shape) if out else "NONE"
print("Exec: %.3fs, output: %s" % (dt, shape))
api2.Release()
print("DONE - DSP Re-Init + LLM works!")
