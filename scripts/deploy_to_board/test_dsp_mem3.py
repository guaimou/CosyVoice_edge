"""Test dual-DSP with small decode DLC."""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fiboaisdk.api_aisdk_py import api_infer_py

dlc_dir = os.path.join(PROJECT_ROOT, "dlc")

# 1. Init f0 INT8 DSP + small decode INT8 DSP simultaneously
sessions = []
for name in ["hift_f0_predictor_int8.dlc", "hift_decode_pre_istft_small_int8.dlc"]:
    path = os.path.join(dlc_dir, name)
    print(f"Init {name}...")
    p = api_infer_py.InferParams(path, "QUALCOMM", "SNPE", "DSP", "ERROR", 5)
    api = api_infer_py.InferAPI()
    ret = api.Init(p)
    print(f"  {name}: {ret}")
    if ret != 0:
        print("FAILED")
        sys.exit(1)
    sessions.append(api)

print(f"Both DSP DLCs alive ({len(sessions)} sessions) - SUCCESS!")

# 2. Load LLM
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cosyvoice_src"))
print("Loading CosyVoice...")
from cosyvoice.cli.cosyvoice import CosyVoice
model = CosyVoice(os.path.join(PROJECT_ROOT, "pretrained/CosyVoice-300M"))
print("LLM loaded OK with 2x small DSP DLCs!")

for api in sessions:
    api.Release()
print("Done")
