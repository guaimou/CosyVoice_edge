"""Test DSP DLC init BEFORE LLM load to check OOM ordering."""
import os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cosyvoice_src"))

from fiboaisdk.api_aisdk_py import api_infer_py

# 1. Init f0 INT8 on DSP FIRST
print("Init f0 INT8 DSP...")
p = api_infer_py.InferParams(
    os.path.join(PROJECT_ROOT, "dlc/hift_f0_predictor_int8.dlc"),
    "QUALCOMM", "SNPE", "DSP", "ERROR", 5)
api = api_infer_py.InferAPI()
ret = api.Init(p)
print(f"f0 Init: {ret}")
api.Release()
print("f0 released, mem ok")

# 2. Now load LLM
print("Loading CosyVoice...")
from cosyvoice.cli.cosyvoice import CosyVoice
model = CosyVoice(os.path.join(PROJECT_ROOT, "pretrained/CosyVoice-300M"))
print("LLM loaded OK after DSP init")
