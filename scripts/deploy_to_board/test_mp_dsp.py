"""Test: two separate Python processes each loading a different DSP DLC."""
import subprocess, sys, os, time

BASE = "/home/fibo/AI model/tts_models/cosyvoice_snpe/dlc"

# Worker code as string
WORKER = """
import os, time, sys
sys.path.insert(0, '/home/fibo/AI model/tts_models/cosyvoice_snpe')
from fiboaisdk.api_aisdk_py import api_infer_py

dlc = sys.argv[1]
tag = sys.argv[2]
print(f'[{tag}] PID={os.getpid()} loading...')
p = api_infer_py.InferParams(dlc, 'QUALCOMM', 'SNPE', 'DSP', 'ERROR', 5)
api = api_infer_py.InferAPI()
ret = api.Init(p)
print(f'[{tag}] Init={ret}')
if ret == 0:
    time.sleep(30)
    api.Release()
print(f'[{tag}] done')
"""

# Write worker script
with open("/tmp/dsp_worker_test.py", "w") as f:
    f.write(WORKER)

# Launch f0 worker
print("Launching f0 DSP worker...")
p1 = subprocess.Popen(
    [sys.executable, "/tmp/dsp_worker_test.py",
     f"{BASE}/hift_f0_predictor_int8.dlc", "f0"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
time.sleep(5)

# Launch decode worker
print("Launching decode DSP worker...")
p2 = subprocess.Popen(
    [sys.executable, "/tmp/dsp_worker_test.py",
     f"{BASE}/hift_decode_pre_istft_int8.dlc", "decode"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

# Collect output
time.sleep(10)
for tag, p in [("f0", p1), ("decode", p2)]:
    print(f"\n=== {tag} output ===")
    try:
        out, _ = p.communicate(timeout=30)
        for line in out.split("\n"):
            if any(kw in line for kw in ["Init", "PID", "done", "loading"]):
                print(f"  {line.strip()}")
    except:
        p.kill()
        print(f"  (killed)")

# Check OOM
os.system("dmesg | grep -i 'oom\\|kill' | tail -3")
print("\nTest done!")
