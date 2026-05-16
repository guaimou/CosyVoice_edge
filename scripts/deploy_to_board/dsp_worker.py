"""DSP worker process: loads one DLC, waits for input files, runs inference, writes output."""
import sys, os, time, numpy as np
from fiboaisdk.api_aisdk_py import api_infer_py

dlc_path = sys.argv[1]
worker_id = sys.argv[2] if len(sys.argv) > 2 else "0"

print(f"[worker {worker_id}] PID={os.getpid()} loading {dlc_path}")
p = api_infer_py.InferParams(dlc_path, "QUALCOMM", "SNPE", "DSP", "ERROR", 5)
api = api_infer_py.InferAPI()
ret = api.Init(p)
print(f"[worker {worker_id}] Init={ret}")

if ret != 0:
    sys.exit(1)

# Signal ready
print(f"[worker {worker_id}] READY", flush=True)

# Wait for work (simple: poll for input file)
work_dir = f"/tmp/dsp_work_{worker_id}"
os.makedirs(work_dir, exist_ok=True)

while True:
    signal_file = os.path.join(work_dir, "signal.txt")
    if os.path.exists(signal_file):
        with open(signal_file) as f:
            cmd = f.read().strip()
        os.remove(signal_file)

        if cmd == "exit":
            break

        # Load inputs
        inputs = {}
        for fname in os.listdir(work_dir):
            if fname.endswith(".raw") and not fname.startswith("out_"):
                name = fname.replace(".raw", "")
                data = np.fromfile(os.path.join(work_dir, fname), dtype=np.float32)
                inputs[name] = data.flatten().tolist()

        if not inputs:
            time.sleep(0.01)
            continue

        # Run inference
        t0 = time.time()
        ret = api.Execute_float(inputs)
        dt = time.time() - t0

        # Fetch all outputs and write
        # We need to know output names - infer from DLC
        if "f0_predictor" in dlc_path:
            out_names = ["f0"]
        elif "decode" in dlc_path:
            out_names = ["magnitude", "phase"]
        elif "campplus" in dlc_path:
            out_names = ["output"]
        elif "estimator" in dlc_path:
            out_names = ["estimator_out"]
        else:
            out_names = []

        if ret == 0 and out_names:
            result = api.FetchOutputs_float(out_names)
            for name, data in result.items():
                arr = np.array(data, dtype=np.float32)
                arr.tofile(os.path.join(work_dir, f"out_{name}.raw"))

        # Signal done
        with open(os.path.join(work_dir, "done.txt"), "w") as f:
            f.write(f"{ret} {dt:.4f}")

    time.sleep(0.01)

api.Release()
print(f"[worker {worker_id}] exited")
