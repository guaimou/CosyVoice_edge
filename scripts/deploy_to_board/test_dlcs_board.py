"""Quick validation of all DLCs on SC171v3 board.

Tests each DLC individually with synthetic data to verify they load and run.
Run on board: python3 test_dlcs_board.py
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fiboaisdk.api_aisdk_py import api_infer_py


def test_dlc(dlc_path, framework, runtime, inputs_dict, output_name, description):
    """Test a single DLC: init, execute, check output shape."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"  DLC: {dlc_path}")
    print(f"  Framework: {framework}, Runtime: {runtime}")

    if not os.path.exists(dlc_path):
        print(f"  SKIP: DLC file not found")
        return False

    params = api_infer_py.InferParams(dlc_path, "QUALCOMM", framework, runtime, "ERROR", 5)
    api = api_infer_py.InferAPI()

    t0 = time.time()
    ret = api.Init(params)
    print(f"  Init: {ret} ({time.time()-t0:.1f}s)")

    # Flatten inputs for the API
    input_lists = {k: v.flatten().tolist() for k, v in inputs_dict.items()}

    t0 = time.time()
    ret = api.Execute_float(input_lists)
    print(f"  Execute: {ret} ({time.time()-t0:.1f}s)")

    result = api.FetchOutputs_float([output_name])
    api.Release()

    if result and output_name in result:
        out_data = np.array(result[output_name], dtype=np.float32)
        print(f"  Output shape: {out_data.shape}")
        print(f"  Output range: [{out_data.min():.4f}, {out_data.max():.4f}]")
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL: no output")
        return False


def main():
    dlc_dir = os.path.join(PROJECT_ROOT, "dlc")
    framework = "SNPE"
    runtime = "CPU"

    tests = [
        {
            "dlc": "campplus.dlc",
            "inputs": {"input": np.random.randn(1, 80, 200).astype(np.float32)},
            "output": "output",
            "desc": "campplus speaker embedding (NCF input [1,80,T])",
        },
        {
            "dlc": "flow.decoder.estimator.fp32.dlc",
            "inputs": {
                "x": np.random.randn(2, 500, 80).astype(np.float32),
                "mask": np.ones((2, 1, 500), dtype=np.float32),
                "mu": np.random.randn(2, 500, 80).astype(np.float32),
                "t": np.zeros(2, dtype=np.float32),
                "spks": np.zeros((2, 80), dtype=np.float32),
                "cond": np.random.randn(2, 500, 80).astype(np.float32),
            },
            "output": "estimator_out",
            "desc": "flow estimator (NFC inputs [B,500,80])",
        },
        {
            "dlc": "hift_f0_predictor.dlc",
            "inputs": {"speech_feat": np.random.randn(1, 80, 200).astype(np.float32)},
            "output": "f0",
            "desc": "HiFT f0 predictor (NCF input [1,80,T])",
        },
        {
            "dlc": "hift_decode_pre_istft.dlc",
            "inputs": {
                "speech_feat": np.random.randn(1, 80, 100).astype(np.float32),
                "s_stft": np.random.randn(1, 18, 6401).astype(np.float32),
            },
            "output": "magnitude",
            "desc": "HiFT decode pre-ISTFT (NCF inputs)",
        },
    ]

    results = {}
    for t in tests:
        dlc_path = os.path.join(dlc_dir, t["dlc"])
        ok = test_dlc(dlc_path, framework, runtime, t["inputs"], t["output"], t["desc"])
        results[t["dlc"]] = ok

    print(f"\n{'='*60}")
    print("SUMMARY:")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")


if __name__ == "__main__":
    main()
