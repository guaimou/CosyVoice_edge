from pathlib import Path
import argparse
import json

import numpy as np
import onnxruntime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
PREP_DIR = OUTPUT_DIR / "flow_decoder_snpe_prep"
MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"
DEFAULT_ESTIMATOR = MODEL_DIR / "flow.decoder.estimator.fp32.onnx"


def load_raw(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def compare(name: str, actual: np.ndarray, reference: np.ndarray) -> dict:
    diff = actual - reference
    abs_diff = np.abs(diff)
    return {
        "name": name,
        "shape": list(actual.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-dir", default=str(PREP_DIR))
    parser.add_argument("--estimator", default=str(DEFAULT_ESTIMATOR))
    args = parser.parse_args()

    prep_dir = Path(args.prep_dir)
    estimator_path = Path(args.estimator)
    if not prep_dir.is_absolute():
        prep_dir = (PROJECT_ROOT / prep_dir).resolve()
    if not estimator_path.is_absolute():
        estimator_path = (PROJECT_ROOT / estimator_path).resolve()

    metadata = json.loads((prep_dir / "prep_metadata.json").read_text(encoding="utf-8"))
    seq_len = metadata["target_seq_len"]
    raw_dir = prep_dir / "raw"

    ncf_inputs = {
        "x": load_raw(raw_dir / "x.raw", (2, 80, seq_len)).transpose(0, 2, 1),
        "mask": load_raw(raw_dir / "mask.raw", (2, 1, seq_len)).transpose(0, 2, 1),
        "mu": load_raw(raw_dir / "mu.raw", (2, 80, seq_len)).transpose(0, 2, 1),
        "t": load_raw(raw_dir / "t.raw", (2,)),
        "spks": load_raw(raw_dir / "spks.raw", (2, 80)),
        "cond": load_raw(raw_dir / "cond.raw", (2, 80, seq_len)).transpose(0, 2, 1),
    }
    nfc_inputs = {
        "x": load_raw(raw_dir / "x.raw", (2, 80, seq_len)),
        "mask": load_raw(raw_dir / "mask.raw", (2, 1, seq_len)),
        "mu": load_raw(raw_dir / "mu.raw", (2, 80, seq_len)),
        "t": load_raw(raw_dir / "t.raw", (2,)),
        "spks": load_raw(raw_dir / "spks.raw", (2, 80)),
        "cond": load_raw(raw_dir / "cond.raw", (2, 80, seq_len)),
    }

    session = onnxruntime.InferenceSession(str(estimator_path), providers=["CPUExecutionProvider"])
    input_specs = [{"name": value.name, "shape": value.shape, "type": value.type} for value in session.get_inputs()]
    output_names = [value.name for value in session.get_outputs()]

    comparison_runs = []
    chosen_output = None
    chosen_layout = None
    last_error = None
    for layout_name, ort_inputs in [("ncf", ncf_inputs), ("nfc", nfc_inputs)]:
        try:
            output = session.run(None, ort_inputs)[0]
            chosen_output = output
            chosen_layout = layout_name
            break
        except Exception as exc:
            comparison_runs.append({"layout": layout_name, "error": str(exc)})
            last_error = exc

    if chosen_output is None:
        raise RuntimeError(str(last_error))

    ort_output = chosen_output.astype(np.float32, copy=False)
    if chosen_layout == "ncf":
        ort_output = ort_output.transpose(0, 2, 1)
    reference = np.load(prep_dir / "reference" / "estimator_reference.npy").astype(np.float32, copy=False)

    report = {
        "estimator": str(estimator_path),
        "seq_len": seq_len,
        "chosen_layout": chosen_layout,
        "attempts": comparison_runs,
        "onnx_io": {
            "inputs": input_specs,
            "outputs": output_names,
        },
        "comparison": compare("onnx_vs_pytorch", ort_output, reference),
    }
    (prep_dir / "ort_compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(prep_dir / "ort_output.npy", ort_output)

    print(f"estimator={estimator_path}")
    print(f"seq_len={seq_len}")
    print(f"chosen_layout={chosen_layout}")
    print(f"inputs={[spec['name'] for spec in input_specs]}")
    print(f"outputs={output_names}")
    print(f"max_abs_diff={report['comparison']['max_abs_diff']}")
    print(f"mean_abs_diff={report['comparison']['mean_abs_diff']}")
    print(f"rmse={report['comparison']['rmse']}")
    print(f"report={prep_dir / 'ort_compare.json'}")


if __name__ == "__main__":
    main()
