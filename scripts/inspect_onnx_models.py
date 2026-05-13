from pathlib import Path
import argparse

import onnxruntime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"


def describe_value(value) -> str:
    shape = [dim if dim is not None else "?" for dim in value.shape]
    return f"name={value.name} type={value.type} shape={shape}"


def inspect_onnx(path: Path) -> None:
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    print(f"onnx_file={path}")
    print(f"providers={session.get_providers()}")
    print("inputs_start")
    for value in session.get_inputs():
        print(describe_value(value))
    print("inputs_end")
    print("outputs_start")
    for value in session.get_outputs():
        print(describe_value(value))
    print("outputs_end")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--pattern", default="*.onnx")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    matches = sorted(model_dir.glob(args.pattern))
    if not matches:
        raise FileNotFoundError(f"no ONNX files matched pattern {args.pattern!r} in {model_dir}")

    print(f"model_dir={model_dir}")
    for path in matches:
        inspect_onnx(path)


if __name__ == "__main__":
    main()
