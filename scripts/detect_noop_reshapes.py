"""Detect all no-op Reshape nodes using ORT with real inputs.

Only checks Reshape nodes whose data input is a float tensor (not int64 shape
computation tensors), avoiding type-conflict errors when adding them as outputs.
"""

from pathlib import Path
import argparse

import numpy as np
import onnx
import onnxruntime
from onnx import helper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"
PREP_DIR = OUTPUT_DIR / "flow_decoder_snpe_prep"
DEFAULT_ESTIMATOR = MODEL_DIR / "flow.decoder.estimator.fp32.onnx"


def load_raw(path: Path, shape: tuple[int, ...], order: str = "C") -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape, order=order)


def detect_noop_reshapes(model: onnx.ModelProto, seq_len: int) -> dict:
    """Run ORT with all float-type Reshape data inputs and outputs to get shapes."""
    # Staticize inputs
    for vi in model.graph.input:
        for dim in vi.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                dim.dim_value = seq_len

    # Collect all Reshape nodes
    reshape_nodes = []
    for node in model.graph.node:
        if node.op_type == "Reshape":
            reshape_nodes.append((node.name, node.input[0], node.output[0]))

    print(f"  Checking {len(reshape_nodes)} Reshape nodes...")

    # Create temp model with added outputs — use empty type so ORT infers it
    temp_model = onnx.ModelProto()
    temp_model.CopyFrom(model)

    existing = {v.name for v in temp_model.graph.output}
    fetch_names = []
    for _, di, out in reshape_nodes:
        if di not in existing:
            temp_model.graph.output.append(helper.make_empty_tensor_value_info(di))
            fetch_names.append(di)
            existing.add(di)
        if out not in existing:
            temp_model.graph.output.append(helper.make_empty_tensor_value_info(out))
            fetch_names.append(out)
            existing.add(out)

    temp_path = OUTPUT_DIR / "_temp_reshape_detect.onnx"
    onnx.save(temp_model, str(temp_path))

    try:
        raw_dir = PREP_DIR / "raw"
        nfc_inputs = {
            "x": load_raw(raw_dir / "x.raw", (2, 80, seq_len)),
            "mask": load_raw(raw_dir / "mask.raw", (2, 1, seq_len)),
            "mu": load_raw(raw_dir / "mu.raw", (2, 80, seq_len)),
            "t": load_raw(raw_dir / "t.raw", (2,)),
            "spks": load_raw(raw_dir / "spks.raw", (2, 80)),
            "cond": load_raw(raw_dir / "cond.raw", (2, 80, seq_len)),
        }
        session = onnxruntime.InferenceSession(str(temp_path), providers=["CPUExecutionProvider"])
        all_names = ["estimator_out"] + fetch_names
        outputs = session.run(all_names, nfc_inputs)
        shape_map = {name: list(val.shape) for name, val in zip(all_names, outputs)}
    finally:
        if temp_path.exists():
            temp_path.unlink()

    # Classify each Reshape
    noop_map = {}
    non_noop = []
    for node_name, di, out in reshape_nodes:
        in_shape = shape_map.get(di)
        out_shape = shape_map.get(out)
        if in_shape is None or out_shape is None:
            continue
        if in_shape == out_shape:
            noop_map[out] = (di, in_shape, node_name)
        else:
            non_noop.append((node_name, di, in_shape, out, out_shape))

    print(f"  Found {len(noop_map)} no-op Reshapes, {len(non_noop)} shape-changing Reshapes")

    if non_noop:
        print("\n  Shape-changing Reshapes (not bypassed):")
        for node_name, di, in_s, out, out_s in non_noop:
            print(f"    {node_name}: {di} {in_s} -> {out} {out_s}")

    return noop_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_ESTIMATOR))
    parser.add_argument("--seq-len", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()

    model = onnx.load(str(input_path))
    noop_map = detect_noop_reshapes(model, args.seq_len)

    print(f"\n=== All {len(noop_map)} no-op Reshape nodes ===\n")
    for output_name, (data_input, shape, node_name) in sorted(noop_map.items()):
        print(f"  {node_name:45s} {output_name:45s} <- {data_input} shape={shape}")

    already_bypassed = {
        "/Reshape_output_0", "/Reshape_1_output_0", "/Reshape_3_output_0",
        "/Reshape_4_output_0", "/Reshape_5_output_0", "/Reshape_6_output_0",
    }
    new = set(noop_map.keys()) - already_bypassed
    if new:
        print(f"\n{len(new)} NEW no-op Reshapes not yet bypassed:")
        for name in sorted(new):
            di, shape, node_name = noop_map[name]
            print(f"    {node_name}: {di} -> {name} shape={shape}")
        print("\n# Add to DEFAULT_NOOP_OUTPUTS in bypass_noop_reshapes.py:")
        for name in sorted(new):
            print(f'    "{name}",')
    else:
        print("\nNo new no-op Reshapes found beyond the 6 already bypassed.")


if __name__ == "__main__":
    main()
