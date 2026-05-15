"""Remove no-op Reshape nodes from the estimator ONNX model to avoid SNPE layout reinterpretation.

The Reshape nodes on x, mu, and cond are no-ops under static seq_len=500 because
they reshape (2,80,500) -> (2,80,500). SNPE's Reshape implementation applies an
internal layout reinterpretation that scrambles the data. Bypassing these no-op
Reshapes eliminates the layout issue.
"""

from pathlib import Path
import argparse

import numpy as np
import onnx
import onnxruntime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"
PREP_DIR = OUTPUT_DIR / "flow_decoder_snpe_prep"
DEFAULT_ESTIMATOR = MODEL_DIR / "flow.decoder.estimator.fp32.onnx"
DEFAULT_OUTPUT = OUTPUT_DIR / "flow.decoder.estimator.noop_reshapes_bypassed.onnx"

# Reshape outputs known to be no-ops when seq_len is staticized.
# Each entry is the output tensor name of the Reshape to bypass.
DEFAULT_NOOP_OUTPUTS = [
    # Early branch input Reshapes
    "/Reshape_output_0",      # Reshape on x: (2,80,500) -> (2,80,500)
    "/Reshape_1_output_0",    # Reshape on mu: (2,80,500) -> (2,80,500)
    "/Reshape_6_output_0",    # Reshape on cond: (2,80,500) -> (2,80,500)
    # Early merge Reshapes
    "/Reshape_3_output_0",    # Reshape on Concat_2 (x+mu): (2,160,500) -> (2,160,500)
    "/Reshape_4_output_0",    # Reshape on Expand (spks): (2,80,500) -> (2,80,500)
    "/Reshape_5_output_0",    # Reshape on Concat_6 (x/mu+spks): (2,240,500) -> (2,240,500)
    # UNet skip-connection Reshapes (mid_blocks / up_blocks boundaries)
    "/Reshape_7_output_0",    # Reshape on Slice_1: (2,256,250) -> (2,256,250), feeds Concat_12
    "/Reshape_8_output_0",    # Reshape on Transpose_3: (2,256,250) -> (2,256,250), feeds Concat_12
    "/Reshape_9_output_0",    # Reshape on Slice_2: (2,256,500) -> (2,256,500), feeds Concat_15
    "/Reshape_10_output_0",   # Reshape on Transpose_1: (2,256,500) -> (2,256,500), feeds Concat_15
]

# NOTE: /Reshape_2 is intentionally excluded — it is a shape-computation Reshape
# (reshape [3] -> [3] for int64 shape tensors) and should not be bypassed.


def load_raw(path: Path, shape: tuple[int, ...], order: str = "C") -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape, order=order)


def run_ort(estimator_path: Path, seq_len: int) -> np.ndarray:
    raw_dir = PREP_DIR / "raw"
    nfc_inputs = {
        "x": load_raw(raw_dir / "x.raw", (2, 80, seq_len)),
        "mask": load_raw(raw_dir / "mask.raw", (2, 1, seq_len)),
        "mu": load_raw(raw_dir / "mu.raw", (2, 80, seq_len)),
        "t": load_raw(raw_dir / "t.raw", (2,)),
        "spks": load_raw(raw_dir / "spks.raw", (2, 80)),
        "cond": load_raw(raw_dir / "cond.raw", (2, 80, seq_len)),
    }
    session = onnxruntime.InferenceSession(str(estimator_path), providers=["CPUExecutionProvider"])
    return session.run(["estimator_out"], nfc_inputs)[0]


def bypass_reshapes(model: onnx.ModelProto, noop_outputs: set[str]) -> onnx.ModelProto:
    """Remove no-op Reshape nodes and reconnect consumers to the data input."""
    nodes_to_remove = []

    for node in model.graph.node:
        if node.op_type == "Reshape" and node.output[0] in noop_outputs:
            data_input = node.input[0]
            reshaped_output = node.output[0]

            # Redirect all consumers of reshaped_output to data_input
            for other in model.graph.node:
                for idx, name in enumerate(other.input):
                    if name == reshaped_output:
                        other.input[idx] = data_input

            nodes_to_remove.append(node)
            print(f"  bypassing {node.name!r}: {data_input} -> {reshaped_output}")

    for node in nodes_to_remove:
        model.graph.node.remove(node)

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_ESTIMATOR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seq-len", type=int, default=500)
    parser.add_argument("--extra-noop", nargs="*", default=[], help="Additional Reshape output names to bypass")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()

    noop_outputs = set(DEFAULT_NOOP_OUTPUTS) | set(args.extra_noop)

    model = onnx.load(str(input_path))

    print(f"Bypassing {len(noop_outputs)} no-op Reshape nodes:")
    model = bypass_reshapes(model, noop_outputs)

    # Save modified model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"\nSaved modified model to {output_path}")

    # Verify ORT output matches original
    print("\nVerifying ORT output unchanged...")
    orig_out = run_ort(input_path, args.seq_len)
    mod_out = run_ort(output_path, args.seq_len)

    diff = np.abs(orig_out - mod_out)
    print(f"  max_abs_diff original vs modified: {diff.max()}")
    print(f"  mean_abs_diff original vs modified: {diff.mean()}")

    if diff.max() < 1e-5:
        print("  PASS: ORT output identical after Reshape bypass")
    else:
        print("  FAIL: ORT output changed after Reshape bypass!")
        return

    # Compare against PyTorch reference
    ref_path = PREP_DIR / "reference" / "estimator_reference.npy"
    if ref_path.exists():
        ref = np.load(ref_path)
        mod_vs_ref = np.abs(mod_out - ref)
        print(f"  modified vs PyTorch reference max_abs_diff: {mod_vs_ref.max()}")
        print(f"  modified vs PyTorch reference mean_abs_diff: {mod_vs_ref.mean()}")


if __name__ == "__main__":
    main()
