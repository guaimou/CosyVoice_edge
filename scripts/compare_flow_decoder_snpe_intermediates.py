from pathlib import Path
import argparse
import json

import numpy as np
import onnx
import onnxruntime
from onnx import helper, shape_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
PREP_DIR = OUTPUT_DIR / "flow_decoder_snpe_prep"
DEBUG_DIR = OUTPUT_DIR / "flow_decoder_case_debug" / "Result_0"
MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"
DEFAULT_ESTIMATOR = MODEL_DIR / "flow.decoder.estimator.fp32.onnx"
TEMP_MODEL = OUTPUT_DIR / "flow_decoder_case_debug" / "estimator_with_intermediates.onnx"


SEMANTIC_GROUPS = {
    "/Cast_output_0": "time_emb_branch",
    "/Unsqueeze_4_output_0": "spks_branch",
    "/Reshape_output_0": "x_branch",
    "/Reshape_1_output_0": "mu_branch",
    "/Concat_2_output_0": "x_mu_merge",
    "/Expand_output_0": "spks_branch",
    "/Reshape_3_output_0": "x_mu_packed",
    "/Reshape_4_output_0": "spks_packed",
    "/Concat_6_output_0": "x_mu_spks_merge",
    "/Reshape_5_output_0": "before_cond_merge",
    "/Reshape_6_output_0": "cond_branch",
    "/Concat_9_output_0": "all_packed",
    "/Reshape_7_output_0": "deep_noop_reshape",
    "/Reshape_8_output_0": "deep_noop_reshape",
    "/Concat_12_output_0": "skip_merge_up0",
    "/Reshape_9_output_0": "deep_noop_reshape",
    "/Reshape_10_output_0": "deep_noop_reshape",
    "/Concat_15_output_0": "skip_merge_up1",
    "estimator_out": "final_output",
}


def load_raw(path: Path, shape: tuple[int, ...], order: str = "C") -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape, order=order)


def compare(actual: np.ndarray, reference: np.ndarray) -> dict:
    diff = actual - reference
    abs_diff = np.abs(diff)
    return {
        "shape": list(actual.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "snpe_mean": float(actual.mean()),
        "ort_mean": float(reference.mean()),
        "snpe_std": float(actual.std()),
        "ort_std": float(reference.std()),
        "abs_diff_sum": float(abs_diff.sum()),
        "reference_abs_sum": float(np.abs(reference).sum()),
        "actual_abs_sum": float(np.abs(actual).sum()),
    }


def compare_variants(actual: np.ndarray, reference: np.ndarray) -> dict:
    variants = {
        "identity": actual,
        "transpose_last_two": np.swapaxes(actual, -1, -2),
        "reshape_c_order": actual.reshape(reference.shape),
        "from_flat_f_order": np.reshape(actual.ravel(order="F"), reference.shape, order="C"),
        "from_flat_reverse": actual.ravel()[::-1].reshape(reference.shape),
    }
    results = {}
    for name, value in variants.items():
        if value.shape != reference.shape:
            continue
        results[name] = compare(value.astype(np.float32, copy=False), reference)
    best_name = min(results, key=lambda key: results[key]["mean_abs_diff"])
    return {
        "best_variant": best_name,
        "variants": results,
    }


def compare_layout_variants(debug_dir: Path, tensor_name: str, expected_shape: tuple[int, ...], reference: np.ndarray) -> dict:
    """Load raw bytes with different axis orders to detect transposed SNPE buffer layouts."""
    results = {}
    if len(expected_shape) != 3:
        return results
    base = tensor_name.split("/")[-1]
    for view_label, suffix in [("raw", ".raw"), ("nfc", ".nfc.raw")]:
        path = debug_dir / f"{base}{suffix}"
        if not path.is_file():
            continue
        raw_bytes = path.read_bytes()
        n_floats = len(raw_bytes) // 4
        if n_floats != int(np.prod(expected_shape)):
            continue
        b, c, t = expected_shape
        shape_attempts = {
            "ncf": (b, c, t),
            "nfc": (b, t, c),
        }
        for shape_name, shape in shape_attempts.items():
            arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)
            if arr.shape == reference.shape:
                results[f"{view_label}_as_{shape_name}"] = compare(arr, reference)
            else:
                # Transpose to match reference shape for comparison
                if shape == (b, t, c) and reference.shape == (b, c, t):
                    arr_t = np.transpose(arr, (0, 2, 1)).copy()
                    results[f"{view_label}_as_{shape_name}_T"] = compare(arr_t, reference)
    return results


def classify_mismatch(metrics: dict, best_variant: str) -> str:
    if metrics["mean_abs_diff"] == 0.0:
        return "exact_match"
    same_mean = abs(metrics["snpe_mean"] - metrics["ort_mean"]) < 1e-6
    same_std = abs(metrics["snpe_std"] - metrics["ort_std"]) < 1e-6
    same_abs_sum = abs(metrics["actual_abs_sum"] - metrics["reference_abs_sum"]) < 1e-3
    if same_mean and same_std and same_abs_sum:
        if best_variant == "transpose_last_two":
            return "axis_swap_candidate"
        if best_variant in {"from_flat_f_order", "reshape_c_order"}:
            return "stride_or_layout_reinterpretation_candidate"
        return "value_preserving_permutation_candidate"
    return "numeric_or_branch_content_mismatch"


def add_intermediate_outputs(source_model: Path, target_model: Path, output_names: list[str]) -> None:
    model = shape_inference.infer_shapes(onnx.load(str(source_model)))
    existing = {value.name for value in model.graph.output}
    value_info = {value.name: value for value in model.graph.value_info}
    value_info.update({value.name: value for value in model.graph.input})
    value_info.update({value.name: value for value in model.graph.output})

    for name in output_names:
        if name in existing:
            continue
        if name not in value_info:
            raise KeyError(f"missing value info for {name}")
        model.graph.output.append(helper.make_empty_tensor_value_info(name))
        model.graph.output[-1].CopyFrom(value_info[name])

    onnx.save(model, str(target_model))


def get_ort_fetches() -> list[str]:
    return [
        # Early branch inputs
        "/Cast_output_0",
        "/Unsqueeze_4_output_0",
        "/Reshape_output_0",
        "/Reshape_1_output_0",
        "/Concat_2_output_0",
        "/Expand_output_0",
        # Early merges
        "/Reshape_3_output_0",
        "/Reshape_4_output_0",
        "/Concat_6_output_0",
        "/Reshape_5_output_0",
        "/Reshape_6_output_0",
        "/Concat_9_output_0",
        # UNet skip connections (mid/up block boundaries)
        "/Reshape_7_output_0",
        "/Reshape_8_output_0",
        "/Concat_12_output_0",
        "/Reshape_9_output_0",
        "/Reshape_10_output_0",
        "/Concat_15_output_0",
        # Final output
        "estimator_out",
    ]


def load_ort_map(estimator_path: Path, raw_dir: Path, seq_len: int, variant: str = "baseline") -> dict[str, np.ndarray]:
    def load_transposed(name: str, ncf_shape: tuple[int, ...]) -> np.ndarray:
        """Load raw data that was serialized with transpose (0,2,1) and undo it for ORT."""
        b, c, t = ncf_shape
        return np.transpose(load_raw(raw_dir / f"{name}.raw", (b, t, c)), (0, 2, 1)).copy()

    if variant in ("transpose_inputs", "transpose_inputs_and_mask"):
        nfc_inputs = {
            "x": load_transposed("x", (2, 80, seq_len)),
            "mu": load_transposed("mu", (2, 80, seq_len)),
            "cond": load_transposed("cond", (2, 80, seq_len)),
            "t": load_raw(raw_dir / "t.raw", (2,)),
            "spks": load_raw(raw_dir / "spks.raw", (2, 80)),
        }
        if variant == "transpose_inputs_and_mask":
            nfc_inputs["mask"] = load_transposed("mask", (2, 1, seq_len))
        else:
            nfc_inputs["mask"] = load_raw(raw_dir / "mask.raw", (2, 1, seq_len))
    elif variant == "transpose_mask_only":
        nfc_inputs = {
            "x": load_raw(raw_dir / "x.raw", (2, 80, seq_len)),
            "mask": load_transposed("mask", (2, 1, seq_len)),
            "mu": load_raw(raw_dir / "mu.raw", (2, 80, seq_len)),
            "t": load_raw(raw_dir / "t.raw", (2,)),
            "spks": load_raw(raw_dir / "spks.raw", (2, 80)),
            "cond": load_raw(raw_dir / "cond.raw", (2, 80, seq_len)),
        }
    else:
        nfc_inputs = {
            "x": load_raw(raw_dir / "x.raw", (2, 80, seq_len)),
            "mask": load_raw(raw_dir / "mask.raw", (2, 1, seq_len)),
            "mu": load_raw(raw_dir / "mu.raw", (2, 80, seq_len)),
            "t": load_raw(raw_dir / "t.raw", (2,)),
            "spks": load_raw(raw_dir / "spks.raw", (2, 80)),
            "cond": load_raw(raw_dir / "cond.raw", (2, 80, seq_len)),
        }
    ort_fetches = get_ort_fetches()
    add_intermediate_outputs(estimator_path, TEMP_MODEL, ort_fetches)
    session = onnxruntime.InferenceSession(str(TEMP_MODEL), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(ort_fetches, nfc_inputs)
    return dict(zip(ort_fetches, ort_outputs))


def tensor_shapes(seq_len: int) -> dict[str, tuple[int, ...]]:
    return {
        "/Cast_output_0": (2, 320),
        "/Unsqueeze_4_output_0": (2, 80, 1),
        "/Reshape_output_0": (2, 80, seq_len),
        "/Reshape_1_output_0": (2, 80, seq_len),
        "/Concat_2_output_0": (2, 160, seq_len),
        "/Expand_output_0": (2, 80, seq_len),
        "/Reshape_3_output_0": (2, 160, seq_len),
        "/Reshape_4_output_0": (2, 80, seq_len),
        "/Concat_6_output_0": (2, 240, seq_len),
        "/Reshape_5_output_0": (2, 240, seq_len),
        "/Reshape_6_output_0": (2, 80, seq_len),
        "/Concat_9_output_0": (2, 320, seq_len),
        "/Reshape_7_output_0": (2, 256, 250),
        "/Reshape_8_output_0": (2, 256, 250),
        "/Concat_12_output_0": (2, 512, 250),
        "/Reshape_9_output_0": (2, 256, 500),
        "/Reshape_10_output_0": (2, 256, 500),
        "/Concat_15_output_0": (2, 512, 500),
        "estimator_out": (2, 80, seq_len),
    }


def candidate_debug_paths(debug_dir: Path, tensor_name: str) -> list[tuple[str, Path]]:
    base = tensor_name.split("/")[-1]
    return [
        ("raw", debug_dir / f"{base}.raw"),
        ("nfc", debug_dir / f"{base}.nfc.raw"),
    ]


def load_best_snpe_view(debug_dir: Path, tensor_name: str, shape: tuple[int, ...], reference: np.ndarray) -> tuple[dict, dict] | tuple[None, None]:
    views = {}
    best_name = None
    best_metrics = None
    best_variant = None
    best_variant_metrics = None
    best_array = None

    for view_name, path in candidate_debug_paths(debug_dir, tensor_name):
        if not path.is_file():
            continue
        array = load_raw(path, shape)
        metrics = compare(array, reference)
        variant_report = compare_variants(array, reference)
        variant_name = variant_report["best_variant"]
        variant_metrics = variant_report["variants"][variant_name]
        mismatch_type = classify_mismatch(metrics, variant_name)
        views[view_name] = {
            "path": str(path),
            "comparison": metrics,
            "best_variant": variant_name,
            "best_variant_metrics": variant_metrics,
            "mismatch_type": mismatch_type,
        }
        if best_variant_metrics is None or variant_metrics["mean_abs_diff"] < best_variant_metrics["mean_abs_diff"]:
            best_name = view_name
            best_metrics = metrics
            best_variant = variant_name
            best_variant_metrics = variant_metrics
            best_array = array

    if best_name is None:
        return None, None

    layout_variants = compare_layout_variants(debug_dir, tensor_name, shape, reference)

    summary = {
        "selected_view": best_name,
        "available_views": list(views.keys()),
        "comparison": best_metrics,
        "best_variant": best_variant,
        "best_variant_metrics": best_variant_metrics,
        "mismatch_type": classify_mismatch(best_metrics, best_variant),
    }
    return {"array": best_array, "views": views, "layout_variants": layout_variants}, summary


def summarize_earliest_divergence(comparisons: dict) -> dict:
    for index, name in enumerate(get_ort_fetches()):
        metrics = comparisons.get(name)
        if metrics is None:
            continue
        if metrics.get("status") == "snpe_debug_missing":
            continue
        if metrics.get("mean_abs_diff", 0.0) > 0.0:
            return {
                "tensor": name,
                "order_index": index,
                "semantic_group": SEMANTIC_GROUPS.get(name, "unknown"),
                "mean_abs_diff": metrics["mean_abs_diff"],
                "selected_view": metrics.get("selected_view"),
                "mismatch_type": metrics.get("mismatch_type"),
            }
    return {
        "tensor": None,
        "order_index": -1,
        "semantic_group": "none",
        "mean_abs_diff": 0.0,
        "selected_view": None,
        "mismatch_type": "exact_match",
    }


def build_report(debug_dir: Path, ort_map: dict[str, np.ndarray], seq_len: int) -> dict:
    shapes = tensor_shapes(seq_len)
    report = {
        "debug_dir": str(debug_dir),
        "seq_len": seq_len,
        "comparisons": {},
        "view_comparisons": {},
        "variant_checks": {},
        "semantic_groups": SEMANTIC_GROUPS,
    }

    for name, reference in ort_map.items():
        reference = reference.astype(np.float32, copy=False)
        view_payload, summary = load_best_snpe_view(debug_dir, name, shapes[name], reference)
        if view_payload is None:
            report["comparisons"][name] = {
                "shape": list(reference.shape),
                "ort_mean": float(reference.mean()),
                "ort_std": float(reference.std()),
                "reference_abs_sum": float(np.abs(reference).sum()),
                "status": "snpe_debug_missing",
                "semantic_group": SEMANTIC_GROUPS.get(name, "unknown"),
            }
            report["view_comparisons"][name] = {}
            report["variant_checks"][name] = {}
            continue
        report["comparisons"][name] = {
            "shape": summary["comparison"]["shape"],
            "max_abs_diff": summary["comparison"]["max_abs_diff"],
            "mean_abs_diff": summary["comparison"]["mean_abs_diff"],
            "rmse": summary["comparison"]["rmse"],
            "snpe_mean": summary["comparison"]["snpe_mean"],
            "ort_mean": summary["comparison"]["ort_mean"],
            "snpe_std": summary["comparison"]["snpe_std"],
            "ort_std": summary["comparison"]["ort_std"],
            "abs_diff_sum": summary["comparison"]["abs_diff_sum"],
            "reference_abs_sum": summary["comparison"]["reference_abs_sum"],
            "actual_abs_sum": summary["comparison"]["actual_abs_sum"],
        }
        report["view_comparisons"][name] = view_payload["views"]
        report["variant_checks"][name] = {
            "best_variant": summary["best_variant"],
            "variants": {
                view_name: payload["best_variant_metrics"]
                for view_name, payload in view_payload["views"].items()
            },
        }
        report["comparisons"][name]["selected_view"] = summary["selected_view"]
        report["comparisons"][name]["semantic_group"] = SEMANTIC_GROUPS.get(name, "unknown")
        report["comparisons"][name]["mismatch_type"] = summary["mismatch_type"]
        report["comparisons"][name]["best_variant"] = summary["best_variant"]
        report["comparisons"][name]["best_variant_mean_abs_diff"] = summary["best_variant_metrics"]["mean_abs_diff"]
        if view_payload.get("layout_variants"):
            best_lv = min(view_payload["layout_variants"].items(), key=lambda kv: kv[1]["mean_abs_diff"]) if view_payload["layout_variants"] else (None, None)
            report["comparisons"][name]["layout_variants"] = view_payload["layout_variants"]
            if best_lv[0] is not None:
                report["comparisons"][name]["best_layout_variant"] = best_lv[0]
                report["comparisons"][name]["best_layout_mean_abs_diff"] = best_lv[1]["mean_abs_diff"]

    key_names = ["/Reshape_output_0", "/Reshape_1_output_0", "/Reshape_6_output_0",
                 "/Reshape_7_output_0", "/Reshape_8_output_0", "/Reshape_9_output_0", "/Reshape_10_output_0"]
    key_values = [report["comparisons"][name].get("mean_abs_diff") for name in key_names if name in report["comparisons"] and "mean_abs_diff" in report["comparisons"][name]]
    report["key_mean_abs_diff"] = float(np.mean(key_values)) if key_values else None
    report["earliest_divergence"] = summarize_earliest_divergence(report["comparisons"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-dir", default=str(PREP_DIR))
    parser.add_argument("--debug-dir", default=str(DEBUG_DIR))
    parser.add_argument("--estimator", default=str(DEFAULT_ESTIMATOR))
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()

    prep_dir = Path(args.prep_dir)
    debug_dir = Path(args.debug_dir)
    estimator_path = Path(args.estimator)
    report_path = Path(args.report_path) if args.report_path else debug_dir.parent / "snpe_vs_ort_intermediates.json"
    if not prep_dir.is_absolute():
        prep_dir = (PROJECT_ROOT / prep_dir).resolve()
    if not debug_dir.is_absolute():
        debug_dir = (PROJECT_ROOT / debug_dir).resolve()
    if not estimator_path.is_absolute():
        estimator_path = (PROJECT_ROOT / estimator_path).resolve()
    if not report_path.is_absolute():
        report_path = (PROJECT_ROOT / report_path).resolve()

    metadata = json.loads((prep_dir / "prep_metadata.json").read_text(encoding="utf-8"))
    seq_len = metadata["target_seq_len"]
    variant = metadata.get("variant", "baseline")
    raw_dir = prep_dir / "raw"
    ort_map = load_ort_map(estimator_path, raw_dir, seq_len, variant)
    report = build_report(debug_dir, ort_map, seq_len)
    report["estimator"] = str(estimator_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in get_ort_fetches():
        metrics = report["comparisons"].get(name)
        if metrics is None:
            print(f"{name}  MISSING from report")
            continue
        print(name)
        print(f"  semantic_group={metrics.get('semantic_group', 'unknown')}")
        if metrics.get("status") == "snpe_debug_missing":
            print(f"  status=snpe_debug_missing (ORT shape={metrics['shape']}, ort_mean={metrics.get('ort_mean')})")
            continue
        print(f"  selected_view={metrics['selected_view']}")
        print(f"  shape={metrics['shape']}")
        print(f"  max_abs_diff={metrics['max_abs_diff']}")
        print(f"  mean_abs_diff={metrics['mean_abs_diff']}")
        print(f"  rmse={metrics['rmse']}")
        print(f"  mismatch_type={metrics['mismatch_type']}")
        print(f"  best_variant={metrics['best_variant']}")
        print(f"  best_variant_mean_abs_diff={metrics['best_variant_mean_abs_diff']}")
        if metrics.get("best_layout_variant") and metrics.get("best_layout_mean_abs_diff", 1.0) < metrics.get("best_variant_mean_abs_diff", 0.0):
            print(f"  best_layout_variant={metrics['best_layout_variant']}")
            print(f"  best_layout_mean_abs_diff={metrics['best_layout_mean_abs_diff']}")
    print(f"key_mean_abs_diff={report['key_mean_abs_diff']}")
    print(f"earliest_divergence={json.dumps(report['earliest_divergence'], ensure_ascii=False)}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
