from pathlib import Path
import argparse
import json
import sys

import numpy as np
import onnxruntime
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
OUTPUT_DIR = PROJECT_ROOT / "output"
UPSTREAM_ROOT = THIRD_PARTY_DIR / "CosyVoice_edge"
DEFAULT_MODEL_DIR = PRETRAINED_DIR / "CosyVoice-300M"
DEFAULT_CASE_DIR = OUTPUT_DIR / "flow_decoder_case"
DEFAULT_PREP_DIR = OUTPUT_DIR / "flow_decoder_snpe_prep"
DEFAULT_ESTIMATOR = DEFAULT_MODEL_DIR / "flow.decoder.estimator.fp32.onnx"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from cosyvoice.cli.cosyvoice import CosyVoice

_ORIGINAL_INFERENCE_SESSION = onnxruntime.InferenceSession


def cpu_inference_session(path, sess_options=None, providers=None, provider_options=None, **kwargs):
    return _ORIGINAL_INFERENCE_SESSION(
        path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
        provider_options=provider_options,
        **kwargs,
    )


def load_array(case_dir: Path, name: str) -> np.ndarray:
    return np.load(case_dir / f"{name}.npy")


def duplicate_batch(value: np.ndarray) -> np.ndarray:
    return np.repeat(value.astype(np.float32, copy=False), 2, axis=0)


def write_raw(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value.astype(np.float32, copy=False).tofile(path)


def resize_for_dlc(value: np.ndarray, target_seq_len: int) -> np.ndarray:
    current = value.shape[-1]
    if current == target_seq_len:
        return value.astype(np.float32, copy=False)
    if current > target_seq_len:
        return value[..., :target_seq_len].astype(np.float32, copy=False)
    pad_shape = list(value.shape)
    pad_shape[-1] = target_seq_len - current
    pad = np.zeros(pad_shape, dtype=np.float32)
    return np.concatenate([value.astype(np.float32, copy=False), pad], axis=-1)


def serialize_for_snpe(name: str, value: np.ndarray, variant: str) -> np.ndarray:
    array = value.astype(np.float32, copy=False)
    if variant == "baseline":
        return array
    if variant == "transpose_inputs":
        if name in {"x", "mu", "cond"}:
            return np.transpose(array, (0, 2, 1)).copy()
        return array
    if variant == "transpose_inputs_and_mask":
        if name in {"x", "mu", "cond", "mask"}:
            return np.transpose(array, (0, 2, 1)).copy()
        return array
    if variant == "transpose_mask_only":
        if name == "mask":
            return np.transpose(array, (0, 2, 1)).copy()
        return array
    if variant == "compensate_userbuffer_map":
        if name in {"x", "mu", "cond"}:
            batch, channels, seq_len = array.shape
            return np.reshape(array, (batch, seq_len, channels)).copy()
        return array
    if variant == "explicit_4d":
        if name in {"x", "mu", "cond"}:
            return np.expand_dims(np.transpose(array, (0, 2, 1)).copy(), axis=1)
        if name == "mask":
            return np.expand_dims(np.transpose(array, (0, 2, 1)).copy(), axis=1)
        return array
    if variant == "explicit_4d_reshape":
        if name in {"x", "mu", "cond"}:
            return np.expand_dims(array.copy(), axis=1)
        return array
    raise ValueError(f"unknown variant: {variant}")


def prepare_inputs(case_dir: Path, prep_dir: Path, target_seq_len: int, variant: str = "baseline") -> tuple[dict, dict]:
    arrays = {name: load_array(case_dir, name) for name in ["x", "mask", "mu", "t", "spks", "cond"]}
    arrays["x"] = resize_for_dlc(arrays["x"], target_seq_len)
    arrays["mask"] = resize_for_dlc(arrays["mask"], target_seq_len)
    arrays["mu"] = resize_for_dlc(arrays["mu"], target_seq_len)
    arrays["cond"] = resize_for_dlc(arrays["cond"], target_seq_len)
    batched = {name: duplicate_batch(value) for name, value in arrays.items()}
    serialized = {name: serialize_for_snpe(name, value, variant) for name, value in batched.items()}
    raw_dir = prep_dir / "raw"
    for name, value in serialized.items():
        write_raw(raw_dir / f"{name}.raw", value)

    input_list_path = prep_dir / "input_list.txt"
    input_list_path.write_text(
        " ".join(
            [
                f"x:={ (raw_dir / 'x.raw').as_posix() }",
                f"mask:={ (raw_dir / 'mask.raw').as_posix() }",
                f"mu:={ (raw_dir / 'mu.raw').as_posix() }",
                f"t:={ (raw_dir / 't.raw').as_posix() }",
                f"spks:={ (raw_dir / 'spks.raw').as_posix() }",
                f"cond:={ (raw_dir / 'cond.raw').as_posix() }",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return arrays, batched, serialized


def run_reference(model_dir: Path, batched: dict) -> np.ndarray:
    onnxruntime.InferenceSession = cpu_inference_session
    try:
        model = CosyVoice(str(model_dir))
    finally:
        onnxruntime.InferenceSession = _ORIGINAL_INFERENCE_SESSION

    estimator = model.model.flow.decoder.estimator
    with torch.inference_mode():
        output = estimator(
            torch.from_numpy(batched["x"]).to(model.model.device),
            torch.from_numpy(batched["mask"]).to(model.model.device),
            torch.from_numpy(batched["mu"]).to(model.model.device),
            torch.from_numpy(batched["t"]).to(model.model.device),
            torch.from_numpy(batched["spks"]).to(model.model.device),
            torch.from_numpy(batched["cond"]).to(model.model.device),
            streaming=False,
        )
    return output.detach().cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", default=str(DEFAULT_CASE_DIR))
    parser.add_argument("--prep-dir", default=str(DEFAULT_PREP_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--estimator", default=str(DEFAULT_ESTIMATOR))
    parser.add_argument("--seq-len", type=int, default=500)
    parser.add_argument("--variant", default="baseline", choices=["baseline", "transpose_inputs", "transpose_inputs_and_mask", "transpose_mask_only", "compensate_userbuffer_map", "explicit_4d", "explicit_4d_reshape"])
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    prep_dir = Path(args.prep_dir)
    model_dir = Path(args.model_dir)
    estimator_path = Path(args.estimator)
    target_seq_len = args.seq_len
    variant = args.variant
    if not case_dir.is_absolute():
        case_dir = (PROJECT_ROOT / case_dir).resolve()
    if not prep_dir.is_absolute():
        prep_dir = (PROJECT_ROOT / prep_dir).resolve()
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()
    if not estimator_path.is_absolute():
        estimator_path = (PROJECT_ROOT / estimator_path).resolve()

    arrays, batched, serialized = prepare_inputs(case_dir, prep_dir, target_seq_len, variant)
    reference = run_reference(model_dir, batched)
    write_raw(prep_dir / "reference" / "estimator_reference.raw", reference)
    np.save(prep_dir / "reference" / "estimator_reference.npy", reference)

    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    prep_metadata = {
        "case_dir": str(case_dir),
        "prep_dir": str(prep_dir),
        "model_dir": str(model_dir),
        "estimator": str(estimator_path),
        "seq_len": int(reference.shape[2]),
        "target_seq_len": int(target_seq_len),
        "variant": variant,
        "reference_shape": list(reference.shape),
        "raw_inputs": {
            name: str((prep_dir / "raw" / f"{name}.raw"))
            for name in ["x", "mask", "mu", "t", "spks", "cond"]
        },
        "input_list": str(prep_dir / "input_list.txt"),
        "reference_raw": str(prep_dir / "reference" / "estimator_reference.raw"),
        "reference_npy": str(prep_dir / "reference" / "estimator_reference.npy"),
        "source_metadata": metadata,
        "source_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "batched_shapes": {name: list(value.shape) for name, value in batched.items()},
    }
    (prep_dir / "prep_metadata.json").write_text(json.dumps(prep_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"case_dir={case_dir}")
    print(f"prep_dir={prep_dir}")
    print(f"estimator={estimator_path}")
    for name in ["x", "mask", "mu", "t", "spks", "cond"]:
        print(f"{name}\tbatched_shape={list(batched[name].shape)}\tdtype={batched[name].dtype}")
    print(f"reference_shape={list(reference.shape)} dtype={reference.dtype}")
    print(f"input_list={prep_dir / 'input_list.txt'}")


if __name__ == "__main__":
    main()
