from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"

CLASSIFIERS = [
    ("onnx", ".onnx"),
    ("torch_checkpoint", ".pt"),
    ("torchscript_zip", ".zip"),
    ("audio_asset", ".wav"),
    ("image_asset", ".png"),
    ("yaml_config", ".yaml"),
    ("json_config", ".json"),
    ("cache_lock", ".lock"),
    ("cache_metadata", ".metadata"),
    ("cache_incomplete", ".incomplete"),
]


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    for kind, ext in CLASSIFIERS:
        if suffix == ext:
            return kind
    if path.name.startswith("."):
        return "hidden"
    return "other"


def is_deployment_relevant(path: Path) -> bool:
    if path.is_dir():
        return False
    parts = {part.lower() for part in path.parts}
    return ".cache" not in parts and "__pycache__" not in parts


def collect(model_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(model_dir.rglob("*")):
        if not is_deployment_relevant(path):
            continue
        rows.append(
            {
                "relative_path": path.relative_to(model_dir).as_posix(),
                "type": classify(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def print_text(rows: list[dict]) -> None:
    print("artifact_inventory_start")
    for row in rows:
        print(f"{row['type']}\t{row['size_bytes']}\t{row['relative_path']}")
    print("artifact_inventory_end")


def print_summary(rows: list[dict]) -> None:
    summary = {}
    for row in rows:
        bucket = summary.setdefault(row["type"], {"count": 0, "size_bytes": 0})
        bucket["count"] += 1
        bucket["size_bytes"] += row["size_bytes"]
    print("artifact_summary_start")
    for artifact_type in sorted(summary):
        bucket = summary[artifact_type]
        print(f"{artifact_type}\tcount={bucket['count']}\tsize_bytes={bucket['size_bytes']}")
    print("artifact_summary_end")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    rows = collect(model_dir)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print(f"model_dir={model_dir}")
    print_text(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
