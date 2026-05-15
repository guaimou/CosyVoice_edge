from pathlib import Path
import argparse
import json
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
PREP_DIR = OUTPUT_DIR / "flow_decoder_coord_case"
RESULT_DIR = OUTPUT_DIR / "flow_decoder_coord_case_result"
MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"
DEFAULT_ESTIMATOR = MODEL_DIR / "flow.decoder.estimator.fp32.onnx"
DEFAULT_DLC = OUTPUT_DIR / "flow.decoder.estimator.fp32.dlc"
DEFAULT_SNPE_PROJECT = "/project/cosyvoice_snpe_snpe"
DEFAULT_SNPE_ROOT = "/opt/2.29.0.241129"
DEFAULT_CONTAINER = "my_work"
SEQ_LEN = 500


def write_raw(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value.astype(np.float32, copy=False).tofile(path)


def build_coord_tensor(channels: int, seq_len: int, batch: int = 2, scale: float = 1.0) -> np.ndarray:
    b = np.arange(batch, dtype=np.float32).reshape(batch, 1, 1) * 1000000.0
    c = np.arange(channels, dtype=np.float32).reshape(1, channels, 1) * 1000.0
    t = np.arange(seq_len, dtype=np.float32).reshape(1, 1, seq_len)
    return (b + c + t) * scale


def prepare_case(prep_dir: Path) -> None:
    raw_dir = prep_dir / "raw"
    x = build_coord_tensor(80, SEQ_LEN, scale=1.0)
    mu = build_coord_tensor(80, SEQ_LEN, scale=0.1)
    cond = build_coord_tensor(80, SEQ_LEN, scale=0.01)
    mask = np.ones((2, 1, SEQ_LEN), dtype=np.float32)
    t = np.zeros((2,), dtype=np.float32)
    spks = np.zeros((2, 80), dtype=np.float32)
    for name, value in {
        "x": x,
        "mask": mask,
        "mu": mu,
        "t": t,
        "spks": spks,
        "cond": cond,
    }.items():
        write_raw(raw_dir / f"{name}.raw", value)

    (prep_dir / "input_list.txt").write_text(
        " ".join(
            [
                f"x:={raw_dir.as_posix()}/x.raw",
                f"mask:={raw_dir.as_posix()}/mask.raw",
                f"mu:={raw_dir.as_posix()}/mu.raw",
                f"t:={raw_dir.as_posix()}/t.raw",
                f"spks:={raw_dir.as_posix()}/spks.raw",
                f"cond:={raw_dir.as_posix()}/cond.raw",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "target_seq_len": SEQ_LEN,
        "mode": "coordinate_probe",
        "raw_inputs": {name: str(raw_dir / f"{name}.raw") for name in ["x", "mask", "mu", "t", "spks", "cond"]},
    }
    (prep_dir / "prep_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run_python(script: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script), *args], check=True)


def run_docker(prep_dir: Path, result_dir: Path, dlc_path: Path, container: str, snpe_project: str, snpe_root: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    container_input = f"{snpe_project}/input_list.txt"
    container_output = f"{snpe_project}/flow_decoder_coord_case_result"
    temp_input = prep_dir / "input_list.container.txt"
    raw_dir = f"{snpe_project}/raw"
    temp_input.write_text(
        " ".join(
            [
                f"x:={raw_dir}/x.raw",
                f"mask:={raw_dir}/mask.raw",
                f"mu:={raw_dir}/mu.raw",
                f"t:={raw_dir}/t.raw",
                f"spks:={raw_dir}/spks.raw",
                f"cond:={raw_dir}/cond.raw",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    subprocess.run(f'docker cp "{temp_input}" {container}:{container_input}', check=True, shell=True)
    subprocess.run(f'docker cp "{prep_dir / "raw"}" {container}:{snpe_project}/', check=True, shell=True)
    command = (
        f'docker exec {container} bash -lc "export SNPE_ROOT={snpe_root} && '
        f'export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH && '
        f'rm -rf {container_output} && mkdir -p {container_output} && '
        f'$SNPE_ROOT/bin/x86_64-linux-clang/snpe-net-run --container {snpe_project}/{dlc_path.name} '
        f'--input_list {container_input} --output_dir {container_output} --runtime_order cpu --userbuffer_float --debug"'
    )
    subprocess.run(command, check=True, shell=True)
    subprocess.run(f'docker cp {container}:{container_output}/Result_0 "{result_dir}"', check=True, shell=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-dir", default=str(PREP_DIR))
    parser.add_argument("--result-dir", default=str(RESULT_DIR))
    parser.add_argument("--estimator", default=str(DEFAULT_ESTIMATOR))
    parser.add_argument("--dlc", default=str(DEFAULT_DLC))
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--snpe-project", default=DEFAULT_SNPE_PROJECT)
    parser.add_argument("--snpe-root", default=DEFAULT_SNPE_ROOT)
    args = parser.parse_args()

    prep_dir = Path(args.prep_dir)
    result_dir = Path(args.result_dir)
    estimator_path = Path(args.estimator)
    dlc_path = Path(args.dlc)
    if not prep_dir.is_absolute():
        prep_dir = (PROJECT_ROOT / prep_dir).resolve()
    if not result_dir.is_absolute():
        result_dir = (PROJECT_ROOT / result_dir).resolve()
    if not estimator_path.is_absolute():
        estimator_path = (PROJECT_ROOT / estimator_path).resolve()
    if not dlc_path.is_absolute():
        dlc_path = (PROJECT_ROOT / dlc_path).resolve()

    prepare_case(prep_dir)
    run_docker(prep_dir, result_dir, dlc_path, args.container, args.snpe_project, args.snpe_root)
    compare_script = PROJECT_ROOT / "scripts" / "compare_flow_decoder_snpe_intermediates.py"
    report_path = result_dir / "coord_report.json"
    run_python(
        compare_script,
        "--prep-dir", str(prep_dir),
        "--debug-dir", str(result_dir / "Result_0"),
        "--estimator", str(estimator_path),
        "--report-path", str(report_path),
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
