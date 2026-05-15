from pathlib import Path
import argparse
import json
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
PREP_ROOT = OUTPUT_DIR / "flow_decoder_4d_explicit"
RESULT_ROOT = OUTPUT_DIR / "flow_decoder_4d_explicit_result"
MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"
DEFAULT_CASE_DIR = OUTPUT_DIR / "flow_decoder_case"
DEFAULT_BASE_ESTIMATOR = MODEL_DIR / "flow.decoder.estimator.fp32.onnx"
DEFAULT_ESTIMATOR = OUTPUT_DIR / "flow.decoder.estimator.input_ncf_explicit_4d.onnx"
DEFAULT_DLC = OUTPUT_DIR / "flow.decoder.estimator.input_ncf_explicit_4d.dlc"
DEFAULT_SNPE_PROJECT = "/project/cosyvoice_snpe_snpe"
DEFAULT_SNPE_ROOT = "/opt/2.29.0.241129"
DEFAULT_CONTAINER = "my_work"


def run_python(script: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script), *args], check=True)


def convert_dlc(estimator_path: Path, dlc_path: Path, container: str, snpe_project: str, snpe_root: str) -> None:
    subprocess.run(f'docker cp "{estimator_path}" {container}:{snpe_project}/{estimator_path.name}', check=True, shell=True)
    command = (
        f'docker exec {container} bash -lc "'
        f'export SNPE_ROOT={snpe_root} && '
        f'export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH && '
        f'export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$SNPE_ROOT/lib/python:$PYTHONPATH && '
        f'\"$SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc\" '
        f'--input_network {snpe_project}/{estimator_path.name} '
        f'--output_path {snpe_project}/{dlc_path.name}'
        f'"'
    )
    subprocess.run(command, check=True, shell=True)


def run_docker(prep_dir: Path, result_dir: Path, dlc_path: Path, container: str, snpe_project: str, snpe_root: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    container_input = f"{snpe_project}/input_list.txt"
    container_output = f"{snpe_project}/flow_decoder_4d_explicit_result"
    raw_dir = f"{snpe_project}/raw"
    temp_input = prep_dir / "input_list.container.txt"
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
    parser.add_argument("--case-dir", default=str(DEFAULT_CASE_DIR))
    parser.add_argument("--base-estimator", default=str(DEFAULT_BASE_ESTIMATOR))
    parser.add_argument("--estimator", default=str(DEFAULT_ESTIMATOR))
    parser.add_argument("--dlc", default=str(DEFAULT_DLC))
    parser.add_argument("--prep-dir", default=str(PREP_ROOT))
    parser.add_argument("--result-dir", default=str(RESULT_ROOT))
    parser.add_argument("--seq-len", type=int, default=500)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--snpe-project", default=DEFAULT_SNPE_PROJECT)
    parser.add_argument("--snpe-root", default=DEFAULT_SNPE_ROOT)
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    base_estimator = Path(args.base_estimator)
    estimator_path = Path(args.estimator)
    dlc_path = Path(args.dlc)
    prep_dir = Path(args.prep_dir)
    result_dir = Path(args.result_dir)
    for name, path in {
        "case_dir": case_dir,
        "base_estimator": base_estimator,
        "estimator": estimator_path,
        "dlc": dlc_path,
        "prep_dir": prep_dir,
        "result_dir": result_dir,
    }.items():
        if not path.is_absolute():
            resolved = (PROJECT_ROOT / path).resolve()
            if name == "case_dir":
                case_dir = resolved
            elif name == "base_estimator":
                base_estimator = resolved
            elif name == "estimator":
                estimator_path = resolved
            elif name == "dlc":
                dlc_path = resolved
            elif name == "prep_dir":
                prep_dir = resolved
            else:
                result_dir = resolved

    make_script = PROJECT_ROOT / "scripts" / "make_flow_decoder_input_ncf_explicit.py"
    prepare_script = PROJECT_ROOT / "scripts" / "prepare_flow_decoder_snpe_case.py"
    compare_script = PROJECT_ROOT / "scripts" / "compare_flow_decoder_snpe_intermediates.py"

    run_python(make_script, "--input", str(base_estimator), "--output", str(estimator_path), "--mode", "4d")
    convert_dlc(estimator_path, dlc_path, args.container, args.snpe_project, args.snpe_root)
    run_python(
        prepare_script,
        "--case-dir", str(case_dir),
        "--prep-dir", str(prep_dir),
        "--estimator", str(estimator_path),
        "--seq-len", str(args.seq_len),
        "--variant", "explicit_4d",
    )
    run_docker(prep_dir, result_dir, dlc_path, args.container, args.snpe_project, args.snpe_root)
    report_path = result_dir / "snpe_vs_ort_intermediates.json"
    run_python(
        compare_script,
        "--prep-dir", str(prep_dir),
        "--debug-dir", str(result_dir / "Result_0"),
        "--estimator", str(estimator_path),
        "--report-path", str(report_path),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"report={report_path}")
    print(f"key_mean_abs_diff={report['key_mean_abs_diff']}")
    print(f"estimator_out_mean_abs_diff={report['comparisons']['estimator_out']['mean_abs_diff']}")


if __name__ == "__main__":
    main()
