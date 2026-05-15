from pathlib import Path
import argparse
import json
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
PREP_ROOT = OUTPUT_DIR / "flow_decoder_layout_sweep"
RESULT_ROOT = OUTPUT_DIR / "flow_decoder_layout_sweep_results"
MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"
DEFAULT_CASE_DIR = OUTPUT_DIR / "flow_decoder_case"
DEFAULT_ESTIMATOR = MODEL_DIR / "flow.decoder.estimator.fp32.onnx"
DEFAULT_DLC = OUTPUT_DIR / "flow.decoder.estimator.fp32.dlc"
DEFAULT_SNPE_PROJECT = "/project/cosyvoice_snpe_snpe"
DEFAULT_SNPE_ROOT = "/opt/2.29.0.241129"
VARIANTS = [
    "baseline",
    "transpose_inputs",
    "transpose_inputs_and_mask",
    "transpose_mask_only",
    "compensate_userbuffer_map",
]


def run_python(script: Path, *args: str) -> None:
    command = [sys.executable, str(script), *args]
    subprocess.run(command, check=True)


def rewrite_input_list_for_container(input_list_path: Path, snpe_project: str) -> str:
    raw_dir = f"{snpe_project}/raw"
    mapping = [
        f"x:={raw_dir}/x.raw",
        f"mask:={raw_dir}/mask.raw",
        f"mu:={raw_dir}/mu.raw",
        f"t:={raw_dir}/t.raw",
        f"spks:={raw_dir}/spks.raw",
        f"cond:={raw_dir}/cond.raw",
    ]
    return " ".join(mapping) + "\n"


def run_docker_snpe(prep_dir: Path, output_dir_name: str, dlc_path: Path, snpe_project: str, snpe_root: str, container_name: str) -> None:
    host_output_dir = (RESULT_ROOT / output_dir_name).resolve()
    host_output_dir.mkdir(parents=True, exist_ok=True)
    container_input = f"{snpe_project}/input_list.txt"
    container_dlc = f"{snpe_project}/{dlc_path.name}"
    container_output = f"{snpe_project}/{output_dir_name}"
    temp_input_list = prep_dir / "input_list.container.txt"
    temp_input_list.write_text(rewrite_input_list_for_container(prep_dir / "input_list.txt", snpe_project), encoding="utf-8")
    subprocess.run(
        f'docker cp "{temp_input_list}" {container_name}:{container_input}',
        check=True,
        shell=True,
    )
    subprocess.run(
        f'docker cp "{prep_dir / "raw"}" {container_name}:{snpe_project}/',
        check=True,
        shell=True,
    )
    command = (
        f'docker exec {container_name} bash -lc "export SNPE_ROOT={snpe_root} && '
        f'export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH && '
        f'rm -rf {container_output} && mkdir -p {container_output} && '
        f'$SNPE_ROOT/bin/x86_64-linux-clang/snpe-net-run --container {container_dlc} '
        f'--input_list {container_input} --output_dir {container_output} --runtime_order cpu --userbuffer_float --debug"'
    )
    subprocess.run(command, check=True, shell=True)
    subprocess.run(
        f'docker cp {container_name}:{container_output}/Result_0 "{host_output_dir}"',
        check=True,
        shell=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", default=str(DEFAULT_CASE_DIR))
    parser.add_argument("--dlc", default=str(DEFAULT_DLC))
    parser.add_argument("--estimator", default=str(DEFAULT_ESTIMATOR))
    parser.add_argument("--seq-len", type=int, default=500)
    parser.add_argument("--snpe-project", default=DEFAULT_SNPE_PROJECT)
    parser.add_argument("--snpe-root", default=DEFAULT_SNPE_ROOT)
    parser.add_argument("--container", default="my_work")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    dlc_path = Path(args.dlc)
    estimator_path = Path(args.estimator)
    if not case_dir.is_absolute():
        case_dir = (PROJECT_ROOT / case_dir).resolve()
    if not dlc_path.is_absolute():
        dlc_path = (PROJECT_ROOT / dlc_path).resolve()
    if not estimator_path.is_absolute():
        estimator_path = (PROJECT_ROOT / estimator_path).resolve()

    summary = {"variants": []}
    prepare_script = PROJECT_ROOT / "scripts" / "prepare_flow_decoder_snpe_case.py"
    compare_script = PROJECT_ROOT / "scripts" / "compare_flow_decoder_snpe_intermediates.py"

    for variant in VARIANTS:
        prep_dir = PREP_ROOT / variant
        result_dir = RESULT_ROOT / variant
        run_python(
            prepare_script,
            "--case-dir", str(case_dir),
            "--prep-dir", str(prep_dir),
            "--estimator", str(estimator_path),
            "--seq-len", str(args.seq_len),
            "--variant", variant,
        )
        run_docker_snpe(prep_dir, variant, dlc_path, args.snpe_project, args.snpe_root, args.container)
        report_path = result_dir / "snpe_vs_ort_intermediates.json"
        run_python(
            compare_script,
            "--prep-dir", str(prep_dir),
            "--debug-dir", str(result_dir / "Result_0"),
            "--estimator", str(estimator_path),
            "--report-path", str(report_path),
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary["variants"].append(
            {
                "variant": variant,
                "report_path": str(report_path),
                "key_mean_abs_diff": report["key_mean_abs_diff"],
                "estimator_out_mean_abs_diff": report["comparisons"]["estimator_out"]["mean_abs_diff"],
            }
        )

    summary["variants"].sort(key=lambda item: item["key_mean_abs_diff"])
    summary_path = RESULT_ROOT / "layout_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")
    for item in summary["variants"]:
        print(
            f"{item['variant']} key_mean_abs_diff={item['key_mean_abs_diff']} "
            f"estimator_out_mean_abs_diff={item['estimator_out_mean_abs_diff']}"
        )


if __name__ == "__main__":
    main()
