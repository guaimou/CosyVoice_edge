# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Repository scope

This workspace is a Windows-first verification project for SC171v3 edge TTS. The working stack is `CosyVoice + HiFiGAN + SNPE`, with local Windows inference already standing up enough to support targeted SNPE validation.

All project-local work should stay inside `cosyvoice_snpe/`. Do not place new project files in `D:/ai_model/` root.

## Current phase

1. Keep the local CosyVoice inference path runnable on Windows.
2. Preserve project-local isolation for model assets, scripts, and generated outputs.
3. **SNPE flow decoder estimator validation COMPLETE (2026-05-15).** DLC replaces PyTorch estimator in full TTS pipeline → generates audible audio.
4. **Verified:** `transpose_inputs` variant + original DLC → all intermediates match ORT, estimator_out mean_abs_diff=0.46, E2E audio (short text) mean_abs_diff=0.025, correlation=0.145.
5. **No-op Reshape bypass:** Correct but unnecessary (original DLC handles them via Transpose replacement). Scripts kept for reference.
6. **Docker:** protobuf must be 3.20.x (not 7.x) for SNPE converter.
7. **Current limitation:** Fixed seq_len=500 limits text to ~10s. DLC conversion needs `--define_symbol seq_len 500`.
8. **Optimization opportunities:** Dynamic seq_len DLC, direct SNPE API (eliminate Docker overhead), HiFiGAN DLC, INT8 quantization, DSP/HTP backend.

## Project layout

- `scripts/` — helper scripts for bootstrap, inference, probing, and export validation
- `third_party/` — upstream CosyVoice source checkout or imported code
- `pretrained/` — local model assets
- `samples/` — test inputs and optional prompt/reference assets
- `assets/` — auxiliary local assets if needed
- `output/` — generated wav files and intermediate outputs

## Common commands

Run commands from `D:/ai_model/cosyvoice_snpe/` unless noted otherwise.

### Setup

- Create or reuse the local environment and install dependencies:
  - `setup.bat`

### Local inference

- Run the smoke-test entrypoint:
  - `run.bat`
- Run the helper script directly:
  - `python scripts/run_infer.py --text "你好，这是一次本地验证。" --out output/smoke.wav`

### Current validation helpers

- Prepare a real flow decoder case from the local zero-shot path:
  - `.venv\Scripts\python.exe scripts\dump_flow_decoder_inputs.py --out-dir output\flow_decoder_case`
- Compare ONNX outputs for the saved case:
  - `.venv\Scripts\python.exe scripts\compare_flow_decoder_onnx.py --case-dir output\flow_decoder_case`
- Prepare an SNPE-ready case bundle:
  - `.venv\Scripts\python.exe scripts\prepare_flow_decoder_snpe_case.py --case-dir output\flow_decoder_case`
- Compare SNPE debug outputs with ONNX Runtime intermediates:
  - `.venv\Scripts\python.exe scripts\compare_flow_decoder_snpe_intermediates.py --prep-dir output\flow_decoder_snpe_prep --debug-dir output\flow_decoder_case_debug\Result_0`
- Compare with coordinate probe case:
  - `.venv\Scripts\python.exe scripts\compare_flow_decoder_snpe_intermediates.py --prep-dir output\flow_decoder_coord_case --debug-dir output\flow_decoder_coord_case_result\Result_0 --report-path output\flow_decoder_coord_case_result\coord_report.json`
- Sweep SNPE input layout variants for the baseline estimator DLC:
  - `.venv\Scripts\python.exe scripts\run_flow_decoder_layout_sweep.py`
- Run coordinate-coded and explicit-input normalization probes when debugging SNPE layout issues:
  - `.venv\Scripts\python.exe scripts\run_flow_decoder_coordinate_probe.py`
  - `.venv\Scripts\python.exe scripts\run_flow_decoder_4d_explicit.py`
  - `.venv\Scripts\python.exe scripts\run_flow_decoder_4d_reshape_explicit.py`
- Detect all no-op Reshape nodes in the staticized estimator graph:
  - `.venv\Scripts\python.exe scripts\detect_noop_reshapes.py`
- Bypass all 10 data-path no-op Reshape nodes to avoid SNPE layout reinterpretation:
  - `.venv\Scripts\python.exe scripts\bypass_noop_reshapes.py`
  - The modified model is at `output/flow.decoder.estimator.noop_reshapes_bypassed.onnx`

## Architecture overview

This project should keep four layers separate:

1. Local wrapper layer
   - batch files, helper scripts, local config, outputs
2. Upstream CosyVoice layer
   - lives under `third_party/CosyVoice_edge/`
3. Vocoder layer
   - keep validation scoped so frontend, decoder, and vocoder boundaries stay visible
4. Export / deployment layer
   - ONNX / SNPE / DLC checks should stay narrow and reproducible instead of attempting end-to-end conversion first

## Repo-specific gotchas

- Prefer project-local paths over machine-global assumptions.
- Keep all downloaded or copied model assets under `pretrained/` unless there is a strong reason not to.
- `scripts/run_infer.py` should import CosyVoice from `third_party/CosyVoice_edge/` and use project-local assets under `pretrained/` and `assets/` only.
- Keep local Windows inference working while adding SNPE-side probes; do not break the runnable local path.
- Treat `flow.decoder.estimator.fp32.onnx` as the first confirmed SNPE-ready artifact in this workspace unless current evidence shows otherwise.
- If upstream CosyVoice expects Linux-centric commands or paths, wrap them in local scripts instead of editing broad project behavior immediately.
- Prefer reproducible tensor dumps and output comparisons over speculative converter changes when validating SNPE behavior.
- When inferring prior progress from scripts alone, distinguish between "script exists for this check" and "this check has been rerun in the current session".
- Existing artifacts under `output/flow_decoder_case/`, `output/flow_decoder_snpe_prep/`, `output/flow_decoder_case_debug/`, `output/flow_decoder_coord_case_result/`, and `output/flow_decoder_layout_sweep_results/` provide concrete evidence that real-case export, ONNX comparison, SNPE debug capture, coordinate probing, and layout sweeps were run previously in this workspace.
- SNPE's Reshape op permutes 3D tensor data due to internal layout reinterpretation. When a Reshape is a no-op under static seq_len (same input and output shape), bypass it in the ONNX model before SNPE conversion rather than trying to fix the layout with host-side transpose hacks.
- `detect_noop_reshapes.py` scanned 399 Reshape nodes in the graph and found 11 no-ops. 10 are data-path (safe to bypass); 1 is shape-computation (`/Reshape_2` on int64 tensors — do NOT bypass). `bypass_noop_reshapes.py` removes all 10 data-path no-op Reshapes and reconnects consumers. ORT confirms bit-identical output.
- Docker SNPE converter requires `protobuf>=3.19,<3.21`. If conversion fails with garbage values in Reshape shape inference, check: `pip install 'protobuf>=3.19,<3.21'`. The issue is that `protobuf 7.x` breaks ONNX constant initializer reading in the C++ backend.
- The `transpose_inputs` data variant is required for correct SNPE inference. Always use `--variant transpose_inputs` with `prepare_flow_decoder_snpe_case.py`. The comparator automatically detects the variant from prep_metadata.json and adjusts ORT loading accordingly.
