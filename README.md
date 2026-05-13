# cosyvoice_snpe

Windows-first local verification project for SC171v3 edge TTS.

This repository is used to stand up and validate a self-contained local CosyVoice inference path before any ONNX, SNPE, or device packaging work begins.

## Current task

The current goal of this project is:

1. Keep a runnable CosyVoice inference workflow inside this repository.
2. Keep model assets, helper scripts, and generated outputs isolated locally.
3. Verify Windows local synthesis first, then move on to export and deployment work later.

## Repository layout

- `scripts/` — local inference helpers
- `third_party/CosyVoice_edge/` — vendored local CosyVoice source used by this project
- `pretrained/` — local model assets, kept out of git
- `assets/` — local prompt/reference assets
- `output/` — generated wav outputs, kept out of git

## Setup

From the project root:

```bat
setup.bat
```

This installs the Python dependencies into the project-local virtual environment when available.

## Run a smoke test

```bat
run.bat
```

This runs:

```bat
python scripts\run_infer.py --text "你好，这是一次本地验证。" --out output\smoke.wav
```

A successful run writes `output/smoke.wav`.

## Direct inference

You can also call the wrapper directly:

```bat
.venv\Scripts\python.exe scripts\run_infer.py --text "你好，这是一次本地验证。" --out output\smoke.wav
```

Optional arguments:

- `--prompt-text`
- `--prompt-wav`
- `--model-dir`
- `--speed`

## SNPE preparation helpers

Before starting export or DLC conversion, use these helpers from the project root:

```bat
.venv\Scripts\python.exe scripts\inspect_model_artifacts.py
.venv\Scripts\python.exe scripts\inspect_onnx_models.py
.venv\Scripts\python.exe scripts\probe_tts_pipeline.py
.venv\Scripts\python.exe scripts\probe_frontend_onnx_stages.py
```

What they do:

- `inspect_model_artifacts.py` classifies the current model files under `pretrained/CosyVoice-300M/` so you can see which artifacts are ONNX, PyTorch checkpoints, TorchScript packages, and auxiliary assets.
- `inspect_onnx_models.py` loads the existing ONNX files with `onnxruntime` and prints provider, input, and output metadata.
- `probe_tts_pipeline.py` runs the frontend side of the zero-shot path and prints the minimal stage map plus tensor shapes that connect the runtime pipeline.
- `probe_frontend_onnx_stages.py` probes the `campplus.onnx` and `speech_tokenizer_v1.onnx` stages directly and compares their outputs with the tensors injected into `frontend_zero_shot`.

Current high-value findings from this workspace:

- Existing ONNX artifacts:
  - `campplus.onnx`
  - `speech_tokenizer_v1.onnx`
  - `flow.decoder.estimator.fp32.onnx`
- Remaining PyTorch checkpoints:
  - `llm.pt`
  - `flow.pt`
  - `hift.pt`
- Existing TorchScript packages:
  - `llm.text_encoder.*.zip`
  - `llm.llm.*.zip`
  - `flow.encoder.*.zip`
- Current SNPE status:
  - `campplus.onnx` is used by the runtime frontend but failed direct SNPE conversion.
  - `speech_tokenizer_v1.onnx` is used by the runtime frontend but failed direct SNPE conversion because of dynamic-shape and operator issues.
  - `flow.decoder.estimator.fp32.onnx` successfully converted to DLC under SNPE 2.29.0.241129 and is the first confirmed SNPE-ready target in this workspace.

This means the most practical first SNPE target is `flow.decoder.estimator.fp32.onnx`, not the full end-to-end TTS pipeline.

## First successful SNPE conversion

The current reproducible SNPE success path is the flow decoder estimator ONNX.

Inside the local Docker environment, the successful conversion used:

```bash
export SNPE_ROOT=/opt/2.29.0.241129
export PYTHONPATH=/opt/2.29.0.241129/lib/python:${PYTHONPATH}
export LD_LIBRARY_PATH=/opt/2.29.0.241129/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
/opt/2.29.0.241129/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
  --input_network /project/cosyvoice_snpe_snpe/flow.decoder.estimator.fp32.onnx \
  --output_path /project/cosyvoice_snpe_snpe/flow.decoder.estimator.fp32.dlc \
  --define_symbol seq_len 500
```

The resulting DLC can be inspected with:

```bash
/opt/2.29.0.241129/bin/x86_64-linux-clang/snpe-dlc-info \
  -i /project/cosyvoice_snpe_snpe/flow.decoder.estimator.fp32.dlc
```

Important properties of this successful conversion:

- SNPE converter version: `2.29.0.241129`
- Staticized symbol: `seq_len=500`
- DLC graph name: `flow.decoder.estimator.fp32`
- Effective input shapes after conversion:
  - `x`: `[2,500,80]`
  - `mu`: `[2,500,80]`
  - `cond`: `[2,500,80]`
  - `mask`: `[2,500,1]`
  - `t`: `[2]`
  - `spks`: `[2,80]`

This is the current narrowest validated SNPE closure and should be treated as the baseline path before attempting broader CosyVoice export work.

## Notes

- The repository is intended to be self-contained for local Windows inference.
- `pretrained/` is ignored by git because model files are large and machine-local.
- SNPE conversion is intentionally deferred until the local inference path is stable.
- A local Docker environment already exposes SNPE 2.29.0.241129, so the next step can validate one of the existing ONNX artifacts inside the container before attempting broader conversion.
