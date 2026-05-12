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

## Notes

- The repository is intended to be self-contained for local Windows inference.
- `pretrained/` is ignored by git because model files are large and machine-local.
- SNPE conversion is intentionally deferred until the local inference path is stable.
