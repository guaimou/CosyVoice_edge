# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Repository scope

This workspace is a Windows-first verification project for SC171v3 edge TTS. The current target stack is `CosyVoice + HiFiGAN + SNPE`, but the immediate phase is **local Windows inference validation only**.

All project-local work should stay inside `cosyvoice_snpe/`. Do not place new project files in `D:/ai_model/` root.

## Current phase

1. Stand up a local CosyVoice inference path on Windows.
2. Keep model assets, scripts, and outputs isolated in this folder.
3. Delay SNPE export and device packaging until the local path is stable.

## Project layout

- `scripts/` — helper scripts for bootstrap, inference, and later export checks
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

## Architecture overview

This project should keep four layers separate:

1. Local wrapper layer
   - batch files, helper scripts, local config, outputs
2. Upstream CosyVoice layer
   - lives under `third_party/CosyVoice_edge/`
3. Vocoder layer
   - validate the usable local path first, then decide whether explicit HiFiGAN separation is needed
4. Export / deployment layer
   - ONNX / SNPE / DLC checks belong to a later phase

## Repo-specific gotchas

- Prefer project-local paths over machine-global assumptions.
- Keep all downloaded or copied model assets under `pretrained/` unless there is a strong reason not to.
- `scripts/run_infer.py` should import CosyVoice from `third_party/CosyVoice_edge/` and use project-local assets under `pretrained/` and `assets/` only.
- Do not begin SNPE conversion work until Windows inference is confirmed working.
- If upstream CosyVoice expects Linux-centric commands or paths, wrap them in local scripts instead of editing broad project behavior immediately.
