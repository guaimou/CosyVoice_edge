# cosyvoice_snpe

Windows-first local verification project for SC171v3 edge TTS. CosyVoice + HiFiGAN + SNPE.

## Current state (2026-05-15)

**SNPE flow decoder estimator validation complete. Deployed to SC171v3 board.** The SNPE DLC replaces the PyTorch estimator in the full CosyVoice TTS pipeline on both Windows+Docker and natively on SC171v3 (via fiboaisdk). End-to-end audio generation verified on both platforms.

### Quick start

```bat
setup.bat
```

```bat
rem PyTorch baseline
.venv\Scripts\python.exe scripts\run_infer.py --text "你好，这是一次本地验证。" --out output\smoke.wav

rem SNPE-integrated inference
.venv\Scripts\python.exe scripts\run_infer_snpe.py --text "你好，SNPE验证。" --out output\snpe_test.wav
```

### Validation results

| Check | Result |
|---|---|
| ONNX vs PyTorch reference | max_abs_diff=2.29e-05 (exact) |
| DLC conversion | Successful (SNPE 2.29.0.241129, seq_len=500) |
| SNPE intermediates vs ORT (early path) | Exact match (key_mean_abs_diff=0.0) |
| SNPE estimator_out vs ORT | mean_abs_diff=0.46 (NFC→NCF format) |
| SNPE TTS audio (short text, seq≈500) | Audible, mean_diff=0.025, corr=0.145 |
| SNPE TTS audio (long text, seq>500) | Degraded due to fixed seq_len truncation |

### Key findings

Three compounding issues were identified and resolved during SNPE validation:

1. **Data preparation bug:** The original layout sweep fed wrong-format data to ORT for transpose variants, making SNPE appear worse than it was. Fixed in `compare_flow_decoder_snpe_intermediates.py`.

2. **Debug dump format:** SNPE DLC internally uses NFC layout for UNet tensors. Loading debug dumps as NCF gave false divergence. The comparator now tests both formats.

3. **SNPE converter protobuf:** `protobuf 7.x` breaks Reshape shape inference in the C++ backend. Fix: `pip install 'protobuf>=3.19,<3.21'` in Docker.

The no-op Reshape bypass was verified correct but unnecessary — the original DLC already handles them via Transpose replacement.

### Repository layout

- `scripts/` — inference, validation, probing, export helpers
- `third_party/CosyVoice_edge/` — vendored CosyVoice source
- `pretrained/` — local model assets (kept out of git)
- `assets/` — prompt/reference audio
- `output/` — generated wav, DLC, debug artifacts

### Validation helpers

```bat
rem === Model inspection ===
.venv\Scripts\python.exe scripts\inspect_model_artifacts.py
.venv\Scripts\python.exe scripts\inspect_onnx_models.py

rem === Real case extraction ===
.venv\Scripts\python.exe scripts\dump_flow_decoder_inputs.py --out-dir output\flow_decoder_case

rem === ONNX comparison ===
.venv\Scripts\python.exe scripts\compare_flow_decoder_onnx.py --case-dir output\flow_decoder_case

rem === SNPE case preparation ===
.venv\Scripts\python.exe scripts\prepare_flow_decoder_snpe_case.py --case-dir output\flow_decoder_case --variant transpose_inputs

rem === SNPE intermediate comparison ===
.venv\Scripts\python.exe scripts\compare_flow_decoder_snpe_intermediates.py --prep-dir output\flow_decoder_snpe_prep --debug-dir output\flow_decoder_case_debug\Result_0

rem === SNPE-integrated TTS ===
.venv\Scripts\python.exe scripts\run_infer_snpe.py --text "你好" --out output\snpe_test.wav

rem === No-op Reshape detection and bypass ===
.venv\Scripts\python.exe scripts\detect_noop_reshapes.py
.venv\Scripts\python.exe scripts\bypass_noop_reshapes.py

rem === Layout sweep ===
.venv\Scripts\python.exe scripts\run_flow_decoder_layout_sweep.py
```

### Docker SNPE commands

```bash
# Convert ONNX to DLC (protobuf must be 3.20.x!)
export SNPE_ROOT=/opt/2.29.0.241129
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
  --input_network flow.decoder.estimator.fp32.onnx \
  --output_path flow.decoder.estimator.fp32.dlc \
  --define_symbol seq_len 500

# Run SNPE inference
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-net-run \
  --container flow.decoder.estimator.fp32.dlc \
  --input_list input_list.txt \
  --output_dir output \
  --runtime_order cpu --userbuffer_float --debug

# Inspect DLC
$SNPE_ROOT/bin/x86_64-linux-clang/snpe-dlc-info -i flow.decoder.estimator.fp32.dlc
```

### SC171v3 board deployment

Project: `/home/fibo/AI model/tts_models/cosyvoice_snpe/` (ADB `28de40d2`, root)

```bash
# Optimized TTS (INT8 DSP flow estimator + CPU rest)
adb shell "cd '/home/fibo/AI model/tts_models/cosyvoice_snpe' && python3 infer_tts_board_v3.py --text '你好' --out /tmp/test.wav"
adb pull /tmp/test.wav
```

Board results (v3, INT8 DSP): LLM 20s + DSP Init 95s + 10 ODE steps 180s = **298s total, 3.6s audio**.
Pure CPU (v1): 383s for 2.3s audio. DSP Exec per step: 0.426s (85x vs CPU), but Python marshalling adds ~17s/step.

### DLC inventory

| DLC | INT8 | Purpose |
|---|---|---|
| flow.decoder.estimator_int8.dlc | 84MB | Flow estimator (DSP, 0.426s/step) |
| hift_f0_predictor_int8.dlc | 3.3MB | F0 predictor (DSP, 0.040s) |
| hift_decode_pre_istft_int8.dlc | 17.7MB | Vocoder decode |
| campplus_int8.dlc | 8.7MB | Speaker embedding |

### Critical deployment notes
- **LLM must load FIRST, then DSP Init.** Reverse order causes OOM (5.5GB limit).
- fiboaisdk only supports one unique DSP DLC per process. Multi-process also OOMs.
- QCS6490 HTP supports INT8 only (not FP16). FP32 must use CPU.
- DSP Init ~95s is one-time; service mode would reuse the session.
- Python numpy↔list conversion costs ~17s/step — eliminating this would bring total to ~40s.

### Current limitations
- DLC `seq_len=500` limits text to ~10s. seq2000 DLC available but not tested on board.
- DSP Init overhead (95s) not cached across requests (single-run mode).
- Python data marshalling dominates inference time.
- Python 3.8.10 requires monkey-patches (wetext, whisper stub, disabled imports).

### Optimization roadmap
1. Eliminate numpy↔list marshalling → total ~40s (10x vs current 298s)
2. DSP session caching / service mode → save 95s Init
3. FP16 TorchScript LLM (772MB, pushed, awaiting integration)
4. INT8 quantization ✅ done (all 4 DLCs)
5. HTP/DSP backend ✅ verified (single-DSP mode)
6. campplus DLC integration into board pipeline

### Notes

- `pretrained/` and `output/` are git-ignored (large model and artifact files).
- SNPE conversion is done inside Docker container `my_work`.
- The local Windows inference path should always remain runnable alongside SNPE experiments.
