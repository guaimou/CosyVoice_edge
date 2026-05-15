# SC171v3 Board Deployment Manifest (2026-05-16)

## DLC Inventory

| DLC | Size | Input Format | Verified |
|---|---|---|---|
| campplus.dlc | 28MB | input [1,80,T] NCF | cosine=0.98 |
| flow.decoder.estimator.fp32.dlc | 315MB | x/mu/cond [2,500,80] NFC, mask [2,1,500], t [2], spks [2,80] | ✅ |
| hift_f0_predictor.dlc | 13MB | speech_feat [1,80,T] NCF | max=0.126 |
| hift_decode_pre_istft.dlc | 67MB | speech_feat [1,80,T], s_stft [1,18,F] NCF | mag=0.0003 |

## Push Commands (when ADB connected)

```bash
# From project root: D:/ai_model/cosyvoice_snpe/

# Push new DLCs
adb push output/campplus.dlc "/home/fibo/AI model/tts_models/cosyvoice_snpe/dlc/campplus.dlc"
adb push output/hift_f0_predictor.dlc "/home/fibo/AI model/tts_models/cosyvoice_snpe/dlc/hift_f0_predictor.dlc"
adb push output/hift_decode_pre_istft.dlc "/home/fibo/AI model/tts_models/cosyvoice_snpe/dlc/hift_decode_pre_istft.dlc"

# Push updated scripts
adb push scripts/deploy_to_board/infer_tts_board.py "/home/fibo/AI model/tts_models/cosyvoice_snpe/infer_tts_board.py"
adb push scripts/deploy_to_board/test_dlcs_board.py "/home/fibo/AI model/tts_models/cosyvoice_snpe/test_dlcs_board.py"

# Push export scripts (reference)
adb push scripts/probe_hift_export.py "/home/fibo/AI model/tts_models/cosyvoice_snpe/probe_hift_export.py"
adb push scripts/export_hift_explicit_pad.py "/home/fibo/AI model/tts_models/cosyvoice_snpe/export_hift_explicit_pad.py"
```

## Board Validation

```bash
# 1. Test all DLCs individually
cd "/home/fibo/AI model/tts_models/cosyvoice_snpe"
python3 test_dlcs_board.py

# 2. Run full TTS (flow estimator DLC only, rest PyTorch)
python3 infer_tts_board.py --text "你好" --out /tmp/test.wav
```

## Format Reference

### NCF format (used by campplus, f0_predictor, decode)
- Shape: [batch, channels, time]
- Write raw directly as numpy C-order

### NFC format (used by flow estimator)
- Shape: [batch, time, channels]
- Transpose from NCF before writing raw

### campplus specific
- CosyVoice code: `kaldi.fbank` → [T,80] → unsqueeze(0) → [1,T,80]
- SNPE DLC expects: transpose to [1,80,T], pad to [1,80,500]
