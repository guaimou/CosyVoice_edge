#!/bin/bash
# Push all DLC files and updated scripts to SC171v3 board
# Run from project root: D:/ai_model/cosyvoice_snpe/
#
# Usage: bash scripts/deploy_to_board/push_dlcs.sh

BOARD_BASE="/home/fibo/AI model/tts_models/cosyvoice_snpe"
BOARD_DLC="$BOARD_BASE/dlc"
BOARD_SCRIPTS="$BOARD_BASE"

echo "=== Pushing new vocoder DLCs ==="
adb push output/hift_f0_predictor.dlc "$BOARD_DLC/hift_f0_predictor.dlc"
adb push output/hift_decode_pre_istft.dlc "$BOARD_DLC/hift_decode_pre_istft.dlc"
adb push output/campplus.dlc "$BOARD_DLC/campplus.dlc"

echo "=== Pushing updated board inference script ==="
adb push scripts/deploy_to_board/infer_tts_board.py "$BOARD_SCRIPTS/infer_tts_board.py"

echo "=== Pushing export scripts (reference) ==="
adb push scripts/probe_hift_export.py "$BOARD_SCRIPTS/probe_hift_export.py"
adb push scripts/export_hift_explicit_pad.py "$BOARD_SCRIPTS/export_hift_explicit_pad.py"

echo "=== Done ==="
echo ""
echo "DLC files on board:"
adb shell ls -lh "$BOARD_DLC/"
echo ""
echo "To run TTS with all DLCs:"
echo "  cd '$BOARD_BASE' && python3 infer_tts_board.py --text '你好' --out /tmp/test.wav"
