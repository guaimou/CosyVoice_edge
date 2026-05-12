from pathlib import Path
import argparse
import sys
import time

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
OUTPUT_DIR = PROJECT_ROOT / "output"
UPSTREAM_ROOT = THIRD_PARTY_DIR / "CosyVoice_edge"
DEFAULT_MODEL_DIR = PRETRAINED_DIR / "CosyVoice-300M"
DEFAULT_PROMPT_WAV = PROJECT_ROOT / "assets" / "zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from cosyvoice.cli.cosyvoice import CosyVoice


def ensure_local_assets() -> None:
    if not (UPSTREAM_ROOT / "cosyvoice").is_dir():
        raise FileNotFoundError(f"upstream CosyVoice code not found: {UPSTREAM_ROOT}")
    if not DEFAULT_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"model directory not found: {DEFAULT_MODEL_DIR}")
    if not DEFAULT_PROMPT_WAV.is_file():
        raise FileNotFoundError(f"prompt wav not found: {DEFAULT_PROMPT_WAV}")


def load_model(model_dir: Path) -> CosyVoice:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    print(f"loading_model={model_dir}")
    start = time.time()
    model = CosyVoice(str(model_dir))
    print(f"model_loaded_in={time.time() - start:.1f}s")
    print(f"sample_rate={model.sample_rate}")
    return model


def save_audio(result: dict, output_path: Path, sample_rate: int) -> None:
    samples = result["tts_speech"].squeeze().numpy()
    sf.write(output_path, samples, sample_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--prompt-wav", default=str(DEFAULT_PROMPT_WAV))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    ensure_local_assets()

    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_wav = Path(args.prompt_wav)
    model_dir = Path(args.model_dir)

    if not prompt_wav.is_absolute():
        prompt_wav = (PROJECT_ROOT / prompt_wav).resolve()
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()

    print(f"project_root={PROJECT_ROOT}")
    print(f"third_party_dir={THIRD_PARTY_DIR}")
    print(f"upstream_root={UPSTREAM_ROOT}")
    print(f"pretrained_dir={PRETRAINED_DIR}")
    print(f"text={args.text}")
    print(f"out={output_path}")
    print(f"prompt_text={args.prompt_text}")
    print(f"prompt_wav={prompt_wav}")
    print(f"model_dir={model_dir}")

    model = load_model(model_dir)

    generated = False
    for index, result in enumerate(
        model.inference_zero_shot(args.text, args.prompt_text, str(prompt_wav), speed=args.speed)
    ):
        current_output = output_path if index == 0 else output_path.with_name(f"{output_path.stem}_{index}{output_path.suffix}")
        save_audio(result, current_output, model.sample_rate)
        duration = result["tts_speech"].shape[1] / model.sample_rate
        print(f"saved={current_output} duration={duration:.2f}s")
        generated = True

    if not generated:
        raise RuntimeError("CosyVoice returned no audio segments")


if __name__ == "__main__":
    main()
