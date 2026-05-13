from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
UPSTREAM_ROOT = THIRD_PARTY_DIR / "CosyVoice_edge"
DEFAULT_MODEL_DIR = PRETRAINED_DIR / "CosyVoice-300M"
DEFAULT_PROMPT_WAV = PROJECT_ROOT / "assets" / "zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
DEFAULT_TEXT = "你好，这是一次最小TTS管道检查。"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from cosyvoice.cli.cosyvoice import CosyVoice


def describe_tensor(name: str, value) -> str:
    shape = list(value.shape)
    dtype = getattr(value, "dtype", type(value).__name__)
    device = getattr(value, "device", "n/a")
    return f"{name}\tshape={shape}\tdtype={dtype}\tdevice={device}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--prompt-wav", default=str(DEFAULT_PROMPT_WAV))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    prompt_wav = Path(args.prompt_wav)
    model_dir = Path(args.model_dir)
    if not prompt_wav.is_absolute():
        prompt_wav = (PROJECT_ROOT / prompt_wav).resolve()
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()

    if not prompt_wav.is_file():
        raise FileNotFoundError(f"prompt wav not found: {prompt_wav}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    model = CosyVoice(str(model_dir))

    normalized_prompt = model.frontend.text_normalize(args.prompt_text, split=False, text_frontend=True)
    normalized_segments = model.frontend.text_normalize(args.text, split=True, text_frontend=True)

    print(f"model_dir={model_dir}")
    print(f"prompt_wav={prompt_wav}")
    print(f"sample_rate={model.sample_rate}")
    print(f"normalized_prompt={normalized_prompt}")
    print("normalized_segments_start")
    for segment in normalized_segments:
        print(segment)
    print("normalized_segments_end")

    first_segment = normalized_segments[0]
    model_input = model.frontend.frontend_zero_shot(first_segment, normalized_prompt, str(prompt_wav), model.sample_rate, '')

    print("model_input_start")
    for key in sorted(model_input):
        print(describe_tensor(key, model_input[key]))
    print("model_input_end")

    prompt_token = model_input["flow_prompt_speech_token"]
    prompt_feat = model_input["prompt_speech_feat"]
    print("stage_map_start")
    print("text_normalize -> frontend_zero_shot -> llm_job -> flow.inference -> hift.inference")
    print(f"prompt_token_length={prompt_token.shape[1]}")
    print(f"prompt_feat_frames={prompt_feat.shape[1]}")
    print("stage_map_end")


if __name__ == "__main__":
    main()
