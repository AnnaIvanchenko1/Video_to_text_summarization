from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)


SHARED_DIR = Path(
    os.getenv("SHARED_DIR", "/shared")
)

SEGMENTS_DIR = SHARED_DIR / "segments"
FRAMES_DIR = SHARED_DIR / "frames"

MODEL_NAME = os.getenv(
    "LLAVA_MODEL",
    "llava-hf/llava-1.5-7b-hf",
)

HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_command(command: list[str]) -> None:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n"
            f"{' '.join(command)}\n"
            f"{result.stderr}"
        )


def get_metadata() -> dict[str, Any]:
    return json.loads(
        (
            SEGMENTS_DIR / "metadata.json"
        ).read_text(encoding="utf-8")
    )


def extract_frame(
    video_path: Path,
    timestamp: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
    )


def load_model() -> tuple[Any, Any]:
    model_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }

    processor_kwargs: dict[str, Any] = {}

    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN
        processor_kwargs["token"] = HF_TOKEN

    if DEVICE == "cuda":
        model_kwargs.update(
            {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        **processor_kwargs,
    )

    model.eval()

    return model, processor


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(DEVICE)


def analyze_frame(
    image_path: Path,
    audio_text: str,
    model: Any,
    processor: Any,
) -> str:
    image = Image.open(image_path).convert("RGB")

    prompt = (
        "USER: <image>\n"
        "Describe the video frame in detail. "
        f"The spoken content in this scene is: "
        f'"{audio_text[:300]}". '
        "Describe visible people, objects, actions, "
        "setting and context.\n"
        "ASSISTANT:"
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    device = model_device(model)

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                inputs[key] = value.to(
                    device=device,
                    dtype=torch.float16
                    if DEVICE == "cuda"
                    else torch.float32,
                )
            else:
                inputs[key] = value.to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    text = processor.batch_decode(
        output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    if "ASSISTANT:" in text:
        text = text.split(
            "ASSISTANT:",
            maxsplit=1,
        )[1]

    return text.strip()


def main() -> None:
    FRAMES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = get_metadata()
    video_path = Path(metadata["video_path"])

    input_path = (
        SEGMENTS_DIR / "selected_segments.json"
    )

    segments = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    model, processor = load_model()

    fragments: list[dict[str, Any]] = []

    for index, segment in enumerate(segments):
        start = segment["start"]
        end = segment["end"]
        middle = (start + end) / 2

        frame_path = FRAMES_DIR / f"frame_{index}.jpg"

        extract_frame(
            video_path,
            middle,
            frame_path,
        )

        try:
            visual_text = analyze_frame(
                frame_path,
                segment["audio_text"],
                model,
                processor,
            )
        except Exception as error:
            visual_text = (
                f"[Visual analysis failed: {error}]"
            )

        fragments.append(
            {
                "start": start,
                "end": end,
                "audio_text": segment["audio_text"],
                "visual_text": visual_text,
                "frame_path": str(frame_path),
            }
        )

        print(
            f"[LLaVA] Processed frame {index + 1}"
        )

    output_path = (
        SEGMENTS_DIR / "synchronized_fragments.json"
    )

    output_path.write_text(
        json.dumps(
            fragments,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[LLaVA] Saved: {output_path}")


if __name__ == "__main__":
    main()