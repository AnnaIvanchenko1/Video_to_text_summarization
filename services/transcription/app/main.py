from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import whisperx


SHARED_DIR = Path(
    os.getenv("SHARED_DIR", "/shared")
)

AUDIO_DIR = SHARED_DIR / "audio"
SEGMENTS_DIR = SHARED_DIR / "segments"

LIGHT_MODEL_NAME = os.getenv(
    "WHISPER_LIGHT_MODEL",
    "tiny",
)

FULL_MODEL_NAME = os.getenv(
    "WHISPER_FULL_MODEL",
    "small",
)

LANGUAGE = os.getenv("LANGUAGE", "en")
MAX_SEGMENTS = int(
    os.getenv("MAX_SEGMENTS", "30")
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


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


def extract_audio_segment(
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
) -> None:
    duration = end - start

    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
    )


def get_input_video() -> Path:
    metadata_path = SEGMENTS_DIR / "metadata.json"

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    return Path(metadata["video_path"])


def get_text(result: dict[str, Any]) -> str:
    return " ".join(
        segment["text"].strip()
        for segment in result.get("segments", [])
        if segment.get("text")
    ).strip()


def transcribe(
    model: Any,
    audio_path: Path,
    batch_size: int,
) -> str:
    result = model.transcribe(
        str(audio_path),
        batch_size=batch_size,
        language=LANGUAGE,
    )

    return get_text(result)


def full_transcribe(
    model: Any,
    audio_path: Path,
) -> str:
    result = model.transcribe(
        str(audio_path),
        batch_size=4 if DEVICE == "cuda" else 1,
        language=LANGUAGE,
    )

    if DEVICE == "cuda":
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=result.get(
                    "language",
                    LANGUAGE,
                ),
                device=DEVICE,
            )

            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                str(audio_path),
                DEVICE,
            )

            del align_model

        except Exception as error:
            print(f"[WhisperX] Alignment failed: {error}")

    return get_text(result)


def main() -> None:
    selected_path = (
        SEGMENTS_DIR / "selected_segments.json"
    )

    candidate_path = (
        SEGMENTS_DIR / "candidate_segments.json"
    )

    candidate_segments = json.loads(
        candidate_path.read_text(encoding="utf-8")
    )

    video_path = get_input_video()

    light_model = whisperx.load_model(
        LIGHT_MODEL_NAME,
        device="cpu",
        compute_type="int8",
    )

    selected_segments: list[dict[str, Any]] = []

    for index, segment in enumerate(candidate_segments):
        if len(selected_segments) >= MAX_SEGMENTS:
            break

        start = segment["start"]
        end = segment["end"]

        audio_path = AUDIO_DIR / f"light_{index}.wav"

        print(
            f"[tiny] Segment {index + 1}: "
            f"{start:.2f}-{end:.2f}"
        )

        extract_audio_segment(
            video_path,
            start,
            end,
            audio_path,
        )

        try:
            text = transcribe(
                light_model,
                audio_path,
                batch_size=1,
            )
        except Exception as error:
            print(f"[tiny] Error: {error}")
            text = ""

        if text:
            selected_segments.append(
                {
                    "start": start,
                    "end": end,
                    "audio_text_light": text,
                }
            )

    del light_model

    if not selected_segments:
        raise RuntimeError(
            "No speech segments were found."
        )

    full_model = whisperx.load_model(
        FULL_MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    result_segments: list[dict[str, Any]] = []

    for index, segment in enumerate(selected_segments):
        start = segment["start"]
        end = segment["end"]

        audio_path = AUDIO_DIR / f"full_{index}.wav"

        extract_audio_segment(
            video_path,
            start,
            end,
            audio_path,
        )

        try:
            text = full_transcribe(
                full_model,
                audio_path,
            )
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                text = ""
            else:
                raise
        except Exception as error:
            print(f"[small] Error: {error}")
            text = ""

        if not text:
            text = segment["audio_text_light"]

        if text:
            result_segments.append(
                {
                    "start": start,
                    "end": end,
                    "audio_text": text,
                }
            )

    if not result_segments:
        raise RuntimeError(
            "Full transcription produced no text."
        )

    selected_path.write_text(
        json.dumps(
            result_segments,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[WhisperX] Saved: {selected_path}"
    )


if __name__ == "__main__":
    main()