from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import webrtcvad


INPUT_VIDEO = Path(
    os.getenv("INPUT_VIDEO", "/data/input/video.mp4")
)

SHARED_DIR = Path(
    os.getenv("SHARED_DIR", "/shared")
)

AUDIO_DIR = SHARED_DIR / "audio"
SEGMENTS_DIR = SHARED_DIR / "segments"

SAMPLE_RATE = 16_000
FRAME_DURATION_MS = 30
VAD_MODE = 2
GAP_THRESHOLD = 0.3
MAX_SEGMENT_LEN = 30.0
MIN_SEGMENT_LEN = 1.0


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


def get_video_duration(video_path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        text=True,
    ).strip()

    return float(output)


def extract_audio(video_path: Path, audio_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(audio_path),
        ]
    )


def detect_speech_intervals(
    audio_path: Path,
) -> list[tuple[float, float]]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        audio_data = wav_file.readframes(frame_count)

    if channels != 1:
        raise ValueError("Audio must be mono")

    if sample_width != 2:
        raise ValueError("Audio must be 16-bit PCM")

    if frame_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {frame_rate}"
        )

    audio = np.frombuffer(audio_data, dtype=np.int16)

    frame_size = int(
        SAMPLE_RATE * FRAME_DURATION_MS / 1000
    )

    num_frames = len(audio) // frame_size
    vad = webrtcvad.Vad(VAD_MODE)

    flags: list[bool] = []

    for frame_index in range(num_frames):
        start = frame_index * frame_size
        end = start + frame_size
        frame = audio[start:end]

        flags.append(
            vad.is_speech(
                frame.tobytes(),
                SAMPLE_RATE,
            )
        )

    intervals: list[tuple[float, float]] = []
    start_frame: int | None = None

    for index, is_speech in enumerate(flags):
        if is_speech and start_frame is None:
            start_frame = index

        elif not is_speech and start_frame is not None:
            intervals.append((start_frame, index))
            start_frame = None

    if start_frame is not None:
        intervals.append((start_frame, len(flags)))

    seconds_per_frame = FRAME_DURATION_MS / 1000

    return [
        (
            start * seconds_per_frame,
            end * seconds_per_frame,
        )
        for start, end in intervals
    ]


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]

        if current_start - last_end <= GAP_THRESHOLD:
            merged[-1] = (
                last_start,
                max(last_end, current_end),
            )
        else:
            merged.append((current_start, current_end))

    return merged


def split_intervals(
    intervals: list[tuple[float, float]],
    video_duration: float,
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []

    for start, end in intervals:
        start = max(0.0, start)
        end = min(video_duration, end)

        if end - start < MIN_SEGMENT_LEN:
            continue

        current = start

        while current < end:
            segment_end = min(
                current + MAX_SEGMENT_LEN,
                end,
            )

            if segment_end - current >= MIN_SEGMENT_LEN:
                result.append(
                    {
                        "start": round(current, 3),
                        "end": round(segment_end, 3),
                    }
                )

            current = segment_end

    return result


def main() -> None:
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}"
        )

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = AUDIO_DIR / "full_audio.wav"

    print(f"[VAD] Input: {INPUT_VIDEO}")
    print("[VAD] Extracting audio...")
    extract_audio(INPUT_VIDEO, audio_path)

    duration = get_video_duration(INPUT_VIDEO)

    print("[VAD] Detecting speech...")
    raw_intervals = detect_speech_intervals(audio_path)

    merged_intervals = merge_intervals(raw_intervals)

    segments = split_intervals(
        merged_intervals,
        video_duration=duration,
    )

    (SEGMENTS_DIR / "vad_intervals.json").write_text(
        json.dumps(
            raw_intervals,
            indent=2,
        ),
        encoding="utf-8",
    )

    (SEGMENTS_DIR / "candidate_segments.json").write_text(
        json.dumps(
            segments,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "video_path": str(INPUT_VIDEO),
        "duration": duration,
        "sample_rate": SAMPLE_RATE,
        "segments_count": len(segments),
    }

    (SEGMENTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"[VAD] Segments created: {len(segments)}")


if __name__ == "__main__":
    main()