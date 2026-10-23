from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


SHARED_DIR = Path(
    os.getenv("SHARED_DIR", "/shared")
)

SEGMENTS_DIR = SHARED_DIR / "segments"
SUMMARY_DIR = SHARED_DIR / "summary"

MODEL_NAME = os.getenv(
    "PHI_MODEL",
    "microsoft/Phi-3.5-mini-instruct",
)

HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> tuple[Any, Any]:
    tokenizer_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }

    if HF_TOKEN:
        tokenizer_kwargs["token"] = HF_TOKEN
        model_kwargs["token"] = HF_TOKEN

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        **tokenizer_kwargs,
    )

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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return tokenizer, model


def get_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(DEVICE)


def build_prompt(
    fragments: list[dict[str, Any]],
) -> list[dict[str, str]]:
    scenes = "\n\n".join(
        [
            (
                f"SCENE {index + 1}\n"
                f"VIDEO: {fragment['visual_text']}\n"
                f"AUDIO: {fragment['audio_text']}"
            )
            for index, fragment in enumerate(fragments)
        ]
    )

    return [
        {
            "role": "system",
            "content": (
                "Write one coherent video description in English. "
                "Start with 'This video is about...' or "
                "'The video shows...'. "
                "Do not mention timestamps, scene numbers, "
                "transcripts or audio recordings. "
                "Use third person. "
                "Combine the spoken content and visual information. "
                "Do not invent unsupported facts. "
                "Keep the description detailed and factual."
            ),
        },
        {
            "role": "user",
            "content": (
                "Synchronized video information:\n\n"
                f"{scenes}\n\n"
                "Write the final description."
            ),
        },
    ]


def generate_summary(
    fragments: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
) -> str:
    messages = build_prompt(fragments)

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n\n".join(
            message["content"]
            for message in messages
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    device = get_model_device(model)

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = output[0][input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = (
        SEGMENTS_DIR / "synchronized_fragments.json"
    )

    fragments = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    tokenizer, model = load_model()

    summary = generate_summary(
        fragments,
        tokenizer,
        model,
    )

    output_path = SUMMARY_DIR / "final_summary.txt"

    output_path.write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FINAL VIDEO DESCRIPTION")
    print("=" * 70)
    print(summary)
    print("=" * 70)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()