# -*- coding: utf-8 -*-
import argparse
import json
import os
import torch
from pathlib import Path
from typing import Any, Dict

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "4bit/llava-v1.5-7b-5GB"
BASE_PROCESSOR_ID = "llava-hf/llava-1.5-7b-hf"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HF_CACHE_DIR = Path(__file__).resolve().parent.parent / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))



def load_model() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for the 4-bit checkpoint. Please run on a GPU-enabled machine.")

    processor = AutoProcessor.from_pretrained(
        BASE_PROCESSOR_ID,
        trust_remote_code=True,
        use_fast=False,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    return {"processor": processor, "model": model}


def run_demo(image_path: Path, output_path: Path, prompt: str) -> None:
    bundle = load_model()
    processor = bundle["processor"]
    model = bundle["model"]

    image = Image.open(image_path).convert("RGB")

    def generate(text_prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        inputs = processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
        )
        generated = processor.batch_decode(output, skip_special_tokens=True)[0]
        # remove the prompt portion
        return generated[len(text_prompt):].strip()

    caption_prompt = f"USER: {prompt}\nASSISTANT:"
    caption = generate(caption_prompt, max_new_tokens=160, temperature=0.2)

    q1_prompt = (
        "USER: Describe the main subjects in the picture in more detail.\nASSISTANT:"
    )
    q2_prompt = (
        "USER: Mention two safety or environmental observations you can infer from the scene.\nASSISTANT:"
    )

    answer1 = generate(q1_prompt, max_new_tokens=200, temperature=0.3)
    answer2 = generate(q2_prompt, max_new_tokens=220, temperature=0.3)

    record = {
        "model_id": MODEL_ID,
        "image_path": str(image_path),
        "caption_prompt": caption_prompt,
        "caption_response": caption,
        "conversation": [
            {"user": q1_prompt.split("ASSISTANT:")[0].replace("USER:", "").strip(), "assistant": answer1},
            {"user": q2_prompt.split("ASSISTANT:")[0].replace("USER:", "").strip(), "assistant": answer2},
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA 1.5 7B 4-bit inference demo")
    parser.add_argument(
        "--image",
        type=str,
        default=str(DATA_DIR / "demo_scene.jpg"),
        help="Path to the image used for the demo",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Provide a concise caption for this image.",
        help="Caption prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "llava_v15_4bit_demo.json"),
        help="Where to save the resulting JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_path = Path(args.output)

    if not image_path.exists():
        raise SystemExit("CUDA device required for the 4-bit checkpoint. Please run on a GPU-enabled machine.")

    run_demo(image_path=image_path, output_path=output_path, prompt=args.prompt)
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()

