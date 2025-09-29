# -*- coding: utf-8 -*-
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "4bit/llava-v1.5-7b-5GB"
BASE_PROCESSOR_ID = "llava-hf/llava-1.5-7b-hf"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
HF_CACHE_DIR = Path(__file__).resolve().parent.parent / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))



class LlavaChat:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA device required for loading the 4-bit LLaVA model.")

        print(f"[INFO] Loading processor ({MODEL_ID}) ...")
        self.processor = AutoProcessor.from_pretrained(
            BASE_PROCESSOR_ID,
            trust_remote_code=True,
            use_fast=False,
            trust_remote_code=True,
        )

        print("[INFO] Loading 4-bit model (this may take a while the first time)...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
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

    def _generate(self, messages: List[dict], image: Image.Image, *, max_new_tokens: int, temperature: float) -> str:
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
        )
        generated_tokens = output[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

    def chat(self, image_path: Path, *, save: bool, max_new_tokens: int, temperature: float) -> None:
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        messages: List[dict] = []
        history: List[dict] = []
        image_attached = False

        print("\n[INFO] Interactive chat ready. Type your question and press Enter; type `exit` / `quit` to finish.\n")
        turn = 1
        while True:
            user_input = input(f"[USER] Turn {turn} question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "bye"}:
                print("[SYSTEM] Conversation terminated.")
                break

            content = [{"type": "text", "text": user_input}]
            if not image_attached:
                content.insert(0, {"type": "image"})
                image_attached = True

            messages.append({"role": "user", "content": content})
            reply = self._generate(messages, image, max_new_tokens=max_new_tokens, temperature=temperature)
            print(f"[ASSISTANT] {reply}\n")

            messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
            history.append({"user": user_input, "assistant": reply})
            turn += 1

        if save and history:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / f"llava_v15_4bit_chat_{timestamp}.json"
            record = {
                "model_id": MODEL_ID,
                "image_path": str(image_path),
                "history": history,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
            with output_path.open("w", encoding="utf-8") as fp:
                json.dump(record, fp, ensure_ascii=False, indent=2)
            print(f"[INFO] Conversation saved to {output_path}")


def collect_images() -> Path:
    candidates = sorted(
        [p for p in DATA_DIR.glob("**/*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    )
    if not candidates:
        raise SystemExit("δ data/ Ŀ¼ҵͼƬʹ --image ָ·ȷͼƬ")

    print("[INFO] Please choose an image to use:")
    for idx, path in enumerate(candidates, start=1):
        rel = path.relative_to(DATA_DIR.parent)
        print(f"  {idx:>2}: {rel}")

    while True:
        choice = input("Enter an index (or type a full path): ").strip()
        if not choice:
            continue
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
            print("[WARN] Index out of range, please try again.")
            continue
        candidate = Path(choice).expanduser()
        if candidate.is_file():
            return candidate
        print("[WARN] Path not recognised, please try again.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive LLaVA 1.5 7B 4-bit chat")
    parser.add_argument("--image", type=str, help="Image path; if omitted will list images under data/")
    parser.add_argument("--save", action="store_true", help="Save conversation transcript")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image) if args.image else collect_images()

    chat = LlavaChat()
    chat.chat(
        image_path=image_path,
        save=args.save,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

