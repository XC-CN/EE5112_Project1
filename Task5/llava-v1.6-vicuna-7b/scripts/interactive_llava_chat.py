# -*- coding: utf-8 -*-
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class LlavaRunner:
    def __init__(self, *, device: str = DEVICE, dtype: torch.dtype = DTYPE) -> None:
        self.device = device
        self.dtype = dtype
        print(f"[INFO] Device: {self.device}  |  Precision: {self.dtype}")
        print(f"[INFO] Loading processor ({MODEL_ID}) ...")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_fast=False,
        )
        print("[INFO] Loading model ... (first load may take longer)")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
        )

    def _prepare_images(self, prompt: str, image: Image.Image):
        num_images = prompt.count("<image>")
        if num_images <= 1:
            return image
        return [image for _ in range(num_images)]

    def _generate(
        self,
        messages: List[dict],
        image: Image.Image,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images = self._prepare_images(prompt, image)
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
        )
        target_device = torch.device(DEVICE) if DEVICE == "cuda" else torch.device("cpu")
        new_inputs = {}
        for k, v in inputs.items():
            tensor = v.to(target_device)
            if tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.to(self.dtype)
            new_inputs[k] = tensor
        outputs = self.model.generate(
            **new_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        generated_tokens = outputs[:, new_inputs["input_ids"].shape[-1]:]
        text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return text.strip()

    def interactive_chat(
        self,
        image: Image.Image,
        *,
        max_new_tokens: int,
        temperature: float,
        log_history: bool = True,
    ) -> List[dict]:
        messages: List[dict] = []
        history: List[dict] = []
        image_attached = False

        print("\n[INFO] Entering chat mode. Ask your question and press Enter; type `exit` / `quit` to finish.\n")
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
            reply = self._generate(
                messages,
                image,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            print(f"[鍔╂墜] {reply}\n")

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": reply}],
                }
            )

            if log_history:
                history.append({"user": user_input, "assistant": reply})

            turn += 1

        return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA 澶氳疆鍥炬枃瀵硅瘽 (GPU 浼樺厛)")
    parser.add_argument(
        "--image",
        type=str,
        help="瑕佷娇鐢ㄧ殑鍥剧墖璺緞锛岄粯璁や粠 data 鐩綍浜掑姩閫夋嫨",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save conversation history into the results directory",
    )
    return parser.parse_args()


def select_image_from_data_dir() -> Path:
    candidates = sorted(
        [
            p
            for p in DATA_DIR.glob("**/*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    if not candidates:
        raise FileNotFoundError(
            "No image found under the data directory; please use --image or place a file manually."
        )

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


def main() -> None:
    args = parse_args()

    try:
        if args.image:
            image_path = Path(args.image).expanduser().resolve()
        else:
            image_path = select_image_from_data_dir()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    print(f"[INFO] Loaded image: {image_path}")

    runner = LlavaRunner()
    history = runner.interactive_chat(
        image,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    if args.save and history:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record = {
            "model_id": MODEL_ID,
            "device": DEVICE,
            "image_path": str(image_path),
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "history": history,
        }
        output_path = RESULTS_DIR / f"llava_chat_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(record, fp, ensure_ascii=False, indent=2)
        print(f"[INFO] 瀵硅瘽璁板綍宸蹭繚瀛樺埌: {output_path}")


if __name__ == "__main__":
    main()


