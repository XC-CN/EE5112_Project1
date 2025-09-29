import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
DEVICE = "cuda"
DTYPE = torch.float16
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class LlavaRunner:
    def __init__(self) -> None:
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_fast=False,
        )
        print("Loading model...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        )
        self.model.to(DEVICE)

    def _prepare_images(self, prompt: str, image: Image.Image):
        num_images = prompt.count("<image>")
        if num_images <= 1:
            return image
        return [image for _ in range(num_images)]

    def _generate(self, messages: List[dict], image: Image.Image, *, max_new_tokens: int, temperature: float) -> str:
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
            tensor = v.to(DEVICE)
            if tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.to(DTYPE)
            new_inputs[k] = tensor
        inputs = new_inputs
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        generated_tokens = output[:, inputs["input_ids"].shape[-1]:]
        text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return text.strip()

    def caption(self, image: Image.Image, prompt: str) -> str:
        print("Generating caption...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages, image, max_new_tokens=200, temperature=0.2)

    def conversation(self, image: Image.Image, prompts: List[str]) -> List[str]:
        messages: List[dict] = []
        responses: List[str] = []
        image_attached = False
        for idx, user_text in enumerate(prompts, start=1):
            print(f"Generating turn {idx}...")
            content = [{"type": "text", "text": user_text}]
            if not image_attached:
                content.insert(0, {"type": "image"})
                image_attached = True
            messages.append({"role": "user", "content": content})
            reply = self._generate(messages, image, max_new_tokens=256, temperature=0.3)
            responses.append(reply)
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": reply}],
                }
            )
        return responses


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = DATA_DIR / "demo_scene.jpg"
    image = Image.open(image_path)

    runner = LlavaRunner()

    caption_prompt = "Provide a concise caption for this image."
    caption = runner.caption(image, caption_prompt)

    conversation_prompts = [
        "What activity are the cats engaged in?",
        "Mention two details about their surroundings.",
    ]
    conversation_responses = runner.conversation(image, conversation_prompts)

    record = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "image_path": str(image_path.relative_to(DATA_DIR.parent)),
        "caption_prompt": caption_prompt,
        "caption_response": caption,
        "conversation": [
            {"user": prompt, "assistant": response}
            for prompt, response in zip(conversation_prompts, conversation_responses)
        ],
    }

    output_path = RESULTS_DIR / "llava_v16_demo.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(record, fp, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()



