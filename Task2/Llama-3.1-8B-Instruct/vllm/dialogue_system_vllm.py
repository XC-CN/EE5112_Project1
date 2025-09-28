#!/usr/bin/env python3
"""基于Transformers的交互式对话系统。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DialogueConfig:
    system_prompt: str
    max_history: int = 6
    save_conversations: bool = True
    conversation_dir: str = "conversations_transformers"


@dataclass
class SamplingConfig:
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.02

    def to_generation_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

        if self.temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            if self.top_k > 0:
                kwargs["top_k"] = self.top_k
        else:
            kwargs["do_sample"] = False
            if self.top_k > 0:
                kwargs["top_k"] = self.top_k

        return kwargs


@dataclass
class ModelConfig:
    model_name: str
    dtype: str = "bfloat16"
    device: Optional[str] = None
    low_cpu_mem_usage: bool = True
    torch_compile: bool = False
    tensor_parallel_size: int = 1  # legacy选项，占位保持兼容
    gpu_memory_utilization: float = 0.8  # legacy选项，占位保持兼容
    enforce_eager: bool = False  # legacy选项，占位保持兼容
    max_model_len: int = 8192  # legacy选项，占位保持兼容


@dataclass
class ConfigBundle:
    model: ModelConfig
    sampling: SamplingConfig
    dialogue: DialogueConfig


def load_config(path: Path) -> ConfigBundle:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    model_section: Dict[str, Any] = {
        "model_name": raw.get("model_name", "meta-llama/Llama-3.1-8B-Instruct"),
        "dtype": raw.get("dtype", "bfloat16"),
        "device": raw.get("device"),
        "low_cpu_mem_usage": raw.get("low_cpu_mem_usage", True),
        "torch_compile": raw.get("torch_compile", False),
    }

    for legacy_key in ("tensor_parallel_size", "gpu_memory_utilization", "enforce_eager", "max_model_len"):
        if legacy_key in raw:
            model_section[legacy_key] = raw[legacy_key]

    sampling_section = raw.get("sampling", {})
    dialogue_section = raw.get("dialogue", {})

    return ConfigBundle(
        model=ModelConfig(**model_section),
        sampling=SamplingConfig(**sampling_section),
        dialogue=DialogueConfig(**dialogue_section),
    )


class TransformersDialogue:
    """纯Transformers多轮对话包装器。"""

    def __init__(self, config: ConfigBundle, *, base_dir: Path, cache_dir: Optional[Path] = None) -> None:
        self.config = config
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.device: Optional[torch.device] = None
        self.messages: List[Dict[str, str]] = []
        self.conversation_id: Optional[str] = None
        self._ensure_conv_dir()

    def _ensure_conv_dir(self) -> None:
        conv_path = self.base_dir / self.config.dialogue.conversation_dir
        conv_path.mkdir(parents=True, exist_ok=True)
        self.conversation_path = conv_path

    def initialize(self) -> None:
        if self.model is not None:
            return

        model_cfg = self.config.model
        print(f"🚀 Loading model: {model_cfg.model_name}")
        print("   Backend: Hugging Face Transformers")
        if os.name == "nt":
            print("ℹ️  检测到Windows环境，使用纯Transformers推理。")

        trust_remote_code = True
        auth_token = os.getenv("HF_TOKEN")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            default_cache = self.base_dir / "hf_cache"
            default_cache.mkdir(parents=True, exist_ok=True)
            self.cache_dir = default_cache

        tokenizer_kwargs: Dict[str, Any] = {
            "use_fast": True,
            "token": auth_token,
            "trust_remote_code": trust_remote_code,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, **tokenizer_kwargs)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        device = self._resolve_device(model_cfg.device)
        torch_dtype = self._resolve_dtype(model_cfg.dtype, device)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": model_cfg.low_cpu_mem_usage,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, **model_kwargs)
        self.model.to(device)
        self.model.eval()

        if model_cfg.torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
            except Exception as compile_error:
                print(f"⚠️  torch.compile失败，已忽略：{compile_error}")

        self.device = device
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("✅ Model loaded. Ready to chat!\n")

    def _resolve_device(self, preferred: Optional[str]) -> torch.device:
        if preferred:
            norm = preferred.lower()
            if norm == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if norm == "cpu":
                return torch.device("cpu")
            if norm in {"mps", "metal"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self, dtype_name: str, device: torch.device) -> Optional[torch.dtype]:
        mapping = {
            "auto": None,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        dtype = mapping.get(dtype_name.lower(), None)
        if dtype in {torch.float16, torch.bfloat16} and device.type == "cpu":
            return torch.float32
        return dtype

    def _build_chat_messages(self) -> List[Dict[str, str]]:
        assert self.tokenizer is not None

        chat_messages: List[Dict[str, str]] = []
        if self.config.dialogue.system_prompt:
            chat_messages.append({"role": "system", "content": self.config.dialogue.system_prompt})

        history_window = self.config.dialogue.max_history
        history_slice = self.messages[-history_window * 2 :] if history_window > 0 else self.messages
        chat_messages.extend(history_slice)
        return chat_messages

    def chat(self, user_input: str) -> str:
        if not self.model or not self.tokenizer or not self.device:
            raise RuntimeError("Model not initialised. Call initialize() first.")

        self.messages.append({"role": "user", "content": user_input})

        chat_messages = self._build_chat_messages()
        input_ids = self.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs = self.config.sampling.to_generation_kwargs()
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generation_kwargs)

        generated_ids = output_ids[0, input_ids.shape[-1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        self.messages.append({"role": "assistant", "content": response})

        if self.config.dialogue.save_conversations:
            self._save_step(user_input, response)

        return response

    def _save_step(self, user_input: str, response: str) -> None:
        if self.conversation_id is None:
            return

        conv_file = self.conversation_path / f"{self.conversation_id}.json"

        if conv_file.exists():
            with conv_file.open("r", encoding="utf-8") as handle:
                conversation = json.load(handle)
        else:
            conversation = {
                "conversation_id": self.conversation_id,
                "model": self.config.model.model_name,
                "start_time": datetime.now().isoformat(),
                "messages": [],
            }

        conversation["messages"].extend(
            [
                {"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()},
            ]
        )

        with conv_file.open("w", encoding="utf-8") as handle:
            json.dump(conversation, handle, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        self.messages.clear()
        print("🧹 Conversation history cleared.")

    def stats(self) -> None:
        print("\n📊 Conversation stats")
        print(f"Model: {self.config.model.model_name}")
        print(f"Messages stored: {len(self.messages)}")
        if self.conversation_id:
            print(f"Conversation ID: {self.conversation_id}")
        if self.device:
            print(f"Device: {self.device}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformers chat runner for Llama 3.1 8B Instruct")
    parser.add_argument("--config", default="config.json", help="Path to JSON config file")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory to cache downloaded model weights",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving of conversations to disk",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_tokens in sampling config",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"❌ Failed to load config: {exc}")
        return 1

    if args.no_save:
        config.dialogue.save_conversations = False

    if args.max_new_tokens is not None:
        config.sampling.max_tokens = args.max_new_tokens

    base_dir = config_path.resolve().parent
    dialogue = TransformersDialogue(
        config,
        base_dir=base_dir,
        cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None,
    )

    try:
        dialogue.initialize()
    except Exception as exc:
        print(f"❌ Failed to initialise model: {exc}")
        return 1

    print("🤖 Llama 3.1 8B (Transformers) ready!")
    print("Commands: 'exit' to quit, 'clear' to reset memory, 'stats' to inspect cache")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered == "exit":
                print("👋 Bye!")
                break
            if lowered == "clear":
                dialogue.clear()
                continue
            if lowered == "stats":
                dialogue.stats()
                continue

            start = time.perf_counter()
            response = dialogue.chat(user_input)
            elapsed = time.perf_counter() - start
            print(f"🤖 Assistant: {response}")
            print(f"⏱️  Elapsed: {elapsed:.2f}s")
        except KeyboardInterrupt:
            print("\n👋 Interrupted. See you next time!")
            break
        except Exception as exc:
            print(f"❌ Error during interaction: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
