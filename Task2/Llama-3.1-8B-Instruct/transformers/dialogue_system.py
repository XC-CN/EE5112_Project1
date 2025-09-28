#!/usr/bin/env python3
"""Interactive dialogue system built on Transformers."""

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
import requests
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
        # 将采样配置转换为Transformers可直接使用的参数字典
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
    tensor_parallel_size: int = 1  # legacy placeholder for compatibility
    gpu_memory_utilization: float = 0.8  # legacy placeholder
    enforce_eager: bool = False  # legacy placeholder
    max_model_len: int = 8192  # legacy placeholder


@dataclass
class ConfigBundle:
    model: ModelConfig
    sampling: SamplingConfig
    dialogue: DialogueConfig


def load_config(path: Path) -> ConfigBundle:
    # 读取JSON配置文件并拆分为模型/采样/对话三个子配置
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


def load_env_file(env_path: Path) -> None:
    # 支持从.env加载密钥（例如HF_TOKEN），未提前设置的变量才会写入环境
    """Load key=value pairs from a .env file without overriding existing env vars."""
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

class TransformersDialogue:
    """Transformers multi-turn dialogue wrapper."""

    def __init__(self, config: ConfigBundle, *, base_dir: Path, cache_dir: Optional[Path] = None) -> None:
        # base_dir用于定位配置与缓存目录，cache_dir可选指定模型缓存位置

        if cache_dir is not None and not isinstance(cache_dir, Path):
            cache_dir = Path(cache_dir)
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
        # 确保对话记录目录存在，避免保存历史时失败
        conv_path = self.base_dir / self.config.dialogue.conversation_dir
        conv_path.mkdir(parents=True, exist_ok=True)
        self.conversation_path = conv_path

    def initialize(self) -> None:
        if self.model is not None:
            return

        model_cfg = self.config.model
        print(f"[INFO] Loading model: {model_cfg.model_name}")
        print("[INFO] Backend: Hugging Face Transformers")
        if os.name == "nt":
            print("[INFO] Detected Windows environment; using pure Transformers inference.")

        # 读取HF_TOKEN以便访问Hugging Face受限模型，用户可在.env或环境变量中配置
        trust_remote_code = True
        auth_token = os.getenv("HF_TOKEN")

        # 优先使用用户传入的缓存目录，若未指定则在项目目录下创建默认缓存
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            default_cache = self.base_dir / "hf_cache"
            default_cache.mkdir(parents=True, exist_ok=True)
            self.cache_dir = default_cache
        # 加载分词器，若在线访问被拒绝则尝试读取本地缓存
        tokenizer_kwargs: Dict[str, Any] = {
            "use_fast": True,
            "trust_remote_code": trust_remote_code,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
        if auth_token:
            tokenizer_kwargs["token"] = auth_token
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, **tokenizer_kwargs)
        except (requests.exceptions.HTTPError, OSError) as http_err:
            if self._has_local_weights(model_cfg.model_name):
                print(f"[WARN] Failed to fetch tokenizer online: {http_err}. Falling back to local cache.")
                tokenizer_kwargs.pop("token", None)
                tokenizer_kwargs["local_files_only"] = True
                self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, **tokenizer_kwargs)
            else:
                raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        device = self._resolve_device(model_cfg.device)
        torch_dtype = self._resolve_dtype(model_cfg.dtype, device)

        # 设置模型加载参数，必要时同样支持离线加载
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": model_cfg.low_cpu_mem_usage,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if auth_token:
            model_kwargs["token"] = auth_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, **model_kwargs)
        except (requests.exceptions.HTTPError, OSError) as http_err:
            if self._has_local_weights(model_cfg.model_name):
                print(f"[WARN] Failed to fetch model online: {http_err}. Falling back to local cache.")
                model_kwargs.pop("token", None)
                model_kwargs["local_files_only"] = True
                self.model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, **model_kwargs)
            else:
                raise

        self.model.to(device)
        self.model.eval()

        if model_cfg.torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
            except Exception as compile_error:
                print(f"[WARN] torch.compile failed; continuing without compilation: {compile_error}")

        self.device = device
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("[INFO] Model loaded. Ready to chat!\n")

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


    def _has_local_weights(self, model_name: str) -> bool:
        """Check whether cached weights exist for the target model."""
        # 检查缓存目录中是否已存在模型快照，以便离线运行时直接加载
        if not self.cache_dir:
            return False
        cache_root = Path(self.cache_dir)
        repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
        snapshot_dir = repo_dir / "snapshots"
        if snapshot_dir.exists():
            for child in snapshot_dir.iterdir():
                if child.is_dir():
                    return True
        return False
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
        # 主推理入口：维护消息历史并调用大模型生成回复
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
        print("[INFO] Conversation history cleared.")

    def stats(self) -> None:
        print("\n[INFO] Conversation stats")
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


def resolve_config_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    script_dir = Path(__file__).resolve().parent
    candidate = (script_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def main() -> int:
    args = parse_args()

    load_env_file(Path(__file__).resolve().parent / ".env")

    config_path = resolve_config_path(args.config)
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load config: {exc}")
        return 1

    if args.no_save:
        config.dialogue.save_conversations = False

    if args.max_new_tokens is not None:
        config.sampling.max_tokens = args.max_new_tokens

    base_dir = config_path.parent
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    dialogue = TransformersDialogue(
        config,
        base_dir=base_dir,
        cache_dir=cache_dir,
    )

    try:
        dialogue.initialize()
    except Exception as exc:
        print(f"[ERROR] Failed to initialise model: {exc}")
        return 1

    print("[INFO] Llama 3.1 8B (Transformers) ready!")
    print("Commands: 'exit' to quit, 'clear' to reset memory, 'stats' to inspect cache")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n[USER] You: ").strip()
            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered == "exit":
                print("[INFO] Bye!")
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
            print(f"[BOT] Assistant: {response}")
            print(f"[INFO] Elapsed: {elapsed:.2f}s")
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted. See you next time!")
            break
        except Exception as exc:
            print(f"[ERROR] Error during interaction: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
