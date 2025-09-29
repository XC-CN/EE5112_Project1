#!/usr/bin/env python3
"""Interactive dialogue system powered by vLLM + Hugging Face Transformers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


@dataclass
class DialogueConfig:
    system_prompt: str
    max_history: int = 6
    save_conversations: bool = True
    conversation_dir: str = "conversations_vllm"


@dataclass
class SamplingConfig:
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.02

    def to_sampling_params(self) -> SamplingParams:
        # Convert to vLLM's sampling parameters.
        return SamplingParams(
            n=1,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k if self.top_k > 0 else -1,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
        )


@dataclass
class ModelConfig:
    model_name: str
    tokenizer_name: Optional[str] = None
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.92
    enforce_eager: bool = False
    max_model_len: Optional[int] = 8192
    max_num_seqs: Optional[int] = 16
    swap_space: int = 6
    download_dir: Optional[str] = None
    quantization: Optional[str] = None


@dataclass
class RuntimeConfig:
    conda_env: Optional[str] = None


@dataclass
class ConfigBundle:
    model: ModelConfig
    sampling: SamplingConfig
    dialogue: DialogueConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_config(path: Path) -> ConfigBundle:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    model_section: Dict[str, Any] = {
        "model_name": raw.get("model_name", "meta-llama/Llama-3.1-8B-Instruct"),
        "tokenizer_name": raw.get("tokenizer_name"),
        "dtype": raw.get("dtype", "float16"),
        "tensor_parallel_size": raw.get("tensor_parallel_size", 1),
        "gpu_memory_utilization": raw.get("gpu_memory_utilization", 0.92),
        "enforce_eager": raw.get("enforce_eager", False),
        "max_model_len": raw.get("max_model_len", 8192),
        "max_num_seqs": raw.get("max_num_seqs", 16),
        "swap_space": raw.get("swap_space", 6),
        "download_dir": raw.get("download_dir"),
        "quantization": raw.get("quantization"),
    }

    sampling_section = raw.get("sampling", {})
    dialogue_section = raw.get("dialogue", {})

    runtime_section = raw.get("runtime", {})

    return ConfigBundle(
        model=ModelConfig(**model_section),
        sampling=SamplingConfig(**sampling_section),
        dialogue=DialogueConfig(**dialogue_section),
        runtime=RuntimeConfig(**runtime_section),
    )


def load_env_file(env_path: Path) -> None:
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


class VLLMDialogue:
    """vLLM-backed multi-turn dialogue wrapper."""

    def __init__(self, config: ConfigBundle, *, base_dir: Path) -> None:
        self.config = config
        self.base_dir = base_dir
        self.tokenizer: Optional[AutoTokenizer] = None
        self.llm: Optional[LLM] = None
        self.messages: List[Dict[str, str]] = []
        self.conversation_id: Optional[str] = None
        self.conversation_path: Optional[Path] = None
        self._ensure_conv_dir()

    def _ensure_conv_dir(self) -> None:
        conv_path = self.base_dir / self.config.dialogue.conversation_dir
        conv_path.mkdir(parents=True, exist_ok=True)
        self.conversation_path = conv_path

    def initialize(self) -> None:
        if self.llm is not None:
            return

        model_cfg = self.config.model
        print(f"[INFO] Loading model with vLLM: {model_cfg.model_name}")

        auth_token = os.getenv("HF_TOKEN")
        download_dir = Path(model_cfg.download_dir) if model_cfg.download_dir else (self.base_dir / "hf_cache")
        download_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_kwargs: Dict[str, Any] = {
            "use_fast": False,
            "trust_remote_code": True,
            "cache_dir": str(download_dir),
        }
        if auth_token:
            tokenizer_kwargs["token"] = auth_token

        tokenizer_name = model_cfg.tokenizer_name or model_cfg.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        llm_kwargs: Dict[str, Any] = {
            "model": model_cfg.model_name,
            "tokenizer": tokenizer_name,
            "tensor_parallel_size": model_cfg.tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": model_cfg.dtype,
            "download_dir": str(download_dir),
            "enforce_eager": model_cfg.enforce_eager,
            "gpu_memory_utilization": model_cfg.gpu_memory_utilization,
            "swap_space": model_cfg.swap_space,
        }
        if model_cfg.max_model_len is not None:
            llm_kwargs["max_model_len"] = model_cfg.max_model_len
        if model_cfg.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = model_cfg.max_num_seqs
        if model_cfg.quantization:
            llm_kwargs["quantization"] = model_cfg.quantization

        self.llm = LLM(**llm_kwargs)
        print("[INFO] Model loaded successfully. Ready for dialogue.")

    def _build_prompt(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        self._trim_history()
        chat_messages: List[Dict[str, str]] = []
        if self.config.dialogue.system_prompt:
            chat_messages.append({"role": "system", "content": self.config.dialogue.system_prompt})
        chat_messages.extend(self.messages)
        assert self.tokenizer is not None
        return self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

    def _trim_history(self) -> None:
        max_hist = max(0, self.config.dialogue.max_history)
        if max_hist == 0:
            self.messages = self.messages[-1:]
            return
        self.messages = self.messages[-max_hist:]

    def generate(self, user_message: str) -> str:
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        prompt = self._build_prompt(user_message)
        outputs = self.llm.generate([prompt], sampling_params=self.config.sampling.to_sampling_params())
        first_output = outputs[0].outputs[0].text.strip()
        self.messages.append({"role": "assistant", "content": first_output})
        return first_output

    def reset(self) -> None:
        self.messages.clear()
        self.conversation_id = None

    def save_conversation(self) -> Optional[Path]:
        if not self.config.dialogue.save_conversations or not self.messages:
            return None
        if self.conversation_path is None:
            return None

        if self.conversation_id is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.conversation_id = f"conversation-{timestamp}"

        conv_file = self.conversation_path / f"{self.conversation_id}.json"
        try:
            conv_file.write_text(json.dumps(self.messages, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"[WARN] Failed to save conversation: {exc}")
            return None
        return conv_file


def interactive_session(dialogue: VLLMDialogue) -> None:
    print("Type 'exit' or 'quit' to end the session. Type 'reset' to clear history.\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Session terminated by user.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "reset":
            dialogue.reset()
            print("[INFO] Dialogue history cleared.")
            continue

        start = time.perf_counter()
        assistant_reply = dialogue.generate(user_input)
        elapsed = time.perf_counter() - start
        print(f"Assistant: {assistant_reply}")
        print(f"[INFO] Generation time: {elapsed:.2f}s\n")

    saved_path = dialogue.save_conversation()
    if saved_path:
        print(f"[INFO] Conversation saved to {saved_path}")


def ensure_conda_environment(target_env: Optional[str]) -> None:
    """如果指定了 Conda 环境且当前未激活，则通过 `conda run` 重新执行脚本。"""

    if not target_env:
        return

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).name == target_env:
        return

    conda_executable = shutil.which("conda")
    if conda_executable is None:
        print(f"[WARN] 未找到 conda 可执行文件，继续使用当前环境。")
        return

    script_path = Path(__file__).resolve()
    command = [
        conda_executable,
        "run",
        "-n",
        target_env,
        "python",
        str(script_path),
    ] + sys.argv[1:]

    print(f"[INFO] 正在切换至 Conda 环境: {target_env}")
    result = subprocess.run(command, check=False)
    sys.exit(result.returncode)


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Llama 3.1 dialogue runner using vLLM.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration JSON file.")
    parser.add_argument("--prompt", type=str, default=None, help="Run a single prompt instead of interactive chat.")
    parser.add_argument("--no-save", action="store_true", help="Disable saving the conversation history.")
    parser.add_argument("--dotenv", type=str, default=None, help="Optional path to .env file for HF tokens.")

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent

    if args.dotenv:
        load_env_file(Path(args.dotenv))

    try:
        config = load_config(base_dir / args.config)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    ensure_conda_environment(config.runtime.conda_env)

    if args.no_save:
        config.dialogue.save_conversations = False

    dialogue = VLLMDialogue(config=config, base_dir=base_dir)
    dialogue.initialize()

    if args.prompt:
        reply = dialogue.generate(args.prompt)
        print(f"Assistant: {reply}")
        saved = dialogue.save_conversation()
        if saved:
            print(f"[INFO] Conversation saved to {saved}")
        return

    interactive_session(dialogue)


if __name__ == "__main__":
    run_cli()
