#!/usr/bin/env python3
"""GPU-accelerated LLM platform for EE5112 Task 2."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from llama_cpp import Llama


class LLMPlatform:
    """Local LLM platform with GPU-first defaults."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, *, workspace_root: Optional[Path] = None) -> None:
        self.config = config or self._get_default_config()
        self.model_config = dict(self.config.get("model_config", {}))
        self.generation_config = dict(self.config.get("generation_config", {}))
        self.dialogue_config = dict(self.config.get("dialogue_config", {}))
        self.hardware_config = dict(self.config.get("hardware_config", {}))
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path(__file__).resolve().parent

        self.model_path = self._resolve_model_path(self.model_config.get("model_path"))
        self.llm: Optional[Llama] = None
        self.conversation_history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _get_default_config(self) -> Dict[str, Any]:
        """Return GPU-oriented default configuration."""

        return {
            "model_config": {
                "model_path": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "n_ctx": 4096,
                "n_threads": max(os.cpu_count() or 1, 8),
                "n_batch": 256,
                "seed": -1,
                "verbose": False,
            },
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "max_tokens": 512,
            },
            "dialogue_config": {
                "system_prompt": "You are a helpful AI assistant. Provide clear, concise, and accurate answers.",
                "max_history": 10,
                "streaming": True,
                "save_conversations": True,
                "conversation_dir": "conversations",
            },
            "hardware_config": {
                "gpu_enabled": True,
                "main_gpu": 0,
                "n_gpu_layers": -1,
                "tensor_split": None,
                "force_gpu_env": True,
            },
        }

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        if not model_path:
            raise ValueError("Model path must be provided in the configuration.")

        path = Path(model_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path

    def _prepare_environment(self) -> None:
        if not self.hardware_config.get("gpu_enabled", True):
            return

        if self.hardware_config.get("force_gpu_env", True):
            os.environ.setdefault("LLAMA_CUBLAS", "1")

        main_gpu = self.hardware_config.get("main_gpu")
        if main_gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(main_gpu)

    def _build_llama_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": int(self.model_config.get("n_ctx", 4096)),
            "n_threads": int(self.model_config.get("n_threads", os.cpu_count() or 8)),
            "seed": int(self.model_config.get("seed", -1)),
            "verbose": bool(self.model_config.get("verbose", False)),
        }

        if "n_batch" in self.model_config:
            kwargs["n_batch"] = int(self.model_config["n_batch"])
        if "rope_freq_base" in self.model_config:
            kwargs["rope_freq_base"] = float(self.model_config["rope_freq_base"])
        if "rope_freq_scale" in self.model_config:
            kwargs["rope_freq_scale"] = float(self.model_config["rope_freq_scale"])
        if "f16_kv" in self.model_config:
            kwargs["f16_kv"] = bool(self.model_config["f16_kv"])

        gpu_enabled = self.hardware_config.get("gpu_enabled", True)
        if gpu_enabled:
            kwargs["n_gpu_layers"] = int(self.hardware_config.get("n_gpu_layers", -1))
            tensor_split = self.hardware_config.get("tensor_split")
            if tensor_split:
                if isinstance(tensor_split, str):
                    tensor_split = [float(x) for x in tensor_split.split(",") if x.strip()]
                kwargs["tensor_split"] = tensor_split
            main_gpu = self.hardware_config.get("main_gpu")
            if main_gpu is not None:
                kwargs["main_gpu"] = int(main_gpu)
        else:
            kwargs["n_gpu_layers"] = 0

        return kwargs

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def load_model(self, *, force_reload: bool = False) -> bool:
        if self.llm is not None and not force_reload:
            return True

        if not self.model_path.exists():
            print(f"❌ Model not found at {self.model_path}")
            return False

        self._prepare_environment()

        try:
            llama_kwargs = self._build_llama_kwargs()

            print(f"📦 Loading model from {llama_kwargs['model_path']}")
            print("⚙️  GPU acceleration enabled" if self.hardware_config.get("gpu_enabled", True) else "⚙️  Running on CPU")

            start = time.perf_counter()
            self.llm = Llama(**llama_kwargs)
            elapsed = time.perf_counter() - start
            print(f"✅ Model ready in {elapsed:.2f}s")
            return True
        except TypeError as exc:
            unexpected = str(exc)
            print(f"⚠️  Parameter mismatch when creating Llama instance: {unexpected}")
            print("   Falling back to minimal parameter set.")
            minimal_kwargs = {
                "model_path": str(self.model_path),
                "n_ctx": int(self.model_config.get("n_ctx", 4096)),
                "n_threads": int(self.model_config.get("n_threads", os.cpu_count() or 8)),
                "seed": int(self.model_config.get("seed", -1)),
                "verbose": bool(self.model_config.get("verbose", False)),
                "n_gpu_layers": int(self.hardware_config.get("n_gpu_layers", -1 if self.hardware_config.get("gpu_enabled", True) else 0)),
            }
            try:
                self.llm = Llama(**minimal_kwargs)
                print("✅ Model ready with fallback parameters.")
                return True
            except Exception as inner_exc:
                print(f"❌ Failed to load model after fallback: {inner_exc}")
                self.llm = None
                return False
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Failed to load model: {exc}")
            self.llm = None
            return False

    # ------------------------------------------------------------------
    # Generation utilities
    # ------------------------------------------------------------------
    def generate_response(self, prompt: str, **overrides: Any) -> str:
        if not self.llm:
            return "Error: Model not loaded. Call load_model() first."

        config = {**self.generation_config, **overrides}
        clean_prompt = self._sanitize_prompt(prompt)

        try:
            response = self.llm(
                clean_prompt,
                max_tokens=int(config.get("max_tokens", 512)),
                temperature=float(config.get("temperature", 0.7)),
                top_p=float(config.get("top_p", 0.9)),
                top_k=int(config.get("top_k", 40)),
                repeat_penalty=float(config.get("repeat_penalty", 1.1)),
                stop=["</s>", "[INST]", "[/INST]", "Human:", "Assistant:"],
                echo=False,
            )
            return response["choices"][0]["text"].strip()
        except Exception as exc:  # pragma: no cover - defensive
            return f"Error generating response: {exc}"

    def stream_response(self, prompt: str, **overrides: Any) -> Iterable[str]:
        if not self.llm:
            yield "(error: model not loaded)"
            return

        config = {**self.generation_config, **overrides}
        clean_prompt = self._sanitize_prompt(prompt)

        try:
            stream = self.llm(
                clean_prompt,
                max_tokens=int(config.get("max_tokens", 512)),
                temperature=float(config.get("temperature", 0.7)),
                top_p=float(config.get("top_p", 0.9)),
                top_k=int(config.get("top_k", 40)),
                repeat_penalty=float(config.get("repeat_penalty", 1.1)),
                stop=["</s>", "[INST]", "[/INST]", "Human:", "Assistant:"],
                echo=False,
                stream=True,
            )

            for chunk in stream:
                if not chunk:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("text", "")
                if delta:
                    yield delta
                finish_reason = choices[0].get("finish_reason")
                if finish_reason is not None:
                    break
        except Exception as exc:  # pragma: no cover - defensive
            yield f"[streaming error: {exc}]"

    def chat(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        self._trim_history()
        context = self._build_conversation_context()
        response = self.generate_response(context)
        self.conversation_history.append({"role": "assistant", "content": response})
        self._trim_history()
        return response

    def stream_chat(self, user_input: str) -> Iterable[str]:
        self.conversation_history.append({"role": "user", "content": user_input})
        self._trim_history()
        context = self._build_conversation_context()

        collected: List[str] = []
        for token in self.stream_response(context):
            collected.append(token)
            yield token

        answer = "".join(collected).strip()
        self.conversation_history.append({"role": "assistant", "content": answer})
        self._trim_history()

    # ------------------------------------------------------------------
    # Conversation utilities
    # ------------------------------------------------------------------
    def _trim_history(self) -> None:
        max_history = int(self.dialogue_config.get("max_history", 10))
        if max_history <= 0:
            return
        # Keep max_history pairs (user+assistant). Allow one extra user message in-flight.
        while len(self.conversation_history) > max_history * 2 + 1:
            self.conversation_history.pop(0)

    def _build_conversation_context(self) -> str:
        system_prompt = self.dialogue_config.get(
            "system_prompt",
            "You are a helpful AI assistant. Provide clear, concise, and accurate answers.",
        )

        context = "<|start_header_id|>system<|end_header_id|>\n\n"
        context += system_prompt.strip() + "\n"
        context += "<|eot_id|>\n\n"

        for message in self.conversation_history[-max(0, int(self.dialogue_config.get("max_history", 10))) * 2 :]:
            role = message.get("role", "user")
            content = message.get("content", "")
            context += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}\n<|eot_id|>\n\n"

        context += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return context

    @staticmethod
    def _sanitize_prompt(prompt: str) -> str:
        marker = "<|begin_of_text|>"
        while prompt.startswith(marker + marker):
            prompt = prompt[len(marker) :]
        return prompt

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def clear_history(self) -> None:
        self.conversation_history.clear()
        print("🧹 Conversation history cleared.")

    def get_model_info(self) -> Dict[str, Any]:
        status = "loaded" if self.llm else "not_loaded"
        return {
            "status": status,
            "model_path": str(self.model_path),
            "model_config": self.model_config,
            "generation_config": self.generation_config,
            "hardware_config": self.hardware_config,
            "conversation_length": len(self.conversation_history),
        }


__all__ = ["LLMPlatform"]
