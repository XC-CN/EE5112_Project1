#!/usr/bin/env python3
"""GPU dialogue system built on top of the local LLM platform."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from llm_platform import LLMPlatform


class DialogueSystem:
    """High-level dialogue loop with conversation persistence."""

    def __init__(self, config_file: str = "config.json") -> None:
        self.config_path = self._resolve_config_path(config_file)
        self.config = self._load_config(self.config_path)
        self.base_dir = self.config_path.parent

        self.platform: Optional[LLMPlatform] = None
        self.conversation_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------
    def _resolve_config_path(self, config_file: str) -> Path:
        path = Path(config_file)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path

    def _load_config(self, path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            print(f"[WARN]  Config file '{path}' not found. Using platform defaults.")
            return LLMPlatform().config
        except json.JSONDecodeError as exc:
            print(f"[WARN]  Invalid JSON in '{path}': {exc}. Using platform defaults.")
            return LLMPlatform().config
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN]  Failed to load config '{path}': {exc}. Using platform defaults.")
            return LLMPlatform().config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        self._ensure_conversation_dir()
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            self.platform = LLMPlatform(self.config, workspace_root=self.base_dir)
            if not self.platform.load_model():
                return False
            print("[OK] Dialogue system initialised with GPU acceleration.")
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[ERROR] Failed to initialise dialogue system: {exc}")
            return False

    def _ensure_conversation_dir(self) -> None:
        dialogue_config = self.config.get("dialogue_config", {})
        conv_dir = dialogue_config.get("conversation_dir", "conversations")
        (self.base_dir / conv_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        if not self.platform:
            return "Error: platform not initialised. Call initialize() first."

        response = self.platform.chat(user_input)
        if self._should_save():
            self._save_conversation_step(user_input, response)
        return response

    def stream_chat(self, user_input: str) -> str:
        if not self.platform:
            return "Error: platform not initialised. Call initialize() first."

        collected = []
        for token in self.platform.stream_chat(user_input):
            print(token, end="", flush=True)
            collected.append(token)
        response = "".join(collected).strip()
        if self._should_save():
            self._save_conversation_step(user_input, response)
        return response

    def clear_conversation(self) -> None:
        if self.platform:
            self.platform.clear_history()

    def show_conversation_stats(self) -> None:
        if not self.platform:
            print("Platform not initialised.")
            return

        info = self.platform.get_model_info()
        print("\n[STATS] Conversation statistics")
        print(f"Model: {Path(info['model_path']).name}")
        print(f"Messages in buffer: {info['conversation_length']}")
        print(f"GPU enabled: {info['hardware_config'].get('gpu_enabled', True)}")
        conv_dir = self.base_dir / self.config.get("dialogue_config", {}).get("conversation_dir", "conversations")
        saved = [p for p in conv_dir.glob("*.json")]
        print(f"Saved conversations: {len(saved)}")
        if self.conversation_id:
            print(f"Current conversation ID: {self.conversation_id}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _should_save(self) -> bool:
        return bool(self.config.get("dialogue_config", {}).get("save_conversations", False))

    def _save_conversation_step(self, user_input: str, response: str) -> None:
        try:
            conv_dir = self.base_dir / self.config.get("dialogue_config", {}).get("conversation_dir", "conversations")
            conv_file = conv_dir / f"{self.conversation_id}.json"

            if conv_file.exists():
                with conv_file.open("r", encoding="utf-8") as handle:
                    conversation = json.load(handle)
            else:
                conversation = {
                    "conversation_id": self.conversation_id,
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
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[WARN]  Failed to save conversation step: {exc}")

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------
    def interactive_mode(self) -> None:
        if not self.platform:
            print("[ERROR] Platform not initialised.")
            return

        streaming_enabled = bool(self.platform.dialogue_config.get("streaming", False))
        print("\n[BOT] Dialogue system ready (GPU mode)")
        print("Commands: 'exit' to quit, 'clear' to reset history, 'stats' for info")
        if streaming_enabled:
            print("Streaming mode enabled: tokens will appear in real time.")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n[USER] You: ").strip()

                if not user_input:
                    continue
                if user_input.lower() == "exit":
                    print("[USER] Bye!")
                    break
                if user_input.lower() == "clear":
                    self.clear_conversation()
                    continue
                if user_input.lower() == "stats":
                    self.show_conversation_stats()
                    continue

                start = time.perf_counter()
                if streaming_enabled:
                    print("[BOT] Assistant: ", end="", flush=True)
                    response = self.stream_chat(user_input)
                    print()
                else:
                    response = self.chat(user_input)
                    print(f"[BOT] Assistant: {response}")
                elapsed = time.perf_counter() - start
                print(f"[TIMER]  Elapsed: {elapsed:.2f}s")
            except KeyboardInterrupt:
                print("\n[USER] Interrupted by user. Bye!")
                break
            except Exception as exc:
                print(f"[ERROR] Error during interaction: {exc}")


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------

def main(argv: Optional[Any] = None) -> None:
    parser = argparse.ArgumentParser(description="GPU-accelerated dialogue system")
    parser.add_argument("--config", "-c", default="config.json", help="Path to configuration JSON file")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args(argv)

    system = DialogueSystem(config_file=args.config)

    if args.no_stream:
        system.config.setdefault("dialogue_config", {})["streaming"] = False

    if not system.initialize():
        sys.exit(1)

    system.interactive_mode()


if __name__ == "__main__":
    main()
