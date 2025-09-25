#!/usr/bin/env python3
"""
LLM Platform for Task2 - EE5112 Human Robot Interaction
本地 LLM 平台 (llama-cpp-python) - 增强：支持流式输出 (token 级)
"""

import os
import sys
from llama_cpp import Llama
from typing import Optional, List, Dict, Any
import json
import time

class LLMPlatform:
    """Local LLM Platform for dialogue systems"""
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM Platform
        
        Args:
            model_path: Path to the GGUF model file
            config: Configuration dictionary for model parameters
        """
        self.model_path = model_path
        self.config = config or self._get_default_config()
        self.llm = None
        self.conversation_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """默认推理参数 (若外部未提供 model_config 则使用)

        注意：这里只放与 llama_cpp.Llama 构造及采样相关的键，避免混淆。
        """
        return {
            "n_gpu_layers": 35,      # GPU 加速层数（-1 表示全部，受显存限制）
            "n_ctx": 4096,           # 上下文窗口
            "n_threads": 8,          # CPU 线程数
            "verbose": False,        # 关闭底层冗长日志
            "seed": -1,              # 随机种子（-1 表示随机）
            # 采样 & 生成相关
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "max_tokens": 512,
        }
    
    def load_model(self) -> bool:
        """
        Load the LLM model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            print(f"📦 Loading model: {self.model_path}")
            print("⏳ First load may take from several seconds to a few minutes (depends on disk/CPU/GPU)...")
            
            # Remove model_path from config if present to avoid duplicate keyword argument
            llama_config = {k: v for k, v in self.config.items() if k != 'model_path'}
            # Llama 默认会插入 BOS token，这里我们已在提示中显式写入 <|begin_of_text|>
            # 为避免重复，默认关闭 add_bos，保留用户手动覆盖的可能
            llama_config.setdefault("add_bos", False)
            
            self.llm = Llama(
                model_path=self.model_path,
                **llama_config
            )
            
            print("✅ Model loaded successfully! You can start chatting now.")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from the model
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response
        """
        if not self.llm:
            return "Error: Model not loaded. Please call load_model() first."
        
        try:
            # Merge default config with provided kwargs
            generation_config = {**self.config, **kwargs}
            
            # Generate response
            response = self.llm(
                prompt,
                max_tokens=generation_config.get("max_tokens", 512),
                temperature=generation_config.get("temperature", 0.7),
                top_p=generation_config.get("top_p", 0.9),
                top_k=generation_config.get("top_k", 40),
                repeat_penalty=generation_config.get("repeat_penalty", 1.1),
                stop=["</s>", "[INST]", "[/INST]", "Human:", "Assistant:"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            return f"Error while generating response: {e}"

    # ----------------------- 流式输出相关 -----------------------
    def stream_response(self, prompt: str, **kwargs):
        """流式生成回复，逐 token (或子词片段) 产出。

        Yields:
            str: 新增的 token 文本片段
        """
        if not self.llm:
            yield "(error: model not loaded)"
            return

        generation_config = {**self.config, **kwargs}
        try:
            stream = self.llm(
                prompt,
                max_tokens=generation_config.get("max_tokens", 512),
                temperature=generation_config.get("temperature", 0.7),
                top_p=generation_config.get("top_p", 0.9),
                top_k=generation_config.get("top_k", 40),
                repeat_penalty=generation_config.get("repeat_penalty", 1.1),
                stop=["</s>", "[INST]", "[/INST]", "Human:", "Assistant:"],
                echo=False,
                stream=True
            )

            for chunk in stream:
                if not chunk:
                    continue
                choices = chunk.get('choices') or []
                if not choices:
                    continue
                delta = choices[0].get('text', '')
                if delta:
                    yield delta
                # 结束原因（有些版本在最后一个块才提供 finish_reason）
                finish_reason = choices[0].get('finish_reason')
                if finish_reason is not None:
                    break
        except Exception as e:
            yield f"[streaming error: {e}]"

    def stream_chat(self, user_input: str):
        """带上下文的流式对话接口 (生成器)

        Args:
            user_input: 用户输入

        Yields:
            str: 模型逐步生成的文本片段
        """
        # 先记录用户消息
        self.conversation_history.append({"role": "user", "content": user_input})
        context = self._build_conversation_context()
        collected = []
        for token in self.stream_response(context):
            collected.append(token)
            yield token
        full_text = ''.join(collected).strip()
        self.conversation_history.append({"role": "assistant", "content": full_text})
    
    def chat(self, user_input: str) -> str:
        """
        Chat with the model using conversation history
        
        Args:
            user_input: User's input message
            
        Returns:
            str: Model's response
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Build conversation context
        context = self._build_conversation_context()
        
        # Generate response
        response = self.generate_response(context)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _build_conversation_context(self) -> str:
        """Build conversation context from history"""
        if not self.conversation_history:
            return ""
        
        context = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        context += "You are a helpful AI assistant. Provide clear, concise, and helpful responses.\n"
        context += "<|eot_id|>\n\n"
        
        for message in self.conversation_history[-6:]:  # Keep last 6 messages
            if message["role"] == "user":
                context += "<|start_header_id|>user<|end_header_id|>\n\n"
                context += message["content"] + "\n"
                context += "<|eot_id|>\n\n"
            else:
                context += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                context += message["content"] + "\n"
                context += "<|eot_id|>\n\n"
        
        context += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return context
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🧹 Conversation history cleared.")
    
    def save_conversation(self, filename: str):
        """Save conversation history to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"💾 Conversation saved to: {filename}")
        except Exception as e:
            print(f"Failed to save conversation: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.llm:
            return {"status": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "config": self.config,
            "conversation_length": len(self.conversation_history),
            "status": "Model loaded successfully"
        }

def main():
    """Main function for testing the LLM Platform"""
    print("=== LLM Platform - Task2 EE5112 (Streaming Demo) ===\n")
    
    # Model path (adjust as needed)
    model_path = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run 'python download_model.py' to download the model first.")
        return
    
    # Initialize platform
    platform = LLMPlatform(model_path)
    
    # Load model
    if not platform.load_model():
        return
    
    print("\n🤖 Platform ready! Type to chat, use 'exit' to quit, 'clear' to reset history, 'info' for model details.")
    print("=" * 50)
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                platform.clear_history()
                continue
            elif user_input.lower() == 'info':
                info = platform.get_model_info()
                print(json.dumps(info, indent=2))
                continue
            elif not user_input:
                continue
            
            # Generate response
            start_time = time.time()
            response = platform.chat(user_input)
            end_time = time.time()
            
            print(f"Assistant: {response}")
            print(f"[Generated in {end_time - start_time:.2f}s]")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
