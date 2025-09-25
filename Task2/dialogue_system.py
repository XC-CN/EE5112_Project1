#!/usr/bin/env python3
"""
Dialogue System for Task2 - EE5112 Human Robot Interaction
Multi-turn dialogue system using local LLM platform
"""

import os
import sys
import json
import time
from datetime import datetime
from llm_platform import LLMPlatform

class DialogueSystem:
    """Multi-turn dialogue system with conversation management"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize dialogue system
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.platform = None
        self.conversation_id = None
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using default config.")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}. Using default config.")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            "model_config": {
                "model_path": "models/Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "n_gpu_layers": 35,
                "n_ctx": 4096
            },
            "generation_config": {
                "temperature": 0.7,
                "max_tokens": 512
            },
            "dialogue_config": {
                "max_history": 10,
                "system_prompt": "You are a helpful AI assistant.",
                "save_conversations": True,
                "conversation_dir": "conversations"
            }
        }
    
    def initialize(self) -> bool:
        """
        Initialize the dialogue system
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create conversation directory
            conv_dir = self.config["dialogue_config"]["conversation_dir"]
            os.makedirs(conv_dir, exist_ok=True)
            
            # Generate conversation ID
            self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Initialize LLM platform
            model_path = self.config["model_config"]["model_path"]
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                print("Please run 'python download_model.py' to download the model first.")
                return False
            
            self.platform = LLMPlatform(model_path, self.config["model_config"])
            
            if not self.platform.load_model():
                return False
            
            print("✅ Dialogue System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing dialogue system: {e}")
            return False
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and generate response
        
        Args:
            user_input: User's input message
            
        Returns:
            str: System's response
        """
        if not self.platform:
            return "Error: System not initialized. Please call initialize() first."
        
        try:
            # Generate response using platform
            response = self.platform.chat(user_input)
            
            # Save conversation if enabled
            if self.config["dialogue_config"]["save_conversations"]:
                self._save_conversation_step(user_input, response)
            
            return response
            
        except Exception as e:
            return f"Error processing input: {e}"
    
    def _save_conversation_step(self, user_input: str, response: str):
        """Save individual conversation step"""
        try:
            conv_file = f"{self.config['dialogue_config']['conversation_dir']}/{self.conversation_id}.json"
            
            # Load existing conversation or create new
            if os.path.exists(conv_file):
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
            else:
                conversation = {
                    "conversation_id": self.conversation_id,
                    "start_time": datetime.now().isoformat(),
                    "messages": []
                }
            
            # Add new messages
            conversation["messages"].extend([
                {"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
            ])
            
            # Save conversation
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save conversation step: {e}")
    
    def clear_conversation(self):
        """Clear current conversation history"""
        if self.platform:
            self.platform.clear_history()
            print("Conversation history cleared.")
    
    def show_conversation_stats(self):
        """Show conversation statistics"""
        if not self.platform:
            print("System not initialized.")
            return
        
        info = self.platform.get_model_info()
        print(f"\n📊 Conversation Statistics:")
        print(f"Model: {os.path.basename(info['model_path'])}")
        print(f"Messages in history: {info['conversation_length']}")
        print(f"Conversation ID: {self.conversation_id}")
        
        # Show saved conversations
        conv_dir = self.config["dialogue_config"]["conversation_dir"]
        if os.path.exists(conv_dir):
            conv_files = [f for f in os.listdir(conv_dir) if f.endswith('.json')]
            print(f"Total saved conversations: {len(conv_files)}")
    
    def interactive_mode(self):
        """Run interactive dialogue mode"""
        if not self.platform:
            print("❌ System not initialized.")
            return
        
        streaming_enabled = self.config.get("dialogue_config", {}).get("streaming", False)
        print("\n🤖 Dialogue System Ready!")
        print("Commands: 'exit' to quit, 'clear' to reset history, 'stats' for metrics")
        if streaming_enabled:
            print("(Streaming enabled: tokens will stream in real time)")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_conversation()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_conversation_stats()
                    continue
                elif not user_input:
                    continue
                
                start_time = time.time()
                streaming_enabled = self.config.get("dialogue_config", {}).get("streaming", False)
                if streaming_enabled:
                    print("🤖 Assistant: ", end="", flush=True)
                    collected = []
                    for token in self.platform.stream_chat(user_input):
                        print(token, end="", flush=True)
                        collected.append(token)
                    response = ''.join(collected).strip()
                    # Save conversation step
                    if self.config["dialogue_config"]["save_conversations"]:
                        self._save_conversation_step(user_input, response)
                    print()  # newline after streaming response
                else:
                    response = self.chat(user_input)
                    print(f"🤖 Assistant: {response}")
                end_time = time.time()
                print(f"⏱️  Elapsed: {end_time - start_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("=== Task2: Dialogue System - EE5112 ===\n")
    
    # Initialize dialogue system
    system = DialogueSystem()
    
    if not system.initialize():
        return
    
    # Run interactive mode
    system.interactive_mode()

if __name__ == "__main__":
    main()
