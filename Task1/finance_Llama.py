from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# 模型ID（金融专用Llama）
model_id = "tarun7r/Finance-Llama-8B"

print("Loading Finance Llama model...")

# 加载模型（带内存优化）
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # 半精度节省显存
        device_map="auto",          # 自动分配设备
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✓ Model loaded with FP16")
except Exception as e:
    print(f"FP16 loading failed: {e}")
    print("Fallback: loading on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✓ Model loaded on CPU")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)
print("✓ Pipeline created successfully!")


def chat():
    """交互式聊天"""
    print("\n" + "="*60)
    print("Finance Llama Chatbot - Interactive Mode")
    print("="*60)
    print("Type 'quit' or 'exit' to stop\n")

    system_prompt = (
        "You are a highly knowledgeable finance chatbot. "
        "Your purpose is to provide accurate, insightful, and actionable financial advice to users."
    )

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Assistant: Goodbye! 👋")
            break
        if not user_input:
            continue

        # 构建prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

        print("Assistant: (thinking...)")

        try:
            outputs = generator(
                prompt,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                use_cache=True,
                max_new_tokens=50
            )

            generated_text = outputs[0]["generated_text"]
            response_start = generated_text.rfind("User:")
            if response_start != -1:
                response = generated_text[response_start:].split("Assistant:")[-1].strip()
            else:
                response = generated_text[len(prompt):].strip()

            print(f"Assistant: {response}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    chat()
