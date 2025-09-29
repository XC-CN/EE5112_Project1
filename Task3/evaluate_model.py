import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer, util

sem_model = SentenceTransformer("all-MiniLM-L6-v2")  # 只加载一次
# -----------------------------
# 配置
# -----------------------------
MODEL_NAME = "databricks/dolly-v2-3b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 372
BATCH_SIZE = 4
SAMPLE_SIZE = 500
MAX_NEW_TOKENS = 20  # 给多一点空间，避免截断

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------------
# 加载模型与 tokenizer
# -----------------------------
# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.float16
)

# 加载 LoRA 适配器
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./dolly-lora-boolq")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./dolly-lora-boolq")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model.eval()
print("已加载微调后的 LoRA 模型")

# -----------------------------
# 推理函数（避免首字符缺失）
# -----------------------------
def batch_generate(prompts, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 贪婪解码
            temperature=0,
            pad_token_id=tokenizer.pad_token_id
        )
    responses = []
    prompt_lens = inputs['attention_mask'].sum(dim=1)  # 每个 prompt 实际长度
    for output, prompt_len in zip(outputs, prompt_lens):
        generated_ids = output[prompt_len:]  # 从实际长度切分
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(text.strip())
    return responses

# -----------------------------
# 答案解析函数（改进版）
# -----------------------------
def normalize_answer(pred: str):
    """
    将模型输出标准化为 True 或 False
    优先规则匹配，其次用 sentence-transformers 语义相似度辅助
    """
    pred_norm = pred.lower().strip()

    # 1. 明确匹配 true/false
    match = re.search(r"\b(true|false)\b", pred_norm)
    if match:
        return True if match.group(1) == "true" else False

    # 2. 规则匹配常见表达
    true_markers = ["yes", "correct", "right", "indeed", "absolutely", "affirmative"]
    false_markers = ["no", "not", "incorrect", "wrong", "negative", "never"]

    if any(word in pred_norm for word in true_markers):
        return True
    if any(word in pred_norm for word in false_markers):
        return False

    # 3. 用 sentence-transformers 判断语义相似度
    candidates = ["True", "False"]
    emb_pred = sem_model.encode(pred, convert_to_tensor=True)
    emb_candidates = sem_model.encode(candidates, convert_to_tensor=True)

    sims = util.cos_sim(emb_pred, emb_candidates)[0]
    best_idx = sims.argmax().item()

    return True if candidates[best_idx].lower() == "true" else False
DATASET = "google/boolq"

results = []
dataset = load_dataset(DATASET, split="validation")

# 随机抽样
indices = np.random.choice(len(dataset), SAMPLE_SIZE, replace=False)
sampled_data = [dataset[i] for i in indices]

prompts = []
references = []

# 构建 prompt 和 reference（使用训练时的格式）
for data in sampled_data:
    prompt = f"Passage: {data['passage']}\nQuestion: {data['question']}\nAnswer only with True/False:"
    prompts.append(prompt)
    references.append(data['answer'])

# 计时开始
start_time = time.time()

# 分批生成
all_preds = []
correct = 0

for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"Generating {DATASET}", unit="batch"):
    batch_prompts = prompts[i:i+BATCH_SIZE]
    batch_refs = references[i:i+BATCH_SIZE]
    
    responses = batch_generate(batch_prompts, max_new_tokens=MAX_NEW_TOKENS)
    all_preds.extend(responses)
    
    # 实时计算累计正确数
    for pred, ref in zip(responses, batch_refs):
        norm = normalize_answer(pred)
        if norm == ref:
            correct += 1
    
    # 实时显示进度和正确率
    processed = min(i + BATCH_SIZE, SAMPLE_SIZE)
    accuracy = correct / processed * 100
    tqdm.write(f"Processed {processed}/{SAMPLE_SIZE} | Current Accuracy: {accuracy:.2f}%")

elapsed_time = time.time() - start_time

# 最终准确率
final_accuracy = correct / SAMPLE_SIZE * 100
results.append({
    "dataset": DATASET,
    "accuracy (%)": f"{final_accuracy:.2f}",
    "inference_time(s)": f"{elapsed_time:.2f}"
})

# 输出表格
df = pd.DataFrame(results)
print("\n" + "="*60)
print("微调后 LoRA 模型评估结果:")
print("="*60)
print(df.to_string(index=False, float_format='%.2f'))
print("="*60)

# 保存结果
df.to_csv("lora_evaluation_results.csv", index=False)
print(f"\n结果已保存到: lora_evaluation_results.csv")

# 显示一些预测示例
print(f"\n前5个预测示例:")
for i in range(min(5, len(all_preds))):
    pred = all_preds[i]
    ref = references[i]
    norm = normalize_answer(pred)
    correct_mark = "✓" if norm == ref else "✗"
    print(f"样本 {i+1}: {correct_mark}")
    print(f"  预测: '{pred}' -> {norm}")
    print(f"  真实: {ref}")
    print()
