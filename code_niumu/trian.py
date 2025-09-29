# -----------------------------
# LoRA 微调 Dolly-v2-3b（BoolQ）
# -----------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# -----------------------------
# 配置
# -----------------------------
MODEL_NAME = "databricks/dolly-v2-3b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 372
BATCH_SIZE = 4
EPOCHS = 3
LR = 5e-4  # 提高学习率，适合小数据集快速收敛
MAX_LENGTH = 128  # prompt + answer
OUTPUT_DIR = "./dolly-lora-boolq"

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# 加载模型和 tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.float16
)

# -----------------------------
# LoRA 配置
# -----------------------------
# 为 Dolly v2 (基于 GPT-NeoX/Pythia) 设置合适的 LoRA 目标层
model_type = getattr(model.config, "model_type", "")
if model_type == "gpt_neox":
    # GPT-NeoX 架构常见线性层命名
    target_modules = [
        "query_key_value",    # 注意力QKV合并层
        "dense",              # 注意力输出投影
        "dense_h_to_4h",      # MLP上投影
        "dense_4h_to_h"       # MLP下投影
    ]
else:
    # 其他模型回退到常见命名（如 LLaMA/GPTJ）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.train()

# -----------------------------
# 加载 BoolQ 数据集（限制样本数量）
# -----------------------------
dataset = load_dataset("google/boolq")
train_data = dataset["train"]
val_data = dataset["validation"]

# 限制训练和验证集大小
TRAIN_SAMPLES = 500
VAL_SAMPLES = 500

# 随机选择训练样本
train_indices = np.random.choice(len(train_data), TRAIN_SAMPLES, replace=False)
train_subset = train_data.select(train_indices)

# 随机选择验证样本
val_indices = np.random.choice(len(val_data), VAL_SAMPLES, replace=False)
val_subset = val_data.select(val_indices)

def preprocess(example):
    prompt = f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer only with True/False:"
    completion = " True" if example["answer"] else " False"
    full_text = prompt + completion

    # 分别编码以获知提示长度，用于标签掩码
    prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, padding=False)["input_ids"]
    tok = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")

    labels = tok["input_ids"].copy()
    # 将提示部分标签置为 -100，仅训练答案部分
    mask_len = min(len(prompt_ids), MAX_LENGTH)
    labels[:mask_len] = [-100] * mask_len

    tok["labels"] = labels
    return tok

train_dataset = train_subset.map(preprocess, remove_columns=train_subset.column_names)
val_dataset = val_subset.map(preprocess, remove_columns=val_subset.column_names)

print(f"训练样本数: {len(train_dataset)}")
print(f"验证样本数: {len(val_dataset)}")

# -----------------------------
# Data collator
# -----------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# -----------------------------
# 训练参数
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,  # 更频繁的日志记录
    save_steps=100,    # 更频繁的保存
    save_total_limit=2,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
    warmup_steps=20,   # 减少预热步数，适合小数据集
    weight_decay=0.01  # 添加权重衰减
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# -----------------------------
# 开始微调
# -----------------------------
trainer.train()

# -----------------------------
# 保存 LoRA 权重
# -----------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA 微调完成，模型保存在:", OUTPUT_DIR)
