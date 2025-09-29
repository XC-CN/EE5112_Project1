# Llama-3.1-8B-Instruct · Transformers 运行指南

本目录提供一个基于 **Hugging Face Transformers** 的纯 Python 对话脚本，可在 Windows、macOS、Linux 上直接运行，无需 vLLM。脚本会读取官方 `meta-llama/Llama-3.1-8B-Instruct` 权重，通过 `AutoModelForCausalLM` 实现多轮对话。

## 环境准备

1. **创建虚拟环境**（建议 Python 3.10+）

   ```bash
   conda create -n llama31 -y
   conda activate llama31
   ```

2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

   默认仅包含 `torch`、`transformers` 等基础依赖，可根据硬件补充 `accelerate`、`bitsandbytes` 等组件。

## Hugging Face 访问凭证

1. 登录 [Hugging Face](https://huggingface.co/)。
2. 在模型主页 [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 接受许可。
3. 在 [Tokens 页面](https://huggingface.co/settings/tokens) 生成 **Read** 权限的个人令牌。
4. 通过环境变量提供令牌：

   ```bash
   export HF_TOKEN="hf_your_read_token"
   ```

   Windows PowerShell 可使用：`$env:HF_TOKEN="hf_xxx"`。

## 运行对话系统

1. 切换到脚本目录：

   ```bash
   cd Task2/Llama-3.1-8B-Instruct/transformers
   ```

2. 启动交互式对话：

   ```bash
   python dialogue_system.py --config config.json
   ```

   可选参数：
   - `--cache-dir`：自定义模型缓存目录，默认 `./hf_cache`。
   - `--no-save`：关闭对话记录写入。
   - `--max-new-tokens`：临时覆盖生成长度配置。

3. 控制台命令：
   - 普通输入：生成回复
   - `stats`：查看模型/上下文信息
   - `clear`：清除记忆
   - `exit`：退出

对话历史默认保存到 `conversations_transformers/` 目录，可用于复盘或记录。

## 模型缓存

首次运行会自动下载 safetensors 权重与分词器并缓存在 `hf_cache/`。该目录可手动迁移或重用，避免重复下载。

## 常见问题

| 问题                         | 解决方案                                                                 |
| ---------------------------- | ------------------------------------------------------------------------ |
| 报错 `Config file not found` | 确认传入 `--config` 参数或在脚本目录运行，或将配置路径改为绝对路径。 |
| 控制台出现编码错误           | 本脚本仅输出 ASCII，若仍提示编码问题，可设置 `PYTHONIOENCODING=utf-8` |
| GPU 显存不足                 | 将配置中的 `dtype` 调整为 `float32` 并使用 CPU，或减少 `max_tokens`。   |
| 加载速度慢                   | 预先下载模型到 `hf_cache`，并确保使用 SSD。

## 目录结构

```
transformers/
├── config.json                 # 模型、采样、会话配置
├── dialogue_system.py          # 交互式对话脚本
├── requirements.txt            # 依赖清单
├── README.md                   # 使用说明
├── hf_cache/                   # (运行后生成) 模型缓存
└── conversations_transformers/ # (运行后生成) 对话记录
```

## 下一步

- 将 `dialogue_system.py` 封装成 REST API 或集成到机器人控制流程。
- 根据需求扩展采样参数，或加入记忆检索/工具调用组件。
- 若需量化部署，可尝试使用 `bitsandbytes` 或 `AWQ` 等方案进一步压缩模型。

祝使用顺利，如需其他自动化脚本可以继续提出需求。
