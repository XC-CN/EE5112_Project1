# llama.cpp GPU 部署指南（Llama-3.1-8B-Instruct）

## 概述

`Task2/Llama-3.1-8B-Instruct/Llama cpp python` 提供了一个针对 NVIDIA CUDA GPU 优化的本地 LLM 对话平台。系统基于 `llama-cpp-python`，默认加载 `../models/Llama-3.1-8B-Instruct-Q4_K_M.gguf` 量化模型，并启用流式输出、会话保存与 GPU 加速配置。

## 环境要求

- **GPU**: NVIDIA RTX 40/50 系列（建议 16GB 显存及以上）
- **驱动**: NVIDIA 驱动 + CUDA 11.8 及以上
- **Python**: 3.9 – 3.12
- **操作系统**: Windows 10/11、Linux 或 macOS（Metal 支持自行调整）

> ⚠️ 建议确保 `llama-cpp-python` 已通过 `pip install llama-cpp-python[cuda]` 安装，以启用 cuBLAS 后端。

## 模型准备

1. 访问 Hugging Face，下载 `Llama-3.1-8B-Instruct-Q4_K_M.gguf`（或其它合适的 GGUF 量化）文件。
2. 将文件放置在 `Task2/Llama-3.1-8B-Instruct/models/` 目录下。
3. 若使用不同的文件名或量化方式，请同步修改 `config.json` 或启动时传入 `--config` 指定自定义路径。

## 快速开始

1. **安装依赖**
   ```powershell
   cd Task2/Llama-3.1-8B-Instruct/'Llama cpp python'
   pip install -r requirements.txt
   ```

2. **验证 CUDA 构建**（可选）
   ```powershell
   python - <<'PY'
   from llama_cpp import Llama
   info = Llama.build_info()
   print("GPU-enabled:", info.get("gpu_type", "unknown"))
   PY
   ```

3. **运行对话系统**
   - PowerShell（默认）
     ```powershell
     .\run_dialogue.ps1
     ```
   - Windows CMD
     ```batch
     run_dialogue.bat
     ```
   - Linux / macOS
     ```bash
     ./run_dialogue.sh
     ```

执行脚本会自动设置 `LLAMA_CUBLAS=1` 以强制 llama.cpp 使用 GPU。首次加载模型会进行权重映射，耗时 10–60 秒。

## 配置说明

所有参数存储在 `config.json` 中，分为四个部分：

- `model_config`：模型路径、上下文长度、线程和批次大小等。
- `generation_config`：采样与生成参数。
- `dialogue_config`：系统提示词、历史长度、是否流式输出和会话保存。
- `hardware_config`：GPU 相关选项（`n_gpu_layers=-1` 表示将所有 Transformer 层迁移到 GPU）。

> `tensor_split` 可用于多 GPU 部署，填写形如 `[0.5, 0.5]` 的权重分配列表即可。

## 目录结构

```
Task2/Llama-3.1-8B-Instruct/Llama cpp python
├── config.json          # GPU 推理参数
├── dialogue_system.py   # 对话系统入口（支持 CLI 参数）
├── llm_platform.py      # GPU 优化的 LLM 平台封装
├── README.md            # 本文档
├── requirements.txt     # 依赖列表（CUDA 版本）
├── run_dialogue.bat     # Windows CMD 启动脚本
├── run_dialogue.ps1     # PowerShell 启动脚本
└── run_dialogue.sh      # Linux/macOS 启动脚本
```

会话日志默认保存在 `Task2/GPU/conversations`，文件命名为时间戳。

## 常见问题

1. **提示未找到模型**：确认 `config.json` 中的 `model_path` 指向 `Task2/models` 目录，或使用绝对路径。
2. **推理仍使用 CPU**：检查 `LLAMA_CUBLAS` 是否为 `1`，确保 `llama-cpp-python` 编译时启用了 CUDA。
3. **显存不足**：
   - 降低 `model_config.n_ctx`
   - 调整 `generation_config.max_tokens`
   - 使用更小的量化模型（如 Q4_K_M → Q4_0）
4. **首轮响应较慢**：这属于权重加载的正常现象，可在脚本中先运行暖机提问，如 `warmup_prompt = "Hello"`。

## 进阶用法

- 禁用流式输出：
  ```powershell
  python dialogue_system.py --no-stream
  ```
- 自定义配置文件：
  ```powershell
  python dialogue_system.py --config custom_config.json
  ```
- 指定多 GPU 拆分：编辑 `config.json` 中的 `tensor_split` 为 JSON 数组，例如 `[0.4, 0.6]`。

---

如需在课程报告中引用，请说明部署基于 `llama-cpp-python` GPU 版本与 `Llama-3.1-8B-Instruct` GGUF 量化模型。祝推理顺利！
