# 使用 vLLM 运行 Llama 3.1 8B

该目录提供了一个轻量对话脚本，结合 Hugging Face `transformers` 的聊天模板能力与 `vllm` 高性能推理。

## 环境准备

1. 安装依赖（确保 PyTorch 版本与本地 CUDA 环境匹配）。

   ```bash
   pip install -r ../transformers/requirements.txt
   pip install vllm==0.5.4.post1
   ```

2. （可选）配置 Hugging Face 访问令牌，便于下载受限模型。

   ```bash
   export HF_TOKEN=hf_xxx
   ```

## 运行方式

交互式对话：

```bash
python dialogue_vllm.py
```

如需指定其他配置文件，可追加 `--config <路径>`。

单轮指令：

```bash
python dialogue_vllm.py --prompt "说明任务"
```

## 16 GB 显存显卡（RTX 5080 16G）建议

- `config.json` 预设 `dtype="float16"`、`max_model_len=6144`、`gpu_memory_utilization=0.90`，以适配 16 GB 显存。
- 若出现 CUDA OOM，可尝试：
  - 降低 `max_model_len`（如 4096）或 `sampling.max_tokens`；
  - 将 `gpu_memory_utilization` 下调至 `0.85`；
  - 若具备相应权重，可将 `"quantization"` 设置为 `"awq"`，使用轻量量化。
- 启动前请确保 GPU 无其他占用任务。

默认会话记录位于 `conversations/`，执行时带上 `--no-save` 参数即可关闭保存。

## Conda 环境自动切换

- `config.json` 中的 `runtime.conda_env` 用于指定目标 Conda 环境名；脚本在启动时会检测当前环境，若不一致会通过 `conda run -n <env>` 重新执行自身。
- 将该值改成你自己的虚拟环境名称（例如 `llama3`），确保该环境已安装全部依赖。
- 若未检测到 `conda` 命令，脚本会提示并继续在当前环境运行。
