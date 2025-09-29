# Task5：LLaVA 多模态任务演示

本目录复现了课程 Task 5 的小组实验流程：从 LLaVA 模型动物园中选择模型，在图像描述与图文对话两个任务上给出实际输出。本机环境已升级到支持 RTX 5080 (SM 120) 的 PyTorch Nightly（CUDA 12.8），脚本默认在 GPU 上运行。

## 1. 环境准备

1. 进入目录：
   cd Task5
2. 安装（或更新）依赖：
   pip install --pre --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   pip install -r requirements.txt
   > 若前一步已完成，可直接执行第二条命令。`requirements.txt` 中的 PyTorch 依赖默认走 CPU；若已装好 Nightly，可忽略警告。
   >

## 2. 运行脚本

    python scripts/run_llava_inference.py

脚本会：

- 加载 `data/demo_scene.jpg`；
- 调用 `llava-hf/llava-interleave-qwen-0.5b-hf` 在 GPU 上生成一条图像描述；
- 与同一张图片进行两轮问答；
- 将结果写入 `results/llava_v15_demo.json`。

## 3. 查看结果

    type results\llava_v15_demo.json

JSON 中记录了提示词、模型回复以及运行设备（`device: "cuda"`）。

## 4. 注意事项

- 目前必须使用支持 SM 120 的 PyTorch Nightly (Cu128)；否则会出现 “no kernel image” 报错。
- 若想在 CPU 上运行，可将 `scripts/run_llava_inference.py` 中的 `DEVICE = "cuda"` 改回 `"cpu"`。
- 可替换图片或提示词以扩展演示，但注意重跑脚本以更新 JSON。
