# Task5：LLaVA 1.5 7B (4-bit) 多模态演示

该目录用于在本地 GPU (RTX 5080 16 GiB) 上部署 4bit/llava-v1.5-7b-5GB 量化模型。相比 0.5B 版本，该模型具备更好的视觉理解能力，同时通过 4-bit 量化使显存占用保持在约 7~8 GB。

## 1. 环境准备

1. 进入目录：
   `powershell
   cd Task5/llava-v1.5-7b-5GB
   `
2. 安装依赖（确保已安装支持 SM 120 的 PyTorch/CUDA 环境）：
   `powershell
   pip install -r requirements.txt
   `
   若尚未安装 nightly 版 PyTorch (CUDA 12.8)，可参考之前的说明：
   `powershell
   pip install --pre --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   `

## 2. 运行示例

执行批量推理脚本：
`powershell
python scripts/run_llava_inference.py --image .\data\demo_scene.jpg --prompt "Provide a concise caption for this image."
`
输出将保存在 esults/llava_v15_4bit_demo.json 中，包括图像描述以及两轮衍生问题的回答。

## 3. 交互式多轮对话

使用 PowerShell 脚本：
`powershell
.\run_llava_chat.ps1
`
- 默认会列出 data/ 目录中的图片；
- 可以使用 -Image 指定图片路径，-Save 保存对话记录；
- 若已手动激活 conda，可加 -SkipConda。

示例：
`powershell
.\run_llava_chat.ps1 -Image .\data\demo_scene.jpg -Save
`

## 4. 注意事项

- 4-bit 模型依赖 itsandbytes，仅支持 GPU 推理；如检测不到 CUDA，会直接退出。
- 初次运行会下载模型权重（约 5 GB），脚本自动把 Hugging Face 缓存重定向到 hf_cache/ 避免占满用户目录。
- 建议把 max_new_tokens 控制在 256 以内，以免显存峰值过高。
- 如需更换图片，直接向 data 目录添加文件或使用 --image/-Image 指定即可。

Enjoy the multimodal assistant!
