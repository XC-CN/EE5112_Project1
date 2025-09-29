# Task 5 – Multimodal Demo with LLaVA (0.5B)

This folder contains everything required to reproduce the Task 5 group submission: we select a model from the official LLaVA model zoo and evaluate it on two downstream tasks (image captioning + multimodal dialogue).

## 1. Environment setup

1. Change to this directory:
       cd Task5
2. Install dependencies (PyTorch 2.4 with CUDA wheels plus Transformers tooling):
       pip install -r requirements.txt
3. (Optional) If you prefer to keep the run entirely on CPU, no additional configuration is needed—the script defaults to CPU execution.

## 2. Run the demo

       python scripts/run_llava_inference.py

The script loads `data/demo_scene.jpg`, queries `llava-hf/llava-interleave-qwen-0.5b-hf` for a single-image caption and a two-turn dialogue, then writes everything to `results/llava_v15_demo.json`.

## 3. Inspecting the outputs

       type results\llava_v15_demo.json

The JSON file stores the prompts and model responses in a structure that can be copied into the report.

## 4. Notes

- The run executes fully offline using the locally downloaded checkpoint.
- You can swap in any other image or adjust the prompts inside `scripts/run_llava_inference.py`.
- If you have a compatible GPU and want faster runs, set `DEVICE = "cuda"` in the script and ensure your PyTorch build supports the card.
