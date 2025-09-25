# Models Directory

This directory contains the downloaded LLM model files.

## Recommended Models for RTX 5080 16GB:

### Llama 3.1 8B (Recommended)
- **File**: `llama-3.1-8b.gguf`
- **Size**: ~5.6GB
- **Download**: Use the download script or manual download from Hugging Face
- **Performance**: Excellent balance of quality and speed

### Alternative Models:
- **Qwen2.5 14B**: ~8-10GB (higher quality, more VRAM)
- **Mistral 7B v0.3**: ~4-5GB (faster inference)
- **Llama 2 7B**: ~4GB (good baseline)

## Download Instructions:
1. Run `python download_model.py` to automatically download models
2. Or manually download GGUF format models from Hugging Face
3. Place the `.gguf` files in this directory

## Model Format:
- Use GGUF format for best performance with llama-cpp-python
- Q4_K_M quantization recommended for RTX 5080 16GB
