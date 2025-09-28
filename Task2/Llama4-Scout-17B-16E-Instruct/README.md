# Llama-4-Scout-17B-16E-Instruct Deployment

This directory contains the deployment for the Llama-4-Scout-17B-16E-Instruct model, optimized for RTX 5080 16GB GPU.

## Model Information

- **Model**: Llama-4-Scout-17B-16E-Instruct
- **Quantization**: Q4_K_S (optimized for 16GB GPU memory)
- **Source**: [unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF)
- **Specialization**: Robotics and Human-Robot Interaction
- **Context Size**: 8192 tokens
- **GPU Optimization**: RTX 5080 16GB with CUDA acceleration

## Directory Structure

```
Llama4-Scout-17B-16E-Instruct/
├── config.json              # Model and hardware configuration
├── dialogue_system.py       # High-level dialogue system
├── llm_platform.py         # Low-level LLM platform wrapper
├── requirements.txt         # Python dependencies
├── run_dialogue.bat        # Windows batch script
├── run_dialogue.ps1        # PowerShell script
├── run_dialogue.sh         # Unix/Linux shell script
├── conversations/          # Saved conversation logs
├── models/                 # Model files directory
└── README.md              # This file
```

## Prerequisites

### 1. Hardware Requirements
- **GPU**: RTX 5080 16GB (or compatible CUDA GPU with 12GB+ VRAM)
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 10GB+ free space for model files

### 2. Software Requirements
- **Python**: 3.9+ (conda environment recommended)
- **CUDA**: 11.8+ or 12.x
- **conda**: Miniconda or Anaconda

### 3. Conda Environment
Ensure you have the `5112Project` conda environment activated:
```bash
conda activate 5112Project
```

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model Files
The model files need to be placed in the `models/` directory. The expected model file is:
- `Llama-4-Scout-17B-16E-Instruct-Q4_K_S.gguf`

You can download it manually from Hugging Face or use the download script:
```python
from huggingface_hub import hf_hub_download

# Download Q4_K_S quantization (recommended for RTX 5080 16GB)
hf_hub_download(
    repo_id="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    filename="Q4_K_S/Llama-4-Scout-17B-16E-Instruct-Q4_K_S-00001-of-00002.gguf",
    local_dir="models"
)
hf_hub_download(
    repo_id="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF", 
    filename="Q4_K_S/Llama-4-Scout-17B-16E-Instruct-Q4_K_S-00002-of-00002.gguf",
    local_dir="models"
)
```

## Usage

### Quick Start
```bash
# Windows
run_dialogue.bat

# PowerShell
./run_dialogue.ps1

# Unix/Linux/macOS
./run_dialogue.sh
```

### Manual Execution
```bash
conda activate 5112Project
python dialogue_system.py
```

### Command Line Options
```bash
python dialogue_system.py --config custom_config.json  # Use custom config
python dialogue_system.py --no-stream                  # Disable streaming
```

## Configuration

The `config.json` file contains all model and hardware configurations:

- **Model Config**: Context size, threads, batch size, RoPE frequency
- **Generation Config**: Temperature, top-p, top-k, repeat penalty
- **Dialogue Config**: System prompt, history length, streaming
- **Hardware Config**: GPU settings, CUDA optimization flags

### RTX 5080 Optimizations
- **GPU Layers**: -1 (all layers on GPU)
- **Flash Attention**: Enabled for faster inference
- **Memory Lock**: Enabled to prevent swapping
- **Memory Mapping**: Enabled for efficient model loading
- **Context Size**: 8192 tokens (optimized for available VRAM)

## Interactive Commands

Once the dialogue system is running, you can use these commands:
- `exit`: Quit the system
- `clear`: Clear conversation history
- `stats`: Show model and conversation statistics

## Troubleshooting

### Common Issues

1. **CUDA Not Found**
   - Ensure CUDA 11.8+ is installed
   - Verify `nvidia-smi` command works
   - Reinstall `llama-cpp-python[cuda]`

2. **Out of Memory**
   - Reduce `n_ctx` in config.json
   - Lower `n_batch` size
   - Consider using Q3_K_S quantization instead

3. **Model Not Loading**
   - Check model file path in config.json
   - Ensure model files are in the `models/` directory
   - Verify file permissions

4. **Slow Performance**
   - Check GPU utilization with `nvidia-smi`
   - Ensure all model layers are on GPU (`n_gpu_layers: -1`)
   - Verify CUDA acceleration is enabled

### Performance Tips

- **Optimal Batch Size**: 512 (already configured)
- **Thread Count**: 12 threads (adjust based on CPU cores)
- **Context Management**: Automatic trimming at 12 message pairs
- **Streaming**: Enabled for real-time response display

## Model Capabilities

Llama-4-Scout is specifically trained and optimized for:
- **Robotics Engineering**: Control systems, kinematics, dynamics
- **Human-Robot Interaction**: HRI principles, social robotics, UX design
- **Sensor Integration**: Computer vision, LIDAR, tactile sensing
- **AI/ML Applications**: Reinforcement learning, neural networks, planning
- **Technical Documentation**: Code generation, system design, troubleshooting

## File Descriptions

- **`config.json`**: Complete system configuration
- **`dialogue_system.py`**: Main interactive dialogue interface
- **`llm_platform.py`**: Low-level model wrapper with GPU optimizations
- **`run_dialogue.*`**: Launch scripts for different platforms
- **`conversations/`**: Automatic conversation logging (JSON format)
- **`models/`**: Model files storage directory

## Conversation Logging

All conversations are automatically saved in the `conversations/` directory with timestamps:
- Format: `YYYYMMDD_HHMMSS.json`
- Includes: Model info, GPU specs, message history, timestamps
- Privacy: Conversations are stored locally only

## License

This deployment follows the same license as the original Llama-4-Scout model. Please refer to the model's Hugging Face page for specific licensing terms.

## Support

For technical issues:
1. Check GPU compatibility and CUDA installation
2. Verify conda environment setup
3. Review model file integrity
4. Monitor system resources during inference

---

**EE5112 Human Robot Interaction - Task 2**  
**Llama-4-Scout Deployment for RTX 5080 16GB**