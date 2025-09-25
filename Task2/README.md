# 任务2：LLM平台开发

## 项目概述

本项目使用 `llama-cpp-python`和Llama 3.2 3B模型实现本地大型语言模型平台，针对RTX 5080 16GB GPU配置进行优化。

## 功能特性

- **本地LLM平台**：本地运行LLM模型，无需网络连接
- **多轮对话**：支持对话式交互
- **GPU加速**：针对RTX 5080 16GB优化，支持CUDA
- **简单配置**：简单的设置和配置选项
- **对话历史**：维护对话上下文
- **流式输出**：支持逐token输出，提升用户体验

## 系统要求

### 硬件要求

- **GPU**: NVIDIA RTX 5080 16GB（推荐）
- **内存**: 16GB以上系统内存
- **存储**: 10GB以上可用空间

### 软件要求

- **Python**: 3.8以上
- **CUDA**: 11.8以上（用于GPU加速）
- **操作系统**: Windows 10/11、macOS或Linux

## 安装步骤

### 1. 克隆和设置

```bash
# 进入Task2目录
cd Task2

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型

模型已预下载：`models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`

### 3. 运行平台

#### Windows (推荐)
```batch
# 使用批处理脚本自动激活conda环境并运行（推荐）
run_dialogue.bat

# 或使用PowerShell脚本
.\run_dialogue.ps1
```

#### Linux/macOS
```bash
# 使用shell脚本
./run_dialogue.sh
```

#### 手动运行
```bash
# 激活conda环境
conda activate 5112Project

# 启动对话系统
python dialogue_system.py

# 或启动基础平台
python llm_platform.py
```

## 项目结构

```
Task2/
├── README.md                 # 本文件
├── requirements.txt          # Python依赖
├── run_dialogue.bat         # Windows批处理脚本（自动激活环境）
├── run_dialogue.ps1         # PowerShell脚本（自动激活环境）
├── run_dialogue.sh          # Shell脚本（Linux/macOS）
├── llm_platform.py          # 核心LLM平台
├── dialogue_system.py       # 对话系统实现
├── test_deployment.py       # 部署测试脚本
├── config.json              # 配置文件
├── models/                  # 模型文件目录
│   ├── README.md           # 模型文档
│   └── *.gguf             # 模型文件（已下载）
└── conversations/          # 保存的对话
    └── *.json             # 对话历史文件
```

## 使用方法

### 交互式对话

```bash
python dialogue_system.py

# 命令:
# - 输入消息并按回车
# - 输入'exit'退出
# - 输入'clear'清除历史
# - 输入'stats'查看统计信息
```

## 配置

### 模型配置

编辑 `config.json`来自定义模型参数：

```json
{
    "model_config": {
        "model_path": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "n_gpu_layers": 35,
        "n_ctx": 4096,
        "n_threads": 8,
        "verbose": false
    },
    "generation_config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "max_tokens": 512
    },
    "dialogue_config": {
        "max_history": 10,
        "system_prompt": "You are a helpful AI assistant. Provide clear, concise, and helpful responses.",
        "save_conversations": true,
        "conversation_dir": "conversations",
        "streaming": true
    },
    "hardware_config": {
        "gpu_enabled": true,
        "max_gpu_layers": 35,
        "memory_fraction": 0.9
    }
}
```

### 支持的模型

| 模型            | 大小    | 显存使用 | 质量       |
| --------------- | ------- | -------- | ---------- |
| Llama 3.2 3B    | ~2GB    | ~3GB     | ⭐⭐⭐⭐⭐ |
| Llama 3.1 8B    | ~5.6GB  | ~6GB     | ⭐⭐⭐⭐⭐ |
| Qwen2.5 14B     | ~8-10GB | ~10GB    | ⭐⭐⭐⭐⭐ |
| Mistral 7B v0.3 | ~4-5GB  | ~5GB     | ⭐⭐⭐⭐   |
| Llama 2 7B      | ~4GB    | ~4GB     | ⭐⭐⭐⭐   |

## 性能优化

### 针对RTX 5080 16GB

- 使用Q4_K_M量化获得最佳平衡
- 设置 `n_gpu_layers=35`最大化GPU利用率
- 根据内存可用性调整 `n_ctx`

### 常见问题

1. **内存不足**

   - 减少 `n_ctx`值
   - 使用更小的模型
   - 启用量化
2. **推理缓慢**

   - 增加 `n_gpu_layers`
   - 检查CUDA安装
   - 使用更小的上下文窗口
3. **流式输出不工作**

   - 确保 `config.json`中 `streaming: true`
   - 检查Python版本兼容性

## 开发

### 添加新模型

1. 下载GGUF格式模型
2. 放置在 `models/`目录
3. 在代码中更新模型路径

### 自定义行为

- 修改 `LLMPlatform`类实现自定义功能
- 在 `_get_default_config()`中调整生成参数
- 实现自定义对话格式

## 贡献

本项目是EE5112人机交互课程的一部分。贡献指南：

1. 遵循课程指南
2. 记录您的更改
3. 在RTX 5080 16GB配置上测试
4. 提交带有清晰描述的拉取请求

## 许可证

本项目用于教育目的，作为NUS EE5112课程的一部分。

## 联系方式

- **课程**: EE5112人机交互
- **机构**: 新加坡国立大学
- **小组**: 第7组

## 致谢

- **Meta AI**提供Llama模型
- **llama-cpp-python**社区
- **Hugging Face**提供模型托管
- **EE5112教学团队**

---

**注意**: 本项目为EE5112人机交互课程的一部分，仅用于教育目的。
