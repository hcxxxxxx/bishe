# 基于自然语言提示的细粒度情感可控文本到语音合成系统（CosyVoice2-0.5B, Inference-only）

本项目面向本科毕业论文实验，目标是：
- 使用 **CosyVoice2-0.5B**（`FunAudioLLM/CosyVoice2-0.5B`）作为 backbone。
- 不做大规模训练，仅进行 inference。
- 通过自然语言 prompt 实现细粒度情感控制：
  - 情感类型控制（happy/sad/angry/...）
  - 强度控制（`slightly` / `moderately` / `very`）
  - 复合情感（主情感 + 次情感）
  - 上下文感知（context-aware）
- 支持 baseline prompt 与优化 prompt 的 A/B 对比实验。
- 支持中英文双语情感控制。

---

## 1. 项目结构

```text
.
├── README.md
├── requirements.txt
├── model_loader.py            # 下载并加载 CosyVoice2-0.5B，兼容多版本 API
├── prompt_engineering.py      # 多层提示词工程：情感模板/强度/复合情感/优化
├── inference_pipeline.py      # 主推理流程（单句+批量），保存音频与元数据
├── evaluation.py              # 实验评估：prompt 对比 + emotion2vec/SenseVoice + MOS 模板
├── demo.py                    # Gradio 交互演示界面
├── data
│   ├── batch_samples.jsonl    # 批量推理示例输入
│   └── eval_samples.csv       # 评估示例输入
├── models/                    # 本地模型目录（自动下载）
├── outputs/                   # 推理输出
├── demo_outputs/              # demo 输出
└── experiments/               # 评估输出
```

---

## 2. 环境配置

### 2.1 Python 与 CUDA
建议：
- Python 3.10/3.11
- CUDA 12.x（与你远程服务器驱动匹配）

### 2.2 创建环境

```bash
conda create -n emo-tts python=3.10 -y
conda activate emo-tts
```

### 2.3 安装 PyTorch（按你的 CUDA 版本选择）

如果你的服务器是 CUDA 12.1：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你的服务器是 CUDA 11.8：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.4 安装其余依赖

```bash
pip install -r requirements.txt
```

### 2.5 安装官方 CosyVoice 运行时（重要）

仅安装 `pip install cosyvoice` 在部分环境会缺少 `cosyvoice.cli`，建议按官方仓库方式安装：

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
pip install -r requirements.txt
cd ..
export COSYVOICE_REPO=$(realpath ./CosyVoice)
```

### 2.6（可选）Hugging Face 镜像或代理
在网络受限环境可设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 3. 快速开始

### 3.1 首次加载模型（自动从 Hugging Face 下载）

```bash
python model_loader.py
```

默认下载到：`./models/CosyVoice2-0.5B`

### 3.2 单句推理（优化 prompt）

```bash
python inference_pipeline.py \
  --text "收到录取通知后，我真的很开心，但也有点紧张。" \
  --emotion happy \
  --intensity very \
  --secondary_emotion fearful \
  --context "和父母分享未来规划" \
  --language zh \
  --prompt_mode optimized
```

### 3.3 单句推理（baseline prompt）

```bash
python inference_pipeline.py \
  --text "收到录取通知后，我真的很开心，但也有点紧张。" \
  --emotion happy \
  --intensity very \
  --secondary_emotion fearful \
  --context "和父母分享未来规划" \
  --language zh \
  --prompt_mode baseline
```

### 3.4 批量推理

```bash
python inference_pipeline.py \
  --batch_file ./data/batch_samples.jsonl \
  --prompt_mode optimized
```

输出：
- 音频：`./outputs/audio/*.wav`
- 元数据：`./outputs/metadata.jsonl`

元数据包含：
- 输入文本
- 情感参数
- 使用的自然语言 prompt
- 生成音频路径
- 模型来源

---

## 4. 提示词工程设计说明

`prompt_engineering.py` 中实现了四层机制：
1. 基础情感模板（中英双语）
2. 强度修饰（`slightly/moderately/very`）
3. 复合情感描述（主情感 + 次情感）
4. Prompt 优化函数（清晰度、自然停连、防夸张约束）

并提供 `build_prompt_pair()` 生成：
- baseline prompt
- optimized prompt
用于论文中的 A/B 对比实验。

注意（CosyVoice2 重要）：
- 对 `inference_instruct2` 而言，`instruct_text` 更接近“风格提示文本通道”，不要把目标 `text` 拼进 `instruct_text`。
- 推荐将 `instruct_text` 写成简短风格描述（如“语气高兴、略紧张”），目标内容仅放在 `tts_text`。

---

## 5. 评估流程

### 5.1 运行自动评估脚本

```bash
python evaluation.py \
  --eval_file ./data/eval_samples.csv \
  --output_dir ./experiments
```

### 5.2 评估输出

- `./experiments/prompt_comparison_results.csv`
  - baseline 与 optimized 的音频路径
  - 对应 prompt
  - 可选 emotion2vec 相似度指标
  - 可选 SenseVoice 情感标签解析结果

- `./experiments/mos_template.csv`
  - 主观评分模板（自然度、情感准确度、强度匹配）

### 5.3 关于 emotion2vec / SenseVoice

`evaluation.py` 采用“best-effort”加载策略：
- 若本地可成功加载 `funasr` 的 emotion2vec/SenseVoice，则自动计算相应指标。
- 若模型不可用，脚本不会崩溃，对应指标记为空，仍可产出 A/B 对比与 MOS 模板。

---

## 6. 可视化 Demo（Gradio）

```bash
python demo.py
```

默认地址：
- 本机：[http://127.0.0.1:7860](http://127.0.0.1:7860)
- 局域网：`http://<server-ip>:7860`

可交互控制：
- 文本
- 语言（中/英）
- 主情感
- 强度
- 次情感
- 上下文
- prompt 版本（baseline / optimized）

---

## 7. 在 8x4090 远程服务器运行建议

1. 先在服务器执行环境安装与依赖安装。
2. 首次运行会下载模型，建议提前拉取，避免实验中断。
3. 推理任务多时，优先使用批量脚本并做好输出目录分组（按实验编号）。
4. 论文实验建议固定：
   - 随机种子（如后续版本添加）
   - 文本集合
   - prompt 模板版本
   以保证可复现性。

---

## 8. 常见问题

### Q1: cosyvoice API 版本变化导致报错怎么办？
`model_loader.py` 已实现多路径兼容加载与推理方法尝试（`inference_instruct2` / `inference_instruct` / `inference`）。
若仍失败，请根据你服务器上的 cosyvoice 版本修改 `model_loader.py` 中方法优先级。

若报错为 `No module named 'cosyvoice.cli'`，通常是运行时安装不完整。优先按“2.5 安装官方 CosyVoice 运行时”修复，并设置 `COSYVOICE_REPO` 环境变量。

### Q2: 为什么建议做 baseline vs optimized 对比？
这是论文中验证提示词工程有效性的关键实验，能直观体现“细粒度控制”带来的收益。

---

## 9. 可用于论文的方法描述（简版）

本系统采用基于自然语言指令的推理式情感控制方案，不对 TTS 主干进行参数更新。通过构建多层次 prompt 模板（情感类别、强度修饰、复合情感、上下文约束）引导 CosyVoice2-0.5B 生成目标风格语音，并通过客观指标（emotion2vec/SenseVoice）与主观 MOS 评分联合评估情感控制效果。
