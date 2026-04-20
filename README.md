# FGEmo-TTS (CosyVoice2 Real Pipeline)

本仓库已切换为真实 CosyVoice2 训练/推理流程（不再使用 mock 声学目标或 mock 推理）。

固定目录关系：
- 当前仓库：`bishe`
- CosyVoice2 代码：`../CosyVoice`
- ESD 数据：`../dataset_esd_sorted`

## 1) 关键入口
- 真实训练编排：[src/fgemo_tts/train/train.py](src/fgemo_tts/train/train.py)
- 真实推理入口：[src/fgemo_tts/infer/infer.py](src/fgemo_tts/infer/infer.py)
- CosyVoice 适配器：[src/fgemo_tts/models/cosyvoice_adapter.py](src/fgemo_tts/models/cosyvoice_adapter.py)
- ESD -> CosyVoice 数据准备：[scripts/prepare_cosyvoice_esd_data.py](scripts/prepare_cosyvoice_esd_data.py)
- 一键数据准备：[scripts/run_prepare_esd.sh](scripts/run_prepare_esd.sh)
- 8卡 ablation：[scripts/run_ablation_8gpu.sh](scripts/run_ablation_8gpu.sh)
- 组装可推理模型目录：[scripts/assemble_cosyvoice_model.py](scripts/assemble_cosyvoice_model.py)

## 2) 环境
```bash
cd /Users/hcx/Desktop/毕业论文/code2/bishe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r ../CosyVoice/requirements.txt
```

## 3) 数据准备（ESD）
```bash
bash scripts/run_prepare_esd.sh ../dataset_esd_sorted ../CosyVoice data/cosyvoice_esd
```

会自动生成三组 ablation 数据：
- `data/cosyvoice_esd/none/...`
- `data/cosyvoice_esd/rule_only/...`
- `data/cosyvoice_esd/full/...`

每组都包含 `train/dev` 的 kaldi 风格文件 + parquet `data.list`。

## 4) 8x4090 训练 ablation（真实 CosyVoice）
```bash
bash scripts/run_ablation_8gpu.sh \
  ../CosyVoice \
  ../CosyVoice/pretrained_models/CosyVoice2-0.5B \
  data/cosyvoice_esd \
  exp/cosyvoice_esd \
  llm,flow
```

## 5) 组装可推理模型目录
训练后把最新 `llm/flow` checkpoint 覆盖到一个新模型目录：
```bash
python3 scripts/assemble_cosyvoice_model.py \
  --base_model_dir ../CosyVoice/pretrained_models/CosyVoice2-0.5B \
  --exp_root exp/cosyvoice_esd/full \
  --output_model_dir exp/cosyvoice_esd/full_infer_model
```

## 6) 推理（自然语言情感 prompt）
```bash
export PYTHONPATH=./src
python3 -m fgemo_tts.infer.infer \
  --cosyvoice_root ../CosyVoice \
  --model_dir exp/cosyvoice_esd/full_infer_model \
  --text "今天我们完成了毕业论文系统的核心实验。" \
  --prompt "请用略带悲伤但温柔的语气说这句话" \
  --speaker_wav ../dataset_esd_sorted/neutral/0001/0001_000001.wav \
  --mode instruct2 \
  --out_wav exp/demo/demo_full.wav
```

## 7) Ablation 定义
- `none`：训练文本不加情感控制提示。
- `rule_only`：训练文本用规则模板提示（情感类别固定映射）。
- `full`：训练文本用更自然的提示模板（强度 + 风格）。

这三组只改 prompt 处理，不改 CosyVoice 主干，便于论文做可控变量实验。
