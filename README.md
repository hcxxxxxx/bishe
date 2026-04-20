# Fine-grained Emotion Prompt Controlled TTS (FGEmo-TTS)

本项目用于本科毕业论文实现：基于自然语言 prompt 的细粒度情感可控 TTS。

## 1. 核心设计
- `prompt_parser.py`: 把自然语言情感描述解析为结构化条件（emotion/intensity/arousal/valence/style）。
- `prompt_encoder.py`: 把结构化条件映射为条件向量。
- `prompt_control_model.py`: 使用 FiLM 风格 adaptor（gamma/beta）把条件注入 backbone 隐层。
- `train.py`: 支持 ablation：`none` / `rule_only` / `full`。
- `f5_adapter.py`: 真实 F5-TTS 的接入模板（你需要按你本地 F5 代码补 forward/infer）。

## 2. 目录
- `src/fgemo_tts/`: 主代码
- `scripts/build_manifest_from_esd_emoemilia.py`: ESD + Emo-Emilia 清单构建
- `scripts/run_ablation_8gpu.sh`: 8 卡 ablation 一键运行

## 3. 环境
```bash
cd /Users/hcx/Desktop/毕业论文/code2/bishe
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. 生成训练清单（ESD + Emo-Emilia）
```bash
mkdir -p data/manifests
python scripts/build_manifest_from_esd_emoemilia.py \
  --esd_root /path/to/esd \
  --emoemilia_root /path/to/emo-emilia \
  --text_table /path/to/utt_text_table.txt \
  --out_train data/manifests/train.jsonl \
  --out_val data/manifests/val.jsonl
```

`--text_table` 是可选，格式为 `utt_id|text`。

## 5. Debug 单卡训练
```bash
export PYTHONPATH=./src
bash scripts/run_single_debug.sh
```

## 6. 8x4090 ablation 训练
```bash
export PYTHONPATH=./src
bash scripts/run_ablation_8gpu.sh data/manifests/train.jsonl data/manifests/val.jsonl exp/ablation
```

## 7. 推理示例
```bash
export PYTHONPATH=./src
python -m fgemo_tts.infer.infer \
  --ckpt exp/ablation/full/last.pt \
  --text "今天我们完成了毕业论文的核心系统实现。" \
  --prompt "请用略带悲伤但温柔的语气说这句话" \
  --out_wav exp/demo/demo_prompt.wav
```

## 8. 与真实 F5/CosyVoice2/XTTS 对接建议
1. 先把你的真实 backbone 封装为 `TTSBackboneBase`。
2. 在 hidden state 注入：`h = h * (1 + gamma) + beta`。
3. 训练时先冻结 backbone，仅训练 prompt encoder + adaptor（建议 10k~20k steps）。
4. 再逐步解冻 backbone 高层做联合微调。

## 9. Ablation 建议写法（论文）
- `none`: 不使用情感 prompt（中性条件）。
- `rule_only`: 使用规则条件向量，不训练 prompt encoder。
- `full`: 规则解析 + 可训练 prompt encoder + adaptor。

可报告指标：MCD、CER/WER、情感分类准确率（外部 SER 模型）、主观 MOS/CMOS。
