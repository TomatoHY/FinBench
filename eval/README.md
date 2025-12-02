# FinBench 评估系统使用说明

## 概述

FinBench评估系统基于7个核心评估指标，完全基于规则，不使用LLM-as-a-Judge，确保评估结果的客观性、可复现性和一致性。

## 评估指标

1. **TSR (Task Success Rate)**: 任务成功率 - 答案匹配 + 至少调用一个工具
2. **FAA (Final Answer Accuracy)**: 最终答案准确率 - 只评估最终答案的正确性
3. **CER (Calculation Error Rate)**: 计算推理错误率 - 工具调用成功但计算错误的比例
4. **AR (Abandonment Rate)**: 放弃率 - 未达到最大轮次就提前放弃的比例
5. **EEP (Execution Error Cost)**: 平均执行错误成本 - 工具调用阶段的平均失败次数
6. **FRR (Failure Resolution Rate)**: 失败解决率 - Agent的纠错能力
7. **LC (Latency Cost)**: 平均时间成本 - 解决每个任务所需的平均端到端时间

## 文件说明

### `eval.py`
统一的评估脚本，支持以下功能：
- **evaluate**: 运行模型评估（支持本地模型和API模型）
- **calculate-metrics**: 计算评估指标
- **compare**: 对比多个模型的评估结果
- **benchmark-comparison**: 对比FinBench与其他benchmark

---

## 一、本地部署模型评估

### 1.1 环境准备

```bash
# 安装依赖
pip install vllm
pip install numpy

# 确保有足够的GPU内存
nvidia-smi
```

### 1.2 单模型评估

```bash
python eval/eval.py evaluate \
    --model-type local \
    --model-path /path/to/model \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --max-tokens 2048 \
    --max-turns 10 \
    --dataset-file-path data/finbench_dataset_classified.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_model_name.jsonl
```

**参数说明**:
- `--model-type local`: 指定使用本地模型
- `--model-path`: 模型路径（本地路径或Hugging Face模型ID）
- `--tensor-parallel-size`: 张量并行大小（GPU数量，例如2表示使用2张GPU）
- `--gpu-memory-utilization`: GPU内存利用率（0.0-1.0，默认0.9）
- `--max-model-len`: 模型最大上下文长度（默认8192）
- `--max-tokens`: 每次生成的最大token数（默认2048）
- `--max-turns`: Agent循环的最大推理轮次（默认10）

### 1.3 批量评估（多模型并行）

使用 `eval.sh` 脚本进行批量评估：

```bash
# 编辑 eval.sh，配置模型列表
vim eval/eval.sh
```

**配置示例**:
```bash
MODELS=(
    "Qwen2.5-Coder-7B-Instruct:/mnt/data/kw/models/Qwen/Qwen2.5-Coder-7B-Instruct:2:0,1"
    "Qwen2.5-Coder-32B-Instruct:/mnt/data/kw/models/Qwen/Qwen2.5-Coder-32B-Instruct:2:2,3"
)
```

**格式说明**:
- 格式: `"模型名称:模型路径:张量并行大小:GPU设备ID"`
- GPU设备ID: `"0,1"` 表示使用GPU 0和1
- 每个模型会在独立的screen会话中运行

**运行批量评估**:
```bash
bash eval/eval.sh
```

**查看运行状态**:
```bash
# 查看所有screen会话
screen -list

# 进入某个会话查看实时输出
screen -r finbench_qwen2.5-coder-7b-instruct

# 查看日志
tail -f eval/log/qwen2.5-coder-7b-instruct.log
```

### 1.4 计算评估指标

```bash
python eval/eval.py calculate-metrics \
    --results-file eval/output/results_model_name.jsonl \
    --output-file eval/output/metrics_model_name.json \
    --max-turns 10 \
    --pretty
```

---

## 二、API模型评估

### 2.1 环境准备

```bash
# 安装API库（根据需要选择）
pip install openai          # OpenAI / Qwen
pip install anthropic          # Claude
pip install google-generativeai # Gemini
```

### 2.2 OpenAI / Qwen API

```bash
# 设置API密钥（可选，也可通过环境变量）
export OPENAI_API_KEY="your-api-key"
# 或
export QWEN_API_KEY="your-api-key"

# 运行评估
python eval/eval.py evaluate \
    --model-type api \
    --provider openai \
    --model-name gpt-4 \
    --api-key $OPENAI_API_KEY \
    --max-tokens 2048 \
    --max-turns 10 \
    --temperature 0.0 \
    --dataset-file-path data/finbench_dataset_classified.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_gpt4.jsonl

# Qwen API示例
python eval/eval.py evaluate \
    --model-type api \
    --provider qwen \
    --model-name qwen-max \
    --api-key $QWEN_API_KEY \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --max-tokens 2048 \
    --max-turns 10 \
    --temperature 0.0 \
    --dataset-file-path data/finbench_dataset_classified.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_qwen_max.jsonl
```

### 2.3 Anthropic Claude API

```bash
# 设置API密钥
export ANTHROPIC_API_KEY="your-api-key"

# 运行评估
python eval/eval.py evaluate \
    --model-type api \
    --provider anthropic \
    --model-name claude-3-opus-20240229 \
    --api-key $ANTHROPIC_API_KEY \
    --max-tokens 2048 \
    --max-turns 10 \
    --temperature 0.0 \
    --dataset-file-path data/finbench_dataset_classified.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_claude3_opus.jsonl
```

### 2.4 Google Gemini API

```bash
# 设置API密钥
export GOOGLE_API_KEY="your-api-key"

# 运行评估
python eval/eval.py evaluate \
    --model-type api \
    --provider google \
    --model-name gemini-pro \
    --api-key $GOOGLE_API_KEY \
    --max-tokens 2048 \
    --max-turns 10 \
    --temperature 0.0 \
    --dataset-file-path data/finbench_dataset_classified.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_gemini_pro.jsonl
```

### 2.5 API模型批量评估

可以创建脚本批量评估多个API模型：

```bash
#!/bin/bash

# OpenAI模型
python eval/eval.py evaluate \
    --model-type api \
    --provider openai \
    --model-name gpt-4 \
    --api-key $OPENAI_API_KEY \
    --max-tokens 2048 \
    --max-turns 10 \
    --dataset-file-path data/finbench_dataset_classified.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_gpt4.jsonl

# Claude模型
python eval/eval.py evaluate \
    --model-type api \
    --provider anthropic \
    --model-name claude-3-opus-20240229 \
    --api-key $ANTHROPIC_API_KEY \
    --max-tokens 2048 \
    --max-turns 10 \
    --dataset-file-path data/finbench_dataset_classified.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_claude3_opus.jsonl

# 计算指标
for model in gpt4 claude3_opus; do
    python eval/eval.py calculate-metrics \
        --results-file eval/output/results_${model}.jsonl \
        --output-file eval/output/metrics_${model}.json \
        --max-turns 10 \
        --pretty
done
```

---

## 三、结果分析

### 3.1 计算评估指标

```bash
python eval/eval.py calculate-metrics \
    --results-file eval/output/results_model_name.jsonl \
    --output-file eval/output/metrics_model_name.json \
    --max-turns 10 \
    --pretty
```

**输出格式**:
```json
{
  "summary": {
    "total_tasks": 624,
    "max_turns": 10
  },
  "overall_metrics": {
    "tsr": 0.65,
    "faa": 0.72,
    "memory_cheating_rate": 0.07,
    "cer": 0.15,
    "ar": 0.10,
    "avg_eep": 0.8,
    "frr": 0.60,
    "avg_lc": 12.5
  },
  "metrics_by_category": {
    "simple": {...},
    "moderate": {...},
    "complex": {...}
  },
  "metrics_by_tool_count": {...}
}
```

### 3.2 对比多个模型

```bash
python eval/eval.py compare \
    --metrics-files \
        eval/output/metrics_model1.json \
        eval/output/metrics_model2.json \
        eval/output/metrics_model3.json \
    --output-file eval/output/comparison.md \
    --format markdown
```

**输出格式**: Markdown表格，包含所有模型的指标对比

### 3.3 Benchmark对比

```bash
python eval/eval.py benchmark-comparison \
    --finbench-metrics eval/output/metrics_finbench.json \
    --other-benchmarks \
        finEval:other_benchmarks/finEval_metrics.json \
        ConvFinQA:other_benchmarks/ConvFinQA_metrics.json \
    --output-file eval/output/benchmark_comparison.json
```

---

## 四、完整评估流程示例

### 4.1 本地模型完整流程

```bash
# 步骤1: 运行评估
python eval/eval.py evaluate \
    --model-type local \
    --model-path /mnt/data/kw/models/Qwen/Qwen2.5-Coder-7B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --max-tokens 2048 \
    --max-turns 10 \
    --dataset-file-path data/finbench_dataset_classified.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_qwen2.5_coder_7b.jsonl

# 步骤2: 计算指标
python eval/eval.py calculate-metrics \
    --results-file eval/output/results_qwen2.5_coder_7b.jsonl \
    --output-file eval/output/metrics_qwen2.5_coder_7b.json \
    --max-turns 10 \
    --pretty

# 步骤3: 查看结果
cat eval/output/metrics_qwen2.5_coder_7b.json | python -m json.tool
```

### 4.2 API模型完整流程

```bash
# 步骤1: 运行评估
export OPENAI_API_KEY="your-api-key"

python eval/eval.py evaluate \
    --model-type api \
    --provider openai \
    --model-name gpt-4 \
    --api-key $OPENAI_API_KEY \
    --max-tokens 2048 \
    --max-turns 10 \
    --temperature 0.0 \
    --dataset-file-path data/finbench_dataset_classified.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_gpt4.jsonl

# 步骤2: 计算指标
python eval/eval.py calculate-metrics \
    --results-file eval/output/results_gpt4.jsonl \
    --output-file eval/output/metrics_gpt4.json \
    --max-turns 10 \
    --pretty
```

---

## 五、输出文件格式

### 5.1 任务结果文件（JSONL格式）

每行一个JSON对象，包含：
- `task_id`: 任务ID
- `question`: 问题文本
- `chain_category`: 任务复杂度分类（simple/moderate/complex）
- `tool_count`: 工具链长度
- `ground_truth`: 正确答案
- `final_answer`: 模型最终答案
- `is_correct`: 答案是否正确
- `tool_call_history`: 工具调用历史记录（包含success、latency等）
- `model_trajectory`: 完整对话轨迹
- `total_turns`: 总推理轮次
- `error_count`: 工具调用错误次数
- `latency`: 任务执行时间（秒）
- `abandoned`: 是否被提前放弃

### 5.2 评估指标文件（JSON格式）

包含：
- `summary`: 总体统计信息
- `overall_metrics`: 总体指标
- `metrics_by_category`: 按任务复杂度分类的指标
- `metrics_by_tool_count`: 按工具链长度分类的指标
- `detailed_breakdown`: 详细分类统计

---

## 六、批量评估脚本使用

### 6.1 配置 eval.sh

编辑 `eval/eval.sh`，配置要评估的模型：

```bash
MODELS=(
    "模型名称:模型路径:张量并行大小:GPU设备ID"
    "Qwen2.5-Coder-7B-Instruct:/mnt/data/kw/models/Qwen/Qwen2.5-Coder-7B-Instruct:2:0,1"
    "Qwen2.5-Coder-32B-Instruct:/mnt/data/kw/models/Qwen/Qwen2.5-Coder-32B-Instruct:2:2,3"
)
```

### 6.2 运行批量评估

```bash
bash eval/eval.sh
```

### 6.3 管理评估任务

```bash
# 查看所有screen会话
screen -list

# 进入某个会话
screen -r finbench_模型名

# 退出会话（不断开进程）
# 按 Ctrl+A 然后按 D

# 终止某个会话
screen -S finbench_模型名 -X quit

# 查看日志
tail -f eval/log/模型名.log
```

---

## 七、常见问题

### 7.1 本地模型相关问题

**Q: 模型加载失败**
- 检查模型路径是否正确
- 检查GPU内存是否足够
- 尝试降低 `--gpu-memory-utilization`

**Q: 张量并行错误**
- 确保 `--tensor-parallel-size` 与GPU数量匹配
- 检查GPU是否可用：`nvidia-smi`

**Q: 评估速度慢**
- 增加 `--tensor-parallel-size`（如果有多张GPU）
- 调整 `--max-tokens` 和 `--max-turns`

### 7.2 API模型相关问题

**Q: API调用失败**
- 检查API密钥是否正确
- 检查网络连接
- 查看API服务状态

**Q: API限流**
- 脚本已内置0.5秒延迟，如需调整可修改代码
- 考虑使用多个API密钥轮换

**Q: 答案格式错误**
- 检查模型是否正确理解 `\boxed{}` 格式要求
- 查看 `model_trajectory` 了解模型输出

### 7.3 指标计算问题

**Q: 答案匹配失败**
- 检查答案格式是否一致
- 查看 `is_answer_match` 函数的容差设置

**Q: 工具调用历史为空**
- 检查模型是否正确调用工具
- 查看 `tool_call_history` 字段格式

---

## 八、相关文档

- [评估指标说明](../data/评估指标说明.md)
- [实验设计](../实验设计.md)
- [质量控制说明](../质量控制说明.md)
- [工具分析报告](../FinBench_工具分析报告.md)

---

## 九、快速参考

### 本地模型评估
```bash
python eval/eval.py evaluate --model-type local --model-path <path> ...
```

### API模型评估
```bash
python eval/eval.py evaluate --model-type api --provider <provider> --model-name <name> ...
```

### 计算指标
```bash
python eval/eval.py calculate-metrics --results-file <file> --output-file <file>
```

### 对比模型
```bash
python eval/eval.py compare --metrics-files <file1> <file2> ...
```

### Benchmark对比
```bash
python eval/eval.py benchmark-comparison --finbench-metrics <file> --other-benchmarks <name:file> ...
```
