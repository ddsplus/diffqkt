# DiffuQKT 推理脚本使用说明

## 脚本概述

`inference.py` 是一个用于 DiffuQKT 模型的推理脚本，支持以下功能：
- 加载训练好的模型
- 在测试数据上进行推理
- 添加可控的噪声（随机翻转答案序列的百分比）
- 计算并输出 **AUC** 和 **ACC** 指标

## 基本用法

```bash
python inference.py --model_path <模型文件路径> --dataset <数据集名称> --noise_ratio <噪声强度>
```

## 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model_path` | str | ✓ | - | 模型权重文件路径（如：`./result/DiffuQKT_assist09_0_20250511_best_model.pkl`） |
| `--dataset` | str | ✓ | - | 数据集名称，可选：`assist09`、`assist17`、`xes3g5m`、`statics2011` |
| `--noise_ratio` | float | ✗ | 0.0 | 噪声强度，范围 0-1，表示随机翻转答案的百分比 |
| `--batch_size` | int | ✗ | 512 | 推理时的批量大小 |
| `--max_seq` | int | ✗ | 200 | 最大序列长度 |
| `--min_seq` | int | ✗ | 3 | 最小序列长度 |
| `--device` | str | ✗ | cuda | 计算设备，可选：`cuda`（GPU）或 `cpu` |

## 使用示例

### 例1：无噪声推理（基准）
```bash
python inference.py \
    --model_path ./result/DiffuQKT_assist09_0_20250511_best_model.pkl \
    --dataset assist09 \
    --noise_ratio 0.0
```

### 例2：添加 20% 的噪声
```bash
python inference.py \
    --model_path ./result/DiffuQKT_assist09_0_20250511_best_model.pkl \
    --dataset assist09 \
    --noise_ratio 0.2
```

### 例3：在不同数据集上测试，添加 30% 噪声，使用 CPU
```bash
python inference.py \
    --model_path ./result/DiffuQKT_assist17_0_20250511_best_model.pkl \
    --dataset assist17 \
    --noise_ratio 0.3 \
    --device cpu
```

### 例4：自定义批量大小和序列长度
```bash
python inference.py \
    --model_path ./result/DiffuQKT_xes3g5m_0_20250511_best_model.pkl \
    --dataset xes3g5m \
    --noise_ratio 0.1 \
    --batch_size 256 \
    --max_seq 150
```

## 输出格式

脚本执行完成后会输出：

```
==================================================
推理完成！
数据集: assist09
噪声强度: 0.20%
AUC: 0.7623
ACC: 0.6842
==================================================
```

## 噪声机制说明

**噪声比例**: 在推理过程中，脚本会随机选择答案序列中的百分比个样本，并翻转其答案值（0 → 1，1 → 0），模拟学生的错误或不确定性。

- `noise_ratio = 0.0`：无噪声，使用原始答案
- `noise_ratio = 0.2`：20% 的答案会被随机翻转
- `noise_ratio = 1.0`：100% 的答案都会被翻转

## 关键特性

1. **灵活的数据集支持**：支持 ASSIST09、ASSIST17、XES3G5M、STATICS2011 四个数据集
2. **可控的噪声注入**：可以通过调整 `noise_ratio` 来测试模型的鲁棒性
3. **自动设备选择**：如果 GPU 不可用，自动降级到 CPU
4. **完整的指标计算**：计算 AUC（曲线下面积）和 ACC（准确率）

## 常见问题

### Q1: 运行时出现 "模块未找到" 错误
**A:** 确保你在项目的 `diffqkt` 目录下运行脚本，并且已经安装了所有依赖项：
```bash
cd e:\desk_file\DIffqkt\diffqkt
pip install torch scikit-learn tqdm pandas numpy
```

### Q2: GPU 内存不足
**A:** 减小 `batch_size` 或使用 CPU：
```bash
python inference.py --model_path ... --dataset ... --batch_size 256 --device cpu
```

### Q3: 模型文件不存在
**A:** 检查模型路径是否正确，可以使用绝对路径或相对路径。

## 输出的指标解释

- **AUC (Area Under Curve)**：ROC 曲线下的面积，范围 0-1，越高越好
- **ACC (Accuracy)**：准确率，预测正确的样本占比，范围 0-1，越高越好
