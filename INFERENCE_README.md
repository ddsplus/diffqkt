# DiffuQKT 推理脚本使用说明

## 脚本概述

`inference.py` 是一个用于 DiffuQKT 模型的推理脚本，支持以下功能：
- 加载训练好的模型
- 在测试数据上进行推理
- 添加可控的噪声（随机翻转答案序列的百分比）
- 计算并输出 **AUC** 和 **ACC** 指标

## 基本用法

```bash
# 默认运行：输出噪声强度 0, 0.1, 0.2, 0.3, 0.4, 0.5 的结果
python inference.py --model_path <模型文件路径> --dataset <数据集名称>

# 自定义噪声强度
python inference.py --model_path <模型文件路径> --dataset <数据集名称> --noise_ratios 0.0 0.1 0.2
```

## 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model_path` | str | ✓ | - | 模型权重文件路径（如：`./result/DiffuQKT_assist09_0_20250511_best_model.pkl`） |
| `--dataset` | str | ✓ | - | 数据集名称，可选：`assist09`、`assist17`、`xes3g5m`、`statics2011` |
| `--noise_ratios` | float | ✗ | [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] | 噪声强度列表，范围 0-1（默认输出 6 个结果） |
| `--batch_size` | int | ✗ | 512 | 推理时的批量大小 |
| `--max_seq` | int | ✗ | 200 | 最大序列长度 |
| `--min_seq` | int | ✗ | 3 | 最小序列长度 |
| `--device` | str | ✗ | cuda | 计算设备，可选：`cuda`（GPU）或 `cpu` |

## 使用示例

### 例1：默认运行（输出 6 个噪声强度的结果）
```bash
python inference.py \
    --model_path ./result/DiffuQKT_assist09_0_20250511_best_model.pkl \
    --dataset assist09
```

### 例2：自定义噪声强度列表
```bash
python inference.py \
    --model_path ./result/DiffuQKT_assist09_0_20250511_best_model.pkl \
    --dataset assist09 \
    --noise_ratios 0.0 0.05 0.1 0.15 0.2
```

### 例3：只测试单个噪声强度
```bash
python inference.py \
    --model_path ./result/DiffuQKT_assist17_0_20250511_best_model.pkl \
    --dataset assist17 \
    --noise_ratios 0.3
```

### 例4：在不同数据集上测试，使用 CPU
```bash
python inference.py \
    --model_path ./result/DiffuQKT_xes3g5m_0_20250511_best_model.pkl \
    --dataset xes3g5m \
    --device cpu
```

### 例5：自定义批量大小和序列长度
```bash
python inference.py \
    --model_path ./result/DiffuQKT_statics2011_0_20250511_best_model.pkl \
    --dataset statics2011 \
    --batch_size 256 \
    --max_seq 150
```

## 输出格式

脚本执行完成后会输出表格格式的结果：

```
============================================================
推理完成！数据集: assist09
============================================================
噪声强度      AUC        ACC       
------------------------------------------------------------
0.0%         0.7623     0.6842    
10.0%        0.7512     0.6721    
20.0%        0.7401     0.6598    
30.0%        0.7289     0.6475    
40.0%        0.7178     0.6352    
50.0%        0.7066     0.6229    
============================================================
```

- **噪声强度**：应用的噪声比例
- **AUC**：曲线下面积（范围 0-1，越高越好）
- **ACC**：准确率（范围 0-1，越高越好）

## 噪声机制说明

**噪声强度**: 在推理过程中，脚本会随机选择整个答案序列中百分比的位置，并直接翻转其答案值（0 ↔ 1），不区分答案是否为0，模拟学生的错误或不确定性。

- `noise_ratio = 0.0`：无噪声，使用原始答案
- `noise_ratio = 0.2`：整个序列的 20% 位置会被随机翻转
- `noise_ratio = 1.0`：整个序列的所有答案都会被翻转

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

### Q4: 如何只测试特定的噪声强度？
**A:** 使用 `--noise_ratios` 参数指定：
```bash
# 只测试 0% 和 50% 噪声
python inference.py --model_path ... --dataset ... --noise_ratios 0.0 0.5
```

### Q5: 脚本运行时间很长
**A:** 这是正常的，因为脚本需要在多个噪声强度下进行推理。可以：
- 减少噪声强度的数量
- 增加 `batch_size`（如果 GPU 内存允许）
- 减少 `max_seq` 长度

## 输出的指标解释

- **AUC (Area Under Curve)**：ROC 曲线下的面积，范围 0-1，越高越好，衡量模型的整体分类性能
- **ACC (Accuracy)**：准确率，预测正确的样本占比，范围 0-1，越高越好

## 性能分析

通过观察不同噪声强度下的 AUC 和 ACC，可以评估模型的**鲁棒性**：
- 如果结果随噪声增加而缓慢下降，说明模型较为鲁棒
- 如果结果快速下降，说明模型对噪声敏感
- 比较不同模型在相同噪声强度下的性能，可以找到更好的模型
