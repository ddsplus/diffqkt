# DiffuQKT 训练指南

## 项目概述

这是DiffuQKT（基于扩散的知识追踪）模型的训练框架。支持ASSIST2009和ASSIST2017两个教育数据集。

## 数据集

### ASSIST2009
- 原始数据: `ASSIST2009/skill_builder_data.csv`
- 数据规模: 346,860个交互记录，4,217名学生
- 技能数: 149个
- 问题数: 17,751个

### ASSIST2017
- 原始数据: `ASSIST2017/anonymized_full_release_competition_dataset.csv`
- 数据规模: 942,816个交互记录
- 数据分割: 80% 训练集，20% 测试集（按学生分割）

## 安装依赖

```bash
pip install torch pandas numpy scikit-learn tqdm
```

## 使用步骤

### 1. 数据预处理

在执行训练前，需要预处理原始数据。

#### 预处理 ASSIST2009

```bash
cd /mnt/sda/hds/2026_spring/DiffuQKT-main
python preprocess_assist09.py
```

生成文件位置: `../data/ASSIST09/`

#### 预处理 ASSIST2017

```bash
cd /mnt/sda/hds/2026_spring/DiffuQKT-main
python preprocess_assist17.py
```

生成文件位置: `../data/ASSIST17/`

### 2. 模型训练

```bash
cd /mnt/sda/hds/2026_spring/DiffuQKT-main
python main.py
```

该脚本会自动训练所有配置的数据集（assist09 和 assist17）。

## 训练配置

在 `main.py` 中可以调整以下超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d | 128 | 嵌入维度 |
| learning_rate | 0.002 | 学习率 |
| epochs | 70 | 最大训练轮数 |
| batch_size | 80 | 批大小 |
| min_seq | 3 | 最小序列长度 |
| max_seq | 200 | 最大序列长度 |
| grad_clip | 15.0 | 梯度裁剪值 |
| head | 2 | 注意力头数 |
| beta_start | 0.0001 | 扩散过程起始β值 |
| beta_end | 0.02 | 扩散过程结束β值 |
| diff_num_step | 100 | 扩散步数 |
| lamda_1 | 0.01 | 正则化参数1 |
| lamda_2 | 0.001 | 正则化参数2 |
| patience | 30 | 早停耐心值 |

## 数据格式说明

### 训练/测试文件格式

每个用户的数据占3行：

```
[空行]
问题ID1,问题ID2,问题ID3,...
答案1,答案2,答案3,...
```

其中：
- 问题ID: 0到problem_max-1的整数（已重新映射）
- 答案: 0（错误）或 1（正确）

### 问题-技能映射文件

CSV格式，包含以下列：
- problem_id: 重新映射后的问题ID
- skill_id: 对应的技能ID

## 训练输出

训练完成后，会生成以下文件：

| 文件名 | 说明 |
|--------|------|
| DiffuQKT_{dataset}_0_model.pkl | 训练好的模型权重 |
| DiffuQKT_{dataset}_0_state.ckpt | 模型训练状态检查点 |
| {dataset}_output.txt | 训练过程中的指标日志 |

## 模型评估指标

训练过程中计算以下指标：

- Loss: 二元交叉熵损失
- ACC: 答案预测准确率
- AUC: ROC曲线下面积（ROC-AUC）

## 常见问题

### Q: 数据预处理失败
A: 确保数据文件位置正确
- ASSIST2009: ASSIST2009/skill_builder_data.csv
- ASSIST2017: ASSIST2017/anonymized_full_release_competition_dataset.csv

### Q: 为什么没有验证集指标？
A: 当前配置只使用训练集训练和测试集评估，不再分割验证集。

### Q: 如何继续训练已保存的模型？
A: 加载模型权重：
```python
model = DiffuQKT(...)
model.load_state_dict(torch.load('./DiffuQKT_assist09_0_model.pkl'))
```

## 许可证

MIT License
