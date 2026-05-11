import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

from model import DiffuQKT
from load_data import getLoader


def add_noise_to_answers(last_ans, next_ans, noise_ratio):
    """
    向答案序列中添加噪声（随机翻转指定百分比的答案）
    
    Args:
        last_ans: 前一步的答案 [batch, seq]
        next_ans: 下一步的答案 [batch, seq]
        noise_ratio: 噪声比例 (0-1)
    
    Returns:
        带噪声的 last_ans 和 next_ans
    """
    if noise_ratio <= 0:
        return last_ans, next_ans
    
    device = last_ans.device
    
    # 复制答案避免修改原始数据
    last_ans_noisy = last_ans.clone()
    next_ans_noisy = next_ans.clone()
    
    # 为 last_ans 添加噪声
    batch_size, seq_len = last_ans.shape
    num_flip = max(1, int(batch_size * seq_len * noise_ratio))
    flip_indices = torch.randint(0, batch_size * seq_len, (num_flip,), device=device)
    
    for idx in flip_indices:
        b = idx // seq_len
        s = idx % seq_len
        if last_ans_noisy[b, s] != 0:  # 只翻转非零位置（有效答案）
            last_ans_noisy[b, s] = 1 - last_ans_noisy[b, s]
    
    # 为 next_ans 添加噪声
    num_flip = max(1, int(batch_size * seq_len * noise_ratio))
    flip_indices = torch.randint(0, batch_size * seq_len, (num_flip,), device=device)
    
    for idx in flip_indices:
        b = idx // seq_len
        s = idx % seq_len
        if next_ans_noisy[b, s] != 0:  # 只翻转非零位置（有效答案）
            next_ans_noisy[b, s] = 1 - next_ans_noisy[b, s]
    
    return last_ans_noisy, next_ans_noisy


def run_inference(model_path, dataset, noise_ratio, batch_size=512, max_seq=200, min_seq=3, device='cpu'):
    """
    运行推理并计算 AUC 和 ACC
    
    Args:
        model_path: 模型权重文件路径
        dataset: 数据集名称 (assist09, assist17, xes3g5m, statics2011)
        noise_ratio: 噪声比例 (0-1)
        batch_size: 批量大小
        max_seq: 最大序列长度
        min_seq: 最小序列长度
        device: 计算设备
    
    Returns:
        auc, acc
    """
    
    # 数据集配置
    dataset_config = {
        'assist09': {
            'ques_skill_path': './data/ASSIST09/ques_skill.csv',
            'test_path': './data/ASSIST09/test_question.txt',
            'test_skill_path': './data/ASSIST09/test_skill.txt',
        },
        'assist17': {
            'ques_skill_path': './data/ASSIST17/ques_skill.csv',
            'test_path': './data/ASSIST17/test_question.txt',
            'test_skill_path': './data/ASSIST17/test_skill.txt',
        },
        'xes3g5m': {
            'ques_skill_path': './data/XES3G5M/ques_skill.csv',
            'test_path': './data/XES3G5M/test_question.txt',
            'test_skill_path': './data/XES3G5M/test_skill.txt',
        },
        'statics2011': {
            'ques_skill_path': './data/STATICS2011/ques_skill.csv',
            'test_path': './data/STATICS2011/test_question.txt',
            'test_skill_path': './data/STATICS2011/test_skill.txt',
        }
    }
    
    if dataset not in dataset_config:
        raise ValueError(f"未知数据集: {dataset}，可选: {list(dataset_config.keys())}")
    
    # 获取数据配置
    config = dataset_config[dataset]
    
    # 计算 skill_max 和 pro_max
    ques_skill_df = pd.read_csv(config['ques_skill_path'])
    skill_max = int(ques_skill_df['skill_id'].max()) + 2
    pro_max = int(ques_skill_df['problem_id'].max()) + 2
    
    # 模型超参数
    d = 128
    p = 0.4
    head = 2
    beta_start = 0.0001
    beta_end = 0.02
    diff_num_step = 100
    lamda_1 = 0.01
    lamda_2 = 0.001
    
    # 创建模型
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DiffuQKT(pro_max, skill_max, d, p, head, beta_start, beta_end, diff_num_step, lamda_1, lamda_2)
    
    # 加载模型权重
    if not torch.cuda.is_available() and device.type == 'cuda':
        # 如果没有 GPU，从 GPU 模型加载到 CPU
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        state_dict = torch.load(model_path, map_location=device)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 加载测试数据
    data_loader = getLoader(pro_max, config['test_path'], config['test_skill_path'], 
                           batch_size, is_train=False, min_problem_num=min_seq, max_problem_num=max_seq)
    
    total_correct = 0
    total_num = 0
    labels = []
    outputs = []
    
    print(f"开始推理... 数据集: {dataset}, 噪声强度: {noise_ratio:.2%}")
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc='推理进度'):
            last_problem, last_skill, last_ans, next_problem, next_skill, next_ans, mask = data
            
            # 添加噪声到答案
            if noise_ratio > 0:
                last_ans_noisy, next_ans_noisy = add_noise_to_answers(last_ans, next_ans, noise_ratio)
            else:
                last_ans_noisy = last_ans
                next_ans_noisy = next_ans
            
            # 推理
            predict, _, _ = model(last_problem, last_skill, last_ans_noisy, 
                                 next_problem, next_skill, next_ans_noisy, mask, is_train=False)
            
            # 提取有效预测和标签
            next_predict = torch.masked_select(predict, mask)
            next_true = torch.masked_select(next_ans, mask)  # 使用原始答案计算指标
            
            labels.extend(next_true.cpu().numpy())
            outputs.extend(next_predict.cpu().numpy())
            
            total_num += len(next_true)
            to_pred = (next_predict >= 0.5).long()
            total_correct += (next_true == to_pred).sum().item()
    
    # 计算指标
    acc = total_correct / total_num if total_num > 0 else 0
    auc = metrics.roc_auc_score(labels, outputs) if len(set(labels)) > 1 else 0
    
    return auc, acc


def main():
    parser = argparse.ArgumentParser(description='DiffuQKT 推理脚本')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['assist09', 'assist17', 'xes3g5m', 'statics2011'],
                       help='数据集名称')
    parser.add_argument('--noise_ratio', type=float, default=0.0, 
                       help='噪声比例 (0-1，表示随机翻转答案的百分比)')
    parser.add_argument('--batch_size', type=int, default=512, help='批量大小')
    parser.add_argument('--max_seq', type=int, default=200, help='最大序列长度')
    parser.add_argument('--min_seq', type=int, default=3, help='最小序列长度')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 验证噪声比例
    if not 0 <= args.noise_ratio <= 1:
        raise ValueError("noise_ratio 必须在 0-1 之间")
    
    # 运行推理
    auc, acc = run_inference(
        model_path=args.model_path,
        dataset=args.dataset,
        noise_ratio=args.noise_ratio,
        batch_size=args.batch_size,
        max_seq=args.max_seq,
        min_seq=args.min_seq,
        device=args.device
    )
    
    # 输出结果
    print("\n" + "="*50)
    print(f"推理完成！")
    print(f"数据集: {args.dataset}")
    print(f"噪声强度: {args.noise_ratio:.2%}")
    print(f"AUC: {auc:.4f}")
    print(f"ACC: {acc:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
