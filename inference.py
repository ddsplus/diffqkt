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
    向答案序列中添加噪声（直接翻转整个序列中百分比的答案，不区分0/1）
    
    Args:
        last_ans: 前一步的答案 [batch, seq]
        next_ans: 下一步的答案 [batch, seq]
        noise_ratio: 噪声强度 (0-1，表示翻转的百分比)
    
    Returns:
        带噪声的 last_ans 和 next_ans
    """
    if noise_ratio <= 0:
        return last_ans, next_ans
    
    device = last_ans.device
    
    # 复制答案避免修改原始数据
    last_ans_noisy = last_ans.clone().float()
    next_ans_noisy = next_ans.clone().float()
    
    # 为 last_ans 添加噪声 - 直接翻转百分比位置的答案
    batch_size, seq_len = last_ans.shape
    num_flip = int(batch_size * seq_len * noise_ratio)
    
    if num_flip > 0:
        # 随机选择要翻转的位置
        flip_indices = torch.randperm(batch_size * seq_len, device=device)[:num_flip]
        
        for idx in flip_indices:
            b = idx // seq_len
            s = idx % seq_len
            # 直接翻转，不区分0/1
            last_ans_noisy[b, s] = 1 - last_ans_noisy[b, s]
    
    # 为 next_ans 添加噪声 - 直接翻转百分比位置的答案
    num_flip = int(batch_size * seq_len * noise_ratio)
    
    if num_flip > 0:
        flip_indices = torch.randperm(batch_size * seq_len, device=device)[:num_flip]
        
        for idx in flip_indices:
            b = idx // seq_len
            s = idx % seq_len
            # 直接翻转，不区分0/1
            next_ans_noisy[b, s] = 1 - next_ans_noisy[b, s]
    
    return last_ans_noisy.long(), next_ans_noisy.long()


def run_inference(model_path, dataset, noise_ratios, batch_size=512, max_seq=200, min_seq=3, device='cpu'):
    """
    运行推理并计算 AUC 和 ACC
    
    Args:
        model_path: 模型权重文件路径
        dataset: 数据集名称 (assist09, assist17, xes3g5m, statics2011)
        noise_ratios: 噪声比例列表或单个值 (0-1)
        batch_size: 批量大小
        max_seq: 最大序列长度
        min_seq: 最小序列长度
        device: 计算设备
    
    Returns:
        结果字典 {noise_ratio: (auc, acc)}
    """
    
    # 确保 noise_ratios 是列表
    if not isinstance(noise_ratios, (list, tuple)):
        noise_ratios = [noise_ratios]
    
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
    
    results = {}
    
    # 对每个噪声强度进行推理
    for noise_ratio in noise_ratios:
        if not 0 <= noise_ratio <= 1:
            print(f"警告: noise_ratio {noise_ratio} 不在 0-1 范围内，跳过")
            continue
        
        total_correct = 0
        total_num = 0
        labels = []
        outputs = []
        
        print(f"\n推理中... 噪声强度: {noise_ratio:.2%}")
        
        with torch.no_grad():
            for data in tqdm(data_loader, desc=f'噪声{noise_ratio:.1%}'):
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
        
        results[noise_ratio] = (auc, acc)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='DiffuQKT 推理脚本')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['assist09', 'assist17', 'xes3g5m', 'statics2011'],
                       help='数据集名称')
    parser.add_argument('--noise_ratios', type=float, nargs='*', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                       help='噪声比例列表 (默认: 0.0 0.1 0.2 0.3 0.4 0.5)')
    parser.add_argument('--batch_size', type=int, default=512, help='批量大小')
    parser.add_argument('--max_seq', type=int, default=200, help='最大序列长度')
    parser.add_argument('--min_seq', type=int, default=3, help='最小序列长度')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 验证噪声比例
    for nr in args.noise_ratios:
        if not 0 <= nr <= 1:
            raise ValueError(f"noise_ratio {nr} 必须在 0-1 之间")
    
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model_path}")
    print(f"噪声强度: {args.noise_ratios}")
    print("=" * 60)
    
    # 运行推理
    results = run_inference(
        model_path=args.model_path,
        dataset=args.dataset,
        noise_ratios=args.noise_ratios,
        batch_size=args.batch_size,
        max_seq=args.max_seq,
        min_seq=args.min_seq,
        device=args.device
    )
    
    # 输出结果表格
    print("\n" + "=" * 60)
    print(f"推理完成！数据集: {args.dataset}")
    print("=" * 60)
    print(f"{'噪声强度':<12} {'AUC':<10} {'ACC':<10}")
    print("-" * 60)
    
    for noise_ratio in sorted(results.keys()):
        auc, acc = results[noise_ratio]
        print(f"{noise_ratio:<12.1%} {auc:<10.4f} {acc:<10.4f}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
