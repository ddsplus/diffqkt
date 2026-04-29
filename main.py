import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import argparse

from model import DiffuQKT
from run import run_epoch

mp2path = {
    'assist09': {
        'ques_skill_path': './data/ASSIST09/ques_skill.csv',
        'train_path': './data/ASSIST09/train_question.txt',
        'test_path': './data/ASSIST09/test_question.txt',
        'train_skill_path': './data/ASSIST09/train_skill.txt',
        'test_skill_path': './data/ASSIST09/test_skill.txt',
        'skill_max': None
    },
    'assist17': {
        'ques_skill_path': './data/ASSIST17/ques_skill.csv',
        'train_path': './data/ASSIST17/train_question.txt',
        'test_path': './data/ASSIST17/test_question.txt',
        'train_skill_path': './data/ASSIST17/train_skill.txt',
        'test_skill_path': './data/ASSIST17/test_skill.txt',
        'skill_max': None  # 将在运行时从数据计算
    },
    'xes3g5m': {
        'ques_skill_path': './data/XES3G5M/ques_skill.csv',
        'train_path': './data/XES3G5M/train_question.txt',
        'test_path': './data/XES3G5M/test_question.txt',
        'train_skill_path': './data/XES3G5M/train_skill.txt',
        'test_skill_path': './data/XES3G5M/test_skill.txt',
        'skill_max': None
    }
}

def main(dataset):
    if dataset not in mp2path:
        raise ValueError(f"未知数据集: {dataset}，可选: {list(mp2path.keys())}")

    with open(f'./result/{dataset}_output.txt', 'w') as file:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if 'ques_skill_path' in mp2path[dataset]:
            ques_skill_path = mp2path[dataset]['ques_skill_path']

        train_path = mp2path[dataset]['train_path']
        test_path = mp2path[dataset]['test_path']

        train_skill_path = mp2path[dataset]['train_skill_path']
        test_skill_path = mp2path[dataset]['test_skill_path']

        skill_max = mp2path[dataset]['skill_max']
        if skill_max is None:
            # 从数据文件读取skill_max
            skill_data = pd.read_csv(mp2path[dataset]['ques_skill_path'])
            skill_max = int(skill_data['skill_id'].max()) + 2

        if 'ques_skill_path' in mp2path[dataset]:
            pro_max = int(pd.read_csv(ques_skill_path)['problem_id'].max()) + 2
        else:
            pro_max = skill_max

        p = 0.4
        d = 128
        learning_rate = 0.002
        epochs = 70
        batch_size = 512
        min_seq = 3
        max_seq = 200
        grad_clip = 15.0
        head = 2
        beta_start = 0.0001
        beta_end = 0.02
        diff_num_step = 100
        lamda_1 = 0.01
        lamda_2 = 0.001
        patience = 30

        avg_auc = 0
        avg_acc = 0
        sublist = []

        # 只训练1次，而不是5次
        for now_step in range(1):
            best_acc = 0
            best_auc = 0
            state = {'auc': 0, 'acc': 0, 'loss': 0}

            model = DiffuQKT(pro_max, skill_max, d, p, head, beta_start, beta_end, diff_num_step, lamda_1, lamda_2)
            model = model.to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

            one_p = 0
            for epoch in range(epochs):
                # 训练
                train_loss, train_acc, train_auc = run_epoch(pro_max, train_path, train_skill_path, batch_size,
                                                             True, min_seq, max_seq, model, optimizer, criterion,
                                                             device,
                                                             grad_clip)
                # 评估测试集
                test_loss, test_acc, test_auc = run_epoch(pro_max, test_path, test_skill_path, batch_size, False,
                                                           min_seq, max_seq, model, optimizer, criterion, device,
                                                           grad_clip)
                print(
                    f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}, test_auc: {test_auc:.4f}')
                # 保存模型
                torch.save(model.state_dict(), f"./result/DiffuQKT_{dataset}_{now_step}_model.pkl")
                state['auc'] = test_auc
                state['acc'] = test_acc
                state['loss'] = test_loss
                torch.save(state, f'./result/DiffuQKT_{dataset}_{now_step}_state.ckpt')
                sublist.append(test_auc)
                # 早停机制（基于测试集AUC）
                if test_auc > best_auc:
                    one_p = 0
                    best_auc = test_auc
                    best_acc = test_acc
                else:
                    one_p += 1
                if one_p >= patience:
                    print(f'早停：{patience}个epoch没有改进，停止训练')
                    break
            avg_auc += best_auc
            avg_acc += best_acc
        print(f'*******************************************************************************')
        print(f'最终结果 - best_acc: {avg_acc:.4f}, best_auc: {avg_auc:.4f}')
        print(f'*******************************************************************************')
        line = '\n'.join(str(item) for item in sublist)
        file.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=list(mp2path.keys()), help='选择要训练的数据集')
    args = parser.parse_args()
    main(args.dataset)
