import torch
from torch import nn
import numpy as np
import argparse
import pickle
from roberta_k_fold import KFoldProcessor
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

class MacroScore(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        self.class_weight = nn.Parameter(torch.rand(class_num), requires_grad=True)
    def forward(self, inputs, targets):
        '''
        :param inputs: shape like (examples nums, class nums)
        :param targets: (examples nums, 1)
        :return:
        '''
        inputs = torch.argmax(self.class_weight * inputs, dim=-1)
        confusion_matrix = torch.zeros(self.class_num, self.class_num)
        for input, target in zip(inputs, targets):
            confusion_matrix[input.long(), target.long()] += 1
        precision = confusion_matrix.diag() / confusion_matrix.sum(0)
        recall = confusion_matrix.diag() / confusion_matrix.sum(1)
        f1 = 2 * precision * recall / (precision + recall)
        loss = -f1.mean()
        return loss, self.class_weight.data

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='模型的预测输出位置，应为一个examples nums * class nums的矩阵')
    parser.add_argument('--target_dir', type=str, help='模型的真实数据，包含句子的真正标签')
    parser.add_argument('--step', default=1000, type=int, help='随机搜索的步数')
    args = parser.parse_args()
    processor = KFoldProcessor(args, data_dir=args.target_dir)
    train_examples = processor.get_train_examples()
    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    targets = [label_map[example.label] for example in train_examples]
    import os
    score_list = []
    voting_list = []
    for path in os.listdir(args.input_dir):
        input_dir = os.path.join(args.input_dir, path)
        oof_dir = os.path.join(input_dir, 'oof_train')
        if not os.path.exists(oof_dir):
            continue
        with open(oof_dir, 'rb') as f:
            inputs = pickle.load(f)
        targets = np.array(targets,dtype=np.long)
        inputs = np.array(inputs, dtype=np.float)
        class_nums = 6
        class_weight = np.ones(class_nums)
        inputs_id = np.argmax(class_weight * inputs, axis=-1)
        f_socre = f1_score(y_pred=inputs_id, y_true=targets, average='macro')
        score_list.append({path:f_socre})
        report = classification_report(y_true=targets, y_pred=inputs_id, target_names=label_list, digits=4)
        print(report)
        if f_socre>0.69:
            voting_list.append(path)
    import json
    print(json.dumps(score_list, indent=4))
    print(voting_list)
    # epoch_nums = args.step
    # best_f_score = f_socre
    # best_weight = class_weight
    # for epoch in tqdm(range(epoch_nums)):
    #     class_weight = np.random.rand(class_nums)
    #     class_weight = softmax(class_weight)
    #     inputs_id = np.argmax(class_weight * inputs, axis=-1)
    #     f_score = f1_score(y_pred=inputs_id, y_true=targets, average='macro')
    #     if f_score > best_f_score:
    #         best_f_score = f_score
    #         print(best_f_score)
    #         best_weight = class_weight
    # print(best_f_score, best_weight)
    # import os
    # ouput_dir, _ = os.path.split(args.input_dir)
    # output_path = os.path.join(ouput_dir, 'weight')
    # print('write to '+output_path)
    # with open(output_path, 'wb') as fw:
    #     pickle.dump(best_weight.tolist(), fw)



    # model = MacroScore(class_num=6)
    # optim = torch.optim.SGD(model.parameters(), lr=0.5)


    # print('class best weight:', weight)

if __name__ == '__main__':
    main()

# nb_classes = 9
#
# confusion_matrix = torch.zeros(nb_classes, nb_classes)
# with torch.no_grad():
#     for i, (inputs, classes) in enumerate(dataloaders['val']):
#         inputs = inputs.to(device)
#         classes = classes.to(device)
#         outputs = model_ft(inputs)
#         _, preds = torch.max(outputs, 1)
#         for t, p in zip(classes.view(-1), preds.view(-1)):
#                 confusion_matrix[t.long(), p.long()] += 1
#
# print(confusion_matrix)
#
# # To get the per-class accuracy: precision
# precision = confusion_matrix.diag()/confusion_matrix.sum(1)
# print(confusion_matrix.diag()/confusion_matrix.sum(1))
#
# recall = confusion_matrix.diag()/confusion_matrix.sum(1)
# print(confusion_matrix.diag()/confusion_matrix.sum(0))
#
# f1 = 2*precision*recall/(precision+recall)
#
# mean = f1.diagonal().mean()