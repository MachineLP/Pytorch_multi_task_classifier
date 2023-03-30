import cv2
import torch

import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
import albumentations as A
from albumentations.pytorch import ToTensorV2

from qdnet_classifier.models import create_model, load_checkpoint

from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import warnings
import numpy as np
from typing import Optional
from typing import List
from mtcnn import MTCNN




# 计算准确率——方式1
# 设定一个阈值，当预测的概率值大于这个阈值，则认为这幅图像中含有这类标签
def calculate_acuracy_mode_one(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num
    return precision.item(), recall.item()
 
# 计算准确率——方式2
# 取预测概率最大的前top个标签，作为模型的预测结果
def calculate_acuracy_mode_two(model_pred, labels):
    # 取前top个预测结果作为模型的预测结果
    precision = 0
    recall = 0
    top = 5
    # 对预测结果进行按概率值进行降序排列，取概率最大的top个结果作为模型的预测结果
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0,pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # 对每一幅图像进行预测准确率的计算
        precision += true_predict_num / top
        # 对每一幅图像进行预测查全率的计算
        recall += true_predict_num / target_one_num
    return precision, recall


loss_fn = nn.BCELoss()
Sigmoid_fun = nn.Sigmoid()

class MultiLabelModel(nn.Module):
    def __init__(self, classifier, n_classes=509, pretrained=False):
        super(MultiLabelModel, self).__init__()

        self.model_classifier = create_model( classifier, pretrained=pretrained, )
        last_channel = 512  # 512, 1024, 2048
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Sequential(
                     nn.Dropout(p=0.5),
                     nn.Linear(in_features=last_channel, out_features=n_classes) )

    def forward(self, x):
        x = self.model_classifier.forward_features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.output(x)

    @staticmethod
    def get_loss(output, target):
        loss = loss_fn(Sigmoid_fun(output), target)
        return loss

    @staticmethod
    def get_accuracy(output, target):
        precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(output), target)
        # precision, recall = calculate_acuracy_mode_two(Sigmoid_fun(outputs), labels)
        return precision, recall
    
    # acc, precision, recall, f1
    def compute_metrics(self, output, target, **kwargs):
        metrics = dict()

        with torch.no_grad():
            predictions = output #torch.softmax(output, 1).max(1)[1].cpu().numpy()
            target = target #target.cpu().numpy()

            metrics["accuracy"] = accuracy_score(target, predictions)
            metrics["per_class"] = classification_report(
                target, predictions,
                output_dict=True,
                labels=list(range(22)),
                target_names=['10019', '10004', '10011', '10015', '10013', '10005', '10012', '10108', '10016', '10006', '10020', '10002', '10009', '10017', '10010', '10007', '10014', '10001', '10018', '10021', '13934', '10003'],
                zero_division=1)

        def to_tensor_recursive(val):
            if not isinstance(val, dict):
                return torch.as_tensor(val).to(output)
            for k, v in val.items():
                val[k] = to_tensor_recursive(v)
            return val

        return metrics
    
    @staticmethod
    def print_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List] = None,
        hide_zeroes: bool = False,
        hide_diagonal: bool = False,
        hide_threshold: Optional[float] = None,
        class_names: Optional[List[str]] = None,
    ):
        """Print a nicely formatted confusion matrix with labelled rows and columns.

        Predicted labels are in the top horizontal header, true labels on the vertical header.

        Args:
            y_true (np.ndarray): ground truth labels
            y_pred (np.ndarray): predicted labels
            labels (Optional[List], optional): list of all labels. If None, then all labels present in the data are
                displayed. Defaults to None.
            hide_zeroes (bool, optional): replace zero-values with an empty cell. Defaults to False.
            hide_diagonal (bool, optional): replace true positives (diagonal) with empty cells. Defaults to False.
            hide_threshold (Optional[float], optional): replace values below this threshold with empty cells. Set to None
                to display all values. Defaults to None.
        """
        if labels is None:
            if class_names is None:
                labels = np.unique(np.concatenate((y_true, y_pred)))
            else:
                labels = np.arange(len(class_names))

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if class_names is not None:
            labels = class_names

        # find which fixed column width will be used for the matrix
        columnwidth = max(
            [len(str(x)) for x in labels] + [5]
        )  # 5 is the minimum column width, otherwise the longest class name
        empty_cell = ' ' * columnwidth

        # top-left cell of the table that indicates that top headers are predicted classes, left headers are true classes
        padding_fst_cell = (columnwidth - 3) // 2  # double-slash is int division
        fst_empty_cell = padding_fst_cell * ' ' + 't/p' + ' ' * (columnwidth - padding_fst_cell - 3)

        # Print header
        print('    ' + fst_empty_cell, end=' ')
        for label in labels:
            print(f'{label:{columnwidth}}', end=' ')  # right-aligned label padded with spaces to columnwidth

        print()  # newline
        # Print rows
        for i, label in enumerate(labels):
            print(f'    {label:{columnwidth}}', end=' ')  # right-aligned label padded with spaces to columnwidth
            for j in range(len(labels)):
                # cell value padded to columnwidth with spaces and displayed with 1 decimal
                cell = f'{cm[i, j]:{columnwidth}}'
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print(cell, end=' ')
            print()


class MultilabelClassifier:
    def __init__(self, device, classifier_model, weights, num_classes, use_ema=False):
        self.device = device 
        self.classifier = MultiLabelModel(
            classifier_model,
            n_classes=num_classes).to(self.device)
        load_checkpoint(self.classifier, weights, use_ema)
        self.classifier.eval()

    def predict(self, img_cv ):
        crop_tensor = self.action_transform(image=cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))["image"].to(self.device)

        output = self.classifier(crop_tensor[None, ...])
        if isinstance(output, (tuple, list)):
            output = output[0]

        prob = Sigmoid_fun(output)

        return prob
