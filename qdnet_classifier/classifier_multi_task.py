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

WEIGHT_COLOR = 0.5
WEIGHT_ACTION = 0.5
assert (WEIGHT_COLOR + WEIGHT_ACTION == 1.0)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

loss_fn = nn.CrossEntropyLoss(reduction='mean')

class MultiLabelModel(nn.Module):
    def __init__(self, classifier, n_color_classes, n_action_classes, pretrained=False):
        super(MultiLabelModel, self).__init__()

        self.model_classifier = create_model( classifier, pretrained=pretrained, )
        last_channel = 512
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.color = nn.Sequential(
                     nn.Dropout(p=0.5),
                     nn.Linear(in_features=last_channel, out_features=n_color_classes) )
        self.action = nn.Sequential(
                     nn.Dropout(p=0.5),
                    nn.Linear(in_features=last_channel, out_features=n_action_classes) )

    def forward(self, x):
        x = self.model_classifier.forward_features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return {  'color': self.color(x),
                  'action': self.action(x), }

    @staticmethod
    def get_loss(output, target):
        loss_color = loss_fn(output['color'], target['color'].cuda())
        loss_action = loss_fn(output['action'], target['action'].cuda())

        loss = WEIGHT_COLOR * loss_color + WEIGHT_ACTION * loss_action
        return loss

    @staticmethod
    def get_accuracy(output, target, topk=(1,)):
        acc1_color, acc3_color = accuracy(output['color'], target['color'].cuda(), topk=topk)
        acc1_action, acc3_action = accuracy(output['action'], target['action'].cuda(), topk=topk)

        acc1 = WEIGHT_COLOR * acc1_color + WEIGHT_ACTION * acc1_action
        acc3 = WEIGHT_COLOR * acc3_color + WEIGHT_ACTION * acc3_action
        return acc1, acc3, {'color': acc1_color, 'action': acc1_action}

    @staticmethod
    def calculate_metrics(output, target):
        predicted_color = output['color'].cpu().argmax(1)
        gt_color = target['color_labels'].cpu()

        predicted_action = output['action'].cpu().argmax(1)
        gt_action = target['action_labels'].cpu()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accuracy_color = balanced_accuracy_score(y_true=gt_color.numpy(), y_pred=predicted_color.numpy())
            accuracy_action = balanced_accuracy_score(y_true=gt_action.numpy(), y_pred=predicted_action.numpy())

        return accuracy_color, accuracy_action


class MultilabelClassifier:
    def __init__(self, device, classifier_model, weights, num_color_classes, num_action_classes, use_ema=False):
        self.device = device 
        self.action_classifier = MultiLabelModel(
            classifier_model,
            n_color_classes=num_color_classes,
            n_action_classes=num_action_classes).to(self.device)
        load_checkpoint(self.action_classifier, weights, use_ema)
        self.action_classifier.eval()

    def predict(self, img_cv ):
        crop_tensor = self.action_transform(image=cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))["image"].to(self.device)

        output = self.action_classifier(crop_tensor[None, ...])
        if isinstance(output, (tuple, list)):
            output = output[0]

        color_prob = F.softmax(output['color'])
        color_output = torch.argmax(color_prob, axis=1)

        action_prob = F.softmax(output['action'])
        action_output = torch.argmax(action_prob, axis=1)

        return color_output, action_output, color_prob, action_prob
