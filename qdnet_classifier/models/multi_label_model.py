from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import warnings

WEIGHT_COLOR = 0.75
WEIGHT_ACTION = 0.25
assert (WEIGHT_COLOR + WEIGHT_ACTION == 1.0)


class MultiLabelModel(nn.Module):
    def __init__(self, model, n_color_classes, n_action_classes):
        super().__init__()
        self.base_model = model.as_sequential_for_ML()
        last_channel = model.num_features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.color = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_color_classes)
        )
        self.action = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_action_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return {
            'color': self.color(x),
            'action': self.action(x),
        }

    @staticmethod
    def get_loss(loss_fn, output, target):
        loss_color = loss_fn(output['color'], target['color_labels'].cuda())
        loss_action = loss_fn(output['action'], target['action_labels'].cuda())

        loss = WEIGHT_COLOR * loss_color + WEIGHT_ACTION * loss_action
        return loss

    @staticmethod
    def get_accuracy(accuracy, output, target, topk=(1,)):
        acc1_color, acc3_color = accuracy(output['color'], target['color_labels'].cuda(), topk=topk)
        acc1_action, acc3_action = accuracy(output['action'], target['action_labels'].cuda(), topk=topk)

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
