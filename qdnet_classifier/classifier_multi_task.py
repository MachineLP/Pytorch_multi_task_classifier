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
