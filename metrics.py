import torch
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import numpy as np

EPSILON = 1e-32


# 定义图像的平均绝对误差（MAE）函数
def mae(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum(abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        # y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target, weight=self.weight,
                             ignore_index=self.ignore_index)

class DiceLoss:
    def __init__(self, smooth=1):
        self.smooth = smooth

    def dice_coef(self, y_true, y_pred):
        intersection = torch.sum(y_true * y_pred)
        return (2. * intersection + self.smooth) / (torch.sum(y_true) + torch.sum(y_pred) + self.smooth)

    def dice_loss(self, y_true, y_pred):
        dice_loss = 1 - self.dice_coef(y_true, y_pred)
        return dice_loss

class JaccardLoss:
    def __init__(self, smooth=1):
        self.smooth = smooth

    def jaccard_similarity(self, y_true, y_pred):
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum((y_true + y_pred) - (y_true * y_pred))
        return intersection / (union + self.smooth)

    def jaccard_loss(self, y_true, y_pred):
        loss = 1 - self.jaccard_similarity(y_true, y_pred)
        return loss


# class Dice_Loss:
#     def __init__(self, smooth=1):
#         self.smooth = smooth
#
#     def dice_coef(self, y_true, y_pred):
#         intersection = np.sum(y_true * y_pred)
#         return (2. * intersection + self.smooth) / (np.sum(y_true) + np.sum(y_pred) + self.smooth)
#
#     def dice_loss(self, y_true, y_pred):
#         dice_loss = 1 - self.dice_coef(y_true, y_pred)
#         return np.float64(dice_loss)
#
# class Jaccard_Loss:
#     def __init__(self, smooth=1):
#         self.smooth = smooth
#
#     def jaccard_similarity(self, y_true, y_pred):
#         intersection = np.sum(y_true * y_pred)
#         union = np.sum((y_true + y_pred) - (y_true * y_pred))
#         return intersection / (union + self.smooth)
#
#     def jaccard_loss(self, y_true, y_pred):
#         loss = 1 - self.jaccard_similarity(y_true, y_pred)
#         return np.float64(loss)


# class BCE_Dice_Loss:
#     def __init__(self, beta=0.25, alpha=0.25, smooth=1):
#         self.beta = beta
#         self.alpha = alpha
#         self.smooth = smooth
#
#     def dice_coef(self, y_true, y_pred):
#         intersection = torch.sum(y_true * y_pred)
#         return (2. * intersection + self.smooth) / (torch.sum(y_true) + torch.sum(y_pred) + self.smooth)
#
#     def bce_dice_loss(self, y_true, y_pred):
#         y_true = torch.tensor(y_true, dtype=torch.float32)
#         y_pred = torch.tensor(y_pred, dtype=torch.float32)
#
#         bce_loss = F.binary_cross_entropy(y_pred, y_true)
#         dice_loss = 1 - self.dice_coef(y_true, y_pred)
#         return (bce_loss + dice_loss) / 2.0


def classwise_iou(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    dims = (0, *range(2, len(output.shape)))
    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output*gt
    union = output + gt - intersection
    classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)

    return classwise_iou


def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    return classwise_f1


def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (0, *range(2, len(output.shape)))

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return classwise_scores 

    return weighted_metric


jaccard_index = make_weighted_metric(classwise_iou)
f1_score = make_weighted_metric(classwise_f1)


if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    print(classwise_iou(output, gt))
