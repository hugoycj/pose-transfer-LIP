import sys
sys.path.append('../')

import torch
import numpy as np

from model_factory.metric_utils import *

def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    im_width = scores.shape[1]
    im_length = scores.shape[2]
    _, ind = scores.topk(k, 1, True, True)
    ind = torch.squeeze(ind, dim=1)
    correct = ind.eq(targets)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size / im_width / im_length)

def pixel_accuracy(output, target):
    return 0

def mean_accuracy(output, target):
    return 0

def mean_iou(predicted_segmentation, gt_segmentation):
    _, predicted_segmentation = predicted_segmentation.topk(1, 1, True, True)
    predicted_segmentation = torch.squeeze(predicted_segmentation, dim=1)

    # print("gt_segmentation:", gt_segmentation.shape)
    predicted_segmentation = predicted_segmentation.detach().numpy()
    # print("predicted_segmentation:", predicted_segmentation.shape)
    
    classes, num_classes  = union_classes(predicted_segmentation, gt_segmentation)
    _, n_classes_gt = extract_classes(gt_segmentation)
    eval_mask, gt_mask = extract_both_masks(predicted_segmentation, gt_segmentation, classes, num_classes)

    iou_list = list([0]) * num_classes

    for i, _ in enumerate(classes):
        curr_eval_mask = eval_mask[:, i, :, :]
        curr_gt_mask = gt_mask[:, i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        intersect = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        union = np.sum(np.logical_or(curr_eval_mask, curr_gt_mask))
        iou_list[i] = intersect / union
 
    mean_iou_value = np.sum(iou_list) / n_classes_gt
    return mean_iou_value

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
