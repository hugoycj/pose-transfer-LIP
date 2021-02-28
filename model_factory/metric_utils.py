import numpy as np

def extract_both_masks(predicted_segmentation, gt_segmentation, classes, num_classes):
    predicted_mask = extract_masks(predicted_segmentation, classes, num_classes)
    gt_mask   = extract_masks(gt_segmentation, classes, num_classes)
    return predicted_mask, gt_mask


def extract_classes(segmentation):
    classes = np.unique(segmentation)
    num_classes = len(classes)

    return classes, num_classes

def union_classes(predicted_segmentation, gt_segmentation):
    predicted_classes, _ = extract_classes(predicted_segmentation)
    gt_classes, _   = extract_classes(gt_segmentation)

    classes = np.union1d(predicted_classes, gt_classes)
    num_classes = len(classes)

    return classes, num_classes


def extract_masks(segmentation, classes, num_classes):
    batch, h, w  = segmentation.shape
    masks = np.zeros((batch, num_classes, h, w))
    # print("segmentation:", segmentation.shape)
    # print("masks:", masks.shape)

    for i, c in enumerate(classes):
        masks[:, i, :, :] = segmentation == c

    return masks


def segmentation_size(segmentation):
    height = segmentation.shape[1]
    width  = segmentation.shape[2]

    return height, width