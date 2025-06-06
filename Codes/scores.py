import numpy as np
from scipy import ndimage

def correctness(TP, FP, eps=1e-12):
    return TP/(TP + FP + eps) # precision

def completeness(TP, FN, eps=1e-12):
    return TP/(TP + FN + eps) # recall

def quality(corr, comp, eps=1e-12):
    return (comp*corr)/(comp-comp*corr+corr + eps)

def f1(corr, comp, eps=1e-12):
    return 2.0/(1.0/(corr+eps) + 1.0/(comp+eps))

def relaxed_confusion_matrix(pred_s, gt_s, slack=3):

    distances_gt = ndimage.distance_transform_edt((np.logical_not(gt_s)))
    distances_pred = ndimage.distance_transform_edt((np.logical_not(pred_s)))

    true_pos_area_gt = distances_gt<=slack
    false_pos_area = distances_gt>slack

    true_pos_area_pred = distances_pred<=slack
    false_neg_area = distances_pred>slack

    true_positives_gt = np.logical_and(true_pos_area_gt, pred_s).sum() # lenght of the matched extraction
    false_positives = np.logical_and(false_pos_area, pred_s).sum()

    true_positives_pred = np.logical_and(true_pos_area_pred, gt_s).sum() # lenght of the matched reference
    false_negatives = np.logical_and(false_neg_area, gt_s).sum()

    return true_positives_gt, true_positives_pred, false_negatives, false_positives

def compute_scores(TP_g, TP_p, FN, FP, eps=1e-12):
    corr = correctness(TP_g, FP, eps)
    comp = completeness(TP_p, FN, eps)
    qual = quality(corr, comp)
    return corr, comp, qual

def correctness_completeness_quality(pred_s, gt_s, slack=3):

    TP_g, TP_p, FN, FP = relaxed_confusion_matrix(pred_s, gt_s, slack)

    corr, comp, qual = compute_scores(TP_g, TP_p, FN, FP)

    return corr, comp, qual