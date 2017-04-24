#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 03:49:05 2017

@author: ks

Detection and Pattern Recognition Utilities
"""

from __future__ import absolute_import

import numpy as np


def binary_metrics(binary_tp, binary_tn, binary_fp, binary_fn, beta=1.0):
    _bb = beta ** 2
    _eps = np.finfo(np.float32).eps

    tp = true_positives = binary_tp
    tn = true_negatives = binary_tn
    fp = false_positives = binary_fp
    fn = false_negatives = binary_fn

    pp = bias = pred_positives = tp + fp
    pn = pred_negatives = tn + fn
    rn = real_negatives = tn + fp
    rp = real_positives = prevalence = tp + fn
    cr = class_ratio = rn / (rp + _eps)

    tpr = recall = sensitivity = hit_rate = tp / (rp + _eps)
    tnr = inverse_recall = specificity = tn / (rn + _eps)
    fpr = fall_out = fp / (rp + _eps)
    fnr = miss_rate = fn / (rn + _eps)

    ppv = precision = confidence = positive_predictive_value = tp / (pp + _eps)
    npv = inverse_precision = negative_predictive_value = tn / (pn + _eps)
    fdr = false_discovery_rate = 1.0 - ppv
    fOr = false_omission_rate = 1.0 - npv

    arithmetic_mean = ameasure = (tpr + ppv) / 2.0
    geometric_mean = gmeasure = np.sqrt(tpr * ppv)
    harmonic_mean = fmeasure = f1score = f1s = (2.0 * tpr * ppv / (tpr + ppv) if tp > 0 else 0)
    fmeasure_2 = 2.0 * tp / (2.0 * tp + fp + fn) if tp > 0 else 0
    fbs = fbeta = 0 if tp <= 0 else  (1 + _bb) * ppv * tpr / (_bb * ppv + tpr)

    acc_2 = accuracy_2 = binary_accuracy_2 = tp + tn
    jcd = jaccard = tanimoto_similarity_coefficient = tp / (tp + fn + fp + _eps)

    auc = (tpr + tnr) / 2.0
    informedness = tpr + tnr - 1
    markedness = ppv + npv - 1


    return {k: v for k, v in locals().items() if not k.startswith('_')}

