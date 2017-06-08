#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 03:49:05 2017
@author: ks
Detection and Pattern Recognition Utilities
"""

from __future__ import absolute_import

import numpy as np


def binary_counts(prediction, reference):
    ct = {}
    ct['binary_tp'] = np.sum(np.logical_and(reference, prediction))
    ct['binary_tn'] = np.sum(np.logical_and(np.logical_not(reference), np.logical_not(prediction)))
    ct['binary_fp'] = np.sum(np.logical_and(np.logical_not(reference), prediction))
    ct['binary_fn'] = np.sum(np.logical_and(reference, np.logical_not(prediction)))
    for k in ct.keys():
        ct[k] = ct[k] / float(len(prediction))
    return ct


def binary_metrics(binary_tp, binary_tn, binary_fp, binary_fn, beta=1.0):
    mt = {}
    mt['binary_tp'], mt['binary_tn'], mt['binary_fp'], mt['binary_fn'], mt['beta'] = \
        binary_tp, binary_tn, binary_fp, binary_fn, beta
    bb = beta ** 2
    _eps = mt['_eps'] = np.finfo(np.float32).eps

    tp = mt['tp'] = mt['true_positives'] = mt['binary_tp']
    tn = mt['tn'] = mt['true_negatives'] = mt['binary_tn']
    fp = mt['fp'] = mt['false_positives'] = mt['binary_fp']
    fn = mt['fn'] = mt['false_negatives'] = mt['binary_fn']

    pp = mt['pp'] = mt['bias'] = mt['pred_positives'] = tp + fp
    pn = mt['pn'] = mt['pred_negatives'] = tn + fn
    rn = mt['rn'] = mt['real_negatives'] = tn + fp
    rp = mt['rp'] = mt['real_positives'] = mt['prevalence'] = tp + fn
    mt['cr'] = mt['class_ratio'] = rn / (rp + _eps)

    tpr = mt['tpr'] = mt['true_positive_rate'] = \
        mt['sensitivity'] = mt['hit_rate'] = mt['recall_'] = tp / (rp + _eps)
    mt['tnr'] = mt['true_negative_rate'] = mt['inverse_recall'] = mt['specificity'] = tn / (rn + _eps)
    mt['fpr'] = mt['false_positive_rate'] = mt['fall_out'] = fp / (rn + _eps)
    mt['fnr'] = mt['false_negative_rate'] = mt['miss_rate'] = fn / (rp + _eps)

    mt['tpr_fnr'] = tpr + mt['fnr']
    mt['tnr_fpr'] = mt['tnr'] + mt['fpr']

    ppv = mt['ppv'] = mt['confidence'] = mt['positive_predictive_value'] = mt['precision_'] = tp / (pp + _eps)
    npv = mt['npv'] = mt['inverse_precision'] = mt['negative_predictive_value'] = tn / (pn + _eps)
    mt['fdr'] = mt['false_discovery_rate'] = 1.0 - ppv
    mt['for'] = mt['false_omission_rate'] = 1.0 - npv

    mt['arithmetic_mean'] = mt['ameasure'] = (tpr + ppv) / 2.0
    mt['geometric_mean'] = mt['gmeasure'] = np.sqrt(tpr * ppv)
    mt['harmonic_mean'] = mt['f1s'] = mt['fmeasure_'] = mt['f1score_'] = \
        ((2.0 * tpr * ppv) / (tpr + ppv)) if tp > 0 else 0
    mt['harmonic_mean2'] = mt['f1s2'] = mt['fmeasure_2'] = mt['f1score_2'] = \
        ((2.0 * tp) / (2.0 * tp + fp + fn)) if tp > 0 else 0
    mt['fbetascore'] = mt['fbs_'] = mt['fbeta_'] = \
        ((1 + bb) * ppv * tpr / (bb * ppv + tpr)) if tp > 0 else 0

    mt['acc_'] = mt['accuracy_'] = mt['binary_accuracy_'] = tp + tn
    mt['jcd'] = mt['jaccard'] = mt['tanimoto_similarity_coefficient'] = tp / (tp + fn + fp + _eps)
    mt['acc_azbm'] = float(rn) / float(tn + tp + fn + fp + _eps)

    mt['auc'] = (tpr + mt['tnr']) / 2.0
    mt['informedness'] = tpr + mt['tnr'] - 1
    mt['markedness'] = ppv + npv - 1

    return mt


def binary_metrics_2(binary_tp, binary_tn, binary_fp, binary_fn, beta=1.0):
    mt = {}
    mt['binary_tp'], mt['binary_tn'], mt['binary_fp'], mt['binary_fn'], mt['beta'] = \
        binary_tp, binary_tn, binary_fp, binary_fn, beta
    bb = beta ** 2
    _eps = mt['_eps'] = np.finfo(np.float32).eps

    tp = mt['tp'] = mt['true_positives'] = mt['binary_tp']
    tn = mt['tn'] = mt['true_negatives'] = mt['binary_tn']
    fp = mt['fp'] = mt['false_positives'] = mt['binary_fp']
    fn = mt['fn'] = mt['false_negatives'] = mt['binary_fn']

    pp = mt['pp'] = mt['bias'] = mt['pred_positives'] = tp + fp
    pn = mt['pn'] = mt['pred_negatives'] = tn + fn
    rn = mt['rn'] = mt['real_negatives'] = tn + fp
    rp = mt['rp'] = mt['real_positives'] = mt['prevalence'] = tp + fn
    mt['cr'] = mt['class_ratio'] = (rn + _eps) / (rp + _eps)

    tpr = mt['tpr'] = mt['true_positive_rate'] = \
        mt['sensitivity'] = mt['hit_rate'] = mt['recall_'] = (tp + _eps) / (rp + _eps)
    mt['tnr'] = mt['true_negative_rate'] = mt['inverse_recall'] = \
        mt['specificity'] = (tn + _eps) / (rn + _eps)
    mt['fpr'] = mt['false_positive_rate'] = mt['fall_out'] = (fp + _eps) / (rn + _eps)
    mt['fnr'] = mt['false_negative_rate'] = mt['miss_rate'] = (fn + _eps) / (rp + _eps)

    mt['tpr_fnr'] = tpr + mt['fnr']
    mt['tnr_fpr'] = mt['tnr'] + mt['fpr']

    ppv = mt['ppv'] = mt['confidence'] = \
        mt['positive_predictive_value'] = mt['precision_'] = (tp + _eps) / (pp + _eps)
    npv = mt['npv'] = mt['inverse_precision'] = \
        mt['negative_predictive_value'] = (tn + _eps) / (pn + _eps)
    mt['fdr'] = mt['false_discovery_rate'] = 1.0 - ppv
    mt['for'] = mt['false_omission_rate'] = 1.0 - npv

    mt['arithmetic_mean'] = mt['ameasure'] = (tpr + ppv) / 2.0
    mt['geometric_mean'] = mt['gmeasure'] = np.sqrt(tpr * ppv)
    mt['harmonic_mean'] = mt['f1s'] = mt['fmeasure_'] = mt['f1score_'] = \
        ((2.0 * tpr * ppv + _eps) / (tpr + ppv + _eps)) if tp > 0 else 0
    mt['harmonic_mean2'] = mt['f1s2'] = mt['fmeasure_2'] = mt['f1score_2'] = \
        ((2.0 * tp + _eps) / (2.0 * tp + fp + fn + _eps)) if tp > 0 else 0
    mt['fbetascore'] = mt['fbs_'] = mt['fbeta_'] = \
        (((1 + bb) * ppv * tpr + _eps) / ((bb * ppv + tpr) + _eps)) if tp > 0 else 0

    mt['acc_'] = mt['accuracy_'] = mt['binary_accuracy_'] = tp + tn
    mt['jcd'] = mt['jaccard'] = mt['tanimoto_similarity_coefficient'] = \
        (tp + _eps) / (tp + fn + fp + _eps)
    mt['acc_azbm'] = float(rn + _eps) / float(tn + tp + fn + fp + _eps)

    mt['auc'] = (tpr + mt['tnr']) / 2.0
    mt['informedness'] = tpr + mt['tnr'] - 1
    mt['markedness'] = ppv + npv - 1

    return mt


