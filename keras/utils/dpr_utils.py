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

    tpr = mt['tpr'] = mt['_recall'] = mt['sensitivity'] = mt['hit_rate'] = tp / (rp + _eps)
    mt['tnr'] = mt['inverse_recall'] = mt['specificity'] = tn / (rn + _eps)
    mt['fpr'] = mt['fall_out'] = fp / (rp + _eps)
    mt['fnr'] = mt['miss_rate'] = fn / (rn + _eps)

    ppv = mt['ppv'] = mt['_precision'] = mt['confidence'] = mt['positive_predictive_value'] = \
        tp / (pp + _eps)
    npv = mt['npv'] = mt['inverse_precision'] = mt['negative_predictive_value'] = tn / (pn + _eps)
    mt['fdr'] = mt['false_discovery_rate'] = 1.0 - ppv
    mt['_for'] = mt['false_omission_rate'] = 1.0 - npv

    mt['arithmetic_mean'] = mt['ameasure'] = (tpr + ppv) / 2.0
    mt['geometric_mean'] = mt['gmeasure'] = np.sqrt(tpr * ppv)
    mt['harmonic_mean'] = mt['_fmeasure'] = mt['_f1score'] = mt['_f1s'] = \
        (2.0 * mt['tpr'] * ppv / (tpr + ppv) if tp > 0 else 0)
    mt['_fmeasure2'] = 2.0 * tp / (2.0 * tp + fp + fn) if tp > 0 else 0
    mt['_fbs'] = mt['_fbeta'] = 0 if tp <= 0 else (1 + bb) * ppv * tpr / (bb * ppv + tpr)

    mt['_acc'] = mt['_accuracy'] = mt['_binary_accuracy'] = tp + tn
    mt['jcd'] = mt['jaccard'] = mt['tanimoto_similarity_coefficient'] = tp / (tp + fn + fp + _eps)

    mt['auc'] = (tpr + mt['tnr']) / 2.0
    mt['informedness'] = tpr + mt['tnr'] - 1
    mt['markedness'] = ppv + npv - 1

    return mt
