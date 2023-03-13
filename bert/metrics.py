import numpy as np
# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2022-04-26 17:01:39
LastEditTime: 2022-04-27 10:40:29
Description: 
'''


def average_precision(gt, pred):
  """
  Computes the average precision.

  This function computes the average prescision at k between two lists of
  items.

  Parameters
  ----------
  gt: set
       A set of ground-truth elements (order doesn't matter)
  pred: list
        A list of predicted elements (order does matter)

  Returns
  -------
  score: double
      The average precision over the input lists
  """

  if not gt:
    return 0.0

  score = 0.0
  num_hits = 0.0
  for i,p in enumerate(pred):
    if p in gt and p not in pred[:i]:
      num_hits += 1.0
      score += num_hits / (i + 1.0)

  return score / max(1.0, len(gt))


def NDCG(gt, pred_gt):
  score = 0.0
  for rank, item in enumerate(pred_gt):
    if item in gt:
      if rank == 0:
        score = item
      else:
        score += item / np.log2(rank + 1)

  norm = 0.0
  for rank, item in enumerate(gt):
    if item in gt:
      if rank == 0:
        norm = item
      else:
        norm += item / np.log2(rank + 1)

  if norm == 0.0:
    return 1
  else:
    return score / norm


def metrics(gt, pred, pred_gt, metrics_map):
  '''
  Returns a numpy array containing metrics specified by metrics_map.
  gt: ground-truth items
  pred: predicted items
  '''
  out = np.zeros((len(metrics_map),), np.float32)

  if ('MAP' in metrics_map):
    avg_precision = average_precision(gt=gt, pred=pred)
    out[metrics_map.index('MAP')] = avg_precision

  if ('RPrec' in metrics_map):
    intersec = len(gt & set(pred[:len(gt)]))
    out[metrics_map.index('RPrec')] = intersec / max(1., float(len(gt)))

  if 'MRR' in metrics_map:
    score = 0.0
    max_gt = int(max(pred_gt))
    for rank, item in enumerate(pred):
      if int(pred_gt[rank]) == max_gt:
        score = 1 / (rank + 1.0)
        break
    out[metrics_map.index('MRR')] = score

  if 'MRR@10' in metrics_map:
    score = 0.0
    max_gt = int(max(pred_gt[:10]))
    for rank, item in enumerate(pred[:10]):
      if int(pred_gt[rank]) == max_gt:
        score = 1 / (rank + 1.0)
        break
    out[metrics_map.index('MRR@10')] = score

  if 'MRR@20' in metrics_map:
    score = 0.0
    max_gt = int(max(pred_gt[:20]))
    for rank, item in enumerate(pred[:20]):
      if int(pred_gt[rank]) == max_gt:
        score = 1 / (rank + 1.0)
        break
    out[metrics_map.index('MRR@20')] = score

  if 'MRR@50' in metrics_map:
    score = 0.0
    max_gt = int(max(pred_gt[:50]))
    for rank, item in enumerate(pred[:50]):
      if int(pred_gt[rank]) == max_gt:
        score = 1 / (rank + 1.0)
        break
    out[metrics_map.index('MRR@50')] = score

  if ('NDCG' in metrics_map):
    out[metrics_map.index('NDCG')] = NDCG(gt, pred_gt)

  return out

if __name__ == '__main__':
  gt = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  pred_gt = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

  print(NDCG(gt, pred_gt))



