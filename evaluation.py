# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter

def nDCG(y_true,y_pred,k=5):
    y_true[y_true==-1] = 0
#    pred = copy.deepcopy(y_pred)
    ndcg = np.zeros(k)
    rank_mat = np.argsort(-y_pred)
    sumY = np.sum(y_true,1).ravel()
    Ypred = np.zeros([y_true.shape[0],k])
    normFac = np.zeros([y_true.shape[0],k])
    
    for i in range(k):
#        Jidx = np.array(pred.argmax(1)).ravel()
        Jidx = rank_mat[:,i]
        Iidx = np.array(range(len(Jidx)))
        lbls = y_true[Iidx,Jidx]
#        pred[Iidx,Jidx] = 0
        Ypred[:,i] = lbls/np.log2(2+i)
        sY = np.sum(Ypred[:,:i+1],1)
        
        
        normFac[:,i] = ( (np.float32(sumY>=i+1)+1e-12)/np.log2(2+i) )
        sF = np.sum(normFac[:,:i+1],1)
        ndcg[i] = np.sum(sY/sF)/y_pred.shape[0]
#        print np.sum(sY),sF.sum()
    return ndcg

def prec_at(y_true,y_pred,k):
    y_true[y_true==-1] = 0
#    pred = copy.deepcopy(y_pred)
    p = np.zeros(k)
    rank_mat = np.argsort(-y_pred)
    add = 0
    for i in range(k):

#        Jidx = np.array(pred.argmax(1)).ravel()
        Jidx = rank_mat[:,i]
        Iidx = np.array(range(len(Jidx)))
        lbls = y_true[Iidx,Jidx]
        add += lbls.sum()
        p[i] = add/(float(i+1)*len(Jidx))
#        pred[Iidx,Jidx] = 0
        
    return p

def patk(predictions, labels):
    pak = np.zeros(3)
    K = np.array([1, 3, 5])
    for i in range(predictions.shape[0]):
        pos = np.argsort(-predictions[i, :])
        y = labels[i, :]
    y = y[pos]
    for j in range(3):
        k = K[j]
        pak[j] += (np.sum(y[:k]) / k)
    pak = pak / predictions.shape[0]
    return pak * 100.

def cm_precision_recall(prediction,truth):
  """Evaluate confusion matrix, precision and recall for given set of labels and predictions
     Args
       prediction: a vector with predictions
       truth: a vector with class labels
     Returns:
       cm: confusion matrix
       precision: precision score
       recall: recall score"""
  confusion_matrix = Counter()

  positives = [1]

  binary_truth = [x in positives for x in truth]
  binary_prediction = [x in positives for x in prediction]

  for t, p in zip(binary_truth, binary_prediction):
    confusion_matrix[t,p] += 1

  cm = np.array([confusion_matrix[True,True], confusion_matrix[False,False], confusion_matrix[False,True], confusion_matrix[True,False]])
  #print cm
  precision = (cm[0]/(cm[0]+cm[2]+0.000001))
  recall = (cm[0]/(cm[0]+cm[3]+0.000001))
  return cm, precision, recall

def bipartition_scores(labels,predictions):
    """ Computes bipartitation metrics for a given multilabel predictions and labels
      Args:
        logits: Logits tensor, float - [batch_size, NUM_LABELS].
        labels: Labels tensor, int32 - [batch_size, NUM_LABELS].
      Returns:
        bipartiation: an array with micro_precision, micro_recall, micro_f1,macro_precision, macro_recall, macro_f1"""
    sum_cm=np.zeros((4))
    macro_precision=0
    macro_recall=0
    for i in range(labels.shape[1]):
        truth=labels[:,i]
        prediction=predictions[:,i]
        cm,precision,recall=cm_precision_recall(prediction, truth)
        sum_cm+=cm
        macro_precision+=precision
        macro_recall+=recall
    
    macro_precision=macro_precision/labels.shape[1]
    macro_recall=macro_recall/labels.shape[1]
    macro_f1 = 2*(macro_precision)*(macro_recall)/(macro_precision+macro_recall+0.000001)
    
    micro_precision = sum_cm[0]/(sum_cm[0]+sum_cm[2]+0.000001)
    micro_recall=sum_cm[0]/(sum_cm[0]+sum_cm[3]+0.000001)
    micro_f1 = 2*(micro_precision)*(micro_recall)/(micro_precision+micro_recall+0.000001)
    bipartiation = np.asarray([micro_precision, micro_recall, micro_f1,macro_precision, macro_recall, macro_f1])
    return bipartiation
