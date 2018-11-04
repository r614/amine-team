import LabelNames as ln
import submissions_test as sn
import pandas as pd
import fastai.conv_learner
import fastai.dataset
import numpy as np
import torch
import torchvision
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt
import tensorflow as tf


PATH = './'
pictures_folder = '../data_READONLY/train'
test_folder = '../data_READONLY/test/'
train_csv = '../data_READONLY/train.csv'
sample_csv = '../data_READONLY/sample_submission.csv'

labels = ln.labelnames()
nw = 2   #number of workers for data loader
arch = resnet34 #specify target architecture

pictures = sn.open_rgby(pictures_folder)

bs = 64
sz = 256
md = sn.get_data(sz, bs)

learner = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learner.opt_fn = optim.Adam
learner.clip = 1.0 #gradient clipping
learner.crit = FocalLoss()
learner.metrics = [acc]

learner.unfreeze()
lrs=np.array([lr/10,lr/3,lr])
learner.fit(lrs/4,4,cycle_len=2,use_clr=(10,20))
learner.fit(lrs/16,1,cycle_len=8,use_clr=(5,20))

def save_pred(pred, th=0.5, fname='protein_classification.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)

    sample_df = pd.read_csv(SAMPLE)
    sample_list = list(sample_df.Id)
    pred_dic = dict((key, value) for (key, value)
                in zip(learner.data.test_ds.fnames,pred_list))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(fname, header=True, index=False)

save_pred(pred)
