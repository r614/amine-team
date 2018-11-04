##################### LOADING PACKAGES #######################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from scipy.misc import imread

import tensorflow as tf
sns.set()

import os
train_path = "../train.csv"
imagepath = "/data_READONLY/test"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

##################### KERNEL SETTINGS ############################

class KernelSettings:
    
    def __init__(self, fit_baseline=False, fit_improved_baseline=False):
        self.fit_baseline = fit_baseline
        self.fit_improved_baseline = fit_improved_baseline

# Reduce computation time by setting fit_baseline and/or fit_improved_baseline to false
# This way prediction probabilties of the corresponding models are loaded as csv from added data source
kernelsettings = KernelSettings(fit_baseline=True, fit_improved_baseline=False)

######################## LOADING DATA ######################

def read_file():
    train_labels = pd.read_csv("../data_READONLY/train.csv")
    train_labels.head()

    for key in label_names.keys():
        train_labels[label_names[key]] = 0

    train_labels = train_labels.apply(fill_targets, axis=1)
    train_labels.head()

def sample_size(train_labels):
    return train_labels.shape[0]

def find_counts(special_target, labels):
    counts = labels[labels[special_target] == 1].drop(
        ["Id", "Target", "number_of_targets"],axis=1
    ).sum(axis=0)
    counts = counts[counts > 0]
    counts = counts.sort_values()
    return counts

def load_image(imagepath, image_id):
    images = np.zeros(shape=(4,512,512))
    images[0,:,:] = imread(basepath + image_id + "_green" + ".png")
    images[1,:,:] = imread(basepath + image_id + "_red" + ".png")
    images[2,:,:] = imread(basepath + image_id + "_blue" + ".png")
    images[3,:,:] = imread(basepath + image_id + "_yellow" + ".png")
    return images

def make_image_row(image, subax, title):
    subax[0].imshow(image[0], cmap="Greens")
    subax[1].imshow(image[1], cmap="Reds")
    subax[1].set_title("stained microtubules")
    subax[2].imshow(image[2], cmap="Blues")
    subax[2].set_title("stained nucleus")
    subax[3].imshow(image[3], cmap="Oranges")
    subax[3].set_title("stained endoplasmatic reticulum")
    subax[0].set_title(title)
    return subax

def make_title(file_id):
    file_targets = train_labels.loc[train_labels.Id==file_id, "Target"].values[0]
    title = " - "
    for n in file_targets:
        title += label_names[n] + " - "
    return title

train_files = listdir("../data_READONLY/train.csv")
test_files = listdir("../test.csv")

def kFold_Cross(train_files, test_files):
    return np.round(len(test_files)/len(train_files) * 100)

def modelParam():
    return ModelParameter(trainpath)
