import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
<<<<<<< HEAD
# # Source: kickback
# # "I think this line is needed if you are running the kernel online as a fork. I just comment out lines like this because I run all of my code on a standalone machine."
# %matplotlib inline
=======
%matplotlib inline
>>>>>>> 36e12efbd8cb2a7de6ae5a5ab982b413d2b2b9a7
from PIL import Image
from scipy.misc import imread

import tensorflow as tf
sns.set()

import os

<<<<<<< HEAD
import LabelNames
import ModelParam

DATA_FOLDER = "..\\data_READONLY\\"
TRAIN_FILE = "train.csv"
TRAIN_PATH = DATA_FOLDER + TRAIN_FILE
IMAGE_FILE = "test"
IMAGE_PATH = DATA_FOLDER + IMAGE_FILE
=======
data_folder = "../data_READONLY/"
train_file = "train.csv"
train_path = data_folder + train_file
image_file = "test"
image_path = data_folder + image_file
>>>>>>> 36e12efbd8cb2a7de6ae5a5ab982b413d2b2b9a7


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def read_file():
    # Import csv with Pandas
<<<<<<< HEAD
    train_labels = pd.read_csv(TRAIN_PATH)
    # Display first few rows of data 
    train_labels.head()

    # Run through labels in LabelNames.py, and set the corresponding training label to 0
    for key in LabelNames.label_names.keys():
        train_labels[LabelNames.label_names[key]] = 0

    
    train_labels = train_labels.apply(LabelNames.fill_targets, axis=1)
=======
    train_labels = pd.read_csv(train_path)
    # Display first few rows of data 
    train_labels.head()

    for key in label_names.keys():
        train_labels[label_names[key]] = 0

    train_labels = train_labels.apply(fill_targets, axis=1)
>>>>>>> 36e12efbd8cb2a7de6ae5a5ab982b413d2b2b9a7
    train_labels.head()

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
<<<<<<< HEAD
    return ModelParam.ModelParameter(trainpath)
=======
    return ModelParameter(trainpath)
>>>>>>> 36e12efbd8cb2a7de6ae5a5ab982b413d2b2b9a7
