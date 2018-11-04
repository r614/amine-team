import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
# # Source: kickback
# # "I think this line is needed if you are running the kernel online as a fork. I just comment out lines like this because I run all of my code on a standalone machine."
# %matplotlib inline
from PIL import Image
from scipy.misc import imread

import tensorflow as tf
sns.set()

import os
import _pickle as cPickle

import LabelNames
import ModelParam

DATA_FOLDER = "../data_READONLY/"
TRAIN_FILE = "train.csv"
TRAIN_PATH = DATA_FOLDER + TRAIN_FILE
IMAGE_FILE = "test"
IMAGE_PATH = DATA_FOLDER + IMAGE_FILE

PICKLE_TRAIN_FILE = "train.pkl"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Read the processed dataset, and cache the object for faster retrieval
def read_file(reload=False):
    if not reload:
        if os.path.isfile(PICKLE_TRAIN_FILE):
            with open(PICKLE_TRAIN_FILE, 'rb') as input:
                train_labels = cPickle.load(input)
                print("Loaded train_labels")
                return train_labels

    # Import csv with Pandas
    train_labels = pd.read_csv(TRAIN_PATH)
    # Display first few rows of data 
    print(train_labels.head())

    # Run through labels in LabelNames.py, and set the corresponding training label to 0
    for key in LabelNames.label_names.keys():
        train_labels[LabelNames.label_names[key]] = 0

    train_labels = train_labels.apply(LabelNames.fill_targets, axis=1)

    with open(PICKLE_TRAIN_FILE, 'wb+') as output:
        cPickle.dump(train_labels, output)
   
    print("Reloaded train_labels")
    return train_labels

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

train_files = os.listdir("../data_READONLY/train")
test_files = os.listdir("../data_READONLY/test")

def kFold_Cross(train_files, test_files):
    return np.round(len(test_files)/len(train_files) * 100)

def modelParam():
    return ModelParam.ModelParameter(TRAIN_PATH)
