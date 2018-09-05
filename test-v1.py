
import os
import string
import tempfile
import tensorflow as tf
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('agg')
import matplotlib.pyplot as plt

filenames = ["/home/asr/Data/classif_task/dev_data/jsgf_calendar"]

# Load all files from a directory in a DataFrame.
def load_dataset(directory):
    data = {}
    data["sentence"] = []
    data["class"] = []
    for file_name in os.listdir(directory):
        print(file_name)
        with tf.gfile.GFile(os.path.join(directory, file_name), "rb") as f:
            ll = [s.strip() for s in f.readlines()]
            for i in range(len(ll)):
                data["sentence"].append(ll[i])
                class_label = int(file_name[-1])
                data["class"].append(class_label)
    return pd.DataFrame.from_dict(data)

train_df = load_dataset('/home/asr/Data/classif_task/dev_data/')

print(train_df.head())
