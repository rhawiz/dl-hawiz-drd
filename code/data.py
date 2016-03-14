import cv2
import csv
import os
import numpy as np
from glob import glob
import pandas as pd


def load_data(folder_path, labels_path, verbose=0, limit=-1, size=256):
    X = []
    y = []
    labels = {}
    with open(labels_path, 'rb') as csvfile:
        for row in csv.reader(csvfile):
            labels[row[0]] = row[1]
    count = 0
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if verbose == 1:
                print "{} Loading image {}".format(count, file)
            if verbose == 2:
                if not count % 100:
                    print "{}. Loading image '{}'".format(count, file)
            if verbose == 3:
                if not count % 1000:
                    print "{}. Loading image '{}'".format(count, file)
            count += 1
            path = os.path.join(subdir, file)
            if '.' not in file:
                continue

            label, file_type = file.split('.')

            img = cv2.imread(path)

            try:
                y.append(labels[label])
                X.append(img)
            except KeyError:
                continue
            if count == limit:
                break

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)

    X = X.reshape(
        -1,  # number of samples, -1 makes it so that this number is determined automatically
        3,
        size,  # first image dimension (vertical)
        size,  # second image dimension (horizontal)
    )

    return X, y