"""Find optimal no. of exemplars in video by elbow method
"""
import numpy as np
import cv2
import math
import util
import logging
import matplotlib.pyplot as plt
import nms
import random
import argparse
import scipy.stats
import time
import itertools

import sklearn.cluster
import sklearn.pipeline

from exemplar_frames import FrameExemplifier


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename')
    parser.add_argument("--log", type=str, default='INFO',
            help="Logging setting (e.g., INFO, DEBUG)")
    args = parser.parse_args()

    # Setting logging parameters
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(message)s')

    exemplifier = FrameExemplifier()

    # Load data
    start, stop, step = 0, 2700, 1
    #start, stop, step = None, None, None
    logging.info('Loading from %s (start frame=%s, end frame=%s, increment=%s)',
        args.input_filename, *map(str, [start, stop, step]))
    sample_inds = list(range(start, stop, step))
    all_frames = util.grab_frame(args.input_filename)
    frame_sample = (util.index(all_frames, sample_inds))

    # Flatten H channel of every item in sample
    H_rows = np.vstack(
            cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:,:,0].ravel()
            for im in frame_sample)

    n_exemplars_list = np.linspace(3, 50, 10, dtype=int)
    n_bins_list = np.linspace(10, 45, 5, dtype=int)
    scores = np.zeros((len(n_bins_list), len(n_exemplars_list), ))

    for j, n_bins in enumerate(n_bins_list):
        X = np.vstack(
                np.histogram(H, bins=n_bins, range=(0, 180.))[0]
                for H in H_rows)
        for i, n_exemplars in enumerate(n_exemplars_list):
            kmeans_obj = exemplifier.pipeline.steps[-1][-1]
            kmeans_obj.n_clusters = n_exemplars

            best_X_inds = exemplifier.from_features(X)
            score = exemplifier.pipeline.score(X)
            scores[j, i] = score

    plt.figure()
    for n_bins, score_list in zip(n_bins_list, scores):
        plt.plot(n_exemplars_list, score_list, label='nbins=%i'%(n_bins))
    #plt.imshow(scores, interpolation='nearest')
    plt.legend(loc='best')
    plt.xlabel('Number of clusters')
    plt.ylabel('KMeans score')
    plt.title('KMeans score versus #clusters and #bins')
    plt.show()

if __name__ == '__main__':

    main()
