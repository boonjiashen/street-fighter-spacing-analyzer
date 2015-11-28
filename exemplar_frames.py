"""Automatically detect examplar frames so that we can minimize the amount of
manual labelling we need to do
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

class FrameExemplifier():

    def __init__(self, num_examplars = 25):
        self.num_examplars = num_examplars
        kmeans = (sklearn.cluster.KMeans,
                {
                    'n_clusters': self.num_examplars,
                    #'n_jobs': -1,
                    'n_init': 10,
                    'max_iter': 100,
                })

        steps = [kmeans]
        self.pipeline = sklearn.pipeline.make_pipeline(
                *[fun(**kwargs) for fun, kwargs in steps])

        util.print_steps(steps)

    def im_BGR_to_features(self, im_BGR):
        im_HSV = cv2.cvtColor(im_BGR, cv2.COLOR_BGR2HSV)
        counts, bins = np.histogram(im_HSV, bins=45, range=(0, 180.))

        return counts, bins

    def from_BGRs(self, im_BGRs):
        """Returns the indices of the examplar of an iterator of BGR images

        Returns a n_examplars length array where each element is in [0, n)
        `im_BGRs` an iterator of color images in BGR space
        `n` length of im_BGRs
        """

        X = np.vstack(self.im_BGR_to_features(im)[0] for im in im_BGRs)

        logging.info('Loaded histograms')

        return self.from_features(X)

    def from_features(self, X):
        """Returns the indices of the examplar of dataset X

        Returns a n_examplars length array where each element is in [0, n)
        `X` a nxm matrix
        `n` number of features
        `m` no. of dimensions in a feature
        """
    
        ### TRAIN PIPELINE ###

        logging.info('Training model...')
        self.pipeline.fit(X)
        logging.info('done.')

        ### Get examplars from each cluster ###

        # Get index (in X) of example closest to cluster center, for each cluster
        cluster_inds = self.pipeline.predict(X)
        cluster_space = self.pipeline.transform(X)
        best_X_inds = []
        for cluster_ind in range(self.num_examplars):
            mask = cluster_inds == cluster_ind
            best_X_ind = np.flatnonzero(mask)[np.argmin(cluster_space[mask, cluster_ind])]
            best_X_inds.append(best_X_ind)

        return best_X_inds


def test():
    """Test that exemplifier returns index of the closest examplars
    in a simulated testset"""

    # Toy example
    n_clusters = 3
    n_examples_per_cluster = 50
    stdev = 0.2
    centers = [(0, 0), (0, 1), (1, 0)]
    X = np.vstack(
            np.random.normal(scale=stdev, size=(n_examples_per_cluster, 2)) + center
            for center in centers)

    exemplifier = FrameExemplifier(n_clusters)
    exemplar_inds = exemplifier.from_features(X)

    plt.figure()
    fig_title = "X at %s" % time.asctime(time.localtime())
    plt.gcf().canvas.set_window_title(fig_title)
    predictions = exemplifier.pipeline.predict(X)
    for cluster_ind, (color, exemplar_ind) in enumerate(zip('rgb', exemplar_inds)):
        mask = predictions == cluster_ind
        mask[exemplar_ind] = False  # don't display the exemplar
        plt.plot(X[mask, 0], X[mask, 1], color+'o', ms=7)
        plt.plot(X[exemplar_ind, 0], X[exemplar_ind, 1], color+'o', ms=9,
                mew=2, label=color+' examplar')

    plt.axis('equal')
    plt.title('Examples and their predictions')
    plt.legend(loc='best')
    plt.show()


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

    n_exemplars = 25
    exemplifier = FrameExemplifier(n_exemplars)

    # Load data
    start, stop, step = 0, 2700, 1
    #start, stop, step = None, None, None
    logging.info('Loading from %s (start frame=%s, end frame=%s, increment=%s)',
        args.input_filename, *map(str, [start, stop, step]))
    sample_inds = list(range(start, stop, step))
    all_frames = util.grab_frame(args.input_filename)
    frame_sample = (util.index(all_frames, sample_inds))

    best_X_inds = exemplifier.from_BGRs(frame_sample)

    # Map index in frame_sample to index in the input video
    frame_inds = np.array(sample_inds)[best_X_inds]
    #frame_inds = np.linspace(0, 2700, 25, dtype=int)
    frame_inds.sort()
    all_frames = util.grab_frame(args.input_filename)
    im_exemplars = list(util.index(all_frames, frame_inds))

    # Display exemplar histograms in one plot
    plt.figure()
    fig_title = "Examplar histograms at %s" % time.asctime(time.localtime())
    for im_exemplar in im_exemplars:
        counts, bins = exemplifier.im_BGR_to_features(im_exemplar)
        plt.plot(bins[:-1], counts)

    num_subplot_rows = math.ceil(n_exemplars**.5)
    plt.figure()
    fig_title = "Examplar images at %s" % time.asctime(time.localtime())
    plt.gcf().canvas.set_window_title(fig_title)
    for i, (frame_ind, im_exemplar) in enumerate(zip(frame_inds, im_exemplars)):
        plt.subplot(num_subplot_rows, num_subplot_rows, i + 1)
        plt.imshow(im_exemplar[:,:,::-1], interpolation='nearest')
        plt.xticks(())  # remove ticks
        plt.yticks(())
        plt.title("Frame #%i" % (frame_ind))
    plt.tight_layout()
    plt.show()

    logging.info('Exemplar frame indices are %s', str(frame_inds))
    return

    ### DISPLAY OUTPUT ###

    WIN = 'Output'
    ESC = 27
    SPACEBAR = 32
    for fi, frame in enumerate(im_HSVs):

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(30)
        if key == ESC:
            break

        # Spacebar pauses video, after while ESC exits video or spacebar
        # resumes. Other keystrokes are ignored during pause.
        elif key == SPACEBAR:
            key = cv2.waitKey()
            while key != SPACEBAR and key != ESC:
                key = cv2.waitKey()
            if key == SPACEBAR:
                continue
            else:
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':

    #test()
    main()
