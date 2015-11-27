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


    # Load data
    start, stop, step = 0, 300, 2    
    #start, stop, step = None, None, None
    logging.info('Loading from %s (start frame=%s, end frame=%s, increment=%s)',
        args.input_filename, *map(str, [start, stop, step]))
    im_originals =  \
            (itertools.islice(util.grab_frame(args.input_filename), start, stop, step))
    logging.info('Loaded images')

    im_HSVs = (cv2.cvtColor(im_original, cv2.COLOR_BGR2HSV)
            for im_original in im_originals)
    Hhistograms = (np.histogram(im[:,:,0], range=(0, 180.))[0] for im in im_HSVs)

    X = np.vstack(Hhistograms)

    ### DEFINE PIPELINE ###

    n_clusters = 10
    kmeans = (sklearn.cluster.KMeans,
            {
                'n_clusters': n_clusters,
                #'n_jobs': -1,
                'n_init': 10,
                'max_iter': 10,
            })

    steps = [kmeans]
    pipeline = sklearn.pipeline.make_pipeline(
            *[fun(**kwargs) for fun, kwargs in steps])

    util.print_steps(steps)

    
    ### TRAIN PIPELINE ###

    logging.info('Training model...')
    pipeline.fit(X)
    logging.info('done.')


    ### Get examplars from each cluster ###

    # Get index (in X) of example closest to cluster center, for each cluster
    cluster_inds = pipeline.predict(X)
    cluster_space = pipeline.transform(X)
    best_X_inds = []
    for cluster_ind in range(n_clusters):
        mask = cluster_inds == cluster_ind
        best_X_ind = np.flatnonzero(mask)[np.argmin(cluster_space[mask, cluster_ind])]
        best_X_inds.append(best_X_ind)

    # Map index in X to index in the input video
    frame_inds = np.arange(start, stop, step)[best_X_inds]
    frame_inds.sort()
    last_frame_ind = np.asscalar(frame_inds[-1])  # islice below doesn't take np.int64
    im_examplars =  \
            (itertools.islice(util.grab_frame(args.input_filename), None, last_frame_ind+1))
    im_examplars = (im for frame_ind, im in enumerate(im_examplars) if frame_ind in frame_inds)

    for frame_ind, im_examplar in zip(frame_inds, im_examplars):
        plt.figure()
        plt.gcf().canvas.set_window_title(str(frame_ind))
        plt.imshow(im_examplar[:,:,::-1])
    plt.show()

    logging.info('Examplar frame indices are %s', str(frame_inds))
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

    main()

