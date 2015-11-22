"""Train SVM using HOG as features
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

import CoatesScaler
import ZCA
import SphericalKMeans

import sklearn.cluster
import sklearn.pipeline


def cluster_frames():

    parser = argparse.ArgumentParser()
    parser.add_argument("data_proportion", nargs='?', type=float, default=1.,
            help="Proportion of full dataset to be used")
    parser.add_argument("--log", type=str, default='INFO',
            help="Logging setting (e.g., INFO, DEBUG)")
    args = parser.parse_args()

    # Setting logging parameters
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(message)s')

    filenames = ['data/' + str(i) + '.png' for i in range(5)];
    logging.info('Loading %i images... ', len(filenames))

    # Load data
    d = 6  # size of patch
    im_originals = [cv2.imread(filename) for filename in filenames]
    im_height, im_width = im_originals[0].shape[:2]
    all_patch_rows =  np.array(list(
            patch.ravel()
            for im in im_originals
            for patch in util.yield_windows(im, (d, d), (1, 1))
            ))
    num_rows_per_im = len(all_patch_rows) // len(im_originals)
    num_im = len(im_originals)
    logging.info('Loaded %i examples from %i images',
        len(all_patch_rows),
        len(im_originals))

    # Randomly sample a subset of the data
    sample_size = int(args.data_proportion * len(all_patch_rows))
    inds = np.random.choice(len(all_patch_rows), sample_size)
    X = all_patch_rows[inds]
    logging.info('Sampled %.1f%% of dataset = %i', 100 * args.data_proportion,
        sample_size)

    ############################# Define pipeline #############################    

    std_scaler = (sklearn.preprocessing.StandardScaler, {})
    coates_scaler = (CoatesScaler.CoatesScaler, {})
    pca = (sklearn.decomposition.PCA,
            {'whiten':True, 'copy':True}
            )
    zca = (ZCA.ZCA, {'regularization': .1})
    mbkmeans = (sklearn.cluster.MiniBatchKMeans,
            {
                'n_clusters': 100,
                'batch_size': 3000,
            })
    n_clusters = 100
    skmeans = (SphericalKMeans.SphericalKMeans,
            {
                'n_clusters': n_clusters,
                'max_iter': 10,
            })
    kmeans = (sklearn.cluster.KMeans,
            {
                'n_clusters': n_clusters,
                #'n_jobs': -1,
                'n_init': 1,
                'max_iter': 10,
            })

    # Define pipeline
    steps = [coates_scaler, zca, skmeans]
    pipeline = sklearn.pipeline.make_pipeline(
            *[fun(**kwargs) for fun, kwargs in steps])

    # Define pointers to certain steps for future processing
    whitener = pipeline.steps[1][1]  # second step
    dic = pipeline.steps[-1][1]  # last step

    util.print_steps(steps)


    ######################### Train pipeline ##################################

    logging.info('Training model...')
    pipeline.fit(X)
    logging.info('done.')

    ######################### Display atoms of dictionary #####################

    frames = util.grab_frame('data/infil.mp4')
    patch_row_chunks = (
            np.array(list(
            patch.ravel()
            for patch in util.yield_windows(im, (d, d), (1, 1))))
            for im in frames)

    do_equalize = False
    if do_equalize:
        # Define histogram equalization based on training data
        logging.info('Computing histogram equalizer...')
        counts, bins = np.histogram(pipeline.predict(all_patch_rows), range(n_clusters + 1))
        cumhist = np.cumsum(counts)
        equalize = lambda x: cumhist[x] / cumhist[-1]
        logging.info('done.')

    def im_displays(do_equalize=False):
        for patch_rows in patch_row_chunks:
            y = pipeline.predict(patch_rows)

            # Map to [0, 1) so that imshow scales across entire colormap spectrum
            if do_equalize:
                y = equalize(y)
            else:
                y = y / n_clusters

            newshape = (im_height - d + 1, im_width - d + 1, )
            segmentation = np.reshape(y, newshape)

            # Apply color map and remove alpha channel
            cmap = plt.cm.Set1
            colored_segmentation = cmap(segmentation)[:, :, :3]
            colored_segmentation = (colored_segmentation * 255).astype(np.uint8)

            yield colored_segmentation


    """
    y = pipeline.predict(all_patch_rows)

    # To consistently get similar clustering labels across runs,
    # we arrange clusters such that lower cluster indices have higher counts
    logging.info('Sorting cluster label by cluster size...')
    counts, bins = np.histogram(y, range(n_clusters + 1))
    rank = scipy.stats.rankdata(counts, method='ordinal').astype(int) - 1
    y = rank[y]
    logging.info('done.')

    # Now to get very distinct colors for each cluster, we do histogram
    # equalization
    logging.info('Equalizing histogram of segmentation image...')
    counts, bins = np.histogram(y, range(n_clusters + 1))
    cumhist = np.cumsum(counts)
    y = cumhist[y] / cumhist[-1]
    logging.info('done.')

    for i, segmentation in enumerate(segmentations, 1):

        plt.figure();
        fig_title = 'im ' + str(i) + " " + time.asctime(time.localtime())
        plt.gcf().canvas.set_window_title(fig_title)
        plt.imshow(segmentation, cmap=plt.cm.Set1, interpolation='nearest')
    
    plt.show()
    """

    WIN = 'Output'
    ESC = 27
    SPACEBAR = 32
    for fi, frame in enumerate(im_displays(do_equalize=do_equalize)):

        util.put_text(frame, str(fi))
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


    return

    logging.info('Displaying atoms of dictionary')

    # Inverse whiten atoms of dictionary
    atom_rows = dic.cluster_centers_ 
    if hasattr(whitener, 'inverse_transform'):
        atom_rows = whitener.inverse_transform(atom_rows)  

    plt.figure()
    for i, atom_row in enumerate(atom_rows):
        patch = atom_row.reshape(d, d, -1)[::-1]
        plt.subplot(10, 10, i + 1)
        plt.imshow(patch, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('Atoms of dictionary learnt from %i patches by %s' %  \
            (len(atom_rows), dic.__class__.__name__))

    plt.figure()
    displayed_patches = X[np.random.choice(len(X), 100)]
    for i, patch in enumerate(displayed_patches):
        plt.subplot(10, 10, i + 1)
        plt.imshow(patch.reshape([d, d, -1])[:,:,::-1], interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.show()

if __name__ == '__main__':

    cluster_frames()

