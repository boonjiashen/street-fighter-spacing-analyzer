"""Train SVM using HOG as features
"""
import numpy as np
import cv2
import math
import util
import logging
import nms
import random
import argparse
import scipy.stats
import time
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import CoatesScaler
import ZCA
import SphericalKMeans

import sklearn.cluster
import sklearn.pipeline

def save_video_with_mpl(im_displays, output_filename):
    dpi = 100
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
            comment='Movie support!')
    writer = FFMpegWriter(fps=30, metadata=metadata)
    fig = plt.figure()

    logging.info('Saving to video %s', output_filename)
    with writer.saving(fig, output_filename, dpi):

        frame = next(im_displays)
        ax = plt.imshow(frame, interpolation='nearest')
        for fi, frame in enumerate(im_displays, 1):

            util.put_text(frame, str(fi))
            ax.set_data(frame)
            writer.grab_frame()
            logging.info('Grabbed frame %i', fi)

    logging.info('Saved to video %s', output_filename)

def cluster_frames():

    seed = 0
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename')
    parser.add_argument("data_proportion", nargs='?', type=float, default=1.,
            help="Proportion of full dataset to be used")
    parser.add_argument("--log", type=str, default='INFO',
            help="Logging setting (e.g., INFO, DEBUG)")
    parser.add_argument('-o', '--output_filename',
        help='Filename of video to be saved (default: does not save)')
    args = parser.parse_args()

    # Setting logging parameters
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(message)s')

    sample_inds = [212, 699, 988, 1105, 2190, 2318]
    logging.info('Loading %i images... ', len(sample_inds))

    # Load data
    d = 6  # size of patch
    all_frames = util.grab_frame(args.input_filename)
    im_originals = list(util.index(all_frames, sample_inds))
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
    n_clusters = 100
    mbkmeans = (sklearn.cluster.MiniBatchKMeans,
            {
                'n_clusters': n_clusters,
                'batch_size': 3000,
            })
    skmeans = (SphericalKMeans.SphericalKMeans,
            {
                'n_clusters': n_clusters,
                'max_iter': 10,
            })
    kmeans = (sklearn.cluster.KMeans,
            {
                'n_clusters': n_clusters,
                #'random_state': np.random.RandomState,
                #'n_jobs': -1,
                #'n_init': 1,
                #'max_iter': 10,
            })

    # Define pipeline
    steps = [coates_scaler, zca, kmeans]
    pipeline = sklearn.pipeline.make_pipeline(
            *[fun(**kwargs) for fun, kwargs in steps])

    # Define pointers to certain steps for future processing
    whitener = pipeline.steps[1][1]  # second step
    dic = pipeline.steps[-1][1]  # last step

    steps = [(obj.__class__, obj.get_params()) for name, obj in pipeline.steps]
    util.print_steps(steps)


    ######################### Train pipeline ##################################

    logging.info('Training model...')
    pipeline.fit(X)
    logging.info('done.')

    ######################### Display atoms of dictionary #####################

    #def save_video_with_mpl(im_displays, output_fmt='output_data/%4i.jpg'):

    frames = util.grab_frame(args.input_filename)
    patch_row_chunks = (
            np.array(list(
            patch.ravel()
            for patch in util.yield_windows(im, (d, d), (1, 1))))
            for im in frames)

    def im_displays():
        for patch_rows in patch_row_chunks:
            y = pipeline.predict(patch_rows)

            # Map to [0, 1) so that imshow scales across entire colormap spectrum
            y = y / n_clusters

            newshape = (im_height - d + 1, im_width - d + 1, )
            segmentation = np.reshape(y, newshape)

            # Apply color map and remove alpha channel
            cmap = plt.cm.Set1
            colored_segmentation = cmap(segmentation)[:, :, :3]
            colored_segmentation = (colored_segmentation * 255).astype(np.uint8)

            yield colored_segmentation

    frames = itertools.islice(im_displays(), 5)
    save_video = args.output_filename is not None
    if save_video:
        save_video_with_mpl(frames, args.output_filename)

    """
    for fi, frame in enumerate(im_displays()):
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
    """

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

