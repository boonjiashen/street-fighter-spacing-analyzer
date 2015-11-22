"""Train SVM using HOG as features
"""
import numpy as np
import cv2
import math
import util
import itertools
import collections
import logging
import matplotlib.pyplot as plt
import nms
import random
import argparse
from PlayerLocalizer import PlayerLocalizer
from CG_fileIO import CG_fileIO


import CoatesScaler
import ZCA
import pickle

import sklearn.datasets
import sklearn.cluster
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.pipeline
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


def filter_bbs(bbs):
    """Filter a list of bounding boxes to reduce no. of overlapping windows.
    
    Returns a list of tuples

    `windows` is a list of tuples
    """
    if not bbs:
        return []

    overlap_thresh = 0.2  # threshold for non max suppression
    subset = nms.non_max_suppression_slow(
            np.vstack(bbs), overlap_thresh)
    return [tuple(x) for x in subset]


def main():

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('video_filename')
    parser.add_argument('CG_filename',
        help='File that contains center-of-gravity info of each frame')
    parser.add_argument('-o', '--output_filename',
        help='Filename of video to be saved (default: does not save)')
    args = parser.parse_args()


    #################### Define classification pipeline ######################

    windowfy = PlayerLocalizer.get_default_windowfy(decimation_factor=5)
    localizer = PlayerLocalizer(windowfy=windowfy)


    #################### Load CG of players ###################################

    # Create a function that maps the frame index of a video to a
    # (p1's CG, p2's CG) 2-ple
    CGs = CG_fileIO.load(args.CG_filename)

    # Drop p2's CG to make mapping simpler
    # Throw away mappings to None
    PLAYER_INDEX = 0
    CGs = dict((frame_index, CG_tuple[PLAYER_INDEX])
            for frame_index, CG_tuple in CGs.items()
            if CG_tuple[PLAYER_INDEX] is not None)


    #################### Construct labeled dataset ############################

    # Get all frames up till and including the last labeled frame
    last_frame_index = max(CGs.keys())  # index of last labeled frame
    frames = itertools.islice(
            util.grab_frame(args.video_filename),
            last_frame_index + 1
            )

    # Get only frames that are labeled
    frames_and_CGs = ((frame, CGs[fi])
            for fi, frame in enumerate(frames)
            if fi in CGs.keys())

    # Extract labeled windows from frames
    windows_and_labels = ((window, label)
        for frame, CG in frames_and_CGs
        for window, label in localizer.labeled_windows(frame, CG)
        )
    X, y = zip(*windows_and_labels)

    try:
        logging.info("Constructed labeled dataset of {0} instances".format(len(X)))
        n_pos = sum(y)
        n_neg = len(y) - n_pos
        logging.info("{0} positive instances".format(n_pos))
        logging.info("{0} negative instances".format(n_neg))
    except TypeError:
        logging.info("Constructed labeled dataset")


    #################### Display training dataset #############################

    if False:

        for desired_label in [True, False]:
            windows = [x for x, label in zip(X, y) if label==desired_label]
            canvas = util.tile(windows, desired_aspect=16/9)
            plt.figure()
            plt.imshow(canvas[:,:,::-1], interpolation="nearest")

        plt.show()
        assert False


    #################### Learn SVM ############################################

    localizer.clf.fit(X, y)

    logging.info("Learnt classifier")


    #################### Predict unlabeled dataset ############################

    def draw_bbs(frame, bbs):
        """Return a copy a frame with bounding boxes drawn
        
        `bbs` is a list of bounding boxes. Each bounding box is a (xTL, yTL,
        xBR, yBR) 4-tuple.
        """

        im_display = frame.copy()
        for bb in bbs:
            cv2.rectangle(im_display, bb[:2], bb[-2:], (0, 0, 0), 3)

        return im_display 

    # Define frames that we want to localize players 1 and 2 in
    frames = util.grab_frame(args.video_filename)
    #frames = itertools.islice(frames, 0, 100)

    im_displays = (
            draw_bbs(frame, filter_bbs(localizer.predict_bbs(frame, allow_empty=False)))
            for frame in frames)


    #################### Display output #######################################

    save_video = args.output_filename is not None
    if save_video:
        cap = cv2.VideoCapture(args.video_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        sz = tuple(map(int, [w, h]))  # NOTE! order is width then height
        cap.release()
        codec = cv2.VideoWriter_fourcc(*'MJPG')

        out = cv2.VideoWriter(args.output_filename, codec, fps, sz)

        print('Saving to video', args.output_filename)

    WIN = 'Output'
    ESC = 27
    SPACEBAR = 32
    for fi, frame in enumerate(im_displays):

        if save_video:
            #print('writing', fi)
            #out.write(np.zeros((360, 640, 3), dtype=np.uint8))
            out.write(frame)

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
    if save_video:
        out.release()
        print('Saved to video', args.output_filename)


def save_frames():
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('video_filename')
    parser.add_argument('n_frames', type=int)
    args = parser.parse_args()

    for ind, frame in enumerate(util.grab_frame(args.video_filename)):
        print("Saved frame ", ind)
        cv2.imwrite(str(ind) + '.png', frame)
        if ind >= args.n_frames:
            break


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

    # Load data
    d = 6  # size of patch
    im = cv2.imread('data/0.png')
    im_height, im_width = im.shape[:2]
    all_patch_rows =  np.array(list(
            patch.ravel()
            for patch in util.yield_windows(im, (d, d), (1, 1))
            ))
    logging.info('Loaded dataset of size %i', len(all_patch_rows))

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
    kmeans = (sklearn.cluster.KMeans,
            {
                'n_clusters': 100,
                #'n_jobs': -1,
                'n_init': 1,
                'max_iter': 10,
            })

    # Define pipeline
    steps = [coates_scaler, zca, kmeans]
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

    y = pipeline.predict(all_patch_rows)
    newshape = (im_height - d + 1, im_width - d + 1, )
    segmentation = np.reshape(y, newshape)
    plt.figure;
    plt.gcf().canvas.set_window_title('Segmentation')
    plt.imshow(segmentation, interpolation='nearest')
    plt.show()

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

