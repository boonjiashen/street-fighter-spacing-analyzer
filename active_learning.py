"""Experiment with active learning to reduce manpower needed to label data
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
from PlayerLocalizer import PlayerLocalizer
from CG_fileIO import CG_fileIO


if __name__ == '__main__':

    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('video_filename')
    parser.add_argument('CG_filename',
        help='File that contains center-of-gravity info of each frame')
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


    #################### Learn SVM ############################################

    localizer.clf.fit(X, y)

    logging.info("Learnt classifier")


    #################### Display output #######################################

    # Keys to control display
    WIN = 'Output'
    ESC = 27
    SPACEBAR = 32

    # Get unlabeled frames
    frames = util.grab_frame(args.video_filename)
    indexed_frames = ((fi, frame)
            for fi, frame in enumerate(frames)
            if fi not in CGs.keys())

    def get_frame_uncertainty(localizer, frame):
        """Get the smallest distance from the decision boundary of all the
        windows of a frame, i.e. the uncertainty of the most uncertain
        classification.
        """

        windows = list(window for window, bb in localizer.windowfy(frame))
        uncertainties = -np.abs(localizer.clf.decision_function(windows))
        max_uncertainty = np.max(uncertainties)

        return max_uncertainty

    # Grab frames in batches and for each batch, ask the human to label the
    # frame with the most uncertain classification
    batch_size = 10
    first_batch = True
    for indexed_batch in util.chunks_of_size_n(indexed_frames, batch_size):

        # Get the frame with the highest
        fi, frame = max(indexed_batch,
                key=lambda x: get_frame_uncertainty(localizer, x[1]))

        # Display frame with frame index
        util.put_text(frame, str(fi))

        # Pre-compute optimal frame while user is labeling
        if not first_batch:
            key = cv2.waitKey()
            if key == ESC:
                break
        else:
            first_batch = False

        cv2.imshow(WIN, frame)
        cv2.waitKey(10)  # Without this line, imshow doesn't show immediately

    cv2.destroyAllWindows()
