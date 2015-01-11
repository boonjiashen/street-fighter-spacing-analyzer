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
from label_CG import CG_fileIO


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


    #################### Calculate the confidence of predictions ##############

    # Define frames that we want to localize players 1
    frames = util.grab_frame(args.video_filename)
    frames = list(itertools.islice(frames, 1, 3))

    #################### Display output #######################################

    # Keys to control display
    WIN = 'Output'
    ESC = 27
    SPACEBAR = 32

    for fi, frame in enumerate(frames):

        windows, bbs = zip(*localizer.windowfy(frame))

        # Distance from decision boundary
        positivities = localizer.clf.decision_function(windows)

        # Get windows ranked by positiveness (first is most positively labeled)
        positivities, windows =  \
                zip(*sorted(zip(positivities, windows), reverse=True))
        positive_windows = util.tile(windows, desired_aspect=16/9)

        # Get windows ranked by uncertainty (first is most uncertain)
        uncertainties = -np.abs(positivities)  # higher no. = more uncertain
        uncertainties, windows =  \
                zip(*sorted(zip(uncertainties, windows), reverse=True))
        uncertain_windows = util.tile(windows, desired_aspect=16/9)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(positive_windows[:, :, ::-1])
        plt.subplot(1, 2, 2)
        plt.imshow(uncertain_windows[:, :, ::-1])
        plt.title(str(fi))

        continue
        cv2.imshow(WIN, ranking)
        key = cv2.waitKey()
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

    plt.show()
    cv2.destroyAllWindows()
