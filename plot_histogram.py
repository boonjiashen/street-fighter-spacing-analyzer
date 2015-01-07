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
import sklearn.pipeline
import sklearn.svm
from label_CG import CG_fileIO
from train_SVM import Hogger



if __name__ == '__main__':

    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('video_filename')
    parser.add_argument('CG_filename',
        help='File that contains center-of-gravity info of each frame')
    args = parser.parse_args()


    #################### Define classification pipeline ######################

    # Get a function that shatters frames into windows
    win_size = np.array([Hogger().win_height, Hogger().win_width])
    windowfy = util.get_windowfier(win_size)


    #################### Load CG of players ###################################

    # Create a function that maps the frame index of a video to a
    # (p1's CG, p2's CG) 2-ple
    CGs = CG_fileIO.load(args.CG_filename)


    #################### Construct labeled dataset ############################

    # Get all frames up till and including the last labeled frame
    last_frame_index = max(CGs.keys())  # index of last labeled frame
    frames = itertools.islice(
            util.grab_frame(args.video_filename),
            last_frame_index + 1
            )

    # Extract windows that contain p1 and p2 CGs
    p1_p2_windows = [[], []]
    for fi, frame in enumerate(frames):

        if fi not in CGs.keys(): continue

        for player_index, CG in enumerate(CGs[fi]):

            if CG is None: continue
            for window, bb in windowfy(frame):
                if util.contains(bb, CG):
                    p1_p2_windows[player_index].append(window)

    for pi, windows in enumerate(p1_p2_windows, 1):
        logging.info('Player %i has %i instances', pi, len(windows))


    ##################### Display training dataset #############################

    for windows in p1_p2_windows:
        canvas = util.tile(windows, desired_aspect=16/9)
        plt.figure()
        plt.imshow(canvas[:,:,::-1], interpolation="nearest")

    plt.show()
