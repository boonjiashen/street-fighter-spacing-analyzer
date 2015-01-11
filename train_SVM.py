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
import BoundingBoxLabeler
import nms
from label_CG import CG_fileIO


class Hogger():
    "Transforms a window to a HoG representation"

    def __init__(self):
        self.hog_obj = cv2.HOGDescriptor()
        self.win_width, self.win_height = self.hog_obj.winSize

    def hogify(self, window):
        "Transform one window to its HoG representation"
        return self.hog_obj.compute(window,
                winStride=(8,8),
                padding=(0,0)).ravel()

    def transform(self, windows):
        return [self.hogify(window) for window in windows]

    def fit(self, *args, **kwargs):
        "Does no fitting"
        return self


class PlayerLocalizer:
    """Pipeline to localize a Street Fighter 4 player (i.e. character) from a
    video frame.
    """

    def __init__(self, windowfy=None, bb2label=None, clf=None):
        "Initialize functions in the pipeline"

        self.windowfy = windowfy  \
                if windowfy is not None  \
                else PlayerLocalizer.get_default_windowfy()

        self.bb2label = bb2label  \
                if bb2label is not None  \
                else PlayerLocalizer.get_default_bb2label()

        self.clf = clf  \
                if clf is not None  \
                else PlayerLocalizer.get_default_clf()

    def get_default_windowfy(decimation_factor = 5):
        """Get a function that generates sliding windows from a SF4 frame
        We construct our test set & training set with this

        `decimation_factor` is the ratio of window size to step size in each
        dimension.
        """
        win_size = np.array([Hogger().win_height, Hogger().win_width])
        return util.get_windowfier(win_size, win_size // decimation_factor)

    def get_default_bb2label():
        """Define a function that labels a window based on its position in the frame
        and the CG of the player
        This function should return True, False or None
        We construct the training set with this
        """
        is_pos = lambda rect, point:  \
                BoundingBoxLabeler.BoundingBoxLabeler.is_central(rect, point, 0.2)
        bb_labeler = lambda frame, CG:  \
            BoundingBoxLabeler.BoundingBoxLabeler.moat(
                    frame,
                    CG,
                    is_pos=is_pos,
                    )
        return bb_labeler

    def get_default_clf():
        "Classifier that maps windows to True/False"
        clf = sklearn.pipeline.Pipeline([
            ('HoG', Hogger()),
            ('SVM', sklearn.svm.LinearSVC())
            ])

        return clf

    def labeled_windows(self, frame, CG):
        "Extract labeled windows from one frame and one CG"
        for window, bb in self.windowfy(frame):
            label = self.bb2label(bb, CG)
            if label is not None:
                yield window, label

    def predict_bbs(self, frame):
        "Return a list of bounding boxes predicted to contain the player"

        windows, bbs = zip(*self.windowfy(frame))
        predictions = self.clf.predict(windows)

        return [bb
                for bb, prediction in zip(bbs, predictions)
                if prediction]


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

    im_displays = (draw_bbs(frame, filter_bbs(localizer.predict_bbs(frame)))
            for frame in frames)


    #################### Display output #######################################

    WIN = 'Output'
    ESC = 27
    SPACEBAR = 32
    for fi, frame in enumerate(im_displays):
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
