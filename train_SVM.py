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


def yield_windows(image, window_size, step_size, yield_bb=False):
    """Yield windows of an image in regular intervals in row-major order.

    `image` - a 2D image

    `window_size` - required (height, width) of window

    `step_size` - (vertical_step, horizontal_step) 2-ple

    `yield_bb' - yields the bounding box of the window if True, i.e., yields a
    (window, (xTL, yTL, xBR, yBR)) tuple, where TL and BR are top-left and
    bottom-right of the window.
    """

    im_height, im_width = image.shape[:2]
    win_height, win_width = window_size
    y_step, x_step = step_size

    # y coord of TL of bottom-most window
    max_y_TL = (im_height - win_height) // y_step * y_step

    # x coord of TL of left-most window
    max_x_TL = (im_width - win_width) // x_step * x_step

    for y_TL in range(0, max_y_TL + 1, y_step):
        for x_TL in range(0, max_x_TL + 1, x_step):
            window = image[
                    y_TL:y_TL + win_height,
                    x_TL:x_TL + win_width]

            # Yield both the window and its coordinates
            if yield_bb:
                bb = (x_TL, y_TL, x_TL + win_width - 1, y_TL + win_height - 1)
                yield window, bb

            # Yield window only
            else:
                yield window


def contains(rectangle, point):
    """Checks if a point is in a rectangle.

    `point` = (x, y) 2-tuple

    `rectangle` = (xTL, yTL, xBR, yBR) 4-tuple
    """

    # Point is in rectangle in x-axis
    contains_x = (rectangle[0] <= point[0] <= rectangle[2])

    # Point is in rectangle in y-axis
    contains_y = (rectangle[1] <= point[1] <= rectangle[3])

    return contains_x and contains_y


if __name__ == '__main__':

    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('video_filename')
    parser.add_argument('CG_filename',
        help='File that contains center-of-gravity info of each frame')
    args = parser.parse_args()


    #################### Define classification pipeline ######################

    def get_windowfier(win_size, step_size=None):
        """Get a function that returns the windows and bounding boxes of a frame,
        given the desired window size

        `win_size` (height, width) tuple

        `step_size` (height, width) tuple. If none, this is half of win_size in
        both dimensions
        """

        # Define step size if it's undefined
        if step_size is None:
            step_size = np.array(win_size) // 2

        def windowfy(frame):
            return yield_windows(frame, win_size, step_size, yield_bb=True)

        return windowfy

    # Get a function that shatters frames into windows
    win_size = np.array([Hogger().win_height, Hogger().win_width])
    windowfy = get_windowfier(win_size)

    # Classifier that maps windows to True/False
    clf = sklearn.pipeline.Pipeline([
        ('HoG', Hogger()),
        ('SVM', sklearn.svm.LinearSVC())
        ])


    #################### Load CG of players ###################################

    # Create a function that maps the frame index of a video to a
    # (p1's CG, p2's CG) 2-ple
    CGs = CG_fileIO.load(args.CG_filename)

    # Drop p2's CG to make mapping simpler
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

    # Shatter frames into windows
    # Labeled training instances for p1
    # X is a list of windows
    # y is a list of True/False
    windows_and_labels = ((window, contains(bb, CG))
            for frame, CG in frames_and_CGs
            for window, bb in windowfy(frame)
            #for original_window, bb in windowfy(frame)
            #for window in [original_window, np.fliplr(original_window)]
            )
    X, y = zip(*windows_and_labels)

    try:
        logging.info("Constructed labeled dataset of {0} instances".format(len(X)))
    except TypeError:
        logging.info("Constructed labeled dataset")


    #################### Display positive instances ###########################

    if False:
        pos_windows = [x for x, label in zip(X, y) if label]
        canvas = util.tile(pos_windows, desired_aspect=16/9)
        plt.figure()
        plt.imshow(canvas[:,:,::-1])

        plt.show()
        #assert False


    #################### Learn SVM ############################################

    clf.fit(X, y)

    logging.info("Learnt classifier")


    #################### Predict unlabeled dataset ############################

    def predict_bbs(frame):
        "Given a frame, predict which bounding boxes are True."

        windows, bbs = zip(*windowfy(frame))
        predictions = clf.predict(windows)

        return [bb
                for bb, prediction in zip(bbs, predictions)
                if prediction]

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

    im_displays = (draw_bbs(frame, predict_bbs(frame))
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
