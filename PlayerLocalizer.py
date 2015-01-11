"""Pipeline to localize a Street Fighter 4 player (i.e. character) from a
video frame.
"""

import sklearn.pipeline
import sklearn.svm
import util
import numpy as np
from Hogger import Hogger
import BoundingBoxLabeler

class PlayerLocalizer:

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


