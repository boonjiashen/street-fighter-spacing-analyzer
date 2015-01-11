"""Transforms a window to a HoG representation

Compatible with sklearn Pipeline objects
"""

import cv2

class Hogger():

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


