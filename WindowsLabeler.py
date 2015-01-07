"""Methods to generate windows and their labels given a frame of a
match and the CG of a player.
"""
import cv2
import util

class WindowsLabeler():

    def __init__(self, windowfy):
        """`windowfy` generates a tuple of (window, bb) given a frame,
        where `bb` is a bounding box, a (xTL, yTL, xBR, yBR) tuple
        """
        self.windowfy = windowfy

    def one_vs_all(self, frame, CG):
        """Window is positive if it contains the CG, negative otherwise
        """

        for window, bb in self.windowfy(frame):
            yield window, util.contains(bb, CG)

    def moat(self, frame, CG):
        """Labels the central window as positive, windows beyond a 'moat'
        around the CG as negative, and skips all other windows.

        Label the window as positive if the CG is in it
        Label the window as negative if it's far enough from the CG
        Otherwise skip the window
        """

        # Define how far a window has to be from the CG to be labeled negative
        radius = 100
        x, y = CG
        box_around_CG = (x - radius, y - radius, x + radius, y + radius)
        bb_far_from_CG = lambda bb: not util.overlaps(bb, box_around_CG)

        for window, bb in self.windowfy(frame):
            if util.contains(bb, CG):
                yield (window, True)
            elif bb_far_from_CG(bb):
                yield (window, False)