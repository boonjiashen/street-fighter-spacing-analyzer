"""Methods to label a window (a crop of a video frame) given its bounding box
and the center-of-gravity of the player of interest.
"""
import cv2
import util

class BoundingBoxLabeler():
    """
    `bb` is a bounding box, a (xTL, yTL, xBR, yBR) tuple

    `CG` is a (x, y) tuple

    Labelers return:
    - True for a positive window,
    - False for a negative window and
    - None for a window that should be discarded.
    """

    def one_vs_all(bb, CG):
        """Window is positive if it contains the CG, negative otherwise
        """
        return util.contains(bb, CG)


    def is_central(rect, point, percentage):
        """Returns True if a point is somewhere in the center of a rectangle

        `percentage` percentage per dimension of the rectangle that is
        considered central.

        `rect` = (xTL, yTL, xBR, yBR) tuple

        `point` = (x, y) tuple
        """

        # Define a rectangle that's the center of the input rectangle
        x1, y1, x2, y2 = rect
        h, w = y2 - y1, x2 - x1
        central_rect = (
                x1 + w * (.5 - percentage/2),
                y1 + h * (.5 - percentage/2),
                x1 + w * (.5 + percentage/2),
                y1 + h * (.5 + percentage/2),
                )
        central_rect = tuple(map(int, central_rect))

        return util.contains(central_rect, point)


    def moat(bb, CG, is_pos=util.contains):
        """Labels the central window as positive, windows beyond a 'moat'
        around the CG as negative, and skips all other windows.

        Label the window as positive if the CG is in it
        Label the window as negative if it's far enough from the CG
        Otherwise skip the window

        `is_pos` determines if a window is labeled positive. It should be a
        Boolean function that takes a rectangle and point as arguments.
        """

        # Define how far a window has to be from the CG to be labeled negative
        radius = 100
        x, y = CG
        box_around_CG = (x - radius, y - radius, x + radius, y + radius)
        bb_far_from_CG = lambda bb: not util.overlaps(bb, box_around_CG)

        if is_pos(bb, CG):
            return True
        if bb_far_from_CG(bb):
            return False
        return None
