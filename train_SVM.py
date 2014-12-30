"""Train SVM using HOG as features
"""
import numpy as np
import cv2
import math
import util
import itertools
import collections
import matplotlib.pyplot as plt
from train_CG import CG_fileIO

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

    parser = argparse.ArgumentParser()
    parser.add_argument('video_filename')
    parser.add_argument('CG_filename',
        help='File that contains center-of-gravity info of each frame')
    args = parser.parse_args()


    #################### Load CG of players ###################################

    # Create a function that maps the frame index of a video to a
    # (p1's CG, p2's CG) 2-ple
    CGs = CG_fileIO.load(args.CG_filename)

    # Make function return (None, None) when frame index is absent in
    # dictionary
    CGs = collections.defaultdict(lambda: (None, None), CGs)


    #################### Construct labeled dataset ############################

    hog_obj = cv2.HOGDescriptor()
    win_width, win_height = hog_obj.winSize

    # Compute descriptors of an image, where window strides are taken in row
    # major order
    hogify = lambda x: hog_obj.compute(x, winStride=(8,8), padding=(0,0))

    # Frames of SF4 match
    frames = util.grab_frame(args.video_filename)
    frames = itertools.islice(frames, 0, 200)

    # Labeled training instances for p1 and p2
    # X is a list of feature vectors
    # y is a list of True/False
    X_p1, X_p2, y_p1, y_p2 = [], [], [], []

    windows = []

    # Shatter frames into windows
    win_size = (win_height, win_width)  # Window size based on HoG descriptor
    step_size = (win_height // 2, win_width // 2)  # Step size is half window
    for fi, frame in enumerate(frames):
        for window, bb in yield_windows(
                frame, win_size, step_size, yield_bb=True):

            p1_CG, p2_CG = CGs[fi] # Get center of gravity of p1 and p2
            hog = hogify(window).ravel() # Get HoG of this window

            # Process label for p1
            for X, y, CG in [(X_p1, y_p1, p1_CG), (X_p2, y_p2, p2_CG)]:
                X.append(hog)
                label = False if CG is None else contains(bb, CG)
                y.append(label)

            windows.append(window)


    #################### Display positive and negative instances ##############

    for y, player in [(y_p1, 'p1'), (y_p2, 'p2')]:
        pos_windows = (window
                for window, is_player in zip(windows, y)
                if is_player)
        canvas = util.tile(pos_windows, desired_aspect=16/9)
        plt.figure()
        plt.imshow(canvas[:,:,::-1])

    plt.show()

    ## Select every N frames
    #step, n_frames = 15, 40
    #frames = itertools.islice(frames, 0, n_frames * step, step)

    ## Select every N windows
    #step, n_windows = 100, 1000
    #windows = itertools.islice(windows, 0, n_windows * step, step)


    #hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    #while(1):
        #ret, img = cap.read()

        ##found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
        #found, _ = hog.detect(img, winStride=(8,8), padding=(32,32))
        #found_filtered = []
        #for ri, r in enumerate(found):
            ##for qi, q in enumerate(found):
                ##if ri != qi and inside(r, q):
                    ##break
            ##else:
                #found_filtered.append(r)

        #found = [(x, y, 8, 8) for x, y in found]
        #draw_detections(img, found)
        ##draw_detections(img, found_filtered, 3)
        ##print('%d (%d) found' % (len(found_filtered), len(found)))
        #cv2.imshow('img', img)
        #ch = 0xFF & cv2.waitKey(30)
        #if ch == 27:
            #break
    #cv2.destroyAllWindows()
