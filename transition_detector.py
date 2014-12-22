"Detect transitions (start/end of rounds) in a Street Fighter 4 match"

import cv2
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np


def find_transitions(frames):
    """Returns frame indices of transitions given frames of a match
    """

    # Look for frames that are very dark
    dark_enough = lambda x: x < 80
    ave_intensities = [np.mean(frame) for frame in frames]
    peak_mask = np.array([dark_enough(x) for x in ave_intensities])

    # Frame index where the transitions are
    transition_inds = [ind + length//2
            for ind, length in find_CC_in_1D_array(peak_mask)]

    return transition_inds


def find_CC_in_1D_array(array):
    """Find connected components in a 1D array

    `array` is a binary 1D array.
    Connected components are connected 1s.
    Returns a list of (start_ind, length) 2-ples

    >>> find_CC_in_1D_array([0, 0, 1, 1, 1, 0])
    [(2, 3)]
    """

    # Connected component is a list of (start_ind, length) 2-ples
    ccs = []
    at_cc = False  # Whether we're currently at a connected component
    for ind in range(len(array)):

        # Check that we just went from neg to pos
        if array[ind] == 1 and not at_cc:
            at_cc = True
            start_ind = ind

        # Check that we just went from pos to neg
        elif array[ind] == 0 and at_cc:
            at_cc = False
            len_cc = ind - start_ind
            ccs.append((start_ind, len_cc))

    # Add last connected component if we ended pos
    if at_cc:
        ccs.append((start_ind, len(array) - start_ind))

    return ccs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('input_filename',
            help='input video file')
    parser.add_argument('--last_frame', type=int,
            help='last frame of region of interest (default: process all frames)')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input_filename)

    FPS = cap.get(cv2.CAP_PROP_FPS)  # frames per sec of video

    frames = []
    #for fi in range(60):
    #for fi in range(1500):
    indices = range(args.last_frame)  \
            if args.last_frame  \
            else itertools.count()
    for fi in indices:
        if not cap.isOpened():
            break

        # Capture frame-by-frame
        read_success, frame = cap.read()

        if not read_success:
            break

        frames.append(frame)

    transition_inds = find_transitions(frames)

    # Print results
    print('Transitions occur at frames indexed ', end='')
    print(*transition_inds, sep=', ')
    print('These occur at ', end='')
    print(*['{0:.2f}'.format(x/FPS) for x in transition_inds], sep=', ', end='')
    print(' sec')
