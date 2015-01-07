"""Useful functions
"""

import cv2
import math
import numpy as np


BLACK = [0, 0, 0]
WHITE = [256 for i in range(3)]


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


def overlaps(rect1, rect2):
    """Checks if two rectangles overlap

    `rect1` `rect2` = (xTL, yTL, xBR, yBR) 4-tuple

    >>> overlaps([0, 0, 5, 5], [6, 5, 10, 10])  # adjacent rectangles
    False
    >>> overlaps([0, 0, 5, 5], [5, 5, 10, 10])  # borders coincide
    True
    """

    # Cast as NumPy arrays for easier indexing
    rect1, rect2 = map(np.array, [rect1, rect2])

    # Cast rects as lines in x and y dimensions
    for axis in [0, 1]:
        line1 = rect1[[axis, axis + 2]]
        line2 = rect2[[axis, axis + 2]]

        # Check if left edge of one exceeds right edge of the other
        if line1[0] > line2[1]:
            return False
        if line1[1] < line2[0]:
            return False
    return True


def put_text(image, text):
#def put_text(image, text, top_left_origin=(0, 0)):
    """Draw black text on a white background on the top-right corner of an
    image
    """

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 1
    (width, height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)

    # Draw a tight white background for text
    cv2.rectangle(image, (0, 0), (width, height), WHITE, -1)

    # Draw text
    cv2.putText(image, text, (0, height),
            font_face, font_scale, BLACK,
            )


def grab_frame(video_filename):
    "Yield successive frames from a video file"

    cap = cv2.VideoCapture(video_filename)

    assert cap.isOpened(), 'Cannot read video!'

    while cap.isOpened():

        # Capture frame-by-frame
        read_success, frame = cap.read()

        if read_success:
            yield frame


def tile(tiles, desired_aspect=1.):
    """Return a canvas from tiling 2D images of the same size

    Tries to return an image as square as possible.

    `tiles` generator or iterator of 2D images of the same size

    `desired_aspect` = width/height, desired aspect ratio of canvas, e.g. 16/9
    when a screen is 16:9
    """

    def unaspectness(tile_size, tiling_factor, desired_aspect=1.):
        """A metric of how close a 2D image is to an aspect ratio when it is
        tiled in both dimensions.

        The smaller the metric, the more square the tiling pattern is.

        `tile_size` = (height, width) size of tile

        `tiling_factor` = (vertical_factor, horizontal_factor) no. of times the tile
        is repeated in each direction

        `desired_aspect` = width/height, desired aspect ratio, e.g. 16/9 when a
        screen is 16:9
        """

        # Height and width of final tiled pattern
        h, w = [sz * factor for sz, factor in zip(tile_size, tiling_factor)]

        # We square the log of the ratios so that unsquaredness of 1/x or x is the
        # same
        unaspectness = math.log(w/h/desired_aspect)**2

        return unaspectness 

    tiles = list(tiles)

    # Make sure that all tiles share the same size
    for tile in tiles:
        assert tile.shape == tiles[0].shape

    # Get optimal tiling factor
    n_tiles = len(tiles)
    tile_size = tiles[0].shape
    tiling_factor = min(
            ((math.ceil(n_tiles / i), i) for i in range(1, n_tiles + 1)),
            key=lambda x: unaspectness(tile_size, x, desired_aspect)
            )

    # Add blank tiles to fill up canvas
    blank_tile = np.zeros_like(tiles[0])
    tiles.extend([blank_tile for i in range(np.prod(tiling_factor) - n_tiles)])

    # Tile tiles
    rows = [np.hstack(tiles[i:i+tiling_factor[1]])
        for i in range(0, len(tiles), tiling_factor[1])]
    canvas = np.vstack(rows)

    return canvas


def demo_overlaps():
    """Demo of overlaps() by generating pairs of random rectangles

    Instructions: Press ESC to exit or any other key to continue
    """

    def generate_rectangle():
        "Generate a random rectangle"

        # Ensure no zero-width edge
        rect = np.random.randint(0, 10, 4)
        while rect[0] == rect[2] or rect[1] == rect[3]:
            rect = np.random.randint(0, 10, 4)

        # Swap values to ensure that xyTL is less than xyBR
        for axis in [0, 1]:
            if rect[axis] > rect[axis + 2]:
                rect[axis], rect[axis + 2] = rect[axis + 2], rect[axis]

        return rect

    import matplotlib.pyplot as plt

    WIN = 'video'
    while True:

        # Generate two rectangles to test for overlap
        rect1, rect2 = [generate_rectangle() for i in range(2)]

        # Draw rectangles on a canvas
        canvas = np.zeros([10, 10])
        for rect, color in zip([rect1, rect2], [.5, 1.]):
            tl, br = map(tuple, [rect[:2], rect[2:]])
            cv2.rectangle(canvas, tl, br, color)
        canvas = cv2.resize(canvas,
                tuple(10*x for x in canvas.shape),
                interpolation=cv2.INTER_NEAREST)

        print('overlaps?', overlaps(rect1, rect2))
        cv2.imshow(WIN, canvas)

        # Wait for user input
        ESC = 27
        if cv2.waitKey() == ESC:
            break
    cv2.destroyAllWindows()


def demo_tile():
    """Demo tile() with stdin

    Instructions: press ESC to exit
    """

    import itertools
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('video_filename')
    args = parser.parse_args()

    # Grab all frames
    full_frames = grab_frame(args.video_filename)

    # Resize all frames to fit better into canvas
    scale = .3  # Resize scale
    mini_frames = (cv2.resize(x, None, fx=scale, fy=scale)
            for x in full_frames)

    # Select every N frames
    step, n_frames = 15, 40
    frames = itertools.islice(mini_frames, 0, n_frames * step, step)

    # Tile selected frames into a canvas
    canvas = tile(frames, desired_aspect=16/9)

    cv2.imshow('1', canvas)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = demo_tile
    print(demo.__doc__)
    demo()
