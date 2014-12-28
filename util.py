"""Useful functions
"""

import cv2
import math
import numpy as np


def grab_frame(video_filename):
    "Yield successive frames from a video file"

    cap = cv2.VideoCapture(video_filename)

    while cap.isOpened():

        # Capture frame-by-frame
        read_success, frame = cap.read()

        if read_success:
            yield frame


def tile(tiles):
    """Return a canvas from tiling 2D images of the same size

    Tries to return an image as square as possible.
    `tiles` generator or iterator of 2D images of the same size
    """

    def unsquareness(tile_size, tiling_factor):
        """A metric of how square a 2D image is when it is tiled in both dimensions

        The smaller the metric, the more square the tiling pattern is.
        `tile_size` (height, width) size of tile
        `tiling_factor` (vertical_factor, horizontal_factor) no. of times the tile
        is repeated in each direction
        """

        # Height and width of final tiled pattern
        h, w = [sz * factor for sz, factor in zip(tile_size, tiling_factor)]

        # We square the log of the ratios so that unsquaredness of 1/x or x is the
        # same
        unsquareness = math.log(h/w)**2

        return unsquareness 

    tiles = list(tiles)

    # Make sure that all tiles share the same size
    for tile in tiles:
        assert tile.shape == tiles[0].shape

    # Get optimal tiling factor
    n_tiles = len(tiles)
    tile_size = tiles[0].shape
    tiling_factor = min(
            ((math.ceil(n_tiles / i), i) for i in range(1, n_tiles + 1)),
            key=lambda x: unsquareness(tile_size, x)
            )

    # Add blank tiles to fill up canvas
    blank_tile = np.zeros_like(tiles[0])
    tiles.extend([blank_tile for i in range(np.prod(tiling_factor) - n_tiles)])

    # Tile tiles
    rows = [np.hstack(tiles[i:i+tiling_factor[1]])
        for i in range(0, len(tiles), tiling_factor[1])]
    canvas = np.vstack(rows)

    return canvas
