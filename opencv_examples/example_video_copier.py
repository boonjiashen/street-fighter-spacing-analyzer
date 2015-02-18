#Example script of copying a video

import cv2
import numpy as np
import itertools
import sys

if __name__ == "__main__":


    import argparse

    # Parse user-defined commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', help='Source video file')
    parser.add_argument('output_filename', help='Destination video file')
    parser.add_argument('--n_frames', type=int, default=100,
            help='Number of frames (from starting frame) to be saved')
    args = parser.parse_args()

    # Make args attributes local for easier addressing, e.g. 'n_frames' rather
    # than args.n_frames
    locals().update(args.__dict__)

    # Open source video
    cap = cv2.VideoCapture(input_filename)
    assert cap.isOpened(), 'Cannot read video!'

    # Define destination video
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    sz = tuple(map(int, [w, h]))  # NOTE! frame size = (w, h)
    out = cv2.VideoWriter(output_filename, codec, fps, sz)

    fi = 0  # Frame index
    while cap.isOpened():

        # Capture frame-by-frame
        read_success, frame = cap.read()

        if read_success:
            out.write(frame)
            sys.stdout.write('Saved frame ' + str(fi) + '\r')
            if fi > n_frames:
                break
            fi += 1

    out.release()
    cap.release()

    print('Saved video', output_filename)
