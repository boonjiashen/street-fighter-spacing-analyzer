"""Label the center of gravity of characters in a Street Fighter match
"""

import numpy as np
import cv2
import argparse

WIN = 'video'
BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG


def onmouse(event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONUP:
        radius = 2
        cv2.circle(frame, (x, y), radius, BLACK, thickness=-1)
        cv2.imshow(WIN, frame)


def grab_frame(video_filename):
    "Yield successive frames from a video file"

    cap = cv2.VideoCapture(video_filename)

    while cap.isOpened():

        # Capture frame-by-frame
        read_success, frame = cap.read()

        if read_success:
            yield frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_filename',
            help='input video file')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input_filename)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, onmouse)

    for frame in grab_frame(args.input_filename):

        cv2.imshow(WIN, frame)

        # Do not progress until we get a ESC or 'n'
        while True:
            ch = cv2.waitKey()
            if ch in (27, ord('n')):
                break
        if ch == 27:
            break
        elif ch == ord('n'):
            continue

    cv2.destroyAllWindows()
