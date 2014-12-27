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
    global frame, p1_CG, p2_CG

    # Update CG of players
    p1_trigger, p2_trigger = cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP
    if event == p1_trigger:
        p1_CG = (x, y)
    elif event == p2_trigger:
        p2_CG = (x, y)

    # Update image
    frame_copy = frame.copy()
    radius = 2
    if event in [p1_trigger, p2_trigger]:
        if p1_CG:
            cv2.circle(frame_copy, p1_CG, radius, BLUE, thickness=-1)
        if p2_CG:
            cv2.circle(frame_copy, p2_CG, radius, BLACK, thickness=-1)
        cv2.imshow(WIN, frame_copy)


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

        p1_CG, p2_CG = None, None

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
