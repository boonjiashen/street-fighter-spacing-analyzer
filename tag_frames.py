"""Tag frames of a Street Fighter match with the center of gravity of
characters in it.
"""

from CG_fileIO import CG_fileIO
import numpy as np
import cv2
import argparse
import util

WIN = 'video'
BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

def onmouse(event,x,y,flags,param):
    "Update CG of players upon appropriate mouse click"

    # Define which button maps to which player
    p1_trigger, p2_trigger = cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP

    # Define colors of players
    p1_color, p2_color = BLUE, GREEN

    # Update CG of players
    if event == p1_trigger:
        onmouse.p1_CG = (x, y)
    elif event == p2_trigger:
        onmouse.p2_CG = (x, y)

    # Update image
    frame_copy = onmouse.frame.copy()
    if event in [cv2.EVENT_MOUSEMOVE, p1_trigger, p2_trigger]:

        h, w = onmouse.rect_size  # Dimensions of boxes to be drawn

        # Box players
        for CG, color in [(onmouse.p1_CG, p1_color), (onmouse.p2_CG, p2_color)]:
            if CG is not None:
                TL = (CG[0] - w//2, CG[1] - h//2)  # top-left
                BR = (CG[0] + w//2, CG[1] + h//2)  # btm+right
                cv2.rectangle(frame_copy, TL, BR, color, thickness=3)

        # Draw cursor as a white box with a black shadow
        TL = (x - w//2, y - h//2)  # top-left
        BR = (x + w//2, y + h//2)  # btm+right
        cv2.rectangle(frame_copy, TL, BR, BLACK, thickness=2)
        cv2.rectangle(frame_copy, TL, BR, WHITE, thickness=1)
        cv2.imshow(onmouse.WIN, frame_copy)
onmouse.frame = None  # target frame of a Street Fighter 4 match
onmouse.p1_CG = None  # CG of player 1
onmouse.p2_CG = None  # CG of player 2
onmouse.rect_size = (100, 50)  # size of rectangle (height, width)
onmouse.WIN = WIN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_filename',
            help='input video file')
    parser.add_argument('output_filename',
            help='output video file')
    parser.add_argument('--start', type=int, default=0,
            help='first frame to be labeled')
    parser.add_argument('--step', type=int, default=10,
            help='step interval between frames')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input_filename)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, onmouse)

    fid = open(args.output_filename, 'a')

    # Initialize frame generator
    frames = util.grab_frame(args.input_filename)

    # Run generator until the desired starting frame
    for i in range(args.start - 1): next(frames)

    # Display frames, allowing user to label
    for fi, onmouse.frame in enumerate(frames, args.start):

        if (fi - args.start) % args.step != 0:
            continue

        cv2.imshow(WIN, onmouse.frame)
        print('Now at frame', fi)

        # Refresh CG as nothing
        onmouse.p1_CG, onmouse.p2_CG = None, None

        # Do not progress until we get a ESC or 'n'
        while True:
            ch = cv2.waitKey()
            if ch in (27, ord('n')):
                break
        if ch == 27:
            break
        elif ch == ord('n'):
            CG_fileIO.saveline(
                    args.output_filename,
                    fi,
                    onmouse.p1_CG,
                    onmouse.p2_CG)
            continue

    fid.close()
    cv2.destroyAllWindows()
