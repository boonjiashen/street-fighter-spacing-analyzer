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

frame = None  # target frame of a Street Fighter 4 match
p1_CG = None  # CG of player 1
p2_CG = None  # CG of player 2
rect_size = (100, 50)  # size of rectangle (height, width)

def onmouse(event,x,y,flags,param):
    "Update CG of players upon appropriate mouse click"

    global frame, p1_CG, p2_CG

    # Define which button maps to which player
    p1_trigger, p2_trigger = cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP

    # Define colors of players
    p1_color, p2_color = BLUE, GREEN

    # Update CG of players
    if event == p1_trigger:
        p1_CG = (x, y)
    elif event == p2_trigger:
        p2_CG = (x, y)

    # Update image
    frame_copy = frame.copy()
    thickness = 2
    if event in [p1_trigger, p2_trigger]:
        for CG, color in [(p1_CG, p1_color), (p2_CG, p2_color)]:
            if CG is not None:
                h, w = rect_size
                TL = (CG[0] - w//2, CG[1] - h//2)  # top-left
                BR = (CG[0] + w//2, CG[1] + h//2)  # btm+right
                cv2.rectangle(frame_copy, TL, BR, color, thickness=thickness)
        cv2.imshow(WIN, frame_copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_filename',
            help='input video file')
    parser.add_argument('output_filename',
            help='output video file')
    parser.add_argument('--start', type=int, default=0,
            help='first frame to be labeled')

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
    for fi, frame in enumerate(frames, args.start):

        if fi%10 != 0:
            continue

        cv2.imshow(WIN, frame)
        print('Now at frame', fi)

        # Refresh CG as nothing
        p1_CG, p2_CG = None, None

        # Do not progress until we get a ESC or 'n'
        while True:
            ch = cv2.waitKey()
            if ch in (27, ord('n')):
                break
        if ch == 27:
            break
        elif ch == ord('n'):
            CG_fileIO.saveline(args.output_filename, fi, p1_CG, p2_CG)
            continue

    fid.close()
    cv2.destroyAllWindows()
