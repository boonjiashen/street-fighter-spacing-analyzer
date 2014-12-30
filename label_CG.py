"""Label the center of gravity of characters in a Street Fighter match
"""

import numpy as np
import cv2
import argparse
import util
import re

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


class CG_fileIO():
    """Loaders and savers for file I/O of center of gravity of Street Fighter 4
    characters in a match
    """

    def saveline(filename, fi, p1_CG, p2_CG, append=True):
        """Save a snippet of CG info into file
        
        `p1_CG` `p2_CG` = (x, y) tuple of two player's CG

        `append` if True, append rather than overwrite file
        """

        fid = open(filename, 'a' if append else 'w')
        snippet = ' '.join(map(str, [fi, p1_CG, p2_CG]))
        fid.write(snippet + '\n')
        fid.close()

    def load(filename):
        """Return a dictionary that maps the frame index of a video to a
        (p1's CG, p2's CG) 2-ple

        Essentially an inverse of the saver.
        """

        CGs = {}
        pattern = r"(.*) (None|\(.*\)) (None|\(.*\))"
        with open(filename, 'r') as fid:
            for line in fid:
                match = re.match(pattern, line)
                frame_index, p1_CG, p2_CG = map(eval, match.groups())
                CGs[frame_index] = (p1_CG, p2_CG)

        return CGs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_filename',
            help='input video file')
    parser.add_argument('output_filename',
            help='output video file')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input_filename)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, onmouse)

    fid = open(args.output_filename, 'a')
    for fi, frame in enumerate(util.grab_frame(args.input_filename)):

        if fi%10 != 0:
            continue

        cv2.imshow(WIN, frame)

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
