"""Display a movie of the Hue channel histogram
"""
import numpy as np
import cv2
import math
import util
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nms
import random
import argparse
import time
import itertools

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename')
    parser.add_argument("--log", type=str, default='INFO',
            help="Logging setting (e.g., INFO, DEBUG)")
    args = parser.parse_args()

    # Setting logging parameters
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(message)s')


    def update_line(frame, line):
        im_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        counts, bins = np.histogram(im_HSV[:,:,0], bins=180)
        data = np.vstack([
                np.arange(180),
                counts / np.max(counts),
                ])
        line.set_data(data)
        return line,

    fig1 = plt.figure()
    l, = plt.plot([], [], 'ro-')
    plt.xlim(0, 180)
    plt.ylim(0, 1)

    all_frames = util.grab_frame(args.input_filename)
    line_ani = animation.FuncAnimation(fig1, update_line, all_frames, fargs=(l, ),
        interval=50, blit=False)
    plt.show()

if __name__ == '__main__':

    #test()
    main()
