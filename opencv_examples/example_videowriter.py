#Example script of writing a video

import cv2
import numpy as np
import itertools

if __name__ == "__main__":

    filename = 'output.avi'  # filename of video to be saved

    # Intensities go from 0 to 255 and back down to 0 and repeat
    intensities = range(0, 256, 10)
    intensities = itertools.cycle(
            itertools.chain(intensities, reversed(intensities)))

    # Define the codec and create VideoWriter object
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.
    sz = (640, 360)  # NOTE! frame size = (w, h)
    out = cv2.VideoWriter(filename, codec, fps, sz)

    n_frames = 1000
    for intensity in itertools.islice(intensities, n_frames):
        image = intensity * np.ones(list(sz)[::-1] + [3], dtype=np.uint8)
        out.write(image)

    out.release()

    print('Saved video', filename)
