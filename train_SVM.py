"""Train SVM using HOG as features
"""
import numpy as np
import cv2

def yield_windows(image, window_size, step_size):
    """Yield windows of an image in regular intervals in row-major order.

    `image` - a 2D image
    `window_size` - required (height, width) of window
    `step_size` - (vertical_step, horizontal_step) 2-ple
    """

    im_height, im_width = image.shape[:2]
    win_height, win_width = window_size
    y_step, x_step = step_size

    max_y_TL = (im_height - win_height) // y_step * y_step
    max_x_TL = (im_width - win_width) // x_step * x_step
    for y_TL in range(0, max_y_TL + 1, y_step):
        for x_TL in range(0, max_x_TL + 1, x_step):
            window = image[
                    y_TL:y_TL + win_height,
                    x_TL:x_TL + win_width]
            yield window


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('img_filename')
    args = parser.parse_args()

    im_target = cv2.imread(args.img_filename)

    hog_obj = cv2.HOGDescriptor()
    win_width, win_height = hog_obj.winSize

    # Compute descriptors of an image, where window strides are taken in row
    # major order
    hogify = lambda x: hog_obj.compute(x, winStride=(8,8), padding=(0,0))

    #im = np.array([1, 2, 3, 4, 5, 6])[np.newaxis, :, np.newaxis]
    #windows = [x for x in yield_windows(im, (1, 4), (1, 2))]

    win_size = (win_height, win_width)
    step_size = (win_height // 2, win_width // 2)
    X = (hogify(window)
            for window in yield_windows(im_target, win_size, step_size))

    #hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    #while(1):
        #ret, img = cap.read()

        ##found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
        #found, _ = hog.detect(img, winStride=(8,8), padding=(32,32))
        #found_filtered = []
        #for ri, r in enumerate(found):
            ##for qi, q in enumerate(found):
                ##if ri != qi and inside(r, q):
                    ##break
            ##else:
                #found_filtered.append(r)

        #found = [(x, y, 8, 8) for x, y in found]
        #draw_detections(img, found)
        ##draw_detections(img, found_filtered, 3)
        ##print('%d (%d) found' % (len(found_filtered), len(found)))
        #cv2.imshow('img', img)
        #ch = 0xFF & cv2.waitKey(30)
        #if ch == 27:
            #break
    #cv2.destroyAllWindows()
