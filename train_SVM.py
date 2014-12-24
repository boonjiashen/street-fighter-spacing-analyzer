"""Train SVM using HOG as features
"""
import numpy as np
import cv2

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('img_filename')
    args = parser.parse_args()

    im_target = cv2.imread(args.img_filename)

    hog = cv2.HOGDescriptor()
    win_width, win_height = hog.winSize

    # Split image into four windows
    im_TL = cv2.cvtColor(im_target[:win_height, :win_width], cv2.COLOR_BGR2GRAY)
    im_TR = cv2.cvtColor(im_target[:win_height, 8:win_width+8], cv2.COLOR_BGR2GRAY)
    im_BR = cv2.cvtColor(im_target[8:win_height+8, 8:win_width+8], cv2.COLOR_BGR2GRAY)
    im_BL = cv2.cvtColor(im_target[8:win_height+8, :win_width], cv2.COLOR_BGR2GRAY)

    # Crop image with all four windows
    im_4windows = cv2.cvtColor(im_target[:win_height+8, :win_width+8], cv2.COLOR_BGR2GRAY)

    # Compute descriptors of an image, where window strides are taken in row
    # major order
    hogify = lambda x: hog.compute(x, winStride=(8,8), padding=(0,0))

    hog_4windows = hogify(im_4windows)
    hog_TL = hogify(im_TL)
    hog_BL = hogify(im_BL)
    hog_TR = hogify(im_TR)
    hog_BR = hogify(im_BR)

    for ind, array in enumerate(np.split(hog_4windows, 4, axis=0)):
        print('array ', ind, ' = TL? ', np.linalg.norm(array - hog_TL))
        print('array ', ind, ' = BR? ', np.linalg.norm(array - hog_BR))
        print('array ', ind, ' = TR? ', np.linalg.norm(array - hog_TR))
        print('array ', ind, ' = BL? ', np.linalg.norm(array - hog_BL))

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
