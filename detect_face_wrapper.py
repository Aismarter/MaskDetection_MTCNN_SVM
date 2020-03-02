from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import detect_face
import os
from tkinter.filedialog import askdirectory
import SVM


class DetectFaceWrapper(object):
    def __init__(self, minsize=20, threshold = [ 0.6, 0.7, 0.7 ], factor = 0.709, gpu_memory_fraction=0.5):
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor
        self.gpu_memory_fraction = gpu_memory_fraction
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)

    def detect_image(self, img):
        bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        return bounding_boxes


def getFace(imgPth, faceSavedPth):
    op = DetectFaceWrapper()
    imgFileList = [x for x in os.listdir(imgPth)]
    print(imgFileList)
    n = 0
    for i in range(len(imgFileList)):
        imgPt = imgPth + '/' + imgFileList[i]
        print(imgPt)
        try:
            img = cv2.imread(imgPt)
            im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = op.detect_image(im)
            print(face_locations)
        except:
            continue
        num_face = face_locations.shape[0]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        if num_face > 0:
            m = 0
            for l in range(num_face):
                im_copy = im.copy()
                try:
                    roi = im_copy[int(face_locations[l, 1]):int(face_locations[l, 3]),
                          int(face_locations[l, 0]):int(face_locations[l, 2])]
                    # roi = binary[y:y+h, x: x+w]
                    # roi = cv2.resize(roi, (int(width/3), int(height/3)))
                    cv2.namedWindow("roi", cv2.WINDOW_FREERATIO)
                    cv2.imshow("roi", roi)
                    cv2.imwrite(faceSavedPth + '/' + str(n) + str(m) + "wq" + imgFileList[i], roi)
                except:
                    abandon = "abandon"
                    if not os.path.exists(abandon):
                        os.mkdir(abandon)
                    cv2.imwrite(abandon + '/' + imgFileList[i], im_copy)

                m += 1

                cv2.rectangle(im_copy, (int(face_locations[l, 0]), int(face_locations[l, 1])),
                              (int(face_locations[l, 2]), int(face_locations[l, 3])), (0, 255, 0), 2)
                # cv.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
                cv2.namedWindow('label image', cv2.WINDOW_FREERATIO)
                cv2.imshow('label image', im_copy)
                cv2.waitKey(5)
                # press_key = cv2.waitKey(0)
                # if press_key == ord('q'):
                #     cv2.destroyAllWindows()
                #     continue
        n += 1


if __name__ == '__main__':
    # op = DetectFaceWrapper()
    # img = cv2.imread('12.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # bb = op.detect_image(img)
    # print(bb)
    # num_face = bb.shape[0]
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # for i in range(num_face):
    #     if bb[i, 4] < 0.95:
    #         continue
    #     cv2.rectangle(img, (int(bb[i, 0]), int(bb[i, 1])), (int(bb[i, 2]), int(bb[i, 3])),(255, 0, 0), 2)
    # cv2.imshow('images', img)
    # cv2.waitKey(0)

    # faceSavedPth = 'facePth'
    # if not os.path.exists(faceSavedPth):
    #     os.makedirs(faceSavedPth)
    # imgPth = askdirectory(title="Multi_face")
    # getFace(imgPth, faceSavedPth)

    op = DetectFaceWrapper()
    capture = cv2.VideoCapture("1.mp4")
    fps = capture.get(cv2.CAP_PROP_FPS)
    while(True):
        ret, frame = capture.read()
        if ret is True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bb = op.detect_image(img)
            print(bb)
            num_face = bb.shape[0]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for i in range(num_face):
                # if bb[i, 4] < 0.95:
                #     continue
                cv2.rectangle(img, (int(bb[i, 0]), int(bb[i, 1])), (int(bb[i, 2]), int(bb[i, 3])),(255, 0, 0), 2)
                # roi = frame[int(bb[i, 1]):int(bb[i, 3]), int(bb[i, 0]):int(bb[i, 2])]
                # cv2.namedWindow("roi", cv2.WINDOW_FREERATIO)
                # cv2.imshow("roi", roi)
                # SVM.elec_detect(roi)
            cv2.imshow('images', img)
            # cv2.waitKey(50)
            # cv2.imshow("video_Input", frame)
            c = cv2.waitKey(50)
            if c == 27:
                break
        else:
            break
