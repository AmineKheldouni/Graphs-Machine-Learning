import matplotlib.pyplot as pyplot
import scipy.misc as sm
import numpy as np
import cv2
import os
import sys

path=os.path.dirname(os.getcwd()+"/code_material_python")
sys.path.append(path)
from helper import *
from harmonic_function_solution.func import *

def offline_face_recognition():

    # Parameters
    cc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    frame_size = 96
    gamma = .95
    # Loading images
    images = np.zeros((100, frame_size ** 2))
    labels = np.zeros(100)
    var=10000**2

    for i in np.arange(10):
      for j in np.arange(10):
        im = sm.imread("data/10faces/%d/%02d.jpg" % (i, j + 1))
        box = cc.detectMultiScale(im)
        top_face = {"area": 0}

        for cfx, cfy, clx, cly in box:
            face_area = clx * cly
            if face_area > top_face["area"]:
                top_face["area"] = face_area
                top_face["box"] = [cfx, cfy, clx, cly]

        fx, fy, lx, ly = top_face["box"]
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_face = gray_im[fy:fy + ly, fx:fx + lx]
        gray_face = cv2.resize(gray_face, (frame_size,frame_size))

        # Preprocessing
        # gf = gray_face
        # gf = cv2.equalizeHist(gray_face)
        gf = cv2.GaussianBlur(gray_face, (5,5), 1)
        #######################################################################
        #######################################################################

        #resize the face and reshape it to a row vector, record labels
        images[j * 10 + i] = gf.reshape((-1))
        labels[j * 10 + i] = i + 1



    plot_the_dataset = 1

    if plot_the_dataset:

     pyplot.figure(1)
     for i in range(100):
        pyplot.subplot(10,10,i+1)
        pyplot.axis('off')
        pyplot.imshow(images[i].reshape(frame_size,frame_size))
        r='{:d}'.format(i+1)
        if i<10:
         pyplot.title('Person '+r)
     pyplot.show()

    # Selecting 4 randomly known labels per class (person)
    Y_masked = np.zeros(labels.shape)
    for i in np.arange(10):
        knownLabels = np.random.choice(np.arange(10), 4, replace=False)
        for j in knownLabels:
            Y_masked[j*10+i] = i+1

    k = 6
    eps = 0
    c_l = 0.96
    c_u = 0.04
    laplacian_regularization = 0.04

    rlabels = hard_hfs(images, Y_masked, laplacian_regularization ,var=var, eps=eps, k=k, laplacian_normalization="rw", real_num_classes=10)

    # Plots #
    pyplot.subplot(121)
    pyplot.imshow(labels.reshape((10, 10)))

    pyplot.subplot(122)
    pyplot.imshow(rlabels.reshape((10, 10)))

    pyplot.title("Acc: {}".format(np.equal(rlabels, labels).mean()))
    pyplot.show()


offline_face_recognition()


def offline_face_recognition_augmented():
    # Parameters
    cc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    frame_size = 96
    gamma = .95
    nbimgs = 50

    # Loading images
    images = np.zeros((10 * nbimgs, frame_size ** 2))
    labels = np.zeros(10 * nbimgs)
    var=10000**2

    for i in np.arange(10):
      imgdir = "data/extended_dataset/%d" % i
      imgfns = os.listdir(imgdir)
      for j, imgfn in enumerate(np.random.choice(imgfns, size=nbimgs)):
        im = sm.imread("{}/{}".format(imgdir, imgfn))
        box = cc.detectMultiScale(im)
        top_face = {"area": 0, "box": (0, 0, *im.shape[:2])}

        for cfx, cfy, clx, cly in box:
            face_area = clx * cly
            if face_area > top_face["area"]:
                top_face["area"] = face_area
                top_face["box"] = [cfx, cfy, clx, cly]


        fx, fy, lx, ly = top_face["box"]
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_face = gray_im[fy:fy + ly, fx:fx + lx]
        gray_face = cv2.resize(gray_face, (frame_size,frame_size))

        # Preprocessing
        # gf = cv2.equalizeHist(gray_face)
        gf = cv2.GaussianBlur(gray_face, (5,5), 1)

        #resize the face and reshape it to a row vector, record labels
        images[j * 10 + i] = gf.reshape((-1))
        labels[j * 10 + i] = i + 1

    # if you want to plot the dataset, set the following variable to 1
    plot_the_dataset = 0

    if plot_the_dataset:

     pyplot.figure(1)
     for i in range(10 * nbimgs):
        pyplot.subplot(nbimgs,10,i+1)
        pyplot.axis('off')
        pyplot.imshow(images[i].reshape(frame_size,frame_size))
        r='{:d}'.format(i+1)
        if i<10:
         pyplot.title('Person '+r)
     pyplot.show()

    # Masking the labels before running HFS classification
    Y_masked = np.zeros(labels.shape)
    for i in np.arange(10):
        knownLabels = np.random.choice(np.arange(10), 4, replace=False)
        for j in knownLabels:
            Y_masked[j*10+i] = i+1

    # Parameters
    k = 10
    c_l = 0.96
    c_u = 0.04
    laplacian_regularization = 0.04

    rlabels = hard_hfs(images, Y_masked, laplacian_regularization ,var=var, eps=eps, k=k, laplacian_normalization="rw", real_num_classes=10)

    ##### Uncomment the following code to make the model #####
    # for i in range(10):
    #     for j, imgfn in enumerate(np.random.choice(imgfns, size=nbimgs)):
    #         if rlabels[j*10+i] != labels[j*10+i]:
    #             pyplot.figure(1)
    #             pyplot.imshow(images[i].reshape(frame_size,frame_size))
    #             pyplot.show()
    #             print("rlabel: ", rlabels[j*10+i])
    #             print("true label: ", labels[j*10+i])
    #             break

    # Plots #
    pyplot.subplot(121)
    pyplot.imshow(labels.reshape((-1, 10)))

    pyplot.subplot(122)
    pyplot.imshow(rlabels.reshape((-1, 10)))
    pyplot.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    pyplot.show()

offline_face_recognition_augmented()
