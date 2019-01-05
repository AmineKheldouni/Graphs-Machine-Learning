import matplotlib.pyplot as pyplot
import scipy.misc as sm
import numpy as np
import cv2 as cv
import os
import sys
from scipy.spatial import distance
import scipy.io as sio
from tqdm import tqdm
import copy

path=os.path.dirname(os.getcwd()+"/code_material_python")
sys.path.append(path)
from helper import *


def iterative_hfs(niter = 20):
    # load the data
    # a skeleton function to perform HFS, needs to be completed
    #  Input
    #  niter:
    #      number of iterations to use for the iterative propagation

    #  Output
    #  labels:
    #      class assignments for each (n) nodes
    #  accuracy

    mat = sio.loadmat("./data/data_iterative_hfs_graph.mat")
    W, Y, Y_masked = mat["W"], mat["Y"], mat["Y_masked"]

    classes = np.unique(Y_masked[Y_masked > 0])

    #####################################
    # Compute the initialization vector f 
    f = np.zeros((Y_masked.shape[0], 2))
    f[np.where(Y_masked == 1)[0], 0] = 1
    f[np.where(Y_masked == 1)[0], 1] = -1
    f[np.where(Y_masked == 2)[0], 0] = -1
    f[np.where(Y_masked == 2)[0], 1] = 1

    #################################################################
    # compute the hfs solution, using iterated averaging
    u_idx = np.where(Y_masked == 0)[0].reshape(-1)

    for _ in tqdm(range(niter)):
       f_old = f.copy()
       for i in u_idx:
           f[i] = W.getcol(i).T.dot(f_old) / W.getcol(i).sum()

    labels = classes[f.argmax(axis = 1)]
    accuracy = (labels == Y.reshape(-1)).mean()

    return labels, accuracy


print(iterative_hfs())
