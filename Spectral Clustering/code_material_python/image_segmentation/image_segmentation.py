# from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.getcwd())+"/code_material_python")
from helper import *
from spectral_clustering.func import *

path = os.getcwd()

def image_segmentation(input_img, k=40, num_classes=5, laplacian='unn', adaptative=True):
    filename = path+'/'+input_img

    X = plt.imread(filename)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    im_side = np.size(X,1)
    Xr = X.reshape(im_side**2,3)

    k = k
    eps = 0
    var = np.var(Xr)

    W, similarities = build_similarity_graph(Xr, var=var, k=k, eps=eps)
    L = build_laplacian(W, laplacian_normalization=laplacian)
    if adaptative:
        Y_rec = spectral_clustering_adaptative(L, num_classes=num_classes)
    else:
        Y_rec = spectral_clustering(L, [0,1,2,3], num_classes=num_classes)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(X)
    plt.subplot(1,2,2)
    Y_rec=Y_rec.reshape(im_side,im_side)
    plt.imshow(Y_rec)
    plt.show()

image_segmentation('fruit_salad.bmp', k=15, num_classes=5, laplacian='rw', adaptative=False)
image_segmentation('four_elements.bmp', k=40, num_classes=5, laplacian='rw')
