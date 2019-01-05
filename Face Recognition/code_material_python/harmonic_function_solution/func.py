import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sys
import os

path=os.path.dirname(os.getcwd()+"/code_material_python")
sys.path.append(path)
from helper import *
#from graph_construction.func import *


def build_laplacian_regularized(X, laplacian_regularization ,var=1, eps=0, k=0, laplacian_normalization=""):

    W = build_similarity_graph(X, var, eps, k)
    L = build_laplacian(W, laplacian_normalization=laplacian_normalization)
    Q = L + laplacian_regularization * np.eye(L.shape[0])

    return Q


def  mask_labels(Y, l):

    num_samples = np.size(Y,0)
    Y_masked = np.zeros(num_samples)

    i = 0

    # Mask all labels except l random examples
    while i < l:
        randomIdx = np.random.randint(0, num_samples)
        Y_masked[randomIdx] = Y[randomIdx]
        i = np.sum(Y_masked!=0)

    return Y_masked


def hard_hfs(X, Y,laplacian_regularization ,var=1, eps=0, k=0, laplacian_normalization="", real_num_classes=2):


    num_samples = np.size(X,0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    # Indexes of labeled data (l_idx) and unlabeled data (u_idx)
    l_idx = np.where(Y != 0)[0].reshape(-1)
    u_idx = np.where(Y == 0)[0].reshape(-1)

    # Construction of the laplacian matrix
    L = build_laplacian_regularized(X, laplacian_regularization, var, eps, k, laplacian_normalization)

    # Computation of f-matrix and prediction of the labels
    Luu = L[:,u_idx][u_idx]
    Lul = L[:,u_idx][l_idx]
    f = np.zeros((num_samples, real_num_classes))
    print(f.shape)
    for i in range(num_samples):
        if Y[i] != 0:
            f[i,Y[i].astype(int)-1] = 1
    f[u_idx,:] = np.linalg.inv(Luu).dot(-Lul.T.dot(f[l_idx,:]))


    labels = np.argmax(f, axis=1)+1

    return labels


def two_moons_hfs():
    # a skeleton function to perform HFS, needs to be completed


    # load the data
    in_data =scipy.io.loadmat(path+'/data/data_2moons_hfs')
    # in_data =scipy.io.loadmat(path+'/data/data_2moons_hfs_large')
    X = in_data['X']
    Y = in_data['Y']

    # automatically infer number of labels from samples
    num_samples = np.size(X,1)
    num_classes = len(np.unique(Y))


    l = 4 # number of labeled (unmasked) nodes provided to the hfs algorithm

    # mask labels
    Y_masked = mask_labels(Y, l)

    # Parameters
    var = 1
    # When data are not compact we use a k-NN Graph
    k = 6
    eps = 0

    # When there is a large sample of data we use an epsilon graph
    # k = 0
    # eps = 0.72

    labels = hard_hfs(X, Y_masked, 0.1 ,var=var, eps=eps, k=k, laplacian_normalization="rw")

    plot_classification(X, Y, labels,  var=var, eps=eps, k=k)
    accuracy = np.mean(labels.reshape(-1) == np.squeeze(Y).reshape(-1))
    print("Accuracy of labeling: ", accuracy)
    return accuracy


accuracy = two_moons_hfs()


def soft_hfs(X, Y, c_l, c_u, laplacian_regularization ,var=1, eps=0, k=0, laplacian_normalization=""):

    num_samples = np.size(X,0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    l_idx = np.where(Y != 0)[0].reshape(-1)
    u_idx = np.where(Y == 0)[0].reshape(-1)


    Cinv = np.zeros((num_samples,num_samples))

    Cinv[u_idx,u_idx] = 1./c_u * np.ones(len(u_idx))
    Cinv[l_idx,l_idx] = 1./c_l * np.ones(len(l_idx))

    Cinv_uu = Cinv[:,u_idx][u_idx]
    Cinv_ll = Cinv[:,l_idx][l_idx]

    L = build_laplacian_regularized(X, laplacian_regularization, var, eps, k, laplacian_normalization)

    Luu = L[:,u_idx][u_idx]
    Lul = L[:,u_idx][l_idx]
    Lll = L[:,l_idx][l_idx]
    u = len(u_idx)
    l = len(l_idx)
    tmp_matrix = np.zeros((u+l,u+l))
    tmp_matrix[0:u,0:u] = Cinv_uu.dot(Luu) + np.eye(u)
    tmp_matrix[u:u+l,u:u+l] = Cinv_ll.dot(Lll) + np.eye(l)
    tmp_matrix[0:u,u:u+l] = Lul.T.dot(Cinv_ll)
    tmp_matrix[u:u+l,0:u] = Lul.dot(Cinv_uu)

    f = np.linalg.inv(tmp_matrix).dot((2*Y-3)*(Y!=0))
    f_u = f[u_idx]
    f_l = f[l_idx]


    labels = np.zeros(num_samples)
    labels[l_idx] = (3+np.sign(f_l))//2
    labels[u_idx] = (3+np.sign(f_u))//2

    return labels


def hard_vs_soft_hfs():

    in_data =scipy.io.loadmat(path+'/data/data_2moons_hfs')
    # in_data =scipy.io.loadmat(path+'/data/data_2moons_hfs_large')
    X = in_data['X']
    Y = in_data['Y']

    # automatically infer number of labels from samples
    num_samples = np.size(X,0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    # randomly sample 20 labels
    l = 20

    # mask labels
    Y_masked =  mask_labels(Y, l)
    Y_masked[Y_masked != 0] = label_noise(Y_masked[Y_masked != 0], 4)


    l = 4 # number of labeled (unmasked) nodes provided to the hfs algorithm

    #Parameters
    var = 1
    # When data are not compact we use a k-NN Graph
    k = 6
    eps = 0
    # When there is a large sample of data we use an epsilon graph
    k = 0
    eps = 0.72

    c_l = 0.96
    c_u = 0.04
    laplacian_regularization = 0.04
    #################################################################
    # compute hfs solution using soft_hfs.m and hard_hfs.m          #
    # remember to use Y_masked (the vector with some labels hidden  #
    # as input and not Y (the vector with all labels revealed)      #
    #################################################################

    hard_labels = hard_hfs(X, Y_masked, laplacian_regularization ,var=var, eps=eps, k=k, laplacian_normalization="rw")
    soft_labels = soft_hfs(X, Y_masked, c_l, c_u, laplacian_regularization,var=var, eps=eps, k=k, laplacian_normalization="rw")

    #################################################################
    #################################################################

    Y_masked[Y_masked == 0] = np.squeeze(Y)[Y_masked == 0]

    plot_classification_comparison(X, Y, hard_labels, soft_labels,var=var, eps=eps, k=k)
    accuracy = [np.mean(hard_labels == np.squeeze(Y)), np.mean(soft_labels == np.squeeze(Y))]
    print("Accuracy of labeling: ", accuracy)
    return accuracy

accuracies = hard_vs_soft_hfs()
