import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sklearn.cluster as skc
import sklearn.metrics as skm
import scipy
import scipy
import sys
import os

sys.path.append(os.path.dirname(os.getcwd())+"/code_material_python")
from helper import *
from graph_construction.func import *
path = os.getcwd()

def build_laplacian(W, laplacian_normalization=""):
    D = np.zeros(W.shape)
    D_inv = np.zeros(W.shape)
    D_invsqrt = np.zeros(W.shape)
    for i in range(W.shape[0]):
        D[i,i] = np.sum(W[i,:])
        D_inv[i,i] = 1./np.sum(W[i,:])
        D_invsqrt[i,i] = np.sqrt(D_inv[i,i])
    if laplacian_normalization == "unn":
        return D - W
    elif laplacian_normalization == "sym":
        return np.eye(W.shape[0]) - D_invsqrt.dot(W).dot(D_invsqrt)
    elif laplacian_normalization == "rw":
        return np.eye(W.shape[0]) - D_inv.dot(W)

def spectral_clustering(L, chosen_eig_indices, num_classes=2):
    [E,U] = np.linalg.eig(L)
    idx = np.argsort(np.real(E))
    E = np.diag(E[idx])
    U = np.real(U[:,idx])
    normalization = np.linalg.norm(U[:,chosen_eig_indices], axis=0)
    Y = U[:,chosen_eig_indices] / normalization
    Y = skc.KMeans(num_classes).fit(Y).labels_
    return Y


def two_blobs_clustering():
    # load the data
    in_data =scipy.io.loadmat(path+'/data/data_2blobs')
    X = in_data['X']
    Y = in_data['Y']
    num_classes = len(np.unique(Y))

    k = 10
    var =  0.5

    laplacian_normalization = 'unn'; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
    chosen_eig_indices =  [0,1]

    # build laplacian
    W, similarities = build_similarity_graph(X, var=var, k=k, eps=0)
    L = build_laplacian(W, laplacian_normalization=laplacian_normalization)
    Y_rec = spectral_clustering(L,chosen_eig_indices)
    plot_clustering_result(X, Y, L, Y_rec, skc.KMeans(num_classes).fit_predict(X), normalized_switch=0)

def choose_eig_function(eigenvalues):
    gap = [eigenvalues[i+1]-eigenvalues[i] for i in range(len(eigenvalues)-1)]
    sortedGapIdx = np.argsort(-abs(np.array(gap)))
    idx = -1
    maxGap = 0
    for i in range(len(sortedGapIdx)):
        if abs(gap[i])>= maxGap:
            idx = i
            maxGap = abs(gap[i])
    return range(idx)


def spectral_clustering_adaptative(L, num_classes=2):
    [E,U] = np.linalg.eig(L)
    idx = np.argsort(np.real(E))
    chosen_eig_indices = choose_eig_function(np.real(E[idx])[:15])
    E = np.diag(np.real(E[idx]))
    U = np.real(U[:,idx])
    Y = U[:,chosen_eig_indices] / np.linalg.norm(U[:,chosen_eig_indices], axis=0)
    Y = skc.KMeans(num_classes).fit(Y).labels_
    return Y


def find_the_bend():
    num_samples = 600

    [X, Y] = blobs(num_samples,4,0.03)

    num_classes = len(np.unique(Y))

    k = 10
    var =  0.5

    laplacian_normalization = 'unn'; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization

    # build the laplacian
    W, similarities = build_similarity_graph(X, var, k=k, eps=0)
    L = build_laplacian(W, laplacian_normalization=laplacian_normalization)

    [E,U] = np.linalg.eig(L)
    idx = np.argsort(np.real(E))
    E = E[idx]
    U = np.real(U[:,idx])
    eigenvalues = E[:15]
    chosen_eig_indices = choose_eig_function(eigenvalues) # indices of the ordered eigenvalues to pick
    normalization = np.linalg.norm(U[:,chosen_eig_indices], axis=0)
    Y_rec = U[:,chosen_eig_indices] / normalization
    Y_rec = skc.KMeans(num_classes).fit_predict(Y_rec)

    plot_the_bend(X, Y, L, Y_rec, eigenvalues)


def two_moons_clustering():
    in_data =scipy.io.loadmat(path+'/data/data_2moons')
    X = in_data['X']
    Y = in_data['Y']
    num_classes = len(np.unique(Y))

    k = 5
    eps = 0.5
    var = 0.5

    laplacian_normalization = 'sym'; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
    chosen_eig_indices = [0, 1] # indices of the ordered eigenvalues to pick

    W, similarities = build_similarity_graph(X, var, k=k, eps=eps)
    L = build_laplacian(W, laplacian_normalization=laplacian_normalization)
    Y_rec = spectral_clustering(L, chosen_eig_indices)

    plot_clustering_result(X, Y, L, Y_rec,skc.KMeans(num_classes).fit_predict(X))



def point_and_circle_clustering():
    in_data =scipy.io.loadmat(path+'/data/data_pointandcircle')
    X = in_data['X']
    Y = in_data['Y']
    num_classes = len(np.unique(Y))

    k = 15
    eps = 0.
    var = 0.5; # exponential_euclidean's sigma^2

    laplacian_normalization = 'rw'; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
    chosen_eig_indices = [0, 1] # indices of the ordered eigenvalues to pick

    #  build the laplacian
    W, similarities = build_similarity_graph(X, var, k=k, eps=eps)
    L_unn = build_laplacian(W, laplacian_normalization='unn')
    L_norm = build_laplacian(W, laplacian_normalization='rw')

    # Distribution of nodes' degrees/connections
    plt.hist(np.sum((L_unn!=0).astype(int),axis=0))
    plt.xlabel("Number of edges (connectivity)", fontsize=16)
    plt.ylabel("Number of nodes", fontsize=16)
    plt.show()

    Y_unn = spectral_clustering(L_unn, chosen_eig_indices)
    Y_norm = spectral_clustering(L_norm, chosen_eig_indices)
    plot_clustering_result(X, Y, L_unn, Y_unn, Y_norm, 1);


def parameter_sensitivity(param='k'):
    num_samples = 500;

    var = 0.5

    laplacian_normalization = 'unn'; #either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
    chosen_eig_indices =  [0, 1]
    if (param == 'k'):
        parameter_candidate = range(1,50,3)   # the number of neighbours for the graph or the epsilon threshold
    else:
        parameter_candidate = np.linspace(0.05,1,20)
    parameter_performance=[]

    for p in parameter_candidate:
        print(param+ " = " + str(p))
        [X, Y] = two_moons(num_samples,1,0.02)
        num_classes = len(np.unique(Y))
        if (param=='k'):
            W, similarities = build_similarity_graph(X, k=p)
        else:
            W, similarities = build_similarity_graph(X, eps=p)
        L =  build_laplacian(W, laplacian_normalization=laplacian_normalization)

        Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes)
        parameter_performance+= [skm.adjusted_rand_score(Y,Y_rec)]

    plt.figure()
    plt.plot(parameter_candidate, parameter_performance)
    plt.title('parameter sensitivity')
    if param=='k':
        plt.xlabel("k", fontsize=16)
    else:
        plt.xlabel("epsilon", fontsize=16)
    plt.ylabel("Performance", fontsize=16)
    plt.show()

two_blobs_clustering()
find_the_bend()
two_moons_clustering()
point_and_circle_clustering()
parameter_sensitivity('k')
parameter_sensitivity('eps')
