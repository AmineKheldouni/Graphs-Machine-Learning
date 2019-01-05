import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sys
import os
import copy

sys.path.append(os.path.dirname(os.getcwd())+"/code_material_python")
from helper import *
from graph_construction.generate_data import *


def build_similarity_graph(X, var=1, eps=0, k=0):
  """     Computes the similarity matrix for a given dataset of samples.

   Input
   X:
       (n x m) matrix of m-dimensional samples
   k and eps:
       controls the main parameter of the graph, the number
       of neighbours k for k-nn, and the threshold eps for epsilon graphs
   var:
       the sigma value for the exponential function, already squared


   Output
   W:
       (n x n) dimensional matrix representing the adjacency matrix of the graph
   similarities:
       (n x n) dimensional matrix containing
       all the similarities between all points (optional output)
  """

  assert eps + k != 0, "Choose either epsilon graph or k-nn graph"

  if eps:
    print("Constructing eps-Graph ...")
  else:
    print("Constructing k-NN Graph ...")

  # euclidean distance squared between points
  dists = sd.squareform(sd.pdist(X))**2
  similarities = np.exp(-dists/(2*var))

  if eps:
    W = similarities * (similarities-eps>=0)
    print("eps-Graph constructed !")
    return W, similarities

  if k:
    W = similarities.copy()
    for i in range(W.shape[0]):
      W[i,i] = 0
      kNearestNodes = np.argsort(-W[i,:])
      kIdx = kNearestNodes[k:]
      W[i,kIdx] = 0
    print("k-NN Graph constructed !")
    return np.maximum(W, W.T), similarities


def plot_similarity_graph(X, Y, eps=0.1, k=0, var=1):

    W, similarities = build_similarity_graph(X, var, eps=eps, k=k)
    plot_graph_matrix(X,Y,W)


def how_to_choose_epsilon(gen_param):

    # the number of samples to generate
    num_samples = 100
    gen_param =   gen_param
    [X, Y] = worst_case_blob(num_samples,gen_param)

    var =  0.5

    dists = sd.squareform(sd.pdist(X))**2
    similarities = np.exp(-dists/(2*var))

    # Building the max spanning tree
    max_tree = max_span_tree(similarities)
    A = similarities*max_tree
    # Finding the optimal epsilon
    eps = np.min(A[np.where(max_tree>0)])
    print("Best epsilon found: ", eps)

    plot_similarity_graph(X, Y, eps=eps, var=var)
    return eps

how_to_choose_epsilon(2)
[X,Y] = blobs(200,2,0.2)
[X, Y] = worst_case_blob(100,3)
[X, Y] = two_moons(200,1,2)
plot_similarity_graph(X, Y, eps=eps, k=0, var=0.5)
plot_similarity_graph(X, Y, eps=0, k=10, var=0.5)
