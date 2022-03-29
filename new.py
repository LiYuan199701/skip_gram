import glob
import os.path as op
import pickle
import random

import numpy as np

import numpy as np
from numpy.core.defchararray import center
from utils.gradcheck import gradcheck_naive
from utils.utils import softmax
from utils.utils import normalize_rows

def dummy():
    random.seed(31415)
    np.random.seed(9265)

    dataset = type('dummy', (), {})()

    def dummy_sample_token_idx():
        return random.randint(0, 4)

    def get_random_context(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset.sample_token_idx = dummy_sample_token_idx
    dataset.get_random_context = get_random_context

    dummy_vectors = normalize_rows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    return dataset, dummy_vectors, dummy_tokens

inputs = {
    'test_word2vec': {
        'current_center_word': "c",
        'window_size': 3,
        'outside_words': ["a", "b", "e", "d", "b", "c"]
    },
    'test_naivesoftmax': {
        'center_word_vec': np.array([-0.27323645, 0.12538062, 0.95374082]).astype(float),
        'outside_word_idx': 3,
        'outside_vectors': np.array([[-0.6831809, -0.04200519, 0.72904007],
                                    [0.18289107, 0.76098587, -0.62245591],
                                    [-0.61517874, 0.5147624, -0.59713884],
                                    [-0.33867074, -0.80966534, -0.47931635],
                                    [-0.52629529, -0.78190408, 0.33412466]]).astype(float)

    },
    'test_sigmoid': {
        'x': np.array([-0.46612273, -0.87671855, 0.54822123, -0.36443576, -0.87671855, 0.33688521
                          , -0.87671855, 0.33688521, -0.36443576, -0.36443576, 0.54822123]).astype(float)
    }
}

outputs = {
    'test_word2vec': {
        'loss': 11.16610900153398,
        'dj_dv': np.array(
            [[0., 0., 0.],
             [0., 0., 0.],
             [-1.26947339, -1.36873189, 2.45158957],
             [0., 0., 0.],
             [0., 0., 0.]]).astype(float),
        'dj_du': np.array(
            [[-0.41045956, 0.18834851, 1.43272264],
             [0.38202831, -0.17530219, -1.33348241],
             [0.07009355, -0.03216399, -0.24466386],
             [0.09472154, -0.04346509, -0.33062865],
             [-0.13638384, 0.06258276, 0.47605228]]).astype(float)

    },
    'test_naivesoftmax': {
        'loss': 2.217424877675181,
        'dj_dvc': np.array([-0.17249875, 0.64873661, 0.67821423]).astype(float),
        'dj_du': np.array([[-0.11394933, 0.05228819, 0.39774391],
                           [-0.02740743, 0.01257651, 0.09566654],
                           [-0.03385715, 0.01553611, 0.11817949],
                           [0.24348396, -0.11172803, -0.84988879],
                           [-0.06827005, 0.03132723, 0.23829885]]).astype(float)
    },
    'test_sigmoid': {
        's': np.array(
            [0.38553435, 0.29385824, 0.63372281, 0.40988622, 0.29385824, 0.5834337, 0.29385824, 0.5834337, 0.40988622,
             0.40988622, 0.63372281]).astype(float),
    }
}

sample_vectors_expected = {
    "female": [
        0.6029723815239835,
        0.16789318536724746,
        0.22520087305967568,
        -0.2887330648792561,
        -0.914615719505456,
        -0.2206997036383445,
        0.2238454978107194,
        -0.27169214724889107,
        0.6634932978039564,
        0.2320323110106518
    ],
    "cool": [
        0.5641256072125872,
        0.13722982658305444,
        0.2082364803517175,
        -0.2929695723456364,
        -0.8704480862547578,
        -0.18822962799771015,
        0.24239616047158674,
        -0.29410091959922546,
        0.6979644655991716,
        0.2147529764765611
    ]
}

def test_naive_softmax_loss_and_gradient():
    print("\t\t\tnaive_softmax_loss_and_gradient\t\t\t")

    dataset, dummy_vectors, dummy_tokens = dummy()

    print("\nYour Result:")
    loss, dj_dvc, dj_du = naive_softmax_loss_and_gradient(
        inputs['test_naivesoftmax']['center_word_vec'],
        inputs['test_naivesoftmax']['outside_word_idx'],
        inputs['test_naivesoftmax']['outside_vectors'],
        dataset
    )

    print(
        "Loss: {}\nGradient wrt Center Vector (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(loss,
                                                                                                                  dj_dvc,
                                                                                                                  dj_du))

    print("Expected Result: Value should approximate these:")
    print(
        "Loss: {}\nGradient wrt Center Vectors(dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            outputs['test_naivesoftmax']['loss'],
            outputs['test_naivesoftmax']['dj_dvc'],
            outputs['test_naivesoftmax']['dj_du']))
    return (outputs['test_naivesoftmax']['loss'], outputs['test_naivesoftmax']['dj_dvc'], outputs['test_naivesoftmax']['dj_du']), (loss, dj_dvc, dj_du)

def sigmoid(x):
  """
  Compute the sigmoid function for the input here.
  Arguments:
  x -- A scalar or numpy array.
  Return:
  s -- sigmoid(x)
  """

  ### START CODE HERE
  s = 1 / (1 + np.exp(-x)) 
  ### END CODE HERE

  return s



def naive_softmax_loss_and_gradient(center_word_vec,outside_word_idx,outside_vectors,dataset):
  """ Naive Softmax loss & gradient function for word2vec models

  Implement the naive softmax loss and gradients between a center word's 
  embedding and an outside word's embedding. This will be the building block
  for our word2vec models.

  Arguments:
  center_word_vec -- numpy ndarray, center word's embedding
                  (v_c in the pdf handout)
  outside_word_idx -- integer, the index of the outside word
                  (o of u_o in the pdf handout)
  outside_vectors -- outside vectors (rows of matrix) for all words in vocab
                    (U in the pdf handout)
  dataset -- needed for negative sampling, unused here.

  Return:
  loss -- naive softmax loss
  grad_center_vec -- the gradient with respect to the center word vector
                   (dJ / dv_c in the pdf handout)
  grad_outside_vecs -- the gradient with respect to all the outside word vectors
                  (dJ / dU)
                  
   Note:
   we usually use column vector convention (i.e., vectors are in column form) for vectors in matrix U and V (in the handout)
   but for ease of implementation/programming we usually use row vectors (representing vectors in row form).
  """

  ### Please use the provided softmax function (imported earlier in this file)
  ### This numerically stable implementation helps you avoid issues pertaining
  ### to integer overflow.
  ### START CODE HERE
  new_out = np.transpose(outside_vectors)
  dot_product = center_word_vec @ new_out
  soft = softmax(dot_product)
  loss = -1 * np.log(soft[outside_word_idx])
  
  y = np.zeros(shape=(outside_vectors.shape[0]))
  y[outside_word_idx] = 1
  diff = (soft - y)
  grad_center_vec = np.transpose(new_out @ diff)
  
  grad_outside_vecs = np.transpose(np.reshape(center_word_vec, (len(center_word_vec), 1)) @ np.reshape(diff, (1, len(diff))))
  ### END CODE HERE

  return loss, grad_center_vec, grad_outside_vecs


def neg_sampling_loss_and_gradient(center_word_vec,outside_word_idx,outside_vectors,dataset,K=10):
  """ Negative sampling loss function for word2vec models

   Arguments/Return Specifications: same as naive_softmax_loss_and_gradient
   K is the number of negative samples to take.

   """

  neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
  indices = [outside_word_idx] + neg_sample_word_indices

  grad_center_vec = np.zeros(center_word_vec.shape)
  grad_outside_vecs = np.zeros(outside_vectors.shape)

  labels = np.array([1] + [-1 for k in range(K)])
  vecs = outside_vectors[indices, :]

  t = sigmoid(vecs.dot(center_word_vec) * labels)
  loss = -np.sum(np.log(t))

  delta = labels * (t - 1)
  grad_center_vec = delta.reshape((1, K + 1)).dot(vecs).flatten()
  grad_outside_vecs_temp = delta.reshape((K + 1, 1)).dot(center_word_vec.reshape(
    (1, center_word_vec.shape[0])))
  for k in range(K + 1):
    grad_outside_vecs[indices[k]] += grad_outside_vecs_temp[k, :]

  return loss, grad_center_vec, grad_outside_vecs

def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset, window_size, word2vec_loss_and_gradient=neg_sampling_loss_and_gradient):
  batchsize = 50
  loss = 0.0
  grad = np.zeros(word_vectors.shape)
  N = word_vectors.shape[0]
  center_word_vectors = word_vectors[:int(N / 2), :]
  outside_vectors = word_vectors[int(N / 2):, :]
  for i in range(batchsize):
    window_size_1 = random.randint(1, window_size)
    center_word, context = dataset.get_random_context(window_size_1)

    c, gin, gout = word2vec_model(center_word, window_size_1, context, word2ind, center_word_vectors,outside_vectors, dataset, word2vec_loss_and_gradient)
    loss += c / batchsize
    grad[:int(N / 2), :] += gin / batchsize
    grad[int(N / 2):, :] += gout / batchsize

  return loss, grad

def skipgram(current_center_word, window_size, outside_words, word2ind, center_word_vectors, outside_vectors, dataset, word2vec_loss_and_gradient=neg_sampling_loss_and_gradient):
  """ Skip-gram model in word2vec

  Implement the skip-gram model in this function.

  Arguments:
  current_center_word -- a string of the current center word
  window_size -- integer, context window size
  outside_words -- list of no more than 2*window_size strings, the outside words
  word2ind -- a dictionary that maps words to their indices in
            the word vector list
  center_word_vectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
  outside_vectors -- outside word vectors (as rows) for all words in vocab
                  (U in pdf handout)
  word2vec_loss_and_gradient -- the loss and gradient function for
                             a prediction vector given the outsideWordIdx
                             word vectors, could be one of the two
                             loss functions you implemented above (do not hardcode any of them).

  Return:
  loss -- the loss function value for the skip-gram model
          (J in the pdf handout)
  grad_center_vecs -- the gradient with respect to the center word vectors
          (dJ / dV in the pdf handout)
  grad_outside_vectors -- the gradient with respect to the outside word vectors
                      (dJ / dU in the pdf handout)
  """

  loss = 0.0
  grad_center_vecs = np.zeros(center_word_vectors.shape)
  grad_outside_vectors = np.zeros(outside_vectors.shape)

  ### START CODE HERE
  c_inx = word2ind[current_center_word]
  for word in outside_words:
      loss_current, gradc, grado = word2vec_loss_and_gradient(center_word_vectors[c_inx, :], word2ind[word], outside_vectors, dataset)
      loss = loss + loss_current
      grad_center_vecs[c_inx, :] = grad_center_vecs[c_inx, :] + gradc
      grad_outside_vectors = grad_outside_vectors + grado
  ### END CODE HERE

  return loss, grad_center_vecs, grad_outside_vectors


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset, dummy_vectors, dummy_tokens = dummy()

    print("==== Gradient check for skip-gram with naive_softmax_loss_and_gradient ====")
    gradcheck_passed = gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naive_softmax_loss_and_gradient),
                    dummy_vectors, "naive_softmax_loss_and_gradient Gradient")

    print("\n\t\t\tSkip-Gram with naive_softmax_loss_and_gradient\t\t\t")

    print("\nYour Result:")
    loss, dj_dv, dj_du = skipgram(inputs['test_word2vec']['current_center_word'], inputs['test_word2vec']['window_size'],
                                    inputs['test_word2vec']['outside_words'],
                                    dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                                    naive_softmax_loss_and_gradient)
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(loss,
                                                                                                                    dj_dv,
                                                                                                                    dj_du))

    print("Expected Result: Value should approximate these:")
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            outputs['test_word2vec']['loss'],
            outputs['test_word2vec']['dj_dv'],
            outputs['test_word2vec']['dj_du']))
    return gradcheck_passed, (outputs['test_word2vec']['loss'], outputs['test_word2vec']['dj_dv'], outputs['test_word2vec']['dj_du']), (loss, dj_dv, dj_du)


test_word2vec()
#test_naive_softmax_loss_and_gradient()

#print(inputs['test_word2vec']['current_center_word'].shape)
#print(dummy())