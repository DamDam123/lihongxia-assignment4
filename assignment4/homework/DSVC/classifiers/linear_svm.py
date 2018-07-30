import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.（权重）
  - X: A numpy array of shape (N, D) containing a minibatch of data.训练集的子集
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means（训练子集对应的标签）
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    #关注不正确分数和正确分数之间的差值
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # for i in range(num_train):
  #   scores = X[i].dot(W)
  #   correct_class_score = scores[y[i]]
  #   for j in range(num_classes):
  #     if j == y[i]:
  #       continue
  #     margin = scores[j] - correct_class_score + 1
  #     if margin > 0 :
  #       loss += margin
  #       dW[:,y[i]] += -X[i,:]
  #       dW[:,j] += X[i,:]
  # loss /= num_train
  # dW /= num_train
  # dW = dW+reg*W
  # W -= dW * 0.01
  # loss += reg * np.sum(W * W)

  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score +1
      if margin>0:
        loss += margin
        dW[:,y[i]] += -X[i,:]
        dW[:,j] += X[i,:]
  loss /= num_train
  dW /= num_train
  dW += reg*W
  # W -= dW*0.001
  loss += reg*np.sum(W*W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # num_train = X.shape[0]
  # num_scores = num_train-1
  # scores = X.dot(W)
  # correct_scores = scores[num_scores,y]
  # correct_scores = np.reshape(correct_scores,(-1,1))
  # margin = scores - correct_scores + 1
  # margin [margin<0] = 0
  # loss = np.sum(margin)/num_train

  num_train = X.shape[0]
  num_train_index = num_train - 1
  scores = X.dot(W)
  correct_class_scores = scores[num_train_index,y]
  correct_class_scores = np.reshape(correct_class_scores,(-1,1))
  margins = scores - correct_class_scores +1.0
  margins[margins<=0] = 0
  margins[np.arange(num_train),y] = 0
  loss = np.sum(margins)/num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins_one = margins
  margins_one[margins_one>0] = 1.0
  sum = np.sum(margins_one,axis=1)
  margins_one[np.arange(num_train),y] = -sum
  dW += X.T.dot(margins_one)/num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
