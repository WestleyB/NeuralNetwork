# -*- coding: utf-8 -*-
#!/usr/bin/python

# https://medium.com/swlh/only-numpy-why-i-do-manual-back-propagation-implementing-multi-channel-layer-convolution-neural-7d83242fcc24

import numpy as np
from sklearn.datasets import load_digits
from scipy.ndimage.filters import maximum_filter
import skimage.measure
from scipy.signal import convolve2d
from scipy import fftpack
from sklearn.utils import shuffle


np.random.seed(42)


class ConvolutionalNeuralNetwork:
	"""
	Params:
	data:
	labels:
	num_epoch: number of epochs
	learning_rate:
	total_error:
	alpha:
	"""
	def __init__(self, data, labels, num_epoch=500, learning_rate=0.0001, total_error=0, alpha=0.00008):

		self.data = data
		self.labels = labels

		self.num_epoch = num_epoch
		self.lr = learning_rate
		self.total_error = total_error
		self.alpha = alpha

		# Weight matrix for Layer 1 with two different channels
		self.w1a = np.random.randn(3,3)
		self.w1b = np.random.randn(3,3)

		# Weight matrix for Layer 2 with four difference channels
		self.w2a = np.random.randn(3,3)
		self.w2b = np.random.randn(3,3)
		self.w2c = np.random.randn(3,3)
		self.w2d = np.random.randn(3,3)

		# Weight matrix for Layer 3 with Fully Connected Weight (W3) Dimension of (16*28)
		self.w3 = np.random.randn(16,28)
		# Weight matrix for Layer 4 with Fully Connected Weight (W4) Dimension of (28*36)
		self.w4 = np.random.randn(28,36)
		# Weight matrix for Layer 5 with Fully Connected Weight (W5) Dimension of (36*1)
		self.w5 = np.random.randn(36,1)

		self.v1a = 0
		self.v1b = 0
		
		self.v2a = 0 
		self.v2b = 0
		self.v2c = 0
		self.v2d = 0
		
		self.v3 = 0
		self.v4 = 0
		self.v5 = 0


	# Activation function.
	def ReLU(self, x):
		mask  = (x >0) * 1.0 
		return mask * x

	# Activation function.
	def d_ReLU(self, x):
		mask  = (x >0) * 1.0 
		return mask 

	# Activation function.
	def tanh(self, x):
		return np.tanh(x)

	# Activation function.
	def d_tanh(self, x):
		return 1 - np.tanh(x) ** 2

	# Activation function.
	def arctanh(self, x):
		return np.arctan(x)

	# Activation function.
	def d_arctan(self, x):
		return 1 / ( 1 + x ** 2)

	# Activation function. simple nonlinearity, convert nums into probabilities between 0 and 1
	def log(self, x):
		return 1 / (1 + np.exp(-1 * x))

	# Derivative of the sigmoid (log) function. used to compute gradients for backpropagation
	def d_log(self, x):
		return self.log(x) * ( 1 - self.log(x))

	def layer_1(self, data):
		"""
		"""
		# Red Star → Layer 1 with two different channels
		# Red Circle → Activation and Max Pooling Layer Applied to Layer 1
		l1aIN, l1bIN = np.pad(data, 1, mode='constant'), np.pad(data, 1, mode='constant')

		l1a = convolve2d(l1aIN, self.w1a, mode='valid')
		l1aA = self.ReLU(l1a)
		l1aM = skimage.measure.block_reduce(l1aA, block_size=(2, 2), func=np.max)

		l1b = convolve2d(l1bIN, self.w1b, mode='valid')
		l1bA = self.arctanh(l1b)
		l1bM = skimage.measure.block_reduce(l1bA, block_size=(2, 2), func=np.max)

		return l1aM, l1bM

	def layer_2(self, l1aM, l1bM):
		# Blue Star → Layer 2 with four difference channels
		# Blue Circle → Activation and Max Pooling operation Applied to Layer 2

		l2aIN, l2bIN = np.pad(l1aM, 1, mode='constant'), np.pad(l1aM, 1, mode='constant')
		l2cIN, l2dIN = np.pad(l1bM, 1, mode='constant'), np.pad(l1bM, 1, mode='constant')

		l2a = convolve2d(l2aIN, self.w2a, mode='valid')
		l2aA = self.arctanh(l2a)
		l2aM = skimage.measure.block_reduce(l2aA, block_size=(2, 2), func=np.max)

		l2b = convolve2d(l2bIN, self.w2b, mode='valid')
		l2bA = self.ReLU(l2b)
		l2bM = skimage.measure.block_reduce(l2bA, block_size=(2, 2), func=np.max)		

		l2c = convolve2d(l2cIN, self.w2c, mode='valid')
		l2cA = self.arctanh(l2c)
		l2cM = skimage.measure.block_reduce(l2cA, block_size=(2, 2), func=np.max)

		l2d = convolve2d(l2dIN, self.w2d, mode='valid')
		l2dA = self.ReLU(l2d)
		l2dM = skimage.measure.block_reduce(l2dA, block_size=(2, 2), func=np.max)

		return l2aM, l2bM, l2cM, l2dM

	def layer_3(self, l2aM, l2bM, l2cM, l2dM):
		# Green Star → Layer 3 with Fully Connected Weight (W3) Dimension of (16*28)
		# Green Circle → Activation Function Applied to Layer 3

		l3IN = np.expand_dims(np.hstack([l2aM.ravel(), l2bM.ravel(), l2cM.ravel(), l2dM.ravel()]), axis=0)

		l3 = l3IN.dot(self.w3)
		l3A = self.arctanh(l3)

		return l3A

	def layer_4(self, l3A):
		# Pink Star → Layer 4 with Fully Connected Weight (W4) Dimension of (28*36)
		# Pink Circle → Activation Layer Applied to Layer 4

		l4 = l3A.dot(self.w4)
		l4A = self.tanh(l4)

		return l4A

	def layer_5(self, l4A):
		# Black Star → Layer 5 with Fully Connected Weight (W5) Dimension of (36*1)
		# Black Circle → Activation Layer Applied to Layer 5

		l5 = l4A.dot(self.w5)
		l5A = self.log(l5)

		return l5A


	def forwardProp(self, current_data):
		"""
		"""
		l1aM, l1bM = self.layer_1(current_data)

		l2aM, l2bM, l2cM, l2dM = self.layer_2(l1aM, l1bM)

		l3A = self.layer_3(l2aM, l2bM, l2cM, l2dM)

		l4A = self.layer_4(l3A)

		l5A = self.layer_5(l4A)

		return l5A


	def backProp(self, l5A, current_label):
		"""
		"""
		# Black Box → Cost Function using the L2 Norm
		cost = np.square(l5A - current_label).sum() * 0.5
		self.total_error += cost

		# Black → Layer 5
		grad_5_part_1 = l5A - current_label
		grad_5_part_2 = self.d_log(l5)
		grad_5_part_3 = l4A
		grad_5 = grad_5_part_3.T.dot(grad_5_part_1 * grad_5_part_2)


		# Pink → Layer 4
		grad_4_part_1 = (grad_5_part_1 * grad_5_part_2).dot(self.w5.T)
		grad_4_part_2 = self.d_tanh(l4)
		grad_4_part_3 = l3A
		grad_4 = grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

		# Green → Layer 3
		grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
		grad_3_part_2 = d_arctan(l3)
		grad_3_part_3 = l3IN
		grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)



	def sampling(self):
		"""
		"""
		for iter in range(self.num_epoch):
			for data_index in range(len(self.data)):

				current_data = self.data[data_index]
				current_label = self.labels[data_index]

				l5A = self.forwardProp(current_data)
				self.backProp(l5A, current_label)







def loadData(shuffle=True):
	# 1. Prepare data
	data = load_digits()
	image = data.images
	label = data.target

	# 1. Prepare only one and only zero + two, three
	only_zero_index = np.asarray(np.where(label == 0))
	only_one_index = np.asarray(np.where(label == 1))
	# only_two_index = np.asarray(np.where(label == 2))
	# only_three_index = np.asarray(np.where(label == 3))

	# 1.5 Prepare label
	only_zero_label = label[only_zero_index].T
	only_one_label = label[only_one_index].T
	# only_two_label = label[only_two_index].T
	# only_three_label = label[only_three_index].T
	image_label = np.vstack((only_zero_label, only_one_label)) # , only_two_label, only_three_label))

	# 2. Prepare matrix image
	only_zero_image = np.squeeze(image[only_zero_index])
	only_one_image = np.squeeze(image[only_one_index])
	# only_two_image = np.squeeze(image[only_two_index])
	# only_three_image = np.squeeze(image[only_three_index])
	image_matrix = np.vstack((only_zero_image, only_one_image)) # , only_two_image, only_three_image))	

	if shuffle:
		image_matrix, image_label = shuffle(image_matrix,image_label)

	image_test_label = image_label[:10]
	image_label = image_label[10:]

	image_test_matrix = image_matrix[:10,:,:]
	image_matrix = image_matrix[10:,:,:]

	return image_matrix, image_label, image_test_matrix, image_test_label



