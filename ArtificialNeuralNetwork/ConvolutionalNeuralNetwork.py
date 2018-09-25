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
	num_epoch: number of epochs
	learning_rate:
	total_error:
	alpha:
	"""
	def __init__(self, num_epoch=500, learning_rate=0.0001, total_error=0, alpha=0.00008):

		self.num_epoch = num_epoch
		self.learning_rate = learning_rate
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
		self.l1aIN, self.l1bIN = np.pad(data, 1, mode='constant'), np.pad(data, 1, mode='constant')

		self.l1a = convolve2d(self.l1aIN, self.w1a, mode='valid')
		self.l1aA = self.ReLU(self.l1a)
		self.l1aM = skimage.measure.block_reduce(self.l1aA, block_size=(2, 2), func=np.max)

		self.l1b = convolve2d(self.l1bIN, self.w1b, mode='valid')
		self.l1bA = self.arctanh(self.l1b)
		self.l1bM = skimage.measure.block_reduce(self.l1bA, block_size=(2, 2), func=np.max)
		return

	def layer_2(self):
		# Blue Star → Layer 2 with four difference channels
		# Blue Circle → Activation and Max Pooling operation Applied to Layer 2

		self.l2aIN, self.l2bIN = np.pad(self.l1aM, 1, mode='constant'), np.pad(self.l1aM, 1, mode='constant')
		self.l2cIN, self.l2dIN = np.pad(self.l1bM, 1, mode='constant'), np.pad(self.l1bM, 1, mode='constant')

		self.l2a = convolve2d(self.l2aIN, self.w2a, mode='valid')
		self.l2aA = self.arctanh(self.l2a)
		self.l2aM = skimage.measure.block_reduce(self.l2aA, block_size=(2, 2), func=np.max)

		self.l2b = convolve2d(self.l2bIN, self.w2b, mode='valid')
		self.l2bA = self.ReLU(self.l2b)
		self.l2bM = skimage.measure.block_reduce(self.l2bA, block_size=(2, 2), func=np.max)		

		self.l2c = convolve2d(self.l2cIN, self.w2c, mode='valid')
		self.l2cA = self.arctanh(self.l2c)
		self.l2cM = skimage.measure.block_reduce(self.l2cA, block_size=(2, 2), func=np.max)

		self.l2d = convolve2d(self.l2dIN, self.w2d, mode='valid')
		self.l2dA = self.ReLU(self.l2d)
		self.l2dM = skimage.measure.block_reduce(self.l2dA, block_size=(2, 2), func=np.max)
		return

	def layer_3(self):
		# Green Star → Layer 3 with Fully Connected Weight (W3) Dimension of (16*28)
		# Green Circle → Activation Function Applied to Layer 3

		self.l3IN = np.expand_dims(np.hstack([self.l2aM.ravel(), self.l2bM.ravel(), self.l2cM.ravel(), self.l2dM.ravel()]), axis=0)

		self.l3 = self.l3IN.dot(self.w3)
		self.l3A = self.arctanh(self.l3)
		return

	def layer_4(self):
		# Pink Star → Layer 4 with Fully Connected Weight (W4) Dimension of (28*36)
		# Pink Circle → Activation Layer Applied to Layer 4

		self.l4 = self.l3A.dot(self.w4)
		self.l4A = self.tanh(self.l4)
		return

	def layer_5(self):
		# Black Star → Layer 5 with Fully Connected Weight (W5) Dimension of (36*1)
		# Black Circle → Activation Layer Applied to Layer 5

		self.l5 = self.l4A.dot(self.w5)
		self.l5A = self.log(self.l5)
		return


	def forwardProp(self, current_data):
		"""
		"""
		self.layer_1(current_data)

		self.layer_2()

		self.layer_3()

		self.layer_4()

		self.layer_5()



	def backProp(self, current_label):
		"""
		"""
		# Black Box → Cost Function using the L2 Norm
		cost = np.square(self.l5A - current_label).sum() * 0.5
		self.total_error += cost

		# Black → Layer 5
		grad_5_part_1 = self.l5A - current_label
		grad_5_part_2 = self.d_log(self.l5)
		grad_5_part_3 = self.l4A
		self.grad_5 = grad_5_part_3.T.dot(grad_5_part_1 * grad_5_part_2)


		# Pink → Layer 4
		grad_4_part_1 = (grad_5_part_1 * grad_5_part_2).dot(self.w5.T)
		grad_4_part_2 = self.d_tanh(self.l4)
		grad_4_part_3 = self.l3A
		self.grad_4 = grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

		# Green → Layer 3
		grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(self.w4.T)
		grad_3_part_2 = self.d_arctan(self.l3)
		grad_3_part_3 = self.l3IN
		self.grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

		# Back Propagation respect to all of W2 (W2a, W2b, W2c, and W2d) Implemented

		# First line of code (Underlined Green) → Back Propagating from previous layer 
		grad_2_part_IN = (grad_3_part_1 * grad_3_part_2).dot(self.w3.T)

		# Green Boxed Region → Calculating Gradient Respect to W2a, W2b, W2c, and W2d in their respective order.
		grad_2_window_a = np.reshape(grad_2_part_IN[:, :4], (2,2))
		grad_2_mask_a = np.equal(self.l2aA, self.l2aM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
		grad_2_part_1_a = grad_2_mask_a * grad_2_window_a.repeat(2, axis=0).repeat(2, axis=1)
		grad_2_part_2_a = self.d_arctan(self.l2a)
		grad_2_part_3_a = self.l2aIN
		self.grad_2_a = np.rot90(convolve2d(grad_2_part_3_a, np.rot90(grad_2_part_2_a * grad_2_part_1_a, 2), mode='valid'), 2)

		grad_2_window_b = np.reshape(grad_2_part_IN[:, 4:8], (2, 2))
		grad_2_mask_b = np.equal(self.l2bA, self.l2bM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
		grad_2_part_1_b = grad_2_mask_b * grad_2_window_b.repeat(2, axis=0).repeat(2, axis=1)
		grad_2_part_2_b = self.d_ReLU(self.l2b)
		grad_2_part_3_b = self.l2bIN
		self.grad_2_b = np.rot90(convolve2d(grad_2_part_3_b, np.rot90(grad_2_part_2_b * grad_2_part_1_b, 2), mode='valid'), 2)

		grad_2_window_c = np.reshape(grad_2_part_IN[:, 8:12], (2, 2))
		grad_2_mask_c = np.equal(self.l2cA, self.l2cM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
		grad_2_part_1_c = grad_2_mask_c * grad_2_window_c.repeat(2, axis=0).repeat(2, axis=1)
		grad_2_part_2_c = self.d_arctan(self.l2c)
		grad_2_part_3_c = self.l2cIN
		self.grad_2_c = np.rot90(convolve2d(grad_2_part_3_c, np.rot90(grad_2_part_2_c * grad_2_part_1_c, 2), mode='valid'), 2)

		grad_2_window_d = np.reshape(grad_2_part_IN[:, 12:16], (2, 2))
		grad_2_mask_d = np.equal(self.l2dA, self.l2dM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
		grad_2_part_1_d = grad_2_mask_d * grad_2_window_d.repeat(2, axis=0).repeat(2, axis=1)
		grad_2_part_2_d = self.d_tanh(self.l2d)
		grad_2_part_3_d = self.l2dIN
		self.grad_2_d = np.rot90(convolve2d(grad_2_part_3_d, np.rot90(grad_2_part_2_d * grad_2_part_1_d, 2), mode='valid'), 2)

		# Back Propagation Respect to all of W1 (W1a and W1b) Implemented

		# First Red Box → Gradient Respect to W1a
		grad_1_part_IN_a = np.rot90(grad_2_part_1_a * grad_2_part_2_a, 2)
		grad_1_part_IN_a_padded = np.pad(self.w2a, 2, mode='constant')
		grad_1_part_a = convolve2d(grad_1_part_IN_a_padded, grad_1_part_IN_a, mode='valid')

		grad_1_part_IN_b = np.rot90(grad_2_part_1_b * grad_2_part_2_b, 2)
		grad_1_part_IN_b_padded = np.pad(self.w2b, 2, mode='constant')
		grad_1_part_b = convolve2d(grad_1_part_IN_b_padded, grad_1_part_IN_b, mode='valid')		

		grad_1_window_a = grad_1_part_a + grad_1_part_b
		grad_1_mask_a = np.equal(self.l1aA, self.l1aM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
		grad_1_part_1_a = grad_1_mask_a * grad_1_window_a.repeat(2, axis=0).repeat(2, axis=1)
		grad_1_part_2_a = self.d_ReLU(self.l1a)
		grad_1_part_3_a = self.l1aIN
		self.grad_1_a = np.rot90(convolve2d(grad_1_part_3_a, np.rot90(grad_1_part_1_a * grad_1_part_2_a, 2), mode='valid'), 2)

		# Second Red Box → Gradient Respect to W2a
		grad_1_part_IN_c = np.rot90(grad_2_part_1_c * grad_2_part_2_c, 2)
		grad_1_part_IN_c_padded = np.pad(self.w2c, 2, mode='constant')
		grad_1_part_c = convolve2d(grad_1_part_IN_c_padded, grad_1_part_IN_c, mode='valid')

		grad_1_part_IN_d = np.rot90(grad_2_part_1_d * grad_2_part_2_d, 2)
		grad_1_part_IN_d_padded = np.pad(self.w2d, 2, mode='constant')
		grad_1_part_d = convolve2d(grad_1_part_IN_d_padded, grad_1_part_IN_d, mode='valid')

		grad_1_window_b = grad_1_part_c + grad_1_part_d
		grad_1_mask_b = np.equal(self.l1bA, self.l1bM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
		grad_1_part_1_b = grad_1_mask_b * grad_1_window_b.repeat(2, axis=0).repeat(2, axis=1)
		grad_1_part_2_b = self.d_arctan(self.l1b)
		grad_1_part_3_b = self.l1bIN
		self.grad_1_b = np.rot90(convolve2d(grad_1_part_3_b, np.rot90(grad_1_part_1_b * grad_1_part_2_b, 2), mode='valid'), 2)
		return

	def update(self):
		# ?????
		self.v5 = self.alpha * self.v5 + self.learning_rate * self.grad_5
		self.v4 = self.alpha * self.v4 + self.learning_rate * self.grad_4
		self.v3 = self.alpha * self.v3 + self.learning_rate * self.grad_3

		self.v2a = self.alpha * self.v2a + self.learning_rate * self.grad_2_a
		self.v2b = self.alpha * self.v2b + self.learning_rate * self.grad_2_b
		self.v2c = self.alpha * self.v2c + self.learning_rate * self.grad_2_c
		self.v2d = self.alpha * self.v2d + self.learning_rate * self.grad_2_d

		self.v1a = self.alpha * self.v1a + self.learning_rate * self.grad_1_a
		self.v1b = self.alpha * self.v1b + self.learning_rate * self.grad_1_b

		self.w5 = self.w5 - self.v5
		self.w4 = self.w4 - self.v4
		self.w3 = self.w3 - self.v3

		self.w2a = self.w2a - self.v2a
		self.w2b = self.w2b - self.v2b
		self.w2c = self.w2c - self.v2c
		self.w2d = self.w2d - self.v2d

		self.w1a = self.w1a - self.v1a
		self.w1b = self.w1b - self.v1b
		return







