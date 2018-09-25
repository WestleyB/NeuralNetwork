# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork


def loadData():
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

	image_matrix, image_label = shuffle(image_matrix,image_label)

	image_test_label = image_label[:10]
	image_label = image_label[10:]

	image_test_matrix = image_matrix[:10,:,:]
	image_matrix = image_matrix[10:,:,:]

	return image_matrix, image_label, image_test_matrix, image_test_label



if __name__ == "__main__":
	print('\n> Launch CNN...')
	start = time.time()

	num_epoch = 500

	image_matrix, image_label, image_test_matrix, image_test_label = loadData()

	CNN = ConvolutionalNeuralNetwork()

	for iter in range(num_epoch):
		for image_index in range(len(image_matrix)):

			current_image = image_matrix[image_index]
			current_image_label = image_label[image_index]

			CNN.forwardProp(current_image)
			CNN.backProp(current_image_label)
			CNN.update()

		print("\tCurrent iter : ", iter, " Current cost: ", CNN.total_error, end='\n')
		CNN.total_error = 0

	print('\n> End of CNN Training in {}'.format(round(time.time() - start, 2)))

	CNN.predict(image_test_matrix, image_test_label)

	print('\n> ...The End')

