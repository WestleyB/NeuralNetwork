# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
from RecurrentNeuralNetwork import RecurrentNeuralNetwork


def LoadText():
	#open text and return input and output data (series of words)
	with open("eminem.txt", "r") as text_file:
		data = text_file.read()
	text = list(data)
	outputSize = len(text)
	data = list(set(text))
	uniqueWords, dataSize = len(data), len(data) 
	returnData = np.zeros((uniqueWords, dataSize))
	for i in range(0, dataSize):
		returnData[i][i] = 1
	returnData = np.append(returnData, np.atleast_2d(data), axis=0)
	output = np.zeros((uniqueWords, outputSize))
	for i in range(0, outputSize):
		index = np.where(np.asarray(data) == text[i])
		output[:,i] = returnData[0:-1,index[0]].astype(float).ravel()  
	return returnData, uniqueWords, output, outputSize, data

#write the predicted output (series of words) to disk
def ExportText(output, data):
	finalOutput = np.zeros_like(output)
	prob = np.zeros_like(output[0])
	outputText = ""
	print(len(data))
	print(output.shape[0])
	for i in range(0, output.shape[0]):
		for j in range(0, output.shape[1]):
			prob[j] = output[i][j] / np.sum(output[i])
		outputText += np.random.choice(data, p=prob)    
	with open("output.txt", "w") as text_file:
		text_file.write(outputText)
	return

#Begin program
print("Beginning")
iterations = 5000
learningRate = 0.001
#load input output data (words)
returnData, numCategories, expectedOutput, outputSize, data = LoadText()
print("Done Reading")
#init our RNN using our hyperparams and dataset
RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate)

#training time!
for i in range(1, iterations):
	#compute predicted next word
	RNN.forwardProp()
	#update all our weights using our error
	error = RNN.backProp()
	#once our error/loss is small enough
	print("Error on iteration ", i, ": ", error)
	if error > -100 and error < 100 or i % 100 == 0:
		#we can finally define a seed word
		seed = np.zeros_like(RNN.x)
		maxI = np.argmax(np.random.random(RNN.x.shape))
		seed[maxI] = 1
		RNN.x = seed  
		#and predict some new text!
		output = RNN.sample()
		print(output)    
		#write it all to disk
		ExportText(output, data)
		print("Done Writing")
print("Complete")