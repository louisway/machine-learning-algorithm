# -*- coding:utf-8 -*-
import numpy as np
import sys
import getopt
def tangent_function(Z):
	positive=np.exp(Z)
	negative=np.exp(-Z)
	return (positive-negative)/(positive+negative)
def deri_tangent_function(Z):
	fz=tangent_function(Z)
	return 1-np.square(fz)
def max_function(Z):
	Z[Z<0]=0
	return Z
def deri_max_function(Z):
	Z[Z>0]=1
	Z[Z<=0]=0
	return Z
def logistic_function(Z):
	return np.reciprocal(1+np.exp(-Z))
def deri_logistic_function(Z):
	return logistic_function(Z)-np.square(logistic_function(Z))
def concate_bias(data):
	bias_param=np.ones(data.shape[0])
	bias_param=bias_param.reshape(bias_param.shape[0],1)
	data=np.concatenate((bias_param,data),axis=1)
	return data
def forward_cast(inputs,weights):
	net=[]
	for i in range(len(weights)-1):
		data_input=inputs[i]
		weight=weights[i]
		inter_output=np.dot(data_input,weight)
		net.append(inter_output)
		inter_output=logistic_function(inter_output)
		inter_output=concate_bias(inter_output)
		#bias_param=np.ones(inter_output.shape[0])
		#bias_param=bias_param.reshape(bias_param.shape[0],1)
		#inter_output=np.concatenate((bias_param,inter_output),axis=1)
		inputs.append(inter_output)
	net_output=np.dot(inputs[-1],weights[-1])
	net.append(net_output)
	inputs.append(logistic_function(net_output))
	return inputs,weights,net

def error_cal(output_data,label_data):
#with Loss function of square error cost function J=1/2|| h-y ||^2
#without using regularization
	output_data=output_data.reshape(1,output_data.shape[0]).T
	label_data=label_data.reshape(1,label_data.shape[0]).T
	bias=label_data-output_data
	sec=np.sum(np.square(bias),axis=0)/(2*output_data.shape[0])
	return bias,sec

def backward_propa(inputs,weights,bias,net):
#using gradient descent
	num_data=inputs[0].shape[0]
	output_dim=weights[-1].shape[1]
	delta_weights=[]
	for weight in weights:
		n,m=weight.shape
		delta_weight=np.zeros((n,m))
		delta_weights.append(delta_weight)
	error=-(bias)*deri_logistic_function(net[-1])
	error=error.reshape(num_data,output_dim)
	#δ(nl)=−(y−a(nl))∙f′(z(nl))
	for i in range(num_data):
		error_propa=error[i,:].reshape(1,error.shape[1])
		for j in reversed(range(len(weights))):
			input_data=inputs[j][i,:]
			input_data=input_data.reshape(input_data.shape[0],1)
			delta_weights[j]+=np.dot(input_data,error_propa)
			#next error
			error_propa=np.dot(weights[j],error_propa.T)
			error_propa=error_propa[1:]
			error_propa=error_propa.reshape(1,error_propa.shape[0])
			#δ(l)=((W(l))Tδ(l+1))∙f′(z(l))
			error_propa*=deri_logistic_function(net[j-1][i,:])
	return delta_weights


def train_nn(data,label_data,units_num,epochs,threshold,lrate,activate_function="logistic_function"):
#assume the type of data is numpy.array
#assume the type of units_num is numpy.array
#assume units_num includes the number of cells in each layer except input layer 
	n,m=data.shape
	data=concate_bias(data)
	weights=[]
	row=m
	for col in units_num:
		weight=np.random.random((row+1,col))*2-1
		weights.append(weight)
		row=col

	for epoch in range(epochs):
		inputs=[]
		inputs.append(data)
		inputs,weight,net=forward_cast(inputs,weights)
		bias,error_rate=error_cal(inputs[-1],label_data)
		print "epochs:%d,error_rate:%f" %(epoch,error_rate)

		if(error_rate<=threshold):
			print "condition satisfied"
			break
		delta_weights=backward_propa(inputs,weights,bias,net)
		for i in range(len(weights)):
			weights[i]-=lrate*delta_weights[i]
		epoch
	return weights

def test(data,weights):
	data=concate_bias(data)
	for weight in weights:
		data=logistic_function(np.dot(data,weight))
		data=concate_bias(data)
	print data[:,1]

if __name__=="__main__":
	units_num=np.array([5,3,2,1])
	train_data=np.array([[0.1,0.2,0.3,0.4],[0.6,0.7,0.8,0.9],[0.2,0.3,0.44,0.25],[0.55,0.85,0.73,0.95]],np.float32)
	label_data=np.array([0,1,0,1])
	weights=train_nn(train_data,label_data,units_num,100000,0.000001,0.05)
	#weights, low level units are on the left, high level units are on the top
	test(train_data,weights)




	