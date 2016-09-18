import numpy as np
from MnistData import load_Mnist_Data as ld
def concate_bias(data):
	bias_param=np.ones(data.shape[0])
	bias_param=bias_param.reshape(bias_param.shape[0],1)
	data=np.concatenate((bias_param,data),axis=1)
	return data

def logistic_function(Z):
	return np.reciprocal(1+np.exp(-Z))

def cost_function(h,label):
	cost_func=-(label*np.log(h)+(1-label)*np.log(1-h))
	return cost_func
def deri_cost_function(h,label):
	deri_cost_func=-label*np.reciprocal(h)+(1-label)*np.reciprocal(1-h)
	return deri_cost_func
def logistic_regression(data,label,epoches,threshold,lrate):
	data=concate_bias(data)
	n,m=data.shape
	weight=np.random.random((m,1))*2-1
	for epoch in range(epoches):
		z=np.dot(data,weight)
		h=logistic_function(z)
		cost_func=cost_function(h,label)
		cost=np.sum(cost_func)/n
		print "epochs:%d,error_rate:%f" %(epoch,cost)
		if cost<=threshold:
			print "condition satisfied."
		#deri_cost=deri_cost_function(h,label)
		deri_cost=h-label
		delta_weight=deri_cost*data
		acc_weight=np.sum(delta_weight,axis=0)/n
		acc_weight=acc_weight.reshape(acc_weight.shape[0],1)
		weight-=lrate*acc_weight
	return weight


def test_Mnist(data_num):
	train_images,train_labels=ld.load_Binary_Mnist_Train_Data(data_num)
	train_labels=train_labels.reshape(train_labels.shape[0],1)
	train_images=train_images.astype(np.float32)/255 #normalization
	weight=logistic_regression(train_images,train_labels,3000,0.0001,0.3)
	return weight

def valid_Mnist(data_num,weight):
	test_images,test_labels=ld.load_Binary_Mnist_Test_Data(data_num)
	test_labels=test_labels.reshape(test_labels.shape[0],1)
	test_images=test_images.astype(np.float32)/255
	test_images=concate_bias(test_images)
	z=np.dot(test_images,weight)
	h=logistic_function(z)
	h[h>0.5]=1
	h[h<=0.5]=0
	error_num=abs(test_labels-h)
	num=np.sum(error_num)
	error_rate=1.0*num/test_labels.shape[0]
	print "error_rate:%f" %(error_rate)

if __name__=="__main__":
	weight=test_Mnist(60000)
	valid_Mnist(10000,weight)
	train_data=np.array([[0.1,0.2,0.3],[0.2,0.3,0.4],[0.3,0.4,0.5],[0.4,0.5,0.6],[0.6,0.7,0.8],[0.7,0.8,0.9]],np.float32)
	label_data=np.array([0,0,0,1,1,1])
	label_data=label_data.reshape(label_data.shape[0],1)
	#weight=logistic_regression(train_data,label_data,500000,0.000001,0.05)
	#print weight

