import numpy as np
def concate_bias(data):
	bias_param=np.ones(data.shape[0])
	bias_param=bias_param.reshape(bias_param.shape[0],1)
	data=np.concatenate((bias_param,data),axis=1)
	return data
def linear_regression(data,label,epochs,threshold,lrate):
	data=concate_bias(data)
	n,m=data.shape
	weight=np.random.random((m,1))
	for epoch in range(epochs):
		h=np.dot(data,weight)
		bias=-(label-h)
		sum_bias=np.sum(np.square(bias),axis=0)/2
		print "epochs:%d,error_rate:%f" %(epoch,sum_bias)
		if sum_bias<threshold:
			print"satistified"
			break
		#print bias
		#print data
		delta_weight=bias*data
		acc_weight=lrate*np.sum(delta_weight,axis=0)
		acc_weight=acc_weight.reshape(acc_weight.shape[0],1)
		weight-=lrate*acc_weight
	return weight


if __name__=="__main__":
	train_data=np.array([[0.1],[0.2],[0.3],[0.4],[0.6],[0.7]],np.float32)
	label_data=np.array([0.1,0.2,0.3,0.4,0.6,0.7])
	label_data=label_data.reshape(label_data.shape[0],1)
	weight=linear_regression(train_data,label_data,100000,0.000001,0.03)
	print weight
	test_data=np.array([0.5]).reshape(1,1)
	test_data=concate_bias(test_data)
	print np.dot(test_data,weight)

