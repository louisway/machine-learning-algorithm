import struct
import numpy as np
import os
base_path=os.path.dirname(os.path.realpath(__file__))
def load_data(filename_image,filename_label,data_num):
	images_num=784*data_num
	images_num=str(images_num)
	images_num='>'+images_num+'B'
	labels_num=str(data_num)
	labels_num='>'+labels_num+'B'

	images_file=open(filename_image,'rb')
	images_buf=images_file.read()
	index=0
	magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',images_buf,index)
	index+=struct.calcsize('>IIII')
	images=struct.unpack_from(images_num,images_buf,index)
	index+=struct.calcsize('>784B')
	images=np.array(images)
	dim=images.shape[0]
	col_dim=784
	row_dim=dim/col_dim
	images=images.reshape(row_dim,col_dim)

	labels_file=open(filename_label,'rb')
	labels_buf=labels_file.read()
	index=0
	magic,numItems=struct.unpack_from('>II',labels_buf,index)
	index+=struct.calcsize('>II')
	labels=struct.unpack_from(labels_num,labels_buf,index)
	labels=np.array(labels)
	labels=labels.reshape(labels.shape[0])
	return images,labels
	
def join_Path(filename):
	return base_path+'/'+filename

def load_Binary_Mnist_Train_Data(data_num):
	filename_train_image=join_Path('train-images.idx3-ubyte')
	filename_train_label=join_Path('train-labels.idx1-ubyte')
	images,labels=load_data(filename_train_image,filename_train_label,data_num)
	index=np.where((labels==1)|(labels==0))[0]
	images=images[index]
	labels=labels[index]
	return images,labels


def load_Binary_Mnist_Test_Data(data_num):
	filename_test_image=join_Path('t10k-images.idx3-ubyte')
	filename_test_label=join_Path('t10k-labels.idx1-ubyte')
	images,labels=load_data(filename_test_image,filename_test_label,data_num)
	index=np.where((labels==1)|(labels==0))[0]
	images=images[index]
	labels=labels[index]
	return images,labels





