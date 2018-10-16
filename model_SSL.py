import os
import string
import numpy as np
from itertools import islice
import random
import csv
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import time
import json
import pandas as pd 
import pickle
import gzip
import struct
import numpy as np
from array import array
import glob 
import pickle
import gzip
#from PIL import Image 
#import scipy 
import random
import pdb

'''
    DCT_data: get labeled and unlabeled data for training based on ratio from csv data packet
    args: 
        fea: the directory that stores the csv files
        lab: the csv files
        preload: in current version, to reduce the indexing overhead of SGD, we load all the data into memeory at initialization.
        ratio: ratio of labeled data for training
    returns:
        (1) numpy array of feature tensors with shape: N x H x W x C
        (2) numpy array of labels with shape: N x 1
'''
def DCT_data(fea, lab, preload=False, ratio=None):
    dat=fea
    label=lab
    with open(lab) as f:
        maxlen=sum(1 for _ in f)
        
        if ratio:
            content = "adjust data from %d " % (maxlen)
            maxlen = int(maxlen*min(ratio, 1.0))
            content += " to %d (%g)" % (maxlen, ratio)
            #print(content)
    if preload:
        print("loading data into the main memory...")
        # get data according to ratio
        if not ratio == 1:    
            ft_buffer, label_buffer, ft_buffer_un, label_buffer_un=readcsv(dat, maxlen, ratio)
            return(ft_buffer, label_buffer, ft_buffer_un, label_buffer_un)
        else:
            ft_buffer, label_buffer=readcsv(dat, maxlen, ratio)
            return(ft_buffer, label_buffer)
    
'''
    readcsv: Read feature tensors from csv data packet
    args:
        target: the directory that stores the csv files
        fealen: the length of feature tensor, related to to discarded DCT coefficients
    returns: (1) numpy array of feature tensors with shape: N x H x W x C
             (2) numpy array of labels with shape: N x 1 
'''
def readcsv(target, maxlen, ratio, fealen=32):
    #read label
    path  = target + '/label.csv'
    label_all = np.genfromtxt(path, delimiter=',')
    #read feature
    feature_all = []
    for dirname, dirnames, filenames in os.walk(target):
        for i in xrange(0, len(filenames)-1):
            if i==0:
                file = '/dc.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature_all.append(featemp)
            else:
                file = '/ac'+str(i)+'.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature_all.append(featemp)          
    #for i in range(len(feature_all)):
    #    print(feature_all[i].shape)
    #adjust amount of training data 
    feature_all = np.rollaxis(np.asarray(feature_all), 0, 3)[:,:,0:fealen]
    if not ratio == 1:
        feature = feature_all[:maxlen]
        label = label_all[:maxlen]
        feature_un = feature_all[maxlen:]
        label_un = label_all[maxlen:]    
        return feature, label, feature_un, label_un
    else:
        return feature_all, label_all
'''
    processlabel: adjust ground truth for biased learning
    args:
        label: numpy array contains labels
        cato : number of classes in the task
        delta1: bias for class 1
        delta2: bias for class 2
    return: softmax label with bias
'''
def processlabel(label, cato=2, delta1 = 0, delta2=0):
    softmaxlabel=np.zeros(len(label)*cato, dtype=np.float32).reshape(len(label), cato)
    for i in range(0, len(label)):
        if int(label[i])==0:
            softmaxlabel[i,0]=1-delta1
            softmaxlabel[i,1]=delta1
        if int(label[i])==1:
            softmaxlabel[i,0]=delta2
            softmaxlabel[i,1]=1-delta2
    return softmaxlabel
'''
    loss_to_bias: calculate the bias term for batch biased learning
    args:
        loss: the average loss of current batch with respect to the label without bias
        threshold: start biased learning when loss is below the threshold
    return: the bias value to calculate the gradient
'''
def loss_to_bias(loss,  alpha, threshold=0.3):
    if loss >= threshold:
        bias = 0
    else:
        bias = 1.0/(1+np.exp(alpha*loss))
    return bias

'''
    forward_crosstask: define the multi-task neural network architecute
    args:
        input: feature tensor batch with size B x H x W x C
        is_training: whether the forward process is training, affect dropout layer
        reuse: undetermined
        scope: undetermined
    return: prediction socre(s) of input batch (both classification and clustering streams)
'''

def forward_crosstask(input, is_training=True, reuse=False, scope='model', flip=False):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            net = slim.conv2d(input, 16, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
            #crosstask part
            net1 = slim.conv2d(net, 32, [3, 3], scope='conv2_1_1')
            net1 = slim.conv2d(net1, 32, [3, 3], scope='conv2_2_1')
            net1 = slim.max_pool2d(net1, [2, 2], stride=2, padding='SAME', scope='pool2_1')
            net1 = slim.flatten(net1)
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
            net = slim.flatten(net)
            w_init = tf.contrib.layers.xavier_initializer(uniform=False)
            net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
            predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
            #crosstask part
            net1 = slim.fully_connected(net1, 250, activation_fn=tf.nn.relu, scope='fc1_1')
            net1 = slim.dropout(net1, 0.5, is_training=is_training, scope='dropout_1')
            predict1 = slim.fully_connected(net1, 2, activation_fn=None, scope='fc2_1')
    return predict,predict1

'''
    data: a class to handle the training and testing data, implement minibatch fetch
    args: 
        fea: feature tensor of whole data set
        lab: labels of whole data set
        ptr: a pointer for the current location of minibatch
        maxlen: length of entire dataset
        preload: in current version, to reduce the indexing overhead of SGD, we load all the data into memeory at initialization.
    methods:
        nextinstance():  returns a single instance and its label from the training set, used for SGD
        nextbatch(): returns a batch of instances and their labels from the training set, used for MGD
            args: 
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label
        sgd_batch(): returns a batch of instances and their labels from the trainin set randomly, number of hs and nhs are equal.
            args:
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label

'''
class data:
    def __init__(self, fea, lab, preload=False, ratio=None):
        self.ptr_n=0
        self.ptr_h=0
        self.ptr=0
        self.dat=fea
        self.label=lab
        with open(lab) as f:
            self.maxlen=sum(1 for _ in f)
            
            if ratio:
                content = "adjust data from %d " % (self.maxlen)
                self.maxlen = int(self.maxlen*min(ratio, 1.0))
                content += " to %d (%g)" % (self.maxlen, ratio)
                print(content)
        if preload:
            print("loading data into the main memory...")
            
            self.ft_buffer, self.label_buffer, self.ft_buffer_un, self.label_buffer_un=readcsv(self.dat, self.maxlen)

    def nextinstance(self):
        temp_fea=[]
        label=None
        idx=random.randint(0,self.maxlen)
        for dirname, dirnames, filenames in os.walk(self.dat):
            for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))        
        with open(self.label) as l:
            temp_label=np.asarray(list(l)[idx]).astype(int)
            if temp_label==0:
                label=[1,0]
            else:
                label=[0,1]
        return np.rollaxis(np.array(temp_fea),0,3),np.array([label])

    def sgd(self, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
          
            labelist=np.asarray(list(l)[:self.maxlen]).astype(int)
        length=labelist.size
        idx=random.randint(0, length-1)
        temp_label=labelist[idx]
        if temp_label==0:
            label=[1,0]
        else:
            label=[0,1]
        ft= self.ft_buffer[idx]

        return ft, np.array(label)
    def sgd_batch_2(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
          
            labelist=np.asarray(list(l)[:self.maxlen]).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch / 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        label = processlabel(label,2, 0,0 )
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label


    def sgd_batch(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
          
            labelist=np.asarray(list(l)[:self.maxlen]).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch / 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
       
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_beta: returns the balalced batch, used for training only
    '''
    def nextbatch_beta(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        with open(self.label) as l:
            labelist=np.asarray(list(l)[:self.maxlen]).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size

        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch/2
            # handle small data size 
            # change num for non-hotspot to batch-num 
            if num >= h_length:
                num = h_length-1
            if num>=n_length or num>=h_length:
                print('ERROR:Batch size exceeds data size')
                print('Abort.')
                quit()
            else:
                if self.ptr_n+(batch-num) <n_length:
                    idxn = labexn[self.ptr_n:self.ptr_n+(batch-num)]
                elif self.ptr_n+(batch-num) >=n_length:
                    idxn = np.concatenate((labexn[self.ptr_n:n_length], labexn[0:self.ptr_n+(batch-num)-n_length]))
                self.ptr_n = update_ptr(self.ptr_n, (batch-num), n_length)
                if self.ptr_h+num <h_length:
                    idxh = labexh[self.ptr_h:self.ptr_h+num]
                elif self.ptr_h+num >=h_length:
                    idxh = np.concatenate((labexh[self.ptr_h:h_length], labexh[0:self.ptr_h+num-h_length]))
                self.ptr_h = update_ptr(self.ptr_h, num, h_length)
                #print self.ptr_n, self.ptr_h
                label = np.concatenate((np.zeros(batch-num), np.ones(num)))
                #label = processlabel(label,2, delta1, delta2)
                ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
                ft_batch_nhs = self.ft_buffer[idxn]
                label_nhs = np.zeros(batch-num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_without_balance: returns the normal batch. Suggest to use for training and validation
    '''
    def nextbatch_without_balance_alpha(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        if self.ptr + batch < self.maxlen:
            label = self.label_buffer[self.ptr:self.ptr+batch]
            ft_batch = self.ft_buffer[self.ptr:self.ptr+batch]
        else:
            label = np.concatenate((self.label_buffer[self.ptr:self.maxlen], self.label_buffer[0:self.ptr+batch-self.maxlen]))
            ft_batch = np.concatenate((self.ft_buffer[self.ptr:self.maxlen], self.ft_buffer[0:self.ptr+batch-self.maxlen]))
        self.ptr = update_ptr(self.ptr, batch, self.maxlen)
        return ft_batch, label
    def nextbatch(self, batch, channel=None, delta1=0, delta2=0):
        #print('recommed to use nextbatch_beta() instead')
        databat=None
        temp_fea=[]
        label=None
        if batch>self.maxlen:
            print('ERROR:Batch size exceeds data size')
            print('Abort.')
            quit()
        if self.ptr+batch < self.maxlen:
            #processing labels
            with open(self.label) as l:
                temp_label=np.asarray(list(l)[self.ptr:self.ptr+batch])
                label=processlabel(temp_label, 2, delta1, delta2)
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
            self.ptr=self.ptr+batch
        elif (self.ptr+batch) >= self.maxlen:
            
            #processing labels
            with open(self.label) as l:
                a=np.genfromtxt(islice(l, self.ptr, self.maxlen),delimiter=',')
            with open(self.label) as l:
                b=np.genfromtxt(islice(l, 0, self.ptr+batch-self.maxlen),delimiter=',')
            #processing data
            if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                temp_label=b
            elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                temp_label=a
            else:
                temp_label=np.concatenate((a,b))
            label=processlabel(temp_label,2, delta1, delta2)
            #print label.shape
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, None, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print a.shape, b.shape, self.ptr
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, 0, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print a.shape, b.shape, self.ptr
            self.ptr=self.ptr+batch-self.maxlen
        #print np.asarray(temp_fea).shape
        return np.rollaxis(np.asarray(temp_fea), 0, 3)[:,:,0:channel], label

'''
    get_batch_withweight_bias: returns a batch of instances, their labels and weight from the training set randomly
    args:
        trainX: feature tensor of training data set
        trainY: labels of training data set
        weight: weights of training data set
        batchsize: batchsize
    returns: the batch for training and the batch for adjusting groud truth of bias learning
'''        
def get_batch_withweight_bias(trainX,trainY,weight,batchsize=32):
    train_nh_X, train_h_X, weight_nh,weight_h = get_batch_nh(trainX,trainY,weight)
    n_length = len(train_nh_X)
    h_length = len(train_h_X)
    if h_length < 16:
        if not h_length ==0:
            idxn = (np.random.rand(batchsize-h_length)*n_length).astype(int)
            ft_batch = np.concatenate((train_nh_X[idxn],train_h_X))
            label = np.concatenate((np.zeros(batchsize-h_length), np.ones(h_length)))
            weight_whole = np.concatenate((weight_nh[idxn],weight_h))
            ft_batch_nhs = train_nh_X[idxn]
            label_nhs = np.zeros(batchsize-h_length)
            weight_nhs = weight_nh[idxn]
        else:
            idxn = (np.random.rand(batchsize)*n_length).astype(int)
            ft_batch = train_nh_X[idxn]
            label = np.zeros(batchsize)
            weight_whole = weight_nh[idxn]
            ft_batch_nhs = ft_batch
            label_nhs = label
            weight_nhs = weight_whole

    else:
        num = batchsize/2
        idxn =(np.random.rand(num)*n_length).astype(int)
        idxh = (np.random.rand(num)*h_length).astype(int)
        ft_batch = np.concatenate((train_nh_X[idxn], train_h_X[idxh]))
        label = np.concatenate((np.zeros(num), np.ones(num)))
        weight_whole = np.concatenate((weight_nh[idxn],weight_h[idxh]))
        ft_batch_nhs = train_nh_X[idxn]
        label_nhs = np.zeros(num)
        weight_nhs = weight_nh[idxn]
    return ft_batch, label, weight_whole, ft_batch_nhs, label_nhs, weight_nhs

'''
    get_batch_withweight: returns a batch of instances, their labels and weight from the training set randomly
    args:
        trainX, trainY, weight, batchsize: see get_batch_withweight_bias
    returns: the batch for training
''' 
def get_batch_withweight(trainX,trainY,weight,batchsize=32):
    train_nh_X, train_h_X, weight_nh,weight_h = get_batch_nh(trainX,trainY,weight)
    n_length = len(train_nh_X)
    h_length = len(train_h_X)
    if h_length < 16:
        if not h_length ==0:
            idxn = (np.random.rand(batchsize-h_length)*n_length).astype(int)
            ft_batch = np.concatenate((train_nh_X[idxn],train_h_X))
            label = np.concatenate((np.zeros(batchsize-h_length), np.ones(h_length)))
            weight_whole = np.concatenate((weight_nh[idxn],weight_h))
        else:
            idxn = (np.random.rand(batchsize)*n_length).astype(int)
            ft_batch = train_nh_X[idxn]
            label = np.zeros(batchsize)
            weight_whole = weight_nh[idxn]
    elif n_length <16:
        if not n_length ==0:
            idxh = (np.random.rand(batchsize-n_length)*h_length).astype(int)
            ft_batch = np.concatenate((train_h_X[idxh],train_nh_X))
            label = np.concatenate((np.ones(batchsize-n_length), np.zeros(n_length)))
            weight_whole = np.concatenate((weight_h[idxh],weight_nh))
        else:
            idxh = (np.random.rand(batchsize)*h_length).astype(int)
            ft_batch = train_h_X[idxh]
            label = np.ones(batchsize)
            weight_whole = weight_h[idxh]
    else:
        num = batchsize/2
        idxn =(np.random.rand(num)*n_length).astype(int)
        idxh = (np.random.rand(num)*h_length).astype(int)
        ft_batch = np.concatenate((train_nh_X[idxn], train_h_X[idxh]))
        label = np.concatenate((np.zeros(num), np.ones(num)))
        weight_whole = np.concatenate((weight_nh[idxn],weight_h[idxh]))
    return ft_batch, label, weight_whole
    
'''
    get_batch_nh: separate instances with different labels
    args:
        trainX, trainY, weight: see get_batch_withweight_bias
    returns: separated instances and their weights
''' 

def get_batch_nh(trainX,trainY,weight):
    trainwhole = list(zip(trainX,trainY,weight))
    train_nh_X = []
    train_h_X = []
    weight_nh = []
    weight_h = []
    for i in trainwhole:
        x,y,z = i
        if y == 0:
            train_nh_X.append(x)
            weight_nh.append(z)
        if y == 1:
            train_h_X.append(x)
            weight_h.append(z)
    train_nh_X = np.array(train_nh_X)
    train_h_X = np.array(train_h_X)
    weight_nh = np.array(weight_nh)
    weight_h = np.array(weight_h)
    return(train_nh_X, train_h_X,weight_nh,weight_h)

    
'''
    get_Dkl: get Kullback-Leibler (KL) divergence of pair instances and the result of the clustering stream's loss function
    args:
        x: prediction socre(s) of input batch (clustering stream), getting from forward_crosstask
        P: matrix with pairwise constraint information      
    returns: 
        D_kl: Kullback-Leibler (KL) divergence of pair instances
        C_loss: the result of the clustering stream's loss function
''' 

def get_Dkl(x,P):
    N = tf.shape(x)[0]
    x_N = tf.reshape(tf.tile(x, [N,1]), [N,N,2])
    x_NT = tf.transpose(x_N, perm=[1,0,2])
    D1 = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(x_NT),logits=x_N)
    D = tf.negative(D1)
    P_D = tf.ones([1,N])[0]
    P_D_diag = tf.diag(P_D)
    Dii_diag = D*P_D_diag
    Dii_1 = tf.reduce_sum(Dii_diag,1)
    P_D_1 = tf.ones_like(D)
    Dii = Dii_1 * P_D_1
    D_kl = Dii - D
    D_kl1 = tf.multiply(D_kl,P)
    P1 = tf.cast(tf.ones_like(P), tf.float32)
    P2 = 2*P1
    P0 = tf.cast((P1 - P),tf.float32)
    D_kl0 = tf.multiply(D_kl,P0)
    D_kl_C = P2 - D_kl0
    P_sign = tf.sign(D_kl_C)
    P_sign_2 = P_sign + P1
    P_sign_1 = tf.div(P_sign_2, 2)
    C0 = tf.multiply(D_kl_C, P_sign_1)
    C0_t = tf.multiply(C0,P0)
    C_whole = D_kl1 + C0_t
    C_loss1 = tf.reduce_mean(C_whole)
    C_loss = (C_loss1)/2
    return D_kl,C_loss

'''
    get_Dkl_C: same as get_DKl, numpy version
''' 

def get_Dkl_C(x,P):
    #pdb.set_trace()
    N = len(x)
    x = softmax_np(x)
    x_N = (np.tile(x, [N,1])).reshape(N,N,2)
    x_NT = np.transpose(x_N,[1,0,2])
    D = softmax_cross_C(x_NT, x_N)
    P_D = np.ones(N)
    P_D_diag = np.diag(P_D)
    Dii_diag = D*P_D_diag
    Dii_1 = np.sum(Dii_diag,1)
    P_D_1 = np.ones_like(D)
    Dii = Dii_1 * P_D_1
    D_kl = Dii - D
    D_kl1 = np.multiply(D_kl,P)
    P1 = np.ones_like(P)
    P1 = P1.astype(np.float32)
    P2 = 2*P1
    P0 = (P1 - P).astype(np.float32)
    D_kl0 = np.multiply(D_kl,P0)
    D_kl_C = P2 - D_kl0
    
    P_sign = np.sign(D_kl_C)
    P_sign_2 = P_sign + P1
    P_sign_1 = P_sign_2/2
    C0 = np.multiply(D_kl_C, P_sign_1)
    C0_t = np.multiply(C0,P0)
    C_whole = D_kl1 + C0_t
    C_loss1 = np.mean(C_whole)
    C_loss = (C_loss1)/2
    return D_kl,C_loss

'''
    get_wi: get weights of training instances
    args:
        D_kl, P: see get_Dkl   
    returns: weights of training instances
''' 
def get_wi(D_kl,P):
    P_whole = (np.sum(P,1)).astype(np.float32)
    D_kli1 = np.multiply(D_kl,P)
    D_kli = np.sum(D_kli1, 1)
    di = D_kli/P_whole
    di_exp = np.exp(np.negative(di))
    di_exp_whole = np.sum(di_exp)
    len_en = len(di)
    wi = di_exp/di_exp_whole*len_en#weight   
    wi = wi/max(wi)
    return wi
    
'''
    softmax_cross_C: calculates cross-entropy of labels and logits
'''    
def softmax_cross_C(labels, logits):
    logits_s  = np.log(logits)
    loss = labels * logits_s
    loss_whole = loss.sum(2)
    return(loss_whole)
 
'''
    softmax_cross_loss: calculates cross-entropy of labels and logits (no sum)
'''    
def softmax_cross_loss(labels, logits):
    logits_softmax = softmax_np(logits)
    logits_s  = np.log(np.ones_like(logits)/logits_softmax)
    loss = labels * logits_s
    loss_un = loss.sum(1)
    return(loss_un)
    
'''
    softmax_np: calculates softmax of x
'''      
def softmax_np(x):
    x_exp = np.exp(x)
    x_exp_whole = x_exp.sum(1)
    for i in range(len(x)):
        x_exp[i]=x_exp[i]/x_exp_whole[i]
    return x_exp

'''
    pairwise_constraint: get matrix with pairwise constraint information
    args:
        trainX, trainY: see getbatch_withweight_bias
    returns: matrix with pairwise constraint information
''' 
def pairwise_constraint(trainX, trainY):
        plen = len(trainX)
        #Ylen = len(trainY[0])
        pconstrain = np.zeros([plen,plen])
        for i in range(plen):
            for j in range(plen):
                if trainY[i] == trainY[j]:
                    pconstrain[i][j] = 1
        return pconstrain

'''
    data_flatten: concatenate a list and a narry
''' 
def data_flatten(predict_label_whole,predict_label):
    predict1 = predict_label.flatten()
    predict2 = predict1.tolist()
    predict_label_whole = predict_label_whole + predict2
    return predict_label_whole
    
'''
    data_flatten2: data flatten
''' 
def data_flatten2(predict_label_whole):
    predict_label2 = np.array(predict_label_whole)
    predict_label1 = predict_label2.flatten()
    return predict_label1

'''
    get_data_un_r: get unlabeled data set
    args:
        un_for_r: numpy array contains unlabeled data set
    returns: 
        X: feature tensors
        Y: pseduo labels
        W: weights
''' 
def get_data_un_r(un_for_r):
    X=[]
    Y=[]
    W=[]
    for i in un_for_r:
        x,y,z,h,d = i
        X.append(x)
        Y.append(y)
        W.append(z)
    return X,Y,W

'''
    data_split: split data samples and their labels, samples and labels are in pair
''' 
def data_split(newdata):
    dataX = []
    dataY = []
    for i in newdata:
        x,y = i
        dataX.append(x)
        dataY.append(y)
    return dataX,dataY

'''
    data_split_sencond: split data samples and their labels, samples and labels have different indexes
''' 
def data_split_sencond(newdata):
    dataX = []
    dataY = []
    for i in range(len(newdata)):
        if i % 2 ==0:
            dataX.append(newdata[i])
        if i % 2 ==1:
            dataY.append(newdata[i])
    return(dataX,dataY)  
    
'''
    batchfor_test: returns a batch for testing
    args:
        testX: testing samples
        testY: labels of testing samples
        num: parameter controling testing progress
        batch: batch size
    returns: instances batch for testing
''' 

def batchfor_test(testX,testY,num,batch=1000):
    testlen=len(testX)
    if batch>testlen:
        print('ERROR:Batch size exceeds data size')
        print('Abort.')
    if num + batch < testlen:
        testX_batch = testX[num:num+batch]
        testY_batch = testY[num:num+batch]
        num = num + batch
    elif num + batch >= testlen:
        testX_batch = testX[num:testlen]
        testY_batch = testY[num:testlen]
        num = num +batch-testlen
    test_batch = []
    test_batch.append(testX_batch)
    testY_batch = [vectorized_result(y) for y in testY_batch]
    test_batch.append(testY_batch)
    return test_batch, num

'''
    batchfor_testX: returns a batch for testing
    args:
        testX, num, batch: see batchfor_test
    returns: instances batch for testing
''' 
def batchfor_testX(testX,num,batch=1000):
    testlen=len(testX)
    if batch>testlen:
        print('ERROR:Batch size exceeds data size')
        print('Abort.')
    if num + batch < testlen:
        testX_batch = testX[num:num+batch]
        #testY_batch = testY[num:num+batch]
        num = num + batch
    elif num + batch >= testlen:
        testX_batch = testX[num:testlen]
        #testY_batch = testY[num:testlen]
        num = num +batch-testlen
    #test_batch = []
    #test_batch.append(testX_batch)
    #testY_batch = [vectorized_result(y) for y in testY_batch]
    #test_batch.append(testY_batch)
    return testX_batch, num
'''
    vectorized_result: vectorized labels of samples
    args:
        j: label with shape: 1
    returns: label with shape: 2 x 1
''' 
def vectorized_result(j):
    e = np.zeros(2)
    if j == 1:
        e[1] = 1
    else:
        e[0] = 1
    return e

