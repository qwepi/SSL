from model_SSL import *
import ConfigParser as cp
import sys
import time
import os
from progress.bar import Bar

'''
Initialize Path and Global Params
'''
infile = cp.SafeConfigParser()
infile.read(sys.argv[1])

test_path   = infile.get('dir','test_path')


model_path = infile.get('dir','model_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))

'''
Prepare the Input
'''
test_data = data(test_path, test_path+'/label.csv')
x_data = tf.placeholder(tf.float32, shape=[None, blockdim*blockdim, fealen])              #input FT
y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label
x      = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])                             #reshap to NHWC
predict, predict1 = forward_crosstask(x, is_training = False)

y      = tf.cast(tf.argmax(predict, 1), tf.int32)                                         
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                               #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
'''
Start testing
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver    = tf.train.Saver()
    saver.restore(sess, model_path)
    chs = 0   #correctly predicted hs
    cnhs= 0   #correctly predicted nhs
    ahs = 0   #actual hs
    anhs= 0   #actual hs
    start   = time.time()
    bar = Bar('Detecting', max=test_data.maxlen/1000+1)
    for titr in xrange(0, test_data.maxlen/1000+1):
        if not titr == test_data.maxlen/1000:
            tbatch = test_data.nextbatch(1000, fealen)
        else:
            tbatch = test_data.nextbatch(test_data.maxlen-titr*1000, fealen)
        tdata = tbatch[0]
        tlabel= tbatch[1]
        tmp_y    = y.eval(feed_dict={x_data: tdata, y_gt: tlabel})
        tmp_label= np.argmax(tlabel, axis=1)
        tmp      = tmp_label+tmp_y
        chs += sum(tmp==2)
        cnhs+= sum(tmp==0)
        ahs += sum(tmp_label)
        anhs+= sum(tmp_label==0)
        bar.next()
    bar.finish()
    print chs, ahs, cnhs, anhs
    if not ahs ==0:
        hs_accu = 1.0*chs/ahs
    else:
        hs_accu = 0
    fs      = anhs - cnhs
    end       = time.time()
print ahs, anhs
print('Hotspot Detection Accuracy is %f'%hs_accu)
print('False Alarm is %f'%fs)
print('Test Runtime is %f seconds'%(end-start))
    



