import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

lr = 0.001
training_iters = 1000000
batch_size = 128

n_inputs = 28 #number of pixels in each line
n_steps = 28 #time steps for the rows of samples
n_hidden_units = 128 # number of neurons in hidden layers
n_classes = 10 #classification numbers

Weights = {
    'in' : tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in' : tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out' : tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X, Weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, Weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], Weights['out']) + biases['out']
    return results



def compute_accuracy(v_xs,v_ys):
    global prediction
    class_count={}
    class_correct={}
    result_precentage={}
    
    y_pre = sess.run(prediction,feed_dict = {xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuray = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuray,feed_dict = {xs:v_xs,ys:v_ys})
    
    print(v_ys.shape)
    for i in range(0, v_ys.shape[0]):
        correct_class=np.argmax(v_ys[i])
        pre_class=np.argmax(y_pre[i])
        if correct_class not in class_count:
            class_count[correct_class]=0
            class_correct[correct_class]=0
        class_count[correct_class]+=1

        if correct_class == pre_class:
            class_correct[correct_class]+=1
    for i in class_correct:
        result_precentage[i]=class_correct[i]/class_count[i]
    return result,result_precentage

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return  tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

xs = tf.placeholder(tf.float32,[None,784]) #第一个位置是none，意思是不限制输入的个数，第二个是输入的维度28*28
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.Variable(tf.constant(0.9))
x_image = tf.reshape(xs,[-1,28,28,1])
#在其中-1是对应的个数，先可以不用管，28*28=784,1代表的是图像深度，因为是黑白，所以为1

#h_conv1
W_conv1 = weight_variable([5,5,1,32])
#shape=5*5,1是输入图片的厚度，32是输出卷积结果的厚度
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # outputsize = 28*28*32(from 28*28*1)
h_pool1 = max_pool_2x2(h_conv1)  #outputsize = 14*14*32(2步长，因此只剩下一半）

#h_conv2
W_conv2 = weight_variable([5,5,32,64])
#shape=5*5,1是输入图片的厚度，32是输出卷积结果的厚度
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) # outputsize = 14*14*64(from 14*14*32)
h_pool2 = max_pool_2x2(h_conv2)  #outputsize = 7*7*64(2步长，因此只剩下一半）

#fc1_layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#fc2_layer
W_fc2 = weight_variable([1024,784])
b_fc2 = bias_variable([784])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

prediction = RNN(h_fc2_drop, Weights, biases)
pred = tf.nn.softmax(prediction)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=ys))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps*n_inputs])
        sess.run([train_step,pred],feed_dict={
            xs:batch_xs,
            ys:batch_ys
        })
        if step % 20 ==0:
            test_xs,test_ys = mnist.test.next_batch(batch_size)
            print(compute_accuracy(test_xs,test_ys))
        step = step + 1
