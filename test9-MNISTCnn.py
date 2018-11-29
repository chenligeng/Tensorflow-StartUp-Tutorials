import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict = {xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuray = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuray,feed_dict = {xs:v_xs,ys:v_ys})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return  tf.Variable(initial)

def conv2d(x,W):
    # strides[1,x_movement,y_movement,1] 意为步长
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    # padding 有same和valid两种，same会经过卷积之后大小不变（新的部分由空白填充）
    # ，valid则是经过卷积之后会变小

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
# [n_samples,7,7,64] ->[n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#fc2_layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(5000):
    batch_xs,batch_ys = mnist.train.next_batch(100) #每次训练只用批处理100个数据
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))