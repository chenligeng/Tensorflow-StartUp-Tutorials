import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict = {xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuray = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuray,feed_dict = {xs:v_xs,ys:v_ys})
    return result

xs = tf.placeholder(tf.float32,[None,784]) #第一个位置是none，意思是不限制输入的个数，第二个是输入的维度28*28
ys = tf.placeholder(tf.float32,[None,10])

l1 = add_layer(xs,784,10,activation_function=tf.nn.tanh)
#隐层使用relu6或者tanh都没问题，但是relu的话可能会产生梯度消失的问题，至少result不work
prediction = add_layer(l1,10,10,activation_function=tf.nn.softmax)
#rediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100) #每次训练只用批处理100个数据
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

