import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28 #number of pixels in each line
n_steps = 28 #time steps for the rows of samples
n_hidden_units = 128 # number of neurons in hidden layers
n_classes = 10 #classification numbers

x = tf.placeholder(tf.float32,[None,n_inputs,n_steps])
y = tf.placeholder(tf.float32,[None,n_classes])

Weights = {
    'in' : tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in' : tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out' : tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X, Weights, biases):
    # X (128 batches. 28steps*28inputs)==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in = W*X + b
    X_in = tf.matmul(X, Weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # Use basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    #forget_bias = 1.0 means forgetting nothing in this stage. 'state_is_tuple' is supposed to be True in new Python version
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # initialize all states

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    #dynamic_rnn is recommended
    #time_major means whether the time-related dimension is in the first place
    # (T,0,0) -> time_major = True, (0,T,0) or (0,0,T) ->time_major = False

    #results = tf.matmul(final_state[1], weights['out']) + biases['out']
    #↑ it is a universal solution for result output

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    #transpose the sequence of tensor from [0,1,2] to [1,0,2]
    #which means we transform [batch_size'128,step_size'28,output'10]
    #to [step_size'28,batch_size'128,output'10]
    #use unstack function to unfold the 3D-tensor to a list with step_size*(batch_size,output)
    results = tf.matmul(outputs[-1], Weights['out']) + biases['out']
    #select the last step_size of list, the 28th of list (batch_size,output)
    #the 128 results of the sequence
    return results

pred = RNN(x, Weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run(train_step,feed_dict={
            x:batch_xs,
            y:batch_ys
        })
        if step % 20 ==0:
            print(sess.run(accuracy,feed_dict={
                x:batch_xs,
                y:batch_ys
        }))
        step = step + 1



















