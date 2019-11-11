#Classification
#[0,0,0,0,0,0,0,0,0,0] = 0
#[1,0,0,0,0,0,0,0,0,0] = 1
#[0,1,0,0,0,0,0,0,0,0] = 2
#......
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#number 1~10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#layer
def add_layer(inputs,in_size,out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def computer_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:batch_xs,ys:batch_ys})
    return result

#placeholder
xs = tf.placeholder(tf.float32,[None,784])#28 x28
ys = tf.placeholder(tf.float32,[None,10])#1~10

#output
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)#softmax for classification

#loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)#extract the database  #save the training time
    sess.run(train_step, feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print(computer_accuracy(mnist.test.images,mnist.test.labels))
