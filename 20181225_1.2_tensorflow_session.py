#對話控制
#
import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2) #matrix multiply(m1,m2)    =   np.dot(m1,m2)

"""
#method 1
sess = tf.Session()#object 切記 大寫"S"ess
result1 = sess.run(product)#執行上述結構
print(result1)
sess.close()
"""

#method 2
with tf.Session() as sess:#會自動關上 和for循環類似   #object 切記 大寫"S"ess
    result2 = sess.run(product)#執行上述結構
    print(result2)
