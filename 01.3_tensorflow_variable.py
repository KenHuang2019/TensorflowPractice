#定義變量
import tensorflow as tf

state = tf.Variable(0,name='counter')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state , one)
update = tf.assign(state, new_value)#將上述state更新成加法結果

init = tf.global_variables_initializer()#設定初始化 #######若定義變量必須做這步

with tf.Session() as sess:#打開此功能
    sess.run(init)#執行初始化
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
