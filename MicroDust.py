import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
from normalization import normalization

#Read excel data to pandas.DataFrame
excel_data = pd.read_excel('AirQuality.xlsx',sheet_name='AirQuality')


#Convert data from '0' to 'NaN'
columns=['MSRDT_DE', 'MSRSTE_NM', 'NO2', 'O3', 'CO','SO2','PM10','PM25']
excel_data[columns] = excel_data[columns].replace({'0':np.nan, 0:np.nan})


#Fill 'NaN' to average of data
excel_data['NO2'] = excel_data['NO2'].fillna(excel_data['NO2'].mean())
excel_data['O3'] = excel_data['O3'].fillna(excel_data['O3'].mean())
excel_data['CO'] = excel_data['CO'].fillna(excel_data['CO'].mean())
excel_data['SO2'] = excel_data['SO2'].fillna(excel_data['SO2'].mean())
excel_data['PM10'] = excel_data['PM10'].fillna(excel_data['PM10'].mean())
excel_data['PM25'] = excel_data['PM25'].fillna(excel_data['PM25'].mean())


#Reduce Columns and Increase the number of data
height = 25
width = 6
n_class = height*width

df = excel_data.values
df = df[:,2:]
x = np.reshape(df,[-1,n_class])


#Normalize data to min-max
df = normalization(df)


#Ready to CNN
#Change Shape to NHWC
df = np.reshape(df,[-1,height,width,1])


#Convolution Neural Network(CNN)

X = tf.placeholder(np.float32,shape=[None,height,width,1])

conv1 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
conv3 = tf.layers.conv2d(inputs=conv2,filters=128,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)

pool_flat = tf.reshape(conv3,[-1,n_class*128])

fc1 = tf.layers.dense(inputs=pool_flat,units=1024,activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1,units=512,activation=tf.nn.relu)
output = tf.layers.dense(inputs=fc2,units=150)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    n = sess.run(output,feed_dict={X:df})


#Divide trainData and testData from CNN output
window_size = 7

trainStartDate = datetime.date(2013,8,1)
testStartDate = datetime.date(2018,7,1)

diff = (testStartDate-trainStartDate).days

trainData = n[:diff]
testData = n[diff-window_size:]

trainX = []
trainY = x[window_size:diff]

for i in range(0,len(trainData)-window_size):
    _x = trainData[i:i+window_size]
    trainX.append(_x)


testX = []
testY = x[diff:]

for i in range(0,len(testData)-window_size):
    _x = testData[i:i+window_size]
    testX.append(_x)


#Recurrent Neural Network(RNN)
n_hidden = 1024
n_layers = 2

X = tf.placeholder(np.float32,[None,window_size,n_class])
Y = tf.placeholder(np.float32,[None,n_class])

cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True)
multiCell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(n_layers)], state_is_tuple=True)
drop = tf.contrib.rnn.DropoutWrapper(multiCell,output_keep_prob=0.7)
outputs,state = tf.nn.dynamic_rnn(drop,X,dtype=tf.float32)

outputs = tf.transpose(outputs,[1,0,2])
outputs = outputs[-1]

W = tf.Variable(tf.random_normal([n_hidden,n_class]))
b = tf.Variable(tf.random_normal([n_class]))

logits = tf.matmul(outputs,W) + b


#Training Data
learning_rate = 0.001
iterations = 2500

cost = tf.reduce_mean(tf.square(logits-Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(iterations):
        sess.run(optimizer,feed_dict={X:trainX,Y:trainY})
        if (i+1) % 100 == 0:
            print(i+1," : ",sess.run(tf.sqrt(cost),feed_dict={X:trainX,Y:trainY}))

    predY = sess.run(logits,feed_dict={X:testX})

predY = np.reshape(predY,[-1,width])
testY = np.reshape(testY,[-1,width])

df = pd.DataFrame(predY)
df.to_excel('PredictData.xlsx', sheet_name='PredictData')

rmse = {}
rmse['microdust'] = np.sqrt(np.mean((predY[:,4]-testY[:,4])**2))
rmse['hypermicrodust'] = np.sqrt(np.mean((predY[:,5]-testY[:,5])**2))

print(rmse)
