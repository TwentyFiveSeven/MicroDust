import pandas as pd
import numpy as np
import datetime
import tensorflow as tf

#Read excel data to pandas.DataFrame
excel_data = pd.read_excel('b.xlsx',sheet_name='Sheet1')

#df =pd.DataFrame(columns=['MSRDT_DE', 'MSRSTE_NM', 'NO2', 'O3', 'CO','SO2','PM10','PM25'])
#df = pd.DataFrame()

#print(excel_data['MSRDT_DE'])

#d = excel_data[(excel_data['MSRDT_DE'] == 20110101)]

#print(type(d.iloc[0]['MSRDT_DE'])) #numpy.int64
#print(type(d.iloc[0]['MSRSTE_NM'])) #str
#print(type(d.iloc[0]['NO2'])) #numpy.float64
#print(type(d.iloc[0]['O3'])) #numpy.float64
#print(type(d.iloc[0]['CO'])) #numpy.float64
#print(type(d.iloc[0]['SO2'])) #numpy.float64
#print(type(d.iloc[0]['PM10'])) #numpy.int64
#print(type(d.iloc[0]['PM25'])) #numpy.int64

#Convert '0' to 'NaN'
columns=['MSRDT_DE', 'MSRSTE_NM', 'NO2', 'O3', 'CO','SO2','PM10','PM25']
excel_data[columns] = excel_data[columns].replace({'0':np.nan, 0:np.nan})


#Fill 'NaN' to average of data
excel_data['NO2'] = excel_data['NO2'].fillna(excel_data['NO2'].mean())
excel_data['O3'] = excel_data['O3'].fillna(excel_data['O3'].mean())
excel_data['CO'] = excel_data['CO'].fillna(excel_data['CO'].mean())
excel_data['SO2'] = excel_data['SO2'].fillna(excel_data['SO2'].mean())
excel_data['PM10'] = excel_data['PM10'].fillna(excel_data['PM10'].mean())
excel_data['PM25'] = excel_data['PM25'].fillna(excel_data['PM25'].mean())


#Have to normalize
#...
#...
#...

#print(np.shape(excel_data))

#print(np.shape(excel_data.iloc[:,2:]))


#Reduce Columns and Increase the number of data
df = excel_data.values

df = np.reshape(df[:,2:],[-1,25,6,1])

#print(np.shape(df))

#Convert pandas.DataFrame to Tensor
#t = tf.convert_to_tensor(df,dtype=tf.float32)


#Doing Convolution Neural Network

height = 25
width = 6
n_class = height*width

X = tf.placeholder(np.float32,shape=[None,height,width,1])
conv1 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
conv3 = tf.layers.conv2d(inputs=conv2,filters=128,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
pool_flat = tf.reshape(conv3,[-1,n_class*128])

fc1 = tf.layers.dense(inputs=pool_flat,units=1024,activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1,units=512,activation=tf.nn.relu)
output = tf.layers.dense(inputs=fc2,units=150)

#print(t.get_shape())
#print(conv1.get_shape())
#print(conv2.get_shape())
#print(conv3.get_shape())
#print(pool_flat.get_shape())
#print(fc1.get_shape())
#print(fc2.get_shape())

print(output.get_shape())


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    n = sess.run(output,feed_dict={X:df})

print(len(n))


trainStartDate = datetime.date(2011,1,1)
testStartDate = datetime.date(2018,7,1)

diff = (testStartDate-trainStartDate).days

#print((testStartDate-trainStartDate).days)
#print(type((testStartDate-trainStartDate).days))

trainData = n[:diff]
testData = n[diff:]

window_size = 7

trainX = []
trainY = trainData[window_size:]
for i in range(0,len(trainData)-window_size):
    _x = trainData[i:i+window_size]
    trainX.append(_x)

testX = []
testY = testData[window_size:]
for i in range(0,len(testData)-window_size):
    _x = testData[i:i+window_size]
    testX.append(_x)


print(np.shape(n))

print(np.shape(trainData))
print(np.shape(trainX))
print(np.shape(trainY))

print(np.shape(testData))
print(np.shape(testX))
print(np.shape(testY))


#Doing Recurrent Neural Network


X = tf.placeholder(np.float32,[None,window_size,n_class])
Y = tf.placeholder(np.float32,[None,n_class])

n_hidden = 1024
n_layers = 2

cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True)
multiCell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(n_layers)], state_is_tuple=True)
drop = tf.contrib.rnn.DropoutWrapper(multiCell,output_keep_prob=0.7)
outputs,state = tf.nn.dynamic_rnn(drop,X,dtype=tf.float32)

outputs = tf.transpose(outputs,[1,0,2])
outputs = outputs[-1]

#with tf.Session() as sess:
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    outputs = sess.run(outputs,feed_dict={X:trainX})
#
#print(np.shape(outputs))


W = tf.Variable(tf.random_normal([n_hidden,n_class]))
b = tf.Variable(tf.random_normal([n_class]))

logits = tf.matmul(outputs,W) + b

learning_rate = 0.001

cost = tf.reduce_mean(tf.square(logits-Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        sess.run(optimizer,feed_dict={X:trainX,Y:trainY})
        if (i+1) % 10 == 0:
            print(i+1," : ",sess.run(tf.sqrt(cost),feed_dict={X:trainX,Y:trainY}))
    sess.run(optimizer,feed_dict={X:trainX,Y:trainY})
    W = sess.run(W)
    b = sess.run(b)

df = pd.DataFrame(W)
df.to_excel('q.xlsx', sheet_name='Sheet1')
df = pd.DataFrame(b)
df.to_excel('w.xlsx', sheet_name='Sheet2')

