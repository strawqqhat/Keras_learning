import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1))/255.0
x_test = x_test.reshape((-1, 28, 28, 1))/255.0
# 换one hot形式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 定义顺序模型
model = Sequential()

# 第一个卷积层
# input_shape 输入平面
# filters 卷积核/滤波器个数
# kenel_size 卷积窗口大小
# strides 步长
# padding 方式same/valid
# activation 激活函数
model.add(Convolution2D(
    input_shape=(28,28,1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu',
    name='conv1'
))

# 第一个池化层
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',

))

# 第二个卷积层
model.add(Convolution2D(64,5,strides=1,padding='same',activation='relu',name='conv2'))
model.add(MaxPooling2D(2,2,'same'))

model.add(Dense(1024,activation='relu'))

# Dropout
# model.add(Dropout(1-0.5))
# 第二个全连接层
model.add(Dense(10,activation='softmax'))


plot_model(model,to_file='model.png',show_shapes=True,show_layer_names='False',rankdir='TB')
plt.figure(figsize=(10,10))
img=plt.imread('model.png')
plt.imshow(img)
plt.axis('off')
plt.show()