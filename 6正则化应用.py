import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
# 负责正则化
from keras.regularizers import l2


#载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# (60000,28,28)
print('x_shape:', x_train.shape)
# (60000)
print('y_shape:', y_train.shape)
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0

# 换one hot格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型：输入784个神经元，输出10个神经元
model = Sequential([
    Dense(units=200, input_dim=784, bias_initializer='one', activation='tanh', kernel_regularizer=l2(0.0003)),
    Dense(units=100, bias_initializer='one', activation='tanh',kernel_regularizer=l2(0.0003)),
    Dense(units=10, bias_initializer='one', activation='softmax',kernel_regularizer=l2(0.0003))
])

# 定义优化器 学习率设置为0.2
sgd = SGD(lr=0.2)

# 定义优化器,loss function,训练过程中计算准确率
# 交叉熵 损失函数
model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=10)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print('test loss', loss)
print('accuracy', accuracy)

loss, accuracy = model.evaluate(x_train, y_train)
print('train loss', loss)
print('accuracy', accuracy)
