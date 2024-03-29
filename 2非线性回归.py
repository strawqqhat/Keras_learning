import keras
import numpy as np
import matplotlib.pyplot as plt
# 按顺序构成的模型
from keras.models import Sequential
# Dense全连接层
from keras.layers import Dense, Activation
from keras.optimizers import SGD


# 使用numpy生成200随机点
x_data = np.linspace(-0.5, 0.5, 200)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

#显示随机点
plt.scatter(x_data, y_data)
plt.show()

# 构建一个顺序模型
model = Sequential()

# 在模型中添加一个全连接层
# 1-10-1
model.add(Dense(units=10, input_dim=1, activation='tanh'))
# model.add(Activation('tanh'))
model.add(Dense(units=1, activation='tanh'))
# model.add(Activation('tanh'))

# 定义优化算法
sgd = SGD(lr=0.3)
# sgd 随机梯度下降法  mse 均方误差
model.compile(optimizer=sgd, loss='mse')


for step in range(3001):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data, y_data)
    if step % 500 == 0:
        print('cost:', cost)

# x_data输入网络中，得到预测值y_pred
y_pred = model.predict(x_data)

# 显示随机点
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()