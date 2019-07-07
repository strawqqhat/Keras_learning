import keras
import numpy as np
import matplotlib.pyplot as plt
# 按顺序构成的模型
from keras.models import Sequential
# Dense全连接层
from keras.layers import Dense

# 使用numpy生成100个随机点
x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data*0.1+0.2+noise

# 构建一个顺序模型
model = Sequential()
# 在模型中添加一个全连接层
model.add(Dense(units=1, input_dim=1))
model.compile(optimizer='sgd', loss='mse')  # sgd 随机梯度下降法  mse 均方误差
for step in range(3001):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data, y_data)
    if step % 500 == 0:
        print('cost', cost)
#  打印权值和偏置值
W, b = model.layers[0].get_weights()
print('W:', W, 'b', b)

# x_data输入网络中，得到预测值y_pred
y_pred = model.predict(x_data)
# 显示随机点
plt.scatter(x_data, y_data)
plt.plot()
plt.show()