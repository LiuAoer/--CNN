import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def open_excel(filename):
    readbook = pd.read_excel(f'{filename}.xlsx', engine='openpyxl')
    nplist = readbook.T.to_numpy()
    data = nplist[0:-1].T

    data = np.float64(data)
    target = nplist[-1]


    return data, target

x_data, y_data = open_excel('qinshigou')
x_data = x_data.reshape(-1, 11, 1, 1)

class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 1), activation='relu', input_shape=(11, 1, 1))
        self.pool1 = MaxPooling2D((2, 1))
        self.flatten = Flatten()
        self.fc1 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.dropout = Dropout(0.5)
        self.fc2 = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        y = self.fc2(x)
        return y

np.random.seed(116)
tf.random.set_seed(116)

n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=116)

roc_auc_train_list = []
roc_auc_val_list = []

checkpoint_init_dir = 'modelcnn_init_checkpoint'
checkpoint_filepath = 'modelcnn_checkpoint'

# 在这里，将路径添加到文件名
checkpoint_init_file = os.path.join(checkpoint_init_dir, 'init_weights')
checkpoint_file = os.path.join(checkpoint_filepath, 'best_weights')

if not os.path.exists(checkpoint_init_dir):
    os.makedirs(checkpoint_init_dir)

if not os.path.exists(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)

model_init = IrisModel()
model_init.save_weights(checkpoint_init_file) # 在这里使用新的文件名

for fold, (train_index, val_index) in enumerate(kfold.split(x_data, y_data)):
    print(f"正在进行第 {fold + 1} 折交叉验证...")
    x_train, x_val = x_data[train_index], x_data[val_index]
    y_train, y_val = y_data[train_index], y_data[val_index]

    model = IrisModel()

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_file,  # 在这里使用新的文件名
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    try:
        model.load_weights(checkpoint_file)  # 在这里使用新的文件名
        print('从以下位置加载模型权重：', checkpoint_file)
    except Exception as e:
        print('没有找到之前的模型权重，或加载失败。从初始化权重检查点文件加载权重。')
        model.load_weights(checkpoint_init_file)  # 在这里使用新的文件名

    history = model.fit(x_train, y_train, batch_size=32, epochs=500,
                        validation_data=(x_val, y_val), validation_freq=1,
                        callbacks=[checkpoint_callback])

    model.summary()
    y_train_pred_probs = model.predict(x_train)
    y_val_pred_probs = model.predict(x_val)

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_probs)
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_train_list.append(roc_auc_train)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_probs)
    roc_auc_val = auc(fpr_val, tpr_val)
    roc_auc_val_list.append(roc_auc_val)

    # 计算平均AUC
    avg_roc_auc_train = np.mean(roc_auc_train_list)
    avg_roc_auc_val = np.mean(roc_auc_val_list)

    # 只在最后一次迭代绘制ROC曲线
    if fold == n_splits - 1:
        plt.plot(fpr_train, tpr_train, 'b', label='训练集平均 ROC 曲线 (面积 = %0.2f)' % avg_roc_auc_train)
        plt.plot(fpr_val, tpr_val, 'r', label='验证集平均 ROC 曲线 (面积 = %0.2f)' % avg_roc_auc_val)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('Iris 数据集的接收者操作特征 (ROC) 曲线 - 10折交叉验证')
        plt.legend(loc="lower right")
        plt.show()
