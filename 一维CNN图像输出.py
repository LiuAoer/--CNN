import numpy as np
from osgeo import gdal
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 栅格图像文件的路径
raster_file = r"E:\\AI\\lunwentu\\network\\raster_fenji.tif"

# 打开栅格文件
dataset = gdal.Open(raster_file)

# 获取栅格图像的通道数
num_channels = dataset.RasterCount

# 读取所有通道的数据并将其存储在一个3D NumPy数组中
raster_data = np.zeros((dataset.RasterYSize, dataset.RasterXSize, num_channels))
for i in range(num_channels):
    raster_data[:, :, i] = dataset.GetRasterBand(i + 1).ReadAsArray()

# 将栅格数据的形状更改为 (num_samples, num_channels, 1, 1)，以便输入到 CNN 中
num_samples = raster_data.shape[0] * raster_data.shape[1]
X_train = raster_data.reshape(num_samples, num_channels, 1, 1)

# 定义 CNNModel 类
class CNNModel(Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 1), activation='relu', input_shape=(num_channels, 1, 1))
        self.pool1 = MaxPooling2D((2, 1))
        self.flatten = Flatten()
        self.fc1 = Dense(64, activation='relu')
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

# 创建模型实例
loaded_model = CNNModel()

# 加载保存的权重
loaded_model.load_weights('./modelcnn_checkpoint/best_weights')

# 使用加载的模型对测试数据进行预测
predictions = loaded_model.predict(X_train)
# 获取每个块的预测概率
predicted_probabilities = predictions[:, 0]

# 重塑结果数组以匹配原始栅格数据的形状
predictions_reshaped = predicted_probabilities.reshape(dataset.RasterYSize, dataset.RasterXSize)

# 设置输出栅格文件的路径
output_raster_file = r"E:\\AI\\lunwentu\\network\\raster2.tif"

# 获取原始栅格数据的地理变换和投影信息
geotransform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

# 创建输出栅格文件
driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(output_raster_file, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)

# 将地理变换和投影信息设置为与输入栅格文件相同
output_dataset.SetGeoTransform(geotransform)
output_dataset.SetProjection(projection)

# 将预测结果写入输出栅格文件
output_band = output_dataset.GetRasterBand(1)
output_band.WriteArray(predictions_reshaped)

#设置 NoData 值（如果需要）
output_band.SetNoDataValue(-9999)
#清除缓存，确保数据被写入文件
output_band.FlushCache()

#关闭输出数据集
output_dataset = None

#关闭输入数据集
dataset = None

print("栅格预测完成，结果已保存到文件：", output_raster_file)
