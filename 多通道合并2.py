import os
import numpy as np
from osgeo import gdal
from sklearn.preprocessing import MinMaxScaler

# 输入文件夹路径
input_folder = r"E:\\course\\yingyuluxjing\\factor125_1"

# 输出文件夹路径
output_folder = r"E:\\AI\\lunwentu\\network"

# 获取文件夹中的所有tif文件
input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".tif")]
input_files = sorted(input_files)  # 按照文件名排序

# 读取第一个tif文件的元数据，用于创建输出文件
ds = gdal.Open(input_files[0])
cols = ds.RasterXSize
rows = ds.RasterYSize
bands = len(input_files)

# 获取原图层的坐标系
proj = ds.GetProjection()
geotrans = ds.GetGeoTransform()

# 创建输出文件
driver = gdal.GetDriverByName("GTiff")
output_file = os.path.join(output_folder, "raster125.tif")
output_ds = driver.Create(output_file, cols, rows, bands, gdal.GDT_Float32)
output_ds.SetProjection(proj)
output_ds.SetGeoTransform(geotrans)

# 将所有tif文件的数据读入数组中
for i, input_file in enumerate(input_files):

    ds = gdal.Open(input_file)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    # 将数据类型设置为浮点数
    data = data.astype(np.float32)
    # 归一化数据
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    # 打印数据范围和类型，以便查看是否有缺失值
    print("Data Range:", np.min(data), np.max(data))
    print("Data Type:", data.dtype)
    # 将数据类型设置为浮点数
    data = data.astype(np.float32)
    # 使用中值填充缺失值
    data[np.isnan(data)] = np.nanmedian(data)
    output_ds.GetRasterBand(i+1).WriteArray(data)
    ds = None

output_ds.FlushCache()
