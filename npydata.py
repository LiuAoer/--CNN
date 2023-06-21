import numpy as np
from osgeo import gdal
import os
def npgget():
    # 定义文件夹和txt文件路径
    folder_path = 'text3'
    txt_file = 'txt/新建文本文档.txt'

    # 获取文件夹中的所有tif文件
    input_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".tif")]

    # 读取txt文件中的文件名列表和对应标签
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    file_list = []
    label_list = []
    for line in lines:
        file_name, label = line.strip().split(' ')
        file_list.append(file_name)
        label_list.append(int(label))

    # 初始化数据集和标签
    data = None
    labels = np.array(label_list, dtype=np.int32)

    # 循环读取每个栅格数据文件
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)


        # 打开栅格数据文件
        dataset = gdal.Open(input_files[i])

        # 获取栅格数据信息
        rows = dataset.RasterYSize
        cols = dataset.RasterXSize
        bands = dataset.RasterCount

        # 初始化数据集
        if data is None:
            data = np.zeros((len(file_list), rows, cols, bands), dtype=np.float32)

        # 读取栅格数据并存入数据集
        for j in range(bands):
            band = dataset.GetRasterBand(j + 1)
            data[i, :, :, j] = band.ReadAsArray().astype(np.float32)

        # 关闭栅格数据文件
        dataset = None

    # 定义npy文件保存路径
    npy_file = 'txt2'

    # 保存数据集和标签为npy文件
    np.save(npy_file, {'data': data, 'labels': labels})

