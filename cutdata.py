# -*- coding: utf-8 -*-
import arcpy
def cut(point_file,raster_file):
    # arcpy.CheckOutExtension('Spatial')
    # arcpy.env.workspace = r"D:\user\1"

    # 输入点图层和栅格图层
    # point_file = r"text1\point3.lyr"
    #
    # raster_file = r"text2\raster.tif"

    # 读取点图层和栅格图层
    point_layer = arcpy.mapping.Layer(point_file)
    raster_layer = arcpy.Raster(raster_file)

    # 获取栅格图层的空间参考和栅格单元大小
    spatial_ref = arcpy.Describe(raster_layer).spatialReference
    cell_size = raster_layer.meanCellWidth
    # 设置输出栅格数据集名称
    out_name = "out_raster"
    # 设置栅格数据类型和波段数
    data_type = "GRID"
    num_bands = 1
    out_path = r"text5"
    # 构建输出栅格数据集的路径
    out_raster_dataset = out_path + "\\" + out_name + ".tif"
    # 创建空的栅格数据集
    #arcpy.CreateRasterDataset_management(out_path, out_name, cell_size,"16_BIT_SIGNED", spatial_ref, num_bands)
    # 获取点图层的空间参考
    #point_desc = arcpy.Describe(point_layer)
    #point_spatial_ref = point_desc.spatialReference
    # 将点图层转换为栅格图层
    #point_raster = arcpy.PointToRaster_conversion(point_layer, "OBJECTID", out_raster_dataset, "MOST_FREQUENT", "", cell_size)
    i=0
    # 遍历点图层中的所有点
    with arcpy.da.SearchCursor(point_layer, ["SHAPE@XY"]) as cursor:
        for row in cursor:
            # 获取点坐标并计算提取栅格的范围
            num=4
            i=i+1
            point_x, point_y = row[0]
            xmin = point_x - num * cell_size
            ymin = point_y - num * cell_size
            xmax = point_x + num * cell_size
            ymax = point_y + num * cell_size
            extent = arcpy.Extent(xmin, ymin, xmax, ymax)
            polygon = arcpy.Polygon(arcpy.Array([arcpy.Point(extent.XMin, extent.YMin),
                                                 arcpy.Point(extent.XMax, extent.YMin),
                                                 arcpy.Point(extent.XMax, extent.YMax),
                                                 arcpy.Point(extent.XMin, extent.YMax)]),
                                    spatial_ref)

            out_raster_dataset = out_path + "\\" + out_name + str(111+i)+".tif"
            # 提取栅格数据并保存输出
            subset_raster = arcpy.sa.ExtractByMask(raster_layer, polygon)
            subset_raster.save(out_raster_dataset)
