import file_helper as FileHelper
import image_helper as ImageHelper
import argparse
import xml.dom.minidom as xml
import numpy as np

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='labels')
args = parser.parse_args()

label_path = args.file_path
file_list = FileHelper.ReadFileList(label_path, r'.xml$')

# 网络图片大小
image_size = (416, 416)

# 缩放后的图片宽高
label_image_wh = []
# 遍历标签文件
for xml_path in file_list:
    # 读取xml文件
    # print('xml_path:', xml_path)
    xml_data = xml.parse(xml_path)
    collection = xml_data.documentElement
    # 读取标签图大小
    label_image_size = collection.getElementsByTagName("size")[0]
    label_image_width = int(label_image_size.getElementsByTagName("width")[0].childNodes[0].data)
    label_image_height = int(label_image_size.getElementsByTagName("height")[0].childNodes[0].data)
    # 对象列表
    object_list = collection.getElementsByTagName("object")

    # 读取目标框信息
    for object_item in object_list:
        # 原始点列表
        object_bndbox = object_item.getElementsByTagName("bndbox")[0]
        point_x1 = float(object_bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
        point_y1 = float(object_bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
        point_x2 = float(object_bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
        point_y2 = float(object_bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
        points = np.array([[point_x1, point_y1, point_x2, point_y2]])
        points = points.reshape((-1,2))
        points, _ = ImageHelper.opencvProportionalResizePoint((label_image_width, label_image_height), image_size, points=points)
        points = points.reshape((-1,4))
        label_image_wh.append([points[0][2]-points[0][0], points[0][3]-points[0][1]])
label_image_wh = np.array(label_image_wh, dtype=np.float32)

print('目标数量：', label_image_wh.shape)
# 初始化9个宽高
anchors_wh = []
for i in range(9):
    anchors_wh.append(label_image_wh[i])
anchors_wh = np.array(anchors_wh, dtype=np.float32)
print('初始化宽高：', anchors_wh)
# 遍历所有宽高
while True:
    anchors_wh_old = anchors_wh.copy()
    # 最近点集合
    anchors_points = [[] for _ in range(len(anchors_wh))]
    for label_image_wh_index in range(label_image_wh.shape[0]):
        min_index = 0
        min_d = None
        for anchors_index in range(len(anchors_wh)):
            d = np.sum(np.square(anchors_wh[anchors_index] - label_image_wh[label_image_wh_index]))
            if min_d is None or d < min_d:
                min_d = d
                min_index = anchors_index
        anchors_points[min_index].append(label_image_wh[label_image_wh_index])
    anchors_points = np.array(anchors_points)
    # 按平均值调整宽高
    for anchors_index in range(len(anchors_wh)):
        anchors_wh[anchors_index] = np.mean(anchors_points[anchors_index], axis=0)
    # print('调整后的宽高：', anchors_wh)
    if (anchors_wh_old == anchors_wh).all():
        break
# 排序
anchors_wh = anchors_wh.tolist()
print('kmeans的宽高：', np.array(anchors_wh))
anchors_wh.sort(key=lambda a: a[0]*a[1])
print('kmeans的宽高(排序)：', np.array(anchors_wh))


