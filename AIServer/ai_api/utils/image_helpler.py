import base64
import io
import cv2
import numpy as np
import math
import random
from PIL import Image


def base64ToBytes(base64_data):
    '''base64转字节数组'''
    bytes_data = base64.b64decode(base64_data)
    return bytes_data


def bytesTobase64(bytes_data):
    '''字节数组转base64'''
    base64_data = base64.b64encode(bytes_data).decode('utf-8')
    return base64_data


def bytesToPILImage(bytes_data):
    '''字节数组转PIL图片'''
    img_bytes = io.BytesIO(bytes_data)
    pil_img = Image.open(img_bytes)
    # pil_img.show()
    return pil_img


def bytesToOpencvImage(bytes_data):
    '''字节数组转Opencv图片'''
    nparr = np.fromstring(bytes_data, np.uint8)
    opencv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.imshow("Opencv", opencv_img)
    # cv2.waitKey(0)
    return opencv_img


def opencvImageToBytes(opencv_img, ext='.jpg'):
    '''Opencv图片转字节数组'''
    res_val, bytes_data = cv2.imencode(ext, opencv_img)
    if not res_val:
        print('opencvImageToBytes:转换失败')
    return bytes_data


def fileToOpencvImage(file_path):
    '''Opencv读取图片文件(中文路径)'''
    nparr = np.fromfile(file_path, dtype=np.uint8)
    opencv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return opencv_img


def opencvImageToFile(file_path, opencv_img, ext='.jpg'):
    '''Opencv保存图片文件(中文路径)'''
    opencvImageToBytes(opencv_img, ext).tofile(file_path)


def opencvToPilImage(opencv_img):
    '''Opencv转PIL图片'''
    pil_img = Image.fromarray(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    return pil_img


def pilToOpencvImage(pil_img):
    '''PIL转Opencv图片'''
    opencv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return opencv_img


def showPILImage(pil_img):
    '''显示PIL图片'''
    pil_img.show()


def showOpencvImage(opencv_img):
    '''显示Opencv图片'''
    cv2.imshow("Opencv", opencv_img)
    cv2.waitKey(0)


def opencvGetImageSize(opencv_img):
    '''Opencv获取图片宽高'''
    height = opencv_img.shape[0]
    width = opencv_img.shape[1]
    return width, height


def opencvScale(opencv_img, x_scale, y_scale, bg_color=(255, 255, 255)):
    '''Opencv缩放图片，大小不变'''
    result_img = opencvPerspective(
        opencv_img, scale=(x_scale, y_scale, 0), bg_color=bg_color)
    return result_img


def opencvOffset(opencv_img, x_offset, y_offset, bg_color=(255, 255, 255)):
    '''Opencv平移图片，大小不变'''
    result_img = opencvPerspective(
        opencv_img, offset=(x_offset, y_offset, 0), bg_color=bg_color)
    return result_img


def opencvRotation(opencv_img, angle, bg_color=(255, 255, 255)):
    '''Opencv旋转图片，大小不变'''
    result_img = opencvPerspective(
        opencv_img, angle=(0, 0, angle), bg_color=bg_color)
    return result_img


def opencvPerspective(opencv_img, angle=(0, 0, 0), offset=(0, 0, 0), scale=(1, 1, 1), bg_color=None, points=None):
    '''Opencv透视变换图片，大小不变'''
    # 图片大小
    width, height = opencvGetImageSize(opencv_img)
    # 角度转弧度
    radian = np.radians(angle)
    # 图片原始四个角
    p_center = np.float32([width/2, height/2, 0, 0])  # 左上
    p1 = np.float32([0, 0, 0, 1]) - p_center  # 左上
    p2 = np.float32([width, 0, 0, 1]) - p_center  # 右上
    p3 = np.float32([0, height, 0, 1]) - p_center  # 左下
    p4 = np.float32([width, height, 0, 1]) - p_center  # 右下
    # 变换矩阵
    M = np.float32([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    # 平移
    M = np.matmul(M,
                  np.float32([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [offset[0], offset[1], offset[2], 1]]))
    # 旋转x轴
    M = np.matmul(M,
                  np.float32([[1, 0, 0, 0],
                              [0, math.cos(radian[0]), -
                               math.sin(radian[0]), 0],
                              [0, -math.sin(radian[0]),
                               math.cos(radian[0]), 0],
                              [0, 0, 0, 1]]))
    # 旋转y轴
    M = np.matmul(M,
                  np.float32([[math.cos(radian[1]), 0, math.sin(radian[1]), 0],
                              [0, 1, 0, 0],
                              [-math.sin(radian[1]), 0,
                               math.cos(radian[1]), 0],
                              [0, 0, 0, 1]]))
    # 旋转z轴
    M = np.matmul(M,
                  np.float32([[math.cos(radian[2]), math.sin(radian[2]), 0, 0],
                              [-math.sin(radian[2]),
                               math.cos(radian[2]), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]))
    # 缩放
    M = np.matmul(M,
                  np.float32([[scale[0], 0, 0, 0],
                              [0, scale[1], 0, 0],
                              [0, 0, scale[2], 0],
                              [0, 0, 0, 1]]))
    # 变换顶点
    dst1 = np.matmul(p1, M)
    dst2 = np.matmul(p2, M)
    dst3 = np.matmul(p3, M)
    dst4 = np.matmul(p4, M)

    list_dst = np.float32([dst1, dst2, dst3, dst4])

    org = np.float32([[0, 0],
                      [width, 0],
                      [0, height],
                      [width, height]])

    dst = np.zeros((4, 2), np.float32)

    # 转换点列表
    result_points = []
    if points is not None:
        for p in points:
            tmp_points = np.float32([p[0], p[1], 0, 1]) - p_center
            tmp_points = np.matmul(tmp_points, M)
            tmp_x = tmp_points[0]*width/(width+tmp_points[2])+p_center[0]
            tmp_y = tmp_points[1]*height/(height+tmp_points[2])+p_center[1]
            result_points.append([tmp_x, tmp_y])
    result_points = np.float32(result_points)

    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i, 0]*width/(width+list_dst[i, 2])+p_center[0]
        dst[i, 1] = list_dst[i, 1]*height/(height+list_dst[i, 2])+p_center[1]
    # print('org', org)
    # print('dst', dst)
    # print('list_dst', list_dst)
    # print('M', M)
    # 随机背景色
    if bg_color is None:
        bg_color = getRandomColor()
    result_img = opencvPerspectiveP(opencv_img, org, dst, bg_color)
    return result_img, org, dst, result_points


def opencvPerspectiveP(opencv_img, org, dst, bg_color=None):
    '''Opencv透视变换图片，大小不变'''
    # 图片大小
    width, height = opencvGetImageSize(opencv_img)
    warpM = cv2.getPerspectiveTransform(org, dst)
    # 随机背景色
    if bg_color is None:
        bg_color = getRandomColor()
    result_img = cv2.warpPerspective(
        opencv_img, warpM, (width, height), borderValue=bg_color)
    return result_img


def getRandomColor():
    '''获取随机颜色'''
    c1 = random.randint(0, 255)
    c2 = random.randint(0, 255)
    c3 = random.randint(0, 255)
    return (c1, c2, c3)


def opencvRandomLines(opencv_img, line_count):
    '''Opencv图片绘制随机线条'''
    # 图片大小
    width, height = opencvGetImageSize(opencv_img)
    result_img = np.copy(opencv_img)
    for i in range(line_count):
        p1 = (random.randint(0, width-1), random.randint(0, height-1))
        p2 = (random.randint(0, width-1), random.randint(0, height-1))
        cv2.line(result_img, p1, p2, getRandomColor(),
                 int(random.random()*4+1))
    return result_img


def opencvNoise(opencv_img):
    '''Opencv图片添加噪点'''
    # 图片大小
    width, height = opencvGetImageSize(opencv_img)
    noise = np.random.random((height, width, 3)) * 30
    noise = noise.astype(np.int32)
    tmp_img = opencv_img.astype(np.int32)
    tmp_img = tmp_img + noise - 15
    tmp_img = np.minimum(tmp_img, 255)
    tmp_img = np.maximum(tmp_img, 0)
    result_img = tmp_img.astype(np.uint8)
    return result_img


def opencvRandomColor(opencv_img):
    '''Opencv图片随机颜色'''
    tmp_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2HSV)
    tmp_img = tmp_img.astype(np.int32)
    # H,色相
    tmp_img[:,:,0] = tmp_img[:,:,0] + int(random.random() * 255) - 127
    # S,饱和度
    tmp_img[:,:,1] = tmp_img[:,:,1] + int(random.random() * 60) - 30
    # V,亮度
    if np.mean(tmp_img[:,:,2]) < 150:
        # print('V:', np.mean(tmp_img[:,:,2]))
        tmp_img[:,:,2] = tmp_img[:,:,2] + int(random.random() * 80) - 40
    else:
        # print('V:', np.mean(tmp_img[:,:,2]))
        tmp_img[:,:,2] = tmp_img[:,:,2] + int(random.random() * 110) - 80
    tmp_img = np.minimum(tmp_img, 255)
    tmp_img = np.maximum(tmp_img, 0)
    result_img = tmp_img.astype(np.uint8)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_HSV2BGR)
    return result_img

def getRandomColor():
    '''获取随机颜色(r,g,b)'''
    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)
    result_color = (r, g, b)
    return result_color


def opencvReflective(opencv_img, bg_img, alpha):
    '''Opencv图片按透明的重叠，模拟反光'''
    # 图片大小
    width, height = opencvGetImageSize(opencv_img)
    bg_img = cv2.resize(bg_img, (width, height), interpolation=cv2.INTER_AREA)
    result_img = cv2.addWeighted(opencv_img, alpha, bg_img, 1 - alpha, 0)
    return result_img


def opencvProportionalResize(opencv_img, size, points=None, bg_color=None):
    '''等比例缩放'''
    # 图片大小
    width, height = opencvGetImageSize(opencv_img)
    new_width, new_height = size[0], size[1]
    #长边缩放为min_side
    if width / height > new_width / new_height:
        resize_width = new_width
        resize_height = int((height / width) * resize_width)
    else:
        resize_height = new_height
        resize_width = int((width / height) * resize_height)
    result_img = cv2.resize(opencv_img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    # 填充边界
    top = (new_height-resize_height)//2
    bottom = new_height-resize_height-top
    left = (new_width-resize_width)//2
    right = new_width-resize_width-left
    # 随机背景色
    if bg_color is None:
        bg_color = getRandomColor()
    result_img = cv2.copyMakeBorder(result_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=bg_color)
    padding = (top, bottom, left, right)
    # 转换点列表
    result_points = []
    if points is not None:
        for p in points:
            tmp_x = p[0]*resize_width/width+left
            tmp_y = p[1]*resize_height/height+top
            result_points.append([tmp_x, tmp_y])
    result_points = np.float32(result_points)
    return result_img, result_points, padding


def opencvCut(opencv_img, box):
    '''
    图片裁剪(如框超出图片边缘会裁剪)
    box:(x1, y1, x2, y2)
    '''
    # 图片大小
    width, height = opencvGetImageSize(opencv_img)
    box[0]=np.maximum(box[0], 0)
    box[1]=np.maximum(box[1], 0)
    box[2]=np.minimum(box[2], width)
    box[3]=np.minimum(box[3], height)
    # height，width
    result_img = opencv_img[box[1]:box[3], box[0]:box[2]]
    return result_img

def main():
    # 读取图片
    img = fileToOpencvImage('./data/a.jpg')
    result_img, _, _ = opencvProportionalResize(img, (400, 400))
    print('result_img', type(result_img))
    # result_img = opencvRandomColor(result_img)
    random_offset_x = random.random()*90-45
    random_offset_y = random.random()*90-45
    random_angle_x = random.random()*60-30
    random_angle_y = random.random()*60-30
    random_scale = random.random()*1.0+0.8
    # random_offset_x = 0
    # random_offset_y = 0
    # random_angle_x = 0
    # random_angle_y = 0
    # random_scale = 1
    # 点列表
    points = np.float32([[50, 50], # 左上
                        [50, 350], # 左下
                        [350, 50], # 右上
                        [350, 350]]) # 右下
    result_img, org, dst, perspective_points = opencvPerspective(result_img, offset=(random_offset_x, random_offset_y, 0),
                                                    angle=(random_angle_x, random_angle_y, 0), scale=(random_scale, random_scale, 1), points=points)
    width, height = opencvGetImageSize(result_img)
    print('size:', width, height)
    showOpencvImage(result_img)
    # 读取反光图片
    bg_img = fileToOpencvImage('./data/b.jpg')
    # 随机反光
    result_img = opencvReflective(result_img, bg_img, 0.85)
    showOpencvImage(result_img)


if __name__ == '__main__':
    main()
