import numpy as np
import os
import sys
import random
import tensorflow as tf

sys.path.append(os.getcwd())
import ai_api.utils.image_helper as ImageHelper

def GetRandomImage(image, points=None):
    '''生成随机图片'''
    random_img = image
    # 图片大小
    # width, height = ImageHelper.opencvGetImageSize(random_img)
    # 画矩形
    # cv2.rectangle(image, (20, 20), (380, 380), tuple(np.random.randint(0, 30, (3), dtype=np.int32)), thickness=8)
    # 增加模糊
    # ksize = random.randint(0, 5)
    # if ksize>0:
    #     random_img = ImageHelper.opencvBlur(random_img, (ksize, ksize))
    # 变换图像, 旋转会导致框不准
    random_offset_x = random.random()*90-45
    random_offset_y = random.random()*90-45
    # random_angle_x = random.random()*60-30
    # random_angle_y = random.random()*60-30
    # random_angle_z = random.random()*40-20
    random_scale = random.random()*1.0+0.5
    # random_offset_x = 0
    # random_offset_y = 0
    random_angle_x = 0
    random_angle_y = 0
    random_angle_z = 0
    # random_scale = 1
    random_img, org, dst, perspective_points = ImageHelper.opencvPerspective(random_img, offset=(random_offset_x, random_offset_y, 0),
                                                                              angle=(random_angle_x, random_angle_y, random_angle_z),
                                                                              scale=(random_scale, random_scale, 1), points=points,
                                                                              bg_color=None, bg_mode=None)
    # 增加线条
    # random_img = image_helper.opencvRandomLines(random_img, 8)
    # 增加噪声
    # random_img = ImageHelper.opencvNoise(random_img)
    # 颜色抖动
    # random_img = ImageHelper.opencvRandomColor(random_img)

    # cv2.imwrite(path, image)
    return random_img, perspective_points


def main():
    pass

if __name__ == '__main__':
    main()
