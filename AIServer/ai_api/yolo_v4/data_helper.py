import numpy as np
import os
import sys
import random
import tensorflow as tf

sys.path.append(os.getcwd())
import ai_api.utils.image_helpler as ImageHelpler


def GetRandomImage(image, points=None):
    '''生成随机图片'''
    # 变换图像
    random_offset_x = random.random()*90-45
    random_offset_y = random.random()*90-45
    random_angle_x = random.random()*60-30
    random_angle_y = random.random()*60-30
    random_angle_z = random.random()*40-20
    random_scale = random.random()*0.8+0.6
    # random_offset_x = 0
    # random_offset_y = 0
    # random_angle_x = 0
    # random_angle_y = 0
    # random_scale = 1
    random_img, org, dst, perspective_points = ImageHelpler.opencvPerspective(image, offset=(random_offset_x, random_offset_y, 0),
                                                                              angle=(random_angle_x, random_angle_y, , random_angle_z), scale=(random_scale, random_scale, 1), points=points)
    # 增加线条
    # random_img = image_helpler.opencvRandomLines(random_img, 8)
    # 增加噪声
    random_img = ImageHelpler.opencvNoise(random_img)
    # 颜色抖动
    random_img = ImageHelpler.opencvRandomColor(random_img)

    # cv2.imwrite(path, image)
    return random_img, perspective_points

def main():
    pass

if __name__ == '__main__':
    main()
