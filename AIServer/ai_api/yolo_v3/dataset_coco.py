import numpy as np
import tensorflow as tf
import random
from PIL import Image, ImageFilter
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import sys
import os
sys.path.append(os.getcwd())
import ai_api.utils.image_helper as ImageHelper


class DataGenerator():
    def __init__(self, image_path, label_path, classes_path, batch_size, anchors, input_shape=(416, 416), is_mean=True):
        self.image_path = image_path
        self.label_path = label_path
        self.classes_path = classes_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.is_mean = is_mean

        # 加载类型
        self.LoadClasses()
        # 加载标签
        self.LoadLabels()

    def LoadClasses(self):
        '''加载类型'''
        print('加载类型：', self.classes_path)
        with open(self.classes_path, 'r', encoding='utf-8') as f:
            self.classes = f.readlines()
        self.classes = [c.strip() for c in self.classes]
        self.classes_num = len(self.classes)
        print('已加载%d个类型' % (self.classes_num))

    def LoadLabels(self):
        '''加载标签'''
        print('加载标签：', self.label_path)
        self.labels = []
        with open(self.label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('|')
                image_full_path = os.path.join(self.image_path, line[0])
                # print('image_full_path:', image_full_path)
                classes = []
                boxes = []
                for i in range(1, len(line)):
                    if line[i] == '':
                        continue
                    info = line[i].split(',')
                    if info[0] not in self.classes:
                        print('标签异常：', info[0], image_full_path)
                        continue
                    classes.append(self.classes.index(info[0]))
                    x1 = float(info[1])
                    y1 = float(info[2])
                    x2 = float(info[3])
                    y2 = float(info[4])
                    # print('boxes:', [x1, y1, x2, y2])
                    boxes.append([x1, y1, x2, y2])
                self.labels.append({
                    'image_path': image_full_path,
                    'classes': classes,
                    'boxes': boxes
                    })
        self.labels_num = len(self.labels)
        print('已加载%d个标签' % (self.labels_num))

    def Rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def GetRandomData(self, label, is_random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True, flip=False):
        '''随机数据扩展'''
        # print(line)
        image = Image.open(label['image_path'])
        iw, ih = image.size
        # print('image size:', iw, ih)
        h, w = self.input_shape
        if len(label['classes'])==0:
            box = np.array([])
        else:
            box = np.concatenate([np.array(label['boxes']), np.expand_dims(np.array(label['classes'], dtype=np.float), axis=-1)], axis=-1)
        # 增加模糊
        ksize = random.randint(0, 5)
        if ksize>0:
            image = image.filter(ImageFilter.GaussianBlur(radius=ksize))

        if not is_random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image_data=0
            if proc_img:
                image = image.resize((nw,nh), Image.BICUBIC)
                new_image = Image.new('RGB', (w,h), (128,128,128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image)/255.

            # correct boxes
            box_data = np.zeros((max_boxes,5))
            if len(box)>0:
                np.random.shuffle(box)
                if len(box)>max_boxes: box = box[:max_boxes]
                box[:, [0,2]] = box[:, [0,2]]*scale + dx
                box[:, [1,3]] = box[:, [1,3]]*scale + dy
                box_data[:len(box)] = box
            return image_data, box_data

        # resize image
        # 随机缩放
        rand = self.Rand
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        # 偏移图片
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        # 随机翻转图片
        if flip:
            flip = rand()<.5
            if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        # 颜色偏移
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x) # numpy array, 0 to 1

        # correct boxes
        # 更新偏移后的框
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box 去掉无效框
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data

    def PreprocessTrueBoxes(self, true_boxes):
        assert (true_boxes[..., 4]<self.classes_num).all(), 'class id must be less than classes_num'
        num_layers = len(self.anchors)//3 # default setting
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

        true_boxes = np.array(true_boxes, dtype='float')
        input_shape = np.array(self.input_shape, dtype='int')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        # 归一化
        true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

        m = true_boxes.shape[0]
        # [(13, 13), (26, 26), (52, 52)]
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        # [(batch_size, 13, 13, 3, 5+classes_num), (batch_size, 26, 26, 3, 5+classes_num), (batch_size, 52, 52, 3, 5+classes_num)]
        y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+self.classes_num),
            dtype='float') for l in range(num_layers)]

        # Expand dim to apply broadcasting.
        # shape=(1, N, 2), 中值为0
        anchors = np.expand_dims(self.anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        # 过滤不合格的框坐标
        valid_mask = boxes_wh[..., 0]>0

        # 遍历每张图的框
        for b in range(m):
            # Discard zero rows.
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh)==0: continue
            # Expand dim to apply broadcasting.
            wh = np.expand_dims(wh, -2)
            # (1, T, 2)
            box_maxes = wh / 2.
            box_mins = -box_maxes
            # 计算IOU
            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # Find best anchor for each true box
            # 与框最匹配的候选框下标
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        # 中心
                        i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int')
                        j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int')
                        # 候选框在当前层的下标
                        k = anchor_mask[l].index(n)
                        # 分类
                        c = true_boxes[b,t, 4].astype('int')
                        # print('box:', b, j, i, k, true_boxes[b,t, 0:4])
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1
        return y_true
        
    def Generate(self):
        # 记录存在分类
        class_list = set()
        # 图片对于分类列表
        image_class_list = {}
        # 读取素材标签，用于素材均衡
        if self.is_mean:
            for label in self.labels:
                image_path = label['image_path']
                image_class_list[image_path]=set()
                for c in label['classes']:
                    class_list.add(c)
                    image_class_list[image_path].add(c)
                image_class_list[image_path]=list(image_class_list[image_path])
            class_list = list(class_list)
            print('存在标签：', class_list)

        n = len(self.labels)
        i = 0
        class_index = 0
        clone_labels = self.labels.copy()
        while True:
            image_data = []
            box_data = []
            batch_count = 0
            while batch_count < self.batch_size:
                if i==0:
                    random.shuffle(clone_labels)
                # 数据平均
                label = clone_labels[i]
                if len(class_list)>0 and self.is_mean:
                    if class_list[class_index] not in image_class_list[label['image_path']]:
                        i = (i+1) % n
                        continue
                    
                    # 找下一个类型
                    if class_index < len(class_list)-1:
                        class_index += 1
                    else:
                        class_index = 0
                # print('image_path:', label['image_path'])
                # input_shape：(416, 416)
                image, box = self.GetRandomData(label, is_random=True)
                image_data.append(image)
                box_data.append(box)
                batch_count += 1
                i = (i+1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            # 这里是归一化后的中心坐标与宽高，不是偏移
            y_true = self.PreprocessTrueBoxes(box_data)
            # print(y_true)
            # 多输出一定要元组
            # print('image_data:', image_data.shape)
            # print('y_true:', y_true[0].shape, y_true[1].shape, y_true[2].shape)
            yield image_data, tuple(y_true)


def GetDataSet(image_path, label_path, classes_path, batch_size, anchors, input_shape=(416, 416), is_mean=True):
    '''获取数据集'''
    data_generator = DataGenerator(image_path, label_path, classes_path, batch_size, anchors, input_shape, is_mean)
    # 数据预处理
    dataset = tf.data.Dataset.from_generator(data_generator.Generate, (tf.float32, (tf.float32, tf.float32, tf.float32)))
    # for x, y in dataset.take(1):
    #     print(x.shape, y[0].shape, y[1].shape, y[2].shape)
    return dataset, data_generator

def main():
    anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
    anchors = np.array(anchors).reshape(-1, 2)
    data_set = GetDataSet(image_path='E:\\MyFiles\\labels\\coco2017\\train2017',
        label_path='E:\\MyFiles\\git\\tensorflow2-yolov4\\AIServer\\data\\coco_train2017_labels.txt',
        classes_path='E:\\MyFiles\\git\\tensorflow2-yolov4\\AIServer\\data\\coco_classes.txt', batch_size=3, anchors=anchors)
    

if __name__ == '__main__':
    main()
