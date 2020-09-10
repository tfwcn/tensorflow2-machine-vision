import tensorflow as tf
import os
import sys
import time
import numpy as np
import random
import cv2
import math

sys.path.append(os.getcwd())
from ai_api.ai_models.utils.radam import RAdam
from ai_api.ai_models.utils.mish import Mish

class Yolov4Model():
    '''Yolov4模型'''
    def __init__(self, anchors_wh, classes_num, image_size, layers_size, train_iou_thresh=0.5,
                 model_path='./data/yolov4_model'):
        # 设置GPU显存自适应
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        print(gpus, cpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # if len(gpus) > 1:
        #     tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        # elif len(cpus) > 0:
        #     tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
        # 模型路径
        self.model_path = model_path
        # 每层候选框大小
        self.anchors_wh = anchors_wh
        # 每层候选框数量
        self.anchors_num = anchors_wh.shape[1]
        # 类型数量
        self.classes_num = classes_num
        # 输入图片大小
        self.image_size = image_size
        # 输出每层大小
        self.layers_size = layers_size
        # IOU低于这个值，则视为背景
        self.train_iou_thresh = train_iou_thresh
        # 加载模型路径
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # 建立模型
        self.BuildInfoModel()
        # 加载模型
        self.LoadModel()

    def BuildInfoModel(self):
        '''建立模型'''
        # 建立特征提取模型
        self.BuildModel()
        # 优化器
        self.optimizer = RAdam(learning_rate=1e-4)
        # 保存模型
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, 
            feature_model=self.feature_model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.model_path, max_to_keep=3)

    def ConvLayer(self, filters, kernel_size, strides=(1, 1), padding='same', use_bias=False):
        '''卷积，默认无bias'''
        def conv_layer(x):
            x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides,
                                       kernel_regularizer=tf.keras.regularizers.l1(5e-4),
                                       use_bias=use_bias)(x)
            return x
        return conv_layer

    def ConvBnMishLayer(self, filters, kernel_size, strides=(1, 1), padding='same'):
        '''卷积+正则化+Mish'''
        def conv_bn_mish_layer(x):
            x = self.ConvLayer(filters, kernel_size,
                               strides=strides, padding=padding)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Mish()(x)
            return x
        return conv_bn_mish_layer

    def ConvBnLReluLayer(self, filters, kernel_size, strides=(1, 1), padding='same'):
        '''卷积+正则化+LeakyReLU'''
        def conv_bn_lrelu_layer(x):
            x = self.ConvLayer(filters, kernel_size,
                               strides=strides, padding=padding)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            return x
        return conv_bn_lrelu_layer

    def BlocksLayer(self, filters, blocks_num):
        '''blocks结构'''
        def blocks_layer(x):
            x = self.ConvBnMishLayer(filters, (3, 3), strides=(2, 2))(x)
            if blocks_num == 1:
                y = self.ConvBnMishLayer(filters, (1, 1))(x)
                z = self.ConvBnMishLayer(filters // 2, (1, 1))(y)
                z = self.ConvBnMishLayer(filters, (3, 3))(z)
                y = tf.keras.layers.Add()([y, z])
                y = self.ConvBnMishLayer(filters, (1, 1))(y)
                x = self.ConvBnMishLayer(filters, (1, 1))(x)
                x = tf.keras.layers.Concatenate()([y, x])
                x = self.ConvBnMishLayer(filters, (1, 1))(x)
            else:
                y = self.ConvBnMishLayer(filters // 2, (1, 1))(x)
                for _ in range(blocks_num):
                    z = self.ConvBnMishLayer(filters // 2, (1, 1))(y)
                    z = self.ConvBnMishLayer(filters // 2, (3, 3))(z)
                    y = tf.keras.layers.Add()([y, z])
                y = self.ConvBnMishLayer(filters // 2, (1, 1))(y)
                x = self.ConvBnMishLayer(filters // 2, (1, 1))(x)
                x = tf.keras.layers.Concatenate()([y, x])
                x = self.ConvBnMishLayer(filters, (1, 1))(x)
            return x
        return blocks_layer

    def LastLayer(self, filters):
        '''最后结果'''
        def last_layer(x, z=None):
            if z is not None:
                z = self.ConvBnLReluLayer(filters, (1, 1))(z)
                z = tf.keras.layers.UpSampling2D((2, 2))(z)
                x = self.ConvBnLReluLayer(filters, (1, 1))(x)
                x = tf.keras.layers.Concatenate()([z, x])
            x = self.ConvBnLReluLayer(filters, (1, 1))(x)
            x = self.ConvBnLReluLayer(filters * 2, (3, 3))(x)
            x = self.ConvBnLReluLayer(filters, (1, 1))(x)
            if z is None:
                # SPP
                x2 = tf.keras.layers.MaxPool2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
                x3 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
                x4 = tf.keras.layers.MaxPool2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
                x = tf.keras.layers.Concatenate()([x2, x3, x4, x])
                x = self.ConvBnLReluLayer(filters, (1, 1))(x)
            x = self.ConvBnLReluLayer(filters * 2, (3, 3))(x)
            x = self.ConvBnLReluLayer(filters, (1, 1))(x)
            return x
        return last_layer

    def OutputLayer(self, filters):
        '''最后结果'''
        def output_layer(x, z=None):
            if z is not None:
                z = self.ConvBnLReluLayer(filters, (3, 3), strides=(2, 2))(z)
                x = tf.keras.layers.Concatenate()([z, x])
                x = self.ConvBnLReluLayer(filters, (1, 1))(x)
                x = self.ConvBnLReluLayer(filters * 2, (3, 3))(x)
                x = self.ConvBnLReluLayer(filters, (1, 1))(x)
                x = self.ConvBnLReluLayer(filters * 2, (3, 3))(x)
                x = self.ConvBnLReluLayer(filters, (1, 1))(x)
            y = self.ConvBnLReluLayer(filters * 2, (3, 3))(x)
            y = self.ConvLayer(self.anchors_num * (5 + self.classes_num), (1, 1), use_bias=True)(y)
            return x, y
        return output_layer

    def BuildModel(self):
        '''建立模型'''
        # 所有参数
        input_classes = tf.keras.Input([416, 416, 3], dtype=tf.float32)
        # (416 * 416)
        x = self.ConvBnLReluLayer(32, (3, 3))(input_classes)
        # (208 * 208)
        x = self.BlocksLayer(64, 1)(x)
        # (104 * 104)
        x = self.BlocksLayer(128, 2)(x)
        # (52 * 52)
        x = self.BlocksLayer(256, 8)(x)
        y3 = x
        # (26 * 26)
        x = self.BlocksLayer(512, 8)(x)
        y2 = x
        # (13 * 13)
        x = self.BlocksLayer(1024, 4)(x)
        y1 = x
        # 计算y1,(13 * 13)
        y1 = self.LastLayer(512)(y1)
        # 计算y2,(26 * 26)
        y2 = self.LastLayer(256)(y2, y1)
        # 计算y3,(52 * 52)
        y3 = self.LastLayer(128)(y3, y2)
        # 计算y3,(52 * 52)
        z3, y3 = self.OutputLayer(128)(y3)
        # 计算y2,(26 * 26)
        z2, y2 = self.OutputLayer(256)(y2, z3)
        # 计算y1,(13 * 13)
        _, y1 = self.OutputLayer(512)(y1, z2)
        self.feature_model = tf.keras.Model(
            inputs=input_classes, outputs=[y1, y2, y3])

    @tf.function
    def GetBoxes(self, y, anchors_wh):
        '''将偏移转换成真实值，范围0到1'''
        # 拆分特征
        # box_xy:(batch_size, y_w, y_h, anchors_num, 2)
        # box_wh:(batch_size, y_w, y_h, anchors_num, 2)
        # confidence:(batch_size, y_w, y_h, anchors_num, 1)
        # classes:(batch_size, y_w, y_h, anchors_num, classes_num)
        boxes_xy, boxes_wh, confidence, classes = tf.split(
            y, (2, 2, 1, self.classes_num), axis=-1)
        confidence = tf.math.sigmoid(confidence)
        classes = tf.math.sigmoid(classes)
        y_pred_raw = y
        # 特征网格对应实际图片的坐标
        grid_shape = tf.shape(y_pred_raw)[1:3] # height, width
        grid_x = tf.range(0, grid_shape[1])
        grid_y = tf.range(0, grid_shape[0])
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        grid_x = tf.reshape(grid_x, (grid_shape[0], grid_shape[1], 1, 1))
        grid_y = tf.reshape(grid_y, (grid_shape[0], grid_shape[1], 1, 1))
        grid_xy = tf.concat([grid_x, grid_y], axis=-1)
        # 计算真实坐标与相对坐标
        # y_pred
        y_pred_raw_xy = boxes_xy
        y_pred_read_xy = (tf.math.sigmoid(y_pred_raw_xy) + tf.cast(grid_xy, dtype=tf.float32)) / tf.cast(grid_shape[::-1], dtype=tf.float32)
        y_pred_raw_wh = boxes_wh
        y_pred_read_wh = tf.math.exp(y_pred_raw_wh) * tf.cast(anchors_wh, dtype=tf.float32)
        y_pred_read_wh = tf.where(tf.math.is_inf(y_pred_read_wh), tf.zeros_like(y_pred_read_wh), y_pred_read_wh)
        # 计算IOU
        y_pred_read_wh_half = y_pred_read_wh / 2
        y_pred_read_mins = y_pred_read_xy - y_pred_read_wh_half
        y_pred_read_maxes = y_pred_read_xy + y_pred_read_wh_half
        y_pred_boxes = tf.concat([y_pred_read_mins, y_pred_read_maxes], axis=-1)
        # 去掉无效框
        mask = tf.math.logical_and(y_pred_boxes[...,2] > y_pred_boxes[...,0], y_pred_boxes[...,3] > y_pred_boxes[...,1])
        y_pred_boxes = tf.boolean_mask(y_pred_boxes, mask)
        confidence = tf.boolean_mask(confidence, mask)
        classes = tf.boolean_mask(classes, mask)
        return y_pred_boxes, confidence, classes

    @tf.function
    def GetDIOUNMS(self,
                   boxes,
                   scores,
                   max_output_size,
                   iou_threshold=0.5):
        # 分数倒序下标
        scores_sort_indexes = tf.argsort(scores, direction='DESCENDING')
        # 排序后的框
        boxes_sort = tf.gather(boxes, scores_sort_indexes)
        # NMS后的下标
        result_indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True)
        def boxes_foreach(idx, result_indexes, boxes_sort, scores_sort_indexes):
            # 取最高分的box
            if idx >= max_output_size:
                return -1, result_indexes, boxes_sort, scores_sort_indexes
            boxes_num = tf.shape(boxes_sort)[0]
            if boxes_num == 0:
                return -1, result_indexes, boxes_sort, scores_sort_indexes
            boxes_top = boxes_sort[0:1, :]
            indexes_top = scores_sort_indexes[0]
            if boxes_num > 1:
                # 计算IOU
                boxes_other = boxes_sort[1:, :]
                indexes_other = scores_sort_indexes[1:]
                _, diou, _ = self.GetIOU(boxes_top, boxes_other)
                iou_mask = diou < iou_threshold
                boxes_sort = tf.boolean_mask(boxes_other, iou_mask)
                scores_sort_indexes = tf.boolean_mask(indexes_other, iou_mask)
                result_indexes = result_indexes.write(idx, indexes_top)
                return idx + 1, result_indexes, boxes_sort, scores_sort_indexes
            else:
                result_indexes = result_indexes.write(idx, indexes_top)
                return -1, result_indexes, boxes_sort, scores_sort_indexes
        _, result_indexes, _, _ = tf.while_loop(lambda i, x1, x2, x3: tf.math.not_equal(i, -1), boxes_foreach, [0, result_indexes, boxes_sort, scores_sort_indexes])
        result_indexes = result_indexes.stack()
        return result_indexes

    @tf.function
    def GetNMSBoxes(self, y1, y2, y3, scores_thresh, iou_thresh):
        '''经过NMS去重后，转换成框坐标'''
        # 拆分维度
        y1 = tf.reshape(y1, [tf.shape(y1)[0], tf.shape(
            y1)[1], tf.shape(y1)[2], self.anchors_num, -1])
        y2 = tf.reshape(y2, [tf.shape(y2)[0], tf.shape(
            y2)[1], tf.shape(y2)[2], self.anchors_num, -1])
        y3 = tf.reshape(y3, [tf.shape(y3)[0], tf.shape(
            y3)[1], tf.shape(y3)[2], self.anchors_num, -1])
        
        y1_pred_boxes, y1_pred_confidence, y1_pred_classes = self.GetBoxes(
            y1, self.anchors_wh[0]/self.image_size)
        y2_pred_boxes, y2_pred_confidence, y2_pred_classes = self.GetBoxes(
            y2, self.anchors_wh[1]/self.image_size)
        y3_pred_boxes, y3_pred_confidence, y3_pred_classes = self.GetBoxes(
            y3, self.anchors_wh[2]/self.image_size)

        y1_pred_mask = tf.math.logical_and(y1_pred_confidence > scores_thresh, 
                                           tf.expand_dims(tf.math.reduce_max(y1_pred_classes, axis=-1), axis=-1) > scores_thresh)
        # y1_pred_mask = y1_pred_confidence > scores_thresh
        y1_pred_boxes = tf.boolean_mask(y1_pred_boxes, y1_pred_mask[..., 0])
        y1_pred_classes = tf.boolean_mask(y1_pred_classes, y1_pred_mask[..., 0])
        y1_pred_confidence = tf.boolean_mask(y1_pred_confidence, y1_pred_mask[..., 0])
        y1_pred_boxes = tf.reshape(
            y1_pred_boxes, [-1, tf.shape(y1_pred_boxes)[-1]])
        # scores
        y1_pred_scores = tf.expand_dims(tf.math.reduce_max(y1_pred_classes, axis=-1), axis=-1)
        # y1_pred_scores = (y1_pred_confidence + y1_pred_scores) / 2
        y1_pred_scores = y1_pred_scores
        y1_pred_scores = tf.reshape(y1_pred_scores, [-1, ])
        # classes
        y1_pred_classes_id = tf.math.argmax(y1_pred_classes, axis=-1)
        y1_pred_classes_id = tf.reshape(y1_pred_classes_id, [-1, ])

        y2_pred_mask = tf.math.logical_and(y2_pred_confidence > scores_thresh, 
                                           tf.expand_dims(tf.math.reduce_max(y2_pred_classes, axis=-1), axis=-1) > scores_thresh)
        # y2_pred_mask = y2_pred_confidence > scores_thresh
        y2_pred_boxes = tf.boolean_mask(y2_pred_boxes, y2_pred_mask[..., 0])
        y2_pred_classes = tf.boolean_mask(y2_pred_classes, y2_pred_mask[..., 0])
        y2_pred_confidence = tf.boolean_mask(y2_pred_confidence, y2_pred_mask[..., 0])
        y2_pred_boxes = tf.reshape(
            y2_pred_boxes, [-1, tf.shape(y2_pred_boxes)[-1]])
        # scores
        y2_pred_scores = tf.expand_dims(tf.math.reduce_max(y2_pred_classes, axis=-1), axis=-1)
        # y2_pred_scores = (y2_pred_confidence + y2_pred_scores) / 2
        y2_pred_scores = y2_pred_scores
        y2_pred_scores = tf.reshape(y2_pred_scores, [-1, ])
        # classes
        y2_pred_classes_id = tf.math.argmax(y2_pred_classes, axis=-1)
        y2_pred_classes_id = tf.reshape(y2_pred_classes_id, [-1, ])

        y3_pred_mask = tf.math.logical_and(y3_pred_confidence > scores_thresh, 
                                           tf.expand_dims(tf.math.reduce_max(y3_pred_classes, axis=-1), axis=-1) > scores_thresh)
        # y3_pred_mask = y3_pred_confidence > scores_thresh
        y3_pred_boxes = tf.boolean_mask(y3_pred_boxes, y3_pred_mask[..., 0])
        y3_pred_classes = tf.boolean_mask(y3_pred_classes, y3_pred_mask[..., 0])
        y3_pred_confidence = tf.boolean_mask(y3_pred_confidence, y3_pred_mask[..., 0])
        y3_pred_boxes = tf.reshape(
            y3_pred_boxes, [-1, tf.shape(y3_pred_boxes)[-1]])
        # scores
        y3_pred_scores = tf.expand_dims(tf.math.reduce_max(y3_pred_classes, axis=-1), axis=-1)
        # y3_pred_scores = (y3_pred_confidence + y3_pred_scores) / 2
        y3_pred_scores = y3_pred_scores
        y3_pred_scores = tf.reshape(y3_pred_scores, [-1, ])
        # classes
        y3_pred_classes_id = tf.math.argmax(y3_pred_classes, axis=-1)
        y3_pred_classes_id = tf.reshape(y3_pred_classes_id, [-1, ])

        y_pred_boxes = tf.concat(
            [y1_pred_boxes, y2_pred_boxes, y3_pred_boxes], axis=0)
        y_pred_classes_id = tf.concat(
            [y1_pred_classes_id, y2_pred_classes_id, y3_pred_classes_id], axis=0)
        y_pred_scores = tf.concat(
            [y1_pred_scores, y2_pred_scores, y3_pred_scores], axis=0)
        y_pred_classes = tf.concat(
            [y1_pred_classes, y2_pred_classes, y3_pred_classes], axis=0)
        y_pred_confidence = tf.concat(
            [y1_pred_confidence, y2_pred_confidence, y3_pred_confidence], axis=0)

        selected_indices = self.GetDIOUNMS(
            y_pred_boxes, y_pred_scores, 500, iou_threshold=iou_thresh)
        selected_boxes = tf.gather(y_pred_boxes, selected_indices)
        selected_classes_id = tf.gather(y_pred_classes_id, selected_indices)
        selected_scores = tf.gather(y_pred_scores, selected_indices)
        selected_classes = tf.gather(y_pred_classes, selected_indices)
        selected_confidence = tf.gather(y_pred_confidence, selected_indices)

        return selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence

    @tf.function
    def GetIOU(self, b1, b2):
        '''
        计算IOU,DIOU,CIOU
        b1:(1, b1_num, (x1, y1, x2, y2))
        b2:(..., b2_num, 1, (x1, y1, x2, y2))
        return:(..., b2_num, b1_num)
        b1与b2前面维度一样或缺少也可以，返回维度与最多的一样
        '''
        # (..., b2_num, b1_num, 2)
        intersect_mins = tf.math.maximum(b1[..., 0:2], b2[..., 0:2])
        intersect_maxes = tf.math.minimum(b1[..., 2:4], b2[..., 2:4])
        intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.)
        # (..., b2_num, b1_num)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # (1, b1_num, 2)
        b1_wh = b1[..., 2:4] - b1[..., 0:2]
        # (..., b2_num, 1, 2)
        b2_wh = b2[..., 2:4] - b2[..., 0:2]
        # (1, b1_num)
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        # (..., b2_num, 1)
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        # (h, w, anchors_num, boxes_num)
        iou = intersect_area / (b1_area + b2_area - intersect_area)
        # 最小外界矩形
        ub_mins = tf.math.minimum(b1[..., 0:2], b2[..., 0:2])
        ub_maxes = tf.math.maximum(b1[..., 2:4], b2[..., 2:4])
        ub_wh = ub_maxes - ub_mins
        c = tf.math.square(ub_wh[..., 0]) + tf.math.square(ub_wh[..., 1])
        b1_xy = (b1[..., 2:4] + b1[..., 0:2]) / 2
        b2_xy = (b2[..., 2:4] + b2[..., 0:2]) / 2
        u = tf.math.reduce_sum(tf.math.square(b1_xy - b2_xy), axis=-1)
        d = u / c
        v = 4 / tf.math.square(math.pi) \
            * tf.math.square(tf.math.atan(b1_wh[..., 0] / b1_wh[..., 1])
            - tf.math.atan(b2_wh[..., 0] / b2_wh[..., 1]))
        alpha = v / (1 - iou + v + 0.000001)
        diou = iou - d
        ciou = diou - alpha * v
        return iou, diou, ciou

    @tf.function
    def GetTarget(self, labels):
        '''
        获取训练Target
        labels:(boxes_num, 6=(batch_index, x1, y1, x2, y2, class_index))
        '''
        image_size = tf.constant(self.image_size, dtype=tf.float32)
        anchors_wh = tf.constant(self.anchors_wh, dtype=tf.float32)
        anchors_num = tf.shape(anchors_wh)[1]
        # (9, 2)
        anchors_wh = tf.reshape(anchors_wh, (-1, 2))
        anchors_wh = anchors_wh / image_size
        layers_size = tf.constant(self.layers_size, dtype=tf.int32)
        layers_num = tf.shape(layers_size)[0]
        classes_num = tf.constant(self.classes_num, dtype=tf.int32)
        # 计算iou
        # (9, 2)
        anchors_mins = tf.zeros_like(anchors_wh)
        anchors_maxes = anchors_wh
        # (9, 4)
        anchors_boxes = tf.concat([anchors_mins, anchors_maxes], axis=-1)
        # (9, 4) => (1, 9, 4)
        anchors_boxes = tf.expand_dims(anchors_boxes, axis=0)
        # (boxes_num, 2)
        boxes_wh = (labels[..., 3:5] - labels[..., 1:3]) / image_size
        boxes_mins = tf.zeros_like(boxes_wh)
        boxes_maxes = boxes_wh
        # (boxes_num, 4)
        boxes = tf.concat([boxes_mins, boxes_maxes], axis=-1)
        # (boxes_num, 4) => (boxes_num, 1, 4)
        boxes = tf.expand_dims(boxes, axis=-2)
        # (boxes_num, 9)
        _, iou, _ = self.GetIOU(boxes, anchors_boxes)
        # (boxes_num, )
        anchors_idx = tf.cast(tf.argmax(iou, axis=-1), tf.int32)
        # tf.print('anchors_idx:', anchors_idx)
        # tf.print('anchors_idx:', anchors_idx.shape)
        # 更新值对应输出的下标
        target_indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True)
        # 更新值对应输出的下标的值
        target_updates = tf.TensorArray(tf.float32, 0, dynamic_size=True)
        batch_size = tf.cast(tf.math.reduce_max(labels[:,0]) + 1, dtype=tf.int32)
        idx = 0
        # 遍历batch
        def batch_foreach(batch_index, idx, target_indexes, target_updates):
            # 遍历boxes
            batch_mask = tf.equal(labels[:, 0], tf.cast(batch_index, dtype=tf.float32))
            # tf.print('batch_mask:', batch_mask)
            batch_labels = tf.boolean_mask(labels, batch_mask)
            batch_anchors_idx = tf.boolean_mask(anchors_idx, batch_mask)
            boxes_size = tf.shape(batch_labels)[0]
            def boxes_foreach(boxes_index, idx, target_indexes, target_updates):
                # 最优层下标
                layer_index = batch_anchors_idx[boxes_index] // layers_num
                # tf.print('layer_index:', layer_index, layer_index.dtype)
                # tf.print('batch_index:', batch_index, batch_index.dtype)
                # 最优候选框下标
                anchor_index = batch_anchors_idx[boxes_index] % layers_num
                # tf.print('anchor_index:', anchor_index, anchor_index.dtype)
                # 计算中心坐标，并换算成层坐标
                boxes_xy = (batch_labels[boxes_index, 3:5] + batch_labels[boxes_index, 1:3]) / 2
                boxes_xy = boxes_xy / image_size
                # tf.print('boxes_xy:', boxes_xy)
                boxes_wh = batch_labels[boxes_index, 3:5] - batch_labels[boxes_index, 1:3]
                boxes_wh = boxes_wh / image_size
                # tf.print('boxes_wh:', boxes_wh)
                layer_xy = tf.cast(tf.math.floor(boxes_xy * tf.cast(layers_size[layer_index], dtype=tf.float32)), dtype=tf.int32)
                # tf.print('layer_xy:', layer_xy, layer_index.dtype)
                # xy下标是反的，因为输入是高宽
                target_indexes = target_indexes.write(idx, [layer_index, batch_index, layer_xy[1], layer_xy[0], anchor_index])
                # 传入的是原始坐标数据
                target_update = tf.concat([boxes_xy, boxes_wh, [1], tf.one_hot(tf.cast(batch_labels[boxes_index, 5], dtype=tf.int32), classes_num)], axis=-1)
                # tf.print('target_update:', target_update, target_update.dtype)
                target_updates = target_updates.write(idx, target_update)
                return boxes_index+1, idx+1, target_indexes, target_updates
            _, idx, target_indexes, target_updates = tf.while_loop(lambda x, y ,z ,a: x<boxes_size, boxes_foreach, [0, idx, target_indexes, target_updates])
            return batch_index+1, idx+1, target_indexes, target_updates
        _, idx, target_indexes, target_updates = tf.while_loop(lambda x, y ,z ,a: x<batch_size, batch_foreach, [0, idx, target_indexes, target_updates])

        target_indexes = target_indexes.stack()
        target_updates = target_updates.stack()
        # 创建0张量，并根据索引赋值
        target_mask = tf.equal(target_indexes[:, 0], 0)
        target1 = tf.scatter_nd(tf.boolean_mask(target_indexes, target_mask)[:,1:], tf.boolean_mask(target_updates, target_mask), (batch_size, layers_size[0, 0], layers_size[0, 1], anchors_num, 5+classes_num))
        target_mask = tf.equal(target_indexes[:, 0], 1)
        target2 = tf.scatter_nd(tf.boolean_mask(target_indexes, target_mask)[:,1:], tf.boolean_mask(target_updates, target_mask), (batch_size, layers_size[1, 0], layers_size[1, 1], anchors_num, 5+classes_num))
        target_mask = tf.equal(target_indexes[:, 0], 2)
        target3 = tf.scatter_nd(tf.boolean_mask(target_indexes, target_mask)[:,1:], tf.boolean_mask(target_updates, target_mask), (batch_size, layers_size[2, 0], layers_size[2, 1], anchors_num, 5+classes_num))
        
        # 去掉因压缩导致的目标重叠
        target1_mask = tf.math.less_equal(target1[...,4:5], 1)
        target1 = target1 * tf.cast(target1_mask, dtype=tf.float32)
        target2_mask = tf.math.less_equal(target2[...,4:5], 1)
        target2 = target2 * tf.cast(target2_mask, dtype=tf.float32)
        target3_mask = tf.math.less_equal(target3[...,4:5], 1)
        target3 = target3 * tf.cast(target3_mask, dtype=tf.float32)
        return (target1, target2, target3)

    @tf.function
    def GetLoss(self, y_true, y_pred):
        '''
        获取损失值
        y_true:坐标还没归一化，[(batch_size, 13, 13, 3, 5+num_classes), (batch_size, 26, 26, 3, 5+num_classes), (batch_size, 52, 52, 3, 5+num_classes)]
        y_pred:[(batch_size, 13, 13, 3, 5+num_classes), (batch_size, 26, 26, 3, 5+num_classes), (batch_size, 52, 52, 3, 5+num_classes)]
        '''
        print('loss_fun:', type(y_true), type(y_pred))
        image_size = tf.constant(self.image_size, dtype=tf.float32)
        # (layers_num, anchors_num, 2)
        anchors_wh = tf.constant(self.anchors_wh, dtype=tf.float32)
        anchors_wh = anchors_wh / image_size
        anchors_num = tf.shape(anchors_wh)[1]
        layers_size = tf.constant(self.layers_size, dtype=tf.int32)
        layers_num = tf.shape(layers_size)[0]
        classes_num = tf.constant(self.classes_num, dtype=tf.int32)
        batch_size = tf.cast(tf.shape(y_true[0])[0], dtype=tf.float32)
        loss = 0.0
        layer_index = 0
        for layer_index in range(3):
            y_true_read = y_true[layer_index]
            y_pred_raw = y_pred[layer_index]
            y_pred_raw = tf.reshape(y_pred_raw, tf.shape(y_true_read))
            # 特征网格对应实际图片的坐标
            grid_shape = tf.shape(y_pred_raw)[1:3] # height, width
            grid_x = tf.range(0, tf.cast(grid_shape[1], dtype=tf.float32), dtype=tf.float32)
            grid_y = tf.range(0, tf.cast(grid_shape[0], dtype=tf.float32), dtype=tf.float32)
            grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
            grid_x = tf.reshape(grid_x, (grid_shape[0], grid_shape[1], 1, 1))
            grid_y = tf.reshape(grid_y, (grid_shape[0], grid_shape[1], 1, 1))
            grid_xy = tf.concat([grid_x, grid_y], axis=-1)
            # 计算真实坐标与相对坐标
            # y_true
            y_true_object = y_true_read[..., 4:5]
            y_true_classes = y_true_read[..., 5:]
            y_true_read_xy = y_true_read[..., 0:2]
            # tf.print('grid_xy:', tf.math.reduce_max(grid_xy), tf.math.reduce_min(grid_xy))
            # tf.print('grid_shape:', grid_shape[::-1])
            y_true_raw_xy = y_true_read_xy * tf.cast(grid_shape[::-1], dtype=tf.float32) - grid_xy
            # tf.print('y_true_raw_xy:', tf.math.reduce_max(y_true_raw_xy), tf.math.reduce_min(y_true_raw_xy))
            # tf.print('y_true_object:', tf.math.reduce_max(y_true_object), tf.math.reduce_min(y_true_object))
            y_true_raw_xy = y_true_object * y_true_raw_xy
            # tf.print('y_true_raw_xy:', tf.math.reduce_max(y_true_raw_xy), tf.math.reduce_min(y_true_raw_xy))
            
            y_true_read_wh = y_true_read[..., 2:4]
            y_true_raw_wh = tf.math.log(y_true_read_wh / anchors_wh[layer_index, ...])
            y_true_raw_wh = tf.where(tf.math.is_inf(y_true_raw_wh), tf.zeros_like(y_true_raw_wh), y_true_raw_wh)
            # tf.print('y_true_raw_wh:', tf.math.reduce_max(y_true_raw_wh), tf.math.reduce_min(y_true_raw_wh))
            
            # y_pred
            y_pred_object = y_pred_raw[..., 4:5]
            y_pred_classes = y_pred_raw[..., 5:]
            y_pred_raw_xy = y_pred_raw[..., 0:2]
            # tf.print('y_pred_raw_xy:', tf.math.reduce_max(y_pred_raw_xy), tf.math.reduce_min(y_pred_raw_xy))
            y_pred_read_xy = (tf.math.sigmoid(y_pred_raw_xy) + grid_xy) / tf.cast(grid_shape[::-1], dtype=tf.float32)
            
            y_pred_raw_wh = y_pred_raw[..., 2:4]
            # tf.print('y_pred_raw_wh:', tf.math.reduce_max(y_pred_raw_wh), tf.math.reduce_min(y_pred_raw_wh))
            y_pred_read_wh = tf.math.exp(y_pred_raw_wh) * anchors_wh[layer_index, ...]
            y_pred_read_wh = tf.where(tf.math.is_inf(y_pred_read_wh), tf.zeros_like(y_pred_read_wh), y_pred_read_wh)
            
            # 框坐标(batch_size, h, w, anchors_num, (x1, y1, x2, y2))
            y_true_read_wh_half = y_true_read_wh / 2
            y_true_read_mins = y_true_read_xy - y_true_read_wh_half
            y_true_read_maxes = y_true_read_xy + y_true_read_wh_half
            y_true_boxes = tf.concat([y_true_read_mins, y_true_read_maxes], axis=-1)
            y_pred_read_wh_half = y_pred_read_wh / 2
            y_pred_read_mins = y_pred_read_xy - y_pred_read_wh_half
            y_pred_read_maxes = y_pred_read_xy + y_pred_read_wh_half
            y_pred_boxes = tf.concat([y_pred_read_mins, y_pred_read_maxes], axis=-1)
            
            ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            def foreach_batch(batch_index, ignore_mask):
                y_true_boxes_one = y_true_boxes[batch_index, ...]
                y_pred_boxes_one = y_pred_boxes[batch_index, ...]
                y_true_object_one = y_true_object[batch_index, ...]
                y_true_boxes_tmp = tf.boolean_mask(y_true_boxes_one, y_true_object_one[..., 0])
                # 计算IOU
                # (boxes_num, 4) => (1, boxes_num, 4)
                y_true_boxes_tmp = tf.expand_dims(y_true_boxes_tmp, axis=0)
                y_pred_boxes_tmp = y_pred_boxes_one
                # (h, w, anchors_num, 4) => (h, w, anchors_num, 1, 4)
                y_pred_boxes_tmp = tf.expand_dims(y_pred_boxes_tmp, axis=-2)
                # (h, w, anchors_num, boxes_num)
                _, _, iou = self.GetIOU(y_true_boxes_tmp, y_pred_boxes_tmp)
                # (h, w, anchors_num)
                best_iou = tf.math.reduce_max(iou, axis=-1)
                # 把IOU<0.5的认为是背景
                ignore_mask = ignore_mask.write(batch_index, tf.cast(best_iou < self.train_iou_thresh, dtype=tf.float32))
                return batch_index + 1, ignore_mask
            # (batch_size, h, w, anchors_num, y_true_boxes_num)
            _, ignore_mask = tf.while_loop(lambda b,*args: b<tf.cast(batch_size, dtype=tf.int32), foreach_batch, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            # (batch_size, h, w, anchors_num)
            ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
            # ignore_mask = tf.where(tf.math.is_nan(ignore_mask), tf.zeros_like(ignore_mask), ignore_mask)
            # tf.print('ignore_mask:', tf.math.reduce_max(ignore_mask), tf.math.reduce_min(ignore_mask))
            # 计算loss
            boxes_loss_scale = 2 - y_true_read_wh[..., 0:1] * y_true_read_wh[..., 1:2]
            # tf.print('boxes_loss_scale:', tf.math.reduce_max(boxes_loss_scale), tf.math.reduce_min(boxes_loss_scale))
            xy_loss = y_true_object * boxes_loss_scale * tf.math.square(y_true_raw_xy - y_pred_raw_xy)
            wh_loss = y_true_object * boxes_loss_scale * 0.5 * tf.math.square(y_true_raw_wh - y_pred_raw_wh)
            object_loss_bc = tf.keras.losses.binary_crossentropy(tf.expand_dims(y_true_object, axis=-1),
                             tf.expand_dims(y_pred_object, axis=-1), from_logits=True)
            # tf.print('object_loss_bc:', tf.math.reduce_max(object_loss_bc), tf.math.reduce_min(object_loss_bc))
            object_loss = y_true_object * object_loss_bc + (1 - y_true_object) * object_loss_bc * ignore_mask
            classes_loss_bc = tf.keras.losses.binary_crossentropy(tf.expand_dims(y_true_classes, axis=-1),
                              tf.expand_dims(y_pred_classes, axis=-1), from_logits=True)
            # tf.print('classes_loss_bc:', tf.math.reduce_max(classes_loss_bc), tf.math.reduce_min(classes_loss_bc))
            classes_loss = y_true_object * classes_loss_bc

            xy_loss = tf.math.reduce_sum(xy_loss) / batch_size
            wh_loss = tf.math.reduce_sum(wh_loss) / batch_size
            object_loss = tf.math.reduce_sum(object_loss) / batch_size
            classes_loss = tf.math.reduce_sum(classes_loss) / batch_size
            # tf.print('loss:', xy_loss, wh_loss, object_loss, classes_loss)
            loss += xy_loss + wh_loss + object_loss + classes_loss
        # tf.print('loss:', loss)
        return loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 6), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    ])
    def TrainStep(self, input_image, target, learning_rate):
        '''
        单步训练
        input_image:图片(416,416,3)
        target_data:(box_size, (batch_index, x1, y1, x2, y2, class_id))
        '''
        print('Tracing with TrainStep', type(input_image),
              type(target))
        print('Tracing with TrainStep', input_image.shape, target.shape)
        tmp_loss = 0.0
        target = self.GetTarget(target)
        with tf.GradientTape() as tape:
            # 预测
            output = self.feature_model(input_image)
            # print('output:', output)
            # 计算损失
            tmp_loss = self.GetLoss(
                y_true=target, y_pred=output)
        trainable_variables = self.feature_model.trainable_variables
        gradients = tape.gradient(tmp_loss, trainable_variables)
        self.optimizer.learning_rate = learning_rate
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return tmp_loss

    def FitGenerator(self, generator, steps_per_epoch, epochs, initial_epoch=1, auto_save=False, learning_rate=1e-4):
        '''批量训练'''
        min_loss = None
        min_loss_num = 0
        lr = learning_rate
        for epoch in range(initial_epoch, epochs+1):
            start = time.process_time()
            epoch_loss = 0
            for steps in range(1, steps_per_epoch+1):
                x, y = next(generator)
                # print('generator', x.shape, y.shape)
                tmp_g_loss = self.TrainStep(x, y, lr)
                epoch_loss += tmp_g_loss
                print('\rsteps:%d/%d, epochs:%d/%d, loss:%0.4f, epoch_loss_avg:%0.4f'
                      % (steps, steps_per_epoch, epoch, epochs, tmp_g_loss, epoch_loss / steps), end='', flush=True)
                if tf.math.is_nan(tmp_g_loss):
                    return
            # 求平均
            # epoch_loss = epoch_loss / steps_per_epoch
            end = time.process_time()
            print('\rsteps:%d/%d, epochs:%d/%d, %0.4f S, loss:%0.4f, epoch_loss:%0.4f, epoch_loss_avg:%0.4f, lr:%e'
                  % (steps, steps_per_epoch, epoch, epochs, (end - start), tmp_g_loss, epoch_loss, epoch_loss / steps_per_epoch, lr), flush=True)
            if min_loss is None or min_loss > epoch_loss:
                min_loss = epoch_loss
                min_loss_num = 0
            else:
                min_loss_num += 1
            if min_loss_num > 10:
                lr *= 0.6
                min_loss_num = 0
                min_loss = None
                print('lr:', lr)
                if lr < 1e-7:
                    return
            if auto_save:
                self.SaveModel()

    @tf.function
    def Predict(self, input_image, scores_thresh=0.5, iou_thresh=0.5):
        '''
        预测(编译模式)
        input_image:图片(416,416,3)
        return:两个指针值(2)
        '''
        # 预测
        start = time.process_time()
        output = self.feature_model(input_image)
        # tf.print('output[0]:', tf.math.reduce_max(output[0]), tf.math.reduce_min(output[0]))
        # tf.print('output[1]:', tf.math.reduce_max(output[1]), tf.math.reduce_min(output[1]))
        # tf.print('output[2]:', tf.math.reduce_max(output[2]), tf.math.reduce_min(output[2]))
        selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence = self.GetNMSBoxes(
            output[0], output[1], output[2], scores_thresh, iou_thresh)
        end = time.process_time()
        tf.print('%s predict time: %f' % (self.__class__, (end - start)))
        return selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence

    def SaveModel(self):
        '''保存模型'''
        save_path = self.checkpoint_manager.save()
        print('保存模型 {}'.format(save_path))

    def LoadModel(self):
        '''加载模型'''
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print('加载模型 {}'.format(self.checkpoint_manager.latest_checkpoint))


def main():
    pass


if __name__ == '__main__':
    main()
