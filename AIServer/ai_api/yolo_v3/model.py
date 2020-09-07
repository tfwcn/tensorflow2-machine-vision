import tensorflow as tf
from ai_api.yolo_v3.loss import Yolov4Loss
import math
from ai_api.yolo_v3.mAP import Get_mAP_one
from ai_api.utils.drop_block import DorpBlock

class DarknetConv2D(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        '''初始化网络'''
        super(DarknetConv2D, self).__init__()
        darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(5e-4)}
        darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
        darknet_conv_kwargs.update(kwargs)
        # print('darknet_conv_kwargs:', darknet_conv_kwargs)
        self.conv1 = tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)

    @tf.function
    def call(self, x, training):
        '''运算部分'''
        x = self.conv1(x)
        return x


class DarknetConv2D_BN_Leaky(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        '''初始化网络'''
        super(DarknetConv2D_BN_Leaky, self).__init__()
        no_bias_kwargs = {'use_bias': False}
        no_bias_kwargs.update(kwargs)
        self.conv1 = DarknetConv2D(*args, **no_bias_kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        # self.dorp_block1 = DorpBlock(0.1, block_size=3)

    @tf.function
    def call(self, x, training):
        '''运算部分'''
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.leaky_relu1(x)
        # x = self.dorp_block1(x, training=training)
        return x


class ResblockBody(tf.keras.Model):

    def __init__(self, num_filters, num_blocks, **args):
        '''初始化网络'''
        super(ResblockBody, self).__init__(**args)
        # 参数
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        
        # 层定义
        self.zero_padding1 = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))
        self.darknet_conv_bn_leaky1 = DarknetConv2D_BN_Leaky(self.num_filters, (3,3), strides=(2,2))
        self.blocks = []
        for i in range(self.num_blocks):
            self.blocks.append([
                DarknetConv2D_BN_Leaky(self.num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(self.num_filters, (3,3)),
                tf.keras.layers.Add()
                ])

    @tf.function
    def call(self, x, training):
        '''运算部分'''
        x = self.zero_padding1(x)
        x = self.darknet_conv_bn_leaky1(x, training=training)
        for block in self.blocks:
            y = block[0](x, training=training)
            y = block[1](y, training=training)
            x = block[2]([x, y])
        return x
        
    def get_config(self):
        '''获取配置，用于保存模型'''
        return {'num_filters': self.num_filters, 'num_blocks': self.num_blocks}


class DarknetBody(tf.keras.Model):

    def __init__(self, **args):
        '''初始化网络'''
        super(DarknetBody, self).__init__(**args)
        
        self.darknet_conv_bn_leaky1 = DarknetConv2D_BN_Leaky(32, (3,3))
        self.resblock_body1 = ResblockBody(64, 1)
        self.resblock_body2 = ResblockBody(128, 2)
        self.resblock_body3 = ResblockBody(256, 8)
        self.resblock_body4 = ResblockBody(512, 8)
        self.resblock_body5 = ResblockBody(1024, 4)

    @tf.function
    def call(self, x, training):
        '''运算部分'''
        x = self.darknet_conv_bn_leaky1(x, training=training)
        x = self.resblock_body1(x, training=training)
        x = self.resblock_body2(x, training=training)
        x = self.resblock_body3(x, training=training)
        y3 = x
        x = self.resblock_body4(x, training=training)
        y2 = x
        x = self.resblock_body5(x, training=training)
        y1 = x
        return y1, y2, y3


class LastLayers(tf.keras.Model):

    def __init__(self, num_filters, out_filters, **args):
        '''初始化网络'''
        super(LastLayers, self).__init__(**args)
        # 参数
        self.num_filters = num_filters
        self.out_filters = out_filters
        
        # 层定义
        self.darknet_conv_bn_leaky1 = DarknetConv2D_BN_Leaky(self.num_filters, (1,1))
        self.darknet_conv_bn_leaky2 = DarknetConv2D_BN_Leaky(self.num_filters*2, (3,3))
        self.darknet_conv_bn_leaky3 = DarknetConv2D_BN_Leaky(self.num_filters, (1,1))
        self.darknet_conv_bn_leaky4 = DarknetConv2D_BN_Leaky(self.num_filters*2, (3,3))
        self.darknet_conv_bn_leaky5 = DarknetConv2D_BN_Leaky(self.num_filters, (1,1))
        self.darknet_conv_bn_leaky6 = DarknetConv2D_BN_Leaky(self.num_filters*2, (3,3))
        self.darknetConv2D1 = DarknetConv2D(self.out_filters, (1,1))

    @tf.function
    def call(self, x, training):
        '''运算部分'''
        x = self.darknet_conv_bn_leaky1(x, training=training)
        x = self.darknet_conv_bn_leaky2(x, training=training)
        x = self.darknet_conv_bn_leaky3(x, training=training)
        x = self.darknet_conv_bn_leaky4(x, training=training)
        x = self.darknet_conv_bn_leaky5(x, training=training)
        y = self.darknet_conv_bn_leaky6(x, training=training)
        y = self.darknetConv2D1(y, training=training)
        return x, y
        
    def get_config(self):
        '''获取配置，用于保存模型'''
        return {'num_filters': self.num_filters, 'out_filters': self.out_filters}


class YoloV3Model(tf.keras.Model):

    def __init__(self, anchors_num, classes_num, anchors, image_size, **args):
        '''初始化网络'''
        super(YoloV3Model, self).__init__(**args)
        # 参数
        self.anchors_num = anchors_num
        self.classes_num = classes_num
        self.anchors = anchors
        self.image_size = image_size
        
        # 层定义
        self.darknet_body1 = DarknetBody()
        self.last_layers1 = LastLayers(512, self.anchors_num*(self.classes_num+5))
        
        self.darknet_conv_bn_leaky1 = DarknetConv2D_BN_Leaky(256, (1,1))
        self.up_sampling1 = tf.keras.layers.UpSampling2D(2)
        self.concatenate1 = tf.keras.layers.Concatenate()
        self.last_layers2 = LastLayers(256, self.anchors_num*(self.classes_num+5))

        self.darknet_conv_bn_leaky2 = DarknetConv2D_BN_Leaky(128, (1,1))
        self.up_sampling2 = tf.keras.layers.UpSampling2D(2)
        self.concatenate2 = tf.keras.layers.Concatenate()
        self.last_layers3 = LastLayers(128, self.anchors_num*(self.classes_num+5))

        self.loss_obj = Yolov4Loss(anchors=self.anchors,classes_num=self.classes_num)

    @tf.function
    def call(self, x, training):
        '''运算部分'''
        y1, y2, y3 = self.darknet_body1(x, training=training)
        
        x, y1 = self.last_layers1(y1, training=training)

        x = self.darknet_conv_bn_leaky1(x, training=training)
        x = self.up_sampling1(x)
        x = self.concatenate1([x, y2])
        x, y2 = self.last_layers2(x, training=training)

        x = self.darknet_conv_bn_leaky2(x, training=training)
        x = self.up_sampling2(x)
        x = self.concatenate2([x, y3])
        x, y3 = self.last_layers3(x, training=training)
        return y1, y2, y3
        
    def get_config(self):
        '''获取配置，用于保存模型'''
        return {'anchors_num': self.anchors_num, 'classes_num': self.classes_num}
    
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
        # tf.print('iou:', tf.math.reduce_max(iou), tf.math.reduce_min(iou))
        # 最小外界矩形
        ub_mins = tf.math.minimum(b1[..., 0:2], b2[..., 0:2])
        ub_maxes = tf.math.maximum(b1[..., 2:4], b2[..., 2:4])
        ub_wh = ub_maxes - ub_mins
        c = tf.math.square(ub_wh[..., 0]) + tf.math.square(ub_wh[..., 1])
        # 计算中心距离
        b1_xy = (b1[..., 2:4] + b1[..., 0:2]) / 2
        b2_xy = (b2[..., 2:4] + b2[..., 0:2]) / 2
        u = tf.math.reduce_sum(tf.math.square(b1_xy - b2_xy), axis=-1)
        # 中心距离越近，d值越小
        d = u / c
        # tf.print('d:', tf.math.reduce_max(d), tf.math.reduce_min(d))
        # 两个框宽高比越接近，v值越小
        v = 4 / tf.math.square(math.pi) * tf.math.square(tf.math.atan(b1_wh[..., 0] / b1_wh[..., 1]) - tf.math.atan(b2_wh[..., 0] / b2_wh[..., 1]))
        # tf.print('v:', tf.math.reduce_max(v), tf.math.reduce_min(v))
        alpha = v / (1 - iou + v)
        # tf.print('alpha:', tf.math.reduce_max(alpha), tf.math.reduce_min(alpha))
        # 目标不相交时，为负值。目标重叠时为0。
        diou = iou - d
        ciou = diou - alpha * v
        # tf.print('iou:', tf.math.reduce_max(iou), tf.math.reduce_min(iou))
        # tf.print('diou:', tf.math.reduce_max(diou), tf.math.reduce_min(diou))
        # tf.print('ciou:', tf.math.reduce_max(ciou), tf.math.reduce_min(ciou))
        return iou, diou, ciou

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
                iou, _, _ = self.GetIOU(boxes_top, boxes_other)
                iou_mask = iou < iou_threshold
                boxes_sort = tf.boolean_mask(boxes_other, iou_mask)
                scores_sort_indexes = tf.boolean_mask(indexes_other, iou_mask)
                result_indexes = result_indexes.write(idx, indexes_top)
                return idx + 1, result_indexes, boxes_sort, scores_sort_indexes
            else:
                result_indexes = result_indexes.write(idx, indexes_top)
                return -1, result_indexes, boxes_sort, scores_sort_indexes
        _, result_indexes, _, _ = tf.while_loop(lambda i, *args: tf.math.not_equal(i, -1), boxes_foreach, [0, result_indexes, boxes_sort, scores_sort_indexes])
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
        
        anchors_wh = self.anchors.reshape((-1, 3, 2))
        y1_pred_boxes, y1_pred_confidence, y1_pred_classes = self.GetBoxes(
            y1, anchors_wh[0]/self.image_size)
        y2_pred_boxes, y2_pred_confidence, y2_pred_classes = self.GetBoxes(
            y2, anchors_wh[1]/self.image_size)
        y3_pred_boxes, y3_pred_confidence, y3_pred_classes = self.GetBoxes(
            y3, anchors_wh[2]/self.image_size)

        y1_pred_mask = tf.math.logical_and(y1_pred_confidence > scores_thresh, 
                                           tf.expand_dims(tf.math.reduce_max(y1_pred_classes, axis=-1), axis=-1) > scores_thresh)
        y1_pred_boxes = tf.boolean_mask(y1_pred_boxes, y1_pred_mask[..., 0])
        y1_pred_classes = tf.boolean_mask(y1_pred_classes, y1_pred_mask[..., 0])
        y1_pred_confidence = tf.boolean_mask(y1_pred_confidence, y1_pred_mask[..., 0])
        y1_pred_boxes = tf.reshape(
            y1_pred_boxes, [-1, tf.shape(y1_pred_boxes)[-1]])
        # scores
        y1_pred_scores = tf.expand_dims(tf.math.reduce_max(y1_pred_classes, axis=-1), axis=-1)
        y1_pred_scores = y1_pred_scores
        y1_pred_scores = tf.reshape(y1_pred_scores, [-1, ])
        # classes
        y1_pred_classes_id = tf.math.argmax(y1_pred_classes, axis=-1)
        y1_pred_classes_id = tf.reshape(y1_pred_classes_id, [-1, ])

        y2_pred_mask = tf.math.logical_and(y2_pred_confidence > scores_thresh, 
                                           tf.expand_dims(tf.math.reduce_max(y2_pred_classes, axis=-1), axis=-1) > scores_thresh)
        y2_pred_boxes = tf.boolean_mask(y2_pred_boxes, y2_pred_mask[..., 0])
        y2_pred_classes = tf.boolean_mask(y2_pred_classes, y2_pred_mask[..., 0])
        y2_pred_confidence = tf.boolean_mask(y2_pred_confidence, y2_pred_mask[..., 0])
        y2_pred_boxes = tf.reshape(
            y2_pred_boxes, [-1, tf.shape(y2_pred_boxes)[-1]])
        # scores
        y2_pred_scores = tf.expand_dims(tf.math.reduce_max(y2_pred_classes, axis=-1), axis=-1)
        y2_pred_scores = y2_pred_scores
        y2_pred_scores = tf.reshape(y2_pred_scores, [-1, ])
        # classes
        y2_pred_classes_id = tf.math.argmax(y2_pred_classes, axis=-1)
        y2_pred_classes_id = tf.reshape(y2_pred_classes_id, [-1, ])

        y3_pred_mask = tf.math.logical_and(y3_pred_confidence > scores_thresh, 
                                           tf.expand_dims(tf.math.reduce_max(y3_pred_classes, axis=-1), axis=-1) > scores_thresh)
        y3_pred_boxes = tf.boolean_mask(y3_pred_boxes, y3_pred_mask[..., 0])
        y3_pred_classes = tf.boolean_mask(y3_pred_classes, y3_pred_mask[..., 0])
        y3_pred_confidence = tf.boolean_mask(y3_pred_confidence, y3_pred_mask[..., 0])
        y3_pred_boxes = tf.reshape(
            y3_pred_boxes, [-1, tf.shape(y3_pred_boxes)[-1]])
        # scores
        y3_pred_scores = tf.expand_dims(tf.math.reduce_max(y3_pred_classes, axis=-1), axis=-1)
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

        # selected_indices = tf.image.non_max_suppression(
        #     y_pred_boxes, y_pred_scores, 500, iou_threshold=iou_thresh)
        selected_indices = self.GetDIOUNMS(
            y_pred_boxes, y_pred_scores, 500, iou_threshold=iou_thresh)
        selected_boxes = tf.gather(y_pred_boxes, selected_indices)
        selected_classes_id = tf.gather(y_pred_classes_id, selected_indices)
        selected_scores = tf.gather(y_pred_scores, selected_indices)
        selected_classes = tf.gather(y_pred_classes, selected_indices)
        selected_confidence = tf.gather(y_pred_confidence, selected_indices)
        # tf.print('y_pred_boxes:', tf.shape(y_pred_boxes))
        # tf.print('selected_boxes:', tf.shape(selected_boxes))
        return selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence

    @tf.function
    def train_step(self, data):
        '''训练'''
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        # print('data:', data)
        x, y = data
        # tf.print('x:', tf.shape(x))
        # tf.print('y:', tf.shape(y[0]), tf.shape(y[1]), tf.shape(y[2]))
        # 维度丢失，需重置维度
        x = tf.reshape(x, (-1,416,416,3))
        y = (tf.reshape(y[0], (-1,13,13,3,(5+self.classes_num))),tf.reshape(y[1], (-1,26,26,3,(5+self.classes_num))),tf.reshape(y[2], (-1,52,52,3,(5+self.classes_num))))

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # print('y_pred:', y_pred)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.loss(y, y_pred, regularization_losses=self.losses)
            loss = self.loss_obj(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        # for var in trainable_vars:
        #     tf.print('trainable_vars:', var.shape)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        # return {m.name: m.result() for m in self.metrics}
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        '''评估'''
        x, y = data
        # 维度丢失，需重置维度
        x = tf.reshape(x, (-1,416,416,3))
        y = (tf.reshape(y[0], (-1,13,13,3,(5+self.classes_num))),tf.reshape(y[1], (-1,26,26,3,(5+self.classes_num))),tf.reshape(y[2], (-1,52,52,3,(5+self.classes_num))))
        y_pred = self(x, training=False)

        selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence = self.GetNMSBoxes(
            y_pred[0], y_pred[1], y_pred[2], scores_thresh=0.2, iou_thresh=0.5)
        # tf.print('selected_boxes:', tf.shape(selected_boxes))
        # tf.print('selected_classes_id:', tf.shape(selected_classes_id))
        # tf.print('selected_scores:', tf.shape(selected_scores))
        # tf.print('selected_classes:', tf.shape(selected_classes))
        # tf.print('selected_confidence:', tf.shape(selected_confidence))
        prediction = tf.concat([selected_boxes,
            tf.cast(tf.expand_dims(selected_classes_id, axis=-1), dtype=tf.float32),
            tf.expand_dims(selected_scores, axis=-1)], axis=-1)
        # tf.print('prediction:', tf.shape(prediction), prediction)
        prediction = prediction
        
        groud_truth1 = self.GetGroudTruth(y[0])
        groud_truth2 = self.GetGroudTruth(y[1])
        groud_truth3 = self.GetGroudTruth(y[2])
        groud_truth = tf.concat([groud_truth1,groud_truth2,groud_truth3], axis=0)
        mAP = tf.numpy_function(Get_mAP_one, (groud_truth, prediction, self.classes_num, 0.5), tf.float64)
        return {'mAP': mAP}
    
    def GetGroudTruth(self, y):
        boxes_xy, boxes_wh, confidence, classes = tf.split(
            y, (2, 2, 1, self.classes_num), axis=-1)
        confidence = confidence[..., 0]
        boxes_wh_half = boxes_wh / 2
        boxes_mins = boxes_xy - boxes_wh_half
        boxes_maxes = boxes_xy + boxes_wh_half
        boxes = tf.concat([boxes_mins, boxes_maxes], axis=-1)
        boxes = tf.boolean_mask(boxes, confidence)
        classes = tf.boolean_mask(classes, confidence)
        groud_truth = tf.concat([boxes, 
            tf.cast(tf.expand_dims(tf.math.argmax(classes, axis=-1), axis=-1), dtype=tf.float32)], axis=-1)
        # tf.print('groud_truth:', tf.shape(groud_truth))
        return groud_truth

    @tf.function
    def Predict(self, input_image, scores_thresh=0.5, iou_thresh=0.5):
        '''
        预测(编译模式)
        input_image:图片(416,416,3)
        return:两个指针值(2)
        '''
        # 预测
        start = time.process_time()
        output = self.feature_model(input_image, training=False)
        tf.print('output[0]:', tf.math.reduce_max(output[0]), tf.math.reduce_min(output[0]), tf.shape(output[0]))
        tf.print('output[1]:', tf.math.reduce_max(output[1]), tf.math.reduce_min(output[1]), tf.shape(output[1]))
        tf.print('output[2]:', tf.math.reduce_max(output[2]), tf.math.reduce_min(output[2]), tf.shape(output[2]))
        selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence = self.GetNMSBoxes(
            output[0], output[1], output[2], scores_thresh, iou_thresh)
        end = time.process_time()
        tf.print('%s predict time: %f' % (self.__class__, (end - start)))
        return selected_boxes, selected_classes_id, selected_scores, selected_classes, selected_confidence

