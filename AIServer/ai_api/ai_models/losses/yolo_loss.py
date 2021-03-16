import tensorflow as tf


class Yolov4Loss(tf.keras.losses.Loss):
    def __init__(self, anchors, classes_num, ignore_thresh=.5, print_loss=False, **args):
        super(Yolov4Loss, self).__init__(**args)
        self.anchors = anchors
        self.classes_num = classes_num
        self.ignore_thresh = ignore_thresh
        self.print_loss = print_loss

    # @tf.function
    def BoxIou(self, b1, b2):
        '''Return iou tensor

        Parameters
        ----------
        b1: tensor, shape=(i1,...,iN, 4), xywh
        b2: tensor, shape=(j, 4), xywh

        Returns
        -------
        iou: tensor, shape=(i1,...,iN, j)

        '''

        # Expand dim to apply broadcasting.
        b1 = tf.expand_dims(b1, axis=-2)
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh/2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        # Expand dim to apply broadcasting.
        b2 = tf.expand_dims(b2, axis=0)
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh/2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = tf.keras.backend.maximum(b1_mins, b2_mins)
        intersect_maxes = tf.keras.backend.minimum(b1_maxes, b2_maxes)
        intersect_wh = tf.keras.backend.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        iou = intersect_area / (b1_area + b2_area - intersect_area)

        return iou

    # @tf.function
    def YoloHead(self, feats, anchors, classes_num, input_shape, calc_loss=False):
        """Convert final layer features to bounding box parameters."""
        # 3
        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = tf.keras.backend.reshape(tf.keras.backend.constant(anchors), [1, 1, 1, num_anchors, 2])
        # 当前层特征高宽
        grid_shape = tf.keras.backend.shape(feats)[1:3] # height, width
        grid_y = tf.keras.backend.tile(tf.keras.backend.reshape(tf.keras.backend.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
            [1, grid_shape[1], 1, 1])
        grid_x = tf.keras.backend.tile(tf.keras.backend.reshape(tf.keras.backend.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
            [grid_shape[0], 1, 1, 1])
        # (height, width, 1, 2)
        grid = tf.keras.backend.concatenate([grid_x, grid_y])
        grid = tf.keras.backend.cast(grid, tf.keras.backend.dtype(feats))
        # 拆分anchors
        feats = tf.keras.backend.reshape(
            feats, [-1, grid_shape[0], grid_shape[1], num_anchors, self.classes_num + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        # tf.print('grid_shape:', grid_shape)
        box_xy = (tf.keras.backend.sigmoid(feats[..., :2]) + grid) / tf.keras.backend.cast(grid_shape[::-1], tf.keras.backend.dtype(feats))
        box_wh = tf.keras.backend.exp(feats[..., 2:4]) * anchors_tensor / tf.keras.backend.cast(input_shape[::-1], tf.keras.backend.dtype(feats))
        box_confidence = tf.keras.backend.sigmoid(feats[..., 4:5])
        box_class_probs = tf.keras.backend.sigmoid(feats[..., 5:])

        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    # @tf.function
    def call(self, y_true, y_pred):
        '''
        获取损失值
        y_true:坐标还没归一化，[(batch_size, 13, 13, 3, 5+classes_num), (batch_size, 26, 26, 3, 5+classes_num), (batch_size, 52, 52, 3, 5+classes_num)]
        y_pred:[(batch_size, 13, 13, 3, 5+classes_num), (batch_size, 26, 26, 3, 5+classes_num), (batch_size, 52, 52, 3, 5+classes_num)]
        '''
        num_layers = len(self.anchors)//3 # default setting
        # 3层输出，[(batch_size, 13, 13, 3*(5+classes_num)), (batch_size, 26, 26, 3*(5+classes_num)), (batch_size, 52, 52, 3*(5+classes_num))]
        yolo_outputs = y_pred
        # 3层正确值，[(batch_size, 13, 13, 3, 5+classes_num), (batch_size, 26, 26, 3, 5+classes_num), (batch_size, 52, 52, 3, 5+classes_num)]
        y_true = y_true
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        # (416, 416)
        input_shape = tf.keras.backend.cast(tf.keras.backend.shape(yolo_outputs[0])[1:3] * 32, tf.keras.backend.dtype(y_true[0]))
        # [(13, 13), (26, 26), (52, 52)]
        grid_shapes = [tf.keras.backend.cast(tf.keras.backend.shape(yolo_outputs[l])[1:3], tf.keras.backend.dtype(y_true[0])) for l in range(num_layers)]
        loss = 0
        m = tf.keras.backend.shape(yolo_outputs[0])[0] # batch size, tensor
        mf = tf.keras.backend.cast(m, tf.keras.backend.dtype(yolo_outputs[0]))

        # 遍历3层
        for l in range(num_layers):
            # 存在目标
            object_mask = y_true[l][..., 4:5]
            # 分类
            true_class_probs = y_true[l][..., 5:]

            # 特征坐标矩阵，拆分anchors后的原特征，真实框信息(归一化后)
            grid, raw_pred, pred_xy, pred_wh = self.YoloHead(yolo_outputs[l],
                self.anchors[anchor_mask[l]], self.classes_num, input_shape, calc_loss=True)
            pred_box = tf.keras.backend.concatenate([pred_xy, pred_wh], axis=-1)

            # Darknet raw box to calculate loss.
            # 真实像素坐标与宽高
            # 目标框坐标与中心点差值，像素级
            raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
            # 目标框与候选框比例
            raw_true_wh = tf.keras.backend.log(y_true[l][..., 2:4] * input_shape[::-1] / self.anchors[anchor_mask[l]])
            raw_true_wh = tf.keras.backend.switch(object_mask, raw_true_wh, tf.keras.backend.zeros_like(raw_true_wh)) # avoid log(0)=-inf
            # 
            box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

            # Find ignore mask, iterate over each of batch.
            # 把iou<0.5，认为是背景
            ignore_mask = tf.TensorArray(tf.keras.backend.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = tf.keras.backend.cast(object_mask, 'bool')
            # 循环体
            def loop_body(b, ignore_mask):
                # (13, 13, 3, 4) => (?, 4)
                true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
                # pred_box[b]:(13, 13, 3, 4), true_box:(?, 4)
                iou = self.BoxIou(pred_box[b], true_box)
                best_iou = tf.keras.backend.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, tf.keras.backend.cast(best_iou<self.ignore_thresh, tf.keras.backend.dtype(true_box)))
                return b+1, ignore_mask
            _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.keras.backend.expand_dims(ignore_mask, -1)

            # tf.keras.losses.binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = object_mask * box_loss_scale * tf.keras.backend.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.math.square(raw_true_wh-raw_pred[...,2:4])
            confidence_loss = object_mask * tf.keras.backend.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
                (1-object_mask) * tf.keras.backend.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
            class_loss = object_mask * tf.keras.backend.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

            xy_loss = tf.keras.backend.sum(xy_loss) / mf
            wh_loss = tf.keras.backend.sum(wh_loss) / mf
            confidence_loss = tf.keras.backend.sum(confidence_loss) / mf
            class_loss = tf.keras.backend.sum(class_loss) / mf
            loss += xy_loss + wh_loss + confidence_loss + class_loss
            # tf.print('loss:', loss, xy_loss, wh_loss, confidence_loss, class_loss)
            if self.print_loss:
                loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, tf.keras.backend.sum(ignore_mask)], message='loss: ')
        return loss
