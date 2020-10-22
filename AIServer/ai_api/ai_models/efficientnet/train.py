import logging
import math
import numpy as np
import tensorflow as tf

import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.efficientnet.efficientdet_net_train import EfficientDetNetTrain
from ai_api.ai_models.datasets.coco_dataset_one import GetDataSet
from ai_api.ai_models.utils.radam import RAdam
from ai_api.ai_models.utils.global_params import get_efficientdet_config
from ai_api.ai_models.efficientnet.utils.anchors import Anchors
from ai_api.ai_models.utils.block_args import EfficientDetBlockArgs

import argparse

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--trainData', default='./train2017')
parser.add_argument('--trainLabels', default='./data/coco_train2017_labels.txt')
parser.add_argument('--valData', default='./val2017')
parser.add_argument('--valLabels', default='./data/coco_val2017_labels.txt')
parser.add_argument('--classesFile', default='./data/coco_classes.txt')
args = parser.parse_args()

trainData = args.trainData
trainLabels = args.trainLabels
valData = args.valData
valLabels = args.valLabels
classesFile = args.classesFile


class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, path):
        '''初始化模型层'''
        super(SaveCallback, self).__init__()
        self.path = path

    def on_epoch_end(self, batch, logs=None):
        self.model.save_weights(self.path)
        # print('\n保存模型:{}'.format(self.path))

class CosineLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Cosine learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, total_steps: int):
    """Build a CosineLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      total_steps: `int`, Total train steps.
    """
    super().__init__()
    logging.info('LR schedule method: cosine')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)

  def __call__(self, step):

    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    cosine_lr = 0.5 * self.adjusted_lr * (
        1 + tf.cos(math.pi * tf.cast(step, tf.float32) / self.decay_steps))
    return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)

def train():
    '''训练'''
    # 加载数据
    batch_size = 2 # note that more GPU memory is required after unfreezing the body
    num_epochs = 300
    
    # Create model config.
    config = get_efficientdet_config(model_name='efficientdet-d1')

    anchors = Anchors(config.min_level,
                     config.max_level,
                     (config.image_size, config.image_size),
                     config.num_scales,
                     config.aspect_ratios,
                     config.anchor_scale)

    blocks_args = [
        EfficientDetBlockArgs(1,3,(1,1),1,32,16,0.25),
        EfficientDetBlockArgs(2,3,(2,2),6,16,24,0.25),
        EfficientDetBlockArgs(2,5,(2,2),6,24,40,0.25),
        EfficientDetBlockArgs(3,3,(2,2),6,40,80,0.25),
        EfficientDetBlockArgs(3,5,(1,1),6,80,112,0.25),
        EfficientDetBlockArgs(4,5,(2,2),6,112,192,0.25),
        EfficientDetBlockArgs(1,3,(1,1),6,192,320,0.25),
    ]
    data_set_train, data_generator_train = GetDataSet(image_path=trainData,
        label_path=trainLabels,
        classes_path=classesFile, batch_size=batch_size, anchors=anchors)
    data_set_val, data_generator_val = GetDataSet(image_path=valData,
        label_path=valLabels,
        classes_path=classesFile, batch_size=1, anchors=anchors, is_train=False)
    steps_per_epoch = max(1, data_generator_train.labels_num//batch_size//10)

    # 构建模型
    model = EfficientDetNetTrain(blocks_args=blocks_args, global_params=config, anchors=anchors)

    # 编译模型
    print('编译模型')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=Yolov4Loss(anchors=anchors,classes_num=data_generator_train.classes_num)) # recompile to apply the change
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
    adjusted_learning_rate = 0.08 * batch_size / 64
    lr_warmup_init = 0.008
    lr_warmup_step = int(1.0 * steps_per_epoch)
    total_steps = int(num_epochs * steps_per_epoch)
    learning_rate = CosineLrSchedule(adjusted_learning_rate,
                            lr_warmup_init, lr_warmup_step,
                            total_steps)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate, momentum=0.9)
        
    import tensorflow_addons as tfa  # pylint: disable=g-import-not-at-top
    optimizer = tfa.optimizers.MovingAverage(
        optimizer, average_decay=0.9998)
    model.compile(optimizer)
    # model.compile(optimizer=RAdam(lr=1e-4))

    # 日志
    log_dir = './data/'
    model_path = log_dir + 'trained_weights_final.h5'
    _ = model(tf.ones((1, config.image_size, config.image_size, 3)))
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print('加载模型:{}'.format(model_path))
    model.summary()
    # for v in model.trainable_weights:
    #     print(v.name, tf.shape(v))

    # logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    #     monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # 训练回调方法
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0, patience=10, verbose=1)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=6, verbose=1, mode='min', min_delta=0.0, min_lr=1e-6)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='min', restore_best_weights=True)

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(data_generator_train.labels_num, data_generator_val.labels_num, batch_size))
    print('开始训练')
    model.fit(data_set_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=data_set_val,
        validation_steps=max(1, data_generator_val.labels_num/10),
        epochs=num_epochs,
        initial_epoch=0,
        callbacks=[early_stopping, SaveCallback(model_path)]
        )
    model.save_weights(model_path)


def main():
    train()


if __name__ == '__main__':
    main()