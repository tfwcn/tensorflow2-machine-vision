import numpy as np
import tensorflow as tf

import sys
import os
sys.path.append(os.getcwd())
from ai_api.yolo_v3.model import YoloV3Model
from ai_api.yolo_v3.dataset_coco import GetDataSet
from ai_api.utils.radam import RAdam

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


def train():
    '''训练'''
    # 加载数据
    batch_size = 4 # note that more GPU memory is required after unfreezing the body
    anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
    anchors = np.array(anchors).reshape(-1, 2)
    data_set_train, data_generator_train = GetDataSet(image_path=trainData,
        label_path=trainLabels,
        classes_path=classesFile, batch_size=batch_size, anchors=anchors)
    data_set_val, data_generator_val = GetDataSet(image_path=valData,
        label_path=valLabels,
        classes_path=classesFile, batch_size=1, anchors=anchors, is_mean=False)

    # 构建模型
    model = YoloV3Model(anchors_num=len(anchors)//3, classes_num=data_generator_train.classes_num, anchors=anchors, image_size=(416, 416))

    # 编译模型
    print('编译模型')
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=Yolov4Loss(anchors=anchors,classes_num=data_generator_train.classes_num)) # recompile to apply the change
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4))
    # model.compile(optimizer=RAdam(lr=1e-4))

    # 日志
    log_dir = './data/'
    model_path = log_dir + 'trained_weights_final.h5'
    _ = model(tf.ones((1, 416, 416, 3)))
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print('加载模型:{}'.format(model_path))
    model.summary()

    # logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    #     monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # 训练回调方法
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=6, verbose=1, mode='min', min_delta=0.0, min_lr=1e-6)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(data_generator_train.labels_num, data_generator_val.labels_num, batch_size))
    print('开始训练')
    model.fit(data_set_train,
        # steps_per_epoch=max(1, data_generator_train.labels_num//batch_size),
        steps_per_epoch=5000,
        validation_data=data_set_val,
        validation_steps=max(1, data_generator_val.labels_num//10),
        # validation_steps=10,
        epochs=300,
        initial_epoch=0,
        callbacks=[reduce_lr, early_stopping, SaveCallback(model_path)]
        )
    model.save_weights(model_path)


def main():
    train()


if __name__ == '__main__':
    main()