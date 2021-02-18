import tensorflow as tf
import tensorflow_addons as tfa
import argparse

import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.facenet.dataset import DataGenerator
from ai_api.ai_models.facenet.facenet_model import FaceNetTrainModel

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('files_path', type=str)
parser.add_argument('--batch_size', type=int,
  help='Number of images to process in a batch.', default=90)
parser.add_argument('--image_size', type=int,
  help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--people_per_batch', type=int,
  help='Number of people per batch.', default=45)
parser.add_argument('--images_per_person', type=int,
  help='Number of images per person.', default=40)
parser.add_argument('--max_nrof_epochs', type=int,
  help='Number of epochs to run.', default=500)
parser.add_argument('--epoch_size', type=int,
  help='Number of batches per epoch.', default=1000)
parser.add_argument('--alpha', type=float,
  help='Positive to negative triplet distance margin.', default=0.2)
parser.add_argument('--embedding_size', type=int,
  help='Dimensionality of the embedding.', default=128)
parser.add_argument('--random_crop', 
  help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
    'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
parser.add_argument('--random_flip', 
  help='Performs random horizontal flipping of training images.', action='store_true')
parser.add_argument('--learning_rate', 
  help='学习速率.', type=float, default=0.001)
parser.add_argument('--moving_average', 
  help='平滑梯度.', action='store_true')
parser.add_argument('--moving_average_decay', 
  help='衰变可用来维持训练变量的移动平均值。仅在有--moving_average参数时有效。', type=float, default=0.9999)
parser.add_argument('--loss_decay', 
  help='衰变可用来维持loss的移动平均值。仅在有--moving_average参数时有效。', type=float, default=0.9)
parser.add_argument('--backbone', 
  help='主体结构：InceptionResNetV1、InceptionResNetV2、InceptionV4、KerasInceptionResNetV2、RepVGG',
  type=str, default='InceptionResNetV1')
parser.add_argument('--dropout_rate', 
  help='dropout率.', type=float, default=0.0)
parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP'],
  help='The optimization algorithm to use', default='ADAGRAD')

# Parameters for validation on LFW
parser.add_argument('--lfw_pairs', type=str,
  help='The file containing the pairs to use for validation.', default='data/pairs.txt')
parser.add_argument('--lfw_dir', type=str,
  help='Path to the data directory containing aligned face patches.', default='')
parser.add_argument('--lfw_nrof_folds', type=int,
  help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
  
args = parser.parse_args()


def main():
  # people_per_batch: 每批取的人数
  # images_per_person: 每人最多取多少张图
  data_generator = DataGenerator(args.files_path, args.people_per_batch, args.images_per_person)
  # 所有图片，随机选一批图片，计算图片特征，根据特征选择三元组，计算三元组loss
  dataset = data_generator.GetDataSet()

  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
  with strategy.scope():
    model = FaceNetTrainModel(alpha=args.alpha,
                              embedding_size=args.embedding_size,
                              batch_size=args.batch_size,
                              people_per_batch=args.people_per_batch,
                              images_per_person=args.images_per_person,
                              image_size=args.image_size,
                              random_crop=args.random_crop,
                              random_flip=args.random_flip,
                              backbone=args.backbone,
                              dropout_rate=args.dropout_rate,
                              moving_average=args.moving_average,
                              moving_average_decay=args.moving_average_decay,
                              loss_decay=args.loss_decay,
                              strategy=strategy)

    o = model(tf.zeros([2,160,160,3], dtype=tf.float32))
    print('out:', o.shape)
    
    learning_rate = args.learning_rate
    optimizer = None
    if args.optimizer=='ADAGRAD':
      optimizer = tf.keras.optimizers.Adagrad(learning_rate)
    elif args.optimizer=='ADADELTA':
      optimizer = tf.keras.optimizers.Adadelta(learning_rate, rho=0.9, epsilon=1e-6)
    elif args.optimizer=='ADAM':
      optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=0.1)
    elif args.optimizer=='RMSPROP':
      optimizer = tf.keras.optimizers.RMSprop(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    else:
      raise ValueError('Invalid optimization algorithm')
    # optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=args.moving_average_decay)
    model.compile(optimizer=optimizer)

    # 加载模型
    log_dir = './data/'
    checkpoint_dir = log_dir + 'facenet/'
    if os.path.exists(checkpoint_dir):
      print('加载模型权重:{}'.format(checkpoint_dir))
      model.load_weights(checkpoint_dir)
      print('加载模型完成！')
    
    # 保存普通记录点
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                    save_weights_only=True,
                                                    verbose=1)
    callbacks = [checkpoint_callback]
    

    model.fit(dataset, epochs=args.max_nrof_epochs,
              steps_per_epoch=args.epoch_size,
              callbacks=callbacks,
              lfw_pairs=args.lfw_pairs,
              lfw_dir=args.lfw_dir,
              lfw_nrof_folds=args.lfw_nrof_folds)

if __name__ == '__main__':
  main()