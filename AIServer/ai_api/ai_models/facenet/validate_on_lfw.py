import tensorflow as tf
import numpy as np
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.facenet.facenet_model import FaceNetModel
import ai_api.ai_models.facenet.lfw as lfw


# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('files_path', type=str)
parser.add_argument('--image_size', type=int,
  help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--embedding_size', type=int,
  help='Dimensionality of the embedding.', default=128)
parser.add_argument('--backbone', 
  help='主体结构：InceptionResNetV1、InceptionResNetV2、InceptionV4、KerasInceptionResNetV2',
  type=str, default='InceptionResNetV1')
parser.add_argument('--batch_size', type=int,
  help='Number of images to process in a batch.', default=90)
parser.add_argument('--lfw_nrof_folds', type=int,
  help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
parser.add_argument('--lfw_pairs', type=str,
  help='The file containing the pairs to use for validation.', default='data/pairs.txt')
  
args = parser.parse_args()


def get_image_list(files_path):
  image_list = []
  for dir_path in os.listdir(files_path):
    if os.path.isdir(os.path.join(files_path, dir_path)):
      one_image_list = []
      for f in os.listdir(os.path.join(files_path, dir_path)):
        ext = os.path.splitext(f)[1]
        if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
          one_image_list.append(os.path.join(files_path, dir_path, f))
      if len(one_image_list):
        image_list.append(one_image_list)
  return image_list


def main():
  model = FaceNetModel(embedding_size=args.embedding_size,
                       image_size=args.image_size,
                       backbone=args.backbone,
                       dropout_rate=0.0,
                       random_crop=False,
                       random_flip=False)

  o = model(tf.zeros([2,160,160,3], dtype=tf.float32), training=False)
  print('out:', o.shape)

  # 加载模型
  log_dir = './data/'
  checkpoint_dir = log_dir + 'facenet/'
  if os.path.exists(checkpoint_dir):
    print('加载模型权重:{}'.format(checkpoint_dir))
    model.load_weights(checkpoint_dir).expect_partial()
    print('加载模型完成！')

  files_path = args.files_path
  # Read the file containing the pairs used for testing
  pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

  # Get the paths for the corresponding images
  image_paths, actual_issame = lfw.get_paths(os.path.expanduser(files_path), pairs)
  # image_paths = get_image_list(files_path)
  embeddings = model.get_embeddings(image_paths, args.batch_size)
  
  tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(embeddings.numpy(), 
      actual_issame, nrof_folds=args.lfw_nrof_folds)

  print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
  print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

  auc = metrics.auc(fpr, tpr)
  print('Area Under Curve (AUC): %1.3f' % auc)
  eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
  print('Equal Error Rate (EER): %1.3f' % eer)

if __name__ == '__main__':
  main()