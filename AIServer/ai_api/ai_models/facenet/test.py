import tensorflow as tf
import tensorflow_addons as tfa
import argparse

import sys
import os
sys.path.append(os.getcwd())
from ai_api.ai_models.facenet.facenet_model import FaceNetModel

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('files_path', type=str, nargs='+')
parser.add_argument('--image_size', type=int,
  help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--embedding_size', type=int,
  help='Dimensionality of the embedding.', default=128)
parser.add_argument('--backbone', 
  help='主体结构：InceptionResNetV1、InceptionResNetV2、InceptionV4、KerasInceptionResNetV2',
  type=str, default='InceptionResNetV1')
  
args = parser.parse_args()


def main():
  model = FaceNetModel(embedding_size=args.embedding_size,
                       image_size=args.image_size,
                       backbone=args.backbone,
                       dropout_rate=0.0,
                       random_crop=False,
                       random_flip=False)

  o = model(tf.zeros([2,160,160,3], dtype=tf.float32), training=False)
  print('out:', o.shape)

  # optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1, decay=0.9, momentum=0.9, epsilon=1.0)
  # optimizer = tfa.optimizers.MovingAverage(
  #     optimizer, average_decay=0.9999, start_step=0, sequential_update=True)
  # model.compile(optimizer=optimizer)

  # 加载模型
  log_dir = './data/'
  checkpoint_dir = log_dir + 'facenet/'
  if os.path.exists(checkpoint_dir):
    print('加载模型权重:{}'.format(checkpoint_dir))
    model.load_weights(checkpoint_dir).expect_partial()
    print('加载模型完成！')

  files_path = args.files_path
  images = model.load_image(files_path)
  embeddings = model(images, training=False)
  for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
      dist = model.euclidean_distance(embeddings[i],embeddings[j])
      print('%d %d: %f' % (i, j, dist))

if __name__ == '__main__':
  main()