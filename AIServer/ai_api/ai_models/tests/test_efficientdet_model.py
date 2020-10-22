# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
A simple example on how to use keras model for inference.
使用模型预测demo
"""
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf
import sys
sys.path.append(os.getcwd())

from ai_api.ai_models.utils.global_params import get_efficientdet_config
from ai_api.ai_models.utils import inference
from ai_api.ai_models.efficientnet.efficientdet_net import EfficientDetModel
from ai_api.ai_models.utils.block_args import EfficientDetBlockArgs

flags.DEFINE_string('image_path', './data/img.png', 'Location of test image.')
flags.DEFINE_string('output_dir', './data/output_imgs', 'Directory of annotated output images.')
flags.DEFINE_string('model_dir', './data/efficientdet-d4', 'Location of the checkpoint to run.')
flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs or a yaml file')
flags.DEFINE_bool('debug', False, 'If true, run function in eager for debug.')
flags.DEFINE_string('saved_model_dir', None, 'Saved model directory')
FLAGS = flags.FLAGS


def main(_):

  # pylint: disable=line-too-long
  # Prepare images and checkpoints: please run these commands in shell.
  # !mkdir tmp
  # !wget https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png -O tmp/img.png
  # !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz -O tmp/efficientdet-d0.tar.gz
  # !tar zxf tmp/efficientdet-d0.tar.gz -C tmp
  imgs = [np.array(Image.open(FLAGS.image_path))]
  # Create model config.
  config = get_efficientdet_config(model_name='efficientdet-d4')
  config.is_training_bn = False
  # config.image_size = 1920
  config.nms_configs.score_thresh = 0.4
  config.nms_configs.max_output_size = 100
  config.override(FLAGS.hparams)

  blocks_args = [
    EfficientDetBlockArgs(1,3,(1,1),1,32,16,0.25),
    EfficientDetBlockArgs(2,3,(2,2),6,16,24,0.25),
    EfficientDetBlockArgs(2,5,(2,2),6,24,40,0.25),
    EfficientDetBlockArgs(3,3,(2,2),6,40,80,0.25),
    EfficientDetBlockArgs(3,5,(1,1),6,80,112,0.25),
    EfficientDetBlockArgs(4,5,(2,2),6,112,192,0.25),
    EfficientDetBlockArgs(1,3,(1,1),6,192,320,0.25),
  ]

  # Use 'mixed_float16' if running on GPUs.
  policy = tf.keras.mixed_precision.experimental.Policy('float32')
  tf.keras.mixed_precision.experimental.set_policy(policy)
  tf.config.experimental_run_functions_eagerly(FLAGS.debug)

  # Create and run the model.
  model = EfficientDetModel(blocks_args=blocks_args, global_params=config)
  model.build((None, None, None, 3))
  print('model_dir:', tf.train.latest_checkpoint(FLAGS.model_dir))
  # for v in model.trainable_weights:
  #   print(v.name, tf.shape(v))
  # model.load_weights(tf.train.latest_checkpoint(FLAGS.model_dir))
  # 将原权重复制到新模型
  for i in range(len(model.trainable_variables)):
    ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    v_new = model.trainable_variables[i]
    v_old = tf.train.load_variable(ckpt_path, v_new.name)
    tf.print(v_new.name)
    if (np.shape(v_old) == np.shape(v_new.numpy())):
      v_new.assign(v_old)
    else:
      tf.print(str(i)+':', v_new.name, np.shape(v_old), np.shape(v_new.numpy()))
  return
  model.summary()
  class ExportModel(tf.Module):
    def __init__(self, model):
      super().__init__()
      self.model = model

    @tf.function
    def f(self, imgs):
      return self.model(imgs, training=False, post_mode='global')

  imgs = tf.convert_to_tensor(imgs, dtype=tf.uint8)
  export_model = ExportModel(model)
  # 默认不保存模型
  if FLAGS.saved_model_dir:
    tf.saved_model.save(
      export_model, FLAGS.saved_model_dir,
      signatures=export_model.f.get_concrete_function(
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8)))
    export_model = tf.saved_model.load(FLAGS.saved_model_dir)

  boxes, scores, classes, valid_len = export_model.f(imgs)

  # Visualize results.
  for i, img in enumerate(imgs):
    length = valid_len[i]
    img = inference.visualize_image(
        img,
        boxes[i].numpy()[:length],
        classes[i].numpy().astype(np.int)[:length],
        scores[i].numpy()[:length],
        label_map=config.label_map,
        min_score_thresh=config.nms_configs.score_thresh,
        max_boxes_to_draw=config.nms_configs.max_output_size)
    output_image_path = os.path.join(FLAGS.output_dir, str(i) + '.jpg')
    Image.fromarray(img).save(output_image_path)
    logging.info('writing annotated image to ', output_image_path)


if __name__ == '__main__':
  flags.mark_flag_as_required('image_path')
  flags.mark_flag_as_required('output_dir')
  flags.mark_flag_as_required('model_dir')
  logging.set_verbosity(logging.ERROR)
  app.run(main)
