import tensorflow as tf

@tf.function
def LoadImage(file_path):
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

@tf.function
def ResizeWithPad(img,
                  target_height,
                  target_width,
                  method=tf.image.ResizeMethod.BILINEAR,
                  antialias=False,
                  random_pad=False):
  '''
  等比缩放图片

  Args:
    antialias:抗锯齿
    random_pad:为Falae时，图片居中。为True时，图片随机偏移，但不会超出图片。
  
  Results:
    p_height：上边缘填充高度
    p_width：左边缘填充宽度
    resized_height：缩放后实际图片高度，f_height / ratio
    resized_width：缩放后实际图片宽度，f_width / ratio
    ratio：缩放比例，
  '''
  img_shape = tf.shape(img)
  f_height = tf.cast(img_shape[0], tf.float32)
  f_width = tf.cast(img_shape[1], tf.float32)
  f_target_height = tf.cast(target_height, tf.float32)
  f_target_width = tf.cast(target_width, tf.float32)
  ratio = tf.math.maximum(f_width / f_target_width, f_height / f_target_height)
  resized_height_float = f_height / ratio
  resized_width_float = f_width / ratio
  resized_height = tf.cast(
      tf.math.floor(resized_height_float), dtype=tf.int32)
  resized_width = tf.cast(
      tf.math.floor(resized_width_float), dtype=tf.int32)
  padding_height = (f_target_height - resized_height_float) / 2.0
  padding_width = (f_target_width - resized_width_float) / 2.0
  f_padding_height = tf.math.floor(padding_height)
  f_padding_width = tf.math.floor(padding_width)
  p_height = tf.math.maximum(0, tf.cast(f_padding_height, dtype=tf.int32))
  p_width = tf.math.maximum(0, tf.cast(f_padding_width, dtype=tf.int32))
  if random_pad:
    p_height = tf.random.uniform((), minval=0, maxval=p_height+1, dtype=tf.int32)
    p_width = tf.random.uniform((), minval=0, maxval=p_width+1, dtype=tf.int32)
  # 等比缩放
  img = tf.image.resize(
      img,
      (resized_height, resized_width),
      method=method,
      antialias=antialias
  )
  # 填充边缘到目标大小
  img = tf.image.pad_to_bounding_box(img, p_height, p_width, target_height,
                                 target_width)
  return img, p_height, p_width, resized_height, resized_width, ratio

@tf.function
def RandomColor(img):
  '''随机变换图片 亮度、对比度、色相、饱和度'''
  random_type = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
  if random_type == 0:
    # 随机亮度
    return tf.image.random_brightness(img, max_delta=0.2)
  elif random_type == 1:
    # 随机对比度
    return tf.image.random_contrast(img, lower=0.2, upper=0.5)
  elif random_type == 2:
    # 随机色相
    return tf.image.random_hue(img, max_delta=0.0)
  else:
    # 随机饱和度
    return tf.image.random_saturation(img, lower=0.2, upper=0.5)

@tf.function
def PadOrCropToBoundingBox(img, offset_height, offset_width, target_height, target_width):
  '''
  裁剪或填充图片

  Args:
    offset_height: 正数填充，负数裁剪，范围[0,target_width-image_weight)
    offset_width: 正数填充，负数裁剪，范围[0,target_height-image_height)
  '''
  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  # 填充
  oh = tf.cast(tf.math.maximum(0, offset_height), dtype=tf.int32)
  ow = tf.cast(tf.math.maximum(0, offset_width), dtype=tf.int32)
  th = tf.cast(tf.math.maximum(target_height, h), dtype=tf.int32)
  tw = tf.cast(tf.math.maximum(target_width, w), dtype=tf.int32)
  img = tf.image.pad_to_bounding_box(
      img, oh, ow, th, tw
  )
  # 裁剪
  oh = tf.cast(tf.math.maximum(0, -offset_height), dtype=tf.int32)
  ow = tf.cast(tf.math.maximum(0, -offset_width), dtype=tf.int32)
  th = tf.cast(target_width, dtype=tf.int32)
  tw = tf.cast(target_width, dtype=tf.int32)
  img = tf.image.crop_to_bounding_box(
      img, oh, ow, th, tw
  )
  return img