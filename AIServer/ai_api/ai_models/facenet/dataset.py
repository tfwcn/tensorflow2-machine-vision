import tensorflow as tf
import numpy as np
import os

class DataGenerator(object):
  def __init__(self, files_path, people_per_batch, images_per_person):
    self.files_path = files_path
    self.people_per_batch = people_per_batch
    self.images_per_person = images_per_person

    # 加载图片列表
    print('图片路径:', self.files_path)
    self.image_list = self.get_image_list()
    print('图片数:', len(self.image_list))

  def get_image_list(self):
    image_list = []
    for dir_path in os.listdir(self.files_path):
      if os.path.isdir(os.path.join(self.files_path, dir_path)):
        one_image_list = []
        for f in os.listdir(os.path.join(self.files_path, dir_path)):
          ext = os.path.splitext(f)[1]
          if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
            one_image_list.append(os.path.join(self.files_path, dir_path, f))
        if len(one_image_list):
          image_list.append(one_image_list)
    return image_list

  def sample_people(self):
    '''
    获取people_per_batch * images_per_person张图片
    同人最多images_per_person张

    Args:
        people_per_batch: 一批多少人
        images_per_person: 每人最大取多少图

    Returns:
        image_paths: 图片路径集合
        num_per_class：同人图片数量,[p1_num,p2_num,...]
    '''
    # 最大取样数
    nrof_images = self.people_per_batch * self.images_per_person

    # Sample classes from the dataset
    # 所有人数
    nrof_classes = len(self.image_list)
    # 生成序号，打乱顺序
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    # 图片路径
    image_paths = []
    # 同人图片数
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    # 一直添加图片，达到取样数量
    while len(image_paths)<nrof_images:
      class_index = class_indices[i]
      # 当前人的图片数
      nrof_images_in_class = len(self.image_list[class_index])
      # 随机同人图
      image_indices = np.arange(nrof_images_in_class)
      np.random.shuffle(image_indices)
      # 如果同人，图片太多就取前images_per_person张，不足则取全部，如果达到最大取样数则停止
      nrof_images_from_class = min(nrof_images_in_class, self.images_per_person, nrof_images-len(image_paths))
      # 通过下标取图片，下标已随机，因此相当于随机取前几张图片
      idx = image_indices[0:nrof_images_from_class]
      # 同人图片路径列表
      image_paths_for_class = [self.image_list[class_index][j] for j in idx]
      # 将图片对应的人物id加入数组
      sampled_class_indices += [class_index]*nrof_images_from_class
      # 将图片累计到列表
      image_paths += image_paths_for_class
      # 当前人物id取样数
      num_per_class.append(nrof_images_from_class)
      i+=1
    # print('image_paths, num_per_class:', len(image_paths), np.sum(num_per_class))
    return image_paths, num_per_class

  def generator(self):
    while True:
      image_paths, num_per_class = self.sample_people()
      yield image_paths, num_per_class

  def GetDataSet(self):
    dataset = tf.data.Dataset.from_generator(self.generator,
      (tf.string, tf.int32),
      (tf.TensorShape([None,]), tf.TensorShape([None,])))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # for image_paths, num_per_class in dataset.take(1):
    #   print('image_paths, num_per_class:', len(image_paths), np.sum(num_per_class))
    return dataset

def main():
  files_path = 'Z:\\Labels\\lfw\\lfw_mtcnnpy_182\\'
  people_per_batch = 45
  images_per_person = 40
  data_generator = DataGenerator(files_path, people_per_batch, images_per_person)
  # 所有图片，随机选一批图片，计算图片特征，根据特征选择三元组，计算三元组loss
  data_generator.GetDataSet()

if __name__ == '__main__':
  main()