import tensorflow as tf

@tf.function
def get_image(image_paths, random_crop, random_flip, image_size):
  images = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  def loop_function(index, images):
    filename = image_paths[index]
    # 读取图片文件
    tf.print('filename:', filename)
    file_contents = tf.io.read_file(filename)
    image = tf.image.decode_image(file_contents, channels=3)
    
    if random_crop:
      # 随机裁剪
      image = tf.image.random_crop(image, [image_size, image_size, 3])
    else:
      # 裁剪或填充到目标大小
      image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    if random_flip:
      # 随机翻转
      image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(tf.cast(image, tf.float32))
    tf.print('index:', index, tf.shape(image))
    images = images.write(index, image)
    return index + 1, images
  _, images = tf.while_loop(lambda index, images: index<tf.shape(image_paths)[0],loop_function,(0, images))
  return images.stack()

@tf.function
def get_embeddings(images, batch_size):
  embeddings = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  nrof_examples = tf.shape(images)[0]
  nrof_batches = tf.cast(tf.math.ceil(nrof_examples / batch_size), tf.int32)
  for index in tf.range(nrof_batches):
    # 取每批batch_size张图，计算图片特征，最后一批小于batch_size，取剩下的所有
    batch_size_one = tf.math.minimum(nrof_examples-index*batch_size, batch_size)
    images_batch = images[index*batch_size:index*batch_size+batch_size_one]
    embeddings_batch = images_batch
    # 将特征加入数组
    for i2 in tf.range(batch_size_one):
      embeddings = embeddings.write(index*batch_size+i2, embeddings_batch[i2])
  return embeddings.stack()

# @tf.function
# def select_triplets(embeddings, image_paths, num_per_class, people_per_batch, alpha):
#   '''
#   选择三元组
  
#   Args:
#     embeddings: 图片特征列表
#     image_paths: 图片地址列表
#     num_per_class: 同人数量列表
#   '''
#   triplets = tf.TensorArray(tf.string, size=0, dynamic_size=True)
#   # 遍历所有人
#   embeddings_start_index = tf.constant(0, dtype=tf.int32)
#   triplets_index = 0
#   for i in range(people_per_batch):
#     # 其他人的特征
#     embeddings_other = tf.concat([embeddings[0:embeddings_start_index],
#                                  embeddings[embeddings_start_index+num_per_class[i]:]], axis=0)
#     image_paths_other = tf.concat([image_paths[0:embeddings_start_index],
#                                  image_paths[embeddings_start_index+num_per_class[i]:]], axis=0)
#     tf.print('embeddings_other:', tf.shape(embeddings_other))
#     tf.print('image_paths_other:', tf.shape(image_paths_other))
#     # 遍历这个人的特征，多少图就有多少特征
#     num_per = num_per_class[i]
#     for i2 in tf.range(1,num_per):
#       # 当前embeddings下标
#       a_idx = embeddings_start_index+i2-1
#       # 当前人的后面其他特征
#       embeddings_one = embeddings[embeddings_start_index+i2:embeddings_start_index+num_per]
#       image_paths_one = image_paths[embeddings_start_index+i2:embeddings_start_index+num_per]
#       # 当前人与其他人的距离，(other_people_num,)
#       neg_dists_sqr = tf.math.reduce_sum(tf.math.square(embeddings[a_idx]-embeddings_other), 1)
#       # 同人距离，(same_people_num,)
#       pos_dist_sqr = tf.math.reduce_sum(tf.math.square(embeddings[a_idx]-embeddings_one), 1)
#       # 插入维度，混合计算所有同人与不同人距离，(同人下标，不同人下标)
#       neg_dists_sqr = tf.reshape(neg_dists_sqr, (1, -1))
#       pos_dist_sqr = tf.reshape(pos_dist_sqr, (-1, 1))
#       # 找到合适的三元组，()
#       all_index = tf.where(tf.math.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))
#       for i in tf.range(tf.shape(all_index)[0]):
#         p_idx = all_index[i][0]
#         n_idx = all_index[i][1]
#         # tf.print('triplets:', a_idx, p_idx, n_idx)
#         # tf.print('triplets:', image_paths[a_idx], image_paths_one[p_idx], image_paths_other[n_idx])
#         triplets = triplets.write(triplets_index,image_paths[a_idx])
#         triplets_index += 1
#         triplets = triplets.write(triplets_index,image_paths_one[p_idx])
#         triplets_index += 1
#         triplets = triplets.write(triplets_index,image_paths_other[n_idx])
#         triplets_index += 1
#     # 起始下标增加
#     embeddings_start_index+=num_per
#   return triplets.stack()

def select_triplets(embeddings, image_paths, num_per_class, people_per_batch, alpha):
  '''
  选择三元组
  
  Args:
    embeddings: 图片特征列表
    image_paths: 图片地址列表
    num_per_class: 同人数量列表
  '''
  triplets = tf.TensorArray(tf.string, size=0, dynamic_size=True)
  # 遍历所有人
  def loop_function(i, embeddings_start_index, triplets_index, triplets):
    # 其他人的特征
    embeddings_other = tf.concat([embeddings[0:embeddings_start_index],
                                embeddings[embeddings_start_index+num_per_class[i]:]], axis=0)
    image_paths_other = tf.concat([image_paths[0:embeddings_start_index],
                                image_paths[embeddings_start_index+num_per_class[i]:]], axis=0)
    # tf.print('embeddings_other:', tf.shape(embeddings_other))
    # tf.print('image_paths_other:', tf.shape(image_paths_other))
    # 遍历这个人的特征，多少图就有多少特征
    num_per = num_per_class[i]
    def loop_function2(i2, triplets_index, triplets):
      # 当前embeddings下标
      a_idx = embeddings_start_index+i2-1
      # 当前人的后面其他特征
      embeddings_one = embeddings[embeddings_start_index+i2:embeddings_start_index+num_per]
      image_paths_one = image_paths[embeddings_start_index+i2:embeddings_start_index+num_per]
      # 当前人与其他人的距离，(other_people_num,)
      neg_dists_sqr = tf.math.reduce_sum(tf.math.square(embeddings[a_idx]-embeddings_other), 1)
      # 同人距离，(same_people_num,)
      pos_dist_sqr = tf.math.reduce_sum(tf.math.square(embeddings[a_idx]-embeddings_one), 1)
      # 插入维度，混合计算所有同人与不同人距离，(同人下标，不同人下标)
      neg_dists_sqr = tf.reshape(neg_dists_sqr, (1, -1))
      pos_dist_sqr = tf.reshape(pos_dist_sqr, (-1, 1))
      # 找到合适的三元组，()
      # all_index = tf.where(tf.math.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))
      # 按条件判断
      all_mask = tf.math.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr)
      print('all_mask:', all_mask)
      # 遍历所有正样本，随机挑选一个负样本，组成三元组
      def loop_function3(i3, triplets_index, triplets):
        # 获取符合条件的其他人
        image_paths_other_mask = tf.boolean_mask(image_paths_other, all_mask[i3])
        if tf.shape(image_paths_other_mask)[0]<=0:
          return i3+1, triplets_index, triplets
        p_idx = i3
        n_idx = tf.random.uniform(shape=[], maxval=tf.shape(image_paths_other_mask)[0], dtype=tf.int32)
        tf.print('triplets:', a_idx, p_idx, n_idx)
        tf.print('triplets:', image_paths[a_idx], image_paths_one[p_idx], image_paths_other[n_idx])
        triplets = triplets.write(triplets_index,image_paths[a_idx])
        triplets_index += 1
        triplets = triplets.write(triplets_index,image_paths_one[p_idx])
        triplets_index += 1
        triplets = triplets.write(triplets_index,image_paths_other[n_idx])
        triplets_index += 1
        return i3+1, triplets_index, triplets
      _, triplets_index, triplets = tf.while_loop(lambda i3,triplets_index, triplets: i3<tf.shape(all_mask)[0],loop_function3,
                                    (0, triplets_index, triplets))
      return i2+1, triplets_index, triplets
    _, triplets_index, triplets = tf.while_loop(lambda i2,triplets_index, triplets: i2<num_per,loop_function2,
                                  (1, triplets_index, triplets))
    # 起始下标增加
    embeddings_start_index+=num_per
    return i+1, embeddings_start_index, triplets_index, triplets
  _, _, _, triplets = tf.while_loop(lambda i,embeddings_start_index, triplets_index, triplets: i<people_per_batch,loop_function,
                                (0, 0, 0, triplets))
  return triplets.stack()

# image_paths = tf.constant(['Z:\\Labels\\lfw\\lfw_mtcnnpy_182\\Aaron_Sorkin\\Aaron_Sorkin_0001.png',
#   'Z:\\Labels\\lfw\\lfw_mtcnnpy_182\\Aaron_Sorkin\\Aaron_Sorkin_0001.png'], dtype=tf.string)
# images = get_image(image_paths, True, True, 160)
# print('images:', images.shape)

# batch_size = 3
# x = tf.zeros([20, 128], dtype=tf.float32)
# embeddings = get_embeddings(x,batch_size)
# tf.print('embeddings:', embeddings.shape)

# embeddings=tf.random.uniform((4,128))
# image_paths=tf.constant(['a.jpg', 'a2.jpg', 'b.jpg', 'b2.jpg'], dtype=tf.string)
# num_per_class=[2,2]
# people_per_batch=2
# alpha=0.2
# triplets=select_triplets(embeddings, image_paths, num_per_class, people_per_batch, alpha)
# tf.print('triplets:',triplets)

# @tf.function
# def test(x):
#   a = tf.constant(1, dtype=tf.int32)
#   b = tf.constant(3, dtype=tf.int32)
#   return x[a:b]

# x = tf.range(10, dtype=tf.float32)
# o = test(x)
# tf.print('o:', o)

x = tf.range(9, dtype=tf.float32)
o = tf.reshape(x,(-1,3))
tf.print('o:', tf.shape(o), o)
o1,o2,o3 = tf.unstack(o,3,1)
tf.print('o1,o2,o3:', o1,o2,o3)


# @tf.function
# def test():
#   triplets_num = 30 // 3
#   batch_size = 20 // 3
#   triplets_batches = tf.cast(tf.math.ceil(triplets_num / batch_size), tf.int32)
#   tf.print('triplets_batches:', triplets_batches)
#   # 分批，因一次计算所有特征，显存不足
#   for i in tf.range(triplets_batches):
#     # 取每批batch_size张图，计算图片特征，最后一批小于batch_size，取剩下的所有
#     batch_size_one = tf.math.minimum(triplets_num-i*batch_size, batch_size)
#     tf.print(i*batch_size*3,(i*batch_size+batch_size_one)*3)
# test()

# def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
#   """ Select the triplets for training
#   选择三元组
#   """
#   trip_idx = 0
#   emb_start_idx = 0
#   num_trips = 0
#   triplets = []
  
#   # VGG Face: Choosing good triplets is crucial and should strike a balance between
#   #  selecting informative (i.e. challenging) examples and swamping training with examples that
#   #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
#   #  the image n at random, but only between the ones that violate the triplet loss margin. The
#   #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
#   #  choosing the maximally violating example, as often done in structured output learning.

#   # 一批多少人，people_per_batch：45
#   for i in range(people_per_batch):
#     # 这人有多少张图
#     nrof_images = int(nrof_images_per_class[i])
#     # 遍历这个人的图
#     for j in range(1,nrof_images):
#       # 当前embeddings下标
#       a_idx = emb_start_idx + j - 1
#       # 当前人与所有人的距离
#       neg_dists_sqr = tf.math.reduce_sum(tf.math.square(embeddings[a_idx] - embeddings), 1)
#       # 循环所有正样本对
#       for pair in range(j, nrof_images): # For every possible positive pair.
#         # 同人下标
#         p_idx = emb_start_idx + pair
#         # 计算同人距离
#         pos_dist_sqr = tf.math.reduce_sum(tf.math.square(embeddings[a_idx]-embeddings[p_idx]))
#         # 在负样本中，把同人距离设置成NaN
#         neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = tf.math.NaN
#         # 按距离选择符合条件的负样本
#         all_neg = tf.where(tf.math.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
#         # all_neg = tf.math.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
#         # 负样本数
#         nrof_random_negs = all_neg.shape[0]
#         # 如果存在负样本
#         if nrof_random_negs>0:
#           # 随机选择一个负样本，加入三元组
#           rnd_idx = tf.random.randint(nrof_random_negs)
#           n_idx = all_neg[rnd_idx]
#           triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
#           #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
#           #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
#           trip_idx += 1

#         num_trips += 1

#     emb_start_idx += nrof_images

#   tf.random.shuffle(triplets)
#   return triplets, num_trips, len(triplets)
