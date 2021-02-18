import tensorflow as tf

class ConvBn(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides=1, padding='same', groups=1, **kwargs):
    super(ConvBn, self).__init__(**kwargs)
    self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                                       groups=groups, use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training):
    x = inputs
    x = self.conv(x)
    x = self.bn(x, training=training)
    return x

class RepVGGBlock(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides=1, padding='same', groups=1, deploy=False, **kwargs):
    super(RepVGGBlock, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.groups = groups
    self.deploy = deploy

  def build(self, input_shape):
    self.in_channels = input_shape[-1]
    if self.deploy:
        # 部署
        self.rbr_reparam = tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding,
                                                  groups=self.groups, use_bias=True)
    else:
        # 训练
        self.rbr_identity = tf.keras.layers.BatchNormalization() if self.in_channels == self.filters and self.strides == 1 else None
        self.rbr_dense = ConvBn(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, groups=self.groups)
        self.rbr_1x1 = ConvBn(self.filters, kernel_size=1, strides=self.strides, padding=self.padding, groups=self.groups)
        # tf.print('RepVGG Block, identity = ', self.rbr_identity)
    super(RepVGGBlock, self).build(input_shape)

  def call(self, inputs, training):
    x = inputs
    if hasattr(self, 'rbr_reparam'):
      return tf.nn.relu(self.rbr_reparam(x))

    if self.rbr_identity:
      id_out = self.rbr_identity(x, training=training)
    else:
      id_out = 0

    x = tf.nn.relu(self.rbr_dense(x, training=training) + self.rbr_1x1(x, training=training) + id_out)
    return x
  
  def get_equivalent_kernel_bias(self):
    '''计算等价kernel与bias'''
    kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
    kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
    kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
    return kernel3x3 + tf.pad(kernel1x1, [[1,1],[1,1],[0,0],[0,0]]) + kernelid, bias3x3 + bias1x1 + biasid

  def _fuse_bn_tensor(self, branch):
    if branch is None:
      return 0, 0
    if isinstance(branch, ConvBn):
      kernel = branch.conv.kernel
      moving_mean = branch.bn.moving_mean
      moving_variance = branch.bn.moving_variance
      gamma = branch.bn.gamma
      beta = branch.bn.beta
      eps = branch.bn.epsilon
    else:
      assert isinstance(branch, tf.keras.layers.BatchNormalization)
      if not hasattr(self, 'id_tensor'):
        input_dim = self.in_channels // self.groups
        kernel_value = tf.zeros((3, 3, input_dim, self.in_channels), dtype=tf.float32)
        for i in range(self.in_channels):
          kernel_value = tf.tensor_scatter_nd_update(kernel_value, [[1, 1, i % input_dim, i]], [1])
        self.id_tensor = kernel_value
      kernel = self.id_tensor
      moving_mean = branch.moving_mean
      moving_variance = branch.moving_variance
      gamma = branch.gamma
      beta = branch.beta
      eps = branch.epsilon
    std = tf.math.sqrt(moving_variance + eps)
    t = tf.reshape(gamma / std, (1, 1, 1, -1))
    return kernel * t, beta - moving_mean * gamma / std

  def repvgg_convert(self):
    kernel, bias = self.get_equivalent_kernel_bias()
    return kernel, bias

class RepVGG(tf.keras.Model):
  def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, **kwargs):
    super(RepVGG, self).__init__(**kwargs)
    self.num_blocks = num_blocks
    self.num_classes = num_classes
    self.width_multiplier = width_multiplier
    self.override_groups_map = override_groups_map or dict()
    self.deploy = deploy

    assert len(self.width_multiplier) == 4
    assert 0 not in self.override_groups_map

    self.in_planes = min(64, int(64 * self.width_multiplier[0]))

  def build(self, input_shape):
    self.stage0 = RepVGGBlock(self.in_planes, kernel_size=3, strides=2, padding='same', deploy=self.deploy)
    self.cur_layer_idx = 1
    self.stage1 = self._make_stage(int(64 * self.width_multiplier[0]), self.num_blocks[0], stride=2)
    self.stage2 = self._make_stage(int(128 * self.width_multiplier[1]), self.num_blocks[1], stride=2)
    self.stage3 = self._make_stage(int(256 * self.width_multiplier[2]), self.num_blocks[2], stride=2)
    self.stage4 = self._make_stage(int(512 * self.width_multiplier[3]), self.num_blocks[3], stride=2)
    
    if self.num_classes:
      self.gap = tf.keras.layers.GlobalAveragePooling2D()
      self.dense = tf.keras.layers.Dense(self.num_classes)
    super(RepVGG, self).build(input_shape)


  def _make_stage(self, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    blocks = []
    for stride in strides:
      cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
      blocks.append(RepVGGBlock(planes, kernel_size=3,
                                strides=stride, padding='same', groups=cur_groups, deploy=self.deploy))
      self.in_planes = planes
      self.cur_layer_idx += 1
    return tf.keras.Sequential(blocks)


  def call(self, inputs, training):
    out = {}
    x = inputs
    x = self.stage0(x, training=training)
    out['stage0'] = x
    x = self.stage1(x, training=training)
    out['stage1'] = x
    x = self.stage2(x, training=training)
    out['stage2'] = x
    x = self.stage3(x, training=training)
    out['stage3'] = x
    x = self.stage4(x, training=training)
    out['stage4'] = x
    if self.num_classes:
      x = self.gap(x, training=training)
      x = self.dense(x)
      return x
    else:
      return out

  def get_all_layers(self):
    layers = {}
    for layer in self.layers:
      if isinstance(layer, tf.keras.Sequential):
        for l in layer.layers:
          layers[layer.name+'/'+l.name]=l
      else:
        layers[layer.name]=layer
    return layers

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_A0(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A1(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)

def create_RepVGG_B0(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1g2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B1g4(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B2g2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B2g4(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B3g2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B3g4(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


func_dict = {
  'RepVGG-A0': create_RepVGG_A0,
  'RepVGG-A1': create_RepVGG_A1,
  'RepVGG-A2': create_RepVGG_A2,
  'RepVGG-B0': create_RepVGG_B0,
  'RepVGG-B1': create_RepVGG_B1,
  'RepVGG-B1g2': create_RepVGG_B1g2,
  'RepVGG-B1g4': create_RepVGG_B1g4,
  'RepVGG-B2': create_RepVGG_B2,
  'RepVGG-B2g2': create_RepVGG_B2g2,
  'RepVGG-B2g4': create_RepVGG_B2g4,
  'RepVGG-B3': create_RepVGG_B3,
  'RepVGG-B3g2': create_RepVGG_B3g2,
  'RepVGG-B3g4': create_RepVGG_B3g4,
}
def get_RepVGG_func_by_name(name):
  return func_dict[name]

#   Use like this:
#   train_model = get_RepVGG_func_by_name('RepVGG-A0')(deploy=False)
#   train train_model
#   deploy_model = repvgg_convert(train_model, 'RepVGG-A0', save_path='repvgg_deploy.pth')
def repvgg_model_convert(model:RepVGG, model_name, save_path=None):
  converted_weights = {}
  for name, layer in model.get_all_layers().items():
    if hasattr(layer, 'repvgg_convert'):
      kernel, bias = layer.repvgg_convert()
      converted_weights[name + '/conv2d/kernel:0'] = kernel.numpy()
      converted_weights[name + '/conv2d/bias:0'] = bias.numpy()
    elif isinstance(layer, tf.keras.layers.Dense):
      converted_weights[name + '/kernel:0'] = layer.kernel.numpy()
      converted_weights[name + '/bias:0'] = layer.bias.numpy()
    else:
      print(name, type(layer))
  build_func = get_RepVGG_func_by_name(model_name)
  deploy_model = build_func(deploy=True)
  deploy_model.build((None, 160, 160, 3))
  trainable_variables = deploy_model.trainable_variables
  for i, var in enumerate(converted_weights.values()):
    deploy_var = trainable_variables[i]
    deploy_var.assign(var)
    tf.print('deploy param: ', deploy_var.name, deploy_var.shape, tf.math.reduce_mean(deploy_var))

  if save_path is not None:
    deploy_model.save(save_path)

  return deploy_model

def main():
  '''发布demo'''
  m = get_RepVGG_func_by_name('RepVGG-A0')(deploy=False)
  x = tf.random.uniform((1, 160, 160, 3), dtype=tf.float32)
  m.build((None, 160, 160, 3))
  m_deploy = repvgg_model_convert(m, 'RepVGG-A0')
  out1 = m(x, training=False)
  out2 = m_deploy(x, training=False)
  tf.print('out1:', out1['classes'])
  tf.print('out2:', out2['classes'])
  # for l in m.get_all_layers():
  #   tf.print(l.name)
  # for v in m.trainable_variables:
  #   tf.print(v.name)

def test():
  '''合并权重'''
  x = tf.random.uniform((1, 4, 4, 1), dtype=tf.float32)
  conv1 = tf.keras.layers.Conv2D(10, 3, strides=1, padding='same',
                                groups=1, use_bias=False)
  bn1 = tf.keras.layers.BatchNormalization()

  conv2 = tf.keras.layers.Conv2D(10, 1, strides=1, padding='same',
                                groups=1, use_bias=False)
  bn2 = tf.keras.layers.BatchNormalization()

  out1 = conv1(x, training=False)
  out1 = bn1(out1, training=False)
  out2 = conv2(x, training=False)
  out2 = bn2(out2, training=False)
  out = out1 + out2

  def get_kb(conv, bn):
    kernel = conv.kernel
    moving_mean = bn.moving_mean
    moving_variance = bn.moving_variance
    gamma = bn.gamma
    beta = bn.beta
    eps = bn.epsilon
    std = tf.math.sqrt(moving_variance + eps)
    t = tf.reshape(gamma / std, (1, 1, 1, -1))
    kernel, bias = kernel * t, beta - moving_mean * gamma / std
    # print('kernel, bias:', kernel, bias)
    return kernel, bias
  kernel1, bias1 = get_kb(conv1, bn1)
  kernel2, bias2 = get_kb(conv2, bn2)
  kernel, bias = kernel1 + tf.pad(kernel2, [[1,1],[1,1],[0,0],[0,0]]), bias1 + bias2
  conv_deploy = tf.keras.layers.Conv2D(10, 3, strides=1, padding='same',
                                      groups=1, use_bias=True)
  conv_deploy(tf.zeros((1, 4, 4, 1), dtype=tf.float32), training=False)
  conv_deploy.kernel.assign(kernel)
  conv_deploy.bias.assign(bias)
  out_deploy = conv_deploy(x, training=False)
  print('out:', out)
  print('out_deploy:', out_deploy)

def test2():
  '''测试不变卷积核'''
  kernel_value = tf.zeros((3, 3, 2, 4), dtype=tf.float32)
  for i in range(4):
    kernel_value = tf.tensor_scatter_nd_update(kernel_value, [[1, 1, i % 2, i]], [1])
  # print('kernel_value:', kernel_value)
  x = tf.random.uniform((1, 4, 4, 4), dtype=tf.float32)
  conv1 = tf.keras.layers.Conv2D(4, 3, strides=1, padding='same',
                                groups=2, use_bias=False)
  out = conv1(x, training=False)
  conv1.kernel.assign(kernel_value)
  out = conv1(x, training=False)
  tf.print('x:', x)
  tf.print('out:', out)

def test3():
  '''测试RepVGGBlock'''
  m = RepVGGBlock(3, 3, strides=1, deploy=False)
  x = tf.random.uniform((1, 4, 4, 3), dtype=tf.float32)
  out = m(x, training=False)
  kernel, bias = m.get_equivalent_kernel_bias()
  # print('kernel, bias:', kernel, bias)
  m_deploy = RepVGGBlock(3, 3, strides=1, deploy=True)
  m_deploy(tf.zeros(x.shape, dtype=tf.float32), training=False)
  m_deploy.rbr_reparam.kernel.assign(kernel)
  m_deploy.rbr_reparam.bias.assign(bias)
  out_deploy = m_deploy(x, training=False)
  print('out:', out)
  print('out_deploy:', out_deploy)

if __name__ == '__main__':
  main()
  # test()
  # test2()
  # test3()