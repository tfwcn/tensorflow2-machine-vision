
import collections

# 相当于定义一个具有下面字段的BlockArgs类，构造函数含这些参数
EfficientDetBlockArgs = collections.namedtuple('EfficientDetBlockArgs', [
    'num_repeat', 'kernel_size', 'strides', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
# 默认全部参数为None
EfficientDetBlockArgs.__new__.__defaults__ = (None,) * len(EfficientDetBlockArgs._fields)
