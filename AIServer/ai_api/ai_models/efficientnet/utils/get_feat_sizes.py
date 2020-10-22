from typing import Tuple, List


def get_feat_sizes(image_size: Tuple[int, int],
                   max_level: int) -> List[Tuple[int, int]]:
  '''计算特征层大小

  Args:
    image_size: (H, W)
    max_level: maximum feature level.

  Returns:
    feat_sizes: [layer_index, (height, width)]
  '''
  feat_size = image_size
  feat_sizes = [feat_size]
  for _ in range(1, max_level + 1):
    feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
    feat_sizes.append(feat_size)
  return feat_sizes

