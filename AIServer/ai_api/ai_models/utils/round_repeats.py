import math

def round_repeats(repeats, global_params):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_coefficient
  return int(math.ceil(multiplier * repeats))
