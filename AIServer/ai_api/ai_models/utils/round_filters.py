
def round_filters(filters, width_coefficient, depth_divisor):
  """Round number of filters based on depth multiplier."""
  multiplier = width_coefficient
  divisor = depth_divisor

  filters *= multiplier
  min_depth = divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)