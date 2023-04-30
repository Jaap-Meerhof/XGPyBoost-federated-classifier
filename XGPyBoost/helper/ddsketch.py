
from customddsketch import DDSketch

import numpy as np
sketch = DDSketch(0.01)
# sketch = ds

values = np.random.normal(size=500)
for v in values:
    sketch.add(v)

quantiles = [sketch.get_quantile_value(q) for q in [0.5, 0.75, 0.9, 1]]

another_sketch = DDSketch(0.02)
other_values = np.random.normal(size=500)
for v in other_values:
  another_sketch.add(v)
sketch.merge(another_sketch)

quantiles = [sketch.get_quantile_value(q) for q in [0.5, 0.75, 0.9, 1]]

pass