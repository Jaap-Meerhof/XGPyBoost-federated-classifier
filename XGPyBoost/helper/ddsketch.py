
from customddsketch import DDSketch
from customddsketch import LogCollapsingLowestDenseDDSketch
from xgboost import QuantileDMatrix
from xgboost import DMatrix

import numpy as np
sketch = DDSketch(0.01)
# sketch = ds

values = np.random.normal(size=500)
weights = np.random.uniform(low=0.0, high=1.0,size=500)
for i in range(len(values)):
    sketch.add(values[i], weights[i])
mean = values.mean()
quantiles = [sketch.get_quantile_value(q) for q in [0.5, 0.75, 0.9, 1]]

another_sketch = DDSketch(0.02)
other_values = np.random.normal(size=500)
for v in other_values:
  another_sketch.add(v)
sketch.merge(another_sketch)

quantiles = [sketch.get_quantile_value(q) for q in [0.5, 0.75, 0.9, 1]]

pass