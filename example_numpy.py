import numpy as np
from modelpicker import modelpicker


predictions = np.array([1,2,3])

labelset = np.array([1,2, 3])

budget = 3

(zt, ut) = modelpicker(predictions,labelset,budget)
print(zt)