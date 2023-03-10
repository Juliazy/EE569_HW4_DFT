# DFT-loss

## Notes
X and y should have data type double\
X and y should be one dimensional (ie. X.shape = y.shape = (nSamples,))

## Building
Make sure Cython is installed in your environment then
```
python setup.py build_ext --inplace
```
to create .so file 
## Example Usage
```
from _loss import PyLoss

loss = PyLoss()
X = ...
y = ...
split = ...
nSamples = X.shape[0]
loss.calc_we(X, y, split, nSamples, num_class)
```
