# EE569_HW4_DFT
Source code for Discriminant-Feature-Test, include MATLAB, Python and Cython version

## MATLAB
credit to Yao Zhu

`py_mnist.mat` example file to run `DFT_utils.m` directly

## Python
credit to Yijing Yang

`DFT_feat_util.py` python file for DFT. Example usage is in its main function. 

If want to speed up, Cython version of DFT loss calculation is in `Cython_DFT_loss` folder. Please refer to its README.md for the usage. 

## C++
credit to Joseph Lin

`Cython_DFT_loss`->`loss.cpp` provide C++ code to calculate weighted entropy loss and squared error loss, but other steps, such as partitioning 1D feature space, calculate loss for all feature dimension are not included. 
