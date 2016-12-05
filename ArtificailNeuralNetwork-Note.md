###1. Artificial Neuro Cell model
We can use a simple proception machine to model the Artificial Neuro Cell approximately like follows,
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/ANNmodel.png "ANN model")


###2. Proception machine
One proception unit can seperate the samples linearly. It can can model three binary functions: AND, OR, NOT, but can not model XOR. For XOR, we need to use two layer proception machine like follows,
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/PreceptronXOR.png "Preceptron XOR")


###3. Proception Training
In order to learn the parameters including weights W and threshold theta, we need to update the them iteratively with the loss. The following figure shows how to update. For example, if `y=0` and `y_hat=1`, then the w is too large and we use `(y-y_hat)xi` as the loss decreasing direction. We can see the loss `y-y_hat` is multipied by `xi`, which can decrease the coefficients most related to `xi`.
P.S. The threshold theta is fixed to zero simplely.
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/PreceptronTrain.png "Preceptron Train")