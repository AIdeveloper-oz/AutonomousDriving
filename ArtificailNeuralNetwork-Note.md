###1. Artificial Neuro Cell model
We can use a simple proception machine to model the Artificial Neuro Cell approximately like follows,
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/ANNmodel.png "ANN model")


###2. Proception machine
One proception unit can seperate the samples linearly. It can can model three binary functions: AND, OR, NOT, but can not model XOR. For XOR, we need to use two layer proception machine like follows,
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/PreceptronXOR.png "Preceptron XOR")


###3. Proception Training
In order to learn the parameters including weights W and threshold theta, we need to update the them iteratively w.r.t the loss. There are two similar learning rules.

1) **Proception rule:** The following figure shows how to update w (the threshold theta is fixed to zero simplely.). For example, if `y=0` and `y_hat=1`, then the w is too large and we use `(y-y_hat)xi` as the loss decreasing direction. We can see the loss `y-y_hat` is multipied by `xi`, which can decrease the coefficients most related to `xi`.
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/PreceptronTrain.png "Preceptron Train")

2) **Gradient Descent:** 
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/GradientDescent.png "Gradient Descent")

As shown below, the above two methods are similar in formulation. But note that `y_hat` and `a` are different, since `y_hat` is the output and is discreate, while `a` is the activation and is continuous. In general, the **Gradient Descent** is more widely used in practice. The comparasion is as follows,
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/Comparision.png "Comparision")