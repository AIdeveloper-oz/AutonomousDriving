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


###4. Sigmoid Unit
In the above figure, we see that the threshold is really pointy spot, so the **Gradient Descent** is non-differentiable. While, we can approximate the curve with a s-like function to soft the threshold, namely **Sigmoid Function**. Sigmoid is differentiable and its derivative itself has a very beautiful form as shown below. Sigmoid is analogous to preceptron, but not a preceptron which use hard threshold. And sigmoid do not guarantee to converge in finite time like preceptron.
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/Sigmoid.png "Sigmoid")


###5. Neural Network
Sigmoid unit is a basis unit in neural network (recently tanh, relu are more popular) which make the whole network differentiable. So that the back propagation can be used to optimize the network parameters. But note that, since there are many layers and each layer have many sigmoid units, then the error space will have many local minimum. So the onpimization can only guarantee to find local optimal.
![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/NeuralNetwork.png "Neural Network")


###6. Optimization
Besides the classical **Gradient Descent**, there are some other kinds of advanced methods to find better local minimum.

1) Momentum, widely used in practice (see figure bellow, image a ball falls down from the mountain, the small local pit can not hold the ball because of large momentum, so the ball can fall into a lower pit)

2) High order derivatives, also widely used in recent NN optimization method such as *Adam*

3) Randomized optimization

4) Penalty for "complexity", widely used in practice. There are three main reason lead to complexity: **more nodes**, **more layers**, **large numbers**. These reasons will make the model learn much about the noise, i.e. overfitting. Specially, the fore two reasons are easy to understand (just like high order in regression) and they are usually hyperparameters (recent work [Convolutional Neural Fabrics](https://github.com/shreyassaxena/convolutional-neural-fabrics) shows that NN structure can also be learned to some extend). While the **large numbers** means that the value of weights `W` is large. `W` is always learned from the data, researcher always constrain it through adding penalty item in loss function.

![alt text](https://github.com/charliememory/AutonomousDriving/blob/master/images/Optimization.png "Optimization")


###7. Restriction Bias 
Besides the classical **Gradient Descent**, there are some other kinds of advanced methods to find better local minimum.
<div style="text-align:center"><img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/RestrictionBias.png" width="700"/></div>