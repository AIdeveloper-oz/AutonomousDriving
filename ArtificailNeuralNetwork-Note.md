###1. Artificial Neuro Cell model
We can use a simple proception machine to model the Artificial Neuro Cell approximately like follows,
<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/ANNmodel.png" width="650"/>
</p>


###2. Proception machine
One proception unit can seperate the samples linearly. It can can model three binary functions: AND, OR, NOT, but can not model XOR. For XOR, we need to use two layer proception machine like follows,
<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/PreceptronXOR.png" width="650"/>
</p>


###3. Proception Training
In order to learn the parameters including weights W and threshold theta, we need to update the them iteratively w.r.t the loss. There are two similar learning rules.

1) **Proception rule:** The following figure shows how to update w (the threshold theta is fixed to zero simplely.). For example, if `y=0` and `y_hat=1`, then the w is too large and we use `(y-y_hat)xi` as the loss decreasing direction. We can see the loss `y-y_hat` is multipied by `xi`, which can decrease the coefficients most related to `xi`.
<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/PreceptronTrain.png" width="650"/>
</p>

2) **Gradient Descent:** 
<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/GradientDescent.png" width="650"/>
</p>

As shown below, the above two methods are similar in formulation. But note that `y_hat` and `a` are different, since `y_hat` is the output and is discreate, while `a` is the activation and is continuous. In general, the **Gradient Descent** is more widely used in practice. The comparasion is as follows,
<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/Comparision.png" width="650"/>
</p>


###4. Sigmoid Unit
In the above figure, we see that the threshold is really pointy spot, so the **Gradient Descent** is non-differentiable. While, we can approximate the curve with a s-like function to soft the threshold, namely **Sigmoid Function**. Sigmoid is differentiable and its derivative itself has a very beautiful form as shown below. Sigmoid is analogous to preceptron, but not a preceptron which use hard threshold. And sigmoid do not guarantee to converge in finite time like preceptron.
<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/Sigmoid.png" width="650"/>
</p>


###5. Neural Network
Sigmoid unit is a basis unit in neural network (recently tanh, relu are more popular) which make the whole network differentiable. So that the back propagation can be used to optimize the network parameters. But note that, since there are many layers and each layer have many sigmoid units, then the error space will have many local minimum. So the onpimization can only guarantee to find local optimal.
<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/NeuralNetwork.png" width="650"/>
</p>


###6. Optimization
Besides the classical **Gradient Descent**, there are some other kinds of advanced methods to find better local minimum.

1) Momentum, widely used in practice (see figure bellow, image a ball falls down from the mountain, the small local pit can not hold the ball because of large momentum, so the ball can fall into a lower pit)

2) High order derivatives, also widely used in recent NN optimization method such as *Adam*

3) Randomized optimization

4) Penalty for "complexity", widely used in practice. There are three main reason lead to complexity: **more nodes**, **more layers**, **large numbers**. These reasons will make the model learn much about the noise, i.e. overfitting. Specially, the fore two reasons are easy to understand (just like high order in regression) and they are usually hyperparameters (recent work [Convolutional Neural Fabrics](https://github.com/shreyassaxena/convolutional-neural-fabrics) shows that NN structure can also be learned to some extend). While the **large numbers** means that the value of weights `W` is large. `W` is always learned from the data, researcher always constrain it through adding penalty item in loss function.

<p align="center">
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/Optimization.png" width="650"/>
</p>


###7. Restriction Bias & Preference Bias [link](http://jmvidal.cse.sc.edu/talks/decisiontrees/restandprefbias.html)

**Restriction Bias** is an inductive bias where the set of hypothesis considered is restricted to a smaller set. E.g. Candidate-Elimination searches an incomplete hypothesis space (it can only represent some hypothesis) but does so completely. 

As shown below, for proception unit, the hypothesis is restricted to half space. And from the hypothesis' view, nework of threshold-like unit (such as proception and sigmoid unit) can only model Boolean function. While, if we have enouth hidden units, each hidden unit can worry about one little patch of the function that the network needs to model, then the patches get set at hidden layers and get stitched at the output layer. So that the network with one hidden layer can model continuous function. And network with two hidden layers can model arbitrary function even itis discontinuous. So if the network is complex enough, we can model anything. But note that, in practise, we give the bounded number of layers and nodes, so the fixed network can only capture whatever it can capture. Another difference between NN and other machine learning methods (such as decision tree and svm) is that the testing error will grow if you train the model with too much iterations. The explaination is the **large numbers** mentioned above.
<p align="center"> 
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/RestrictionBias.png" width="650"/>
</p>

**Preference Bias** is an inductive bias where some hypothesis are preferred over others. E.g. ID3 searches a complete hypothesis space but does so incompletely since once it finds a good hypothesis it stops (cannot find others).

Two important related problem:

1) How to choose model structure? Using the principle of occam's razor.

2) How to initilize weights? Using **Small Random Values** (also can be explained with occam's razor). **Small** can prevent **large numbers**, and **Random** can help to skip bad local minimum.
<p align="center"> 
  <img src ="https://github.com/charliememory/AutonomousDriving/blob/master/images/PreferenceBias.png" width="650"/>
</p>