###1. Linear model's advantages & liminations
**Advantages**
1) Efficient computation, expecially using GPUs.
2) Stability: a) small changes of inputs result in small changes of outputs; b) for `y=wx+b`, `dy/dx=w` and 'dy/dx=w', both of which are constant (nothing can be more stable than constant).
**Liminations**
Linear model can only represente linear relationships which is too simple for many problems. So we need to introduce non-linearities between linear matrix multiplication.

<p align="center">
  <img src ="./images/Linear.png" width="700"/>
</p>

###2. Non-linearity
A simple but efficient way to introduce non-linearity is insert ReLU units into our pipeline as a layer. ReLU is simple to compute and has simple gradient. 

<p align="center">
  <img src ="./images/ReLU.png" width="700"/>
</p>

BTW, the first layer effectively consists of the set of weights and biases applied to X and passed through ReLUs. The output of this layer is fed to the next one, but is not observable outside the network, hence it is known as a hidden layer.

###3. Why deeper?
1) We can typically get much more performance with fewer parameters by going deeper rather than wider.
2) Deep model can capture hierarchical structure of objects, i.e. low layers capture simple edge feature, middle layers capture object-part-feature, high layers capture object-like feature.

<p align="center">
  <img src ="./images/DeeperOrWider.png" width="700"/>
</p>
