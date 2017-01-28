###1. Rule of thumb for validation set size
Typically, in Statistics, the model parameters become better if more than 30 examples are corrected from FP to TP. So, in make accuracy figures significant to the first decimal place (i.e. 0.1% valid figure), we usually need more than 30000 examples for validation. In practice, we may not have some many data, and we can use cross-validation to mitigate this problem.

<p align="center">
  <img src ="./images/ValidationSetSize.png" width="600"/>
</p>

###2. Gradient Descent (GD) vs Stochastic Gradient Descent (SGD)
When dealing with large scale data， the compute for GD is expensive, usually 3 times more (why?). We can get estimate of the gradient descent of all data, just choose subset randomly and use their gradient descent. This estimation may be bad, so we need enough randomness and more iterations. While SGD is very fast and simple, so it performs well in practice.

<p align="center">
  <img src ="./images/GD.png" width="600"/>
</p>
<p align="center">
  <img src ="./images/SGD.png" width="600"/>
</p>

Top is GD, bottom is SGD.
