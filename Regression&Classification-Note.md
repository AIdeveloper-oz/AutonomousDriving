###1. Dfferences between Regression and Classification
**Regression**: the output is continuous, such as real-value.
**Classification**: the output is discreate, such as binary True/False, multiple category.
For example, *Human Age Estimation* can be treated as regression problem if fraction is allowed, i.e. unlimited categories. While, it can also be treated as classification problem if there are only limited categories, like 1,2,3,...,150.

###2. Regression (Function Approximate)
When use square function as the fit error in regression, we can use **calculus**. The minimum of the sum of squares is found by setting the gradient to zero. (cf. https://en.wikipedia.org/wiki/Least_squares). Besides, note that if there are `N` samples in data, then the highest order should be no more then `N-1` (start from `0`).

When to fit linear function `X*W=Y`, we can use `X'X*W=X'*Y` --> `W=(X'X)\(X'*Y)`. It is because that `X` may not be square matrix and may not have good inverse, while `X'X` is square matrix and always have good inverse. Above is to show that the linear fitting has analytical solution.
