# Rust AI

Machine learning is the phenomenon that a machine can learn without being explicitly programmed what it's doing.

## Terminology

Training set:
The data used to train the model.

### Notation

x is the input variable, or feature.

y is the output variable, or target variable.

m is the number of training examples, or data points.

(x, y) is a single training example.

To index a data point of the training set, we use the label of the data point, between brackets, in super-script notation above the points in the set.

Cost function:
The function that quantifies to what degree the model makes accurate predictions. Lower is better.

## Supervised learning

Supervised learning is a machine learning algorithm that is trained on correctly labeled data. It predicts the label.

There are two main forms of supervised learning:
- Regression
- Classification

The training set contains features and targets. The machine learning algorithm outputs a function that maps the input data `x` to the predicted or estimated output, often written as `y-hat`.

How to represent the function f, is the question what model you choose. 

The function is an expression that maps the input to the output through some calculation, based on the inputs in relation to a set of parameters.

The choice of machine learning algorithm is to find an efficient algorithm that tunes the parameters in the model that reduce the cost function.


### Regression 

Regression is a learning algorithm where the output of the prediction is an ordered value, like a (continuous) number. 

Models
- Univariate linear regression: a straight line fitting the training set with one input variable: `f(x) = w*x + b`
- Multiple linear regression: a straight line fitting the training set with a vector of input variables: `f(x) = theta_0 + theta_1*x1 + theta_2*x2 + ... + theta_n*xn`, where x is the vector of inputs and theta is the vector of feature weights.
N.B.: Multivariate regression is actually something else, but that's out of scope for this project.

#### Example

Predict the price of a house, based on its size. We have the data with for each data point the size and price of a house. 

Since the price is a number and the data is labeled with the output we care about, regression is a good method for this case.

Input: size in square feet
Output: price in $1000s

For univariate linear regression, the cost function is the average of the squared errors, with the error being the enumration of the differences between the predicted and actual data points.

The goal is to minimize the cost function J, w.r.t. its paramters w, b. So J is a function of w and b, and its minimum is the solution to finding the parameters that give the chosen model the best fit to the data.

#### Multiple linear regression

For multiple linear regression, we generalise the w and b coefficients to a vector theta, and use the vector x for the data of the inputs. For easier and more understandable implementation, we prepend a 1.0 floating point number to the data in x, so the first value in theta refers to the constant, or y-intercept of the line that fits the data.

### Classification

Classification has a discrete, finite set of possible output values. It predicts to what output category or class the input data points most likely belong.

## Unsupervised learning

Useful when given data has no labels. The goal is to find some pattern, structure in the data.

### Clustering

Grouping together different similar data points.

### Anomaly detection

Find rare unusual data points

### Dimensionality reduction

Compress data using fewer dimensions to increase efficiency of the algorithm. The aim is to retain most of the information in the data set.

## Algorithms

### Gradient descent

Gradient descent is an algorithm that serves to find the parameters that iteratively minimize the cost function.

w = w - alpha * d/(dw)J(w,b)
and 
b = b - alpha * d/(db)J(w,b)

alpha is the learning rate (how big of a step you take downhill). A larger alpha speeds up the process, but it may make the algorithm digerge rather than converge on the minimum. A smaller alpha can make the computation expensive.

We simultaneously update all parameters. So that means to calculate the partial derivatives and then update both. 

Start out with some parameters. Keep changing the parameters until we have found a minimum. Mathematically, we use the derivative to find the steepest descent.

If you're already at or near a local minimum, an iteration of gradient descent will not or barely change w.

Batch gradient descent specifies that each step of the descent uses all training data, rather than subsets.
