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

#### Feature scaling

Feature scaling is the translation of data such that the ranges of values of each feature has a reasonably similar scale. The contour plot of the cost function will be more circular, rather than elliptic. This speeds up the gradient descent, i.e. fewer iterations are needed to arrive at the minimum of the cost function.

One method to do so is by mean normalisation. First, find the average, maximum and minimum of all values per feature. Or, the maximum and minimum that data can reasonably take on as values. Then, for each data point i and feature j, the new data point ij is calculated by (x_ij - avg_j)/(max_j - min_j). Now, all values should for all features of all data points should be centered around zero in the range from -1 to 1.

Another method is the Z-score normalistion. This is done similarly. Each data point x_ij is transformed as follows: (x_ij - avg_j) / std_dev_j, where std_dev_j is the standard deviation of all data points i of feature j.

### Classification

Classification has a discrete, finite set of possible output values. It predicts to what output category or class the input data points most likely belong.

The target variable y can be one of a number of values. If there are two possible values, it is called a binary classification problem.

Linear regression is a bad choice to fit classification data. The least squares may give too much weight to outliers, which means data points closer to the decision boundary may unnecessarily be predicted to have the wrong class. Another reason is that the values a line can take on includes the entire real number line, where everything outside the range from zero to one has little meaning.

Logistic regression is a good model to predict a binary classification problem. We can use the sigmoid function, which is a logistic function that outputs between 0 and 1.

g(z) = 1/(1+e^-z)

When z is very large, g(z) approaches 1. When z is a large negative number, g(z) approaches 0. When z = 0, g(z) = 0.5.

We combine the linear regression with the sigmoid function to find logistic regression.

f_wb(x) = g(w*x+b) = 1/(1+e^-(w*x+b))

The decision boundary is a threshold number. If the output of the logistic regression is greater than the boundary, the estimate of the target will be 1. With the logistic regression model, it turns out that the decision boundary is the line that crosses the y-axis.

The cost function is still the average loss, but the loss is calculated differently compared to that of linear regression.

The loss function L is conditional on the value of the target (sub and super scripts omitted for readability):
L(f, y) = {
    -log(f(x)) if y = 1
    -log(1- f(x)) if y = 0
}

### Regularisation

Regulaisation addresses the problem of overfitting. It lets you keep the high amount of features, but reduces their impact. It offers a middle ground between keeping all features and removing features altogether.

In practice, regularisation is implemented by adding a term to the cost function: lambda/(2*m) * sum_over_j(w_j^2). That is, sum all feature weights squared - to get a sense of how much weight is put to all features, and divide by the amount of training examples. Lambda is some positive regulasation parameter that can be tuned. We divide by m to make the same regularisation parameter work for more data sets.

### Neural networks

Neural networks are a modeling choice for supervised learning, because the data set that the algorithm is trained on uses labeled data. The design uses layers of activation functions to let the learning algorithm figure out complex connections between the features of the training examples and the output variable.

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
