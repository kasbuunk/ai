# Rust AI

_This project implements various machine learning algorithms with the Rust programming language. It showcases how easy it can be to create useful algorithms with few dependencies and safe, high-level functional programming principles. Further optimisation can be done to use crates to leverage parallel computations, such as the rayon and ndarray crates, which should be easy to integrate. Moreover, the linfa framework is promising to be a fruitful ecosystem to build real ML programs in._

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

The final step is to find the vector theta such that the cost function is minimised. That is achieved by applying the gradient descent algorithm.

#### Multiclass classification

A classification problem where there are more than two potential classes the output target should be able to predict.

Softmax regression, rather than logistic regression, is an algorithm that can predict multiclass classification problems. Use the SparseCategoryCrossEntropy function as the activation function, because only one of the potential categories should be predicted to be the outcome.

#### Multilabel classification

A classification problem where where a subset of the potential categories is predicted. Since each of the last activation layer is separately a binary outcome, we can use the sigmoid activation function.

### Regularisation

Regulaisation addresses the problem of overfitting. It lets you keep the high amount of features, but reduces their impact. It offers a middle ground between keeping all features and removing features altogether.

In practice, regularisation is implemented by adding a term to the cost function: lambda/(2*m) * sum_over_j(w_j^2). That is, sum all feature weights squared - to get a sense of how much weight is put to all features, and divide by the amount of training examples. Lambda is some positive regulasation parameter that can be tuned. We divide by m to make the same regularisation parameter work for more data sets.

### Neural networks

Neural networks are a modeling choice for supervised learning, because the data set that the algorithm is trained on uses labeled data. The design uses layers of activation functions to let the learning algorithm figure out complex connections between the features of the training examples and the output variable.

First, we define the model. The model of a neural network, as with the other supervised learning algorithms, is the function that maps the input features to an estimated output variable. An example of a neural network model is a sequence of activation vectors, starting with the input features, where each activation vector is the product of multiplying the previous activation vector by the weight vector of the current activation layer to forward propagate the estimation of, in the end, a single value that can be interpreted as the estimation. Or, even this can be a vector, if multiple output values are defined.

Then, we define a loss function, such as the binary cross entropy or mean squared error function.

Finally, we 'fit' or train the model on the data, resulting in the feature weight matrices. The training is done with the backpropagation algorithm.

We can still use neural networks to do bothh classification and regression. Whether you should choose a neural network for your model depends more on the type of pattern in the data, than the type of output you fit the data on.

#### Layers

A typical choice for the layer is the Dense layer, which simply maps all neurons from the previous layer to all neurons in the current Dense layer.

Another choice is a Convolutional layer. This lets each neuron look only at a subset of the previous layer's outputs. It speeds up the computation and is less prone to overfitting, or needs fewer training examples. Each neuron can look at an overlapping window of the input vector. The next layer may also be a convolutional layer, i.e. its neurons each also only take a specific subset of the previous layer as input.

#### When to use?

Neural networks work well on structured and unstructured data. A large neural network may be slower than a decison tree. It can work with transfer learning. Multiple models can be built or strung together.

### Decision trees

A decision tree is a data structure that represents how an algorithm may decide to predict a data point fits a particular category, by walking the tree across decision nodes until a leaf node is found. The label of the leaf node represents the prediction.

How to choose what feature to split on at each node? Maximise purity: choose the feature that best separates the data per subbranch.

When do you stop splitting? When a node is 100% one class, is an easy way if possible. Another way is to set a maximum depth and disallow splitting further if the tree length meets the limit. Another way is if the improvements in purity are below some threshold, indicating further refinement is not rewarding the performance. Also, when the number of examples in a node is below some threshold, it may indicate the algorithm would not generalise well to new data.

#### Entropy

Entropy is the measure of impurity. H(p1) = -p1 * log2(p1) - p0*log2(p0), with p_i the fraction of the population being in category i.

Choosing the right split can be done by picking the split with the lowest average entropy. We compute the reduction in average (or expected) entropy, which comes down to the highest information gain.

The same principle also applies to splitting on a continuous variable. As long as the output variable is binary, you can pick the threshold that maximises information gain.

#### Regression trees

We now not just use the information gain, but take the reduction in variance into account as the weight per split option of the respective decrease in weighted variance/entropy.

#### Tree ensembles

Multiple decision trees are build concurrently, because although maximising information gain may at first be the most reasonable choice, later anothor choice may have been better. Allowing multiple trees gives the opportunity to make the choice later, when more is known what the long-term effects are of a particular choice. This improves the robustness of the algorithm. To predict the outcome given the data point, the algorithm can weigh the different predictions that each decision tree makes.

#### Random forests

Also called a bagged decision tree, a random forest is a tree ensemble that uses sampling with replacement, i.e. take a sample from the original data set that may have data points repeating. Choose a number of trees, B, and for each tree, pick a sample of training examples. In addition, the random forest also picks a random amount of features, n, like the square root of n.

On prediction, the trees all vote on a particular outcome, where a typical decision boundary is to pick the outcome with the most votes.

#### XGBoost

EXtreme Gradient Boosting, or XGBoost, is similar to the bagged decision tree, but it emphasises to use training examples that are misclassified with a higher probability. This, in order to boost the performance for data points that it performs most poorly on.

#### When to use?

Decision trees work well on tabular or structured data. It does not work well on unstructured data, like images, audio, video or text. It's very fast and in small cases it may be more human interpretable than a neural network.

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

### Adam

Adaptive Moment estimation, or Adam, is essentially the same as gradient descent, but it automatically adjusts its learning rate to move faster to an optimum.

## Diagnostics

### Evaluating a model

To evaluate the model, you can split the data set into a training and test set, e.g. 70 and 30%, respectively. We first train the parameters by minimising the cost function. Then, we compute the test error (without regularisation) to know how well the model generalises to new data. Or in other words, how much the model overfits the training data. If the test error is quite high compared to the train error, it's likely the model suffers from overfitting. Remember to compute the training error a second time at the end, also without the regularisation parameter, which was only necessary for training purposes. For classification problems, it may be more indicative to review and compare the fraction of the test and train set that the algorithm has misclassified.

We can also choose a model programmatically. We can predefine a set of possible models, like different order of polynomials or different compositions of layers in a neural network. But now we cannot simply use the test error to estimate the cost of the model. We actually need an extra set of isolated training examples, which we call the cross-validation set. It's also called the validation, development or dev set. We ten use the cross-validation set to carry out model selection, e.g. to select the polynomial degree. We check what model does best on the validation set and use that model to review the test cost, to have a fair, unbiased view of the perfomance of how your model generalises.

A baseline error level can help serve as a benchmark to evaluate the performance of the model on the train, cv and test data. Examples are: human level performance, a competing algorithm or just a prior guess.

#### High bias 

High bias means the training error is high. The model underfits the training data. It will likely also underfit the cv data. More training data is unlikely to help. But you can try to find a better model. 

Try to:
- Decrease the regularisation parameter lambda.
- Enrich the data by finding additional features.
- Choose a more complex model, such as (more layers in) a neural network or adding polynomial features.

#### High variance

High variance means the model overfits the training data. It may help to collect more data. And you can try a simpler model, if the training error also remains low, i.e. a simpler model does not cause underfitting.

Try to:
- Get more training examples.
- Increase the regularisation parameter lambda.
- Pick a smaller set of features.
- Choose a simpler model.

#### Learning rate

You can plot the train and cv error as a function of the amount of training examples. This indicates whether adding additional data is helpful for improving the performance.

#### Neural networks

Large neural networks can often achieve low bias. If it does not yet, make it a bigger network and at some time the bias must become low. If it doet not do well on the cv set yet, more data is the solution. Of course, gathering data and bigger networks both come with their costs, if feasible at all. 

In practice, larger neural networks with regularisation often do a lot better than simpler networks without regularisation. It hardly ever hurts to make the network bigger, as long as regularisation is chosen appropriately.

## Machine learning development

Steps involve a loop: First, choose an architecture. That includes the model, what the type of the output should be represented with, the algorithm to minimise the cost function, regularisation parameter, etc. Potentially this step includes engineering the existing data or finding additional data.

Next, train the model.

Finally, diagnose the performance of the model. Based on the results, go back to the architecture.

### Error analysis

A way to find misclassifications is to manually go through the data that a sample of the wrong output is predicted. If you observe a pattern, this may inspire whether collecting additional examples or features of the existing data may help.

### Adding data

If error analysis has shown a specific subset has poor perforamnce, adding data of a particular type may be more fruitful than adding data on any target type.

#### Data augmentation

In augmentation, you apply distortians or transformations to existing training examples to create extra data that has the same output labels. Random or meaningless noise is usually not helpful for the performance.

#### Data synthesis

Artificial data synthesis is creation of fake data. It can be a lot of work to generate this, but can be a lot cheaper than gathering the actual data. It can be just as effective as real data.

#### Transfer learning

Transfer learning is useful for ML problems with very few data. It leverages existing, pre-trained models. Use the parameters, replacing the last output layer with your own output units. 

The input data needs to match the input of the pre-trained model. Some preprocessing, or even a layer in-between, may be necessary as some sort of translation layer.

##### Suprevised pre-training

Next, an option is to only train the output layer parameters and keep all previous layers as they are. THere is a large amount of pre-trained models that can be found online to leverage this.

##### Fine-tuning

Another option is to train all parameters, effectively fine-tuning the pretrained model.

### Skewed data

If some outcomes are rare, or the distribution of numeric values is skewed in the sense that some ranges of values occur significantly more often, the data is skewed. This may bias the algorithm to just predict the more common case and the cost function will reward it anyway.

We can use precision and recall as metrics to measure this. The precision is the fraction of true positives divided by the predicted positives. Of all cases where we predict y=1, what fraction of those predictions were accurate? The recall is the fraction of the true positives divided by the actual positives. Of all data points where y=1, how many were correctly predicted as such?

To balance between precision and recall, we can set the threshold higher or lower, in case of binary classes as outcome.

If a single metric is desired, use the F1 score. The average between the two can be very bad at balancing if the data is skewed. F1 = 2*P*R / (P+R). This is also called the Harmonic Mean.
