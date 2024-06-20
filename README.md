# 23b2231_.LLM

# Machine Learning
## Basics of Machine Learning
#### Machine learning (ML) is a subset of artificial intelligence (AI) that involves the development of algorithms and statistical models enabling computers to perform specific tasks without explicit instructions. Instead, ML systems learn from and make decisions based on data. The primary goal of ML is to allow computers to learn and adapt through experience, improving their performance over time. ML is utilized in various applications, from recommendation systems and image recognition to autonomous vehicles and natural language processing, making it a crucial technology in today's data-driven world.

## Types of Machine Learning
#### Machine learning can be broadly categorized into three types: supervised, unsupervised, and reinforcement. Each type serves different purposes and is chosen based on the nature of the task and the available data. You can just read ahead to know more about each of them.

### Supervised Learning
#### In supervised learning, the algorithm is given a dataset containing inputs and their corresponding correct outputs. The main objective is to learn a mapping from inputs to outputs, which can then be used to predict the output of new, unseen inputs. For example, in a supervised learning task to classify emails as spam or not spam, the algorithm would be trained on a dataset labeled as "spam" or "not spam" and then used to classify new emails.

### Unsupervised Learning
#### Unlike supervised learning, unsupervised learning deals with unlabeled data. The goal is to find hidden patterns or intrinsic structures in the input data. For instance, in customer segmentation, unsupervised learning can group customers based on purchasing behavior without pre-labeled categories. Clustering algorithms, such as k-means, and dimensionality reduction techniques, like PCA, are commonly used in unsupervised learning.

### Reinforcement Learning
#### Reinforcement learning is inspired by behavioral psychology, where an agent learns to behave in an environment, performing actions and receiving rewards. The agent's objective is to maximize its total reward. Unlike supervised and unsupervised learning, reinforcement learning is dynamic and focuses on the sequence of actions rather than static data. Applications include training AI to play games, where the agent improves its strategy by learning from the outcomes of its moves.

# Deep Learning

## Definition of Deep Learning
#### Deep learning is formally defined as a class of machine learning techniques that use multi-layered neural networks to model and understand complex patterns in data. Each layer in the network transforms the input data into a more abstract and composite representation. Deep learning algorithms are capable of supervised, unsupervised, and reinforcement learning tasks. The depth of the model allows it to learn intricate patterns and relationships, making it particularly suited for high-dimensional data.

## Evolution of Deep Learning
#### Deep learning has evolved from simple neural network models to complex architectures like convolutional neural networks (CNNs) and recurrent neural networks (RNNs). The advancements in computational power, availability of large datasets, and improved algorithms have fueled its growth. Early neural networks were limited by computational constraints and insufficient data, but modern GPUs and large-scale data collections have enabled the development of deep learning models that outperform traditional machine learning techniques in various domains.

## Representation Learning
#### Representation learning, or feature learning, involves discovering the representations or features needed for machine learning tasks directly from raw data. Deep learning models automatically learn to extract useful features from data, eliminating the need for manual feature engineering. This process is crucial in applications such as image and speech recognition, where the model learns hierarchical representations, with lower layers capturing basic features like edges and higher layers capturing complex patterns and objects.

## Perceptron and How It Relates to Neurons
#### The perceptron is a simplified model of a biological neuron, used as a fundamental building block in artificial neural networks. It takes multiple input signals, applies weights to them, sums them up, and passes the result through an activation function to produce an output. Just as biological neurons process and transmit information through synaptic connections, perceptrons in neural networks process input data and contribute to decision-making processes.

## Logistic Regression as Neural Network
#### Logistic regression can be viewed as a simple neural network with no hidden layers. It models the probability that a given input belongs to a certain class. The model applies weights to input features, sums them, and passes the result through a sigmoid activation function to produce a probability output. This probabilistic interpretation and use of the sigmoid function make logistic regression a foundational concept for understanding more complex neural networks.

## Multi-layer Perceptron
#### A multi-layer perceptron (MLP) is a type of feedforward artificial neural network with multiple layers of nodes (neurons), including an input layer, one or more hidden layers, and an output layer. Each node in one layer connects to every node in the next layer, with weights assigned to each connection. MLPs use activation functions to introduce non-linearity, allowing them to model complex relationships in data. They are trained using backpropagation to minimize error and improve predictive accuracy.

## Perceptron Training
#### Perceptron training involves adjusting the weights of the model to minimize classification errors on the training data. The learning algorithm updates weights iteratively based on the difference between the predicted and actual outputs. If the prediction is correct, no changes are made. If incorrect, the weights are adjusted to reduce future errors. This process continues until the perceptron correctly classifies all training examples or reaches a maximum number of iterations.

## Multi-layer Perceptron Training
#### Training a multi-layer perceptron involves forward propagation to compute predictions and backpropagation to adjust weights based on errors. During forward propagation, input data is passed through the network, layer by layer, to generate output predictions. In backpropagation, the error is calculated by comparing the predicted output with the actual output, and the gradients of the error with respect to each weight are computed and used to update the weights. This process iteratively reduces the error and improves model performance.

## Backpropagation Training
#### Backpropagation is a key algorithm for training neural networks, particularly multi-layer perceptrons. It involves propagating the error backward from the output layer to the input layer through the network, updating the weights of each connection to minimize the overall error. The algorithm uses the chain rule of calculus to compute gradients for each layer, enabling efficient weight adjustment. Backpropagation has significantly advanced the development and application of deep learning models.

## Activation Functions and Derivation
#### Activation functions introduce non-linearity into neural networks, allowing them to model complex data. Common activation functions include the sigmoid, hyperbolic tangent (tanh), and rectified linear unit (ReLU). The sigmoid function outputs values between 0 and 1, making it useful for binary classification. Tanh outputs values between -1 and 1, often used in hidden layers. ReLU outputs the input directly if positive, otherwise zero, which helps mitigate the vanishing gradient problem. The derivatives of these functions are essential for the backpropagation algorithm.





