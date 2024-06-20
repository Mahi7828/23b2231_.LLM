# 23b2231_.LLM

<p style="font-family: Times New Roman;">
  
# Machine Learning
  
## Basics of Machine Learning

Machine learning (ML) is a subset of artificial intelligence (AI) that involves the development of algorithms and statistical models enabling computers to perform specific tasks without explicit instructions. Instead, ML systems learn from and make decisions based on data. The primary goal of ML is to allow computers to learn and adapt through experience, improving their performance over time. ML is utilized in various applications, from recommendation systems and image recognition to autonomous vehicles and natural language processing, making it a crucial technology in today's data-driven world.

## Types of Machine Learning
Machine learning can be broadly categorized into three types: supervised, unsupervised, and reinforcement. Each type serves different purposes and is chosen based on the nature of the task and the available data. You can just read ahead to know more about each of them.

### Supervised Learning
In supervised learning, the algorithm is given a dataset containing inputs and their corresponding correct outputs. The main objective is to learn a mapping from inputs to outputs, which can then be used to predict the output of new, unseen inputs. For example, in a supervised learning task to classify emails as spam or not spam, the algorithm would be trained on a dataset labeled as "spam" or "not spam" and then used to classify new emails.

### Unsupervised Learning
Unlike supervised learning, unsupervised learning deals with unlabeled data. The goal is to find hidden patterns or intrinsic structures in the input data. For instance, in customer segmentation, unsupervised learning can group customers based on purchasing behavior without pre-labeled categories. Clustering algorithms, such as k-means, and dimensionality reduction techniques, like PCA, are commonly used in unsupervised learning.

### Reinforcement Learning
Reinforcement learning is inspired by behavioral psychology, where an agent learns to behave in an environment, performing actions and receiving rewards. The agent's objective is to maximize its total reward. Unlike supervised and unsupervised learning, reinforcement learning is dynamic and focuses on the sequence of actions rather than static data. Applications include training AI to play games, where the agent improves its strategy by learning from the outcomes of its moves.

# Deep Learning

## Definition of Deep Learning
Deep learning is formally defined as a class of machine learning techniques that use multi-layered neural networks to model and understand complex patterns in data. Each layer in the network transforms the input data into a more abstract and composite representation. Deep learning algorithms are capable of supervised, unsupervised, and reinforcement learning tasks. The depth of the model allows it to learn intricate patterns and relationships, making it particularly suited for high-dimensional data.

## Evolution of Deep Learning
Deep learning has evolved from simple neural network models to complex architectures like convolutional neural networks (CNNs) and recurrent neural networks (RNNs). The advancements in computational power, availability of large datasets, and improved algorithms have fueled its growth. Early neural networks were limited by computational constraints and insufficient data, but modern GPUs and large-scale data collections have enabled the development of deep learning models that outperform traditional machine learning techniques in various domains.

## Representation Learning
Representation learning, or feature learning, involves discovering the representations or features needed for machine learning tasks directly from raw data. Deep learning models automatically learn to extract useful features from data, eliminating the need for manual feature engineering. This process is crucial in applications such as image and speech recognition, where the model learns hierarchical representations, with lower layers capturing basic features like edges and higher layers capturing complex patterns and objects.

## Perceptron and How It Relates to Neurons
The perceptron is a simplified model of a biological neuron, used as a fundamental building block in artificial neural networks. It takes multiple input signals, applies weights to them, sums them up, and passes the result through an activation function to produce an output. Just as biological neurons process and transmit information through synaptic connections, perceptrons in neural networks process input data and contribute to decision-making processes.

## Logistic Regression as Neural Network
Logistic regression can be viewed as a simple neural network with no hidden layers. It models the probability that a given input belongs to a certain class. The model applies weights to input features, sums them, and passes the result through a sigmoid activation function to produce a probability output. This probabilistic interpretation and use of the sigmoid function make logistic regression a foundational concept for understanding more complex neural networks.

## Multi-layer Perceptron
A multi-layer perceptron (MLP) is a type of feedforward artificial neural network with multiple layers of nodes (neurons), including an input layer, one or more hidden layers, and an output layer. Each node in one layer connects to every node in the next layer, with weights assigned to each connection. MLPs use activation functions to introduce non-linearity, allowing them to model complex relationships in data. They are trained using backpropagation to minimize error and improve predictive accuracy.

## Perceptron Training
Perceptron training involves adjusting the weights of the model to minimize classification errors on the training data. The learning algorithm updates weights iteratively based on the difference between the predicted and actual outputs. If the prediction is correct, no changes are made. If incorrect, the weights are adjusted to reduce future errors. This process continues until the perceptron correctly classifies all training examples or reaches a maximum number of iterations.

## Multi-layer Perceptron Training
Training a multi-layer perceptron involves forward propagation to compute predictions and backpropagation to adjust weights based on errors. During forward propagation, input data is passed through the network, layer by layer, to generate output predictions. In backpropagation, the error is calculated by comparing the predicted output with the actual output, and the gradients of the error with respect to each weight are computed and used to update the weights. This process iteratively reduces the error and improves model performance.

## Backpropagation Training
Backpropagation is a key algorithm for training neural networks, particularly multi-layer perceptrons. It involves propagating the error backward from the output layer to the input layer through the network, updating the weights of each connection to minimize the overall error. The algorithm uses the chain rule of calculus to compute gradients for each layer, enabling efficient weight adjustment. Backpropagation has significantly advanced the development and application of deep learning models.

## Activation Functions and Derivation
Activation functions introduce non-linearity into neural networks, allowing them to model complex data. Common activation functions include the sigmoid, hyperbolic tangent (tanh), and rectified linear unit (ReLU). The sigmoid function outputs values between 0 and 1, making it useful for binary classification. Tanh outputs values between -1 and 1, often used in hidden layers. ReLU outputs the input directly if positive, otherwise zero, which helps mitigate the vanishing gradient problem. The derivatives of these functions are essential for the backpropagation algorithm.



</p>

# Natual Language Learning

## What’s NLTK:

The Natural Language Toolkit (NLTK) is a comprehensive library in Python designed for working with human language data (text). It provides easy-to-use interfaces to over 50 corpora and lexical resources, including WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

## Tokenization:
Tokenization in the context of deep learning and NLTK involves breaking down a text into smaller units called tokens, which are typically words or subwords. In deep learning, tokenization is a crucial preprocessing step that converts raw text into a format suitable for neural networks to process.
In deep learning frameworks like TensorFlow or PyTorch, tokenization is often implemented using libraries such as NLTK. NLTK provides various tokenizers that can handle different types of tokenization tasks, such as word tokenization, sentence tokenization, or even tokenization based on specific rules or patterns.

### PunkT library:
The PunkT library, part of the NLTK (Natural Language Toolkit), is specifically designed for sentence tokenization in natural language processing tasks. Sentence tokenization involves splitting a text into individual sentences. The PunkT algorithm uses unsupervised learning techniques to learn parameters for sentence boundary detection from large corpora of text.

### The TreebankWordTokenizer:
It is an NLTK tokenizer that follows the Penn Treebank corpus conventions. It splits text into individual words and punctuation, handling contractions and hyphenated words consistently. It’s commonly used for NLP tasks that require standard tokenization rules, such as preparing data for models trained on Treebank-style datasets. This tokenizer is integrated into NLTK, making it easily accessible for text processing in Python.

## STEMMING:
Stemming is a text normalization process in natural language processing (NLP) that reduces words to their base or root form. The goal of stemming is to strip affixes (prefixes, suffixes, infixes, and circumfixes) from words to create a common base form. This helps in reducing the number of unique words in a dataset, thereby simplifying text processing and analysis.

### Porter Stemmer:
The Porter Stemmer is a widely used algorithm in natural language processing for reducing words to their root form. It uses a series of heuristic rules to remove common suffixes from English words. This helps in normalizing words with the same base meaning for text analysis tasks. For example, "running," "runner," and "ran" are all reduced to "run." The Porter Stemmer is known for its simplicity and effectiveness, making it a popular choice for tasks like search indexing and text mining.

### Regexp Stemmer:
 The RegexpStemmer is a customizable stemming tool in natural language processing that uses user-defined regular expressions to strip affixes from words. Unlike standard stemmers, it allows for flexible and specific pattern matching to remove suffixes or prefixes, making it adaptable to various languages and specialized vocabularies. It is particularly useful for domain-specific text processing where conventional stemming rules may not apply.

### SnowBall stemmer:
The Snowball Stemmer, is a robust algorithm for reducing words to their base or root form, part of the Snowball framework. It supports multiple languages, including English, French, German, Spanish, and more, making it highly versatile. Compared to the original Porter Stemmer, the Snowball Stemmer is more efficient in processing speed and memory usage and offers greater adaptability through the Snowball language, which allows for customization of stemming rules.



## Lammetization:
Lemmatization in NLTK (Natural Language Toolkit) is a process of reducing words to their base or dictionary form (known as a lemma) while ensuring that the lemma belongs to the language. Unlike stemming, which simply chops off prefixes and suffixes of words, lemmatization considers the context and meaning of a word to derive its lemma.

### Word Net Lemmatizer:

WordNet is a lexical database that organizes words into synonym sets (synsets) and provides information on word meanings, relationships, and lexical hierarchies. The WordNet Lemmatizer uses WordNet to access lemma information.
The lemmatizer requires POS tagging to accurately identify the part of speech of each word (noun, verb, adjective, adverb). This ensures that words are lemmatized correctly based on their grammatical context.

### Stopwords
Stop words in NLTK (Natural Language Toolkit) refer to commonly used words (such as "the", "is", "at", "which", etc.) that are filtered out during the preprocessing of text data before natural language processing tasks. These words are typically removed because they do not contribute much to the overall meaning of the text and can introduce noise or unnecessary complexity in tasks like text analysis, information retrieval, and text mining.



## One Hot encoding:
One hot encoding is commonly used in tasks like text classification, where words or tokens are encoded as vectors to train machine learning models. It ensures that the categorical data is represented in a way that algorithms can effectively interpret and process. one hot encoding transforms categorical variables into binary vectors, making them suitable for machine learning algorithms that require numerical input, such as classification and regression tasks. 
Suppose we have three categories: "red", "green", and "blue". One hot encoding would represent these categories as:

"red" = [1, 0, 0]
"green" = [0, 1, 0]
"blue" = [0, 0, 1]


### Disadvantages: 
1.	Sparse matrix: size of vocalbulary  leads to overfitting
2.	ML Algorithms  Fixed size Input
3.	No semantic meaning is getting captured.
4.	It can also lead to out of vocabulary situation.

## Bag of Words:
In this technique, it Breaks down text into individual words or tokens and then Constructs a vector where each element represents the count of a word in the document.
### Advantages: 
1.	simple and intuitive
2.	Fixed size input.
### Disadvantages:
1.	Sparse Matrix or arrays overfitting
2.	Ordering of word gets changed
3.	Out of vocabulary.
4.	Semantic meaning is still not captured.
Eg.  For the sentence "The cat sat on the mat".
After tokenization: ["The", "cat", "sat", "on", "the", "mat"]
BoW representation: [1, 1, 1, 1, 2, 1] (assuming the order of words does not matter)

## N grams:
N-grams are contiguous sequences of N items (words, characters, or other units extracted from a text or speech corpus.
Types: Uni , Bi, Tri – grams.
N-grams are extracted by sliding a window of size N over the text. Each position in the text generates a new N-gram until all possible N-grams have been extracted. Its mostly used for Lnaguage Modelling,Informational retrieval etc.

## Term Frequency – Inverse Document Frequency:
TF-IDF is a powerful technique for assessing the significance of terms in documents by considering both their local frequency and global distribution across a collection of documents, facilitating effective information retrieval and analysis in various text-based applications.

TF(t,d) = (No. of repetition of words in sentence/ No. of words in sentence)
IDF(t,D) = loge(No. of sentences/No. of sentences containing the word)
Based on this, we generate a sent vs words table , giving us unique vector representation.
TF-IDF(t,d,D)  =  TF(t,d)×IDF(t,D)
### Advantages: 
1.	Intuitive
2.	Fixed Size
3.	Word Importance is getting captured.
### Disadvantages:
1.	Sparsity still exists.
2.	Out of vocabulary.
## Word-Embeddings:
In NLP, word embedding is a term used for the representation of words for text analysis, typically in the form of a real-values vector that encodes the meaning of word such that the words that are closer in the vector space are expected to be similar in meaning. 
## Word2vec:
It is a technique for NLP. The algorithm uses a neural network model to learn word association form a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. It represents each distinct word with a particular list of numbers called a vector. 
 

 
For small datasets: We use CBOW [CONTNUOUS BAG OF WORDS]
For large datasets: We use Skipgram
### Advantages of word2vec:
1.	Sparse matrix  Dense Matrix
2.	Schematic info is getting captured.
3.	Vocabulary size – fixed
4.	Out of vocabulary is solved.



## Transformers:

Transformers are a class of deep learning models that have revolutionized natural language processing (NLP) by leveraging self-attention mechanisms to capture relationships between words or tokens in a sequence. They are known for their ability to handle long-range dependencies and have set new benchmarks in tasks such as machine translation, text generation, and sentiment analysis.

### Usage of Transformers:

1. **Hugging Face Transformers Library**:
   - **Library Overview**: Hugging Face provides a popular open-source library called `transformers`, which offers pre-trained transformer models and utilities for working with them.
   - **Model Support**: The library supports a wide range of transformer architectures such as BERT, GPT (Generative Pre-trained Transformer), RoBERTa, T5, and many others.
   - **Easy Integration**: Users can easily load pre-trained models and fine-tune them on specific tasks using straightforward APIs.

2. **Key Features and Capabilities**:
   - **Model Variants**: Offers various transformer models tailored for specific tasks (e.g., BERT for language understanding, GPT for text generation).
   - **Tokenization**: Provides efficient tokenization utilities to convert text into tokens suitable for input into transformer models.
   - **Training and Fine-Tuning**: Facilitates fine-tuning on custom datasets for tasks like text classification, named entity recognition, and question answering.
   - **Community and Resources**: Active community support, documentation, and access to pre-trained models through the Hugging Face model hub.

3. **Applications**:
   - **Text Classification**: Classifying text into predefined categories (e.g., sentiment analysis, spam detection).
   - **Named Entity Recognition (NER)**: Identifying and categorizing entities (such as names, dates, and locations) in text.
   - **Question Answering**: Generating answers based on questions posed in natural language.
   - **Text Generation**: Creating coherent and contextually relevant text based on prompts.

4. **Integration with PyTorch and TensorFlow**:
   - The `transformers` library supports both PyTorch and TensorFlow frameworks, providing flexibility for developers and researchers to choose their preferred deep learning framework.

5. **Model Deployment**:
   - Hugging Face models can be deployed in various environments, from cloud-based solutions to edge devices, depending on performance and latency requirements.

In summary, Hugging Face’s `transformers` library plays a crucial role in democratizing access to state-of-the-art transformer models, enabling developers and researchers to leverage advanced NLP capabilities with ease and efficiency across a wide range of applications and domains.








