## Setup and Imports

`
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
`

## Linear Regression

`
import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
`

*Equation for a line of best fit for this type of graph*

`
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()
`

## Data

*The following is an example of importing training dating.*

`
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
`

*The .head() method from pandas will show the first 5 items in the dataframe.* 

**Training shape** 

`dftrain.shape(627, 9)`

* This would train 627 entries and 9 features

## Training vs. Testing Data

* When we train a model we need two sets of data: training and testing. 

The **training** data is what we feed to the model so that it can develop and learn. 
The **testing data** is what we use to evaluate the model and see how well it is performing. 

## Feature Columns 

`
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)
`

* This document details feature columns and how they can be used as inputs to neural networks using TensorFlow

## The Training Process 

Usually, not all the data is fed into the modal at once, but small batches of entries are insteads. We will feed these batches into the model multiple times according to the number of **epochs**. 

An **epoch** is simply one stream of our entire dataset. The number of epochs we define ***is the amount of times our model will see the entire dataset***. We use multiple epochs in hope that after seeing the same data multiple times hte model will better determine how to estimate it. 

## Input Function

`
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
`

## Creating the Model 

`
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier
`

* Using the linear estimator to utilize the linear regression algorithm. 

## Training the Model 

`
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model
`

## Classification

`
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
`

### Another Input Function 

`
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)
`

## Building the Model 

* Building use DNNClassifier (Deep Neural Network)

`
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.

`
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)
`

## Training

`
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
`

## Predictions

`
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))
`

## Clustering 

Clustering is a mchine learning technique that involves the grouping of data points. In theory, data points that are in the same group should have similar properties and/or features, while data points in different groups should have highly dissimilar properties and/or features. 

Unfortunately, there are issues with the current version of TensorFlow and the implementation of KMeans. This means we cannot use KMeans without writing the algorithm from scratch. 

Basic Algorithm for K-Means

* Step 1: Randomly pick K points to place K centroids. 
* Step 2: Assign all the data points ot the centroids by distance. The closest centroid to a point is hte one it is assigned to. 
* Step 3: Average all the points belonging to each centroied to find the middle of those clusters (center of mass). Place the corresponding centroids into that position. 
* Step 4: Reassign every point once again to the closest centroid 
* Step 5: Repeat steps 3 and 4 until no point changes which centroid it belongs to. 

## Data

Let's start by discussing the type of data we use when we work with a hidden markov model. 

In the previous sectoins we worked with large datasets of 100's of different entries. For a markov model we are only interested in probability distribution that have to do with states. 

We can find these probabilities from large datasets or may laready have these values. We'll run through an example in a second that should clear some things up, but let's discuss the components of a markov model. 

States: In each markov model we have a finite set of states. These states could be something like "warm" and "cold" or "high" and "lov" or even "red", "green" and "blue". These states are "hidden" within the mode, which means we do not directly observer them. 

**Observations:** Each state has a particular outcome or observation associated with it based on a probability distribution. An example of this is the following: **On a hot dya Tim has an 80% change of being happy and a 20% chance of being sad.**

**Transitions:** Each state will hvae a probability defining the likelyhood of transitioning to a different state. An example is the following: a cold day has a 30% change of being followed by a hot dat and a 70% change of being followed by another cold day. 


New Notes section
