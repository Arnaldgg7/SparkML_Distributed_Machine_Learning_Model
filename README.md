# SparkML_Distributed_Machine_Learning_Model
We exploit to the full the capabilities of distributed processing using a Distributed Linear Support Vector Machine to retrieve, train the model and predict a medical dataset.

The present dataset that is used to perform this distributed machine learning implementation using Spark and its well-known librart SparkML (previously known as SparkMLLib) is inspired in a real problem aimed at determining the days a patient is going to stay in the hospital based on their medical history. Since the dataset is artifficially created, the expected global Accuracy is around 65%. In our implementation though, we get a 68.24% of global Accuracy, which outperforms the theoretical expected one.

In the real problem, predicting exactly the days that a patient is going to stay in the hospital was extremely difficult. Therefore, such a regression problem was reconverted to a classification problem, thereby establishing ranges of days as classes and classify the individuals accordingly into them.

In order to apply the Linear SVM to the data, which becomes the most suitable Machine Learning classifier given the huge dimensionality of the data once we vectorize the list variables, we must perform the following pre-processing steps:

**1. Discretizing continuous variables into range classes, so that they pave higher accuracies by using ranges of values as classes.**
**2. One-Hot encode the categorical variables, so that they can be passed to the classifier in the required format.**
**3. Tokenize and vectorize the list variables into a sparse vector with each position as a different encountered value in those variables.**
**4. Perform an assembling of all 'features' or Predictor Variables into one single vector, so that it can be passed to the classifier, along with the Response Variable.**

Overall, this project offers an overview of the potentialities of distributed processing when using powerful Machine Learning algorithms that have been implemented in a stochastic way to perfectly suit distributed environments and make the most of the processing power of a cluster, all of this implemented under the Spark ML library.
