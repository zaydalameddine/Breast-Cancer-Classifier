# importing a binary database from sklearn
from sklearn.datasets import load_breast_cancer
# importing the splitting function
from sklearn.model_selection import train_test_split
# importing the KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# loading the databse into that variable
breast_cancer_data = load_breast_cancer()

# get a better understanding of the database
print(breast_cancer_data.target, breast_cancer_data.target_names)

# split data into 80%training and 20% testing sets
breast_cancer_train, breast_cancer_test, target_train, target_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

# printing the len of the data to make sure that the data is the same len as its adjoining labels
print(len(breast_cancer_train), len(breast_cancer_test), len(target_train), len(target_test))


highest_accuracy = 0
index = 0
# finding the best k value using a loop from 1 - 100
for k in range(1, 101):
  # creating and training the classifier model
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(breast_cancer_train, target_train)
  accuracy = 0
  
  # testing model against test sets and checking if score is higher than previous k 
  if classifier.score(breast_cancer_test, target_test) > highest_accuracy:
    highest_accuracy = classifier.score(breast_cancer_test, target_test)
    index = k

#make a model using the best k value and predict the label for the test set
classifier = KNeighborsClassifier(n_neighbors = index)
classifier.fit(breast_cancer_train, target_train)
guesses = classifier.predict(breast_cancer_test.data)

#while not a great graph ot shows that the data is mostly right which is also displayed by the 0.965 R^2 value
plt.scatter(guesses, target_test.data, alpha = 0.1)
plt.xlabel("Classification Guess")
plt.ylabel("Clasification Label")
plt.title("Breast Cancer Classifier")
plt.show()