#Author: Nazariy Dumanskyy
#Date: March 23, 2016
import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn import svm 
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

class BuildModel:
	TRAIN_DATA_FILENAME = "train_potus_by_county.csv"
	TEST_SIZE = 0.10

	def __init__(self):
		train_dataframe = pd.read_csv(self.TRAIN_DATA_FILENAME)
		train_labels = train_dataframe.pop("Winner")
		#convert labels to 1s and 0s
		labels  = list(set(train_labels))
		train_labels = np.array([labels.index(x) for x in train_labels])

		#Since the labels have been removed train_dataframe is train_features
		train_features = train_dataframe

		#Split the data into train/test data
		#10% is test data and 90% is train data
		self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(train_features, train_labels, test_size=self.TEST_SIZE, random_state=42)

	#Purpose: Run the Naive Bayes Classifier
	#Precondition: features and labels are set
	def run_naive_bayes(self):
		print "Running......"
		clf = GaussianNB()
		clf.fit(self.features_train, self.labels_train)
		pred = clf.predict(self.features_test)
		accuracy = clf.score(self.features_test, self.labels_test)
		#Save model and performance
		self.save_model(clf, "Naive Bayes")
		self.save_performance("Naive Bayes", accuracy)

	#Purpose: Run the Support Vector Machine classifier
	#Precondition: features and labels are set
	def run_svm(self):
		clf = svm.SVC(kernel="linear")
		clf.fit(self.features_train, self.labels_train)
		pred = clf.predict(self.features_test)
		accuracy = accuracy_score(pred, self.labels_test)
		self.save_performance("SVM", accuracy)

	#Purpose: Run the Decision Tree classifier
	#Precondition: features and labels are set
	def run_decision_tree(self):
		clf = tree.DecisionTreeClassifier(min_samples_leaf = 50)
		clf = clf.fit(self.features_train,self.labels_train)
		accuracy = clf.score(self.features_test,self.labels_test)
		self.save_performance("Decision Tree", accuracy)

	#Purpose: Serialize the model for later use
	#Precondition: clf and model_type are both set
	def save_model(self, clf, model_type):
		with open("model/{}.pkl".format(model_type), "w") as output:
			pickle.dump(clf, output)
		print "Saved model to model/{}.pkl".format(model_type) 

	#Purpose: Log the performance of a model
	#Precondition: model_type and accuracy are both set
	def save_performance(self, model_type, accuracy):
		accuracy *= 100
		accuracy = round(accuracy,3)
		print "Accuracy: {}%".format(accuracy)
		performance = "{} with accuracy of {}% \n".format(model_type, accuracy)
		with open("performance_log.txt", "a") as output:
			output.write(performance)
		print "Saved performance data to performance_log.txt"

### Initiate the class and run the classifier
model = BuildModel()
model.run_naive_bayes()



