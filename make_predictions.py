#Author: Nazariy Dumanskyy
#Date: March 23, 2016
import pandas as pd
import numpy as np
import pickle
import csv
import sys
from sklearn import svm 
from sklearn.naive_bayes import GaussianNB

class MakePrediction:
	TEST_DATA_FILENAME = "test_potus_by_county.csv"
	PREDICTION_FILENAME = "predictions.csv"
	LABELS = ['Mitt Romney', 'Barack Obama']
	PREDICTION_MODEL = "Naive Bayes"

	#Purpose: Load the model together with test data
	def __init__(self):
		self.clf = self.load_model()
		test_dataframe = pd.read_csv(self.TEST_DATA_FILENAME)
		self.features_test = np.array(test_dataframe)

	#Purpose: Make a prediction
	#Precondition: features_test and clf are set
	def make_predicitons(self):
		print "Running....."
		pred = self.clf.predict(self.features_test)
		self.save_predictions(pred)

	#Purpose: Load the trained model from a file
	#Precondition: External file exists
	def load_model(self):
		print "Loading the model"
		with open("model/{}.pkl".format(self.PREDICTION_MODEL), "r") as file_input:
			try:
				clf = pickle.load(file_input)
			except:
				print "Please make sure to run build_model.py first"
				sys.exit(0)
		return clf

	#Purpose: Save predictions to a csv file
	#Precondition: pred is set
	def save_predictions(self, pred):
		field_names = ["prediction"]
		with open(self.PREDICTION_FILENAME, "w") as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=field_names)

			writer.writeheader()
			for prediction in pred:
				row = {
					"prediction" : self.LABELS[prediction]
				}
				writer.writerow(row)
		print "Predictions saved to {}".format(self.PREDICTION_FILENAME)

### Initiate the class and run the classifier to make predictions
prediction = MakePrediction()
prediction.make_predicitons()



