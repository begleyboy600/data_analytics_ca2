import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Shows max rows and columns in pandas dataframe when being displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# DATA IMPORT
cwd = os.getcwd()
print(cwd)
os.chdir("C://Users//conor//PycharmProjects//data_science_ca2//")

data = pd.read_csv("processed_data.csv")

# print(data.head())

# print(data.info())
"""
0   Unnamed: 0                   2119 non-null   int64  
 1   address                      2119 non-null   object 
 2   state                        2119 non-null   object 
 3   baths                        2119 non-null   float64
 4   beds                         2119 non-null   float64
 5   list_price                   2119 non-null   float64
 6   is_foreclosure               2119 non-null   int64  
 7   latitude                     2119 non-null   float64
 8   longitude                    2119 non-null   float64
 9   CityName                     2119 non-null   int64  
 10  CountyName                   2119 non-null   int64  
 11  NeighborhoodName             2119 non-null   int64  
 12  ZipCodeName                  2119 non-null   int64  
 13  StatusInactive               2119 non-null   int64  
 14  StatusActive                 2119 non-null   int64  
 15  StatusPending                2119 non-null   int64  
 16  StatusPublicRecord           2119 non-null   int64  
 17  TypeCondo/Coop               2119 non-null   int64  
 18  TypeSingleFamilyResidential  2119 non-null   int64  
 19  TypeMulti Family             2119 non-null   int64  
 20  TypeOther                    2119 non-null   int64  
 21  TypeTownhouse                2119 non-null   int64  
 22  TypeCondo/Co-op              2119 non-null   int64  
 23  TypeGarage                   2119 non-null   int64  
"""

# Feature Engineering
# check data clean - All Ok
print(data.isnull().sum())

# drop address column as there is no need for it
data.drop('address', axis=1, inplace=True)

# print out all unique values in state column
print("unique state: ", data.state.unique())

# drop state column as there is only 1 unique value
data.drop('state', axis=1, inplace=True)

# drop first column with indexes
data.drop('Unnamed: 0', axis=1, inplace=True)

# PLEASE NOTE ALL OTHER COLUMN VARIABLES ARE CHANGED TO NUMERICAL DATA TYPES ALREADY. THIS WAS DONE THE exploreData.py FILE

# PLEASE NOTE HEATMAP OF DATAFRAME COLUMNS HAS ALREADY BEEN GENERATED. IT CAN BE FOUND IN THE data_visualization DIRECTORY

# Modelling. Split up the data into training, validation, and testing
# print out all columns in dataframe
print(data.columns)

# set the Response and the predictor variables
x = data[['baths', 'beds', 'is_foreclosure', 'latitude',
       'longitude', 'CityName', 'CountyName', 'NeighborhoodName',
       'ZipCodeName', 'StatusInactive', 'StatusActive', 'StatusPending',
       'StatusPublicRecord', 'TypeCondo/Coop', 'TypeSingleFamilyResidential',
       'TypeMulti Family', 'TypeOther', 'TypeTownhouse', 'TypeCondo/Co-op',
       'TypeGarage']]

y = data[['list_price']]

# nirst split training - test 60-40
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# now split test set into  validation - test  equally
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

print(len(x_train))
print(len(x_val))
print(len(x_test))

# https://keras.io/examples/structured_data/structured_data_classification_from_scratch/


def model_selection_for_nueral_net(layers, iterations, x_train_, y_train_, x_val_, y_val_):
    model = MLPClassifier(hidden_layer_sizes=layers, max_iter=iterations)

    # Select the model using the training data
    model.fit(x_train_, y_train_)

    #########Modelling - Step 3: Model Evaluation Based on TEST set.

    # Find the predicted values from the test set
    predictions = model.predict(x_val_)

    # Calculate performance metrics Accuracy, Error Rate, Precision and Recall from the confusion matrix

    confusionMatrix = confusion_matrix(y_val_, predictions)
    # print(confusionMatrix)

    # Check numbers
    numberSurvivedTest = y_val_.value_counts()

    accuracy = (confusionMatrix[0, 0] + confusionMatrix[1, 1]) / len(predictions)
    # errorRate = 1 - accuracy
    # precision = (confusionMatrix[1, 1]) / (confusionMatrix[1, 1] + confusionMatrix[0, 1])
    # recall = (confusionMatrix[1, 1]) / (confusionMatrix[1, 1] + confusionMatrix[1, 0])
    # print("Accuracy: " + str(accuracy))
    # print("Error Rate: " + str(errorRate))
    # print("Precision: " + str(precision))
    # print("Recall: " + str(recall))
    # Key value is the Accuracy which for the run was 0.8619
    return accuracy


nueral_network_accuracy = []
hidden_layers_breakdown = 0
for layer in range(2, 13, 1):
    for node in range(5, 20, 5):
        print(hidden_layers_breakdown)
        set_layers = tuple([node for i in range(layer)])
        print(set_layers)
        acc = model_selection_for_nueral_net(layers=set_layers, iterations=500, x_train_=x_train, y_train_=y_train, x_val_=x_val, y_val_=y_val)
        print(acc)
        nueral_network_accuracy.append(acc)
        hidden_layers_breakdown = hidden_layers_breakdown + 1


best_accuracy = max(nueral_network_accuracy)
accuracy_index = nueral_network_accuracy.index(best_accuracy)
print("best model by accuracy: ", best_accuracy, " index: ", accuracy_index)




