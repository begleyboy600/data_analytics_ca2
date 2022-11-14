from datetime import datetime
import keras
import pandas as pd
import numpy as np
import os
from keras import layers
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pickle
import tensorboard
import tensorflow as tf
import datetime


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
feats = ['baths', 'beds', 'list_price', 'is_foreclosure', 'latitude',
         'longitude', 'CityName', 'CountyName', 'NeighborhoodName',
         'ZipCodeName', 'StatusInactive', 'StatusActive', 'StatusPending',
         'StatusPublicRecord', 'TypeCondo/Coop', 'TypeSingleFamilyResidential',
         'TypeMulti Family', 'TypeOther', 'TypeTownhouse', 'TypeCondo/Co-op',
         'TypeGarage']

x = data[feats]

y = data.list_price

# first split training - test 60-40
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# now split test set into  validation - test  equally
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)


def linear_regression_model(x_train_, y_train_, x_val_, y_val_):
    model = LinearRegression()
    model.fit(x_train_, y_train_)
    predictions = model.predict(x_val_)
    prediction_MAE = sum(abs(predictions - y_val_)) / len(y_val_)
    prediction_MAPE = sum(abs((predictions - y_val_) / y_val_)) / len(y_val_)
    prediction_RMSE = (sum((predictions - y_val_) ** 2) / len(y_val_)) ** 0.5
    print(f"mae: {prediction_MAE:.15f}")
    print(f"mape: {prediction_MAPE:.15f}")
    print(f"rmse: {prediction_RMSE:.15f}")

    df_preds = pd.DataFrame({'Actual': y_val_.squeeze(), 'Predicted': predictions.squeeze()})
    print(df_preds.head())
    return model


def linear_regression_model_test(model, x_test_, y_test_):
    predictions = model.predict(x_test_)
    # preds = [pr[0] for pr in predictions]
    prediction_MAE = sum(abs(predictions - y_test_)) / len(y_test_)
    prediction_MAPE = sum(abs((predictions - y_test_) / y_test_)) / len(y_test_)
    prediction_RMSE = (sum((predictions - y_test_) ** 2) / len(y_test_)) ** 0.5
    print(f"mae: {prediction_MAE:.15f}")
    print(f"mape: {prediction_MAPE:.15f}")
    print(f"rmse: {prediction_RMSE:.15f}")

    # df_preds = pd.DataFrame({'Actual': y_test_.squeeze(), 'Predicted': predictions.squeeze()})
    # print(df_preds.head())


def find_best_ai_architecture(x_train_, y_train_):
    nueral_network_accuracy = []
    hidden_layers_breakdown = 0
    for layer in range(2, 13, 1):
        for node in range(5, 20, 5):
            print(hidden_layers_breakdown)
            set_layers = tuple([node for i in range(layer)])
            print(set_layers)
            acc = model_selection_for_neural_net(layers=set_layers, iterations=500, x_train_=x_train_,
                                                 y_train_=y_train_,
                                                 x_val_=x_val, y_val_=y_val)
            print(acc)
            nueral_network_accuracy.append(acc)
            hidden_layers_breakdown = hidden_layers_breakdown + 1

    min_accuracy = min(nueral_network_accuracy)
    min_accuracy_index = nueral_network_accuracy.index(min_accuracy)
    print("model by min accuracy: ", min_accuracy, " index: ", min_accuracy_index)

    max_accuracy = max(nueral_network_accuracy)
    max_accuracy_index = nueral_network_accuracy.index(max_accuracy)
    print("model by max accuracy: ", max_accuracy, " index: ", max_accuracy_index)

    # model by min accuracy:  1.1010587094603724e-05  index:  19
    # model by max accuracy:  1.2850688659865277  index:  24

    # mean absolute percentage error: 1.1010587094603724e-05 = 0.0000110106
    # model accuracy: 0.9999889894
    # model layers: (10, 10, 10, 10, 10, 10, 10, 10)


def model_selection_for_neural_net(layers, iterations, x_train_, y_train_, x_val_, y_val_):
    model = MLPRegressor(hidden_layer_sizes=layers, max_iter=iterations)

    # Select the model using the training data
    model.fit(x_train_, y_train_)
    print("model trained")

    # Find the predicted values from the test set
    predictions = model.predict(x_val_)
    # prediction_MAE = sum(abs(predictions - y_val_)) / len(y_val_)
    prediction_MAPE = sum(abs((predictions - y_val_) / y_val_)) / len(y_val_)
    # prediction_RMSE = (sum((predictions - y_val_) ** 2) / len(y_val_)) ** 0.5

    return prediction_MAPE


def final_neural_network(layers, iterations, x_train_, y_train_):
    model = MLPRegressor(hidden_layer_sizes=layers, max_iter=iterations)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Select the model using the training data
    model.fit(x_train_, y_train_)
    print("model trained")
    return model


def neural_network_model_with_tf(x_train_, y_train_, x_test_, y_test_):
    model = keras.Sequential(
        [
            layers.Dense(10, activation="relu", name="layer1"),
            layers.Dense(10, activation="relu", name="layer2"),
            layers.Dense(10, activation="relu", name="layer3"),
            layers.Dense(10, activation="relu", name="layer4"),
            layers.Dense(10, activation="relu", name="layer5"),
            layers.Dense(10, activation="relu", name="layer6"),
            layers.Dense(10, activation="relu", name="layer7"),
            layers.Dense(10, activation="relu", name="layer8"),
            layers.Dense(1, name="layer9"),
        ]
    )
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(loss='mean_absolute_percentage_error', optimizer=tf.keras.optimizers.Adam(0.001))

    model.fit(x=x_train_, y=y_train_, epochs=200, validation_data=(x_test_, y_test_), callbacks=[tensorboard_callback])
    return model


def save_ai_model(model, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
    print(f"{file_name} was created successfully")


def save_tf_model(model, file_name):
    model.save(file_name)
    print(f"{file_name} was created successfully")


def load_tf_model(file_name):
    model = keras.models.load_model(file_name)
    return model


def load_ai_model(file_name):
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model


def ai_model_evaluation(loaded_model, x_test_, y_test_):
    predictions = loaded_model.predict(x_test_)
    prediction_MAE = sum(abs(predictions - y_test_)) / len(y_test_)
    prediction_MAPE = sum(abs((predictions - y_test_) / y_test_)) / len(y_test_)
    prediction_RMSE = (sum((predictions - y_test_) ** 2) / len(y_test_)) ** 0.5
    print(f"MAE: {prediction_MAE:.15f}")
    print(f"MAPE: {prediction_MAPE:.15f}")
    print(f"RMSE: {prediction_RMSE:.15f}")

    model_accuracy = 1 - prediction_MAPE
    print(f"Model Accuracy: {model_accuracy:.15f}")
    return predictions

# find the best neural network architecture
# find_best_ai_architecture(x_train, y_train)
# best neural

# call linear regression, neural network and tensorflow neural network models
# ai_model = final_neural_network(layers=(10, 10, 10, 10, 10, 10, 10, 10), iterations=500, x_train_=x_train, y_train_=y_train)
# model = neural_network_model_with_tf(x_train_=x_train, y_train_=y_train, x_test_=x_test, y_test_=y_test)
# lr_model = linear_regression_model(x_train_=x_train, y_train_=y_train, x_val_=x_val, y_val_=y_val)

# pickle file name
file_name = "neural_network_model.pkl"
file_name2 = "neural_network_model_with_tensorboard.h5"
file_name3 = "basic_linear_regression_model.pkl"

# Save neural network
# save_ai_model(model, file_name=file_name)
# save_tf_model(model, file_name2)
# save_ai_model(lr_model, file_name3)

# Load neural network
loaded_lr_model = load_ai_model(file_name3)
loaded_sk_model = load_ai_model(file_name)
loaded_tf_model = load_tf_model(file_name2)

# install tensorboard: pip install tensorboard
# to run the tensorboard navigate to the project folder and then insert the following command:
# tensorboard --logdir=logs/fit

# get ai model evaluation
print("linear regression model evaluation: ")
predictions_lr = ai_model_evaluation(loaded_model=loaded_lr_model, x_test_=x_test, y_test_=y_test)

print("sklearn neural network model evaluation: ")
predictions_sk = ai_model_evaluation(loaded_model=loaded_sk_model, x_test_=x_test, y_test_=y_test)

# print("tensorflow neural network model evaluation: ")
# predictions_tf = ai_model_evaluation(loaded_model=loaded_tf_model, x_test_=x_test, y_test_=y_test)

# visualizations for linear regression model evaluations
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_lr)
plt.title("Predictions v Actual Test Values For Linear Regression Model", fontsize=22)
plt.xlabel("Actual values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.show() # Should be close to a straight line

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_lr - y_test)
plt.title("Predictions v Actual Test Values For Linear Regression Model", fontsize=22)
plt.xlabel("Actual values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.show()

# visualizations for sklearn model evaluations
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_sk)
plt.title("Predictions v Actual Test Values For Sklearn Model", fontsize=22)
plt.xlabel("Actual values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.show() # Should be close to a straight line

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_sk - y_test)
plt.title("Predictions v Actual Test Values For Sklearn Model", fontsize=22)
plt.xlabel("Actual values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.show()

