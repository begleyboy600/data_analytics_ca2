import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder

# Shows max rows and columns in pandas dataframe when being displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# DATA IMPORT
cwd = os.getcwd()
print(cwd)
os.chdir("C://Users//conor//PycharmProjects//data_science_ca2//")

data = pd.read_csv("data.csv")

# print first five rows of data.csv
# print(data.head())

# print information about the dataframe e.g index, column name, non-null count, dtype
# print(data.info())
"""
#   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   id               2158 non-null   int64  
 1   address          2150 non-null   object 
 2   zip_code         2158 non-null   int64  
 3   city             2158 non-null   object 
 4   county           2158 non-null   object 
 5   state            2158 non-null   object 
 6   neighborhood     2158 non-null   object 
 7   neighborhood_id  2158 non-null   int64  
 8   type             2158 non-null   object 
 9   baths            2138 non-null   float64
 10  beds             2137 non-null   float64
 11  mls_id           2080 non-null   object 
 12  status           2158 non-null   object 
 13  list_price       2149 non-null   float64
 14  image_url        2111 non-null   object 
 15  is_foreclosure   2158 non-null   int64  
 16  sqft             1969 non-null   float64
 17  lot_size         818 non-null    float64
 18  latitude         2158 non-null   float64
 19  longitude        2158 non-null   float64
dtypes: float64(7), int64(4), object(9)
memory usage: 337.3+ KB
None
"""

# prints out the count, mean, std, min, max, 25%, 50%, 75%, and max for every numerical column in the dataframe
# print(data.describe())

# DATA CLEANING
# prints out all unique vales in zip code, cities, county, state, neighborhood, type, baths, bed in dataframe
print("unique zip codes: ", data.zip_code.unique())
print("unique cities: ", data.city.unique())
print("unique county: ", data.county.unique())
print("unique state: ", data.state.unique())
print("unique neighborhood: ", data.neighborhood.unique())
print("unique baths: ", data.baths.unique())
print("unique beds: ", data.beds.unique())
print("unique status: ", data.status.unique())
print("unique type: ", data.type.unique())
print("unique is_foreclosuree: ", data.is_foreclosure.unique())

# prints out the number of null values in each column
# print(data.isnull().sum())

# removes rows with null values in address, baths, beds, list_price column
data = data.drop(data[data.address.isnull()].index)
data = data.drop(data[data.baths.isnull()].index)
data = data.drop(data[data.beds.isnull()].index)
data = data.drop(data[data.list_price.isnull()].index)

# drops the lot_size column because there is too many null values (1340) null values
data.drop(['lot_size'], axis=1, inplace=True)
# drops the mls_id column because it's not important
data.drop(['mls_id'], axis=1, inplace=True)
# drops the image_url column because it's not important
data.drop(['image_url'], axis=1, inplace=True)
# drops the sqft column because there is too many null values (189) null values
data.drop(['sqft'], axis=1, inplace=True)

# print(data.isnull().sum())

# prints out number of rows in the dataframe
print("number of rows in dataframe: ", len(data.index))

# Converting city variables into numerical variables
city_mapper = {'Miami': 1, 'Miami Beach': 2, 'Coconut Grove': 3, 'Coral Gables': 4, 'Doral': 5, 'Hialeah': 6}
data["CityName"] = data["city"].replace(city_mapper)
print("unique cities: ", data.city.unique())
print("unique cityNames: ", data.CityName.unique())

# Converting county variables into numerical variables
county_mapper = {'MIAMI-DADE COUNTY': 1, 'MIAMI-DADE': 2, 'Miami-Dade': 3, 'OTHER': 4, 'Miami-Dade County': 5, 'DADE': 6}
data["CountyName"] = data["county"].replace(county_mapper)
print("unique counties: ", data.county.unique())
print("unique countyNames: ", data.CountyName.unique())
print(data.head())
print(data.loc[189])

