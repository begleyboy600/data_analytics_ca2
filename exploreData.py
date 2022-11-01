# have to process zip code data
# then data exploration and visualisation
# then business case at top of file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.pyplot import figure

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

# print(data.isnull().sum())

# prints out number of rows in the dataframe
print("number of rows in dataframe: ", len(data.index))

# Converting city variables into numerical variables
city_mapper = {'Miami': 1, 'Miami Beach': 2, 'Coconut Grove': 3, 'Coral Gables': 4, 'Doral': 5, 'Hialeah': 6}
data["CityName"] = data["city"].replace(city_mapper)
# print("unique cities: ", data.city.unique())
print("unique cityNames: ", data.CityName.unique())

# drop city column
data.drop(['city'], axis=1, inplace=True)

# Converting county variables into numerical variables
county_mapper = {'MIAMI-DADE COUNTY': 1, 'MIAMI-DADE': 2, 'Miami-Dade': 3, 'OTHER': 4, 'Miami-Dade County': 5,
                 'DADE': 6}
data["CountyName"] = data["county"].replace(county_mapper)
# print("unique counties: ", data.county.unique())
print("unique countyNames: ", data.CountyName.unique())

# drop county column
data.drop(['county'], axis=1, inplace=True)
# print(data.head())

neighborhood_mapper = {'Upper Eastside': 1, 'Brickell': 2, 'Overtown': 3, 'Downtown Miami': 4,
                       'North-East Coconut Grove': 5, 'Miami Islands': 6, 'San Marco Island': 7, 'Coral Way': 8,
                       'Watson Island': 9, 'Little Haiti': 10, 'Biscayne Island': 11, 'Liberty City': 12,
                       'Wynwood - Edgewater': 13, 'South-West Coconut Grove': 14, 'Allapattah': 15,
                       'Coral Gables': 16, 'Fair Isle': 17, 'Douglas Park': 18, 'Flagami': 19, 'Shenandoah': 20,
                       'Little Havana': 21, 'Alameda - West Flagler': 22, 'North Bayfront': 23, 'Virginia Key': 24}

data["NeighborhoodName"] = data["neighborhood"].replace(neighborhood_mapper)
# print("unique neighborhood: ", data.neighborhood.unique())
print("unique neighborhood names: ", data.NeighborhoodName.unique())
# drop neighborhood column
data.drop(['neighborhood'], axis=1, inplace=True)
# print(data.head())


zip_code_mapper = {33160: 1, 33154: 2, 33141: 3, 33140: 4, 33139: 5, 33035: 6, 33034: 7, 33189: 8,
                   33186: 9, 33133: 10, 33185: 11, 33175: 12, 33129: 13, 33131: 14, 33132: 15, 33136: 16,
                   33128: 17, 33130: 18, 33126: 19, 33178: 20, 33182: 21, 33166: 22, 33010: 23, 33013: 24,
                   33018: 25, 33169: 26, 33138: 27, 33161: 28, 33181: 29, 33149: 30}

data["ZipCodeName"] = data["zip_code"].replace(zip_code_mapper)
# print("unique zip_code: ", data.zip_code.unique())
print("unique zip_codes names: ", data.ZipCodeName.unique())
# drop zip_code column
data.drop(['zip_code'], axis=1, inplace=True)
# print(data.head())


# Status column has 4 unique columns: ['inactive' 'Active' 'Pending' 'public-record']. I will create 3 new columns and convert status variable to binary (1 or 0)
# 1 means property is inactive and 0 means it is not
data['StatusInactive'] = np.where(data.status == "inactive", 1, 0)

# 1 means property is active and 0 means it is not
data['StatusActive'] = np.where(data.status == "Active", 1, 0)

# 1 means property is pending and 0 means it is not
data['StatusPending'] = np.where(data.status == "Pending", 1, 0)

# 1 means property is public-record and 0 means it is not
data['StatusPublicRecord'] = np.where(data.status == "public-record", 1, 0)

# drop status column
data.drop(['status'], axis=1, inplace=True)

# Type column has 7 unique columns: ['Condo/Coop' 'Single Family Residential' 'Multi Family' 'Other'
#  'Townhouse' 'Condo/Co-op' 'Garage']
# 1 means type is Condo/Coop and 0 means it is not
data['TypeCondo/Coop'] = np.where(data.type == "Condo/Coop", 1, 0)

# 1 means type is Single Family Residential and 0 means it is not
data['TypeSingleFamilyResidential'] = np.where(data.type == "Single Family Residential", 1, 0)

# 1 means type is Multi Family and 0 means it is not
data['TypeMulti Family'] = np.where(data.type == "Multi Family", 1, 0)

# 1 means type is Other and 0 means it is not
data['TypeOther'] = np.where(data.type == "Other", 1, 0)

# 1 means type is Townhouse and 0 means it is not
data['TypeTownhouse'] = np.where(data.type == "Townhouse", 1, 0)

# 1 means type is Condo/Co-op and 0 means it is not
data['TypeCondo/Co-op'] = np.where(data.type == "Condo/Co-op", 1, 0)

# 1 means type is Condo/Coop and 0 means it is not
data['TypeGarage'] = np.where(data.type == "Garage", 1, 0)

# Dropping other non-revalent columns
# drops the lot_size column because there is too many null values (1340) null values
data.drop(['lot_size'], axis=1, inplace=True)

# drops the mls_id column because it's not important
data.drop(['mls_id'], axis=1, inplace=True)

# drops the image_url column because it's not important
data.drop(['image_url'], axis=1, inplace=True)

# drops the sqft column because there is too many null values (189) null values
data.drop(['sqft'], axis=1, inplace=True)

# drops the neighborhood_id column because it's not important
data.drop(['neighborhood_id'], axis=1, inplace=True)

# drops the address column because it's not important
data.drop(['address'], axis=1, inplace=True)

# drops the id column because it's not important
data.drop(['id'], axis=1, inplace=True)

# drops the type column because it's not important
data.drop(['type'], axis=1, inplace=True)

print(data.head())
"""
# prints out all unique vales in zip code, cities, county, state, neighborhood, type, baths, bed in dataframe
print("unique zip codes: ", data.zip_code.unique())
print("unique baths: ", data.baths.unique())
print("unique beds: ", data.beds.unique())
print("unique is_foreclosuree: ", data.is_foreclosure.unique())

# prints out if there is any null values
print(pd.isnull(data).sum())
"""

"""
# Outliers
# boxplot of baths column in dataframe
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot of baths column")
sns.boxplot(x=data.baths)
plt.show()

# boxplot of beds column in dataframe
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot of beds column")
sns.boxplot(x=data.beds)
plt.show()

# boxplot of list_price column in dataframe
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot of list_price column")
sns.boxplot(x=data.list_price)
plt.show()

# boxplot of NeighborhoodName column in dataframe
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot of NeighborhoodName column")
sns.boxplot(x=data.NeighborhoodName)
plt.show()

# boxplot of list_price column in dataframe
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot of list_price column")
sns.boxplot(x=data.ZipCodeName)
plt.show()
"""

# Exploratory data analysis - univariate analysis
"""
# data visualization for baths column
number_baths = data.baths.value_counts()  # Gives a pandas series
number_baths.plot.bar()
plt.xticks(rotation=360)
plt.title("Bar Chart Of Baths Column")
plt.xlabel("Number Of Baths")
plt.ylabel("Number Of Properties")
plt.show()

# data visualization for beds column
number_beds = data.beds.value_counts()  # Gives a pandas series
number_beds.plot.bar()
plt.xticks(rotation=360)
plt.title("Bar Chart Of Beds Column")
plt.xlabel("Number Of Beds")
plt.ylabel("Number Of Properties")
plt.show()

# data visualization for is_foreclosure column
number_is_foreclosure = data.is_foreclosure.value_counts()  # Gives a pandas series
number_is_foreclosure.plot.bar()
plt.xticks(rotation=360)
plt.title("Bar Chart Of Is_Foreclosure Column")
plt.xlabel("Number Of Is_Foreclosure")
plt.ylabel("Number Of Properties")
plt.show()

figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
number_is_foreclosure_pie = data.is_foreclosure.value_counts(normalize=True)
number_is_foreclosure_pie.plot.pie(radius=2200, frame=True, autopct='%.0f%%')
plt.legend(["yes", "no"], loc='upper right')
plt.title("Pie Chart Of Is_Foreclosure Column")
plt.show()

# data visualization for CityName column
city_labels = ['Miami', 'Miami Beach', 'Coconut Grove', 'Coral Gables', 'Doral', 'Hialeah']
default_city_ticks = range(len(city_labels))
number_city_names = data.CityName.value_counts()  # Gives a pandas series
number_city_names.plot.bar()
plt.title("Bar Chart Of City_Names Column")
plt.xticks(default_city_ticks, city_labels, rotation=45)
plt.xlabel("Number Of City_Names")
plt.ylabel("Number Of Properties")
plt.show()


# data visualization for CountyName column
county_labels = ['MIAMI-DADE COUNTY', 'MIAMI-DADE', 'Miami-Dade', 'OTHER', 'Miami-Dade County', 'DADE']
default_county_ticks = range(len(county_labels))
number_county_names = data.CountyName.value_counts()  # Gives a pandas series
number_county_names.plot.bar()
plt.xticks(default_county_ticks, county_labels, rotation=360)
plt.title("Bar Chart Of County_Names Column")
plt.xlabel("Number Of County_Names")
plt.ylabel("Number Of Properties")
plt.show()

figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
number_county_names_pie = data.CountyName.value_counts(normalize=True)
number_county_names_pie.plot.pie(radius=2200, frame=True, autopct='%.0f%%')
plt.legend(county_mapper, loc='upper right')
plt.title("Pie Chart Of County_Names Column")
plt.show()
"""

"""
# data visualization for NeighborhoodName column
neighborhood_labels = ['Upper Eastside', 'Brickell', 'Overtown', 'Downtown Miami',
                       'North-East Coconut Grove', 'Miami Islands', 'San Marco Island', 'Coral Way',
                       'Watson Island', 'Little Haiti', 'Biscayne Island', 'Liberty City',
                       'Wynwood - Edgewater', 'South-West Coconut Grove', 'Allapattah',
                       'Coral Gables', 'Fair Isle', 'Douglas Park', 'Flagami', 'Shenandoah',
                       'Little Havana', 'Alameda - West Flagler', 'North Bayfront', 'Virginia Key']
default_neighborhood_ticks = range(len(neighborhood_labels))
number_neighborhood_names = data.NeighborhoodName.value_counts()  # Gives a pandas series
number_neighborhood_names.plot.bar()
plt.xticks(default_neighborhood_ticks, neighborhood_labels, rotation=360)
plt.title("Bar Chart Of Neighborhood_Names Column")
plt.xlabel("Number Of Neighborhood_Names")
plt.ylabel("Number Of Properties")
plt.show()

figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
number_county_names_pie = data.NeighborhoodName.value_counts(normalize=True)
number_county_names_pie.plot.pie(radius=2200, frame=True, autopct='%.0f%%')
plt.legend(neighborhood_mapper, loc='upper right')
plt.title("Pie Chart Of Neighborhood_Names Column")
plt.show()
"""

# Exploratory data analysis - bivariate analysis

# data for beds and price column
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.beds, data.list_price)
plt.title("Scatter plot of beds and price")
plt.xlabel("beds")
plt.ylabel("price")
plt.show()

# calculate correlation for beds and price column
correlation = np.corrcoef(data.beds, data.list_price)
print("correlation of beds and price: ", correlation)

# data for baths and price column
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.baths, data.list_price)
plt.title("Scatter plot of baths and price")
plt.xlabel("baths")
plt.ylabel("price")
plt.show()

# calculate correlation for baths and price column
correlation2 = np.corrcoef(data.baths, data.list_price)
print("correlation of baths and price: ", correlation2)

# Exploratory data analysis - multivariate analysis
# heatmap of all variables in dataframe
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(data.corr(), annot=True, cmap='Reds')
plt.show()

# geospatial visualization


