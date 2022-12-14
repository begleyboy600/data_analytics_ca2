"""
Business Case
The main goal of my project is for real estate agents price client real estate appropriately by predicting the price of
a property by using variables that are relevant to the property. Another goal for this project is for real estate agents
to be able to understand which feature (data variable) affect the price of a property
"""
# have to process zip code data - done
# then data exploration and visualisation - done
# then business case at top of file
# save all visualization as images
# clean up file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from folium import plugins
from matplotlib.pyplot import figure
import folium
from folium.plugins import HeatMap
from keplergl import KeplerGl
import operator

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
"""

# prints out the count, mean, std, min, max, 25%, 50%, 75%, and max for every numerical column in the dataframe
print(data.describe())

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

print(data.isnull().sum())

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
# data.drop(['address'], axis=1, inplace=True)

# drops the id column because it's not important
data.drop(['id'], axis=1, inplace=True)

# drops the type column because it's not important
data.drop(['type'], axis=1, inplace=True)

# prints out if there is any null values
print(pd.isnull(data).sum())

# Outliers
# boxplot of baths column in dataframe
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot Of Number Of Baths In Each Property", fontsize=22)
plt.xlabel("baths", fontsize=14)
sns.boxplot(x=data.baths, color='#2CBDFE')
plt.show()
# Summary: Most properties have between 2 and 4 bathrooms. One property has 40 bathrooms which is an outlier

# boxplot of beds column in dataframe
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot Of Number Of Beds In Each Property", fontsize=22)
sns.boxplot(x=data.beds, color='#2CBDFE')
plt.xlabel("beds", fontsize=14)
plt.show()
# Summary: Most properties have between 2 and 4 bedrooms. One property has 40 bedrooms and another one with 20 bedrooms, these properties are seen as an outlier

# boxplot of list_price column in dataframe
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot Of Price For Each Property", fontsize=22)
plt.ticklabel_format(style='plain')
plt.xlabel("prices", fontsize=14)
sns.boxplot(x=data.list_price, color='#2CBDFE')
plt.show()
# Summary: Majority of the properties are listed for less than 5 million. However there is a few outliers. With one property being listed at 35 million

# boxplot of NeighborhoodName column in dataframe
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot Of neighborhood Names For Each Property", fontsize=22)
sns.boxplot(x=data.NeighborhoodName, color='#2CBDFE')
plt.xlabel("neighborhood names", fontsize=14)
plt.show()
# Summary: Majority of properties are in neighborhoods between 2 and 19. NOTE PLEASE SEE variable legend TEXT FILE TO SEE APPROPRIATE LEGEND

# boxplot of zipCodeName column in dataframe
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title("Boxplot Of Zip Code Names For Each Property", fontsize=22)
plt.xlabel("zip code names", fontsize=14)
sns.boxplot(x=data.ZipCodeName, color='#2CBDFE')
plt.show()

# Summary: Majority of properties are in zip codes between 10 and 22. NOTE PLEASE SEE variable legend TEXT FILE TO SEE APPROPRIATE LEGEND


# Exploratory data analysis - univariate analysis (ALL VISUALIZATIONS ARE SAVE IN data_visualizations DIRECTORY.)
# data visualization for baths column
number_baths = data.baths.value_counts()  # Gives a pandas series
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
number_baths.plot.bar(color='#2CBDFE')
plt.xticks(rotation=360)
plt.title("Bar Chart Comparing The Number Of Baths And The Number Of Properties", fontsize=22)
plt.xlabel("Number Of Baths", fontsize=14)
plt.ylabel("Number Of Properties", fontsize=14)
plt.show()

# Summary: Over 1,000 properties have 2 bathrooms

# data visualization for beds column
number_beds = data.beds.value_counts()  # Gives a pandas series
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
number_beds.plot.bar(color='#2CBDFE')
plt.xticks(rotation=360)
plt.title("Bar Chart Comparing The Number Of Beds And The Number Of Properties", fontsize=22)
plt.xlabel("Number Of Beds", fontsize=14)
plt.ylabel("Number Of Properties", fontsize=14)
plt.show()

# Summary: Over 700 properties have 2 bedrooms, followed closely by 600 properties that have 3 bedrooms

# data visualization for is_foreclosure column
number_is_foreclosure = data.is_foreclosure.value_counts()  # Gives a pandas series
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
number_is_foreclosure.plot.bar(color='#2CBDFE')
plt.xticks(rotation=360)
plt.title("Bar Chart Comparing The Number Of Property Foreclosures And The Number Of Properties", fontsize=22)
plt.xlabel("Number Of Is_Foreclosure", fontsize=14)
plt.ylabel("Number Of Properties", fontsize=14)
plt.show()

# Summary: Over 2,000 properties are not foreclosure properties

figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
number_is_foreclosure_pie = data.is_foreclosure.value_counts(normalize=True)
number_is_foreclosure_pie.plot.pie(radius=2200, frame=True, autopct='%.0f%%')
plt.legend(["no", "yes"], loc='upper right')
plt.title("Pie Chart Of Percentage of Foreclosures", fontsize=22)
plt.show()

# Summary: 96% of properties are not foreclosure properties

# data visualization for CityName column
city_labels = ['Miami', 'Miami Beach', 'Coconut Grove', 'Coral Gables', 'Doral', 'Hialeah']
default_city_ticks = range(len(city_labels))
number_city_names = data.CityName.value_counts()  # Gives a pandas series
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
number_city_names.plot.bar(color='#2CBDFE')
plt.title("Bar Chart Of The Number Of Properties In Each City", fontsize=22)
plt.xticks(default_city_ticks, city_labels, rotation=45)
plt.xlabel("Number Of City_Names", fontsize=14)
plt.ylabel("Number Of Properties", fontsize=14)
plt.show()

# Summary: Over 2000 properties are in Miami

# data visualization for CountyName column
county_labels = ['MIAMI-DADE COUNTY', 'MIAMI-DADE', 'Miami-Dade', 'OTHER', 'Miami-Dade County', 'DADE']
default_county_ticks = range(len(county_labels))
number_county_names = data.CountyName.value_counts()  # Gives a pandas series
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
number_county_names.plot.bar(color='#2CBDFE')
plt.xticks(default_county_ticks, county_labels, rotation=360)
plt.title("Bar Chart Of The Number Of Properties In Each County", fontsize=22)
plt.xlabel("Number Of County_Names", fontsize=14)
plt.ylabel("Number Of Properties", fontsize=14)
plt.show()

# Summary: over 1,600 properties are in the county of MIAMI-DADE COUNTY

figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
number_county_names_pie = data.CountyName.value_counts(normalize=True)
number_county_names_pie.plot.pie(radius=2200, frame=True, autopct='%.0f%%')
plt.legend(county_mapper, loc='upper right')
plt.title("Pie Chart Of The Number Of Properties In Each County", fontsize=22)
plt.show()

# Summary: 78% of properties are in the county of MIAMI-DADE COUNTY

# data visualization for NeighborhoodName column
neighborhood_labels = ['Upper Eastside', 'Brickell', 'Overtown', 'Downtown Miami',
                       'North-East Coconut Grove', 'Miami Islands', 'San Marco Island', 'Coral Way',
                       'Watson Island', 'Little Haiti', 'Biscayne Island', 'Liberty City',
                       'Wynwood - Edgewater', 'South-West Coconut Grove', 'Allapattah',
                       'Coral Gables', 'Fair Isle', 'Douglas Park', 'Flagami', 'Shenandoah',
                       'Little Havana', 'Alameda - West Flagler', 'North Bayfront', 'Virginia Key']
default_neighborhood_ticks = range(len(neighborhood_labels))
number_neighborhood_names = data.NeighborhoodName.value_counts()  # Gives a pandas series
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
number_neighborhood_names.plot.bar(color='#2CBDFE')
plt.xticks(default_neighborhood_ticks, rotation=360)
plt.title("Bar Chart Of The Number Of Properties In Each Neighborhood", fontsize=22)
plt.xlabel("Number Of Neighborhood_Names", fontsize=14)
plt.ylabel("Number Of Properties", fontsize=14)
plt.show()

# Summary: Over 600 properties are in neighborhoods Flagami (19) and Upper Eastside (1). NOTE PLEASE SEE variable legend TEXT FILE TO SEE APPROPRIATE LEGEND

neighborhood_names = dict(sorted(number_neighborhood_names.items(), key=operator.itemgetter(1), reverse=True))
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
number_neighborhood_names.plot.pie(radius=2200, frame=True, autopct='%.0f%%')
plt.legend(neighborhood_names, loc="upper right")
plt.title("Pie Chart Of The Number Of Properties In Each Neighborhood", fontsize=22)
plt.show()

# Summary: The neighborhoods Flagami (19) and Upper Eastside (1) make up 39% of all properties

# Exploratory data analysis - bivariate analysis (ALL VISUALIZATIONS ARE SAVE IN data_visualizations DIRECTORY.)
# data for beds and price column
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.beds, data.list_price, color='#2CBDFE')
plt.title("Scatter plot of beds and price", fontsize=22)
plt.xlabel("beds", fontsize=14)
plt.ylabel("price in $", fontsize=14)
plt.show()

# calculate correlation for beds and price column
correlation = np.corrcoef(data.beds, data.list_price)
print("correlation of beds and price: ", correlation)

# summary: The beds and list_price column have a correlation of 0.26256629, which means these columns have little correlation

# data for baths and price column
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.baths, data.list_price, color='#2CBDFE')
plt.title("Scatter plot of baths and price", fontsize=22)
plt.xlabel("baths", fontsize=14)
plt.ylabel("price in $", fontsize=14)
plt.show()

# calculate correlation for baths and price column
correlation2 = np.corrcoef(data.baths, data.list_price)
print("correlation of baths and price: ", correlation2)

# Summary: The baths and price column have a correlation of 0.42760621, which means these columns are have an even correlation

figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.baths, data.beds, color='#2CBDFE')
plt.title("Scatter plot of baths and beds", fontsize=22)
plt.xlabel("baths", fontsize=14)
plt.ylabel("beds", fontsize=14)
plt.show()

# calculate correlation for baths and price column
correlation3 = np.corrcoef(data.baths, data.beds)
print("correlation of baths and beds: ", correlation3)

# summary: The baths and beds column have a correlation of 0.78879735, which means these columns are very correlated

# Exploratory data analysis - multivariate analysis (ALL VISUALIZATIONS ARE SAVE IN data_visualizations DIRECTORY.)
# heatmap of all variables in dataframe
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title("Heatmap Of Property Dataframe Variables", fontsize=22)
sns.heatmap(data.corr(), annot=True, cmap='Reds')
plt.show()

# Geospatial visualization (PLEASE NOTE THAT ALL HTML FILES WITH VISUALIZATIONS ARE IN THE html_files DIRECTORY AND
# SAVING THE MAPS COMMAND IS COMMENTED. BUT FEEL FREE TO UNCOMMENT AND RUN YOURSELF)
map = folium.Map(location=[data.latitude.mean(), data.longitude.mean()], zoom_start=12, control_scale=True)
data['latitude'] = data['latitude'].astype(np.float)
data['longitude'] = data['longitude'].astype(np.float)
for index, row in data.iterrows():
    folium.CircleMarker((row.latitude, row.longitude), popup=f"address: {row.address}", radius=6, color="#3186cc", fill=True,
                        fill_color="#3186cc").add_to(map)
# map.save('geospatial_data_visualization.html')

map2 = folium.Map(location=[data.latitude.mean(), data.longitude.mean()], zoom_start=12, control_scale=True)
for index, row in data.iterrows():
    folium.CircleMarker((row.latitude, row.longitude), popup=f"address: {row.address}", radius=2, weight=2, color='red',
                        fill_color='red', fill_opacity=.5).add_to(map2)
heat_data = [[row['latitude'], row['longitude']] for index, row in data.iterrows()]
hm = HeatMap(heat_data, min_opacity=0.4, blur=18)
map2.add_child(plugins.HeatMap(data=heat_data, radius=25, blur=10))
# map2.save('geospatial_heatmap_data_visualization.html')

custom_config = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [],
      "interactionConfig": {}
    },
    "mapState": {
      "bearing": -4.928571428571431,
      "dragRotate": True,
      "latitude": 25.77062694667296,
      "longitude": -80.23216984426615,
      "pitch": 49.18440507924836,
      "zoom": 10.655984704565685,
      "isSplit": False
    },
    "mapStyle": {
      "styleType": "muted_night",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "label": False,
        "road": False,
        "border": False,
        "building": False,
        "water": True,
        "land": True
      }
    }
  }
}

kepler_map = KeplerGl(height=500)
kepler_map.add_data(data=data, name='Miami Property Data')
kepler_map.config = custom_config
# kepler_map.save_to_html(file_name="html_files/keplerDataVisualizationBaseMap.html")

# print(data.head())
# print(data.info())
"""
UPDATED DATAFRAME 
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   address                      2119 non-null   object 
 1   state                        2119 non-null   object 
 2   baths                        2119 non-null   float64
 3   beds                         2119 non-null   float64
 4   list_price                   2119 non-null   float64
 5   is_foreclosure               2119 non-null   int64  
 6   latitude                     2119 non-null   float64
 7   longitude                    2119 non-null   float64
 8   CityName                     2119 non-null   int64  
 9   CountyName                   2119 non-null   int64  
 10  NeighborhoodName             2119 non-null   int64  
 11  ZipCodeName                  2119 non-null   int64  
 12  StatusInactive               2119 non-null   int32  
 13  StatusActive                 2119 non-null   int32  
 14  StatusPending                2119 non-null   int32  
 15  StatusPublicRecord           2119 non-null   int32  
 16  TypeCondo/Coop               2119 non-null   int32  
 17  TypeSingleFamilyResidential  2119 non-null   int32  
 18  TypeMulti Family             2119 non-null   int32  
 19  TypeOther                    2119 non-null   int32  
 20  TypeTownhouse                2119 non-null   int32  
 21  TypeCondo/Co-op              2119 non-null   int32  
 22  TypeGarage                   2119 non-null   int32  
"""

# save data to new csv file
data.to_csv('processed_data.csv')


