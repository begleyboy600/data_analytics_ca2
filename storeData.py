import pandas as pd
import requests
import csv

url = "https://mashvisor-api.p.rapidapi.com/city/listings"

"""
PLEASE NOTE: The API required a zip code to be passed into the API query call so I used the following zip codes to get 
the data:
[33160, 33154, 33141, 33140, 33139, 33035, 33034, 33189, 33186, 33133, 33185, 33175, 33129, 33131, 33132, 33136, 33128, 
 33130, 33126, 33178, 33182, 33166, 33010, 33013, 33018, 33169, 33138, 33161, 33181, 33149]
 """

querystring = {"page":"1","state":"FL","city":"Miami","zip_code":"33149"}
headers = {
	"X-RapidAPI-Key": "e45e16fe17mshdee338759b8fe19p15b612jsnef7bfb3c8fe1",
	"X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
}

# requests response from api and stores data in response variable
response = requests.request("GET", url, headers=headers, params=querystring)

res = response.text
# print(res)

# reads response as json
json_data = pd.read_json(res)

# dig into json response structure
content = json_data['content']
data = content['properties']
details = [data]
# print(details)

# converts data into dataframe
data_df = pd.DataFrame(data)

# print first 5 rows of the dataframe
print(data_df.head())

# passes dataframe into a csv file called data.csv
data_df.to_csv('data.csv', index=False, mode='a', header=False)

# https://keras.io/examples/structured_data/structured_data_classification_from_scratch/

