import pandas as pd
import requests
import csv

url = "https://mashvisor-api.p.rapidapi.com/city/listings"

querystring = {"page":"1","state":"FL","city":"Miami","zip_code":"33149"}
headers = {
	"X-RapidAPI-Key": "e45e16fe17mshdee338759b8fe19p15b612jsnef7bfb3c8fe1",
	"X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

res = response.text
# print(res)

json_data = pd.read_json(res)
content = json_data['content']
data = content['properties']
details = [data]
# print(details)
data_df = pd.DataFrame(data)
print(data_df.head())
data_df.to_csv('properties.csv', index=False, mode='a', header=False)



