
#imported libraries
import json
import pandas as pd

#reading json file and appending each tweet's data to tweets list
tweets = []
for line in open('005.csv', 'r'):
    if line != '\n':
        tweets.append(json.loads(line))

#normalizing data and converting to a dataframe
data = pd.json_normalize(data = tweets, meta = [])

#converting dataframe to a csv file for storage
data.to_csv('semiclean_data.csv')
