#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:14:15 2020

@author: aidanosullivan
"""
import pandas as pd
import os 
import numpy as np


from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

import json
import ast

pd.options.display.max_seq_items = None
pd.set_option('display.max_columns', None)



os.chdir("/Users/aidanosullivan/Desktop/UCLA_Extension/Capstone_Project/Data")

links = pd.read_csv("links.csv")
links_small = pd.read_csv("/Users/aidanosullivan/Desktop/UCLA_Extension/Capstone_Project/Data/links_small.csv")
credits = pd.read_csv("/Users/aidanosullivan/Desktop/UCLA_Extension/Capstone_Project/Data/credits.csv")
keywords = pd.read_csv("/Users/aidanosullivan/Desktop/UCLA_Extension/Capstone_Project/Data/keywords.csv")
movies_meta = pd.read_csv("/Users/aidanosullivan/Desktop/UCLA_Extension/Capstone_Project/Data/movies_metadata.csv")
ratings = pd.read_csv("/Users/aidanosullivan/Desktop/UCLA_Extension/Capstone_Project/Data/ratings.csv")
ratings_small = pd.read_csv("/Users/aidanosullivan/Desktop/UCLA_Extension/Capstone_Project/Data/ratings_small.csv")



keywords.shape
movies_meta.shape
ratings.shape #26 million rows!!!
ratings_small.shape
links_small.shape

#value_counts()


#first, let's condense the ratings dataframe down by grouping it on critic ID
ratings = ratings.groupby(['movieId']).mean()

#first merge:
ratings_links = pd.merge(ratings, links, on = ['movieId'])

#then, before any further merges, we also have to clean up the imdb_id column in movies_meta, 
#because it has some character values attached to the Id values
movies_meta['imdb_id'] = movies_meta['imdb_id'].astype(str).str.replace('\D+', '')
movies_meta = movies_meta.drop(['homepage', 'overview', 'poster_path', 'tagline'], axis=1)
print(movies_meta.loc[:,['imdb_id']])

#at this point, the column imdb_id in movies_meta is an object instead of a float
#because there are some incorrect rows in it. Therefore I need to remove the problem
#rows, which are displayed when I try to run: movies_meta['id'].astype('float64')


#the first incorrect ID is the date '1997-08-20'
movies_meta[movies_meta['id' ] == '1997-08-20']
#so let's check whats going on on that row
movies_meta.iloc[19730,:]
#it's a pretty wacked up row, not much salvaging to be done, so let's drop it
movies_meta = movies_meta.drop(19730, axis = 0)

#now rerun trying to coerce the column to float64

movies_meta[movies_meta['id' ] == '2012-09-29']
movies_meta = movies_meta.drop(29503, axis = 0)

movies_meta[movies_meta['id' ] == '2014-01-01']
movies_meta = movies_meta.drop(35587, axis = 0)


#now i can finally coerce the column to float64 once and for all
movies_meta['id'].astype('float64')


#let's coerce the column back to being being an Int
movies_meta['id'] = movies_meta['id'].astype(int)


#finally, let's merge again
ratings_links_meta = pd.merge(ratings_links, movies_meta, left_on = ["tmdbId"], right_on = ['id'])

#shorten the release_year column
ratings_links_meta['release_year'] = pd.DatetimeIndex(ratings_links_meta['release_date']).year
ratings_links_meta['release_year'] = ratings_links_meta['release_year'].astype(pd.Int32Dtype())
ratings_links_meta['release_year']

#add a months column from release_year
ratings_links_meta['release_month'] = pd.DatetimeIndex(ratings_links_meta['release_date']).month
ratings_links_meta['release_month'] = ratings_links_meta['release_month'].astype(pd.Int32Dtype())
ratings_links_meta['release_month']

#finally, let's drop the release_date column because it's no longer necessary
ratings_links_meta.drop('release_date', axis =1, inplace = True)


#drop columns that are useless for either modeling or interpreation or both
ratings_links_meta.drop(['userId', 'timestamp'], axis = 1, inplace = True)

#make "belongs_to_collection" a binary variable instead of a most empty list of dictionaries
ratings_links_meta['collection'] = np.where(ratings_links_meta['belongs_to_collection']
                                            .isna(),0,1)
ratings_links_meta.drop('belongs_to_collection', axis = 1, inplace = True)


#if we check which columns have NaN values, some columns have 10+ Nan values, but
#many have only 3 rows with Nan value
column_names = ratings_links_meta.columns.tolist()
for i in range(len(ratings_links_meta.columns)):
    b = column_names[i]
    c = ratings_links_meta.iloc[:,i].isna().sum()
    print(f"{b} : {c}")
    
#the code below also works, but as we add more and more columns, it will begin
#to condense the output, which makes understand the data difficult. I prefer the
#for loop
ratings_links_meta.isna().sum()

#Let's find out which rows are causing the Na values
ratings_links_meta[ratings_links_meta['popularity'].isna()]

#and let's drop those rows!
ratings_links_meta.drop([19777, 29224, 35206], axis = 0, inplace = True)

#if we rerun ratings_links_meta.isn().sum(), we see that now all the columns
#that previously had only 3 Nan values have been eliminated

#next step is to chage all the columns full of dictionaries into categorical variables
#currently, those columns are: genres, production companies, production countries, and spoken languages

#first, let's remind ourselves of the shape of our dataframe - we want to keep the same number of rows
ratings_links_meta.shape



#undo the genres dictionary
def getGenres(string):
    final_list = []
    string = string.replace("'", '"') 
    dict_string = json.loads(string)
    for i in dict_string:
        final_list.append(i["name"])
    return final_list

ratings_links_meta['genre_name'] = ratings_links_meta['genres'].apply(getGenres)


#UNDO PRODUCTION COMPANIES
def getProductionCompanies(string):
    final_list = []
    dict_i = ast.literal_eval(string)
    for j in dict_i:
        final_list.append(j["name"]) 
    return final_list

ratings_links_meta['top_production_companies'] = ratings_links_meta['production_companies'].apply(getProductionCompanies) 
ratings_links_meta['top_production_companies'] = [[x.replace(" Corporation", '') for x in l] for l in ratings_links_meta['top_production_companies']]
ratings_links_meta['top_production_companies'].shape
c = Counter(x for xs in ratings_links_meta['top_production_companies'] for x in set(xs))
len(c) #there are 23383
c.most_common(13)

#UNDO PRODUCTION COUNTRIES
def getProductionCountries(string):
    final_list = []
    dict_i = ast.literal_eval(string)
    for j in dict_i:
        final_list.append(j["name"])
    return final_list
ratings_links_meta['production_locations'] = ratings_links_meta['production_countries'].apply(getProductionCountries)
ratings_links_meta['production_locations'].shape

mylist = []
for i in list(ratings_links_meta["production_countries"]):
    dict_i = ast.literal_eval(i)
    for j in dict_i:
        mylist.append(j["name"])
        
count = Counter(mylist) #there are 159 different production countries
len(count)
top = count.most_common(30)
top_countries =  []  
for i in top:
    top_countries.append(i[0])

#UNDO SPOKEN LANGUAGES 
def getLanguages(string):
    final_list = []
    string = string.replace("'", '"') 
    string = string.replace("\\", "")
    dict_string = json.loads(string)
    for i in dict_string:
        final_list.append(i["name"])
    return final_list

ratings_links_meta['language'] = ratings_links_meta['spoken_languages'].apply(getLanguages)
ratings_links_meta['language'].shape
   

#the above code works, but on top of that I want to know the top ten spoken languages
mylist = []
for i in list(ratings_links_meta["spoken_languages"]):
    i = i.replace("'", '"')
    i = i.replace('\\', '')
    dict_i = json.loads(i)
    for j in dict_i:
        mylist.append(j["name"])  
        
count = Counter(mylist)
len(count) #there are 75 unique languages spoken
top_lang = count.most_common(20)
top_languages = []
for i in top_lang:
    top_languages.append(i[0])
    

#Let's Make all the dummy variable dataset, cut then down so only the most important
#columns are included, and then concat them to ratings_links_meta
messy_df = ratings_links_meta

ratings_links_meta.columns
#for genres dataset
genres_data = ratings_links_meta['genre_name'].str.join(sep = '*').str.get_dummies(sep='*')
genres_columns = genres_data.columns
ratings_links_meta = pd.concat([ratings_links_meta, genres_data], axis =1)

#for production companies dataset
#unfortunately, there are too many production companies to change the format to a dummy variable. 
#so let's take the top 20 production companies

#prod_comps = ratings_links_meta['top_production_companies'].str.join(sep = '*').str.get_dummies(sep='*')


#for production countries dataset
prod_countries = ratings_links_meta['production_locations'].str.join(sep = '*').str.get_dummies(sep='*')
prod_countries = prod_countries.filter(top_countries)
prod_countries.columns
ratings_links_meta = pd.concat([ratings_links_meta, prod_countries], axis =1)
#where top_countries was previouly defined as the top 30 countries where movies were filmed

#for spoken languages
lang_spok = ratings_links_meta['language'].str.join(sep = '*').str.get_dummies(sep='*')
lang_spok = lang_spok.filter(top_languages)
lang_spok.columns
ratings_links_meta = pd.concat([ratings_links_meta, lang_spok], axis =1)

#we are almost done with cleaning the data. Now it is time to see the dtypes that we have
#and to drop the redundant columns (many columns are not in dummy form)
pd.set_option('display.max_rows', 120)
ratings_links_meta.dtypes

messy_df = ratings_links_meta


movie_titles = ratings_links_meta['title']

ratings_links_meta.columns

#here are the last columns of dtype object that can be changed into floats or integers
ratings_links_meta['popularity'] = ratings_links_meta['popularity'].astype('float64')
ratings_links_meta['budget'] = ratings_links_meta['budget'].astype(int)
ratings_links_meta['adult'] = (ratings_links_meta['adult'] == 'True').astype(int)
ratings_links_meta['video'] = (ratings_links_meta['video'] == 'True').astype(int)

#finally, let's drop all the old dictionary columns
ratings_links_meta.drop(['movieId', 'genres', 'imdb_id', 'original_language', 'original_title', 
               'production_companies', 'production_countries', 'spoken_languages', 
               'genre_name', 'top_production_companies','production_locations', 'language', 'title', 'tmdbId'], 
              axis =1, inplace = True)

#last step is to write 


ratings_links_meta.to_csv("movies_data.csv", index = False, header = True)



