#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:14:15 2020

@author: aidanosullivan
"""
import pandas as pd
import os 
import numpy as np
import seaborn as sns
import sklearn as skl
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from autoimpute.imputations import MultipleImputer
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
print(c)

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

prod_comps = ratings_links_meta['top_production_companies'].str.join(sep = '*').str.get_dummies(sep='*')
prod_comps.head()

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

ratings_links_meta.shape
ratings_links_meta.columns

#finally, let's drop all the old dictionary columns
ratings_links_meta.drop(['production_countries', 'spoken_languages', 'genres', 'production_companies'], axis =1, inplace = True)


a = [[x.replace(" Corporation", '') for x in l] for l in ratings_links_meta['top_production_companies']]
for lists in a:
    for i in range(len(lists)):
        if (lists[i] == 'Warner Bros.' or lists[i] == 'Metro-Goldwyn-Mayer (MGM)' or lists[i] == 'Paramount Pictures' 
        or lists[i] == 'Columbia Pictures' or lists[i] == 'Universal Pictures' or lists[i] == 'Twentieth Century Fox Film'
        or lists[i] == 'Canal+' or lists[i] == 'United Artists' or lists[i] == 'New Line Cinema' or lists[i] == 'RKO Radio Pictures' 
        or lists[i] == 'Walt Disney Pictures' or lists[i] == 'Touchstone Pictures' or lists[i] == 'TriStar Pictures'):
            next
        else:
            lists[i] = re.sub('^.*', "Other Production Company", lists[i])
print(a)
print(ratings_links_meta['genre_name'])
print(ratings_links_meta['top_production_companies'])
b = ratings_links_meta['top_production_companies']
b = [[x.replace(" Corporation", '') for x in l] for l in ratings_links_meta['top_production_companies']]
print(b)
aba = Counter(x for xs in a for x in set(xs))

aba_data = a.str.join(sep = '*').str.get_dummies(sep='*')
print(a[2])          
import re
for i in range(len(a[2])):
    if a[2][i] == 'Warner Bros.':
        next
    else:
        print(i)
        a[2][i] = re.sub('^.*', "", a[2][i])
len(a[0])
Counter(a[2])     
range(len(a[2])) 
a[2]

a = pd.DatetimeIndex(ratings_links_meta['release_date']).month
a.astype(pd.Int32Dtype())
print(a)
a[-4]
a.isna().sum()
type(a[-4])
aa = str(a[-4])
print(str(aa))
str(aa) == 'nan'
a[-4] == np.nan 
for i in range(len(a)):
    j = str(a[i])
    if j == 'nan':
        i+=1
    j = j.replace(".0", "")
    a[i] = int(j)



def getGenres(string):
    final_list = []
    string = string.replace("'", '"') 
    dict_string = json.loads(string)
    for i in dict_string:
        final_list.append(i["name"])
    return final_list

#undo the genres dictionary
ratings_links_meta['genre_name'] = ratings_links_meta['genres'].apply(getGenres)
ratings_links_meta.drop(['genres'], axis = 1, inplace = True)
genres_data = ratings_links_meta['genre_name'].str.join(sep = '*').str.get_dummies(sep='*')
genres_columns = genres_data.columns
ratings_links_meta = pd.concat([ratings_links_meta, genres_data], axis =1)



#undo production companies dictionary
ratings_links_meta['production_companies']
ratings_links_meta['production_comp'] = ratings_links_meta['production_companies'].apply(getGenres)

#undo production countries
ratings_links_meta['production_countries']
ratings_links_meta['spoken_languages']

#undo spoken languages
def getLanguages(string):
    final_list = []
    string = string.replace("'", '"') 
    string = string.replace("\\", "")
    dict_string = json.loads(string)
    for i in dict_string:
        final_list.append(i["iso_639_1"])
    return final_list

ratings_links_meta['language'] = ratings_links_meta['spoken_languages'].apply(getLanguages)
ratings_links_meta.drop('language', axis =1, inplace = True)
print(ratings_links_meta['language'])
#let's try to impute some data. I will use AutoImpute
ratings_links_meta.isna().sum()
#currently, we have missing values in 4 columns - original_language, release_date, runtime, and status


ratings_links_meta['original_language']
ratings_links_meta['status'].value_counts()
ratings_links_meta.status.unique()




#quickly, let's just drop the rows with NA values and try to run a few quick models
#later, I will go back and try to impute the data

data = ratings_links_meta.dropna()
#let's also drop all of the indexing columns because we don't need those are unnecessary
data.drop(["movieId", "imdbId", "tmdbId", "id", "imdb_id"], axis = 1, inplace = True)


#let's start with a quick random forest model to see what that's like 

sns.distplot(ratings_links_meta['rating'])
sns.distplot(np.log1p(ratings_links_meta['revenue']))
sum(ratings_links_meta['revenue'] == 0)
sns.distplot(ratings_links_meta['popularity'])
sns.distplot(ratings_links_meta['vote_count'])

ratings_links_meta[ratings_links_meta['popularity'].astype('float64') > 200]
ratings_links_meta.iloc[30390,]
ratings_links_meta.iloc[24429,]


ratings_links_meta[ratings_links_meta['movieId'] ==135887]['popularity']

ratings_links_meta['popularity'].astype('float64')

#todo - get distrubtion of rating column before grouping by userId
#also find average score by userId
#could use median instead of mean


#df.convert_objects(convert_numeric=True)
#movies_meta = movies_meta[:,'id'].to_numeric(errors = "coerce")
movies_meta['id'] = pd.to_numeric(movies_meta['id'],errors='coerce')
#movies_meta['id'].notna()
movies_meta.dropna(subset=['id'],inplace=True)
movies_meta[movies_meta['id'].notna(),]

# imdb_id_col = movies_meta.loc[:,["imdb_id"]]
# pract = movies_meta['imdb_id']
# print(pract)

# a = (imdb_id_col.iloc[0,0])
# print(a)
# a = a.strip("tt")

# print(movies_meta.columns)
# p = []
# for i in range(len(imdb_id_col)):
#     imdb_id_col.iloc[i,0].strip("tt")
# print(imdb_id_col)    
# range(len(imdb_id_col))
# p = imdb_id_col.iloc[0].replace("tt", '')

# print(imdb_id_col.iloc[3,0].strip("tt"))
# print(p)
# pract = map(character, pract)
# p = ([s.strip("tt") for s in pract])
# print(imdb_id_col)

# type(imdb_id_col.iloc[2,0])
dataset = data




ele = list(ratings_links_meta['production_companies'])[4]
ele = ele.replace("'", '"')
print(ele)
ele_dict = json.loads(ele)
print(ele_dict)
for j in ele_dict:
    mylist.append(j["name"])


print(mylist)
len(mylist)
print()
ratings_links_meta.drop([28], axis = 0, inplace  = True)
mylist = []
count = 0
for i in list(ratings_links_meta["production_companies"]):
    i = i.replace("'", '"')
    print(i)
    dict_i = json.loads(i)
    count += 1
    print(count)
    for j in dict_i:
        mylist.append(j["name"])
        
        
ratings_links_meta.loc[19777, 'production_companies']
stupid = list(ratings_links_meta["spoken_languages"])
stupid[17008]
        

first = ratings_links_meta['spoken_languages'][0]
print(first)
first = first.replace("'", '"')
print(first)
repr(first)
first = json.loads(first)
print(first)
print(type(first[0]))



for i in ratings_links_meta.iloc[28, :]:
    print(i)

mylist = []
for i in list(ratings_links_meta["spoken_languages"]):
    i = i.replace("'", '"')
    i = i.replace('\\', '')
    print(i)
    
    dict_i = json.loads(i)
    for j in dict_i:
        mylist.append(j["iso_639_1"])  
        
      
count = Counter(mylist)
count.most_common(10)

ratings_links_meta.shape

for i in range(len(ratings_links_meta.production_companies)):
    if "Procirep" in ratings_links_meta['production_companies'][i]:
        print(i)
    

    
 print(ratings_links_meta['production_companies'][2])    

print
import json

def getGenres(string):
    final_list = []
    string = string.replace("'", '"') 
    dict_string = json.loads(string)
    for i in dict_string:
        final_list.append(i["name"])
    return final_list

mylist = []
for i in list(ratings_links_meta["production_countries"]):
    dict_i = ast.literal_eval(i)
    for j in dict_i:
        mylist.append(j["iso_3166_1"])
        
dict_i = [{'iso_3166_1': 'CA', 'name': 'Canada'}, {'iso_3166_1': 'DE', 'name': 'Germany'}, {'iso_3166_1': 'GB', 'name': 'United Kingdom'}, {'iso_3166_1': 'US', 'name': 'United States of America'}]       
print(dict_i)

for i in dict_i:
    print(i)
    mylist.append(i['iso_3166_1'])
print(mylist)      
def getProductionCountries(string):
    final_list = []
    dict_i = ast.literal_eval(string)
    for j in dict_i:
        final_list.append(j["iso_3166_1"])
    return final_list

Prod = ratings_links_meta['production_countries'].apply(getProductionCountries)       
print(Prod[0:10])   
print(ratings_links_meta['production_countries'])
print(Prod)  
len(Prod)
print(mylist) 
aa = "[{'iso_3166_1': 'US', 'name': 'United States of America'}]"
a_a = '[{"iso_3166_1": "US", "name": "United States of America"}]'
import ast
ast.literal_eval(aa)
json.loads(a_a)
type(aa)
type(a_a)
ratings_links_meta.columns
 a= [{"name": "Procirep", "id": 311}, {"name": "Constellation Productions", "id": 590}, {"name": "France 3 Cinéma", "id": 591}, {"name": "Claudie Ossard Productions", "id": 592}, {"name": "Eurimages", "id": 850}, {"name": "MEDIA Programme of the European Union", "id": 851}, {"name": "Cofimage 5", "id": 1871}, {"name": "Televisión Española (TVE)", "id": 6639}, {"name": "Tele München Fernseh Produktionsgesellschaft (TMG)", "id": 7237}, {"name": "Club d"Investissement Média", "id": 8170}, {"name": "Canal+ España", "id": 9335}, {"name": "Elías Querejeta Producciones Cinematográficas S.L.", "id": 12009}, {"name": "Centre National de la Cinématographie (CNC)", "id": 18367}, {"name": "Victoires Productions", "id": 25020}, {"name": "Constellation", "id": 25021}, {"name": "Lumière Pictures", "id": 25129}, {"name": "Canal+", "id": 47532}, {"name": "Studio Image", "id": 68517}, {"name": "Cofimage 4", "id": 79437}, {"name": "Ossane", "id": 79438}, {"name": "Phoenix Images", "id": 79439}]
repr(a)      

mylist = []
for i in list(ratings_links_meta["production_companies"]):
    #print(i)
    #i = i.replace("'", '\\"')
    
    #i = i.replace('\\', '')
    #print(i) 
    #dict_i = json.loads(i)
    dict_i = ast.literal_eval(i)
    for j in dict_i:
        mylist.append(j["name"]) 
print(mylist[0:10])

def getProductionCompanies(string):
    final_list = []
    dict_i = ast.literal_eval(string)
    for j in dict_i:
        final_list.append(j["name"]) 
    return final_list

for i in mylist:
    if i == "Columbia Pictures Corporation":
        #print(i)
         i = i.replace("Corporation", "")
        mylist[i]
a = ratings_links_meta['production_companies'].apply(getProductionCompanies) 
a = [[x.replace(" Corporation", '') for x in l] for l in a]

c = Counter(x for xs in a for x in set(xs))
print(a) 
Counter(a)
for arrays in a:
    for i in range(len(arrays)):
        arrays[i] = arrays[i].replace(" Corporation", '')
        #[w.replace(" Corporation", '') for w in i]
 

         
mylist = [w.replace(" Corporation", '') for w in mylist]
len(mylist)
c = Counter(mylist)
len(c)
c.most_common(13)  
# from sklearn.preprocessing import MultiLabelBinarizer
# let's separate out the genres as binary variables
# practice = MultiLabelBinarizer()
# practice.fit_transform(dataset['genres'].apply(getGenres))

print(dataset['genres'])
dataset.columns
dataset.drop(genres_data.columns, axis = 1)


dataset['genres'].isnull().sum()


dataset['genre_name'] = dataset['genres'].apply(getGenres)

print(dataset['genre_name'])
#genres_data = pd.get_dummies(dataset['genre_name'].apply(pd.Series).stack()).sum(level=0)
#where x in function lamba is the row
#genres_data = dataset[['genre_name']].groups.apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0, downcast='infer')


#this one works
genres_data = dataset['genre_name'].str.join(sep = '*').str.get_dummies(sep='*')
dataset['genre_name']
genres_data.shape
dataset.shape
len(dataset['genres'])

dataset = pd.concat([dataset, genres_data], axis =1, ignore_index=True, sort=False)

'*'.join(['AAA','BBB','CCC'])

#practice AutoImpute
messy_df = ratings_links_meta
ratings_links_meta = messy_df
messy_df.shape
ratings_links_meta.shape


pd.set_option('display.max_rows', 120)
messy_df.dtypes
for columns in messy_df.columns:
   print(messy_df[columns][0], messy_df[columns].dtype)
   
   
messy_df.columns
   #print(f" {b} : {c}")

b = messy_df.columns[20]
messy_df[b].dtype
class(messy_df.columns[0])

messy_df = ratings_links_meta

messy_df.drop(['genres', 'imdb_id', 'original_language', 'original_title', 
               'production_companies', 'production_countries', 'spoken_languages', 
               'genre_name', 'top_production_companies','production_locations', 'language', 'title'], 
              axis =1, inplace = True)

messy_df['popularity'] = messy_df['popularity'].astype('float64')
messy_df['budget'] = messy_df['budget'].astype(int)
messy_df['adult'] = (messy_df['adult'] == 'True').astype(int)
messy_df['video'] = (messy_df['video'] == 'True').astype(int)



prac = pd.concat([messy_df, genres_data], axis = 1)

imp = MultipleImputer()
print(imp)
imp.fit_transform(messy_df)

print(ratings_links_meta['release_month'])
Counter(messy_df['adult'])
messy_df.video


messy_df.video.unique()
messy_df.adult.unique()


