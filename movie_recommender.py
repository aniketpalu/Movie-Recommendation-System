import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df = pd.read_csv('D:\\Projects\\Recommendattion\\movie_dataset.csv')
			#print(df)
			#print(df.columns)
			#print(df.shape)
##Step 2: Select Features

features = ['keywords', 'cast','genres','director']
##Step 3: Create a column in DF which combines all selected features
for feature in features:
	df[feature] = df[feature].fillna('')

def combine_feature(row):
	try:
		return row['keywords'] +' '+ row['cast'] + " " + row['genres'] + ' ' +row['director']
	except:
		print('Error : \n', row)
df['combined_features'] = df.apply(combine_feature, axis=1)
#print(df['combined_features'].head())
##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
cv_fit=cv.fit_transform(df['combined_features'])
##Step 5: Compute the Cosine Similarity based on the count_matrix
similarity = cosine_similarity(cv_fit )
movie_user_likes = input("Enter your loved movie")

## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(similarity[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies, key= lambda x: x[1], reverse = True)

## Step 8: Print titles of first 50 movies
i = 0
for mo in sorted_similar_movies:
	print(get_title_from_index(mo[0]))
	i = i+ 1
	if i>50:
		break