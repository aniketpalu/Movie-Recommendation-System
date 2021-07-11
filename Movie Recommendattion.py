import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ['london paris london', 'paris paris london']
cv = CountVectorizer()
cv_fit=cv.fit_transform(text)
#print(cv.get_feature_names())
#print (cv_fit.toarray())

similarity = cosine_similarity(cv_fit )
print(similarity)