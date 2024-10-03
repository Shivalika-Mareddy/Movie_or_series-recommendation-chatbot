import pandas as pd
import numpy as np
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
path='C:/Users/shiva/Desktop/movie_recommender/netflix_titles.csv.zip'
df=pd.read_csv(path)
dates=['October 2, 2013','April 14, 2013','September 23, 2003','September 25, 2003','January 7, 2008','March 31, 2010','September 16, 2012','July 13, 2016','April 1, 2015','August 31, 2015']
nan_row=df[df['date_added'].isna()].index.tolist()
nan_indices = df[df['date_added'].isna()].index
df.loc[nan_indices, 'date_added'] =dates
df.replace({'type':{'Movie':0,'TV Show':1}}, inplace=True)
df['country']=df['country'].replace(np.nan,'Unknown')
df['director']=df['director'].replace(np.nan,'Unknown')
df['cast']=df['cast'].replace(np.nan,'Unknown')
df['date_added']=df['date_added'].apply(pd.to_datetime)
df['year']=df['date_added'].dt.year
df['month']=df['date_added'].dt.month_name()
df['description'] = df['description'].fillna('')
def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)


    tokens = [word.lower() for word in tokens if word.isalpha()]


    tokens = [word for word in tokens if word not in stop_words]


    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

df['listed_in_processed'] = df['listed_in'].apply(preprocess_text)
df['description_processed'] = df['description'].apply(preprocess_text)

df['content'] = df['title'] + ' ' + df['listed_in_processed'] + ' ' + df['description_processed']
df['description'] = df['description'].fillna('')
df['content'] = df['title'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['listed_in'] + ' ' + df['description']

stop_words = stopwords.words('english')
tfidf = TfidfVectorizer(stop_words=stop_words)

tfidf_matrix = tfidf.fit_transform(df['content'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


indices = pd.Series(df.index, index=df['title']).drop_duplicates()
def get_recommendations(title, content_type, release_year, cast, country, cosine_sim=cosine_sim):
    """Function to get movie recommendations based on multiple filters."""
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]


    filtered_df = df.iloc[movie_indices]

    if content_type != '':
        content_type = 'Movie' if content_type == '0' else 'TV Show'
        filtered_df = filtered_df[filtered_df['type'] == content_type]

    if release_year != '':
        filtered_df = filtered_df[filtered_df['release_year'] == int(release_year)]

    if cast != '':
        filtered_df = filtered_df[filtered_df['cast'].str.contains(cast, case=False, na=False)]

    if country != '':
        filtered_df = filtered_df[filtered_df['country'].str.contains(country, case=False, na=False)]

    recommendations = filtered_df['title'].tolist()


    if not recommendations:
        recommendations = [df['title'].iloc[movie_indices[0]]]
        recommendations += df['title'].iloc[movie_indices[1:3]].tolist()

    return recommendations
st.title('Movie/TV Show Recommendation System')

# Input fields for user query
title_input = st.text_input('Enter a movie or TV show title:')
content_type_input = st.selectbox('Content Type:', ['Any', 'Movie', 'TV Show'])
release_year_input = st.text_input('Release Year (optional):')
cast_input = st.text_input('Cast (optional):')
country_input = st.text_input('Country (optional):')

# Convert content type to correct format
if content_type_input == 'Any':
    content_type_input = ''
elif content_type_input == 'Movie':
    content_type_input = 'Movie'
else:
    content_type_input = 'TV Show'

# Button to generate recommendations
if st.button('Get Recommendations'):
    recommendations = get_recommendations(
        title=title_input,
        content_type=content_type_input,
        release_year=release_year_input,
        cast=cast_input,
        country=country_input,
        cosine_sim=cosine_sim
    )

    if recommendations:
        st.write(f"Recommendations based on '{title_input}':")
        for idx, rec in enumerate(recommendations, 1):
            st.write(f"{idx}. {rec}")
    else:
        st.write("No recommendations found.")
