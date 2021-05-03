# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb #to visualise data
from textblob import TextBlob

# Loading data
yt_comments = pd.read_csv("GBcomments.csv", error_bad_lines=False) #error_bad_lines to fix 'ParserError' because our data contain lines with too many fields

yt_comments.head()

# Data preprocessing
yt_comments.isna().sum() #checking missing values by getting the summation of all the missing values available

# Dropping the missing values
yt_comments.dropna(inplace=True) # inplace to keep the DF with valid entries in the same variable

"""# Applying TextBlob library to get the polarity"""

#Getting the polarity of comments
sentiment_polarity =[]
for i in yt_comments["comment_text"]:
  sentiment_polarity.append(TextBlob(i).sentiment.polarity)

# Adding the polarity list to my dataframe as a new column
yt_comments['Polarity']=sentiment_polarity


#displaying the new dataframe
yt_comments.head(10)


#Filtering positive opinions only
positives_comments = yt_comments[yt_comments['Polarity']==1]

positives_comments.shape # display how many lines & columns on positives data

positives_comments.head()



# Wordcloud presentation


from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

all_comments = ' '.join(positives_comments['comment_text']) # Join all items in a tuple into one string, using a space as separator:
#all_comments

wordcloud = WordCloud(width=1200, height= 800, stopwords=stopwords).generate(all_comments)

plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
