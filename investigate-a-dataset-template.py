#!/usr/bin/env python
# coding: utf-8

# 
# # Project: TMDB movie Data Analysis 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# # <a id='intro'></a>
# ## Introduction
# ### Project Overview
#     In this project, we have to analyze the TMDB movie dataset and then communicate our findings about it. We will use the Python libraries NumPy, pandas, and Matplotlib to make the analysis easier
#     the TMDB movie data set This data set contains information about 10,000 movies collected from the Movie Database (TMDb), including user ratings and revenue, Certain columns, like ‘cast’ and ‘genres’ and characters.
# ### Questions to be asked about the dataset:
#     We are gonna ask some questions about this dataset like:
#     Which genres are most popular?
#     What kinds of properties are associated with movies that have high revenues?
#     who is the most cast actor?
#     what is the highest rated movie with the biggest budget?
#     what is the lowest rated movie with the biggest budget?
#     what are the overrated and underrated movies based on budget and popularity?
#     what is the most month with movie releases?
#     what are the years with highest number of releases? 
#     what are the years with highest rated movies? 
#     what's average runtime?
#     what are the longest 10 movies and shortest 10 movies?
# 
#     
# 

# In[18]:


#import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import csv 
import seaborn as sns
from wordcloud import WordCloud


# <a id='wrangling'></a>
# ## Data Wrangling
#     check for cleanliness of data, delete unnecessary columns, check for NAN values and fix them, etc.
# 
# 
# ### General Properties

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
#loading the csv file and storing it in the variable "df"
df = pd.read_csv('tmdb-movies.csv')

#printing first five rows to check the data
df.head()


# 
# 
# ### Data Cleaning 
# #### Based on my observations from the first 5 rows, and from previous datasets we need to do the following:
#     1. The columns id, imdb_id, homepage, budget_adj, revenue adj are useless, hence, we need to delete them
#     2. remove dublicates
#     3. check for NAN values 
#     4. replace null values with NAN 
#     5. release_date needs to be formatted into standard date format 
#     

# ### First we remove useless columns

# In[3]:


#After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
#remove useless columns
df.drop(["id", "imdb_id", "homepage", "budget_adj", "revenue_adj"], axis=1, inplace=True)
df.head()


# ### finding the dimensions of the dataframe after modification

# In[4]:


df.shape


# There are 10865 movie entries and 16 columns 

# ### finding data information

# In[5]:


df.info()


# #### There are values missing in the following columns (director, cast, tagline, keywords, overview, genres, production companies)
#       regarding the size of the data we can afford to delete rows where cast, director, genres, overview are missing since the missing values are small portions. However, we can't do the same for tagline, keywords, and production companies so we replace their missing values with NAN

# #### deleting rows with null values:

# In[6]:


df.dropna(axis = 0, inplace = True, subset = ['genres'])
df.dropna(axis = 0, inplace = True, subset = ['cast'])
df.dropna(axis = 0, inplace = True, subset = ['overview'])
df.dropna(axis = 0, inplace = True, subset = ['director'])
df.info()


# #### In keywords and tagline and production companies, there are null values but I might not need these columns after all so I'll leave them as they are 

# #### review dataset

# In[19]:


df.hist(figsize(8,8));


# #### there are multiple 0's in revenue, budget, and runtime replace them with NAN

# In[20]:


df['runtime'] =df['runtime'].replace(0, np.NAN)

df.info()


# #### check for duplicates

# In[21]:


df.duplicated().sum()


# ####  drop duplicates snce it's only 1

# In[22]:


df = df.drop_duplicates(subset=None, keep="first", inplace=False)


# In[23]:


df.duplicated().sum()


# In[24]:


df.shape


# In[25]:


df.describe()


# #### Now we convert release_date to datetime format

# In[26]:


df.release_date = pd.to_datetime(df['release_date'])


# In[27]:


df.head()


# In[28]:


#checking if all the rows have proper datatypes
df.dtypes


# In[29]:


#checking unique values in genres
df.genres.str.get_dummies(',').stack().sum(level=1)


# In[30]:


#check for unique values
df.nunique()


# ### Now our data is ready to work on 

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > Now that we've trimmed and cleaned the data, we're ready to move on to exploration. In this section we'll compute statistics and create visualizations with the goal of addressing the research questions that we posed in the Introduction section. 
# 
# ### Research Question 1: What are the most common movie genres?

# > First thing we notice how the genres column is pipe separated and when we tried to check for unique values in returned all genres at once so we need to separate them
# 

# In[31]:


#Concatenate strings in the Series/Index with given separator.
genre_c = pd.Series(df['genres'].str.cat(sep = '|').split('|')).value_counts() 
genre_c


# #### Conclusion:

# > Drama movies are the most common genre followed by comedy and Thriller, Tv movies and Western movies are the least common genres
# 

# In[32]:


#get the visualization of this conclusion:
plot_genre = genre_c.plot.bar()
plot_genre.set(title = 'Most popular Genres')
plot_genre.set_xlabel('Genre')
plot_genre.set_ylabel('Number of Movies')
# Show the plot
plot_genre


# In[33]:


genre_c.plot.pie()
plt.show()


# ### Research Question 2: Who are the most cast actors?

# > Notice that the cast column is separated the same way as the genres column so it's basically the same process

# In[34]:


cast_c = pd.Series(df['cast'].str.cat(sep = '|').split('|')).value_counts() 
cast_c


# #### conclusion:

# > Robert De Niro is the most cast actor with 72 movies followed by Samuel L. Jackson with 71 movies, Bruce Willis comes third with 62 movies...
# Now we find the most common genre for each of these actors as well as the average rate for the movies they appeared in 

# > Here we can get the top 10 actors based on number of appearances in movies:

# In[35]:


cast_c = cast_c[:10,]
ax = sns.barplot(x= cast_c.index, y= cast_c.values)
sns.set(rc={'figure.figsize':(12,12)}, font_scale=1.5)
ax.set(xlabel='Actors', ylabel='Movies count', title = 'Top 10 actors based on the number of the appearances in movies')

#rotate x-axis' text
for item in ax.get_xticklabels():
    item.set_rotation(70)
    

plt.show()


# #### Conclusion:
# > Top 3 actors based on movie appearance are Robert De Niro, Samuel L.Jackson and Bruce Willis, we can now see a detailed visualization of the movies of the most popular actor

# ##### visualization of Robert De Niro movies:

# In[36]:


filter1 = df[df['cast'].str.contains('Robert De Niro')]
filter1.popularity.plot(kind="hist")


# In[37]:


filter1.vote_average.plot(kind="box")


# In[38]:


filter1.vote_average.mean()


# In[39]:


filter1.runtime.mean()


# #### Conclusion:
# > from the previous examples we find that the average runtime of the most popular actor movies is 115 minutes, average rating is 6.33 and the popularity seems over 12

# ### Research Question 3:  What production company produces the most movies?

# In[90]:


#split companies
companies_c = pd.Series(df['production_companies'].str.cat(sep = '|').split('|')).value_counts() 
companies_c


# In[91]:


# plot 
companies_c = companies_c[:10,]
ax = sns.barplot(x= companies_c.index, y= companies_c.values)
sns.set(rc={'figure.figsize':(12,12)}, font_scale=1.5)
ax.set(xlabel='Companies', ylabel='Movies count', title = 'Top 10 production companies based on the number movies released')

#rotate x-axis' text
for item in ax.get_xticklabels():
    item.set_rotation(70)
    

plt.show()


# #### Conclusion:
#     Universal Pictures is the company that produces most movies followed by Warner Bros. in 2nd place and Paramount pictures in 3rd

# ### Research Question 4: What are the top movies in terms of profit?

# #### First we look at the top movies based on revenue and budget:

# In[40]:


movies_revenue = df[['original_title','revenue']]
movies_budget= df[['original_title','budget']]
movies_revenue.sort_values(by="revenue", ascending=False).head(10)


# In[41]:


movies_budget.sort_values(by="budget", ascending=False).head(10)


# In[42]:


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
sns.set(rc={'figure.figsize':(12,9)}, font_scale=1.3)


ax = sns.barplot(
    movies_revenue.sort_values(by = "revenue", ascending=False).head(10).original_title, 
    movies_revenue.sort_values(by = "revenue", ascending=False).head(10).revenue)



#rotate x-axis' text
for item in ax.get_xticklabels():
    item.set_rotation(70)
    

ax.set(xlabel='movie titles', ylabel='revenue', title = 'Top 10 movies based on their revenue')
plt.show()


# In[43]:


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
sns.set(rc={'figure.figsize':(12,12)}, font_scale=1.3)


ax = sns.barplot(
    movies_budget.sort_values(by = "budget", ascending=False).head(10).original_title, 
    movies_budget.sort_values(by = "budget", ascending=False).head(10).budget)



#rotate x-axis' text
for item in ax.get_xticklabels():
    item.set_rotation(70)
    

ax.set(xlabel='movie titles', ylabel='budget', title = 'Top 10 movies based on their budget')
plt.show()


# #### Add new column called profit
# > This column gets the profit of each movie (revenue - budget) to measure how profitable the movies were and base success on this 

# In[44]:


#Add new column
df.insert(3, "profit" ,df['revenue']-df['budget'])
df.head()


# In[46]:


movies_profit= df[['original_title','profit']]
movies_profit.sort_values(by="profit", ascending=False).head(10)


# In[48]:


#draw conclusion
#To avoid errors
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
sns.set(rc={'figure.figsize':(12,12)}, font_scale=1.3)


ax = sns.barplot(
    movies_profit.sort_values(by = "profit", ascending=False).head(10).original_title, 
    movies_profit.sort_values(by = "profit", ascending=False).head(10).profit)



#rotate x-axis' text
for item in ax.get_xticklabels():
    item.set_rotation(70)
    

ax.set(xlabel='movie titles', ylabel='profit', title = 'Top 10 movies based on their profit')
plt.show()


# #### Conclusion:
# > 1. Avatar is the most successul movie based on profit, followed by Star Wars and Titanic in 3rd
# 2. The movies with the most revenue weren't necessarily the most profitable and the movies with the most budget weren't necessarily the most profitable 

# ### Research Question 5: What are the top movies based on popularity?

# In[49]:


high1 = df.nlargest(10, ['popularity'])
high1


# > These are the top 10 movies based on popularity, let's examine their budget and see if it's higher that average or not

# In[50]:


#average budget:
df['budget'].mean()


# In[51]:


# how many of the most popular movies are above average budget?
high1.query('budget > 102.3')


# In[52]:


#How many of these movies are below average budget?
high1.query('budget < 102.3')


# #### Conclusion:
# most popular movies tend to have a big budget 

# ### Research Question 6: What are the top movies based on viewer rating?

# In[53]:


high = df.nlargest(10,['vote_average'])
high


# > These are the top 10 movies based on viewer rating
# notice if we try to get the average budget by mean it wouldn't give us an accurate value because there's a movie that's 900 minutes of runtime so we use median instead 

# ##### Average runtime of the highest rated movies:

# In[54]:


high['runtime'].median()


# #### Conclusion:
# > Average runtime of the highest rated movies is 140 minutes

# ### Research Question 7: What are the most common keywords?

# In[55]:


#Creating the text variable
text = " ".join(df['keywords'].str.cat(sep = '|').split('|'))
# Creating word_cloud with text as argument in .generate() method
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
# Display the generated Word Cloud
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# > Here we have a figure of the most common keywords in the movie dataset 

# ### Research Question 8: what is the best month to release a movie?

# > Best month to release a movie can be decided by the highest revenue

# In[56]:


#first create a column called month and store months in it
df['month'] = df['release_date'].apply(lambda x: x.month)
df.head()


# In[57]:


monthly_revenue = df.groupby('month')['revenue'].sum()
monthly_revenue


# In[58]:


from importlib import reload
plt=reload(plt)
#Now we plot the results
plt.bar([1,2,3,4,5,6,7,8,9,10,11,12], monthly_revenue, tick_label =[1,2,3,4,5,6,7,8,9,10,11,12])
# title and labels
plt.ylabel('Revenue')
plt.xlabel('Month')
plt.title('Revenue by Month released')


# #### Conclusion:
# > June is the most profitable month followed by November and December

# ### Research question 9: Is the budget related to a higher average vote?

# In[60]:


from importlib import reload
plt=reload(plt)
plt.scatter(x=df['budget'], y=df['vote_average'])
plt.title("budget vs rating")
plt.xlabel("Budget")
plt.ylabel('Vote Average')
plt.show()


# #### Conlusions:
# > From this scatter plot we can see that a higher budget does have a correlation with a higher rate, but many highest rated movies have lower budgets

# ### Research Question 10: what's the correlation between runtime and vote average, budget and popularity?

# In[61]:


plt.scatter(x=df['runtime'], y=df['vote_average'])
plt.title("runtime vs rating")
plt.xlabel("Runtime")
plt.ylabel('Vote Average')
plt.show()


# In[62]:


plt.scatter(x=df['runtime'], y=df['popularity'])
plt.title("runtime vs popularity")
plt.xlabel("Runtime")
plt.ylabel('popularity')
plt.show()


# In[63]:


plt.scatter(x=df['runtime'], y=df['budget'])
plt.title("Runtime vs budget")
plt.xlabel("Runtime")
plt.ylabel('budget')
plt.show()


# #### Conclusion: 
# > 1. The higher the runtime, the bigger the budget gets
# 2. Movies shorter than 200 minutes are way more popular than movies longer than 200 minutes
# 3. There isn't a big correlation between rating and runtime. however, longer movies have higher ratings 

# ### Research Question 11: Who are the most successful directors?
#     Most successful director is the one who generated the most revenue

# In[87]:


dir_rev = df.groupby(['director']).sum()['revenue'].nlargest(10)
dir_rev


# In[88]:


dir_rev.plot(kind = 'bar', figsize=(13,6))
plt.title("Top 10 Directors by largest Revenue")
plt.xticks(rotation=70)
plt.xlabel("Director")
plt.ylabel("Total Revenue")
plt.show()


# #### Conclusion: 
# > 1. Steven Spielberg is the most successful director in terms of revenue 
# 2. he's followed by Peter Jackson, while James Cameron comes 3rd

# ### Research Question 12: How did the runtime of movies change over the years? What Movie has the longest runtime? what movie has the shortest runtime? what's the average movie runtime?

# #### Average movie runtime:
# 

# In[64]:


df['runtime'].mean()


# #### Longest movie: 

# In[66]:


df.loc[df['runtime'].idxmax()]


# #### Shortest movie:

# In[67]:


df.loc[df['runtime'].idxmin()]


# #### Runtime visualization:

# In[73]:


# x-axis
plt.xlabel('Runtime of Movies')
# y-axis
plt.ylabel('Number of Movies')
# Title of the histogram
plt.title('Runtime distribution movies')
# Plot a histogram
plt.hist(df['runtime'], bins = 50)


# #### Change of runtime over the years:

# In[82]:


year_runtime = df['runtime'].groupby(df['release_year']).describe()
average_runtime = year_runtime['mean']
f, axs = plt.subplots(figsize=(15,15))
axs.plot(average_runtime)
axs.set_title("change of average runtime over the years")
axs.set_xlabel("Years")
axs.set_ylabel("Average runtime")
plt.show()


# #### Conclusion:
# > 1. The longest movie is "The Story of Film: An Odyssey" and its runtime is 900 minutes
# 2. The shortest movie is "Batman: Strange Days" and its runtime is : 3 minutes
# 3. The Average movie runtime is 102 minutes
# 4. The majority of movies tend to be between 40 to 200 minutes
# 5. Average runtime got slightly shorter over the years

# <a id='conclusions'></a>
# ## Conclusions
#     1. Most cast actor is Robert De Niro, even though the average vote count of his movies isn't as high as I would have imagined
#     2. if a certain company wants to release a movie they better do it in June, October or December, festive seasons tend to get the highest revenue
#     3. relationship, woman, sex, independent seem to be the most common keywords people look for when searching for movies
#     4. average runtime of the highest rated movies is 140 minutes which means that most movies that are slightly under/over 2 hours are popular
#     5. Usually, the bigger the bidget the more popularity
#     6. longer movies have higher budget
#     7. The absolute longest movies takes the lion share of high ratings
#     8. Highest profitable movies tend to have the most revenue but not necessarily the most budget
#     9. The Average movie runtime is 102 minutes

# In[ ]:




