#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 2 - Get Started with Python**

# Welcome to the TikTok Project!
# 
# You have just started as a data professional at TikTok.
# 
# The team is still in the early stages of the project. You have received notice that TikTok's leadership team has approved the project proposal. To gain clear insights to prepare for a claims classification model, TikTok's provided data must be examined to begin the process of exploratory data analysis (EDA).
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 2 End-of-course project: Inspect and analyze data**
# 
# In this activity, you will examine data provided and prepare it for analysis.
# <br/>
# 
# **The purpose** of this project is to investigate and understand the data provided. This activity will:
# 
# 1.   Acquaint you with the data
# 
# 2.   Compile summary information about the data
# 
# 3.   Begin the process of EDA and reveal insights contained in the data
# 
# 4.   Prepare you for more in-depth EDA, hypothesis testing, and statistical analysis
# 
# **The goal** is to construct a dataframe in Python, perform a cursory inspection of the provided dataset, and inform TikTok data team members of your findings.
# <br/>
# *This activity has three parts:*
# 
# **Part 1:** Understand the situation
# * How can you best prepare to understand and organize the provided TikTok information?
# 
# **Part 2:** Understand the data
# 
# * Create a pandas dataframe for data learning and future exploratory data analysis (EDA) and statistical activities
# 
# * Compile summary information about the data to inform next steps
# 
# **Part 3:** Understand the variables
# 
# * Use insights from your examination of the summary data to guide deeper investigation into variables
# 
# <br/>
# 
# To complete the activity, follow the instructions and answer the questions below. Then, you will us your responses to these questions and the questions included in the Course 2 PACE Strategy Document to create an executive summary.
# 
# Be sure to complete this activity before moving on to Course 3. You can assess your work by comparing the results to a completed exemplar after completing the end-of-course project.

# # **Identify data types and compile summary information**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.
# 
# # **PACE stages**
# 
# <img src="images/Pace.png" width="100" height="100" align=left>
# 
#    *        [Plan](#scrollTo=psz51YkZVwtN&line=3&uniqifier=1)
#    *        [Analyze](#scrollTo=mA7Mz_SnI8km&line=4&uniqifier=1)
#    *        [Construct](#scrollTo=Lca9c8XON8lc&line=2&uniqifier=1)
#    *        [Execute](#scrollTo=401PgchTPr4E&line=2&uniqifier=1)

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response:
# 
# 

# ### **Task 1. Understand the situation**
# 
# *   How can you best prepare to understand and organize the provided information?
# 
# 
# *Begin by exploring your dataset and consider reviewing the Data Dictionary.*

# Read in the data, view the data dictionary, and explore the dataset to identify the key variables for the stakeholder.

# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2a. Imports and data loading**
# 
# Start by importing the packages that you will need to load and explore the dataset. Make sure to use the following import statements:
# *   `import pandas as pd`
# 
# *   `import numpy as np`
# 

# In[1]:


# Import packages
### YOUR CODE HERE ###
import pandas as pd
import numpy as np


# Then, load the dataset into a dataframe. Creating a dataframe will help you conduct data manipulation, exploratory data analysis (EDA), and statistical activities.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[3]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# ### **Task 2b. Understand the data - Inspect the data**
# 
# View and inspect summary information about the dataframe by **coding the following:**
# 
# 1. `data.head(10)`
# 2. `data.info()`
# 3. `data.describe()`
# 
# *Consider the following questions:*
# 
# **Question 1:** When reviewing the first few rows of the dataframe, what do you observe about the data? What does each row represent?
# 
# **Question 2:** When reviewing the `data.info()` output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?
# 
# **Question 3:** When reviewing the `data.describe()` output, what do you notice about the distributions of each variable? Are there any questionable values? Does it seem that there are outlier values?
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[4]:


# Display and examine the first ten rows of the dataframe
### YOUR CODE HERE ###
data.head(10)


# In[5]:


# Get summary info
### YOUR CODE HERE ###
data.info()


# In[6]:


# Get summary statistics
### YOUR CODE HERE ###
data.describe()


# 1. Each row represents an individual TikTok video. The videos are distinct as they have an id, and metadata is supplied as well, like views, likes, shares etc.
# 
# 2. There are integers, floats and objects in the dataset. There are 19,382 entries, so the only columns with no null values at all include #, video id, video duration, verified status and author banned status.
# 
# 3.The variables that end in 'count' have higher range outliers -- which makes sense, since some videos go "viral" while others do not.

# ### **Task 2c. Understand the data - Investigate the variables**
# 
# In this phase, you will begin to investigate the variables more closely to better understand them.
# 
# You know from the project proposal that the ultimate objective is to use machine learning to classify videos as either claims or opinions. A good first step towards understanding the data might therefore be examining the `claim_status` variable. Begin by determining how many videos there are for each different claim status.

# In[7]:


# What are the different values for claim status and how many of each are in the data?
### YOUR CODE HERE ###
data['claim_status'].value_counts()


# **Question:** What do you notice about the values shown?
# 
# **Answer:** The claims and opinions are quite balanced.

# Next, examine the engagement trends associated with each different claim status.
# 
# Start by using Boolean masking to filter the data according to claim status, then calculate the mean and median view counts for each claim status.

# In[8]:


# What is the average view count of videos with "claim" status?
### YOUR CODE HERE ###
claims = data[data['claim_status'] == 'claim']
print('Mean view count claims:', claims['video_view_count'].mean())
print('Median view count claims:', claims['video_view_count'].median())


# In[9]:


# What is the average view count of videos with "opinion" status?
### YOUR CODE HERE ###
opinions = data[data['claim_status'] == 'opinion']
print('Mean view count opinions:', opinions['video_view_count'].mean())
print('Median view count opinions:', opinions['video_view_count'].median())


# **Question:** What do you notice about the mean and media within each claim category?
# 
# **Answer:** The mean and median for each category are very close, however the difference in both numbers for claims vs. those for opinions is very, very vast.
# 
# Now, examine trends associated with the ban status of the author.
# 
# Use `groupby()` to calculate how many videos there are for each combination of categories of claim status and author ban status.

# In[10]:


# Get counts for each group combination of claim status and author ban status
### YOUR CODE HERE ###
data.groupby(['claim_status', 'author_ban_status']).count()[['#']]


# **Question:** What do you notice about the number of claims videos with banned authors? Why might this relationship occur?
# 
# **Answer:** The claims section has many more banned authors. This makes sense, because false claims are more harmful than disagreeable opinions.
# 
# Continue investigating engagement levels, now focusing on `author_ban_status`.
# 
# Calculate the median video share count of each author ban status.

# In[11]:


### YOUR CODE HERE ###
data.groupby(['author_ban_status']).agg(
    {'video_view_count': ['mean', 'median'],
     'video_like_count': ['mean', 'median'],
     'video_share_count': ['mean', 'median']})


# In[12]:


# What's the median video share count of each author ban status?
### YOUR CODE HERE ###
data.groupby(['author_ban_status']).median(numeric_only=True)[
    ['video_share_count']]


# **Question:** What do you notice about the share count of banned authors, compared to that of active authors? Explore this in more depth.
# 
# **Answer:**: Banned authors have exponentially more shares than active authors -- over 30x the rate, in fact.
# 
# Use `groupby()` to group the data by `author_ban_status`, then use `agg()` to get the count, mean, and median of each of the following columns:
# * `video_view_count`
# * `video_like_count`
# * `video_share_count`
# 
# Remember, the argument for the `agg()` function is a dictionary whose keys are columns. The values for each column are a list of the calculations you want to perform.

# In[13]:


### YOUR CODE HERE ###
data.groupby(['author_ban_status']).agg(
    {'video_view_count': ['count', 'mean', 'median'],
     'video_like_count': ['count', 'mean', 'median'],
     'video_share_count': ['count', 'mean', 'median']
     })


# **Question:** What do you notice about the number of views, likes, and shres for banned authors compared to active authors?
# 
# **Answer:** Banned and under review authors have a much higher mean and median view, like and share count than active authors. This is likely because controversy attracts a lot of eyes. Aslo, the mean is substanstially higher than than the median for active authors, again showing that some videos go viral while others stay contained to a much smaller circle.
# 
# Now, create three new columns to help better understand engagement rates:
# * `likes_per_view`: represents the number of likes divided by the number of views for each video
# * `comments_per_view`: represents the number of comments divided by the number of views for each video
# * `shares_per_view`: represents the number of shares divided by the number of views for each video

# In[14]:


# Create a likes_per_view column
### YOUR CODE HERE ###

data['likes_per_view'] = data['video_like_count'] / data['video_view_count']

# Create a comments_per_view column
### YOUR CODE HERE ###

data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']

# Create a shares_per_view column
### YOUR CODE HERE ###

data['shares_per_view'] = data['video_share_count'] / data['video_view_count']


# Use `groupby()` to compile the information in each of the three newly created columns for each combination of categories of claim status and author ban status, then use `agg()` to calculate the count, the mean, and the median of each group.

# In[15]:


### YOUR CODE HERE ###
data.groupby(['claim_status', 'author_ban_status']).agg(
    {'likes_per_view': ['count', 'mean', 'median'],
     'comments_per_view': ['count', 'mean', 'median'],
     'shares_per_view': ['count', 'mean', 'median']})


# **Questions:**
# 
# How does the data for claim videos and opinion videos compare or differ? Consider views, comments, likes, and shares.
# 
# At this point in the investigation, what do you notice about the correlation between engagement level and claim status?
# 
# **Answer:** We know that videos by banned authors and those under review tend to get far more views, likes, and shares than videos by non-banned authors. However, when a video does get viewed, its engagement rate is less related to author ban status and more related to its claim status.
# 
# Also, we know that claim videos have a higher view rate than opinion videos, but this tells us that claim videos also have a higher rate of likes on average, so they are more favorably received as well. Furthermore, they receive more engagement via comments and shares than opinion videos.
# 
# Note that for claim videos, banned authors have slightly higher likes/view and shares/view rates than active authors or those under review. However, for opinion videos, active authors and those under review both get higher engagement rates than banned authors in all categories.
# 
# Returning to the purpose of this analysis, at this point in the investigation it seems that engagement level is strongly correlated with claim status, and this should be a focus of further inquiry.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# 
# **Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project.
# 
# 
# 

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response.

# ### **Given your efforts, what can you summarize for Rosie Mae Bradshaw and the TikTok data team?**
# 
# *Note for Learners: Your answer should address TikTok's request for a summary that covers the following points:*
# 
# 
# 
# *   A summary of the data types of the variables
# *   The minimum, mean, and maximum values for the two variables chosen
# 

# After looking at the dataset, the two variables that are most likely to help build a predictive model are claim_status and opinion_status. Those two variables are most likely to be helpful because those variables denote claims and opinions. That way, we can create a classification model to differentiate the two variables, as instructed from the employer.
# 
# The mean is larger than the median, so the data is right-skewed. Therefore, it is recommended to consider applying a variable transformation, such as taking logarithms, prior to modeling.
