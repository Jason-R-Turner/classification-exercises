#!/usr/bin/env python
# coding: utf-8

# # Prepare Data
# 
# Plan - Acquire - **Prepare** - Explore - Model - Deliver

# ## What we are doing and why:
# 
# **What:** Clean and tidy our data so that it is ready for exploration, analysis and modeling
# 
# **Why:** Set ourselves up for certainty! 
# 
#     1) Ensure that our observations will be sound:
#         Validity of statistical and human observations
#     2) Ensure that we will not have computational errors:
#         non numerical data cells, nulls/NaNs
#     3) Protect against overfitting:
#         Ensure that have a split data structure prior to drawing conclusions

# ## High level Roadmap:
# 
# **Input:** An aquired dataset (One Pandas Dataframe) 
# 
# **Output:** Cleaned data split into Train, Validate, and Test sets (Three Pandas Dataframes)
# 
# **Processes:** Inspect and summarize the data ---> Clean the data ---> Split the data

# ## Inspect and Summarize

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer

# filter out warnings
import warnings
warnings.filterwarnings('ignore')

# our own acquire script:
import acquire


# ## Inspect and Summarize

# In[2]:


# Importing our data
df = acquire.get_titanic_data()


# In[3]:


# Take a look at the data
df.head()


# In[68]:


df[df['embark_town'].isna()]


# In[4]:


df.info()


# In[5]:


pd.crosstab(df['class'], df.pclass)


# In[6]:


pd.crosstab(df.sibsp, df.alone)


# In[7]:


pd.crosstab(df.embarked, df.embark_town)


# #### Gather our takeaways, i.e., what we are going to do when we clean:
# - survived is our target variable (it will not be a 'feature' of our model)
# - passenger_id needs to be removed
# - pclass and class are the same data, we will need to decide which one is worth keeping
# - embarked and embark_town represent the same data (we will need to decide which one is worth keeping)
# - Some people don't have siblings or spouses but are not alone

# In[8]:


df.describe()


# In[9]:


df.columns


# In[10]:


numcols = [col for col in df.columns if df[col].dtype != 'O']


# In[11]:


numcols


# In[12]:


catcols = [col for col in df.columns if df[col].dtype == 'O']


# In[13]:


catcols


# In[14]:


# Describe the object columns
for col in catcols:
    print(f"Column: {col}")
    print(df[col].value_counts())
    print("---------")
    print(df[col].value_counts(normalize=True, dropna=False))
    print("======================")


# In[15]:


# Histograms of numeric columns
for col in numcols:
    print(col)
    df[col].hist()
    plt.show()


# ### Takeaways
# - Remove embarked
# - Remove class
# - Remove passenger_id
# - Remove deck
#     - Has too many nulls, I would need to build a predictive model - not worth the time
# - Lots of missing information in age
#     - Going to have to impute nulls
# - Two nulls in embark_town
#     - Going to have to impute these nulls (maybe just use mode)

# ## Clean

# In[16]:


# Drop duplicates
df.drop_duplicates(inplace=True)


# In[17]:


df.shape # No duplicates after all


# In[18]:


columns_to_drop = ['embarked', 'class', 'passenger_id', 'deck']


# In[19]:


data = df.drop(columns = columns_to_drop) # Saved this change to a new variable so that I don't mess up the original data


# #### Encoding: Turning Categorical Values into Boolean Values (0,1)
#  - We have two options: simple encoding or one-hot encoding

# In[20]:


# Encoding steps
# 1. Make a dataframe out of "dummy" coluns
# 2. Concatenate our dummy dataframe to our original dataframe

dummy_df = pd.get_dummies(data[['sex', 'embark_town']], dummy_na=False, drop_first=[True, True]) # you need as many Trues (or Falses) for as many columns that you're asking about
# get_dummies makes a whole new dataframe that we'll need to append on the end


# In[21]:


dummy_df


# In[22]:


# Concatenate my dummy_df to my data

data = pd.concat([data, dummy_df], axis=1) # axis=1 so that you don't append it as extra rows if it somehow worked
data


# ## Putting our Work Into a Function

# In[23]:


def clean_titanic_data(df):
    '''
    Takes in a titianic dataframe and returns a cleaned dataframe
    Arguments: df = a pandas dataframe with the expected feature names and columns
    Return: clean_df - a dataframe with the cleaning operations performed on it
    ''' # triple quotes includes information the use can find using the help function
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Drop columns
    columns_to_drop = ['embarked', 'class', 'passenger_id', 'deck']
    df = df.drop(columns = columns_to_drop)
    # encoded categorical variables
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_df], axis=1)
    return df.drop(columns=['sex', 'embark_town'])


# In[24]:


df = acquire.get_titanic_data()
clean_df = clean_titanic_data(df)
clean_df


# In[25]:


clean_df.info()


# ## Train, Validate, Test Split

# In[40]:


train, test = train_test_split(clean_df, 
                               train_size = 0.8, 
                               stratify = clean_df.survived, 
                               random_state=1234)
# stratify is only used for categorical targets
# random_state lets other people get the same result when running the code


# In[30]:


train.shape


# In[31]:


test.shape


# In[41]:


# third set can be taken from either train or test sets but preferably take it out
# of the bigger set which is train in this case

train, validate = train_test_split(train,
                                    train_size = 0.7,
                                    stratify = train.survived,
                                    random_state=1234)


# In[34]:


train.shape


# In[37]:


validate.shape


# In[38]:


test.shape


# In[42]:


train.head()


# In[43]:


validate.head()


# In[44]:


test.head()


# ## Option for Missing Values: Impute
# 
# We can impute values using the mean, median, mode (most frequent), or a constant value. We will use sklearn.imputer.SimpleImputer to do this.  
# 
# 1. Create the imputer object, selecting the strategy used to impute (mean, median or mode (strategy = 'most_frequent'). 
# 2. Fit to train. This means compute the mean, median, or most_frequent (i.e. mode) for each of the columns that will be imputed. Store that value in the imputer object. 
# 3. Transform train: fill missing values in train dataset with that value identified
# 4. Transform test: fill missing values with that value identified

# 1. Create the `SimpleImputer` object, which we will store in the variable `imputer`. In the creation of the object, we will specify the strategy to use (`mean`, `median`, `most_frequent`). Essentially, this is creating the instructions and assigning them to a variable we will reference.  

# In[45]:


imputer = SimpleImputer(strategy='mean', missing_values=np.nan)


# In[46]:


type(imputer)


# 2. `Fit` the imputer to the columns in the training df.  This means that the imputer will determine the `most_frequent` value, or other value depending on the `strategy` called, for each column.   

# In[49]:


imputer = imputer.fit(train[['age']]) # double brackets to make it a dataframe


# 3. It will store that value in the imputer object to use upon calling `transform.` We will call `transform` on each of our samples to fill any missing values.  

# In[50]:


train[['age']] = imputer.transform(train[['age']])


# In[51]:


train.info()


# Create a function that will run through all of these steps, when I provide a train and test dataframe, a strategy, and a list of columns. 

# In[54]:


train.age  # imputer is smart enough to only overwrite the null values


# In[57]:


validate[['age']] = imputer.transform(validate[['age']])


# In[58]:


test[['age']] = imputer.transform(test[['age']])


# In[60]:


def impute_age(train, validate, test):
    '''
    Imputes the mean age of train to all three datasets
    '''
    imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    imputer = imputer.fit(train[['age']])
    train[['age']] = imputer.transform(train[['age']])
    validate[['age']] = imputer.transform(validate[['age']])
    test[['age']] = imputer.transform(test[['age']])
    return train, validate, test


# Blend the clean, split and impute functions into a single prep_data() function. 

# In[66]:


def prep_titanic_data(df):
    df = clean_titanic_data(df)
    train, test = train_test_split(df, 
                               train_size = 0.8, 
                               stratify = df.survived, 
                               random_state=1234)
    train, validate = train_test_split(train,
                                    train_size = 0.7,
                                    stratify = train.survived,
                                    random_state=1234)
    train, validate, test = impute_age(train, validate, test)
    return train, validate, test


# In[67]:


df = acquire.get_titanic_data()
train, validate, test = prep_titanic_data(df)
train.head()


# In[ ]:


# Can import it late to clean code
# Can also make one that does everything


# ## Exercises
# 
# The end product of this exercise should be the specified functions in a python script named `prepare.py`.
# Do these in your `classification_exercises.ipynb` first, then transfer to the prepare.py file. 
# 
# This work should all be saved in your local `classification-exercises` repo. Then add, commit, and push your changes.
# 
# Using the Iris Data:  
# 
# 1. Use the function defined in `acquire.py` to load the iris data.  
# 
# 1. Drop the `species_id` and `measurement_id` columns.  
# 
# 1. Rename the `species_name` column to just `species`.  
# 
# 1. Create dummy variables of the species name. 
# 
# 1. Create a function named `prep_iris` that accepts the untransformed iris data, and returns the data with the transformations above applied.  

# In[ ]:




