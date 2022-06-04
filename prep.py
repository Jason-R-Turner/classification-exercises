import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prep_telco(df):
    '''
    Prepares acquired telco dataframe for exploration
    '''
    
# splits the full data set 80/20 into train and test dataframes stratified 
# around churn, the target variable, using the train_test_split function
    train, test = train_test_split(df, 
                               train_size = 0.6, 
                               stratify = df.churn, 
                               random_state=5678)

# splits the train dataframe 70/30 into the new train and validate dataframes
# they're stratified around churn again using the train_test_split function
    train, validate = train_test_split(train,
                                    train_size = 0.6,
                                    stratify = train.churn,
                                    random_state=5678)
    
# returns the three dataframes we'll use for training, validation, and testing
    return train, validate, test


def clean_telco_cat(df):
    '''
    Prepares Telco data by dropping continuous/index-like data, duplicate or 
    redundant columns and converting  remaining columns into binary data.
    '''
    
# drops all, any-source duplicate columns we may have from the df
    df = df.drop_duplicates()
    
# creates a variable to hold all the columns to be dropped in a subsequent step
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 
                    'contract_type_id', 'customer_id', 'tenure', 
                    'monthly_charges', 'total_charges']
    
# uses drop fx to remove columns id'd in previous step from df
    df = df.drop(columns=cols_to_drop)
    
# uses get_dummies fx to convert strings to binary values suitable for machine
# learning dropping any columns with already represented data
    dummy_df = pd.get_dummies(df, dummy_na=False, drop_first=[True, True])
    
# returns the new, final df
    return dummy_df


def telco_clean_services(df):
    '''
    Drops extraneous columns that are virtual duplicates or are not services offered by telco except for churn.
    '''
    
# drops all, any-source duplicate columns we may have from the df
    df = df.drop_duplicates()

# creates a variable to hold all the columns to be dropped in a subsequent step
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 
                    'contract_type_id', 'customer_id', 'gender', 
                    'senior_citizen', 'partner', 'dependents', 'tenure', 
                    'monthly_charges', 'total_charges', 'contract_type', 
                    'payment_type']
    
# uses drop fx to remove columns id'd in previous step from df
    df = df.drop(columns=cols_to_drop)

# returns the new, final df
    return df