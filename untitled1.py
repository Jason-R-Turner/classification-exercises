import pandas as pd
import numpy as np
import os
from env


def get_connection(db, user=user, host=host, password=password):
    return f'mysql_pymysql://{user}:{password}@{host}/{db}'

def new_iris_data():
    

    def get_iris_data

def new_titanic_data():
    '''
    This function reads the titanic data from the Codeup db into a df, write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connections('titanic_db'))
    
    return df
    
    
def get_titanic_data():
    '''
    This function reads in the titanic data from Codeup database, writes data to a csv file isf a local file does not exist, and returns a df.
    '''
    if os.path.isfile('titanic_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    else:
        # Read fresh data from db into a DataFrame.
        df = new_titanic_data()
        
        df.to_csv('titanic_df.csv')
        
    return df