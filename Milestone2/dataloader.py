import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm
from FeatureEngineering_2 import *

import os

def extract_train_data():
    """
    Extract training data from nl_data/ json files
    """
    if not os.path.exists("csv"):
        os.makedirs("csv")
    for year in [2016, 2017, 2018, 2019]: # 2020 has been excluded to be reserved as the test set
        read_a_season("nhl_data/",year).to_csv('csv/tidy_{season}.csv'.format(season = year), sep = ',', index = False)

    directory = 'csv'
    
    # Get a list of csv file names within the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Sort the list of csv files
    csv_files.sort()
    
    # List to hold your DataFrames
    dataframes_list = []
    
    # Loop over the sorted list of csv files with tqdm for progress indication
    for filename in tqdm(csv_files, desc="Loading files", unit="file"):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        dataframes_list.append(df)
    
    # Concatenate all the DataFrames in the list into a single DataFrame
    train_df = pd.concat(dataframes_list, ignore_index=True)
    
    # Sort the DataFrame first by 'game_id' and then by 'event_idx'
    train_df = train_df.sort_values(by=['game_id', 'event_idx'])
    
    # Reset the index of the sorted DataFrame
    train_df = train_df.reset_index(drop=True)

    return train_df

def select_features_for_model(x, col_list):
    """ Selector columns fromt he train datafame"""
    return x[col_list]

def split_train_valid_sets(X:pd.DataFrame, Y:pd.DataFrame):
    """ Split data into training and validation sets"""
    
    #splitting into train and validation sets (in numpy format)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, X_valid, Y_train, Y_valid

def clean_dataset(dataset):
    """ Cleaning data to remove NAN values"""    
    for col in dataset.columns:
        print(f"NAs in {col}: {(dataset[col].isna()).astype(int).sum()}")

    #removing empty_net and strength due to high NAs
    if "empty_Net" in dataset.columns:
        dataset = dataset.drop(columns=["empty_Net"])
    if "strength" in dataset.columns:
        dataset = dataset.drop(columns=["strength"])
    
    clean_dataset = dataset.dropna() #axis=0, rows, how='any'
    print("No. of rows in cleaned dataset: ", len(clean_dataset))

    return clean_dataset

def onehot_generator(column):
    """ Does one hot encoding of categorical values"""
    #ref:https://stackoverflow.com/questions/69302224/how-to-one-hot-encode-a-dataframe-column-in-python
    one_hot_encoder = OneHotEncoder()
    values = one_hot_encoder.fit_transform(column)

    df = pd.DataFrame(values.toarray(),columns=one_hot_encoder.get_feature_names_out()).astype(int)
    return df

def prepare_train_valid_dataset(feature_list):

    #prepare training set from data
    # train_dataset = extract_train_data()
    train_dataset = pd.read_csv("extract_nhl_data_mile_2.csv")
    # print("train columns: ", train_dataset.columns)
    # print("anglr_net_uniqeue:", train_dataset.angle_net.unique())
    # print("last event type_uniqeue:", train_dataset.last_event_type.unique())


    #FEAUTRE SELECTION
    train_dataset = select_features_for_model(train_dataset, feature_list + ["is_goal"])


    #handle categorical features
    if 'shot_type' in train_dataset.columns:
        shot_type_1hot = onehot_generator(train_dataset[['shot_type']])
        train_dataset = pd.concat([train_dataset,
                                         shot_type_1hot.set_index(train_dataset.index),
                                        ],
                                        axis=1,
                                        )
        train_dataset = train_dataset.drop(columns=['shot_type'])
    if 'last_event_type' in train_dataset.columns:
        last_event_type_1hot = onehot_generator(train_dataset[['last_event_type']])
        train_dataset = pd.concat([train_dataset,
                                         last_event_type_1hot.set_index(train_dataset.index),
                                        ],
                                        axis=1,
                                        )
        train_dataset = train_dataset.drop(columns=['last_event_type'])

    #DATA CLEANING
    clean_train_dataset = clean_dataset(train_dataset)

    #separate Labels "is_goal", from actual features
    Y = clean_train_dataset["is_goal"]
    X = clean_train_dataset.drop(columns=["is_goal"])
    print("X set columns:", len(X.columns), X.columns)

    X_train, X_valid, Y_train, Y_valid = split_train_valid_sets(X, Y)

    return X_train, X_valid, Y_train, Y_valid



def prepare_test_dataset(feature_list):


    if not os.path.exists("csv"):
        os.makedirs("csv")
    # for year in [2016, 2017, 2018, 2019]: # 2020 has been excluded to be reserved as the test set
    year=2020
    read_a_season("nhl_data/",year).to_csv('csv/tidy_{season}.csv'.format(season = year), sep = ',', index = False)

    test_dataset = pd.read_csv('csv/tidy_2020.csv')
    
    #FEAUTRE SELECTION
    test_dataset = select_features_for_model(test_dataset, feature_list + ["is_goal"])


    #handle categorical features
    if 'shot_type' in test_dataset.columns:
        shot_type_1hot = onehot_generator(test_dataset[['shot_type']])
        test_dataset = pd.concat([test_dataset,
                                         shot_type_1hot.set_index(test_dataset.index),
                                        ],
                                        axis=1,
                                        )
        test_dataset = test_dataset.drop(columns=['shot_type'])
    if 'last_event_type' in test_dataset.columns:
        last_event_type_1hot = onehot_generator(test_dataset[['last_event_type']])
        test_dataset = pd.concat([test_dataset,
                                         last_event_type_1hot.set_index(test_dataset.index),
                                        ],
                                        axis=1,
                                        )
        test_dataset = test_dataset.drop(columns=['last_event_type'])


    #DATA CLEANING
    clean_test_dataset = clean_dataset(test_dataset)

    #separate Labels "is_goal", from actual features
    Y_test = clean_test_dataset["is_goal"]
    X_test = clean_test_dataset.drop(columns=["is_goal"])
    print("X set columns:", len(X_test.columns), X_test.columns)

    return X_test, Y_test
    
