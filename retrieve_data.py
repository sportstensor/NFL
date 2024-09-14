import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import os
import openpyxl
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal

def get_data() -> pd.DataFrame:       
     
    file_path = 'data_and_models/df_01.csv'
    #file_path = 'data_and_models/df_02.csv'

    # check number of entries with na's removed and all stats    
    all_stats_df = pd.read_csv(file_path)
    all_stats_df.rename(columns={'home_team': 'HT', 'away_team': 'AT', 'home_score':'HT_SC', 'away_score':'AT_SC', 'home_moneyline':'HT_ML', 'away_moneyline':'AT_ML', 'spread_line':'SL', 'temp':'T', 'home_elo':'HT_ELO', 'away_elo':'AT_ELO', 'home_points_diff':'HT_PD', 'away_points_diff':'AT_PD', 'home_qb_rating':'HT_QBR', 'away_qb_rating':'AT_QBR'}, inplace=True)
    
    all_stats_df.to_csv(file_path)

    all_stats_df = all_stats_df.dropna()
    all_stats_df = remove_outliers(all_stats_df)
    all_stats_df = all_stats_df.dropna()

    print('database 1 length', len(all_stats_df))
    
    data = all_stats_df
    teamcode = {
        'KC':32,
        'BUF':31,
        'LA':30,
        'NO':29,
        'BAL':28,
        'PHI':27,
        'GB':26,
        'SF':25,
        'DAL':24,
        'PIT':23,
        'NE':22,
        'MIN':21,
        'SEA':20,
        'TEN':19,
        'TB':18,
        'LAC':17,
        'MIA':16,
        'CIN':15,
        'IND':14,
        'CLE':13,
        'ATL':12,
        'OAK':11,
        'LV':11,
        'CHI':10,
        'DET':9,
        'HOU':8,
        'JAX':7,
        'DEN':6,
        'WAS':5,
        'ARI':4,
        'CAR':3,
        'NYG':2,
        'NYJ':1
        }
    data['HT'] = data['HT'].map(teamcode)
    data['AT'] = data['AT'].map(teamcode)
    data = data.dropna()

    return data

def scale_data(data:pd.DataFrame) -> Tuple[dict, np.ndarray, np.ndarray]:

    ## Scaling data so it is normalized and ready for ingestion ##
    #, 'HT_QBR', 'AT_QBR'
    X_scaled = data[['HT', 'AT', 'HT_ELO', 'AT_ELO', 'HT_PD', 'AT_PD']].values
    y_scaled = data[['HT_SC', 'AT_SC']].values.astype(float)

    columns_for_model_input = ['HT', 'AT', 'HT_ELO', 'AT_ELO', 'HT_PD', 'AT_PD', 'HT_SC', 'AT_SC'] 

    # Scale features
    scalers = {}
    index = 0
    for column in columns_for_model_input:
        scaler = MinMaxScaler(feature_range=(0, 1))

        if index < X_scaled.shape[1]:
            X_scaled[:,index] = scaler.fit_transform(X_scaled[:,index].reshape(-1,1)).reshape(1,-1)
        else:
            y_scaled[:,index-X_scaled.shape[1]] = scaler.fit_transform(y_scaled[:,index-X_scaled.shape[1]].reshape(-1,1)).reshape(1,-1)

        scalers[column] = scaler
        index += 1

    return scalers, X_scaled, y_scaled

def prep_pred_input(fixture:dict, scalers:dict) -> np.array:

    date_formatted = datetime.strptime(fixture['DATE'], '%Y-%m-%d')

    current_team_stats = pd.read_excel('data_and_models/current_team_db.xlsx')

    home_abbrv = current_team_stats[(current_team_stats['team'] == fixture['HT'])]['abbrv'].values[0]
    away_abbrv = current_team_stats[(current_team_stats['team'] == fixture['AT'])]['abbrv'].values[0]

    home_abbrv_dict = home_abbrv.split(' ')[4]
    away_abbrv_dict = away_abbrv.split(' ')[4]

    home_stats = current_team_stats[(current_team_stats['abbrv'] == home_abbrv)]
    away_stats = current_team_stats[(current_team_stats['abbrv'] == away_abbrv)]

    input = {}
    input['HT'] = get_teamcode_from_abrv(home_abbrv_dict)
    input['AT'] = get_teamcode_from_abrv(away_abbrv_dict)
    input['HT_ELO'] = home_stats['elo'].to_numpy()[0]
    input['AT_ELO'] = away_stats['elo'].to_numpy()[0]
    input['HT_PD'] = home_stats['pd'].to_numpy()[0]
    input['AT_PD'] = away_stats['pd'].to_numpy()[0]

    input = np.array(list(input.values())).reshape(1,-1)

    index = 0
    for column in scalers.keys():                
        if index < input.shape[1]:
            input[:,index] = scalers[column].transform(input[:,index].reshape(-1,1)).reshape(1,-1)
        index += 1       

    output = '...'

    return input, output

def update_current_team_database():

    ### Update database for current statistics ###

    print('Database updated successfully')

def get_teamcode_from_abrv(abbrv:float):

    teamcode = {
        'KC':32,
        'BUF':31,
        'LA':30,
        'NO':29,
        'BAL':28,
        'PHI':27,
        'GB':26,
        'SF':25,
        'DAL':24,
        'PIT':23,
        'NE':22,
        'MIN':21,
        'SEA':20,
        'TEN':19,
        'TB':18,
        'LAC':17,
        'MIA':16,
        'CIN':15,
        'IND':14,
        'CLE':13,
        'ATL':12,
        'OAK':11,
        'LV':11,
        'CHI':10,
        'DET':9,
        'HOU':8,
        'JAX':7,
        'DEN':6,
        'WAS':5,
        'ARI':4,
        'CAR':3,
        'NYG':2,
        'NYJ':1
        }
    return teamcode[abbrv]

def remove_outliers(data):

    # Removing outliers 
    ht_sc_95 = np.percentile(data['HT_SC'], 95)
    at_sc_95 = np.percentile(data['AT_SC'], 95)
    
    ht_pd_5 = np.percentile(data['HT_PD'], 5)
    ht_pd_95 = np.percentile(data['HT_PD'], 95)

    at_pd_5 = np.percentile(data['AT_PD'], 5)
    at_pd_95 = np.percentile(data['AT_PD'], 95)
    
    ht_elo_5 = np.percentile(data['HT_ELO'], 3)
    ht_elo_95 = np.percentile(data['HT_ELO'], 97)

    at_elo_5 = np.percentile(data['AT_ELO'], 3)
    at_elo_95 = np.percentile(data['AT_ELO'], 97)
    
    data = data[(data['HT_SC'] <= ht_sc_95) & (data['AT_SC'] <= at_sc_95)]
    data = data[(data['HT_PD'] >= ht_pd_5) & (data['HT_PD'] <= ht_pd_95)]
    data = data[(data['AT_PD'] >= at_pd_5) & (data['AT_PD'] <= at_pd_95)]
    data = data[(data['HT_ELO'] >= ht_elo_5) & (data['HT_ELO'] <= ht_elo_95)]
    data = data[(data['AT_ELO'] >= at_elo_5) & (data['AT_ELO'] <= at_elo_95)]

    return data