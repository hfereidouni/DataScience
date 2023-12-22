#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:04:09 2023

@author: Hamidreza Fereidouni
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import task3_serving_client as serving_client
import task3_game_client as game_client
from game_loader import *
import Tidy_Data_new as td
import matplotlib.colors as mcolors

selected_model = None

# Initialize the serving and game clients
serving_cli = serving_client.ServingClient(ip='localhost', port=8080)
game_cli = game_client.GameClient()


def color_scale(val, max_val):
    """
    Apply a color scale to the values in the DataFrame.
    Interpolates from white (lowest value) to green (highest value).
    """
    proportion = val / max_val if max_val > 0 else 0
    green_color = mcolors.to_rgba('green')
    mixed_color = np.array(mcolors.to_rgba('white')) * (1 - proportion) + np.array(green_color) * proportion
    color_hex = mcolors.to_hex(mixed_color)
    return f'background-color: {color_hex}'

def download_model(workspace, model, version):
    """Download the model based on user selections."""
    with open('tracker.json', 'w') as file:
        json.dump({}, file)
    serving_cli.download_registry_model(workspace=workspace, model=model, version=version)
    st.sidebar.write('Model downloaded successfully.')

def fetch_and_display_game_data(game_id):
    """Fetch game data for the specified game ID and display it."""
    global selected_model
    
    if game_id:
        #game_cli.get_game(game_id)
        #data, is_live, period, time_remaining, home_team, away_team, home_score, away_score = game_cli.ping_game(game_id)
        data, is_live, period, time_remaining, home_team, away_team, home_score, away_score = game_cli.ping_game(game_id)
        #print("--------------------------")
        #print(data.columns)
        # Check if the game data is available
        if data.empty:
            #st.write(f"No data available for game ID {game_id}.")
            st.write(" ")
            return

        # Display basic game information
        #st.subheader(f"Details for Game: {game_id}")
        #st.markdown(f"<strong>{home_team}</strong> vs <strong>{away_team}</strong>", unsafe_allow_html=True)
        #if is_live:
            #st.write(f"Current Period: {period}  |  Time Remaining: {time_remaining}")
        #else:
            #st.write("Game has concluded.")

        # Predict using the model and update the DataFrame
        #predicted_xG = serving_cli.predict(data)
        #model_features = get_model_features()
        #sliced_data = data[model_features]
        
        # for Log_Reg_shot_dist_only
        #sliced_data = data[['shot_dist']]
        
        # for Log_Reg_shot_dist_and_angle
        #sliced_data = data[['shot_dist', 'angle_net']]
        
        
        if selected_model == "Log_Reg_shot_dist_only":
            sliced_data = data[['shot_dist']]
        elif selected_model == "Log_Reg_shot_dist_and_angle":
            sliced_data = data[['shot_dist', 'angle_net']]
        else:
            st.write("Invalid model selection.")
            return
        
        #print(data[['shot_dist']])
        #print(type(sliced_data))  # This should output <class 'pandas.core.frame.DataFrame'>
        predicted_xG = serving_cli.predict(sliced_data)
        
        #print(predicted_xG)
        
        data['chance_of_goal'] = predicted_xG['chance of goal'].tolist()
        #print(data['chance_of_goal'])
        
        home_xg = data.apply(lambda row:row["chance_of_goal"]*1 if row["home_or_away"]=="home" else 0,axis=1)
        away_xg = data.apply(lambda row:row["chance_of_goal"]*1 if row["home_or_away"]=="away" else 0,axis=1)
        
        data['home_xg']=home_xg.cumsum()
        data['away_xg']=away_xg.cumsum()
        
        #print("--------------------------")
        #print(home_xg)
        #print(data['home_xg'])
        #print("--------------------------")
        
        # Aggregate xG scores
        #home_xG = data[data['home_score'] >= 1]['chance_of_goal'].sum()
        home_xG = home_xg.cumsum().iloc[-1]
        #print("--------------------------")
        #print(home_xG)
        #print("--------------------------")
        away_xG = away_xg.cumsum().iloc[-1]
        #away_xG = data[data['away_score'] >= 1]['chance_of_goal'].sum()

        # Displaying the xG scores and actual scores
        #cols = st.columns(2)
        #cols[0].metric(label=f" {home_team}:  Actual Score (xG)",
        #value=f"{home_score} ({home_xG:.4f})",
        #delta=f"{home_score - home_xG:.4f}")
        
        #cols[1].metric(label=f"{away_team}: Actual Score (xG)",
        #value=f"{away_score} ({away_xG:.4f})",
        #delta=f"{away_score - away_xG:.4f}")

        # Display the processed DataFrame
        #print("----------")
        #st.subheader("Event Data and Predictions")
        final_df = data
        game_details = {
            "is_live": is_live,
            "period": period,
            "time_remaining": time_remaining,
            "home_team": home_team,
            "away_team": away_team
        }
        xg_scores = {
            "home_score": home_score,
            "away_score": away_score,
            "home_xG": home_xG,  # You need to define this in your function
            "away_xG": away_xG   # You need to define this in your function
        }

        return final_df, game_details, xg_scores

        #st.dataframe(data['shot_dist'])
        #st.subheader("predicted_xG")
        #st.write(predicted_xG)
        #st.dataframe(data['xG'])
    else:
        st.write("Please enter a valid Game ID.")
        #st.write(" ")

def sidebar_configuration():
    global selected_model
    """Configure the sidebar for workspace, model, and version selection."""
    workspace = st.sidebar.selectbox("Workspace: ", ["hfereidouni"])
    model = st.sidebar.selectbox("Model: ", ["Log_Reg_shot_dist_only", "Log_Reg_shot_dist_and_angle"])
    version = st.sidebar.selectbox("Version: ", ["1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0", "1.8.0", "1.9.0", "1.10.0", "1.11.0", "1.12.0", "1.13.0", "1.14.0", "1.15.0", "1.16.0", "1.17.0"])
    
    # Dictionary mapping model names to their latest versions
    model_versions = {
        #"log_reg_angle_only": "1.11.0",
        "Log_Reg_shot_dist_only": "1.17.0",
        "Log_Reg_shot_dist_and_angle": "1.9.0",
        #"random_classifer_random_classifier": "1.7.0",
        #"xgboost_feature_select_no_hp_tune": "1.0.0",
        #"xgboost_feature_select_hp_tune": "1.0.0"
        }

    # Iterate through the dictionary and check if the model is in the dictionary
    for model_name, version in model_versions.items():
        if model_name in model:
            st.sidebar.warning(f"For '{model_name}', the latest version is {version}.")
            # If you need to set the version, you can uncomment the next line
            #version = version'
            
    selected_model = model
    if st.sidebar.button('Download Model'):
        download_model(workspace, model, version)


def main():
    """Main function to run the Streamlit app."""
    st.title("Hockey ._/ Analytics Dashboard")

    # Configure the sidebar for user inputs
    sidebar_configuration()

    # Main section for game data interaction
    game_id = st.text_input("Enter Game ID (e.g. 2022020064)")
    if st.button('Fetch Game Data'):
        all_data_frames = []  # List to store data from each game
        last_game_details = None  # Variable to store the last game's details
        last_xg_scores = None  # Variable to store the last xG scores

        while True:
            result = fetch_and_display_game_data(game_id)
            if result is None:
                break  # Break the loop if the function returns None

            game_data, game_details, xg_scores = result
            if game_data.empty:
                break  # Break the loop if no more data is available

            all_data_frames.append(game_data)
            last_game_details = game_details  # Update the last game's details
            last_xg_scores = xg_scores  # Update the last xG scores

        # Concatenate all data frames into one
        master_df = pd.concat(all_data_frames, ignore_index=True) if all_data_frames else pd.DataFrame()
        
        # Initialize cumulative xG scores
        master_df['home_xg'] = master_df.apply(lambda row: row['chance_of_goal'] if row['home_or_away'] == 'home' else 0, axis=1).cumsum()
        master_df['away_xg'] = master_df.apply(lambda row: row['chance_of_goal'] if row['home_or_away'] == 'away' else 0, axis=1).cumsum()
        
        last_xg_scores['away_xG'] = master_df['home_xg'].tail(1).values[0]
        last_xg_scores['home_xG'] = master_df['away_xg'].tail(1).values[0]
        
        # Display the last game's details
        if last_game_details:
            st.subheader(f"Details for Game: {game_id}")
            st.markdown(f"<strong>{last_game_details['home_team']}</strong> vs <strong>{last_game_details['away_team']}</strong>", unsafe_allow_html=True)
            if last_game_details['is_live']:
                st.write(f"Current Period: {last_game_details['period']}  |  Time Remaining: {last_game_details['time_remaining']}")
            else:
                st.write("Game has concluded.")

        # Display the last xG scores and actual scores
        if last_xg_scores:
            cols = st.columns(2)
            cols[0].metric(label=f" {last_game_details['home_team']}:  xG (Actual Score)",
                           value=f" {last_xg_scores['home_xG']:.4f} ({last_xg_scores['home_score']})",
                           delta=f"{last_xg_scores['home_score'] - last_xg_scores['home_xG']:.4f}")
            
            cols[1].metric(label=f"{last_game_details['away_team']}: xG (Actual Score)",
                           value=f"{last_xg_scores['away_xG']:.4f} ({last_xg_scores['away_score']})",
                           delta=f"{last_xg_scores['away_score'] - last_xg_scores['away_xG']:.4f}")

        # Display the aggregated DataFrame
        final = master_df[['event_idx', 'home_name', 'away_name', 'home_score', 'away_score', 'is_goal', 'home_or_away', 'chance_of_goal',  'home_xg', 'away_xg', 'period', 'period_time_rem', 'angle_net', 'shot_dist']]
        st.subheader("Aggregated Event Data and Predictions")
        
        # Get the maximum value from the 'chance_of_goal' column for scaling
        max_val = final['chance_of_goal'].max()

        # Apply the color scale to the 'chance_of_goal' column
        styled_df = final.style.applymap(lambda val: color_scale(val, max_val), subset=['chance_of_goal'])
        st.dataframe(styled_df)

if __name__ == "__main__":
    main()
    #fetch_and_display_game_data('2022020011')