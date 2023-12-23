#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:04:09 2023

@author: Hamidreza Fereidouni
"""

import streamlit as st
import pandas as pd
import requests
import json
import game_client as serving_client
import game_client as game_client
import Tidy_Data_new as td

# Initialize the serving and game clients
serving_cli = serving_client.ServingClient(ip='localhost', port=8080)
game_cli = game_client.GameClient()

def download_model(workspace, model, version):
    """Download the model based on user selections."""
    with open('tracker.json', 'w') as file:
        json.dump({}, file)
    serving_cli.download_registry_model(workspace=workspace, model=model, version=version)
    st.sidebar.write('Model downloaded successfully.')

def fetch_and_display_game_data(game_id):
    """Fetch game data for the specified game ID and display it."""

    if game_id:
        game_cli.get_game(game_id)
        #data, is_live, period, time_remaining, home_team, away_team, home_score, away_score = game_cli.ping_game(game_id)
        data, is_live, period, time_remaining, home_team, away_team, home_score, away_score = game_cli.ping_game(game_id)
        print("--------------------------")
        print(data)
        # Check if the game data is available
        if data.empty:
            st.write(f"No data available for game ID {game_id}.")
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
        sliced_data = data[['shot_dist']]
        print(data[['shot_dist']])
        print(type(sliced_data))  # This should output <class 'pandas.core.frame.DataFrame'>
        predicted_xG = serving_cli.predict(sliced_data)
        
        print(predicted_xG)
        
        data['xG'] = predicted_xG['chance of goal'].tolist()
        print(data['xG'])
        
        
        # Aggregate xG scores
        home_xG = data[data['home_score'] >= 1]['xG'].sum()
        away_xG = data[data['away_score'] >= 1]['xG'].sum()

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
        final_df = data[['home_name', 'away_name', 'home_score', 'away_score', 'is_goal', 'xG', 'period', 'period_time_rem', 'angle_net', 'shot_dist']]
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

def sidebar_configuration():
    """Configure the sidebar for workspace, model, and version selection."""
    workspace = st.sidebar.selectbox("Workspace: ", ["hfereidouni"])
    model = st.sidebar.selectbox("Model: ", ["log_reg_angle_only", "Log_Reg_shot_dist_only", "log_reg_shot_dist_and_angle", "random_classifer_random_classifier", "xgboost_feature_select_no_hp_tune", "xgboost_feature_select_hp_tune"])
    version = st.sidebar.selectbox("Version: ", ["1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0", "1.8.0", "1.9.0", "1.10.0", "1.11.0", "1.12.0", "1.13.0", "1.14.0", "1.15.0", "1.16.0", "1.17.0"])
    
    # Dictionary mapping model names to their latest versions
    model_versions = {
        "log_reg_angle_only": "1.11.0",
        "Log_Reg_shot_dist_only": "1.17.0",
        "log_reg_shot_dist_and_angle": "1.9.0",
        "random_classifer_random_classifier": "1.7.0",
        "xgboost_feature_select_no_hp_tune": "1.0.0",
        "xgboost_feature_select_hp_tune": "1.0.0"
        }

    # Iterate through the dictionary and check if the model is in the dictionary
    for model_name, version in model_versions.items():
        if model_name in model:
            st.sidebar.warning(f"For '{model_name}', the latest version is {version}.")
            # If you need to set the version, you can uncomment the next line
            # version = version
    
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
        data_frames = []  # List to store data from each game
        last_game_details = None  # Variable to store the last game's details
        last_xg_scores = None  # Variable to store the last xG scores

        for _ in range(7):
            game_data, game_details, xg_scores = fetch_and_display_game_data(game_id)
            data_frames.append(game_data)
            last_game_details = game_details  # Update the last game's details
            last_xg_scores = xg_scores  # Update the last xG scores

        # Concatenate all data frames into one
        master_df = pd.concat(data_frames, ignore_index=True)

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
        st.subheader("Aggregated Event Data and Predictions")
        st.dataframe(master_df)

if __name__ == "__main__":
    main()