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
import task3_serving_client as serving_client
import task3_game_client as game_client
import Tidy_Data_new as td

# Initialize the serving and game clients
serving_cli = serving_client.ServingClient(ip='serving', port=4000)
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
        st.subheader(f"Details for Game: {game_id}")
        st.markdown(f"<strong>{home_team}</strong> vs <strong>{away_team}</strong>", unsafe_allow_html=True)
        if is_live:
            st.write(f"Current Period: {period}  |  Time Remaining: {time_remaining}")
        else:
            st.write("Game has concluded.")

        # Predict using the model and update the DataFrame
        #predicted_xG = serving_cli.predict(data)
        predicted_xG = serving_cli.predict(data['shot_dist'])
        
        #data['xG'] = pd.DataFrame(predicted_xG.items())

        # Aggregate xG scores
        #home_xG = data[data['team'] == home_team]['xG'].sum()
        #away_xG = data[data['team'] == away_team]['xG'].sum()

        # Displaying the xG scores and actual scores
        #cols = st.columns(2)
        #cols[0].metric(label=f"{home_team} xG (Actual Score)",
        #value=f"{home_xG} ({home_score})",
        #delta=int(home_score - home_xG))
        
        #cols[1].metric(label=f"{away_team} xG (Actual Score)",
        #value=f"{away_xG} ({away_score})",
        #delta=int(away_score - away_xG))

        # Display the processed DataFrame
        st.subheader("Event Data and Predictions")
        st.dataframe(data)
        st.dataframe(data['shot_dist'])
        st.subheader("predicted_xG")
        st.write(predicted_xG)
        #st.dataframe(data['xG'])
    else:
        st.write("Please enter a valid Game ID.")

def sidebar_configuration():
    """Configure the sidebar for workspace, model, and version selection."""
    workspace = st.sidebar.selectbox("Workspace: ", ["hfereidouni"])
    model = st.sidebar.selectbox("Model: ", ["log_reg_angle_only", "log_reg_shot_dist_only", "log_reg_shot_dist_and_angle", "random_classifer_random_classifier", "xgboost_feature_select_no_hp_tune", "xgboost_feature_select_hp_tune"])
    version = st.sidebar.selectbox("Version: ", ["1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0", "1.8.0", "1.9.0", "1.10.0", "1.11.0", "1.12.0", "1.13.0", "1.14.0", "1.15.0", "1.16.0", "1.17.0"])
    
    # Dictionary mapping model names to their latest versions
    model_versions = {
        "log_reg_angle_only": "1.11.0",
        "log_reg_shot_dist_only": "1.17.0",
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
        fetch_and_display_game_data(game_id)

if __name__ == "__main__":
    main()
    #fetch_and_display_game_data("2022020011")