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
from io import BytesIO
import json
import task3_serving_client as serving_client
import task3_game_client as game_client
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns


# Global variable to hold the selected model information
selected_model = None

# Initialize the serving and game clients
game_cli = game_client.GameClient()
serving_cli = serving_client.ServingClient(ip='localhost', port=8080)


def create_shot_heatmap(data):
    """
    Create and display a heatmap of shot locations.

    Args:
        data (DataFrame): The DataFrame containing shot data including 'x_coord' and 'y_coord'.
    """
    plt.figure(figsize=(20, 10))

    # Improved color palette and transparency
    cmap = sns.cubehelix_palette(start=2.8, rot=0.2, light=0.9, n_colors=6, as_cmap=True)

    # Create the KDE plot with adjusted settings
    sns.kdeplot(x=data['x'], y=data['y'], cmap=cmap, shade=True, alpha=0.6, bw_adjust=2)

    # Add shot markers
    plt.scatter(data['x'], data['y'], color='blue', alpha=0.6, s=10, edgecolor='black', linewidth=1.5)

    plt.title('Shot Heatmap', fontsize=20)
    plt.xlabel('Feet', fontsize=18)
    plt.ylabel('Feet', fontsize=18)

    # Fetch and overlay the heatmap on a rink image from a URL
    url = 'https://upload.wikimedia.org/wikipedia/commons/2/21/Hockeyrink_IIHF.png'
    response = requests.get(url)
    img = plt.imread(BytesIO(response.content))
    plt.imshow(img, extent=[-100, 100, -42.5, 42.5])  # Ensure correct aspect ratio

    # Annotations and Legends
    plt.colorbar(label='Shot Density')
    plt.grid(False)  # Disable grid to avoid clutter

    plt.tight_layout()
    plt.show()
    
    
def color_scale(val, max_val):
    """
    Apply a color scale to values in a DataFrame column.
    
    Args:
        val (float): The value to be colored.
        max_val (float): The maximum value in the column for scaling.

    Returns:
        str: A string representing the CSS style for background color.
    """
    # Calculate the proportion of 'val' relative to 'max_val'
    proportion = val / max_val if max_val > 0 else 0
    green_color = mcolors.to_rgba('green')
    mixed_color = np.array(mcolors.to_rgba('white')) * (1 - proportion) + np.array(green_color) * proportion
    color_hex = mcolors.to_hex(mixed_color)
    return f'background-color: {color_hex}'


def download_model(workspace, model, version):
    """
    Download a model based on user selections from the sidebar.

    Args:
        workspace (str): The workspace name.
        model (str): The model name.
        version (str): The model version.
    """
    with open('tracker.json', 'w') as file:
        json.dump({}, file)
    serving_cli.download_registry_model(workspace=workspace, model=model, version=version)
    st.sidebar.write('Model downloaded successfully.')


def fetch_and_display_game_data(serving_cli, game_cli, game_id):
    """
    Fetch game data for the specified game ID and display it using Streamlit.

    Args:
        serving_cli (ServingClient): An instance of ServingClient to make predictions.
        game_cli (GameClient): An instance of GameClient to fetch game data.
        game_id (str): The ID of the game to fetch data for.

    Returns:
        tuple or None: A tuple containing the final DataFrame, game details, and xG scores, or None if no game ID is provided.
    """
    
    global selected_model
    
    if game_id:
        # Process to fetch and display game data
        #game_cli.get_game(game_id)
        data, is_live, period, time_remaining, home_team, away_team, home_score, away_score = game_cli.ping_game(game_id)

        if data.empty:
            #st.write(f"No data available for game ID {game_id}.")
            #st.write(" ")
            return
        
        if selected_model == "Log_Reg_shot_dist_only":
            sliced_data = data[['shot_dist']]
        elif selected_model == "Log_Reg_shot_dist_and_angle":
            sliced_data = data[['shot_dist', 'angle_net']]
        else:
            st.write("Invalid model selection.")
            return

        predicted_xG = serving_cli.predict(sliced_data)
        data['chance_of_goal'] = predicted_xG['chance of goal'].tolist()
        
        home_xg = data.apply(lambda row:row["chance_of_goal"]*1 if row["home_or_away"]=="home" else 0,axis=1)
        away_xg = data.apply(lambda row:row["chance_of_goal"]*1 if row["home_or_away"]=="away" else 0,axis=1)
        
        data['home_xg']=home_xg.cumsum()
        data['away_xg']=away_xg.cumsum()
        
        home_xG = home_xg.cumsum().iloc[-1]
        away_xG = away_xg.cumsum().iloc[-1]
        

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

    else:
        st.write("Please enter a valid Game ID.")


def sidebar_configuration():
    """
    Configure the Streamlit sidebar for workspace, model, and version selection.
    """
    
    global selected_model
    # Configure the sidebar for workspace, model, and version selection.
    workspace = st.sidebar.selectbox("Workspace: ", ["hfereidouni"])
    
    # Dictionary mapping model names to their latest versions
    model_versions = {
        "Log_Reg_shot_dist_only": "1.17.0",
        "Log_Reg_shot_dist_and_angle": "1.9.0",
    }

    model = st.sidebar.selectbox("Model: ", list(model_versions.keys()))

    # Automatically set the version to the latest one
    latest_version = model_versions[model]
    version = st.sidebar.selectbox("Version: ", [latest_version])

    # Display a warning if not the latest version
    st.sidebar.warning(f"The latest version for '{model}' is {latest_version}.")

    selected_model = model
    if st.sidebar.button('Download Model'):
        download_model(workspace, model, version)


def main():
    """
    Main function to run the Streamlit app.
    """
    # Setup and configuration of the Streamlit app
    
    st.title("Hockey ._/ Analytics Dashboard")
    st.markdown("""
    <h1 style='font-size: 16px; color: orange; margin-bottom: 0.5em;'>First configure the sidebar for workspace, model, and version selection!</h1>
    """, unsafe_allow_html=True)

    # Configure the sidebar for user inputs
    sidebar_configuration()

    # Main section for game data interaction
    game_id = st.text_input("Enter Game ID (e.g. 2022020064)")
    if st.button('Fetch Game Data'):
        all_data_frames = []  # List to store data from each game
        last_game_details = None  # Variable to store the last game's details
        last_xg_scores = None  # Variable to store the last xG scores

        while True:
            result = fetch_and_display_game_data(serving_cli, game_cli, game_id)
            if result is None :
                print("result is none")
                break  # Break the loop if the function returns None

            game_data, game_details, xg_scores = result
            if game_data.empty:
                print("game dat is empty")
                break  # Break the loop if no more data is available

            all_data_frames.append(game_data)
            last_game_details = game_details  # Update the last game's details
            last_xg_scores = xg_scores  # Update the last xG scores
            print("--55555----len of all events", len(all_data_frames))
            # for previous games
            if not game_details["is_live"]:
                print("a prev game")
                break

        # Concatenate all data frames into one
        print("------len of all events", len(all_data_frames))
        master_df = pd.concat(all_data_frames, ignore_index=True) if all_data_frames else pd.DataFrame()
        print("master df shape:", master_df.shape)
        
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
                           value=f" {last_xg_scores['home_xG']:.2f} ({last_xg_scores['home_score']})",
                           delta=f"{last_xg_scores['home_score'] - last_xg_scores['home_xG']:.2f}")
            
            cols[1].metric(label=f"{last_game_details['away_team']}: xG (Actual Score)",
                           value=f"{last_xg_scores['away_xG']:.2f} ({last_xg_scores['away_score']})",
                           delta=f"{last_xg_scores['away_score'] - last_xg_scores['away_xG']:.2f}")

        # Display the aggregated DataFrame
        final = master_df[['home_name', 'away_name', 'home_score', 'away_score', 'is_goal', 'home_or_away', 'chance_of_goal',  'home_xg', 'away_xg', 'period', 'period_time_rem', 'angle_net', 'shot_dist']]
        st.subheader("Aggregated Event Data and Predictions")
        
        # Get the maximum value from the 'chance_of_goal' column for scaling
        max_val = final['chance_of_goal'].max()

        # Apply the color scale to the 'chance_of_goal' column
        styled_df = final.style.applymap(lambda val: color_scale(val, max_val), subset=['chance_of_goal'])
        st.dataframe(styled_df)
        
        master_df['x'] = master_df['coordinate'].apply(lambda coord: coord['x'])
        master_df['y'] = master_df['coordinate'].apply(lambda coord: coord['y'])
        
        # Additional section for displaying the Shot Heatmap
        #if st.button('Show Shot Heatmap'):
            # Assuming 'x_coord' and 'y_coord' are columns in your data
        create_shot_heatmap(master_df[['x', 'y']])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

if __name__ == "__main__":
    main()
    #fetch_and_display_game_data('2022020011')