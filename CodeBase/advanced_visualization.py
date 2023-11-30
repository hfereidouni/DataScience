"""
Advanced code visualization

Author: Jerome Francis
Description : Prepares the shot data for NHL teams for visualization in Plotly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



def get_x_y(ele):
    """
    Extracts the coordinates x & y 
    from a dictionary to separate values.

    Parameters:
    ele: Coordinate dictionary

    Returns:
    x: coordinate x
    y: coordinate y
    """
    ele_dict = eval(ele)
    try:
        x = ele_dict['x']
    except:
        x = np.nan
    try:
        y = ele_dict['y']
    except:
        y = np.nan
    return x,y


def prepare_shot_pic(path):
    """
    Prepares the rink image compatible 
    to the plotly graph. Includes 
    cropping and transformation of the nhl_rink.png
    from 1100 X 467 to 467 X 550.

    Parameters:
    path: image path

    Returns:
    rink_img: PIL Image object
    """
    from PIL import Image
    image = Image.open('nhl_rink.png')
    length, height = image.size
    rink_img = image.crop((length/2, 0, length, height)) #left-uper point to right-lower point
    rink_img = rink_img.rotate(90, expand=True)
    return rink_img

"""
Shot data preparation
"""
def filter_offensive_shot_data(shot_data):
    """
    Filter out the offensive zone coordinates.

    Paramters:
    shot_data: Dataframe for shots

    Returns:
    offensive_shot_df: Offensitve shot df
    """
    attacking_shot_with_right_rink_side_condition = (shot_data.rink_side=="right") & (shot_data.coordinate_x < 0)
    attacking_shot_with_left_rink_side_condition = (shot_data.rink_side=="left") & (shot_data.coordinate_x > 0)
    
    offensive_shot_df = shot_data[attacking_shot_with_right_rink_side_condition | attacking_shot_with_left_rink_side_condition]
    
    return offensive_shot_df

def compute_league_shot_rate_per_hour(df, start_year, end_year):
    """
    Computes excess shot rate per hour for all teams.
    
    Parameters:
    df: Dataframe containing NHL data
    start_year: start year of the season/ period
    end_year: end year of the season/ period
    
    Returns:
    league_df: Dataframe contain league shot info
    """
    
    # YEAR TO game_id (INT) MAPPING
    start_year = start_year * 1000000
    end_year = end_year * 1000000
    # print(start_year, end_year)
    
    league_df = df[( df.game_id >= start_year ) & ( df.game_id <= end_year )]
    league_df = league_df.loc[(df.play_type=="Shot"), :]
    
    # irrespective of rink_side, doing abs(x_coordinates)
    #gives freq. of shot in x,y location in half rink
    league_df.loc[:, 'coordinate_x'] = np.abs(league_df.coordinate_x)
    league_df = league_df.groupby(by=['coordinate_x', 'coordinate_y'])['period'].count().reset_index()
    # print(len(league_df[league_df.period.isna()]))
    league_df = league_df.rename(columns={'period': 'frequency'})
    
    # Assuming 1hour to complete per game & 82 games per season/league
    TOTAL_HOURS_PER_LEAGUE = 1 * 82
    league_df.loc[:, 'frequency'] = league_df.loc[:, 'frequency'] / TOTAL_HOURS_PER_LEAGUE
    
    return league_df

def compute_excess_shot_rate_per_hour(df, start_year, end_year):
    """
    Computes excess shot rate per hour for all teams.
    
    Parameters:
    df: Dataframe containing NHL data
    start_year: start year of the season/ period
    end_year: end year of the season/ period
    
    Returns:
    excess_shot_df: Dataframe contain excess shot rate info
    """
    
    leagure_shot_rate_df = compute_league_shot_rate_per_hour(df, start_year, end_year)
    
    # YEAR TO game_id (INT) MAPPING
    start_year = start_year * 1000000
    end_year = end_year * 1000000
    
    q = df[( df.game_id >= start_year ) & ( df.game_id <= end_year )].sort_values(by=["game_id", "event_idx"])
    q = q.loc[q.play_type=="Shot",:]
    
    #for shot map prep
    q = filter_offensive_shot_data(q)
    q.loc[:, 'coordinate_x'] = np.abs(q.coordinate_x)
    res=q.groupby(by=["attack_team_name", 'coordinate_x', 'coordinate_y'])['period'].count().reset_index()
    # calculate frequency
    res = res.rename(columns={'period': 'team_shot_frequency'})
    
    # Assuming 1hour to complete per game & 82 games per season/league
    TOTAL_HOURS_PER_LEAGUE = 1 * 82
    res.loc[:, 'team_shot_frequency'] = res.loc[:, 'team_shot_frequency'] / TOTAL_HOURS_PER_LEAGUE
    
    # Combine the data with league shot rate df to calculate excess shot rate
    excess_shot_df = res.merge(leagure_shot_rate_df, how='inner',
            left_on=['coordinate_x', 'coordinate_y'],
            right_on=['coordinate_x', 'coordinate_y'],
            suffixes=('_team', '_league')
            )
    excess_shot_df['excess_shot_frequency'] = excess_shot_df['team_shot_frequency'] - excess_shot_df['frequency']
    
    #calculate normal density estimate
    team_freq_data = excess_shot_df.excess_shot_frequency.to_numpy()
    team_freq_mean = np.mean(team_freq_data)
    team_freq_cov = np.cov(team_freq_data)
    
    # how to calculate density estimate given mean and covariance using pyhton-gpt
    from scipy.stats import multivariate_normal
    multivariate_dist = multivariate_normal.pdf(team_freq_data, mean=team_freq_mean, cov=team_freq_cov)
    excess_shot_df['normal_density_estimate'] = pd.Series(multivariate_dist)
    
    return excess_shot_df



def plot_shot_map(df_extract, team_name, start_year, end_year, RINK_IMAGE_PATH):
    """
    Creates shot map for a single team from start to end year
    
    Paramter:
    df_extract: DataFrame of extracted data
    team_name: Name of the team
    start_year: start year of the season/ period
    end_year: end year of the season/ period
    
    Returns:
    None
    """
    excess_df = compute_excess_shot_rate_per_hour(df_extract, start_year, end_year)
    san_df = excess_df[excess_df.attack_team_name==team_name]
    
    import numpy as np
    import plotly.express as px
    
    # Create a 2D histogram (heatmap) in Plotly
    dense = px.density_contour(
        san_df,
        x='coordinate_y',
        y='coordinate_x',
        z='normal_density_estimate', # normlaized version of excess_shot_frequency
        histfunc='count',
        histnorm='percent',
        range_x = [-42.5, 42.5],
        range_y = [0, 100],
        orientation='h',
        # nbinsx=10, nbinsy=10
    )
    
    dense.update_traces(contours_coloring="fill", 
                        # line_color="red", 
                        colorscale="rdbu", 
                        opacity=0.6,
                        colorbar=dict(title='Excess shots per hour (in %)', 
                                    titleside='top',
                                    titlefont={'size': 12},
                                    tickfont={'size': 12},
                                    ticksuffix='%',
                                    )
                       )
    # Update axis labels and title
    dense.update_xaxes(title_text="Distance from the center of Goal")
    dense.update_yaxes(title_text="Distance from mid line to rink wall")
    dense.update_layout()

    # Add the image as a background
    # Specify the path to your image
    rink_img = prepare_shot_pic(RINK_IMAGE_PATH)
    rink_width, rink_height = rink_img.size
    dense.update_layout(
        title=f'{team_name} Shot Map for {start_year}-{end_year} season',
        width=rink_width +100,
        height=rink_height,
        images=[dict(
                    source=rink_img,
                    x=-42.5,
                    y=100,
                    xref="x",
                    yref="y",
                    sizex=85,
                    sizey=100,
                    opacity=1,
                    sizing='stretch',
                    layer="below"
                )])
    
    # Show the plot
    dense.show()


# Run the app
if __name__ == '__main__':
    
    EXTRACTED_DATA_PATH = "../../extracted_nhl_data_tidy_final.csv"
    RINK_IMAGE_PATH = 'nhl_rink.png'
    
    # Data tidy extra part
    df_extract = pd.read_csv(EXTRACTED_DATA_PATH)
    plot_shot_map(df_extract, 'Vancouver Canucks', 2017, 2018, RINK_IMAGE_PATH)
    print("Visual complete")




