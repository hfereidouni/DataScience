"""
Advanced code visualization

Author: Jerome Francis
Description : Prepares the setup for shot data visualization in Plotly Dash.
"""


import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from advanced_visualization import compute_excess_shot_rate_per_hour, prepare_shot_pic

EXTRACTED_DATA_PATH = "extracted_nhl_data_tidy_final.csv"
RINK_IMAGE_PATH = 'nhl_rink.png'

"""
List of references used:
1. # https://plotly.com/python/2d-histogram-contour/

Chat GPT queries:
1. how to adjust dropdown size for dcc.dropdown in dash plotly
2. how to place dropdowns in a single line
3. how to reduce the opacity of px.density_contour function?
4. what is justify_content values and how to use it in plotly CSS when dealing with dcc.dropdown?
"""



# Initializing Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
                        html.Label('Select Team:'),
                        dcc.Dropdown(
                            id='team-dropdown',
                            options=[
                                    {'label': 'Buffalo Sabres', 'value': 'Buffalo Sabres'},
                                    {'label': 'Anaheim Ducks', 'value': 'Anaheim Ducks'},
                                    {'label': 'Los Angeles Kings', 'value': 'Los Angeles Kings'},
                                    {'label': 'New York Rangers', 'value': 'New York Rangers'},
                                    {'label': 'Toronto Maple Leafs', 'value': 'Toronto Maple Leafs'},
                                    {'label': 'Dallas Stars', 'value': 'Dallas Stars'},
                                    {'label': 'Boston Bruins', 'value': 'Boston Bruins'},
                                    {'label': 'Arizona Coyotes', 'value': 'Arizona Coyotes'},
                                    {'label': 'New Jersey Devils', 'value': 'New Jersey Devils'},
                                    {'label': 'Vancouver Canucks', 'value': 'Vancouver Canucks'},
                                    {'label': 'St. Louis Blues', 'value': 'St. Louis Blues'},
                                    {'label': 'Columbus Blue Jackets', 'value': 'Columbus Blue Jackets'},
                                    {'label': 'Washington Capitals', 'value': 'Washington Capitals'},
                                    {'label': 'Winnipeg Jets', 'value': 'Winnipeg Jets'},
                                    {'label': 'Carolina Hurricanes', 'value': 'Carolina Hurricanes'},
                                    {'label': 'Pittsburgh Penguins', 'value': 'Pittsburgh Penguins'},
                                    {'label': 'Philadelphia Flyers', 'value': 'Philadelphia Flyers'},
                                    {'label': 'New York Islanders', 'value': 'New York Islanders'},
                                    {'label': 'Tampa Bay Lightning', 'value': 'Tampa Bay Lightning'},
                                    {'label': 'Montréal Canadiens', 'value': 'Montréal Canadiens'},
                                    {'label': 'Minnesota Wild', 'value': 'Minnesota Wild'},
                                    {'label': 'Chicago Blackhawks', 'value': 'Chicago Blackhawks'},
                                    {'label': 'Ottawa Senators', 'value': 'Ottawa Senators'},
                                    {'label': 'Florida Panthers', 'value': 'Florida Panthers'},
                                    {'label': 'San Jose Sharks', 'value': 'San Jose Sharks'},
                                    {'label': 'Edmonton Oilers', 'value': 'Edmonton Oilers'},
                                    {'label': 'Vegas Golden Knights', 'value': 'Vegas Golden Knights'},
                                    {'label': 'Detroit Red Wings', 'value': 'Detroit Red Wings'},
                                    {'label': 'Calgary Flames', 'value': 'Calgary Flames'},
                                    {'label': 'Nashville Predators', 'value': 'Nashville Predators'},
                                    {'label': 'Colorado Avalanche', 'value': 'Colorado Avalanche'},
                                    ],
                            value='Anaheim Ducks',
                            style={ 'width': "50%"},
                            ),
                        html.Div([                            
                        html.Label('Select Start Year:'),
                        dcc.Dropdown(
                                    id='season-start-dropdown',
                                    options=[
                                                {'label': '2016', 'value': 2016},
                                                {'label': '2017', 'value': 2017},
                                                {'label': '2018', 'value': 2018},
                                                {'label': '2019', 'value': 2019},
                                                {'label': '2020', 'value': 2020},
                                            ],
                                    value='2016',
                                    ),
                        html.Label('     Select End Year:'),
                        dcc.Dropdown(
                                    id='season-end-dropdown',
                                    options=[
                                                {'label': '2016', 'value': 2016},
                                                {'label': '2017', 'value': 2017},
                                                {'label': '2018', 'value': 2018},
                                                {'label': '2019', 'value': 2019},
                                                {'label': '2020', 'value': 2020},
                                                {'label': '2021', 'value': 2021},
                                            ],
                                    value='2017',
                                    ),
                        ], style={'display': 'flex', 'justify-content': 'stretch'}),

                        dcc.Graph(id='shot-map'),  
                    ])

# Callback to update the shot map based on team selection
@app.callback(
    Output('shot-map', 'figure'),
    Input('team-dropdown', 'value'),
    Input('season-start-dropdown', 'value'),
    Input('season-end-dropdown', 'value'),
)

def update_shot_map(team_name, start_year, end_year):
    """
    Updates the shot map based on team_name , start
    and end year of the season. Update handled
    interactively using callbacks to Dash app.

    Parameters:
    team_name: Name of the team
    start_year: start year of the season/ period
    end_year: end year of the season/ period

    Returns:
    dense: Plotly graph object
    """
    
    df_extract = pd.read_csv(EXTRACTED_DATA_PATH)
    excess_df = compute_excess_shot_rate_per_hour(df_extract, int(start_year), int(end_year))
    san_df = excess_df[excess_df.attack_team_name == team_name]

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
        title=f'Team Shot Map wrt. League shot Average \n for {start_year}-{end_year} season',
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
    dense.write_html(f"Shot_Map_{start_year}_{end_year}.html")

    return dense


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

    # app.to_html("export_shotmap.html")
    # # Render the app layout to an HTML string
    # html_string = app.index_string()

    # # Save the HTML string to an HTML file
    # with open('exported_shot_map.html', 'w') as f:
    #     f.write(html_string)
