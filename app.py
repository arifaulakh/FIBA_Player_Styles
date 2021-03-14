#!/usr/bin/env python
# coding: utf-8

# In[1]:


from jupyter_dash import JupyterDash
import os
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_table

colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# In[2]:

cluster_centres = pd.read_csv('Clustering/cluster-centres.csv', index_col=0).transpose().reset_index()
cluster_centres.index = cluster_centres['index'] + ' (' + cluster_centres.index.astype(str) + ')'
cluster_centres = cluster_centres.drop(columns=['index'])
cluster_labels = cluster_centres.index

cluster_centres_display = pd.read_csv('Clustering/cluster-centres.csv', index_col=0)
cluster_centres_display.index = '% ' + cluster_centres_display.index
cluster_centres_display.index.name = ""
cluster_centres_display.reset_index(inplace=True)
s = cluster_centres_display.select_dtypes(include=[np.number])*100
s = s.round(2)
cluster_centres_display[s.columns] = s

df = pd.read_csv("Clustering/NBA_INTL_FIBA_Clustering_Encoded.csv", encoding='latin-1', index_col=0)
all_names = ['League Average'] + df.index.tolist() + cluster_centres.index.tolist()

dfs = [df[[c for c in df.columns if c.startswith(name)]].dropna() for name in ['FIBA', 'NBA', 'INTL']]
# Split up into separate dataframes


play_types = [c[5:] for c in dfs[0].columns.tolist()[:-1]]
play_types[0] = 'P&R Ball<br>Handler'
play_types[5] = 'P&R<br>Roll<br>Man'

# In[3]:


# Normalize dataframes, add row for 'Average', put cluster series in list

cluster_series = [] # League-specific series of player clusters
sample_size_series = [] # League-specific series of player sample sizes (number of total plays)
name_series, raw_dfs, scaled_dfs, z_dfs = [], [], [], []

for i in range(len(dfs)):
    cluster_series.append(dfs[i].iloc[:, -1]) # Add cluster column to cluster_series
    df = dfs[i].iloc[:, :-1] # Remove cluster column from dataframe
    sample_size_series.append(df.sum(axis=1)) # Add sample size to sample_size_series before normalization

    df = pd.concat([df, pd.DataFrame(df.mean().rename("League Average")).transpose()]) # Add row with league average
    # TODO include League Average for closest players?
    df = df.div(df.sum(axis=1), axis=0) # Normalize by row
    
    name_series.append(df.index.values)
    raw_df = df.reset_index(drop=True)
    raw_dfs.append(raw_df.values)
    scaled_dfs.append(((raw_df-raw_df.min())/(raw_df.max()-raw_df.min())).values)
    z_dfs.append(((raw_df-raw_df.mean())/raw_df.std()).values)
    
    # Add cluster centres
    cluster_centres.columns = df.columns
    dfs[i] = pd.concat([df, cluster_centres])
    
for d in dfs:
    print(d.shape)


# In[4]:


clusters_df = pd.concat(cluster_series, axis=1)
clusters_df = clusters_df.reset_index().rename(columns={'index': 'Player'})
labels = ['Stretch bigs', 'Playmakers', 'Ball handlers', 'Driving bigs', 'Spot-up Shooters', 'Low-post bigs', 'Off-ball wings']
labels = [labels[i] + ' (' + str(i) + ')' for i in range(len(labels))]
clusters_df


# In[5]:
def get_n_closest_players(arr, player_index, n):
    """ arr is df.values """
    temp = arr - arr[player_index]
    l = np.sum(np.square(temp), axis=1)
    indices = np.argpartition(l, n+1)[:n+1]
    return indices[indices != player_index]

player_details_df = pd.concat([x for y in zip(cluster_series, sample_size_series) for x in y], axis=1)
player_details_df = player_details_df.reset_index()
player_details_df.columns = ['Player', 'FIBA Cluster', 'FIBA # Plays', 'NBA_Cluster', 'NBA # Plays', 'INTL Cluster', 'INTL # Plays']
player_details_df = player_details_df.fillna('N/A').replace([i for i in range(7)], [l for l in cluster_labels])
player_details_df


cluster_num = 3
dfs_by_cluster = []
for i in range(7):
    dfs_by_cluster.append(pd.DataFrame())
    for j in range(3):
        s = (cluster_series[j] == i)
        dfs_by_cluster[i] = pd.concat([dfs_by_cluster[i], dfs[j].loc[s.index[s]]])
    dfs_by_cluster[i] = dfs_by_cluster[i].mean()
df_by_cluster = pd.concat(dfs_by_cluster, axis=1)
df_by_cluster


# In[6]:
def get_polar_plot_cluster(cluster):
    """ cluster is cluster number, 0-6 """
    r = cluster_centres.iloc[cluster].tolist()
    theta = play_types
    return go.Scatterpolar(r = r + [r[0]], theta = theta + [theta[0]], mode = 'lines',
                            name = cluster_labels[cluster], line = {'color': colours[cluster]}, legendgroup = cluster)

cluster_fig = px.scatter_polar()
for i in range(7):
    cluster_fig.add_trace((get_polar_plot_cluster(i)))
cluster_fig.update_layout(polar_angularaxis_rotation = 0, polar_angularaxis_direction = 'counterclockwise',
                        legend = {'xanchor': 'left', 'x': 0.1, 'yanchor': 'middle', 'y': 0.5})
cluster_fig.update_polars(radialaxis_range=[0, 0.65])
cluster_fig.show()

def get_player_table(players):
    if isinstance(players, str):
        players = [players]
    ps = [p for p in players if p != 'League Average']
    return player_details_df[player_details_df['Player'].isin(ps)]

def get_most_similar_players(df_index, player):
    try:
        player_idx = np.where(name_series[df_index] == player)[0][0]
    except IndexError:
        return 'N/A'
    players = get_n_closest_players(scaled_dfs[df_index], player_idx, 5)
    return '\n'.join(name_series[df_index][players])

def get_polar_plots_league(df, players):
    """
    df: A dataframe with the play type distribution for each player in a particular league
    players: A list of players to look up, possibly including 'League Average'
    If a player is not in the dataframe, does not plot this player
    """
    global colours
    
    if not players: # No players passed
        return ((go.Scatterpolar(r = [0 for i in range(10)], theta = play_types, showlegend=False),), 0)
    
    valid_players = df.index.intersection(players)
    
    gs = []
    max_prop = 0
    for i in range(len(players)):
        p = players[i]
        if p in valid_players:
            r = df.loc[p].tolist()
            #theta = df.columns.tolist()
            theta = play_types
            g = go.Scatterpolar(r = r + [r[0]], theta = theta + [theta[0]], mode = 'lines', name = p,
                               line = {'color': colours[i%10]}, legendgroup = i, showlegend = (i not in existing_groups))
            existing_groups.add(i)
            gs.append(g)
            max_prop = max(max_prop, max(r))
    return (tuple(gs), max_prop)


# In[7]:




# In[8]:


dropdown_options = [{"label":i, "value": i} for i in all_names]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

VALID_USERNAME_PASSWORD_PAIRS = {
    'phil': 'CBxUTSPAN2021',
    'arif': 'CBxUTSPAN2021',
    'ethan': 'CBxUTSPAN2021',
    'jamal': 'CBxUTSPAN2021',
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

server = app.server

# app.layout = html.Div([
#     html.H1("Play Type Distribution of Players"),
#     html.P("Select players to display from the dropdown. Selecting 'League Average' shows the mean play type distribution for all players in that competition."),
#     html.P("Click on a player in the legend below to hide/show that series. Zoom in by clicking and dragging, and reset the zoom by double-clicking."),
#     dcc.Dropdown(id="selected_players", options=dropdown_options, value="League Average", multi=True, placeholder="Search for a player or 'League Average'",),
#     dcc.Graph(id='play_type_dist_by_player'),
#     html.P('\n'.join(col_map), style={'whiteSpace': 'pre-wrap'}),
#     dash_table.DataTable(id='player_clusters', columns=[{"name": i, "id": i} for i in clusters_df.columns],
#                         style_header={'fontWeight': 'bold'}, sort_action='native'),
#     dcc.Graph(id='play_type_dist_by_cluster', figure=cluster_fig)])
    
    
# @app.callback(Output('play_type_dist_by_player', 'figure'), [Input("selected_players", "value")])
# def update_figure(players):
#     global existing_groups
#     existing_groups = set()
#     if isinstance(players, str):
#         players = [players]
#     fig = make_subplots(cols=3, specs=[[{"type": "polar"}, {"type": "polar"}, {"type": "polar"}]],
#                        subplot_titles=("<b>FIBA</b>", "<b>NBA</b>", "<b>INTL</b>"))
#     for i in range(3):
#         for g in get_polar_plots_league(dfs[i], players):
#             fig.add_trace(g, row=1, col=i+1)
#             fig.layout.annotations[i].update(y=1.05)
    
#     fig.update_polars(radialaxis_range=[0, 0.85])
#     fig.update_layout(legend = {'orientation': "h", 'yanchor': "bottom", 'y': -0.15}, margin = {'l': 0, 'r': 0, 't': 50})
    
#     return fig

# @app.callback(Output('player_clusters', 'data'), [Input("selected_players", "value")])
# def display_clusters(players):
#     if isinstance(players, str):
#         players = [players]
#     ps = [p for p in players if p != 'League Average']
#     d = clusters_df[clusters_df['Player'].isin(ps)].fillna(' N/A')
#     return d.replace([i for i in range(7)], [l for l in labels]).to_dict('records')

app.layout = html.Div([
    html.H1("Analyzing Players' Play Styles Using Clustering"),
    html.H2("The Seven Clusters"),
    html.P("Below is a graph of the average play type distributions for each cluster. You can show/hide certain clusters by clicking on them in the legend. Hover over a point to print the proportion."),
    dcc.Graph(id='play_type_dist_by_cluster', figure=cluster_fig, config={'scrollZoom': True, "toImageButtonOptions": {"width": None, "height": None}}), 
    html.P("Below is a table of the cluster centres."),
    dash_table.DataTable(id='cluster_centres', columns=[{"name": i, "id": i} for i in cluster_centres_display.columns], data=cluster_centres_display.to_dict('records'), style_header={'fontWeight': 'bold'}, export_format="csv"),
    html.H2("Player Stats"),
    html.P("Select players from the dropdown to show their play type distributions in each league, or show a cluster centre by searching for the cluster number or name. Selecting 'League Average' shows the mean play type distribution for all players in that competition."),
    html.P("Click on a player in the legend to hide/show their stats. Zoom in by clicking and dragging, and reset the zoom by double-clicking."),
    dcc.Dropdown(id="selected_players", options=dropdown_options, value="League Average", multi=True, placeholder="Search for a player or 'League Average'",),
    dcc.Graph(id='play_type_dist_by_player', config={'scrollZoom': True, "toImageButtonOptions": {"width": None, "height": None}}),
    # TODO scrollZoom not working right now
    #html.P('\n'.join(col_map), style={'whiteSpace': 'pre-wrap'}),
    html.P("Once a player is added to the graphs, their clusters and total play counts are shown in the table below. Click 'Export' to export the table as a CSV file."),
    dash_table.DataTable(id='player_clusters', columns=[{"name": i, "id": i} for i in player_details_df.columns],
                        style_header={'fontWeight': 'bold'}, sort_action='native', export_format="csv"),
    html.H2("Similar Players"),
    html.P("The table below shows the most similar players in the competition type to each selected player based on the z-scores of their play type proportions."),
    dash_table.DataTable(id='similar_players', columns=[{"name": i, "id": i} for i in ['Player', 'FIBA Similar', 'NBA Similar', 'INTL Similar']],
                        style_header={'fontWeight': 'bold'}, style_cell = {'whiteSpace': 'pre-line', 'textAlign': 'center'})])
    
@app.callback(Output('play_type_dist_by_player', 'figure'), [Input("selected_players", "value")])
def update_figure(players):
    global existing_groups
    existing_groups = set()
    if isinstance(players, str):
        players = [players]
    fig = make_subplots(cols=3, specs=[[{"type": "polar"}, {"type": "polar"}, {"type": "polar"}]],
                       subplot_titles=("<b>FIBA</b>", "<b>NBA</b>", "<b>INTL</b>"))
    max_prop = 0
    for i in range(3):
        gs, m = get_polar_plots_league(dfs[i], players)
        max_prop = max(max_prop, m)
        for g in gs:
            fig.add_trace(g, row=1, col=i+1)
            fig.layout.annotations[i].update(y=1.05)
    
    if not players: # Hide tick labels if no data
        fig.update_polars(radialaxis_showticklabels = False)
    
    fig.update_polars(radialaxis_range=[0, max_prop + 0.01])
    fig.update_layout(legend = {'orientation': "h", 'yanchor': "bottom", 'y': -0.2}, margin = {'l': 30, 'r': 30, 't': 50})
    #fig.update_yaxes(automargin=True)
    #fig.add_annotation(x=0.75, y=0.5, text='<br>'.join(col_map), showarrow=False, yshift=10)
    return fig
# @app.callback(Output('cluster_centres', 'data'), [Input("cluster_centres", "value")])
# def display_cluster_centres(players):
#     return cluster_centres.to_dict('records')
@app.callback(Output('player_clusters', 'data'), [Input("selected_players", "value")])
def display_player_details(players):
    d = get_player_table(players)
    return d.to_dict('records')

@app.callback(Output('similar_players', 'data'), [Input("selected_players", "value")])
def get_similar_players_table(players):
    # TODO behaviour if no players selected?
    d = []
    if isinstance(players, str):
        players = [players]
    for p in players:
        d.append({'Player': p,
                 'FIBA Similar': get_most_similar_players(0, p),
                 'NBA Similar': get_most_similar_players(1, p),
                 'INTL Similar': get_most_similar_players(2, p)})
    return d
if __name__ == '__main__':
    app.run_server(debug = True)    

#app.run_server(port=8056)
#app.run_server(mode="inline", debug=True, port=8056)

