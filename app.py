#!/usr/bin/env python
# coding: utf-8

# In[1]:


from jupyter_dash import JupyterDash
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_table


# In[2]:


df = pd.read_csv("NBA_INTL_FIBA_Clustering.csv", encoding='latin-1', index_col=0)
all_names = ['League Average'] + df.index.tolist()

# Split up into separate dataframes
FIBA_df = df[[c for c in df.columns if c.startswith('FIBA')]].dropna()
NBA_df = df[[c for c in df.columns if c.startswith('NBA')]].dropna()
INTL_df = df[[c for c in df.columns if c.startswith('INTL')]].dropna()
dfs = [FIBA_df, NBA_df, INTL_df]
for d in dfs:
    print(d.shape)


# In[3]:


# Normalize dataframes, add row for 'Average', put cluster series in list
cluster_series = []
new_col_names = ['BH', 'SU', 'TR', 'IS', 'PU', 'RM', 'CT', 'OR', 'OS', 'HO']
colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i in range(len(dfs)):
    cluster_series.append(dfs[i].iloc[:, -1])
    df = dfs[i].iloc[:, :-1]
    if i == 0:
        original_col_names = [c[5:] for c in df.columns]
    df.columns = new_col_names
    #df = df.rename(columns={c: c[5:] for c in df.columns})
    
    df = df.div(df.sum(axis=1), axis=0) # Normalize by row
    mean_vals = df.mean().rename("League Average")
    dfs[i] = pd.concat([df, pd.DataFrame(mean_vals).transpose()])

col_map = [(new_col_names[i] + ':\t' + original_col_names[i]) for i in range(len(original_col_names))]
for d in dfs:
    print(d.shape)


# In[4]:


clusters_df = pd.concat(cluster_series, axis=1)
clusters_df = clusters_df.reset_index().rename(columns={'index': 'Player'})
labels = ['Stretch bigs', 'Playmakers', 'Ball handlers', 'Driving bigs', 'Spot-up Shooters', 'Low-post bigs', 'Off-ball wings']
labels = [labels[i] + ' (' + str(i) + ')' for i in range(len(labels))]
clusters_df


# In[5]:


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


def get_polar_plots_league(df, players):
    """
    df: A dataframe with the play type distribution for each player in a particular league
    players: A list of players to look up, possibly including 'Average'
    If a player is not in the dataframe, does not plot this player
    """
    global colours
    
    if not players: # No players passed
        return px.line_polar(r = [0 for i in range(10)], theta = df.columns)
    
    valid_players = df.index.intersection(players)
    
    gs = []
    for i in range(len(players)):
        p = players[i]
        if p in valid_players:
            r = df.loc[p].tolist()
            theta = df.columns.tolist()
            g = go.Scatterpolar(r = r + [r[0]], theta = theta + [theta[0]], mode = 'lines', name = p,
                               line = {'color': colours[i%10]}, legendgroup = i, showlegend = (i not in existing_groups))
            existing_groups.add(i)
            gs.append(g)

    return tuple(gs)


# In[7]:


def get_polar_plot_cluster(cluster):
    """
    cluster: the cluster number
    """
    r = df_by_cluster[cluster].tolist()
    theta = df_by_cluster.index.tolist()
    g = go.Scatterpolar(r = r + [r[0]], theta = theta + [theta[0]], mode = 'lines',
                        name = labels[cluster], line = {'color': colours[cluster]}, legendgroup = cluster,
                       )
    return g

cluster_fig = px.scatter_polar()
for i in range(7):
    cluster_fig.add_trace((get_polar_plot_cluster(i)))
cluster_fig.update_layout(polar_angularaxis_rotation=0, polar_angularaxis_direction='counterclockwise')

cluster_fig.show()


# In[8]:


dropdown_options = [{"label":i, "value": i} for i in all_names]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H1("Play Type Distribution of Players"),
    html.P("Select players to display from the dropdown. Selecting 'League Average' shows the mean play type distribution for all players in that competition."),
    html.P("Click on a player in the legend below to hide/show that series. Zoom in by clicking and dragging, and reset the zoom by double-clicking."),
    dcc.Dropdown(id="selected_players", options=dropdown_options, value="League Average", multi=True, placeholder="Search for a player or 'League Average'",),
    dcc.Graph(id='play_type_dist_by_player'),
    html.P('\n'.join(col_map), style={'whiteSpace': 'pre-wrap'}),
    dash_table.DataTable(id='player_clusters', columns=[{"name": i, "id": i} for i in clusters_df.columns],
                        style_header={'fontWeight': 'bold'}, sort_action='native'),
    dcc.Graph(id='play_type_dist_by_cluster', figure=cluster_fig)])
    
    
@app.callback(Output('play_type_dist_by_player', 'figure'), [Input("selected_players", "value")])
def update_figure(players):
    global existing_groups
    existing_groups = set()
    if isinstance(players, str):
        players = [players]
    fig = make_subplots(cols=3, specs=[[{"type": "polar"}, {"type": "polar"}, {"type": "polar"}]],
                       subplot_titles=("<b>FIBA</b>", "<b>NBA</b>", "<b>INTL</b>"))
    for i in range(3):
        for g in get_polar_plots_league(dfs[i], players):
            fig.add_trace(g, row=1, col=i+1)
            fig.layout.annotations[i].update(y=1.05)
    
    fig.update_polars(radialaxis_range=[0, 0.85])
    fig.update_layout(legend = {'orientation': "h", 'yanchor': "bottom", 'y': -0.15}, margin = {'l': 0, 'r': 0, 't': 50})
    
    return fig

@app.callback(Output('player_clusters', 'data'), [Input("selected_players", "value")])
def display_clusters(players):
    if isinstance(players, str):
        players = [players]
    ps = [p for p in players if p != 'League Average']
    d = clusters_df[clusters_df['Player'].isin(ps)].fillna(' N/A')
    return d.replace([i for i in range(7)], [l for l in labels]).to_dict('records')

if __name__ == '__main__':
    app.run_server(debug = True)    

#app.run_server(port=8056)
#app.run_server(mode="inline", debug=True, port=8056)

