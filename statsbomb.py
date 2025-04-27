#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # run the app using this line in commnands :streamlit run --theme.base "light" statsbomb.py
# streamlit
# numpy
# matplotlib
# statsbombpy
# highlight_text
# mplsoccer
# plotly
# pip3 freeze > requirements.txt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer.pitch import Pitch, VerticalPitch
from highlight_text import fig_text
import plotly.express as px
from dash import dcc
from mplsoccer import FontManager
import matplotlib.patheffects as path_effects

## competitions
com = sb.competitions()

com_name = com['competition_name']
com_id = com['competition_id']
season_name = com['season_name']
season_id = com['season_id']

com_dict = dict(zip(com_name, com_id))
season_dict = dict(zip(season_name, season_id))


## Matches
def matches_id(data):
    match_id = []
    match_name = []
    match_index = []
    for i in range(len(data)):
        match_index.append(i)
        match_id.append(data['match_id'][i])
        match_name.append(data['home_team'][i] + ' vs ' + data['away_team'][i] + ' ' + data['competition_stage'][i])
    match_dict_id = dict(zip(match_name, match_id))
    match_dict_index = dict(zip(match_name, match_index))
    return match_name, match_dict_index, match_dict_id


def match_data(data, match_index):
    home_team = data['home_team'][match_index]
    away_team = data['away_team'][match_index]
    home_score = data['home_score'][match_index]
    away_score = data['away_score'][match_index]
    stadium = data['stadium'][match_index]
    home_manager = data['home_managers'][match_index]
    away_manager = data['away_managers'][match_index]
    comp_stats = data['competition_stage'][match_index]
    return home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats


## Lineups
def lineups(h, w, data):
    home_lineups = []
    away_lineups = []
    for i in range(len(data)):
        home_lineups.append(data[h]['player_name'])
        away_lineups.append(data[w]['player_name'])
    return home_lineups[0].values, away_lineups[0].values


## substitutions
def sub(events, h, w):
    home_sub_player = []
    away_sub_player = []
    for i in range(len(events['substitutions'])):
        if events['substitutions']['possession_team'][i] == h:
            home_sub_player.append(
                f" {events['substitutions']['player'][i]}     {events['substitutions']['position'][i]}")
        elif events['substitutions']['possession_team'][i] == w:
            away_sub_player.append(
                f" {events['substitutions']['player'][i]}     {events['substitutions']['position'][i]}")
        return home_sub_player, away_sub_player


## events
def shots_goal(shots, h, w, match_id):
    # Size of the pitch in yards (!!!)
    pitchLengthX = 120
    pitchWidthY = 80

    home_team_required = h
    away_team_required = w

    pitch = Pitch(pitch_type='statsbomb', line_color='gray')
    fig, ax = pitch.draw()

    # Plot the shots
    for i, shot in shots.iterrows():
        x = shot['location'][0]
        y = shot['location'][1]

        goal = shot['shot_outcome'] == 'Goal'
        team_name = shot['team']

        circleSize = 2
        circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 5

        if (team_name == home_team_required):
            if goal:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color="red")
                plt.text((x + 1), pitchWidthY - y + 2, shot['player'])
            else:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color="red")
                shotCircle.set_alpha(.2)
        elif (team_name == away_team_required):
            if goal:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color="blue")
                plt.text((pitchLengthX - x + 1), y + 2, shot['player'])
            else:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color="blue")
                shotCircle.set_alpha(.2)

        ax.add_patch(shotCircle)
    plt.text(15, 75, away_team_required + ' shots')
    plt.text(80, 75, home_team_required + ' shots')

    total_shots = len(shots)
    fig_text(s=f'Total Shots: {total_shots}',
             x=.40, y=.80, fontsize=14, fontfamily='Andale Mono', color='black')
    fig.text(.10, .12, f'@ahmedtarek26 / Github', fontstyle='italic', fontsize=12, fontfamily='Andale Mono',
             color='black')
    fig.set_size_inches(10, 7)
    plt.savefig(f'graphs/shots-{match_id}.png', dpi=300)
    st.image(f'graphs/shots-{match_id}.png')
### Goals
def goals(shots, h, w, match_id):
    # Size of the pitch in yards (!!!)
    pitchLengthX = 120
    pitchWidthY = 80

    home_team_required = h
    away_team_required = w

    pitch = Pitch(pitch_type='statsbomb', line_color='#c7d5cc')
    fig, ax = pitch.draw()

    # Plot the shots
    for i, shot in shots.iterrows():
        x = shot['location'][0]
        y = shot['location'][1]
        x_end = shot['shot_end_location'][0]
        y_end = shot['shot_end_location'][1]

        goal = shot['shot_outcome'] == 'Goal'
        team_name = shot['team']

        circleSize = 2
        circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 5

        if (team_name == home_team_required):
            if goal:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color="red")
                plt.text((x - 10), pitchWidthY - y - 2, shot['shot_body_part'], fontsize=12)
                plt.text((x - 10), pitchWidthY - y, f"XG: {round(shot['shot_statsbomb_xg'], 2)}", fontsize=12)
                pitch.arrows(x, pitchWidthY - y, x_end, pitchWidthY - y_end, color='black', width=1,
                             headwidth=5, headlength=5, ax=ax)
            else:
                continue
        elif (team_name == away_team_required):
            if goal:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color="blue")
                plt.text((pitchLengthX - x - 10), y - 2, shot['shot_body_part'], fontsize=12)
                plt.text((pitchLengthX - x - 10), y + 2, f"XG: {round(shot['shot_statsbomb_xg'], 2)}", fontsize=12)
                pitch.arrows(pitchLengthX - x, y, pitchLengthX - x_end, y_end, color='black', width=2,
                             headwidth=5, headlength=5, ax=ax)
            else:
                continue

        ax.add_patch(shotCircle)

    fig.text(.10, .12, f'@ahmedtarek26 / Github', fontstyle='italic', fontsize=12,color='black')
    fig.set_size_inches(10, 7)
    plt.savefig(f'graphs/goals-{match_id}.png', dpi=300)
    st.image(f'graphs/goals-{match_id}.png')
## Dripples
def dribbles(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    # The statsbomb pitch from mplsoccer
    pitch = Pitch(pitch_type='statsbomb',
                  pitch_color='grass', line_color='#c7d5cc', stripe=True)

    pitch.draw(ax=ax)

    # I invert the axis to make it so I am viewing it how I want
    plt.gca().invert_yaxis()

    # plot the points, you can use a for loop to plot the different outcomes if you want
    x_h = []
    y_h = []
    x_w = []
    y_w = []
    for i, shot in events['dribbles'].iterrows():
        if events['dribbles']['possession_team'][i] == h:
            x_h.append(shot['location'][0])
            y_h.append(shot['location'][1])
        elif events['dribbles']['possession_team'][i] == w:
            x_w.append(shot['location'][0])
            y_w.append(shot['location'][1])
    plt.scatter(x_h, y_h, s=100, c='red', alpha=.7, label=h)

    plt.scatter(x_w, y_w, s=100, c='blue', alpha=.7, label=w)
    plt.legend(loc="upper left")

    total_shots = len(events['dribbles'])

    fig_text(s=f'Total Dribbles: {total_shots}',
             x=.49, y=.67, fontsize=14, color='white')

    fig.text(.22, .14, f'@ahmedtarek / Github', fontstyle='italic', fontsize=12,
             color='white')

    plt.savefig(f'graphs/dribbles-{match_id}.png', dpi=300, bbox_inches='tight', facecolor='#486F38')
    st.image(f'graphs/dribbles-{match_id}.png')


## passes
def home_team_passes(events, home_team, match_id):
    x_h = []
    y_h = []

    for i, shot in events['passes'].iterrows():
        if events['passes']['possession_team'][i] == home_team:
            x_h.append(shot['location'][0])
            y_h.append(shot['location'][1])

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color='gray', pitch_color='#22312b')
    bins = (6, 4)

    fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
    fig_text(s=f'{home_team} Passes: {len(x_h)}',
             x=.49, y=.67, fontsize=14, color='yellow')
    fig.text(.22, .14, f'@ahmedtarek26 / Github', fontstyle='italic', fontsize=12, color='yellow')
    # plot the heatmap - darker colors = more passes originating from that square

    bs_heatmap = pitch.bin_statistic(x_h, y_h, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues')
    plt.savefig(f'graphs/{home_team}passes-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/{home_team}passes-{match_id}.png')


def away_team_passes(events, away_team, match_id):
    x_w = []
    y_w = []
    for i, shot in events['dribbles'].iterrows():
        if events['dribbles']['possession_team'][i] == away_team:
            x_w.append(shot['location'][0])
            y_w.append(shot['location'][1])

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color='gray', pitch_color='#22312b')
    bins = (6, 4)
    fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
    fig_text(s=f'{away_team} Passes: {len(x_w)}',
             x=.49, y=.67, fontsize=14, color='yellow')
    fig.text(.22, .14, f'@ahmedtarek26 / Github', fontstyle='italic', fontsize=12, color='yellow')
    fig.set_facecolor('#22312b')
    # plot the heatmap - darker colors = more passes originating from that square

    bs_heatmap = pitch.bin_statistic(x_w, y_w, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Reds')
    plt.savefig(f'graphs/{away_team}passes-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/{away_team}passes-{match_id}.png')

def pass_network(events, team_name, match_id, color="#BF616A"):
    """
    Create a pass network visualization for a team in Streamlit.
    
    Parameters:
    - events: Dictionary containing match events
    - team_name: Name of the team to analyze
    - match_id: Match ID for saving the image
    - color: Team color for visualization
    """
    try:
        # Verify we have passes data
        if 'passes' not in events:
            st.warning(f"No passes data found for {team_name}")
            return
            
        passes = events['passes']
        
        # Filter for team
        team_passes = passes[passes['team'] == team_name]
        if len(team_passes) == 0:
            st.warning(f"No passes found for {team_name}")
            return
            
        # Filter successful passes
        successful_passes = team_passes[team_passes['pass_outcome'].isna()].copy()
        if len(successful_passes) == 0:
            st.warning(f"No successful passes found for {team_name}")
            return
            
        # Extract coordinates
        locations = successful_passes['location'].apply(lambda x: pd.Series(x, index=['x', 'y']))
        successful_passes[['x', 'y']] = locations
        
        # Calculate average positions
        avg_locations = successful_passes.groupby('player')[['x', 'y']].mean()
        
        # Calculate pass counts
        pass_counts = successful_passes['player'].value_counts()
        avg_locations['pass_count'] = avg_locations.index.map(pass_counts)
        
        # Scale node sizes (300-1500 range)
        avg_locations['marker_size'] = 300 + (1200 * (avg_locations['pass_count'] / pass_counts.max()))
        
        # Calculate pass connections
        pass_connections = successful_passes.groupby(
            ['player', 'pass_recipient']).size().reset_index(name='count')
        
        # Merge positions
        pass_connections = pass_connections.merge(
            avg_locations[['x', 'y']], 
            left_on='player', 
            right_index=True
        )
        pass_connections = pass_connections.merge(
            avg_locations[['x', 'y']], 
            left_on='pass_recipient', 
            right_index=True,
            suffixes=['', '_end']
        )
        
        # Scale connection widths (1-5 range)
        pass_connections['width'] = 1 + (4 * (pass_connections['count'] / pass_connections['count'].max()))
        
        # Setup pitch
        pitch = Pitch(pitch_type="statsbomb", pitch_color="white", 
                     line_color="black", linewidth=1)
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.set_facecolor("white")
        
        # Draw connections
        pitch.lines(
            pass_connections.x,
            pass_connections.y,
            pass_connections.x_end,
            pass_connections.y_end,
            lw=pass_connections.width,
            color=color,
            zorder=1,
            ax=ax
        )
        
        # Draw nodes
        pitch.scatter(
            avg_locations.x,
            avg_locations.y,
            s=avg_locations.marker_size,
            color=color,
            edgecolors="black",
            linewidth=0.5,
            alpha=1,
            ax=ax
        )
        
        # Add inner circles
        pitch.scatter(
            avg_locations.x,
            avg_locations.y,
            s=avg_locations.marker_size/2,
            color="white",
            edgecolors="black",
            linewidth=0.5,
            alpha=1,
            ax=ax
        )
        
        # Add player names
        for index, row in avg_locations.iterrows():
            text = ax.text(
                row.x, row.y,
                index.split()[-1],
                color="black",
                va="center",
                ha="center",
                size=10,
                weight="bold"
            )
            text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="white")])
        
        # Add title and credit
        ax.set_title(f"{team_name} Pass Network", fontsize=16, pad=20)
        fig.text(0.1, 0.02, '@ahmedtarek26 / Github', 
                fontstyle='italic', fontsize=12, color='black')
        
        # Save and display in Streamlit
        plt.savefig(f'graphs/pass_network_{team_name}_{match_id}.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        st.image(f'graphs/pass_network_{team_name}_{match_id}.png')
        
    except Exception as e:
        st.error(f"Error creating pass network for {team_name}: {str(e)}")


## foul committeds
def foul_committed(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    # The statsbomb pitch from mplsoccer
    pitch = Pitch(pitch_type='statsbomb',
                  pitch_color='grass', line_color='#c7d5cc', stripe=True)

    pitch.draw(ax=ax)

    # I invert the axis to make it so I am viewing it how I want
    plt.gca().invert_yaxis()

    # plot the points, you can use a for loop to plot the different outcomes if you want
    x_h = []
    y_h = []
    x_w = []
    y_w = []
    for i, foul in events['foul_committeds'].iterrows():
        if events['foul_committeds']['possession_team'][i] == h:
            x_h.append(foul['location'][0])
            y_h.append(foul['location'][1])
        elif events['foul_committeds']['possession_team'][i] == w:
            x_w.append(foul['location'][0])
            y_w.append(foul['location'][1])

    plt.scatter(x_h, y_h, s=100, c='red', alpha=.7, label=h)

    plt.scatter(x_w, y_w, s=100, c='blue', alpha=.7, label=w)
    plt.legend(loc="upper left")

    total_foul_committed = len(events['foul_committeds'])

    fig_text(s=f'Total Foul Committed: {total_foul_committed}',
             x=.16, y=.81, fontsize=14, color='black')

    fig.text(.22, .14, f'@ahmedtarek / Github', fontstyle='italic', fontsize=12, color='white')

    plt.savefig(f'graphs/committed-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/committed-{match_id}.png')


## foul wons
def foul_won(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    # The statsbomb pitch from mplsoccer
    pitch = Pitch(pitch_type='statsbomb',
                  pitch_color='grass', line_color='#c7d5cc', stripe=True)

    pitch.draw(ax=ax)

    # I invert the axis to make it so I am viewing it how I want
    plt.gca().invert_yaxis()

    # plot the points, you can use a for loop to plot the different outcomes if you want
    x_h = []
    y_h = []
    x_w = []
    y_w = []
    for i, foul in events['foul_wons'].iterrows():
        if events['foul_wons']['possession_team'][i] == h:
            x_h.append(foul['location'][0])
            y_h.append(foul['location'][1])
        elif events['foul_wons']['possession_team'][i] == w:
            x_w.append(foul['location'][0])
            y_w.append(foul['location'][1])

    plt.scatter(x_h, y_h, s=100, c='red', alpha=.7, label=h)

    plt.scatter(x_w, y_w, s=100, c='blue', alpha=.7, label=w)
    plt.legend(loc="upper left")

    total_foul_wons = len(events['foul_wons'])

    fig_text(s=f'Total Foul wons: {total_foul_wons}',
             x=.16, y=.81, fontsize=14, color='black')

    fig.text(.22, .14, f'@ahmedtarek / Github', fontstyle='italic', fontsize=12, color='white')
    plt.savefig(f'graphs/wons-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/wons-{match_id}.png')


## carrys
def carrys(events, player, match_id):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    # The statsbomb pitch from mplsoccer
    pitch = Pitch(pitch_type='statsbomb',
                  pitch_color='grass', line_color='#c7d5cc', stripe=True)

    pitch.draw(ax=ax)

    # I invert the axis to make it so I am viewing it how I want
    plt.gca().invert_yaxis()

    # plot the points, you can use a for loop to plot the different outcomes if you want
    x = []
    y = []
    x_end = []
    y_end = []

    for i, carry in events['carrys'].iterrows():
        if events['carrys']['player'][i] == player:
            x.append(carry['location'][0])
            y.append(carry['location'][1])
            x_end.append(carry['carry_end_location'][0])
            y_end.append(carry['carry_end_location'][1])

    plt.scatter(x, y, s=100, c='red', alpha=.7, label='Start')
    plt.scatter(x_end, y_end, s=100, c='blue', alpha=.7, label='End')
    plt.legend(loc='upper left')

    pitch.arrows(x, y, x_end, y_end,
                 color='#C7B097', width=2,
                 headwidth=5, headlength=5, ax=ax)

    fig.text(.22, .14, f'@ahmedtarek / Github', fontstyle='italic', fontsize=12, color='white')
    plt.savefig(f'graphs/carry-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/carry-{match_id}.png')


## interception
def interception(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    # The statsbomb pitch from mplsoccer
    pitch = Pitch(pitch_type='statsbomb',
                  pitch_color='grass', line_color='#c7d5cc', stripe=True)

    pitch.draw(ax=ax)

    # I invert the axis to make it so I am viewing it how I want
    plt.gca().invert_yaxis()

    # plot the points, you can use a for loop to plot the different outcomes if you want
    x_h = []
    y_h = []
    x_w = []
    y_w = []
    for i, foul in events['interceptions'].iterrows():
        if events['interceptions']['possession_team'][i] == h:
            x_h.append(foul['location'][0])
            y_h.append(foul['location'][1])
        elif events['interceptions']['possession_team'][i] == w:
            x_w.append(foul['location'][0])
            y_w.append(foul['location'][1])

    plt.scatter(x_h, y_h, s=100, c='red', alpha=.7, label=h)

    plt.scatter(x_w, y_w, s=100, c='blue', alpha=.7, label=w)
    plt.legend(loc="upper left")

    total_interceptions = len(events['interceptions'])

    fig_text(s=f'Total Interceptions: {total_interceptions}',
             x=.16, y=.81, fontsize=14, color='black')

    fig.text(.22, .14, f'@ahmedtarek / Github', fontstyle='italic', fontsize=12, color='white')
    plt.savefig(f'graphs/interception-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/interception-{match_id}.png')


## dispossesseds
def dispossesseds(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    # The statsbomb pitch from mplsoccer
    pitch = Pitch(pitch_type='statsbomb',
                  pitch_color='grass', line_color='#c7d5cc', stripe=True)

    pitch.draw(ax=ax)

    # I invert the axis to make it so I am viewing it how I want
    plt.gca().invert_yaxis()

    # plot the points, you can use a for loop to plot the different outcomes if you want
    x_h = []
    y_h = []
    x_w = []
    y_w = []
    for i, foul in events['dispossesseds'].iterrows():
        if events['dispossesseds']['possession_team'][i] == h:
            x_h.append(foul['location'][0])
            y_h.append(foul['location'][1])
        elif events['dispossesseds']['possession_team'][i] == w:
            x_w.append(foul['location'][0])
            y_w.append(foul['location'][1])

    plt.scatter(x_h, y_h, s=100, c='red', alpha=.7, label=h)

    plt.scatter(x_w, y_w, s=100, c='blue', alpha=.7, label=w)
    plt.legend(loc="upper left")

    total_dispossesseds = len(events['dispossesseds'])

    fig_text(s=f'Total Dispossesseds: {total_dispossesseds}',
             x=.16, y=.81, fontsize=14, color='black')

    fig.text(.22, .14, f'@ahmedtarek / Github', fontstyle='italic', fontsize=12, color='white')
    plt.savefig(f'graphs/dispossessed-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/dispossessed-{match_id}.png')


## miscontrols
def miscontrols(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    # The statsbomb pitch from mplsoccer
    pitch = Pitch(pitch_type='statsbomb',
                  pitch_color='grass', line_color='#c7d5cc', stripe=True)

    pitch.draw(ax=ax)

    # I invert the axis to make it so I am viewing it how I want
    plt.gca().invert_yaxis()

    # plot the points, you can use a for loop to plot the different outcomes if you want
    x_h = []
    y_h = []
    x_w = []
    y_w = []
    for i, foul in events['miscontrols'].iterrows():
        if events['miscontrols']['possession_team'][i] == h:
            x_h.append(foul['location'][0])
            y_h.append(foul['location'][1])
        elif events['miscontrols']['possession_team'][i] == w:
            x_w.append(foul['location'][0])
            y_w.append(foul['location'][1])

    plt.scatter(x_h, y_h, s=100, c='red', alpha=.7, label=h)

    plt.scatter(x_w, y_w, s=100, c='blue', alpha=.7, label=w)
    plt.legend(loc="upper left")

    total_miscontrols = len(events['miscontrols'])

    fig_text(s=f'Total Miscontrols: {total_miscontrols}',
             x=.16, y=.81, fontsize=14, color='black')

    fig.text(.22, .14, f'@ahmedtarek / Github', fontstyle='italic', fontsize=12, color='white')
    plt.savefig(f'graphs/miscontrol-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/miscontrol-{match_id}.png')


## blocks
def blocks(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(13, 8.5))

    # The statsbomb pitch from mplsoccer
    pitch = Pitch(pitch_type='statsbomb',
                  pitch_color='grass', line_color='#c7d5cc', stripe=True)

    pitch.draw(ax=ax)

    # I invert the axis to make it so I am viewing it how I want
    plt.gca().invert_yaxis()

    # plot the points, you can use a for loop to plot the different outcomes if you want
    block_id = events['blocks']['block'].dropna()
    x_h = []
    y_h = []
    annotation_h = []
    player_h = []
    x_w = []
    y_w = []
    annotation_w = []
    player_w = []
    for i in range(len(block_id)):
        if events['blocks']['possession_team'][block_id.keys()[i]] == h:
            x_h.append(events['blocks']['location'][block_id.keys()[i]][0])
            y_h.append(events['blocks']['location'][block_id.keys()[i]][1])
            annotation_h.append(events['blocks']['block'][block_id.keys()[i]])
            player_h.append(events['blocks']['player'][block_id.keys()[i]])

        elif events['blocks']['possession_team'][block_id.keys()[i]] == w:
            x_w.append(events['blocks']['location'][block_id.keys()[i]][0])
            y_w.append(events['blocks']['location'][block_id.keys()[i]][1])
            annotation_w.append(events['blocks']['block'][block_id.keys()[i]])
            player_w.append(events['blocks']['player'][block_id.keys()[i]])

    plt.scatter(x_h, y_h, s=100, c='red', alpha=.7, label=h)

    plt.scatter(x_w, y_w, s=100, c='blue', alpha=.7, label=w)
    plt.legend(loc="upper left")

    for a in range(len(annotation_h)):
        plt.annotate(annotation_h[a], (x_h[a] + 0.5, y_h[a]), fontsize=15)

    for b in range(len(annotation_w)):
        plt.annotate(annotation_w[b], (x_w[b] + 0.5, y_w[b]))

    fig.text(.22, .14, f'@ahmedtarek / Github', fontstyle='italic', fontsize=12, color='white')

    plt.savefig('bcnjuveshots.png')
    plt.savefig(f'graphs/blocks-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/blocks-{match_id}.png')


## streamlit app
st.title('Discover the competition like couches 😉')
competition = st.selectbox('Choose the competition', (com_dict.keys()))

season = st.selectbox('Choose the season', (season_dict.keys()))
data = sb.matches(competition_id=com_dict[competition], season_id=season_dict[season])
matches_names, matches_idx, matches_id = matches_id(data)
match = st.selectbox('Select the match', matches_names)
sub_2 = st.button('Analyze')
if sub_2:
    home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(
        data, matches_idx[match])
    home_lineup, away_lineup = lineups(home_team, away_team, data=sb.lineups(match_id=matches_id[match]))
    col1, col2, col3 = st.columns([2, 1, 2])
    # st.subheader(f'{home_team} {home_score} : {away_score} {away_team}')
    col1.subheader(f'{home_team}')
    col1.markdown(f'### \t{home_score}')
    col2.subheader('\n')
    col2.subheader('Goals')
    col2.subheader('Manager\n')
    col1.write(f'\n {home_manager}')
    col2.subheader(f'Lineup ')
    for i in range(len(home_lineup)):
        col1.write('\n \n \n')
        col1.write(f'- {home_lineup[i]}')
    col3.subheader(f'\n{away_team}')
    col3.markdown(f'### \t{away_score}')
    col3.write(f'\n {away_manager}')
    for i in range(len(away_lineup)):
        col3.write('\n \n \n')
        col3.write(f'- {away_lineup[i]}')

    events = sb.events(match_id=matches_id[match], split=True)
    # st.subheader('Substitutions')
    # home_sub, away_sub = sub(events, home_team, away_team)
    # for i in range(len(home_sub)):
    #     st.write(f'- {home_sub[i]}')
    # for i in range(len(away_sub)):
    #     st.write(f'{away_sub[i]}')
    st.subheader(f'{stadium} Stadium')
    st.subheader(f'{comp_stats} Stage')
    # st.subheader(f"Injury Stoppages Time period")
    # inj_time = []
    # x = len(events['injury_stoppage'])
    # for i in range(x):
    #     inj_time.append(
    #         f"- {events['injury_stoppages']['player'][i]} Time period  {events['injury_stoppages']['minute'][i]}:{events['injury_stoppages']['second'][i]}")
    # for j in range(len(inj_time)):
    #     st.write(inj_time[i])
    st.subheader(f'{home_team} shots vs {away_team} shots')
    shots_goal(events['shots'], home_team, away_team, matches_id[match])

    st.subheader('Goals')
    goals(events['shots'], home_team, away_team, matches_id[match])

    st.subheader('Dribbles')
    dribbles(events, home_team, away_team, matches_id[match])

    st.subheader(f'{home_team} pass map')
    home_team_passes(events, home_team, matches_id[match])
    st.subheader(f'{away_team} pass map')
    home_team_passes(events, away_team, matches_id[match])


    st.subheader(f'{home_team} Pass Network')
    pass_network(events, home_team, matches_id[match], color="#BF616A")  # Red color

    st.subheader(f'{away_team} Pass Network')
    pass_network(events, away_team, matches_id[match], color="#5E81AC")  # Blue color


    st.subheader('Foul Committeds in the match')
    foul_committed(events, home_team, away_team, matches_id[match])
    st.plotly_chart(px.bar(events['foul_committeds'], x=['player', 'position'], color='position'))

    st.subheader('Foul Wons in the match')
    foul_won(events, home_team, away_team, matches_id[match])
    st.plotly_chart(px.bar(events['foul_wons'], x=['player', 'position'], color='position'))

    st.subheader('Clearances in the match')
    st.plotly_chart(px.bar(events['clearances'], x=['player'], color='possession_team'))

    st.subheader('Carrys in the match')
    st.write('A carry is defined as any movement of the ball by a player which is greater than five metres from where '
             'they received the ball.')
    st.plotly_chart(px.bar(events['carrys'], x=['player', 'position'], color='position'))
    for i in range(len(events['carrys']['player'].unique())):
        st.write(f"#### {events['carrys']['player'].unique()[i]} ")
        carrys(events, events['carrys']['player'].unique()[i], matches_id[match])
    st.subheader('Interceprions in the match')
    st.write('Intercepting involves stealing the ball from your opposition')
    interception(events, home_team, away_team, matches_id[match])
    st.plotly_chart(px.bar(events['interceptions'], x=['player', 'position'], color='position'))

    st.subheader('Dispossessed in the match')
    st.write('Dispossessed means being tackled by an opponent without attempting to dribble past them.')
    dispossesseds(events, home_team, away_team, matches_id[match])
    st.plotly_chart(px.bar(events['dispossesseds'], x=['player', 'position'], color='position'))

    st.subheader('Miscontrols in the match')
    miscontrols(events, home_team, away_team, matches_id[match])
    st.plotly_chart(px.bar(events['miscontrols'], x=['player', 'position'], color='position'))

    # st.subheader(f'actions with blocks')
    # blocks(events, home_team, away_team, matches_id[match])
