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

# Custom styling
PITCH_COLOR = '#f8f9fa'
LINE_COLOR = '#212529'
TEXT_COLOR = '#212529'
HOME_COLOR = '#d62828'  # Red
AWAY_COLOR = '#003049'  # Dark blue
FONT = 'DejaVu Sans'
FONT_BOLD = 'DejaVu Sans'
FONT_SIZE_SM = 10
FONT_SIZE_MD = 12
FONT_SIZE_LG = 14
FONT_SIZE_XL = 16
FIG_BG_COLOR = '#ffffff'

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

    pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)

    # Plot the shots
    for i, shot in shots.iterrows():
        x = shot['location'][0]
        y = shot['location'][1]

        goal = shot['shot_outcome'] == 'Goal'
        team_name = shot['team']

        circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 8  # Increased size for better visibility

        if (team_name == home_team_required):
            if goal:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                plt.text((x + 1), pitchWidthY - y + 2, shot['player'], 
                        fontsize=FONT_SIZE_SM, fontfamily=FONT)
            else:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                shotCircle.set_alpha(.3)
        elif (team_name == away_team_required):
            if goal:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                plt.text((pitchLengthX - x + 1), y + 2, shot['player'], 
                        fontsize=FONT_SIZE_SM, fontfamily=FONT)
            else:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                shotCircle.set_alpha(.3)

        ax.add_patch(shotCircle)
    
    # Team labels
    plt.text(15, 75, away_team_required + ' shots', 
             fontsize=FONT_SIZE_MD, fontfamily=FONT_BOLD, color=TEXT_COLOR)
    plt.text(80, 75, home_team_required + ' shots', 
             fontsize=FONT_SIZE_MD, fontfamily=FONT_BOLD, color=TEXT_COLOR)

    total_shots = len(shots)
    fig_text(s=f'Total Shots: {total_shots}',
             x=.40, y=.80, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)
    fig.text(.10, .12, f'@ahmedtarek26 / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(f'graphs/shots-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/shots-{match_id}.png')

### Goals
def goals(shots, h, w, match_id):
    # Size of the pitch in yards (!!!)
    pitchLengthX = 120
    pitchWidthY = 80

    home_team_required = h
    away_team_required = w

    pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)

    # Plot the shots
    for i, shot in shots.iterrows():
        x = shot['location'][0]
        y = shot['location'][1]
        x_end = shot['shot_end_location'][0]
        y_end = shot['shot_end_location'][1]

        goal = shot['shot_outcome'] == 'Goal'
        team_name = shot['team']

        circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 8  # Increased size for better visibility

        if (team_name == home_team_required):
            if goal:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                plt.text((x - 10), pitchWidthY - y - 2, shot['shot_body_part'], 
                         fontsize=FONT_SIZE_SM, fontfamily=FONT)
                plt.text((x - 10), pitchWidthY - y, f"xG: {round(shot['shot_statsbomb_xg'], 2)}", 
                         fontsize=FONT_SIZE_SM, fontfamily=FONT)
                pitch.arrows(x, pitchWidthY - y, x_end, pitchWidthY - y_end, 
                            color='black', width=1, headwidth=5, headlength=5, ax=ax)
            else:
                continue
        elif (team_name == away_team_required):
            if goal:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                plt.text((pitchLengthX - x - 10), y - 2, shot['shot_body_part'], 
                         fontsize=FONT_SIZE_SM, fontfamily=FONT)
                plt.text((pitchLengthX - x - 10), y + 2, f"xG: {round(shot['shot_statsbomb_xg'], 2)}", 
                         fontsize=FONT_SIZE_SM, fontfamily=FONT)
                pitch.arrows(pitchLengthX - x, y, pitchLengthX - x_end, y_end, 
                            color='black', width=2, headwidth=5, headlength=5, ax=ax)
            else:
                continue

        ax.add_patch(shotCircle)

    fig.text(.10, .12, f'@ahmedtarek26 / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(f'graphs/goals-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/goals-{match_id}.png')

## Dribbles
def dribbles(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR, stripe=True)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

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
    
    plt.scatter(x_h, y_h, s=100, c=HOME_COLOR, alpha=.7, label=h)
    plt.scatter(x_w, y_w, s=100, c=AWAY_COLOR, alpha=.7, label=w)
    
    # Improved legend
    legend = plt.legend(loc="upper left", framealpha=0.8, edgecolor='none')
    legend.get_frame().set_facecolor(FIG_BG_COLOR)
    
    total_shots = len(events['dribbles'])
    fig_text(s=f'Total Dribbles: {total_shots}',
             x=.49, y=.67, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)
    fig.text(.22, .14, f'@ahmedtarek / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/dribbles-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/dribbles-{match_id}.png')


## passes
def home_team_passes(events, home_team, match_id):
    x_h = []
    y_h = []

    for i, shot in events['passes'].iterrows():
        if events['passes']['possession_team'][i] == home_team:
            x_h.append(shot['location'][0])
            y_h.append(shot['location'][1])

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    bins = (6, 4)

    fig, ax = pitch.draw(figsize=(12, 8), constrained_layout=True, tight_layout=False)
    fig.set_facecolor(FIG_BG_COLOR)
    
    fig_text(s=f'{home_team} Passes: {len(x_h)}',
             x=.49, y=.67, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)
    fig.text(.22, .14, f'@ahmedtarek26 / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)

    bs_heatmap = pitch.bin_statistic(x_h, y_h, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Reds')
    
    plt.tight_layout()
    plt.savefig(f'graphs/{home_team}passes-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/{home_team}passes-{match_id}.png')


def away_team_passes(events, away_team, match_id):
    x_w = []
    y_w = []
    for i, shot in events['dribbles'].iterrows():
        if events['dribbles']['possession_team'][i] == away_team:
            x_w.append(shot['location'][0])
            y_w.append(shot['location'][1])

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    bins = (6, 4)
    fig, ax = pitch.draw(figsize=(12, 8), constrained_layout=True, tight_layout=False)
    fig.set_facecolor(FIG_BG_COLOR)
    
    fig_text(s=f'{away_team} Passes: {len(x_w)}',
             x=.49, y=.67, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)
    fig.text(.22, .14, f'@ahmedtarek26 / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)

    bs_heatmap = pitch.bin_statistic(x_w, y_w, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues')
    
    plt.tight_layout()
    plt.savefig(f'graphs/{away_team}passes-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/{away_team}passes-{match_id}.png')

def pass_network(events, team_name, match_id, color="#d62828"):
    """
    Create a pass network visualization with heatmap overlay for a team in Streamlit.
    """
    try:
        if 'passes' not in events:
            st.warning(f"No passes data found for {team_name}")
            return
            
        passes = events['passes']
        team_passes = passes[passes['team'] == team_name]
        if len(team_passes) == 0:
            st.warning(f"No passes found for {team_name}")
            return
            
        successful_passes = team_passes[team_passes['pass_outcome'].isna()].copy()
        if len(successful_passes) == 0:
            st.warning(f"No successful passes found for {team_name}")
            return
            
        locations = successful_passes['location'].apply(lambda x: pd.Series(x, index=['x', 'y']))
        successful_passes[['x', 'y']] = locations
        
        avg_locations = successful_passes.groupby('player')[['x', 'y']].mean()
        pass_counts = successful_passes['player'].value_counts()
        avg_locations['pass_count'] = avg_locations.index.map(pass_counts)
        avg_locations['marker_size'] = 300 + (1200 * (avg_locations['pass_count'] / pass_counts.max()))
        
        pass_connections = successful_passes.groupby(
            ['player', 'pass_recipient']).size().reset_index(name='count')
        
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
        pass_connections['width'] = 1 + (4 * (pass_connections['count'] / pass_connections['count'].max()))
        
        pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_COLOR, 
                     line_color=LINE_COLOR, linewidth=1)
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.set_facecolor(FIG_BG_COLOR)
        
        heatmap_bins = (6, 4)
        bs_heatmap = pitch.bin_statistic(successful_passes['x'], successful_passes['y'], 
                                        statistic='count', bins=heatmap_bins)
        pitch.heatmap(bs_heatmap, ax=ax, cmap='Reds' if color == "#d62828" else 'Blues', 
                     alpha=0.2, zorder=0.5)
        
        pitch.lines(
            pass_connections.x,
            pass_connections.y,
            pass_connections.x_end,
            pass_connections.y_end,
            lw=pass_connections.width,
            color=color,
            zorder=1,
            ax=ax,
            alpha=0.6
        )
        
        pitch.scatter(
            avg_locations.x,
            avg_locations.y,
            s=avg_locations.marker_size,
            color=color,
            edgecolors="black",
            linewidth=0.5,
            alpha=1,
            ax=ax,
            zorder=2
        )
        
        pitch.scatter(
            avg_locations.x,
            avg_locations.y,
            s=avg_locations.marker_size/2,
            color="white",
            edgecolors="black",
            linewidth=0.5,
            alpha=1,
            ax=ax,
            zorder=3
        )
        
        for index, row in avg_locations.iterrows():
            text = ax.text(
                row.x, row.y,
                index.split()[-1],
                color=TEXT_COLOR,
                va="center",
                ha="center",
                size=FONT_SIZE_SM,
                weight="bold",
                zorder=4,
                fontfamily=FONT
            )
            text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="white")])
        
        ax.set_title(f"{team_name} Pass Network", fontsize=FONT_SIZE_LG, pad=20, fontfamily=FONT_BOLD)
        fig.text(0.1, 0.02, '@ahmedtarek26 / Github', 
                fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
        
        plt.tight_layout()
        plt.savefig(f'graphs/pass_network_{team_name}_{match_id}.png', 
                   dpi=300, bbox_inches='tight')
        st.image(f'graphs/pass_network_{team_name}_{match_id}.png')
        
    except Exception as e:
        st.error(f"Error creating pass network for {team_name}: {str(e)}")
        

## foul committeds
def foul_committed(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

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

    plt.scatter(x_h, y_h, s=100, c=HOME_COLOR, alpha=.7, label=h)
    plt.scatter(x_w, y_w, s=100, c=AWAY_COLOR, alpha=.7, label=w)
    
    legend = plt.legend(loc="upper left", framealpha=0.8, edgecolor='none')
    legend.get_frame().set_facecolor(FIG_BG_COLOR)

    total_foul_committed = len(events['foul_committeds'])

    fig_text(s=f'Total Foul Committed: {total_foul_committed}',
             x=.16, y=.81, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)

    fig.text(.22, .14, f'@ahmedtarek / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(f'graphs/committed-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/committed-{match_id}.png')


## foul wons
def foul_won(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

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

    plt.scatter(x_h, y_h, s=100, c=HOME_COLOR, alpha=.7, label=h)
    plt.scatter(x_w, y_w, s=100, c=AWAY_COLOR, alpha=.7, label=w)
    
    legend = plt.legend(loc="upper left", framealpha=0.8, edgecolor='none')
    legend.get_frame().set_facecolor(FIG_BG_COLOR)

    total_foul_wons = len(events['foul_wons'])

    fig_text(s=f'Total Foul Wons: {total_foul_wons}',
             x=.16, y=.81, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)

    fig.text(.22, .14, f'@ahmedtarek / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/wons-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/wons-{match_id}.png')


## carrys
def carrys(events, player, match_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

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

    plt.scatter(x, y, s=100, c=HOME_COLOR, alpha=.7, label='Start')
    plt.scatter(x_end, y_end, s=100, c=AWAY_COLOR, alpha=.7, label='End')
    
    legend = plt.legend(loc='upper left', framealpha=0.8, edgecolor='none')
    legend.get_frame().set_facecolor(FIG_BG_COLOR)

    pitch.arrows(x, y, x_end, y_end,
                 color='#6c757d', width=2,
                 headwidth=5, headlength=5, ax=ax)

    fig.text(.22, .14, f'@ahmedtarek / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/carry-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/carry-{match_id}.png')


## interception
def interception(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

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

    plt.scatter(x_h, y_h, s=100, c=HOME_COLOR, alpha=.7, label=h)
    plt.scatter(x_w, y_w, s=100, c=AWAY_COLOR, alpha=.7, label=w)
    
    legend = plt.legend(loc="upper left", framealpha=0.8, edgecolor='none')
    legend.get_frame().set_facecolor(FIG_BG_COLOR)

    total_interceptions = len(events['interceptions'])

    fig_text(s=f'Total Interceptions: {total_interceptions}',
             x=.16, y=.81, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)

    fig.text(.22, .14, f'@ahmedtarek / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/interception-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/interception-{match_id}.png')


## dispossesseds
def dispossesseds(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

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

    plt.scatter(x_h, y_h, s=100, c=HOME_COLOR, alpha=.7, label=h)
    plt.scatter(x_w, y_w, s=100, c=AWAY_COLOR, alpha=.7, label=w)
    
    legend = plt.legend(loc="upper left", framealpha=0.8, edgecolor='none')
    legend.get_frame().set_facecolor(FIG_BG_COLOR)

    total_dispossesseds = len(events['dispossesseds'])

    fig_text(s=f'Total Dispossesseds: {total_dispossesseds}',
             x=.16, y=.81, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)

    fig.text(.22, .14, f'@ahmedtarek / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/dispossessed-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/dispossessed-{match_id}.png')


## miscontrols
def miscontrols(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

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

    plt.scatter(x_h, y_h, s=100, c=HOME_COLOR, alpha=.7, label=h)
    plt.scatter(x_w, y_w, s=100, c=AWAY_COLOR, alpha=.7, label=w)
    
    legend = plt.legend(loc="upper left", framealpha=0.8, edgecolor='none')
    legend.get_frame().set_facecolor(FIG_BG_COLOR)

    total_miscontrols = len(events['miscontrols'])

    fig_text(s=f'Total Miscontrols: {total_miscontrols}',
             x=.16, y=.81, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)

    fig.text(.22, .14, f'@ahmedtarek / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/miscontrol-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/miscontrol-{match_id}.png')


## blocks
def blocks(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

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

    plt.scatter(x_h, y_h, s=100, c=HOME_COLOR, alpha=.7, label=h)
    plt.scatter(x_w, y_w, s=100, c=AWAY_COLOR, alpha=.7, label=w)
    
    legend = plt.legend(loc="upper left", framealpha=0.8, edgecolor='none')
    legend.get_frame().set_facecolor(FIG_BG_COLOR)

    for a in range(len(annotation_h)):
        plt.annotate(annotation_h[a], (x_h[a] + 0.5, y_h[a]), 
                    fontsize=FONT_SIZE_SM, fontfamily=FONT)

    for b in range(len(annotation_w)):
        plt.annotate(annotation_w[b], (x_w[b] + 0.5, y_w[b]),
                    fontsize=FONT_SIZE_SM, fontfamily=FONT)

    fig.text(.22, .14, f'@ahmedtarek / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/blocks-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/blocks-{match_id}.png')


## streamlit app
st.set_page_config(layout="wide", page_title="Football Match Analysis")

# Custom CSS for better styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .stSelectbox > div > div {
            background-color: white;
            border-radius: 8px;
        }
        .stButton>button {
            background-color: #003049;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #d62828;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #212529;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title('⚽ Discover the Competition Like Coaches 😉')
competition = st.selectbox('Choose the competition', (com_dict.keys()))

season = st.selectbox('Choose the season', (season_dict.keys()))
data = sb.matches(competition_id=com_dict[competition], season_id=season_dict[season])
matches_names, matches_idx, matches_id = matches_id(data)
match = st.selectbox('Select the match', matches_names)
sub_2 = st.button('Analyze Match')

if sub_2:
    home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(
        data, matches_idx[match])
    home_lineup, away_lineup = lineups(home_team, away_team, data=sb.lineups(match_id=matches_id[match]))
    
    # Match header
    st.markdown(f"""
        <div style="background-color:#003049;padding:1.5rem;border-radius:12px;margin-bottom:2rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;color:white;">
                <div style="text-align:center;flex:1;">
                    <h2 style="color:white;margin-bottom:0;">{home_team}</h2>
                    <h1 style="color:white;margin-top:0;">{home_score}</h1>
                </div>
                <div style="text-align:center;flex:1;">
                    <h3 style="color:white;">vs</h3>
                    <p style="color:white;margin-bottom:0;">{comp_stats}</p>
                    <p style="color:white;margin-top:0;">{stadium}</p>
                </div>
                <div style="text-align:center;flex:1;">
                    <h2 style="color:white;margin-bottom:0;">{away_team}</h2>
                    <h1 style="color:white;margin-top:0;">{away_score}</h1>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Team lineups in columns
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.subheader(f'{home_team} Lineup')
        st.markdown(f"**Manager:** {home_manager}")
        for player in home_lineup:
            st.markdown(f"- {player}")
    
    with col3:
        st.subheader(f'{away_team} Lineup')
        st.markdown(f"**Manager:** {away_manager}")
        for player in away_lineup:
            st.markdown(f"- {player}")

    events = sb.events(match_id=matches_id[match], split=True)
    
    # Visualization sections
    st.markdown("---")
    st.header("📊 Match Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Shots & Goals", "Passing", "Defensive Actions", "Ball Progression"])
    
    with tab1:
        st.subheader(f'{home_team} shots vs {away_team} shots')
        shots_goal(events['shots'], home_team, away_team, matches_id[match])

        st.subheader('Goals Analysis')
        goals(events['shots'], home_team, away_team, matches_id[match])

    with tab2:
        st.subheader(f'{home_team} Pass Network')
        pass_network(events, home_team, matches_id[match], color=HOME_COLOR)

        st.subheader(f'{away_team} Pass Network')
        pass_network(events, away_team, matches_id[match], color=AWAY_COLOR)

        st.subheader(f'{home_team} pass map')
        home_team_passes(events, home_team, matches_id[match])
        
        st.subheader(f'{away_team} pass map')
        home_team_passes(events, away_team, matches_id[match])

    with tab3:
        st.subheader('Defensive Actions')
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Foul Committeds')
            foul_committed(events, home_team, away_team, matches_id[match])
            st.plotly_chart(px.bar(events['foul_committeds'], x=['player', 'position'], color='position',
                                 color_discrete_sequence=[HOME_COLOR, AWAY_COLOR]))

        with col2:
            st.subheader('Foul Wons')
            foul_won(events, home_team, away_team, matches_id[match])
            st.plotly_chart(px.bar(events['foul_wons'], x=['player', 'position'], color='position',
                                 color_discrete_sequence=[HOME_COLOR, AWAY_COLOR]))

        st.subheader('Interceptions')
        interception(events, home_team, away_team, matches_id[match])
        st.plotly_chart(px.bar(events['interceptions'], x=['player', 'position'], color='position',
                             color_discrete_sequence=[HOME_COLOR, AWAY_COLOR]))

    with tab4:
        st.subheader('Dribbles')
        dribbles(events, home_team, away_team, matches_id[match])

        st.subheader('Carrys')
        st.write('A carry is defined as any movement of the ball by a player which is greater than five metres from where they received the ball.')
        for i in range(len(events['carrys']['player'].unique())):
            st.write(f"#### {events['carrys']['player'].unique()[i]} ")
            carrys(events, events['carrys']['player'].unique()[i], matches_id[match])