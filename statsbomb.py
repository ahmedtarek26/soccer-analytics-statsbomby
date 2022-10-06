#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # run the app using this line in commnands :streamlit run --theme.base "light" statsbomb.py

import streamlit as st
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer.pitch import Pitch, VerticalPitch
from highlight_text import fig_text
import plotly.express as px
from dash import dcc

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


## events
def shots(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    # The statsbomb pitch from mplsoccer
    pitch = Pitch(pitch_type='statsbomb', half=True,
                  pitch_color='grass', line_color='#c7d5cc', stripe=True)

    pitch.draw(ax=ax)

    # I invert the axis to make it so I am viewing it how I want
    plt.gca().invert_yaxis()

    # plot the points, you can use a for loop to plot the different outcomes if you want
    x_h = []
    y_h = []
    x_w = []
    y_w = []
    for i, shot in events['shots'].iterrows():
        if events['shots']['possession_team'][i] == h:
            x_h.append(shot['location'][0])
            y_h.append(shot['location'][1])
        elif events['shots']['possession_team'][i] == w:
            x_w.append(shot['location'][0])
            y_w.append(shot['location'][1])

    plt.scatter(x_h, y_h, s=100, c='red', alpha=.7, label=h)

    plt.scatter(x_w, y_w, s=100, c='blue', alpha=.7, label=w)
    plt.legend(loc="upper left")

    total_shots = len(events['shots'])
    fig_text(s=f'Total Shots: {total_shots}',
             x=.49, y=.67, fontsize=14, color='white')

    fig.text(.22, .14, f'@ahmedtarek26 / Github', fontstyle='italic', fontsize=12,
             color='white')

    plt.savefig(f'graphs/shots-{match_id}.png', dpi=300, bbox_inches='tight', facecolor='#486F38')
    st.image(f'graphs/shots-{match_id}.png')


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


## clearances
def clearance(events, match_id):
    data = events['clearances']
    fig = px.bar(data, x='player', color='possession_team')
    st.plotly_chart(fig)


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

    st.subheader(f'{stadium} Stadium')
    st.subheader(f'{comp_stats} Stage')

    events = sb.events(match_id=matches_id[match], split=True, flatten_attrs=False)
    # st.subheader(f"Injury Stoppages")
    # inj_time = []
    # for i in range(0, len(events['injury_stoppage']) - 1):
    #     inj_time.append(
    #         f"- {events['injury_stoppages']['player'][i]} Time period  {events['injury_stoppages']['minute'][i]}:{events['injury_stoppages']['second'][i]}")
    # for j in range(0, len(inj_time) - 1):
    #     st.write(inj_time[i])
    st.subheader(f'{home_team} shots vs {away_team} shots')
    shots(events, home_team, away_team, matches_id[match])
    st.subheader('Dribbles')
    dribbles(events, home_team, away_team, matches_id[match])
    st.subheader('Clearances in the match')
    clearance(events, matches_id[match])
    st.subheader(f'{home_team} pass map')
    home_team_passes(events, home_team, matches_id[match])
    st.subheader(f'{away_team} pass map')
    home_team_passes(events, away_team, matches_id[match])
