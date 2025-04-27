#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# Custom styling with dark mode support
DARK_MODE = True  # Set to False for light mode

if DARK_MODE:
    PITCH_COLOR = '#1a1a1a'
    LINE_COLOR = '#cccccc'
    TEXT_COLOR = '#f8f9fa'
    FIG_BG_COLOR = '#2d2d2d'
    SELECTBOX_BG = '#3d3d3d'
    BUTTON_COLOR = '#4CAF50'
    BUTTON_HOVER = '#45a049'
    HOME_COLOR = '#ff6b6b'  # Soft red
    AWAY_COLOR = '#64b5f6'  # Soft blue
    DRIBBLE_PITCH = '#2d2d2d'
    DRIBBLE_LINE = '#cccccc'
else:
    PITCH_COLOR = '#f8f9fa'
    LINE_COLOR = '#212529'
    TEXT_COLOR = '#212529'
    FIG_BG_COLOR = '#ffffff'
    SELECTBOX_BG = '#ffffff'
    BUTTON_COLOR = '#4285F4'
    BUTTON_HOVER = '#3367D6'
    HOME_COLOR = '#d62828'  # Red
    AWAY_COLOR = '#003049'  # Dark blue
    DRIBBLE_PITCH = '#f8f9fa'
    DRIBBLE_LINE = '#212529'

FONT = 'Segoe UI'
FONT_BOLD = 'Segoe UI Semibold'
FONT_SIZE_SM = 10
FONT_SIZE_MD = 12
FONT_SIZE_LG = 14
FONT_SIZE_XL = 16

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
def shots_goal(shots, h, w, match_id):
    pitchLengthX = 120
    pitchWidthY = 80

    pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    fig, ax = pitch.draw(figsize=(10, 6.5))  # Reduced size
    fig.set_facecolor(FIG_BG_COLOR)

    for i, shot in shots.iterrows():
        x = shot['location'][0]
        y = shot['location'][1]
        goal = shot['shot_outcome'] == 'Goal'
        team_name = shot['team']
        circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 6  # Slightly smaller

        if (team_name == h):
            if goal:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                plt.text((x + 1), pitchWidthY - y + 2, shot['player'], 
                        fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
            else:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                shotCircle.set_alpha(.3)
        elif (team_name == w):
            if goal:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                plt.text((pitchLengthX - x + 1), y + 2, shot['player'], 
                        fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
            else:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                shotCircle.set_alpha(.3)

        ax.add_patch(shotCircle)
    
    plt.text(15, 75, w + ' shots', fontsize=FONT_SIZE_MD, fontfamily=FONT_BOLD, color=TEXT_COLOR)
    plt.text(80, 75, h + ' shots', fontsize=FONT_SIZE_MD, fontfamily=FONT_BOLD, color=TEXT_COLOR)

    total_shots = len(shots)
    fig_text(s=f'Total Shots: {total_shots}',
             x=.40, y=.80, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)
    fig.text(.10, .12, f'@ahmedtarek26 / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/shots-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/shots-{match_id}.png')

def goals(shots, h, w, match_id):
    pitchLengthX = 120
    pitchWidthY = 80

    pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    fig, ax = pitch.draw(figsize=(10, 6.5))  # Reduced size
    fig.set_facecolor(FIG_BG_COLOR)

    for i, shot in shots.iterrows():
        x = shot['location'][0]
        y = shot['location'][1]
        x_end = shot['shot_end_location'][0]
        y_end = shot['shot_end_location'][1]
        goal = shot['shot_outcome'] == 'Goal'
        team_name = shot['team']
        circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 6  # Slightly smaller

        if (team_name == h):
            if goal:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                plt.text((x - 10), pitchWidthY - y - 2, shot['shot_body_part'], 
                         fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
                plt.text((x - 10), pitchWidthY - y, f"xG: {round(shot['shot_statsbomb_xg'], 2)}", 
                         fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
                pitch.arrows(x, pitchWidthY - y, x_end, pitchWidthY - y_end, 
                            color=TEXT_COLOR, width=1, headwidth=5, headlength=5, ax=ax)
        elif (team_name == w):
            if goal:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                plt.text((pitchLengthX - x - 10), y - 2, shot['shot_body_part'], 
                         fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
                plt.text((pitchLengthX - x - 10), y + 2, f"xG: {round(shot['shot_statsbomb_xg'], 2)}", 
                         fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
                pitch.arrows(pitchLengthX - x, y, pitchLengthX - x_end, y_end, 
                            color=TEXT_COLOR, width=2, headwidth=5, headlength=5, ax=ax)

        if goal:
            ax.add_patch(shotCircle)

    fig.text(.10, .12, f'@ahmedtarek26 / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/goals-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/goals-{match_id}.png')

def dribbles(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(10, 6.5))  # Reduced size
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=DRIBBLE_PITCH, line_color=DRIBBLE_LINE)
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
    
    plt.scatter(x_h, y_h, s=80, c=HOME_COLOR, alpha=.7, label=h)  # Smaller markers
    plt.scatter(x_w, y_w, s=80, c=AWAY_COLOR, alpha=.7, label=w)  # Smaller markers
    
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

def home_team_passes(events, home_team, match_id):
    x_h = []
    y_h = []

    for i, shot in events['passes'].iterrows():
        if events['passes']['possession_team'][i] == home_team:
            x_h.append(shot['location'][0])
            y_h.append(shot['location'][1])

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    bins = (6, 4)

    fig, ax = pitch.draw(figsize=(10, 6.5))  # Reduced size
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

def pass_network(events, team_name, match_id, color):
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
        fig, ax = pitch.draw(figsize=(10, 6.5))
        fig.set_facecolor(FIG_BG_COLOR)
        
        heatmap_bins = (6, 4)
        bs_heatmap = pitch.bin_statistic(successful_passes['x'], successful_passes['y'], 
                                        statistic='count', bins=heatmap_bins)
        pitch.heatmap(bs_heatmap, ax=ax, cmap='Reds' if color == HOME_COLOR else 'Blues', 
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

## streamlit app
st.set_page_config(layout="wide", page_title="Football Match Analysis")

# Custom CSS with dark mode support
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {'#121212' if DARK_MODE else '#f8f9fa'};
        }}
        .stSelectbox > div > div {{
            background-color: {SELECTBOX_BG};
            border-radius: 12px;
            border: 1px solid {'#555' if DARK_MODE else '#ddd'};
            color: {TEXT_COLOR};
        }}
        .stSelectbox label {{
            color: {TEXT_COLOR} !important;
        }}
        .stButton>button {{
            background-color: {BUTTON_COLOR};
            color: white;
            border-radius: 20px;
            padding: 0.5rem 1.5rem;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {BUTTON_HOVER};
            transform: scale(1.05);
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_COLOR};
        }}
        .css-1aumxhk {{
            background-color: {'#2d2d2d' if DARK_MODE else '#ffffff'};
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,{'0.2' if DARK_MODE else '0.1'});
            border: 1px solid {'#444' if DARK_MODE else '#eee'};
        }}
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
            font-size: 16px;
            font-weight: bold;
        }}
    </style>
""", unsafe_allow_html=True)

st.title('⚽ Discover the Competition Like Coaches 😉')
competition = st.selectbox('Choose the competition', (com_dict.keys()))

season = st.selectbox('Choose the season', (season_dict.keys()))
data = sb.matches(competition_id=com_dict[competition], season_id=season_dict[season])
matches_names, matches_idx, matches_id = matches_id(data)
match = st.selectbox('Select the match', matches_names)
sub_2 = st.button('Analyze Match 🚀', key='analyze_btn')

if sub_2:
    home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(
        data, matches_idx[match])
    home_lineup, away_lineup = lineups(home_team, away_team, data=sb.lineups(match_id=matches_id[match]))
    
    # Match header with improved styling
    st.markdown(f"""
        <div style="background:{'linear-gradient(135deg, #2d2d2d, #1a1a1a)' if DARK_MODE else 'linear-gradient(135deg, #003049, #004d7a)'};
                padding:1.5rem;border-radius:12px;margin-bottom:2rem;color:white;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="text-align:center;flex:1;">
                    <h2 style="color:white;margin-bottom:0;">{home_team}</h2>
                    <h1 style="color:white;margin-top:0;font-size:3rem;">{home_score}</h1>
                </div>
                <div style="text-align:center;flex:1;">
                    <h3 style="color:white;">vs</h3>
                    <p style="color:white;margin-bottom:0;font-size:1.1rem;">{comp_stats}</p>
                    <p style="color:white;margin-top:0;font-size:1.1rem;">🏟️ {stadium}</p>
                </div>
                <div style="text-align:center;flex:1;">
                    <h2 style="color:white;margin-bottom:0;">{away_team}</h2>
                    <h1 style="color:white;margin-top:0;font-size:3rem;">{away_score}</h1>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Team lineups in columns with better styling
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        with st.container():
            st.subheader(f'👥 {home_team} Lineup')
            st.markdown(f"**👨‍💼 Manager:** {home_manager}")
            for player in home_lineup:
                st.markdown(f"- ⚽ {player}")
    
    with col3:
        with st.container():
            st.subheader(f'👥 {away_team} Lineup')
            st.markdown(f"**👨‍💼 Manager:** {away_manager}")
            for player in away_lineup:
                st.markdown(f"- ⚽ {player}")

    events = sb.events(match_id=matches_id[match], split=True)
    
    # Visualization sections with tabs
    st.markdown("---")
    st.header("📊 Match Visualizations")
    
    tab1, tab2 = st.tabs(["⚽ Shots & Goals", "🔄 Passing & Movement"])
    
    with tab1:
        st.subheader(f'🎯 {home_team} shots vs {away_team} shots')
        shots_goal(events['shots'], home_team, away_team, matches_id[match])

        st.subheader('🥅 Goals Analysis')
        goals(events['shots'], home_team, away_team, matches_id[match])

        st.subheader('🏃‍♂️ Dribbles')
        dribbles(events, home_team, away_team, matches_id[match])

    with tab2:
        st.subheader(f'🔴 {home_team} Pass Network')
        pass_network(events, home_team, matches_id[match], color=HOME_COLOR)

        st.subheader(f'🔵 {away_team} Pass Network')
        pass_network(events, away_team, matches_id[match], color=AWAY_COLOR)

        st.subheader(f'🔴 {home_team} pass map')
        home_team_passes(events, home_team, matches_id[match])
        
        st.subheader(f'🔵 {away_team} pass map')
        home_team_passes(events, away_team, matches_id[match])