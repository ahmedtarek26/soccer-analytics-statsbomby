#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer.pitch import Pitch
from highlight_text import fig_text
import plotly.express as px
import os
import matplotlib.patheffects as path_effects

# Set page config must be first Streamlit command
st.set_page_config(layout="wide", page_title="Football Match Analysis")

# Then proceed with the rest of your code
os.makedirs('graphs', exist_ok=True)

# Theme configuration
DARK_MODE = st.sidebar.checkbox("Dark Mode", value=True)

if DARK_MODE:
    # Dark theme colors
    BG_COLOR = "#121212"
    PITCH_COLOR = "#1a1a1a"
    LINE_COLOR = "#efefef"
    TEXT_COLOR = "#ffffff"
    HOME_COLOR = "#ff6b6b"  # Soft red
    AWAY_COLOR = "#64b5f6"  # Soft blue
    BUTTON_COLOR = "#4CAF50"
    BUTTON_HOVER = "#45a049"
    FIG_BG_COLOR = "#2d2d2d"
    SELECTBOX_BG = "#3d3d3d"
else:
    # Light theme colors
    BG_COLOR = "#f8f9fa"
    PITCH_COLOR = "#f8f9fa"
    LINE_COLOR = "#000000"
    TEXT_COLOR = "#000000"
    HOME_COLOR = "#d62828"  # Red
    AWAY_COLOR = "#003049"  # Dark blue
    BUTTON_COLOR = "#4285F4"
    BUTTON_HOVER = "#3367D6"
    FIG_BG_COLOR = "#ffffff"
    SELECTBOX_BG = "#ffffff"


# Font settings
FONT = 'DejaVu Sans'
FONT_BOLD = 'DejaVu Sans'
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
    fig, ax = pitch.draw(figsize=(10, 6.5))
    fig.set_facecolor(FIG_BG_COLOR)

    for i, shot in shots.iterrows():
        x = shot['location'][0]
        y = shot['location'][1]
        goal = shot['shot_outcome'] == 'Goal'
        team_name = shot['team']
        circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 5

        if (team_name == h):
            if goal:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                # Improved player name text
                text = ax.text(x + 1, pitchWidthY - y + 2, shot['player'], 
                             fontsize=FONT_SIZE_SM, color=TEXT_COLOR, 
                             ha='center', va='center', fontfamily=FONT)
                text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
            else:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                shotCircle.set_alpha(.2)
        elif (team_name == w):
            if goal:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                # Improved player name text
                text = ax.text(pitchLengthX - x + 1, y + 2, shot['player'], 
                             fontsize=FONT_SIZE_SM, color=TEXT_COLOR,
                             ha='center', va='center', fontfamily=FONT)
                text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
            else:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                shotCircle.set_alpha(.2)

        ax.add_patch(shotCircle)
    
    plt.text(15, 75, w + ' shots', color=TEXT_COLOR, fontfamily=FONT)
    plt.text(80, 75, h + ' shots', color=TEXT_COLOR, fontfamily=FONT)

    total_shots = len(shots)
    fig_text(s=f'Total Shots: {total_shots}',
             x=.40, y=.80, fontsize=FONT_SIZE_LG, fontfamily=FONT_BOLD, color=TEXT_COLOR)
    # Moved credit to bottom left with smaller font
    fig.text(.02, .02, f'@ahmedtarek26 / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM-2, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/shots-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/shots-{match_id}.png')

def goals(shots, h, w, match_id):
    pitchLengthX = 120
    pitchWidthY = 80

    pitch = Pitch(pitch_type='statsbomb', line_color='#c7d5cc', pitch_color=PITCH_COLOR)
    fig, ax = pitch.draw(figsize=(10, 6.5))
    fig.set_facecolor(FIG_BG_COLOR)

    for i, shot in shots.iterrows():
        x = shot['location'][0]
        y = shot['location'][1]
        x_end = shot['shot_end_location'][0]
        y_end = shot['shot_end_location'][1]
        goal = shot['shot_outcome'] == 'Goal'
        team_name = shot['team']
        circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 5

        if (team_name == h):
            if goal:
                shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                # Improved text styling
                text1 = ax.text(x - 10, pitchWidthY - y - 2, shot['shot_body_part'], 
                              fontsize=FONT_SIZE_SM, color=TEXT_COLOR, fontfamily=FONT)
                text1.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
                
                text2 = ax.text(x - 10, pitchWidthY - y, f"xG: {round(shot['shot_statsbomb_xg'], 2)}", 
                              fontsize=FONT_SIZE_SM, color=TEXT_COLOR, fontfamily=FONT)
                text2.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
                
                pitch.arrows(x, pitchWidthY - y, x_end, pitchWidthY - y_end, 
                           color=TEXT_COLOR, width=1, headwidth=5, headlength=5, ax=ax)
        elif (team_name == w):
            if goal:
                shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                # Improved text styling
                text1 = ax.text(pitchLengthX - x - 10, y - 2, shot['shot_body_part'], 
                              fontsize=FONT_SIZE_SM, color=TEXT_COLOR, fontfamily=FONT)
                text1.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
                
                text2 = ax.text(pitchLengthX - x - 10, y + 2, f"xG: {round(shot['shot_statsbomb_xg'], 2)}", 
                              fontsize=FONT_SIZE_SM, color=TEXT_COLOR, fontfamily=FONT)
                text2.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
                
                pitch.arrows(pitchLengthX - x, y, pitchLengthX - x_end, y_end, 
                           color=TEXT_COLOR, width=2, headwidth=5, headlength=5, ax=ax)

        if goal:
            ax.add_patch(shotCircle)

    # Moved credit to bottom left with smaller font
    fig.text(.02, .02, f'@ahmedtarek26 / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM-2, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/goals-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/goals-{match_id}.png')

def dribbles(events, h, w, match_id):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)

    pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
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
    
    plt.scatter(x_h, y_h, s=80, c=HOME_COLOR, alpha=.7, label=h)
    plt.scatter(x_w, y_w, s=80, c=AWAY_COLOR, alpha=.7, label=w)
    
    legend = plt.legend(loc="upper left", framealpha=0.8)
    legend.get_frame().set_facecolor(FIG_BG_COLOR)
    
    total_shots = len(events['dribbles'])
    fig_text(s=f'Total Dribbles: {total_shots}',
             x=.49, y=.67, fontsize=FONT_SIZE_LG, color=TEXT_COLOR, fontfamily=FONT)
    # Moved credit to bottom left with smaller font
    fig.text(.02, .02, f'@ahmedtarek / Github', 
             fontstyle='italic', fontsize=FONT_SIZE_SM-2, fontfamily=FONT, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f'graphs/dribbles-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/dribbles-{match_id}.png')

# Pass map functions (commented out but kept in code)
'''
def home_team_passes(events, home_team, match_id):
    x_h = []
    y_h = []

    for i, shot in events['passes'].iterrows():
        if events['passes']['possession_team'][i] == home_team:
            x_h.append(shot['location'][0])
            y_h.append(shot['location'][1])

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color='gray', pitch_color='#22312b')
    bins = (6, 4)

    fig, ax = pitch.draw(figsize=(10, 6.5))
    fig.set_facecolor(FIG_BG_COLOR)
    
    fig_text(s=f'{home_team} Passes: {len(x_h)}',
             x=.49, y=.67, fontsize=14, color='yellow')
    fig.text(.22, .14, f'@ahmedtarek26 / Github', 
             fontstyle='italic', fontsize=12, color='yellow')

    bs_heatmap = pitch.bin_statistic(x_h, y_h, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues')
    
    plt.tight_layout()
    plt.savefig(f'graphs/{home_team}passes-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/{home_team}passes-{match_id}.png')
'''

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
        
        pitch = Pitch(pitch_type="statsbomb", pitch_color="white", 
                     line_color="black", linewidth=1)
        fig, ax = pitch.draw(figsize=(10, 6.5))
        fig.set_facecolor(FIG_BG_COLOR)
        
        heatmap_bins = (6, 4)
        bs_heatmap = pitch.bin_statistic(successful_passes['x'], successful_passes['y'], 
                                        statistic='count', bins=heatmap_bins)
        pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues' if color == "#BF616A" else 'Reds', 
                     alpha=0.3, zorder=0.5)
        
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
                color="black",
                va="center",
                ha="center",
                size=10,
                weight="bold",
                zorder=4
            )
            text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="white")])
        
        ax.set_title(f"{team_name} Pass Network", fontsize=16, pad=20)
        # Moved credit to bottom left with smaller font
        fig.text(.02, .02, '@ahmedtarek26 / Github', 
                fontstyle='italic', fontsize=FONT_SIZE_SM-2, color='black')
        
        plt.tight_layout()
        plt.savefig(f'graphs/pass_network_{team_name}_{match_id}.png', 
                   dpi=300, bbox_inches='tight')
        st.image(f'graphs/pass_network_{team_name}_{match_id}.png')
        
    except Exception as e:
        st.error(f"Error creating pass network for {team_name}: {str(e)}")

## streamlit app
st.set_page_config(layout="wide", page_title="Football Match Analysis")

# Custom CSS with dark/light mode support
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {BG_COLOR};
        }}
        .stSelectbox > div > div {{
            background-color: {SELECTBOX_BG};
            border-radius: 8px;
            border: 1px solid {'#555' if DARK_MODE else '#ddd'};
            color: {TEXT_COLOR};
        }}
        .stSelectbox label {{
            color: {TEXT_COLOR} !important;
        }}
        .stButton>button {{
            background-color: {BUTTON_COLOR};
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {BUTTON_HOVER};
            color: white;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_COLOR};
        }}
        .css-1aumxhk {{
            background-color: {FIG_BG_COLOR};
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
            font-size: 16px;
            font-weight: bold;
        }}
    </style>
""", unsafe_allow_html=True)

st.title('⚽ Football Match Analysis')

# Competition selection
try:
    competition = st.selectbox('Choose the competition', com_dict.keys())
    season = st.selectbox('Choose the season', season_dict.keys())
    
    # Load matches data
    data = sb.matches(competition_id=com_dict[competition], season_id=season_dict[season])
    matches_names, matches_idx, matches_id_dict = matches_id(data)
    match = st.selectbox('Select the match', matches_names)
    
    if st.button('Analyze Match'):
        home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(
            data, matches_idx[match])
        
        if None in [home_team, away_team]:
            st.error("Could not load match data")
            st.stop()
        
        match_id = matches_id_dict[match]
        
        # Load lineups
        home_lineup, away_lineup = lineups(home_team, away_team, sb.lineups(match_id=match_id))
        
        # Match header
        st.markdown(f"""
            <div style="background-color:{'#2d2d2d' if DARK_MODE else '#003049'};
                    padding:1.5rem;border-radius:12px;margin-bottom:2rem;color:white;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="text-align:center;flex:1;">
                        <h2 style="color:white;margin-bottom:0;">{home_team}</h2>
                        <h1 style="color:white;margin-top:0;">{home_score}</h1>
                    </div>
                    <div style="text-align:center;flex:1;">
                        <h3 style="color:white;">vs</h3>
                        <p style="color:white;margin-bottom:0;">{comp_stats}</p>
                        <p style="color:white;margin-top:0;">🏟️ {stadium}</p>
                    </div>
                    <div style="text-align:center;flex:1;">
                        <h2 style="color:white;margin-bottom:0;">{away_team}</h2>
                        <h1 style="color:white;margin-top:0;">{away_score}</h1>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Lineups
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

        # Load events
        events = sb.events(match_id=match_id, split=True)
        
        # Visualizations
        st.markdown("---")
        st.header("📊 Match Visualizations")
        
        tab1, tab2 = st.tabs(["⚽ Shots & Goals", "🔄 Passing"])
        
        with tab1:
            if 'shots' in events:
                st.subheader(f'🎯 {home_team} shots vs {away_team} shots')
                shots_goal(events['shots'], home_team, away_team, match_id)

                st.subheader('🥅 Goals Analysis')
                goals(events['shots'], home_team, away_team, match_id)
            else:
                st.warning("No shots data available for this match")

        with tab2:
            st.subheader(f'🔴 {home_team} Pass Network')
            pass_network(events, home_team, match_id, HOME_COLOR)

            st.subheader(f'🔵 {away_team} Pass Network')
            pass_network(events, away_team, match_id, AWAY_COLOR)

            if 'dribbles' in events:
                st.subheader('🏃‍♂️ Dribbles')
                dribbles(events, home_team, away_team, match_id)
            else:
                st.warning("No dribbles data available for this match")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")