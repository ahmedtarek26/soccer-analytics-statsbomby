#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Updated for Streamlit Cloud deployment with enhanced design

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer.pitch import Pitch, VerticalPitch
from highlight_text import fig_text
import plotly.express as px
from mplsoccer import FontManager
import matplotlib.patheffects as path_effects

# ========== SETUP & CONFIGURATION ==========
# Create graphs directory if not exists
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Custom Font Setup
font_url = "https://raw.githubusercontent.com/google/fonts/main/ofl/raleway/Raleway%5Bwght%5D.ttf"
custom_font = FontManager(font_url)

# Color Palette
PALETTE = {
    'background': '#f8f9fa',
    'text': '#212529',
    'primary': '#4361ee',  # Blue
    'secondary': '#3a0ca3',  # Dark Blue
    'accent': '#7209b7',
    'home': '#e63946',  # Red
    'away': '#457b9d',  # Blue
    'pitch': '#f8f9fa',
    'line': '#495057'
}

# Pitch Style
PITCH_STYLE = {
    'pitch_type': 'statsbomb',
    'pitch_color': PALETTE['pitch'],
    'line_color': PALETTE['line'],
    'linewidth': 1.5
}

# ========== HELPER FUNCTIONS ==========
def setup_figure(title="", size=(12, 8)):
    """Create standardized figure with consistent styling"""
    fig, ax = plt.subplots(figsize=size)
    fig.set_facecolor(PALETTE['background'])
    ax.patch.set_facecolor(PALETTE['background'])
    if title:
        fig.suptitle(title, fontproperties=custom_font.prop, fontsize=20, color=PALETTE['text'])
    return fig, ax

def add_credit(fig):
    """Add consistent credit to visualizations"""
    fig.text(0.1, 0.02, '@ahmedtarek26 / Github', 
            fontstyle='italic', fontsize=10, 
            color=PALETTE['text'], fontproperties=custom_font.prop)

# ========== ORIGINAL PROJECT FUNCTIONS (UPDATED) ==========
## competitions
com = sb.competitions()
com_dict = dict(zip(com['competition_name'], com['competition_id']))
season_dict = dict(zip(com['season_name'], com['season_id']))

## Matches
def matches_id(data):
    match_id = []
    match_name = []
    match_index = []
    for i in range(len(data)):
        match_index.append(i)
        match_id.append(data['match_id'][i])
        match_name.append(f"{data['home_team'][i]} vs {data['away_team'][i]} ({data['competition_stage'][i]})")
    return match_name, dict(zip(match_name, match_index)), dict(zip(match_name, match_id))

def match_data(data, match_index):
    return (
        data['home_team'][match_index],
        data['away_team'][match_index],
        data['home_score'][match_index],
        data['away_score'][match_index],
        data['stadium'][match_index],
        data['home_managers'][match_index],
        data['away_managers'][match_index],
        data['competition_stage'][match_index]
    )

## Lineups
def lineups(h, w, data):
    return data[h]['player_name'].values, data[w]['player_name'].values

## Visualizations (updated with new design)
def shots_goal(shots, h, w, match_id):
    fig, ax = setup_figure(f"{h} vs {w} Shot Map", (12, 8))
    pitch = Pitch(**PITCH_STYLE)
    pitch.draw(ax=ax)
    
    for _, shot in shots.iterrows():
        x, y = shot['location']
        team = shot['team']
        goal = shot['shot_outcome'] == 'Goal'
        color = PALETTE['home'] if team == h else PALETTE['away']
        
        if team == h:
            y = 80 - y  # Flip for home team
        
        size = np.sqrt(shot['shot_statsbomb_xg']) * 8
        alpha = 1 if goal else 0.5
        
        circle = plt.Circle((x, y), size, color=color, alpha=alpha)
        ax.add_patch(circle)
        
        if goal:
            plt.text(x, y + 2, shot['player'], 
                    fontsize=10, ha='center', 
                    fontproperties=custom_font.prop)
    
    fig_text(s=f'Total Shots: {len(shots)}',
             x=0.4, y=0.85, fontsize=14,
             fontproperties=custom_font.prop)
    add_credit(fig)
    plt.savefig(f'graphs/shots-{match_id}.png', dpi=300)
    st.image(f'graphs/shots-{match_id}.png')

def pass_network(events, team_name, match_id, color=PALETTE['primary']):
    try:
        passes = events['passes']
        team_passes = passes[passes['team'] == team_name]
        successful = team_passes[team_passes['pass_outcome'].isna()].copy()
        
        # Extract locations
        locs = successful['location'].apply(pd.Series)
        successful[['x', 'y']] = locs
        
        # Calculate metrics
        avg_loc = successful.groupby('player')[['x', 'y']].mean()
        pass_counts = successful['player'].value_counts()
        avg_loc['size'] = 300 + (1200 * pass_counts / pass_counts.max())
        
        # Connections
        connections = successful.groupby(['player', 'pass_recipient']).size().reset_index(name='count')
        connections = connections.merge(avg_loc, left_on='player', right_index=True)
        connections = connections.merge(avg_loc, left_on='pass_recipient', right_index=True, suffixes=['', '_end'])
        connections['width'] = 1 + (4 * connections['count'] / connections['count'].max())
        
        # Visualization
        fig, ax = setup_figure(f"{team_name} Pass Network", (14, 9))
        pitch = Pitch(**PITCH_STYLE)
        pitch.draw(ax=ax)
        
        # Heatmap
        bs_heatmap = pitch.bin_statistic(successful['x'], successful['y'], bins=(12, 8))
        pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues' if color == PALETTE['primary'] else 'Reds', alpha=0.25)
        
        # Connections
        pitch.lines(
            connections.x, connections.y,
            connections.x_end, connections.y_end,
            lw=connections.width,
            color=color,
            ax=ax,
            alpha=0.6
        )
        
        # Nodes
        pitch.scatter(
            avg_loc.x, avg_loc.y,
            s=avg_loc['size'],
            color=color,
            edgecolors='white',
            linewidth=1,
            ax=ax
        )
        
        # Labels
        for idx, row in avg_loc.iterrows():
            ax.text(row.x, row.y, idx.split()[-1],
                   ha='center', va='center',
                   fontsize=10, color='white',
                   fontproperties=custom_font.prop,
                   path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        
        add_credit(fig)
        plt.savefig(f'graphs/pass_network_{team_name}_{match_id}.png', dpi=300)
        st.image(f'graphs/pass_network_{team_name}_{match_id}.png')
        
    except Exception as e:
        st.error(f"Error creating pass network: {str(e)}")

# ========== STREAMLIT APP ==========
def main():
    st.set_page_config(
        page_title="Football Analytics",
        page_icon="⚽",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown(f"""
        <style>
            .stApp {{
                background-color: {PALETTE['background']};
            }}
            h1, h2, h3 {{
                color: {PALETTE['text']};
                font-family: 'Raleway';
            }}
            .stSelectbox > label {{
                font-weight: bold;
            }}
        </style>
    """, unsafe_allow_html=True)
    
    st.title('⚽ Football Match Analysis')
    
    # Competition selection
    col1, col2 = st.columns(2)
    with col1:
        competition = st.selectbox('Competition', com_dict.keys())
    with col2:
        season = st.selectbox('Season', season_dict.keys())
    
    # Match selection
    data = sb.matches(competition_id=com_dict[competition], season_id=season_dict[season])
    matches, matches_idx, matches_id_dict = matches_id(data)
    match = st.selectbox('Select Match', matches)
    
    if st.button('Analyze', type="primary"):
        with st.spinner('Loading match data...'):
            analyze_match(data, matches_idx[match], matches_id_dict[match])

def analyze_match(data, match_idx, match_id):
    home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(data, match_idx)
    home_lineup, away_lineup = lineups(home_team, away_team, sb.lineups(match_id=match_id))
    events = sb.events(match_id=match_id, split=True)
    
    # Match header
    st.header(f"{home_team} {home_score} - {away_score} {away_team}")
    st.caption(f"{stadium} | {comp_stats}")
    
    # Lineups
    with st.expander("Team Lineups", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.subheader(home_team)
            st.write(f"Manager: {home_manager}")
            st.write("**Starting XI:**")
            for player in home_lineup:
                st.write(f"- {player}")
        with col2:
            st.metric("Score", f"{home_score} - {away_score}")
        with col3:
            st.subheader(away_team)
            st.write(f"Manager: {away_manager}")
            st.write("**Starting XI:**")
            for player in away_lineup:
                st.write(f"- {player}")
    
    # Visualizations
    st.subheader("Match Analysis")
    tab1, tab2 = st.tabs(["Shots & Goals", "Passing Networks"])
    
    with tab1:
        st.subheader("Shot Map")
        shots_goal(events['shots'], home_team, away_team, match_id)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{home_team} Passing")
            pass_network(events, home_team, match_id, color=PALETTE['home'])
        with col2:
            st.subheader(f"{away_team} Passing")
            pass_network(events, away_team, match_id, color=PALETTE['away'])

if __name__ == "__main__":
    main()