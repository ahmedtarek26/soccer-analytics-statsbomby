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
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from matplotlib.patches import FancyArrowPatch

# Configuration
st.set_page_config(layout="wide", page_title="Football Match Analysis")
DEBUG_MODE = False  # Set to True for development with sampled data

# Theme configuration
DARK_MODE = st.sidebar.checkbox("Dark Mode", value=False)
if DARK_MODE:
    BG_COLOR, PITCH_COLOR = "#121212", "#1a1a1a"
    HOME_COLOR, AWAY_COLOR = "#ff6b6b", "#64b5f6"
else:
    BG_COLOR, PITCH_COLOR = "#ffffff", "#f8f9fa"
    HOME_COLOR, AWAY_COLOR = "#d62828", "#7fbfff"

# Font settings
FONT = 'DejaVu Sans'
FONT_BOLD = 'DejaVu Sans'

# Caching decorators
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_competitions():
    return sb.competitions()

@st.cache_data(ttl=3600)
def get_cached_matches(competition_id, season_id):
    return sb.matches(competition_id=competition_id, season_id=season_id)

@st.cache_data(ttl=3600)
def get_cached_events(match_id):
    events = sb.events(match_id=match_id, split=True)
    return {k: v.sample(frac=0.3) if DEBUG_MODE else v for k, v in events.items()}

# Optimized data processing
def process_player_events_parallel(events, players):
    def process_single_player(player):
        player_data = {}
        for action_type in ['shots', 'passes', 'dribbles']:
            if action_type in events:
                player_data[action_type] = events[action_type][events[action_type]['player'] == player]
        return player, player_data
    
    with ThreadPoolExecutor() as executor:
        return dict(executor.map(process_single_player, players))

# Core visualization functions
def create_pitch_figure(title):
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=BG_COLOR)
    pitch = Pitch(pitch_type='statsbomb', line_color='#000000', pitch_color=PITCH_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()
    return fig, ax

def plot_optimized_heatmap(events, player_name):
    fig, ax = create_pitch_figure(f"{player_name} Heatmap")
    locations = [e['location'] for e in events if 'location' in e]
    if locations:
        x, y = zip(*locations)
        hb = ax.hexbin(x, y, gridsize=15, cmap='Reds', alpha=0.7)
        plt.colorbar(hb, ax=ax)
    st.pyplot(fig)

# Main app function
def main():
    st.title('⚽ Football Match Analysis')
    
    try:
        # Initialize session state
        if 'precomputed' not in st.session_state:
            st.session_state.precomputed = {}
            st.session_state.analyzed = False
        
        # Load competitions with caching
        com = get_cached_competitions()
        com_dict = dict(zip(com['competition_name'], com['competition_id']))
        
        # Competition selection
        competition = st.selectbox('Choose the competition', com_dict.keys())
        
        # Filter seasons for selected competition
        available_seasons = com[com['competition_id'] == com_dict[competition]]
        season_dict = dict(zip(available_seasons['season_name'], available_seasons['season_id']))
        season = st.selectbox('Choose the season', season_dict.keys())
        
        # Load matches with caching
        data = get_cached_matches(com_dict[competition], season_dict[season])
        matches_names, matches_idx, matches_id_dict = matches_id(data)
        match = st.selectbox('Select the match', matches_names)
        
        if st.button('Analyze Match') or st.session_state.analyzed:
            with st.spinner('Optimizing data...'):
                st.session_state.analyzed = True
                match_id = matches_id_dict[match]
                
                # Parallel loading of data
                with ThreadPoolExecutor() as executor:
                    lineup_future = executor.submit(sb.lineups, match_id=match_id)
                    events_future = executor.submit(get_cached_events, match_id)
                    
                    lineup_data = lineup_future.result()
                    events = events_future.result()
                
                # Process lineups
                home_team, away_team = data['home_team'][matches_idx[match]], data['away_team'][matches_idx[match]]
                home_lineup, away_lineup = lineups(home_team, away_team, lineup_data)
                
                # Precompute player stats in parallel
                all_players = list(home_lineup) + list(away_lineup)
                if 'player_stats' not in st.session_state.precomputed:
                    st.session_state.precomputed['player_stats'] = process_player_events_parallel(events, all_players)
                
                # Display match header and lineups
                display_match_header(data, matches_idx[match], home_lineup, away_lineup)
                
                # Visualizations in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚽ Attack", "🛡️ Defense", "👤 Player Actions", "📊 Stats", "👤 Player Stats"])
                
                with tab1:
                    display_attack_analysis(events, home_team, away_team, match_id)
                
                with tab2:
                    display_defense_analysis(events, home_team, away_team, match_id)
                
                with tab3:
                    display_player_actions(events, home_team, away_team, home_lineup, away_lineup)
                
                with tab4:
                    display_match_stats(events, home_team, away_team)
                
                with tab5:
                    display_player_stats_tab(events, home_team, away_team, home_lineup, away_lineup)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Helper functions would be defined here following the same optimization patterns
# ...

if __name__ == "__main__":
    main()