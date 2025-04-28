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
from matplotlib.patches import FancyArrowPatch

# Set page config must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Football Match Analysis")

# Theme configuration - Light mode as default
DARK_MODE = st.sidebar.checkbox("Dark Mode", value=False)

if DARK_MODE:
    BG_COLOR = "#121212"
    PITCH_COLOR = "#1a1a1a"
    LINE_COLOR = "#efefef"
    TEXT_COLOR = "#ffffff"
    HOME_COLOR = "#ff6b6b"
    AWAY_COLOR = "#64b5f6"
    FIG_BG_COLOR = "#121212"
    PLOTLY_TEMPLATE = "plotly_dark"
    XG_BAR_COLOR = "#64b5f6"
else:
    BG_COLOR = "#ffffff"
    PITCH_COLOR = "#f8f9fa"
    LINE_COLOR = "#000000"
    TEXT_COLOR = "#000000"
    HOME_COLOR = "#d62828"
    AWAY_COLOR = "#7fbfff"  # Lighter blue for away team
    FIG_BG_COLOR = "#ffffff"
    PLOTLY_TEMPLATE = "plotly_white"
    XG_BAR_COLOR = "#1e90ff"

# Font settings - using smoother font
FONT = 'DejaVu Sans'
FONT_BOLD = 'DejaVu Sans'
FONT_SIZE_SM = 10
FONT_SIZE_MD = 12
FONT_SIZE_LG = 14
FONT_SIZE_XL = 16

# Ensure graphs directory exists
os.makedirs('graphs', exist_ok=True)

# --------------------------
# DATA PROCESSING FUNCTIONS
# --------------------------

def matches_id(data):
    match_id = []
    match_name = []
    match_index = []
    for i in range(len(data)):
        match_index.append(i)
        match_id.append(data['match_id'][i])
        match_name.append(f"{data['home_team'][i]} vs {data['away_team'][i]} {data['competition_stage'][i]}")
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

def lineups(h, w, data):
    return data[h]['player_name'].values, data[w]['player_name'].values

# --------------------------
# VISUALIZATION FUNCTIONS
# --------------------------

def create_pitch_figure(title, figsize=(10, 6.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=FIG_BG_COLOR)
    pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()
    fig_text(s=title, x=0.5, y=0.95, fontsize=FONT_SIZE_LG, 
            color=TEXT_COLOR, fontfamily=FONT_BOLD, ha='center')
    return fig, ax

def save_and_display(fig, filename):
    fig.text(0.02, 0.02, '@ahmedtarek26 / GitHub', 
            fontstyle='italic', fontsize=FONT_SIZE_SM-2, 
            color=TEXT_COLOR, fontfamily=FONT)
    plt.tight_layout()
    plt.savefig(f'graphs/{filename}', dpi=300, bbox_inches='tight', facecolor=FIG_BG_COLOR)
    st.image(f'graphs/{filename}')

def dribbles(events, h, w, match_id):
    try:
        fig, ax = create_pitch_figure('Dribbles')
        
        if 'dribbles' in events:
            x_h = []
            y_h = []
            x_w = []
            y_w = []
            
            for i, dribble in events['dribbles'].iterrows():
                if events['dribbles']['possession_team'][i] == h:
                    x_h.append(dribble['location'][0])
                    y_h.append(dribble['location'][1])
                elif events['dribbles']['possession_team'][i] == w:
                    x_w.append(dribble['location'][0])
                    y_w.append(dribble['location'][1])
            
            ax.scatter(x_h, y_h, s=80, c=HOME_COLOR, alpha=.7, label=h)
            ax.scatter(x_w, y_w, s=80, c=AWAY_COLOR, alpha=.7, label=w)
            
            legend = ax.legend(loc='upper right', framealpha=0.8)
            legend.get_frame().set_facecolor(FIG_BG_COLOR)
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)
            
            total_dribbles = len(events['dribbles'])
            fig_text(s=f'Total Dribbles: {total_dribbles}', x=0.15, y=0.85,
                    fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
            save_and_display(fig, f'dribbles-{match_id}.png')
        else:
            st.warning("No dribbles data available")
    except Exception as e:
        st.error(f"Dribbles visualization error: {str(e)}")

def shots_goal(shots, h, w, match_id):
    try:
        pitchLengthX = 120
        pitchWidthY = 80
        fig, ax = create_pitch_figure(f'{h} shots vs {w} shots')

        for i, shot in shots.iterrows():
            x = shot['location'][0]
            y = shot['location'][1]
            goal = shot['shot_outcome'] == 'Goal'
            team_name = shot['team']
            circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 8  # Increased size

            if team_name == h:
                plot_x = x
                plot_y = pitchWidthY - y
                color = HOME_COLOR
            else:
                plot_x = pitchLengthX - x
                plot_y = y
                color = AWAY_COLOR

            # Apply opacity for non-goals
            alpha = 0.8 if goal else 0.3
            shotCircle = plt.Circle((plot_x, plot_y), circleSize, color=color, alpha=alpha)
            ax.add_patch(shotCircle)

            if goal:
                # Use last name only with smoother font
                player_name = shot['player'].split()[-1]
                text = ax.text(plot_x + 1, plot_y + 2, player_name, 
                              fontsize=FONT_SIZE_SM, color=TEXT_COLOR, 
                              ha='left', va='center', fontfamily=FONT,
                              fontweight='normal')  # Changed to normal weight
                text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])

        total_shots = len(shots)
        fig_text(s=f'Total Shots: {total_shots}', x=0.4, y=0.85, 
                fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
        
        home_patch = plt.Circle((0,0), 1, color=HOME_COLOR, label=h)
        away_patch = plt.Circle((0,0), 1, color=AWAY_COLOR, label=w)
        legend = ax.legend(handles=[home_patch, away_patch], loc='upper right',
                         facecolor=FIG_BG_COLOR, edgecolor=FIG_BG_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
        
        save_and_display(fig, f'shots-{match_id}.png')

    except Exception as e:
        st.error(f"Shots visualization error: {str(e)}")

def goals(shots, h, w, match_id):
    try:
        pitchLengthX = 120
        pitchWidthY = 80
        fig, ax = create_pitch_figure('Goals Analysis')

        goals_df = shots[shots['shot_outcome'] == 'Goal']
        if goals_df.empty:
            st.warning("No goals data available")
            return

        for i, shot in goals_df.iterrows():
            x = shot['location'][0]
            y = shot['location'][1]
            team_name = shot['team']
            circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 8  # Increased size

            if team_name == h:
                plot_x = x
                plot_y = pitchWidthY - y
                color = HOME_COLOR
            else:
                plot_x = pitchLengthX - x
                plot_y = y
                color = AWAY_COLOR

            # Add player name inside the bubble with opacity
            player_name = shot['player'].split()[-1]
            ax.text(plot_x, plot_y, player_name, 
                   fontsize=FONT_SIZE_SM-1, color='white',
                   ha='center', va='center', fontfamily=FONT,
                   fontweight='normal')  # Smoother font weight
            
            shotCircle = plt.Circle((plot_x, plot_y), circleSize, color=color, alpha=0.7)
            ax.add_patch(shotCircle)

            # Additional info below the bubble
            info_text = f"{shot['shot_body_part'].title()}\nxG: {shot['shot_statsbomb_xg']:.2f}"
            text = ax.text(plot_x, plot_y + circleSize + 3, info_text, 
                          fontsize=FONT_SIZE_SM-1, color=TEXT_COLOR,
                          ha='center', va='bottom', fontfamily=FONT)
            text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])

        save_and_display(fig, f'goals-{match_id}.png')
    except Exception as e:
        st.error(f"Goals visualization error: {str(e)}")

# [Rest of the functions remain the same...]

def main():
    st.title('⚽ Football Match Analysis')
    
    try:
        # Load competitions data
        com = sb.competitions()
        com_dict = dict(zip(com['competition_name'], com['competition_id']))
        season_dict = dict(zip(com['season_name'], com['season_id']))
        
        competition = st.selectbox('Choose the competition', com_dict.keys())
        season = st.selectbox('Choose the season', season_dict.keys())
        
        data = sb.matches(competition_id=com_dict[competition], season_id=season_dict[season])
        matches_names, matches_idx, matches_id_dict = matches_id(data)
        match = st.selectbox('Select the match', matches_names)
        
        if st.button('Analyze Match'):
            home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(
                data, matches_idx[match])
            
            match_id = matches_id_dict[match]
            lineup_data = sb.lineups(match_id=match_id)
            home_lineup, away_lineup = lineups(home_team, away_team, lineup_data)
            events = sb.events(match_id=match_id, split=True)
            
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
                    st.subheader(f'{home_team} Lineup')
                    st.markdown(f"**Manager:** {home_manager}")
                    for player in home_lineup:
                        st.markdown(f"- {player}")
            
            with col3:
                with st.container():
                    st.subheader(f'{away_team} Lineup')
                    st.markdown(f"**Manager:** {away_manager}")
                    for player in away_lineup:
                        st.markdown(f"- {player}")

            # Visualizations
            st.markdown("---")
            st.header("Match Visualizations")
            
            tab1, tab2, tab3, tab4 = st.tabs(["⚽ Attack", "🛡️ Defense", "👤 Player Actions", "📊 Stats"])
            
            with tab1:
                shots_goal(events.get('shots', pd.DataFrame()), home_team, away_team, match_id)
                goals(events.get('shots', pd.DataFrame()), home_team, away_team, match_id)
                
                if 'dribbles' in events:
                    st.subheader("Dribbles")
                    dribbles(events, home_team, away_team, match_id)
                
                st.subheader("Pass Networks")
                col1, col2 = st.columns(2)
                with col1:
                    pass_network(events, home_team, match_id, HOME_COLOR)
                with col2:
                    pass_network(events, away_team, match_id, AWAY_COLOR)
            
            with tab2:
                defensive_actions(events, home_team, away_team, match_id, 'foul_committeds')
                defensive_actions(events, home_team, away_team, match_id, 'foul_wons')
                defensive_actions(events, home_team, away_team, match_id, 'interceptions')
                defensive_actions(events, home_team, away_team, match_id, 'dispossesseds')
                defensive_actions(events, home_team, away_team, match_id, 'miscontrols')
            
            with tab3:
                st.subheader("Player Actions")
                
                action_type = st.selectbox("Select action type", 
                                         ['carrys', 'passes', 'shots', 'dribbles'])
                
                st.subheader(f"{home_team} Players")
                player_actions_grid(events, home_team, list(home_lineup), action_type, HOME_COLOR)
                
                st.subheader(f"{away_team} Players")
                player_actions_grid(events, away_team, list(away_lineup), action_type, AWAY_COLOR)
            
            with tab4:
                st.subheader("Match Statistics")
                
                if 'shots' in events:
                    shots_df = events['shots'].groupby(['player', 'team']).size().reset_index(name='count')
                    st.plotly_chart(px.bar(shots_df, 
                                        x='player', y='count', 
                                        color='team',
                                        color_discrete_map={home_team: HOME_COLOR, away_team: AWAY_COLOR},
                                        title="Shots by Player", 
                                        template=PLOTLY_TEMPLATE))
                
                if 'passes' in events:
                    passes_df = events['passes'].groupby(['player', 'team']).size().reset_index(name='count')
                    st.plotly_chart(px.bar(passes_df, 
                                        x='player', y='count', 
                                        color='team',
                                        color_discrete_map={home_team: HOME_COLOR, away_team: AWAY_COLOR},
                                        title="Passes by Player", 
                                        template=PLOTLY_TEMPLATE))
                
                if 'foul_committeds' in events:
                    fouls_df = events['foul_committeds'].groupby(['player', 'team']).size().reset_index(name='count')
                    st.plotly_chart(px.bar(fouls_df, 
                                        x='player', y='count', 
                                        color='team',
                                        color_discrete_map={home_team: HOME_COLOR, away_team: AWAY_COLOR},
                                        title="Fouls Committed", 
                                        template=PLOTLY_TEMPLATE))

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()