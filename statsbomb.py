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

# Set page config must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Football Match Analysis")

# Theme configuration
DARK_MODE = st.sidebar.checkbox("Dark Mode", value=True)

if DARK_MODE:
    BG_COLOR = "#121212"
    PITCH_COLOR = "#1a1a1a"
    LINE_COLOR = "#efefef"
    TEXT_COLOR = "#ffffff"
    HOME_COLOR = "#ff6b6b"
    AWAY_COLOR = "#64b5f6"
    BUTTON_COLOR = "#4CAF50"
    BUTTON_HOVER = "#45a049"
    FIG_BG_COLOR = "#2d2d2d"
    SELECTBOX_BG = "#3d3d3d"
else:
    BG_COLOR = "#f8f9fa"
    PITCH_COLOR = "#f8f9fa"
    LINE_COLOR = "#000000"
    TEXT_COLOR = "#000000"
    HOME_COLOR = "#d62828"
    AWAY_COLOR = "#003049"
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

# Ensure graphs directory exists
os.makedirs('graphs', exist_ok=True)

# Define all functions first
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

def create_pitch_figure(title, figsize=(10, 6.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(FIG_BG_COLOR)
    ax.patch.set_facecolor(FIG_BG_COLOR)
    pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()
    fig_text(s=title, x=0.5, y=0.95, fontsize=FONT_SIZE_LG, 
            color=TEXT_COLOR, fontfamily=FONT_BOLD, ha='center')
    return fig, ax

# All visualization functions with updated styling
def shots_goal(shots, h, w, match_id):
    try:
        pitchLengthX = 120
        pitchWidthY = 80
        fig, ax = create_pitch_figure(f'🎯 {h} shots vs {w} shots')

        for i, shot in shots.iterrows():
            x = shot['location'][0]
            y = shot['location'][1]
            goal = shot['shot_outcome'] == 'Goal'
            team_name = shot['team']
            circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 5

            if team_name == h:
                plot_x = x
                plot_y = pitchWidthY - y
                color = HOME_COLOR
            else:
                plot_x = pitchLengthX - x
                plot_y = y
                color = AWAY_COLOR

            shotCircle = plt.Circle((plot_x, plot_y), circleSize, color=color, alpha=1 if goal else 0.2)
            ax.add_patch(shotCircle)

            if goal:
                text = ax.text(plot_x + 1, plot_y + 2, shot['player'], 
                              fontsize=FONT_SIZE_SM, color=TEXT_COLOR, 
                              ha='left', va='center', fontfamily=FONT)
                text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])

        total_shots = len(shots)
        fig_text(s=f'Total Shots: {total_shots}', x=0.4, y=0.85, 
                fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
        save_and_display(fig, f'shots-{match_id}.png')

    except Exception as e:
        st.error(f"Shots visualization error: {str(e)}")

def goals(shots, h, w, match_id):
    try:
        pitchLengthX = 120
        pitchWidthY = 80
        fig, ax = create_pitch_figure('🥅 Goals Analysis')

        for i, shot in shots[shots['shot_outcome'] == 'Goal'].iterrows():
            x = shot['location'][0]
            y = shot['location'][1]
            team_name = shot['team']
            circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 5

            if team_name == h:
                plot_x = x
                plot_y = pitchWidthY - y
                color = HOME_COLOR
            else:
                plot_x = pitchLengthX - x
                plot_y = y
                color = AWAY_COLOR

            shotCircle = plt.Circle((plot_x, plot_y), circleSize, color=color)
            ax.add_patch(shotCircle)

            # Text annotations with improved styling
            text_args = {
                'fontsize': FONT_SIZE_SM,
                'color': TEXT_COLOR,
                'fontfamily': FONT,
                'ha': 'center',
                'va': 'center'
            }
            text1 = ax.text(plot_x, plot_y - 3, shot['shot_body_part'].title(), **text_args)
            text2 = ax.text(plot_x, plot_y + 3, f"xG: {shot['shot_statsbomb_xg']:.2f}", **text_args)
            
            for text in [text1, text2]:
                text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])

        save_and_display(fig, f'goals-{match_id}.png')
    except Exception as e:
        st.error(f"Goals visualization error: {str(e)}")

def defensive_actions(events, h, w, match_id, action_type):
    try:
        title_map = {
            'foul_committeds': '🔴 Fouls Committed',
            'foul_wons': '🟢 Fouls Won',
            'interceptions': '🛡️ Interceptions',
            'dispossesseds': '💥 Dispossessions',
            'miscontrols': '⚠️ Miscontrols'
        }
        fig, ax = create_pitch_figure(title_map[action_type])
        
        if action_type in events:
            df = events[action_type]
            for team, color in [(h, HOME_COLOR), (w, AWAY_COLOR)]:
                team_data = df[df['possession_team'] == team]
                if not team_data.empty:
                    x = team_data['location'].apply(lambda loc: loc[0])
                    y = team_data['location'].apply(lambda loc: loc[1])
                    ax.scatter(x, y, s=100, color=color, alpha=0.7, label=team)
            
            legend = ax.legend(loc="upper left", framealpha=0.8)
            legend.get_frame().set_facecolor(FIG_BG_COLOR)
            fig_text(s=f'Total Actions: {len(df)}', x=0.15, y=0.85,
                    fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
            save_and_display(fig, f'{action_type}-{match_id}.png')
        else:
            st.warning(f"No {action_type.replace('_', ' ')} data available")
    except Exception as e:
        st.error(f"{action_type} visualization error: {str(e)}")

def carrys(events, h, w, match_id):
    try:
        fig, ax = create_pitch_figure('🏃‍♂️ Carries Analysis')
        
        if 'carrys' in events:
            for i, carry in events['carrys'].iterrows():
                start = carry['location']
                end = carry['carry_end_location']
                team = carry['possession_team']
                color = HOME_COLOR if team == h else AWAY_COLOR
                
                ax.scatter(start[0], start[1], s=50, color=color, alpha=0.7, marker='o')
                ax.scatter(end[0], end[1], s=50, color=color, alpha=0.7, marker='x')
                pitch.arrows(start[0], start[1], end[0], end[1], 
                            color=color, width=2, ax=ax, alpha=0.5)
            
            fig_text(s=f'Total Carries: {len(events["carrys"])}', x=0.15, y=0.85,
                    fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
            save_and_display(fig, f'carrys-{match_id}.png')
        else:
            st.warning("No carry data available")
    except Exception as e:
        st.error(f"Carry visualization error: {str(e)}")

def save_and_display(fig, filename):
    fig.text(0.02, 0.02, '@ahmedtarek26 / GitHub', 
            fontstyle='italic', fontsize=FONT_SIZE_SM-2, 
            color=TEXT_COLOR, fontfamily=FONT)
    plt.tight_layout()
    plt.savefig(f'graphs/{filename}', dpi=300, bbox_inches='tight')
    st.image(f'graphs/{filename}')

# Main app function with all visualizations
def main():
    st.title('⚽ Football Match Analysis')
    
    try:
        com = sb.competitions()
        com_dict = dict(zip(com['competition_name'], com['competition_id']))
        season_dict = dict(zip(com['season_name'], com['season_id']))
        
        competition = st.selectbox('Choose the competition', com_dict.keys())
        season = st.selectbox('Choose the season', season_dict.keys())
        
        data = sb.matches(competition_id=com_dict[competition], season_id=season_dict[season])
        matches_names, matches_idx, matches_id_dict = matches_id(data)
        match = st.selectbox('Select the match', matches_names)
        
        if st.button('Analyze Match', type='primary'):
            home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(
                data, matches_idx[match])
            
            match_id = matches_id_dict[match]
            events = sb.events(match_id=match_id, split=True)
            
            # Main visualization tabs
            tab1, tab2, tab3 = st.tabs(["⚔️ Attack Analysis", "🛡️ Defense Analysis", "📊 Match Stats"])
            
            with tab1:
                shots_goal(events.get('shots', pd.DataFrame()), home_team, away_team, match_id)
                goals(events.get('shots', pd.DataFrame()), home_team, away_team, match_id)
                
                st.subheader("🎯 Dribbles")
                dribbles(events, home_team, away_team, match_id)
                
                st.subheader("📶 Pass Networks")
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
                carrys(events, home_team, away_team, match_id)
            
            with tab3:
                st.subheader("📈 Match Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🏟️ Stadium", stadium)
                    st.metric("📅 Competition Stage", comp_stats)
                    st.metric("👨‍💼 Home Manager", home_manager)
                    st.metric("👨‍💼 Away Manager", away_manager)
                
                with col2:
                    stats_data = {
                        'Category': ['Shots', 'Passes', 'Fouls', 'Corners'],
                        home_team: [
                            len(events.get('shots', [])),
                            len(events.get('passes', [])),
                            len(events.get('foul_committeds', [])),
                            len([e for e in events.get('shots', []) if e.get('shot_type') == 'Corner'])
                        ],
                        away_team: [
                            len(events.get('shots', [])),
                            len(events.get('passes', [])),
                            len(events.get('foul_committeds', [])),
                            len([e for e in events.get('shots', []) if e.get('shot_type') == 'Corner'])
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Commented pass map functions (keep in code but not called)
"""
def home_team_passes(events, home_team, match_id):
    # Existing implementation here
    pass

def away_team_passes(events, away_team, match_id):
    # Existing implementation here
    pass
"""

if __name__ == "__main__":
    main()