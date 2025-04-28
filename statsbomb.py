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
    AWAY_COLOR = "#64b5f6"  # Darker blue for dark mode
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

# Font settings - using more modern font
FONT = 'Arial'
FONT_BOLD = 'Arial'
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
    """Extract match information and create mapping dictionaries"""
    match_id = []
    match_name = []
    match_index = []
    for i in range(len(data)):
        match_index.append(i)
        match_id.append(data['match_id'][i])
        match_name.append(f"{data['home_team'][i]} vs {data['away_team'][i]} {data['competition_stage'][i]}")
    return match_name, dict(zip(match_name, match_index)), dict(zip(match_name, match_id))

def match_data(data, match_index):
    """Extract match details for a specific match index"""
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
    """Get lineups for home and away teams"""
    return data[h]['player_name'].values, data[w]['player_name'].values

# --------------------------
# VISUALIZATION FUNCTIONS
# --------------------------

def create_pitch_figure(title, figsize=(10, 6.5)):
    """Create a standard pitch figure with consistent styling"""
    fig, ax = plt.subplots(figsize=figsize, facecolor=FIG_BG_COLOR)
    pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()
    fig_text(s=title, x=0.5, y=0.95, fontsize=FONT_SIZE_LG, 
            color=TEXT_COLOR, fontfamily=FONT_BOLD, ha='center')
    return fig, ax

def save_and_display(fig, filename):
    """Save figure to file and display in Streamlit"""
    fig.text(0.02, 0.02, '@ahmedtarek26 / GitHub', 
            fontstyle='italic', fontsize=FONT_SIZE_SM-2, 
            color=TEXT_COLOR, fontfamily=FONT)
    plt.tight_layout()
    plt.savefig(f'graphs/{filename}', dpi=300, bbox_inches='tight', facecolor=FIG_BG_COLOR)
    st.image(f'graphs/{filename}')

def plot_player_actions(events, player_name, action_type, color, ax):
    """Plot individual player actions on pitch"""
    player_events = events[events['player'] == player_name]
    if len(player_events) == 0:
        return
    
    if action_type == 'carrys':
        # Add legend
        start_marker = ax.scatter([], [], s=50, color=color, alpha=0.7, marker='o', label='Start')
        end_marker = ax.scatter([], [], s=50, color=color, alpha=0.7, marker='x', label='End')
        ax.legend(handles=[start_marker, end_marker], 
                 facecolor=FIG_BG_COLOR, 
                 edgecolor=FIG_BG_COLOR)
        
        # Plot arrows instead of lines
        for _, event in player_events.iterrows():
            x_start, y_start = event['location']
            x_end, y_end = event['carry_end_location']
            dx, dy = x_end - x_start, y_end - y_start
            ax.arrow(x_start, y_start, dx, dy, 
                    head_width=1.5, head_length=2, 
                    fc=color, ec=color, alpha=0.5)
    else:
        x = player_events['location'].apply(lambda loc: loc[0])
        y = player_events['location'].apply(lambda loc: loc[1])
        ax.scatter(x, y, s=80, color=color, alpha=0.7)

def shots_goal(shots, h, w, match_id):
    """Visualize shots and goals"""
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
                # Use last name only like in pass network
                player_name = shot['player'].split()[-1]
                text = ax.text(plot_x + 1, plot_y + 2, player_name, 
                              fontsize=FONT_SIZE_SM, color=TEXT_COLOR, 
                              ha='left', va='center', fontfamily=FONT,
                              fontweight='bold')
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
    """Visualize goals with details"""
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

            shotCircle = plt.Circle((plot_x, plot_y), circleSize, color=color, alpha=0.8)
            ax.add_patch(shotCircle)

            # Use last name only and improved formatting
            player_name = shot['player'].split()[-1]
            info_text = f"{player_name}\n{shot['shot_body_part'].title()}\nxG: {shot['shot_statsbomb_xg']:.2f}"
            text = ax.text(plot_x, plot_y + 5, info_text, 
                          fontsize=FONT_SIZE_SM, color=TEXT_COLOR,
                          ha='center', va='bottom', fontfamily=FONT,
                          fontweight='bold')
            text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])

        save_and_display(fig, f'goals-{match_id}.png')
    except Exception as e:
        st.error(f"Goals visualization error: {str(e)}")

def player_actions_grid(events, team_name, players, action_type, color):
    """Grid of player actions for a team"""
    try:
        if not players:
            st.warning(f"No players available for {team_name}")
            return
            
        if action_type not in events or events[action_type].empty:
            st.warning(f"No {action_type} data available for {team_name}")
            return
        
        n_players = len(players)
        n_cols = min(3, n_players)
        n_rows = (n_players + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.set_facecolor(FIG_BG_COLOR)
        
        if n_players == 1:
            axes = np.array([axes])
        
        for i, (player, ax) in enumerate(zip(players, axes.flatten())):
            ax.set_facecolor(FIG_BG_COLOR)
            pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
            pitch.draw(ax=ax)
            ax.invert_yaxis()
            
            plot_player_actions(events[action_type], player, action_type, color, ax)
            
            ax.set_title(player, color=TEXT_COLOR, fontsize=FONT_SIZE_MD)
        
        for j in range(i+1, n_rows*n_cols):
            axes.flatten()[j].axis('off')
        
        fig.suptitle(f"{team_name} {action_type.title()} by Player", 
                    color=TEXT_COLOR, fontsize=FONT_SIZE_LG, y=0.98)
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating player actions grid: {str(e)}")

# --------------------------
# MAIN APP FUNCTION
# --------------------------

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