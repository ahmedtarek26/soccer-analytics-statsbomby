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

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'match_data' not in st.session_state:
    st.session_state.match_data = None

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
    AWAY_COLOR = "#003049"
    FIG_BG_COLOR = "#ffffff"
    PLOTLY_TEMPLATE = "plotly_white"
    XG_BAR_COLOR = "#1e90ff"

# Font settings
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

def plot_player_actions(events, player_name, action_type, color, ax):
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

def plot_player_xg(shots, player_name, team_name, competition_info, home_team):
    """
    Creates an Opta-style xG visualization
    """
    try:
        player_shots = shots[shots['player'] == player_name]
        if player_shots.empty:
            st.warning(f"No shots data for {player_name}")
            return

        # Calculate metrics
        goals = player_shots[player_shots['shot_outcome'] == 'Goal'].shape[0]
        xg = player_shots['shot_statsbomb_xg'].sum().round(1)
        total_shots = player_shots.shape[0]
        xg_per_shot = (xg / total_shots).round(2) if total_shots > 0 else 0
        games = player_shots['match_id'].nunique()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 3), facecolor='white')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)

        # Main horizontal bars
        bar_height = 0.4
        goal_bar = ax.barh([''], [goals], height=bar_height, color=XG_BAR_COLOR, alpha=0.9)
        xg_bar = ax.barh([''], [xg], height=bar_height, color=XG_BAR_COLOR, alpha=0.4)

        # Add value labels inside bars
        ax.text(goals/2, 0, f"{goals}", 
               ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        ax.text(xg/2, 0, "", 
               ha='center', va='center', color=XG_BAR_COLOR, fontweight='bold')

        # Set x-axis limit
        max_value = max(goals, xg)
        ax.set_xlim(0, max_value * 1.3)

        # Remove spines and ticks
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add metrics on right side
        metrics_text = (
            f"{xg} xG\n"
            f"{total_shots} shots\n"
            f"{xg_per_shot} xG per shot\n"
            f"{games} games"
        )
        ax.text(max_value * 1.15, 0, metrics_text, 
               ha='left', va='center', linespacing=1.8)

        # Add xG scale indicator
        ax.text(0.02, -0.8, "Low xG", transform=ax.transAxes, 
               fontsize=9, color='#666666')
        ax.text(0.98, -0.8, "High xG", transform=ax.transAxes, 
               fontsize=9, color='#666666', ha='right')
        ax.plot([0.1, 0.9], [-0.5, -0.5], transform=ax.transAxes, 
               color='#666666', linewidth=2, clip_on=False)

        # Add title with team color
        player_color = HOME_COLOR if team_name == home_team else AWAY_COLOR
        fig.text(0.05, 0.9, player_name, 
                fontsize=16, fontweight='bold', ha='left', color=player_color)
        fig.text(0.05, 0.82, f"{team_name} | {competition_info}", 
                fontsize=11, color='#666666', ha='left')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error creating xG visualization: {str(e)}")

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
                text = ax.text(plot_x + 1, plot_y + 2, shot['player'].split()[-1], 
                              fontsize=FONT_SIZE_SM, color=TEXT_COLOR, 
                              ha='left', va='center', fontfamily=FONT)
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

            # Updated player name text with foot notation
            player_name = shot['player'].split()[-1]
            body_part = 'R' if shot['shot_body_part'] == 'Right Foot' else 'L' if shot['shot_body_part'] == 'Left Foot' else shot['shot_body_part'][0]
            info_text = f"{player_name}\n{body_part}"
            text = ax.text(plot_x, plot_y + 5, info_text, 
                          fontsize=FONT_SIZE_MD,
                          color=color,
                          ha='center', va='bottom', 
                          fontfamily=FONT_BOLD,
                          weight='bold')
            text.set_path_effects([path_effects.withStroke(linewidth=2, foreground="black")])

        save_and_display(fig, f'goals-{match_id}.png')
    except Exception as e:
        st.error(f"Goals visualization error: {str(e)}")

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

def defensive_actions(events, h, w, match_id, action_type):
    try:
        title_map = {
            'foul_committeds': 'Fouls Committed',
            'foul_wons': 'Fouls Won',
            'interceptions': 'Interceptions',
            'dispossesseds': 'Dispossessions',
            'miscontrols': 'Miscontrols'
        }
        fig, ax = create_pitch_figure(title_map[action_type])
        
        if action_type in events:
            df = events[action_type]
            for team, color in [(h, HOME_COLOR), (w, AWAY_COLOR)]:
                team_data = df[df['possession_team'] == team]
                if not team_data.empty:
                    x = team_data['location'].apply(lambda loc: loc[0])
                    y = team_data['location'].apply(lambda loc: loc[1])
                    ax.scatter(x, y, s=80, color=color, alpha=0.7, label=team)
            
            legend = ax.legend(loc='upper right', framealpha=0.8)
            legend.get_frame().set_facecolor(FIG_BG_COLOR)
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)
            
            total_actions = len(df)
            fig_text(s=f'Total: {total_actions}', x=0.15, y=0.85,
                    fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
            save_and_display(fig, f'{action_type}-{match_id}.png')
        else:
            st.warning(f"No {action_type.replace('_', ' ')} data available")
    except Exception as e:
        st.error(f"{action_type} visualization error: {str(e)}")

def player_actions_grid(events, team_name, players, action_type, color):
    try:
        if not players:
            st.warning(f"No players available for {team_name}")
            return
            
        if action_type not in events:
            st.warning(f"Action type '{action_type}' not found in event data")
            return
            
        if events[action_type].empty:
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
        pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues' if color == HOME_COLOR else 'Reds', 
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
        fig.text(0.02, 0.02, '@ahmedtarek26 / GitHub', 
                fontstyle='italic', fontsize=FONT_SIZE_SM-2, color='black')
        
        plt.tight_layout()
        plt.savefig(f'graphs/pass_network_{team_name}_{match_id}.png', 
                   dpi=300, bbox_inches='tight')
        st.image(f'graphs/pass_network_{team_name}_{match_id}.png')
        
    except Exception as e:
        st.error(f"Error creating pass network for {team_name}: {str(e)}")

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
        

        if st.button('Analyze Match') or st.session_state.analyzed:
            st.session_state.analyzed = True
            if st.session_state.match_data is None:
                # Load and process the data only if it's not already in session state
                home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(
                    data, matches_idx[match])
                
                match_id = matches_id_dict[match]
                lineup_data = sb.lineups(match_id=match_id)
                home_lineup, away_lineup = lineups(home_team, away_team, lineup_data)
                events = sb.events(match_id=match_id, split=True)
                
                # Store all the necessary data in session state
                st.session_state.match_data = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'stadium': stadium,
                    'home_manager': home_manager,
                    'away_manager': away_manager,
                    'comp_stats': comp_stats,
                    'match_id': match_id,
                    'home_lineup': home_lineup,
                    'away_lineup': away_lineup,
                    'events': events
                }
            else:
                # Retrieve data from session state
                match_data_dict = st.session_state.match_data  # Changed variable name here
                home_team = match_data_dict['home_team']
                away_team = match_data_dict['away_team']
                home_score = match_data_dict['home_score']
                away_score = match_data_dict['away_score']
                stadium = match_data_dict['stadium']
                home_manager = match_data_dict['home_manager']
                away_manager = match_data_dict['away_manager']
                comp_stats = match_data_dict['comp_stats']
                match_id = match_data_dict['match_id']
                home_lineup = match_data_dict['home_lineup']
                away_lineup = match_data_dict['away_lineup']
                events = match_data_dict['events']
            
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
                st.subheader("Individual Player Actions")
                
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

            # Add reset button
            if st.button('Analyze New Match'):
                st.session_state.analyzed = False
                st.session_state.match_data = None
                st.experimental_rerun()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()