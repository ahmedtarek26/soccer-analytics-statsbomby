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
from matplotlib.lines import Line2D
import os
import matplotlib.patheffects as path_effects
from sklearn.cluster import KMeans
import random

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
    try:
        player_shots = shots[shots['player'] == player_name]
        if player_shots.empty:
            st.warning(f"No shots data for {player_name}")
            return

        goals = player_shots[player_shots['shot_outcome'] == 'Goal'].shape[0]
        xg = player_shots['shot_statsbomb_xg'].sum().round(1)
        total_shots = player_shots.shape[0]
        xg_per_shot = (xg / total_shots).round(2) if total_shots > 0 else 0
        games = player_shots['match_id'].nunique()

        fig, ax = plt.subplots(figsize=(8, 3), facecolor='white')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)

        bar_height = 0.4
        goal_bar = ax.barh([''], [goals], height=bar_height, color=XG_BAR_COLOR, alpha=0.9)
        xg_bar = ax.barh([''], [xg], height=bar_height, color=XG_BAR_COLOR, alpha=0.4)

        ax.text(goals/2, 0, f"{goals}", 
               ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        ax.text(xg/2, 0, "", 
               ha='center', va='center', color=XG_BAR_COLOR, fontweight='bold')

        max_value = max(goals, xg)
        ax.set_xlim(0, max_value * 1.3)

        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])

        metrics_text = (
            f"{xg} xG\n"
            f"{total_shots} shots\n"
            f"{xg_per_shot} xG per shot\n"
            f"{games} games"
        )
        ax.text(max_value * 1.15, 0, metrics_text, 
               ha='left', va='center', linespacing=1.8)

        ax.text(0.02, -0.8, "Low xG", transform=ax.transAxes, 
               fontsize=9, color='#666666')
        ax.text(0.98, -0.8, "High xG", transform=ax.transAxes, 
               fontsize=9, color='#666666', ha='right')
        ax.plot([0.1, 0.9], [-0.5, -0.5], transform=ax.transAxes, 
               color='#666666', linewidth=2, clip_on=False)

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

        total_shots = len(shots)
        fig_text(s=f'Total Shots: {total_shots}', x=0.15, y=0.85, 
                fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
        
        home_patch = plt.Circle((0,0), 1, color=HOME_COLOR, label=h)
        away_patch = plt.Circle((0,0), 1, color=AWAY_COLOR, label=w)
        legend = ax.legend(handles=[home_patch, away_patch], loc='upper right',
                         facecolor=FIG_BG_COLOR, edgecolor=FIG_BG_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
        
        save_and_display(fig, f'shots-{match_id}.png')
        
        # Analytical Description for Shots
        home_shots = shots[shots['team'] == h].shape[0]
        away_shots = shots[shots['team'] == w].shape[0]
        home_goals = shots[(shots['team'] == h) & (shots['shot_outcome'] == 'Goal')].shape[0]
        away_goals = shots[(shots['team'] == w) & (shots['shot_outcome'] == 'Goal')].shape[0]
        top_shooter = shots['player'].value_counts().idxmax() if not shots.empty else "N/A"
        
        description = f"{h} took {home_shots} shots, scoring {home_goals} goals, while {w} had {away_shots} shots and scored {away_goals}."
        if home_shots > away_shots:
            description += f" {h} was more aggressive in attack."
        elif away_shots > home_shots:
            description += f" {w} created more shooting opportunities."
        else:
            description += " Both teams had an equal number of shots."
        if top_shooter != "N/A":
            description += f" {top_shooter} was the most active shooter."
        
        st.markdown(
            f"""
            <div style="background-color:{'#2d2d2d' if DARK_MODE else '#f8f9fa'};
                        padding:1rem;border-radius:8px;margin-top:1rem;color:{TEXT_COLOR};">
                <strong>Shots Analysis:</strong> {description}
            </div>
            """, unsafe_allow_html=True
        )
        
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

        cmap = plt.cm.Reds
        
        legend_entries = {}
        
        for i, shot in goals_df.iterrows():
            x = shot['location'][0]
            y = shot['location'][1]
            team_name = shot['team']
            circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 8
            
            if team_name == h:
                plot_x = x
                plot_y = pitchWidthY - y
                team_color = HOME_COLOR
            else:
                plot_x = pitchLengthX - x
                plot_y = y
                team_color = AWAY_COLOR
            
            time_norm = shot['minute'] / (goals_df['minute'].max() + 1)
            color = cmap(time_norm)
            
            shotCircle = plt.Circle((plot_x, plot_y), circleSize, color=color, alpha=0.8)
            ax.add_patch(shotCircle)
            
            player_name = shot['player'].split()[-1]
            text_color = 'black' if time_norm < 0.5 else 'white'
            
            if shot['shot_statsbomb_xg'] < 0.1:
                display_text = player_name[0]
                fontsize = FONT_SIZE_SM-2
            else:
                display_text = player_name
                fontsize = FONT_SIZE_SM-1
            
            ax.text(plot_x, plot_y, display_text, 
                   fontsize=fontsize, color=text_color,
                   ha='center', va='center', fontfamily=FONT,
                   fontweight='bold')
            
            legend_entries[player_name[0]] = player_name
            
            foot = shot['shot_body_part']
            if foot == 'Left Foot':
                foot_text = 'L'
                offset_x = -circleSize - 2
                offset_y = 0
            elif foot == 'Right Foot':
                foot_text = 'R'
                offset_x = circleSize + 2
                offset_y = 0
            else:
                foot_text = 'O'
                offset_x = 0
                offset_y = -circleSize - 2
            
            ax.text(plot_x + offset_x, plot_y + offset_y, foot_text,
                   fontsize=FONT_SIZE_SM, color=TEXT_COLOR,
                   ha='center', va='center', fontfamily=FONT_BOLD,
                   bbox=dict(facecolor=FIG_BG_COLOR, edgecolor=TEXT_COLOR, 
                            boxstyle='round,pad=0.2', alpha=0.7))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=goals_df['minute'].max()))
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=20)
        cbar.set_label('Minute', color=TEXT_COLOR, fontsize=FONT_SIZE_SM-2, labelpad=2)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=FONT_SIZE_SM-3)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Left Foot (L)',
                  markerfacecolor='gray', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Right Foot (R)',
                  markerfacecolor='gray', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Other (O)',
                  markerfacecolor='gray', markersize=6)
        ]
        
        player_legend = "\n".join([f"{k}: {v}" for k,v in sorted(legend_entries.items())])
        legend_elements.append(Line2D([0], [0], marker='', color='w', 
                                   label=f"Players:\n{player_legend}",
                                   markersize=0))
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 facecolor=FIG_BG_COLOR, edgecolor=FIG_BG_COLOR,
                 fontsize=FONT_SIZE_SM-2, handlelength=1.5)
        
        save_and_display(fig, f'goals-{match_id}.png')
        
        # Analytical Description for Goals
        home_goals = goals_df[goals_df['team'] == h].shape[0]
        away_goals = goals_df[goals_df['team'] == w].shape[0]
        top_scorer = goals_df['player'].value_counts().idxmax() if not goals_df.empty else "N/A"
        
        description = f"{h} scored {home_goals} goals, while {w} scored {away_goals}."
        if home_goals > away_goals:
            description += f" {h} was more clinical in front of goal."
        elif away_goals > home_goals:
            description += f" {w} capitalized better on their chances."
        else:
            description += " Both teams were equally effective in scoring."
        if top_scorer != "N/A":
            description += f" {top_scorer} was the key goal scorer."
        
        st.markdown(
            f"""
            <div style="background-color:{'#2d2d2d' if DARK_MODE else '#f8f9fa'};
                        padding:1rem;border-radius:8px;margin-top:1rem;color:{TEXT_COLOR};">
                <strong>Goals Analysis:</strong> {description}
            </div>
            """, unsafe_allow_html=True
        )
        
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
            
            # Analytical Description for Dribbles
            home_dribbles = len(x_h)
            away_dribbles = len(x_w)
            description = f"{h} attempted {home_dribbles} dribbles, while {w} had {away_dribbles}."
            if home_dribbles > away_dribbles:
                description += f" {h} was more aggressive in taking on defenders."
            elif away_dribbles > home_dribbles:
                description += f" {w} showed greater intent to beat players."
            else:
                description += " Both teams attempted an equal number of dribbles."
            
            st.markdown(
                f"""
                <div style="background-color:{'#2d2d2d' if DARK_MODE else '#f8f9fa'};
                            padding:1rem;border-radius:8px;margin-top:1rem;color:{TEXT_COLOR};">
                    <strong>Dribbles Analysis:</strong> {description}
                </div>
                """, unsafe_allow_html=True
            )
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
            
            # Analytical Description for Defensive Actions
            home_actions = df[df['possession_team'] == h].shape[0]
            away_actions = df[df['possession_team'] == w].shape[0]
            description = f"{h} committed {home_actions} {action_type.replace('_', ' ')}, while {w} had {away_actions}."
            if home_actions > away_actions:
                description += f" {h} showed greater defensive intensity."
            elif away_actions > home_actions:
                description += f" {w} was more active in {action_type.replace('_', ' ')}."
            else:
                description += f" Both teams had an equal number of {action_type.replace('_', ' ')}."
            
            st.markdown(
                f"""
                <div style="background-color:{'#2d2d2d' if DARK_MODE else '#f8f9fa'};
                            padding:1rem;border-radius:8px;margin-top:1rem;color:{TEXT_COLOR};">
                    <strong>{title_map[action_type]} Analysis:</strong> {description}
                </div>
                """, unsafe_allow_html=True
            )
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
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Analytical Description for Player Actions
        top_player = events[action_type][events[action_type]['player'].isin(players)]['player'].value_counts().idxmax() if not events[action_type].empty else "N/A"
        description = f"{team_name}'s players were active in {action_type}, with {top_player} leading the way."
        
        st.markdown(
            f"""
            <div style="background-color:{'#2d2d2d' if DARK_MODE else '#f8f9fa'};
                        padding:1rem;border-radius:8px;margin-top:1rem;color:{TEXT_COLOR};">
                <strong>Player Actions Analysis ({action_type}):</strong> {description}
            </div>
            """, unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Error creating player actions grid: {str(e)}")

# --------------------------
# PASS NETWORK ENHANCEMENTS
# --------------------------

def infer_formation(avg_locations):
    """
    Infer team formation using k-means clustering on player y-positions.
    
    Args:
        avg_locations (pd.DataFrame): DataFrame with columns 'x', 'y' for player positions.
    
    Returns:
        str: Inferred formation (e.g., '4-3-3') or 'Unknown'.
    """
    try:
        if len(avg_locations) < 3:
            return 'Unknown'
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        avg_locations['cluster'] = kmeans.fit_predict(avg_locations[['y']])
        
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_clusters = np.argsort(cluster_centers)
        
        def_count = sum(avg_locations['cluster'] == sorted_clusters[0])
        mid_count = sum(avg_locations['cluster'] == sorted_clusters[1])
        att_count = sum(avg_locations['cluster'] == sorted_clusters[2])
        
        formation_map = {
            (4, 3, 3): '4-3-3',
            (4, 2, 4): '4-2-3-1',
            (4, 4, 2): '4-4-2',
            (5, 3, 2): '5-3-2',
            (3, 4, 3): '3-4-3'
        }
        
        formation_key = (def_count, mid_count, att_count)
        return formation_map.get(formation_key, f"{def_count}-{mid_count}-{att_count}")
    except Exception as e:
        return 'Unknown'

def analyze_pass_network(team_passes, successful_passes, pass_connections, avg_locations):
    """
    Analyze pass network to extract tactical insights.
    
    Args:
        team_passes (pd.DataFrame): All passes for the team.
        successful_passes (pd.DataFrame): Successful passes for the team.
        pass_connections (pd.DataFrame): Passing connections between players.
        avg_locations (pd.DataFrame): Average player locations.
    
    Returns:
        dict: Statistics and insights for description.
    """
    try:
        stats = {}
        
        total_passes = len(team_passes)
        successful_count = len(successful_passes)
        pass_accuracy = (successful_count / total_passes * 100) if total_passes > 0 else 0
        stats['total_passes'] = total_passes
        stats['pass_accuracy'] = round(pass_accuracy, 1)
        
        stats['possession_style'] = (
            'possession-based' if pass_accuracy > 80 else 
            'direct' if pass_accuracy < 70 else 'balanced'
        )
        
        stats['formation'] = infer_formation(avg_locations)
        
        wing_backs = avg_locations[
            ((avg_locations['x'] < 20) | (avg_locations['x'] > 100)) & 
            (avg_locations['y'] > 50)
        ]
        wing_back_names = []
        wing_back_passes = 0
        if not wing_backs.empty:
            for wb in wing_backs.index:
                wb_passes = pass_connections[
                    (pass_connections['role'] == wb) & 
                    (pass_connections['y_end'] > 65)
                ]['count'].sum()
                if wb_passes > 5:
                    wing_back_names.append(f"Role {wb}")
                    wing_back_passes += wb_passes
        
        stats['wing_back_insight'] = (
            f"wing-backs {', '.join(wing_back_names)} actively contributing to attacks with {int(wing_back_passes)} forward passes"
            if wing_back_names else "limited wing-back involvement in attacking play"
        )
        
        # if not pass_connections.empty:
        #     top_pair = pass_connections.loc[pass_connections['count'].idxmax()]
        #     stats['key_connection'] = (
        #         f"Role {top_pair['role']} and Role {top_pair['pass_recipient_role']} "
        #         f"linked up {int(top_pair['count'])} times"
        #     )
        # else:
        #     stats['key_connection'] = "no dominant passing connections"
        
        return stats
    except Exception as e:
        return {
            'total_passes': 0,
            'pass_accuracy': 0,
            'possession_style': 'unknown',
            'formation': 'Unknown',
            'wing_back_insight': 'no wing-back data available',
            # 'key_connection': 'no passing connections available'
        }

def generate_pass_network_description(team, stats):
    """
    Generate a tactical description for the pass network.
    
    Args:
        team (str): Team name.
        stats (dict): Statistics and insights from analyze_pass_network.
    
    Returns:
        str: Formatted description.
    """
    try:
        templates = {
            'base': [
                "{team} completed {pass_accuracy}% of {total_passes} passes, indicating a {possession_style} approach.",
                "{team} achieved a {pass_accuracy}% pass completion rate across {total_passes} attempts, suggesting a {possession_style} style."
            ],
            'formation': [
                "They appeared to play in a {formation} formation",
                "The team lined up in a {formation} setup"
            ],
            'wing_back': [
                ", with {wing_back_insight}.",
                ", where {wing_back_insight}."
            ],
            # 'connection': [
            #     " {key_connection}.",
            #     " Notably, {key_connection}."
            # ]
        }
        
        description = (
            random.choice(templates['base']).format(team=team, **stats) + " " +
            random.choice(templates['formation']).format(**stats) +
            random.choice(templates['wing_back']).format(**stats) +
            random.choice(templates['connection']).format(**stats)
        )
        
        return description
    except Exception as e:
        return f"No tactical insights available for {team}."

def pass_network(events, team_name, match_id, color):
    """
    Plot pass network for a team with 11 bubbles representing roles based on average position similarities.
    
    Args:
        events (dict): Match event data.
        team_name (str): Name of the team.
        match_id (int): Match ID.
        color (str): Color for visualization.
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
            
        # Calculate average positions and successful passes for all players
        locations = successful_passes['location'].apply(lambda x: pd.Series(x, index=['x', 'y']))
        successful_passes[['x', 'y']] = locations
        player_avg_locations = successful_passes.groupby('player')[['x', 'y']].mean()
        
        # Cluster players into 11 roles based on average positions
        kmeans = KMeans(n_clusters=11, random_state=42)
        player_avg_locations['role'] = kmeans.fit_predict(player_avg_locations[['x', 'y']])
        
        # Map players to their roles
        player_to_role = player_avg_locations['role'].to_dict()
        successful_passes['role'] = successful_passes['player'].map(player_to_role)
        successful_passes['pass_recipient_role'] = successful_passes['pass_recipient'].map(player_to_role)
        
        # Calculate average locations and pass counts by role
        role_avg_locations = successful_passes.groupby('role')[['x', 'y']].mean()
        role_pass_counts = successful_passes.groupby('role').size()
        role_avg_locations['pass_count'] = role_pass_counts
        role_avg_locations['marker_size'] = 300 + (1200 * (role_avg_locations['pass_count'] / role_pass_counts.max()))
        
        # Aggregate pass connections by role
        role_pass_connections = successful_passes.groupby(
            ['role', 'pass_recipient_role']
        ).size().reset_index(name='count')
        role_pass_connections = role_pass_connections.merge(
            role_avg_locations[['x', 'y']], 
            left_on='role', 
            right_index=True
        ).merge(
            role_avg_locations[['x', 'y']], 
            left_on='pass_recipient_role', 
            right_index=True, 
            suffixes=('', '_end')
        )
        role_pass_connections['width'] = 1 + (4 * (role_pass_connections['count'] / role_pass_connections['count'].max()))
        
        # Plotting
        pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
        fig, ax = pitch.draw(figsize=(10, 6.5))
        fig.set_facecolor(FIG_BG_COLOR)
        
        # Heatmap
        heatmap_bins = (6, 4)
        bs_heatmap = pitch.bin_statistic(successful_passes['x'], successful_passes['y'], 
                                        statistic='count', bins=heatmap_bins)
        pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues' if color == HOME_COLOR else 'Reds', 
                     alpha=0.3, zorder=0.5)
        
        # Pass connections
        pitch.lines(
            role_pass_connections.x,
            role_pass_connections.y,
            role_pass_connections.x_end,
            role_pass_connections.y_end,
            lw=role_pass_connections.width,
            color=color,
            zorder=1,
            ax=ax
        )
        
        # Role bubbles
        pitch.scatter(
            role_avg_locations.x,
            role_avg_locations.y,
            s=role_avg_locations.marker_size,
            color=color,
            edgecolors="black",
            linewidth=0.5,
            alpha=1,
            ax=ax,
            zorder=2
        )
        
        # Inner white circles for contrast
        pitch.scatter(
            role_avg_locations.x,
            role_avg_locations.y,
            s=role_avg_locations.marker_size/4,
            color="white",
            edgecolors="black",
            linewidth=0.5,
            alpha=1,
            ax=ax,
            zorder=3
        )
        
        # Role labels
        # for role, row in role_avg_locations.iterrows():
        #     text = ax.text(
        #         row.x, row.y,
        #         f"Role {role}",
        #         color="black",
        #         va="center",
        #         ha="center",
        #         size=10,
        #         weight="bold",
        #         zorder=4
        #     )
        #     text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="white")])
        
        ax.set_title(f"{team_name} Pass Network", fontsize=16, pad=20, color=TEXT_COLOR)
        fig.text(0.02, 0.02, '@ahmedtarek26 / GitHub', 
                fontstyle='italic', fontsize=FONT_SIZE_SM-2, color=TEXT_COLOR)
        
        plt.tight_layout()
        plt.savefig(f'graphs/pass_network_{team_name}_{match_id}.png', 
                   dpi=300, bbox_inches='tight', facecolor=FIG_BG_COLOR)
        st.image(f'graphs/pass_network_{team_name}_{match_id}.png')
        
        # Generate and display tactical description
        stats = analyze_pass_network(team_passes, successful_passes, role_pass_connections, role_avg_locations)
        description = generate_pass_network_description(team_name, stats)
        st.markdown(
            f"""
            <div style="background-color:{'#2d2d2d' if DARK_MODE else '#f8f9fa'};
                        padding:1rem;border-radius:8px;margin-top:1rem;color:{TEXT_COLOR};">
                <strong>Tactical Analysis:</strong> {description}
            </div>
            """, unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Error creating pass network for {team_name}: {str(e)}")

def player_stats_tab(events, home_team, away_team, home_lineup, away_lineup):
    with st.expander("Player Statistics", expanded=True):
        all_players = list(home_lineup) + list(away_lineup)
        selected_player = st.selectbox("Select Player", all_players)
        
        player_team = home_team if selected_player in home_lineup else away_team
        team_color = HOME_COLOR if player_team == home_team else AWAY_COLOR
        
        st.subheader(f"Performance Analysis: {selected_player}")
        
        tab1, tab2, tab3 = st.tabs(["Heatmap", "Event Stats", "Passing Network"])
        
        with tab1:
            plot_player_heatmap(events, selected_player, player_team, team_color)
            
        with tab2:
            show_player_stats(events, selected_player)
            
        with tab3:
            plot_player_passing_network(events, selected_player, player_team, team_color)

def plot_player_heatmap(events, player_name, team_name, color):
    try:
        player_events = pd.concat([events[action_type] 
                                 for action_type in events 
                                 if 'player' in events[action_type].columns])
        player_events = player_events[player_events['player'] == player_name]
        
        if player_events.empty:
            st.warning(f"No position data available for {player_name}")
            return
            
        locations = player_events['location'].dropna()
        if len(locations) == 0:
            st.warning(f"No location data available for {player_name}")
            return
            
        x = [loc[0] for loc in locations]
        y = [loc[1] for loc in locations]
        
        fig, ax = create_pitch_figure(f"{player_name} Heatmap")
        pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
        pitch.draw(ax=ax)
        
        hb = pitch.hexbin(x, y, ax=ax, cmap='Reds', gridsize=15, alpha=0.7)
        
        cb = fig.colorbar(hb, ax=ax, shrink=0.7)
        cb.set_label('Activity Density', color=TEXT_COLOR)
        cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=TEXT_COLOR)
        
        st.pyplot(fig)
        
        # Analytical Description for Heatmap
        avg_x = np.mean(x)
        if avg_x < 40:
            position = "defensive areas"
        elif avg_x < 80:
            position = "midfield"
        else:
            position = "attacking zones"
        description = f"{player_name} was most active in {position}."
        
        st.markdown(
            f"""
            <div style="background-color:{'#2d2d2d' if DARK_MODE else '#f8f9fa'};
                        padding:1rem;border-radius:8px;margin-top:1rem;color:{TEXT_COLOR};">
                <strong>Heatmap Analysis:</strong> {description}
            </div>
            """, unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Heatmap error for {player_name}: {str(e)}")

def show_player_stats(events, player_name):
    try:
        stats = {
            "Shots": 0,
            "Goals": 0,
            "Passes": 0,
            "Successful Passes": 0,
            "Dribbles": 0,
            "Tackles": 0,
            "Interceptions": 0
        }
        
        if 'shots' in events:
            player_shots = events['shots'][events['shots']['player'] == player_name]
            stats["Shots"] = len(player_shots)
            stats["Goals"] = len(player_shots[player_shots['shot_outcome'] == 'Goal'])
        
        if 'passes' in events:
            player_passes = events['passes'][events['passes']['player'] == player_name]
            stats["Passes"] = len(player_passes)
            stats["Successful Passes"] = len(player_passes[player_passes['pass_outcome'].isna()])
        
        if 'dribbles' in events:
            stats["Dribbles"] = len(events['dribbles'][events['dribbles']['player'] == player_name])
        
        if 'interceptions' in events:
            stats["Interceptions"] = len(events['interceptions'][events['interceptions']['player'] == player_name])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Shots", stats["Shots"])
            st.metric("Goals", stats["Goals"])
            
        with col2:
            st.metric("Passes", stats["Passes"])
            st.metric("Pass Accuracy", 
                     f"{(stats['Successful Passes']/stats['Passes']*100 if stats['Passes'] > 0 else 0):.1f}%")
            
        with col3:
            st.metric("Dribbles", stats["Dribbles"])
            st.metric("Interceptions", stats["Interceptions"])
            
        # Analytical Description for Player Stats
        pass_accuracy = (stats['Successful Passes'] / stats['Passes'] * 100) if stats['Passes'] > 0 else 0
        description = f"{player_name} attempted {stats['Passes']} passes with {pass_accuracy:.1f}% accuracy."
        if pass_accuracy > 85:
            description += " They were highly accurate in their passing."
        elif pass_accuracy < 70:
            description += " Their passing accuracy was below average."
        else:
            description += " Their passing was solid but not exceptional."
        
        st.markdown(
            f"""
            <div style="background-color:{'#2d2d2d' if DARK_MODE else '#f8f9fa'};
                        padding:1rem;border-radius:8px;margin-top:1rem;color:{TEXT_COLOR};">
                <strong>Player Stats Analysis:</strong> {description}
            </div>
            """, unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Stats error for {player_name}: {str(e)}")

def plot_player_passing_network(events, player_name, team_name, color):
    try:
        if 'passes' not in events:
            st.warning("No passes data available")
            return
            
        player_passes = events['passes'][events['passes']['player'] == player_name]
        if player_passes.empty:
            st.warning(f"No passes data for {player_name}")
            return
            
        fig, ax = create_pitch_figure(f"{player_name} Passing Network")
        pitch = Pitch(pitch_type='statsbomb', line_color=LINE_COLOR, pitch_color=PITCH_COLOR)
        pitch.draw(ax=ax)
        
        for _, pass_event in player_passes.iterrows():
            if isinstance(pass_event['location'], list) and isinstance(pass_event['pass_end_location'], list):
                start_x, start_y = pass_event['location'][0], pass_event['location'][1]
                end_x, end_y = pass_event['pass_end_location'][0], pass_event['pass_end_location'][1]
                
                if team_name != pass_event['team']:
                    start_x, start_y = 120 - start_x, 80 - start_y
                    end_x, end_y = 120 - end_x, 80 - end_y
                
                if pd.isna(pass_event['pass_outcome']):
                    pitch.arrows(start_x, start_y, end_x, end_y, 
                               ax=ax, color=color, width=2, headwidth=4, headlength=4)
                else:
                    pitch.arrows(start_x, start_y, end_x, end_y, 
                               ax=ax, color='gray', width=1, headwidth=3, headlength=3, alpha=0.5)
        
        avg_x = player_passes['location'].apply(lambda x: x[0]).mean()
        avg_y = player_passes['location'].apply(lambda x: x[1]).mean()
        pitch.scatter(avg_x, avg_y, ax=ax, s=300, color=color, edgecolors='black', linewidth=1)
        
        legend_elements = [
            Line2D([0], [0], color=color, lw=2, label='Successful Pass'),
            Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Unsuccessful Pass'),
            Line2D([0], [0], marker='o', color=color, label='Player Position Avg',
                   markerfacecolor=color, markersize=10, alpha=1)
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        st.pyplot(fig)
        
        # Analytical Description for Player Passing Network
        total_passes = len(player_passes)
        successful_passes = len(player_passes[player_passes['pass_outcome'].isna()])
        pass_accuracy = (successful_passes / total_passes * 100) if total_passes > 0 else 0
        description = f"{player_name} attempted {total_passes} passes with {pass_accuracy:.1f}% accuracy."
        if pass_accuracy > 85:
            description += " They were highly accurate in their passing."
        elif pass_accuracy < 70:
            description += " Their passing accuracy was below average."
        else:
            description += " Their passing was solid but not exceptional."
        
        st.markdown(
            f"""
            <div style="background-color:{'#2d2d2d' if DARK_MODE else '#f8f9fa'};
                        padding:1rem;border-radius:8px;margin-top:1rem;color:{TEXT_COLOR};">
                <strong>Passing Network Analysis:</strong> {description}
            </div>
            """, unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Passing network error for {player_name}: {str(e)}")

# --------------------------
# MAIN APP FUNCTION
# --------------------------

def main():
    st.title('⚽ Football Match Analysis')
    
    try:
        if 'analyzed' not in st.session_state:
            st.session_state.analyzed = False
        if 'match_data' not in st.session_state:
            st.session_state.match_data = None
        
        com = sb.competitions()
        com_dict = dict(zip(com['competition_name'], com['competition_id']))
        
        if 'competitions_data' not in st.session_state:
            st.session_state.competitions_data = com
            st.session_state.available_seasons = {}
        
        competition = st.selectbox('Choose the competition', com_dict.keys())
        
        if competition:
            selected_comp_id = com_dict[competition]
            available_seasons = st.session_state.competitions_data[
                st.session_state.competitions_data['competition_id'] == selected_comp_id
            ]
            
            season_dict = dict(zip(available_seasons['season_name'], available_seasons['season_id']))
            
            season = st.selectbox('Choose the season', season_dict.keys())
            
            try:
                data = sb.matches(competition_id=selected_comp_id, 
                                season_id=season_dict[season])
                
                matches_names, matches_idx, matches_id_dict = matches_id(data)
                match = st.selectbox('Select the match', matches_names)
                
                if st.button('Analyze Match') or st.session_state.analyzed:
                    st.session_state.analyzed = True
                    if st.session_state.match_data is None or st.session_state.match_data.get('match_id') != matches_id_dict[match]:
                        home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(
                            data, matches_idx[match])
                        
                        match_id = matches_id_dict[match]
                        lineup_data = sb.lineups(match_id=match_id)
                        home_lineup, away_lineup = lineups(home_team, away_team, lineup_data)
                        events = sb.events(match_id=match_id, split=True)
                        
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
                        match_data_dict = st.session_state.match_data
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

                    st.markdown("---")
                    st.header("Match Visualizations")
                    
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚽ Attack", "🛡️ Defense", "👤 Player Actions", "📊 Stats", "👤 Player Stats"])
                    
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
                            shots_df = events['shots'].copy()
                            shots_df['player_lastname'] = shots_df['player'].apply(lambda name: name.split()[-1])
                            shots_summary = shots_df.groupby(['player_lastname', 'team']).size().reset_index(name='count')
                            st.plotly_chart(px.bar(
                                shots_summary,
                                y='player_lastname', 
                                x='count', 
                                color='team',
                                color_discrete_map={home_team: HOME_COLOR, away_team: AWAY_COLOR},
                                title="Shots by Player", 
                                template=PLOTLY_TEMPLATE
                            ))
                        
                        if 'passes' in events:
                            passes_df = events['passes'].copy()
                            passes_df['player_lastname'] = passes_df['player'].apply(lambda name: name.split()[-1])
                            passes_summary = passes_df.groupby(['player_lastname', 'team']).size().reset_index(name='count')
                            st.plotly_chart(px.bar(
                                passes_summary, 
                                y='player_lastname', 
                                x='count', 
                                color='team',
                                color_discrete_map={home_team: HOME_COLOR, away_team: AWAY_COLOR},
                                title="Passes by Player", 
                                template=PLOTLY_TEMPLATE
                            ))

                        if 'foul_committeds' in events:
                            fouls_df = events['foul_committeds'].copy()
                            fouls_df['player_lastname'] = fouls_df['player'].apply(lambda name: name.split()[-1])
                            fouls_summary = fouls_df.groupby(['player_lastname', 'team']).size().reset_index(name='count')
                            st.plotly_chart(px.bar(
                                fouls_summary, 
                                y='player_lastname', 
                                x='count', 
                                color='team',
                                color_discrete_map={home_team: HOME_COLOR, away_team: AWAY_COLOR},
                                title="Fouls Committed", 
                                template=PLOTLY_TEMPLATE
                            ))
                    
                    with tab5:
                        player_stats_tab(events, home_team, away_team, home_lineup, away_lineup)

                    if st.button('Analyze New Match'):
                        st.session_state.analyzed = False
                        st.session_state.match_data = None
                        st.experimental_rerun()
                        
            except Exception as e:
                st.warning(f"Please select a valid season for this competition. Error: {str(e)}")
                st.stop()
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()