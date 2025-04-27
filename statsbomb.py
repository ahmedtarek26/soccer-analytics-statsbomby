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

def shots_goal(shots, h, w, match_id):
    try:
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

            if team_name == h:
                if goal:
                    shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                    text = ax.text(x + 1, pitchWidthY - y + 2, shot['player'], 
                                 fontsize=FONT_SIZE_SM, color=TEXT_COLOR, 
                                 ha='center', va='center', fontfamily=FONT)
                    text.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
                else:
                    shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                    shotCircle.set_alpha(.2)
            elif team_name == w:
                if goal:
                    shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
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
        fig.text(.02, .02, f'@ahmedtarek26 / Github', 
                 fontstyle='italic', fontsize=FONT_SIZE_SM-2, fontfamily=FONT, color=TEXT_COLOR)
        
        plt.tight_layout()
        plt.savefig(f'graphs/shots-{match_id}.png', dpi=300, bbox_inches='tight')
        st.image(f'graphs/shots-{match_id}.png')
    except Exception as e:
        st.error(f"Error creating shots visualization: {str(e)}")

def goals(shots, h, w, match_id):
    try:
        pitchLengthX = 120
        pitchWidthY = 80

        pitch = Pitch(pitch_type='statsbomb', line_color='#c7d5cc', pitch_color=PITCH_COLOR)
        fig, ax = pitch.draw(figsize=(10, 6.5))
        fig.set_facecolor(FIG_BG_COLOR)

        for i, shot in shots.iterrows():
            if shot['shot_outcome'] == 'Goal':
                x = shot['location'][0]
                y = shot['location'][1]
                x_end = shot['shot_end_location'][0]
                y_end = shot['shot_end_location'][1]
                team_name = shot['team']
                circleSize = np.sqrt(shot['shot_statsbomb_xg']) * 5

                if team_name == h:
                    shotCircle = plt.Circle((x, pitchWidthY - y), circleSize, color=HOME_COLOR)
                    text1 = ax.text(x - 10, pitchWidthY - y - 2, shot['shot_body_part'], 
                                  fontsize=FONT_SIZE_SM, color=TEXT_COLOR, fontfamily=FONT)
                    text1.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
                    
                    text2 = ax.text(x - 10, pitchWidthY - y, f"xG: {round(shot['shot_statsbomb_xg'], 2)}", 
                                  fontsize=FONT_SIZE_SM, color=TEXT_COLOR, fontfamily=FONT)
                    text2.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
                    
                    pitch.arrows(x, pitchWidthY - y, x_end, pitchWidthY - y_end, 
                               color=TEXT_COLOR, width=1, headwidth=5, headlength=5, ax=ax)
                elif team_name == w:
                    shotCircle = plt.Circle((pitchLengthX - x, y), circleSize, color=AWAY_COLOR)
                    text1 = ax.text(pitchLengthX - x - 10, y - 2, shot['shot_body_part'], 
                                  fontsize=FONT_SIZE_SM, color=TEXT_COLOR, fontfamily=FONT)
                    text1.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
                    
                    text2 = ax.text(pitchLengthX - x - 10, y + 2, f"xG: {round(shot['shot_statsbomb_xg'], 2)}", 
                                  fontsize=FONT_SIZE_SM, color=TEXT_COLOR, fontfamily=FONT)
                    text2.set_path_effects([path_effects.withStroke(linewidth=1, foreground="black")])
                    
                    pitch.arrows(pitchLengthX - x, y, pitchLengthX - x_end, y_end, 
                               color=TEXT_COLOR, width=2, headwidth=5, headlength=5, ax=ax)

                ax.add_patch(shotCircle)

        fig.text(.02, .02, f'@ahmedtarek26 / Github', 
                 fontstyle='italic', fontsize=FONT_SIZE_SM-2, fontfamily=FONT, color=TEXT_COLOR)
        
        plt.tight_layout()
        plt.savefig(f'graphs/goals-{match_id}.png', dpi=300, bbox_inches='tight')
        st.image(f'graphs/goals-{match_id}.png')
    except Exception as e:
        st.error(f"Error creating goals visualization: {str(e)}")

def dribbles(events, h, w, match_id):
    try:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.set_facecolor(FIG_BG_COLOR)
        ax.patch.set_facecolor(FIG_BG_COLOR)

        pitch = Pitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
        pitch.draw(ax=ax)
        plt.gca().invert_yaxis()

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
            
            plt.scatter(x_h, y_h, s=80, c=HOME_COLOR, alpha=.7, label=h)
            plt.scatter(x_w, y_w, s=80, c=AWAY_COLOR, alpha=.7, label=w)
            
            legend = plt.legend(loc="upper left", framealpha=0.8)
            legend.get_frame().set_facecolor(FIG_BG_COLOR)
            
            total_dribbles = len(events['dribbles'])
            fig_text(s=f'Total Dribbles: {total_dribbles}',
                    x=.49, y=.67, fontsize=FONT_SIZE_LG, color=TEXT_COLOR, fontfamily=FONT)
            fig.text(.02, .02, '@ahmedtarek / Github', 
                    fontstyle='italic', fontsize=FONT_SIZE_SM-2, fontfamily=FONT, color=TEXT_COLOR)
            
            plt.tight_layout()
            plt.savefig(f'graphs/dribbles-{match_id}.png', dpi=300, bbox_inches='tight')
            st.image(f'graphs/dribbles-{match_id}.png')
        else:
            st.warning("No dribbles data available for this match")
    except Exception as e:
        st.error(f"Error creating dribbles visualization: {str(e)}")

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
        fig.text(.02, .02, '@ahmedtarek26 / Github', 
                fontstyle='italic', fontsize=FONT_SIZE_SM-2, color='black')
        
        plt.tight_layout()
        plt.savefig(f'graphs/pass_network_{team_name}_{match_id}.png', 
                   dpi=300, bbox_inches='tight')
        st.image(f'graphs/pass_network_{team_name}_{match_id}.png')
        
    except Exception as e:
        st.error(f"Error creating pass network for {team_name}: {str(e)}")

# Main app function
def main():
    st.title('⚽ Football Match Analysis')
    
    try:
        # Load competitions data
        com = sb.competitions()
        com_name = com['competition_name']
        com_id = com['competition_id']
        season_name = com['season_name']
        season_id = com['season_id']
        
        com_dict = dict(zip(com_name, com_id))
        season_dict = dict(zip(season_name, season_id))
        
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
                return
            
            match_id = matches_id_dict[match]
            
            # Load lineups
            lineup_data = sb.lineups(match_id=match_id)
            home_lineup, away_lineup = lineups(home_team, away_team, lineup_data)
            
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

# Run the app
if __name__ == "__main__":
    main()