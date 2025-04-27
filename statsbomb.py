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
    FIG_BG_COLOR = "#2d2d2d"
else:
    BG_COLOR = "#f8f9fa"
    PITCH_COLOR = "#f8f9fa"
    LINE_COLOR = "#000000"
    TEXT_COLOR = "#000000"
    HOME_COLOR = "#d62828"
    AWAY_COLOR = "#003049"
    FIG_BG_COLOR = "#ffffff"

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

def dribbles(events, h, w, match_id):
    try:
        fig, ax = create_pitch_figure('🏃‍♂️ Dribbles')
        
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
            
            legend = ax.legend(loc="upper left", framealpha=0.8)
            legend.get_frame().set_facecolor(FIG_BG_COLOR)
            
            total_dribbles = len(events['dribbles'])
            fig_text(s=f'Total Dribbles: {total_dribbles}', x=0.15, y=0.85,
                    fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
            save_and_display(fig, f'dribbles-{match_id}.png')
        else:
            st.warning("No dribbles data available")
    except Exception as e:
        st.error(f"Dribbles visualization error: {str(e)}")

def foul_committed(events, h, w, match_id):
    try:
        fig, ax = create_pitch_figure('🔴 Fouls Committed')
        
        if 'foul_committeds' in events:
            x_h = []
            y_h = []
            x_w = []
            y_w = []
            
            for i, foul in events['foul_committeds'].iterrows():
                if events['foul_committeds']['possession_team'][i] == h:
                    x_h.append(foul['location'][0])
                    y_h.append(foul['location'][1])
                elif events['foul_committeds']['possession_team'][i] == w:
                    x_w.append(foul['location'][0])
                    y_w.append(foul['location'][1])
            
            ax.scatter(x_h, y_h, s=80, c=HOME_COLOR, alpha=.7, label=h)
            ax.scatter(x_w, y_w, s=80, c=AWAY_COLOR, alpha=.7, label=w)
            
            legend = ax.legend(loc="upper left", framealpha=0.8)
            legend.get_frame().set_facecolor(FIG_BG_COLOR)
            
            total_fouls = len(events['foul_committeds'])
            fig_text(s=f'Total Fouls: {total_fouls}', x=0.15, y=0.85,
                    fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
            save_and_display(fig, f'fouls-{match_id}.png')
        else:
            st.warning("No fouls data available")
    except Exception as e:
        st.error(f"Fouls visualization error: {str(e)}")

def interception(events, h, w, match_id):
    try:
        fig, ax = create_pitch_figure('🛡️ Interceptions')
        
        if 'interceptions' in events:
            x_h = []
            y_h = []
            x_w = []
            y_w = []
            
            for i, interception in events['interceptions'].iterrows():
                if events['interceptions']['possession_team'][i] == h:
                    x_h.append(interception['location'][0])
                    y_h.append(interception['location'][1])
                elif events['interceptions']['possession_team'][i] == w:
                    x_w.append(interception['location'][0])
                    y_w.append(interception['location'][1])
            
            ax.scatter(x_h, y_h, s=80, c=HOME_COLOR, alpha=.7, label=h)
            ax.scatter(x_w, y_w, s=80, c=AWAY_COLOR, alpha=.7, label=w)
            
            legend = ax.legend(loc="upper left", framealpha=0.8)
            legend.get_frame().set_facecolor(FIG_BG_COLOR)
            
            total_interceptions = len(events['interceptions'])
            fig_text(s=f'Total Interceptions: {total_interceptions}', x=0.15, y=0.85,
                    fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
            save_and_display(fig, f'interceptions-{match_id}.png')
        else:
            st.warning("No interceptions data available")
    except Exception as e:
        st.error(f"Interceptions visualization error: {str(e)}")

def dispossesseds(events, h, w, match_id):
    try:
        fig, ax = create_pitch_figure('💥 Dispossessions')
        
        if 'dispossesseds' in events:
            x_h = []
            y_h = []
            x_w = []
            y_w = []
            
            for i, dispossessed in events['dispossesseds'].iterrows():
                if events['dispossesseds']['possession_team'][i] == h:
                    x_h.append(dispossessed['location'][0])
                    y_h.append(dispossessed['location'][1])
                elif events['dispossesseds']['possession_team'][i] == w:
                    x_w.append(dispossessed['location'][0])
                    y_w.append(dispossessed['location'][1])
            
            ax.scatter(x_h, y_h, s=80, c=HOME_COLOR, alpha=.7, label=h)
            ax.scatter(x_w, y_w, s=80, c=AWAY_COLOR, alpha=.7, label=w)
            
            legend = ax.legend(loc="upper left", framealpha=0.8)
            legend.get_frame().set_facecolor(FIG_BG_COLOR)
            
            total_dispossessions = len(events['dispossesseds'])
            fig_text(s=f'Total Dispossessions: {total_dispossessions}', x=0.15, y=0.85,
                    fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
            save_and_display(fig, f'dispossessions-{match_id}.png')
        else:
            st.warning("No dispossessions data available")
    except Exception as e:
        st.error(f"Dispossessions visualization error: {str(e)}")

def miscontrols(events, h, w, match_id):
    try:
        fig, ax = create_pitch_figure('⚠️ Miscontrols')
        
        if 'miscontrols' in events:
            x_h = []
            y_h = []
            x_w = []
            y_w = []
            
            for i, miscontrol in events['miscontrols'].iterrows():
                if events['miscontrols']['possession_team'][i] == h:
                    x_h.append(miscontrol['location'][0])
                    y_h.append(miscontrol['location'][1])
                elif events['miscontrols']['possession_team'][i] == w:
                    x_w.append(miscontrol['location'][0])
                    y_w.append(miscontrol['location'][1])
            
            ax.scatter(x_h, y_h, s=80, c=HOME_COLOR, alpha=.7, label=h)
            ax.scatter(x_w, y_w, s=80, c=AWAY_COLOR, alpha=.7, label=w)
            
            legend = ax.legend(loc="upper left", framealpha=0.8)
            legend.get_frame().set_facecolor(FIG_BG_COLOR)
            
            total_miscontrols = len(events['miscontrols'])
            fig_text(s=f'Total Miscontrols: {total_miscontrols}', x=0.15, y=0.85,
                    fontsize=FONT_SIZE_MD, color=TEXT_COLOR, fontfamily=FONT)
            save_and_display(fig, f'miscontrols-{match_id}.png')
        else:
            st.warning("No miscontrols data available")
    except Exception as e:
        st.error(f"Miscontrols visualization error: {str(e)}")

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

def save_and_display(fig, filename):
    fig.text(0.02, 0.02, '@ahmedtarek26 / GitHub', 
            fontstyle='italic', fontsize=FONT_SIZE_SM-2, 
            color=TEXT_COLOR, fontfamily=FONT)
    plt.tight_layout()
    plt.savefig(f'graphs/{filename}', dpi=300, bbox_inches='tight')
    st.image(f'graphs/{filename}')

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
        
        if st.button('Analyze Match'):  # Removed the 'type' parameter
            home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(
                data, matches_idx[match])
            
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
            
            tab1, tab2, tab3 = st.tabs(["⚔️ Attack Analysis", "🛡️ Defense Analysis", "📊 Match Stats"])
            
            with tab1:
                shots_goal(events.get('shots', pd.DataFrame()), home_team, away_team, match_id)
                goals(events.get('shots', pd.DataFrame()), home_team, away_team, match_id)
                dribbles(events, home_team, away_team, match_id)
                
                st.subheader("📶 Pass Networks")
                col1, col2 = st.columns(2)
                with col1:
                    pass_network(events, home_team, match_id, HOME_COLOR)
                with col2:
                    pass_network(events, away_team, match_id, AWAY_COLOR)
            
            with tab2:
                foul_committed(events, home_team, away_team, match_id)
                interception(events, home_team, away_team, match_id)
                dispossesseds(events, home_team, away_team, match_id)
                miscontrols(events, home_team, away_team, match_id)
            
            with tab3:
                st.subheader("📈 Match Statistics")
                if 'shots' in events:
                    st.plotly_chart(px.bar(events['shots'], x=['player', 'position'], color='team', 
                                         title="Shots by Player"))
                if 'foul_committeds' in events:
                    st.plotly_chart(px.bar(events['foul_committeds'], x=['player', 'position'], color='team',
                                         title="Fouls Committed"))
                if 'foul_wons' in events:
                    st.plotly_chart(px.bar(events['foul_wons'], x=['player', 'position'], color='team',
                                        title="Fouls Won"))
                if 'carrys' in events:
                    st.plotly_chart(px.bar(events['carrys'], x=['player', 'position'], color='team',
                                       title="Carries"))

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Commented pass map functions (keep in code but not called)
"""
def home_team_passes(events, home_team, match_id):
    x_h = []
    y_h = []

    for i, shot in events['passes'].iterrows():
        if events['passes']['possession_team'][i] == home_team:
            x_h.append(shot['location'][0])
            y_h.append(shot['location'][1])

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color='gray', pitch_color='#22312b')
    bins = (6, 4)

    fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
    fig_text(s=f'{home_team} Passes: {len(x_h)}',
             x=.49, y=.67, fontsize=14, color='yellow')
    fig.text(.22, .14, f'@ahmedtarek26 / Github', fontstyle='italic', fontsize=12, color='yellow')
    bs_heatmap = pitch.bin_statistic(x_h, y_h, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues')
    plt.savefig(f'graphs/{home_team}passes-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/{home_team}passes-{match_id}.png')

def away_team_passes(events, away_team, match_id):
    x_w = []
    y_w = []
    for i, shot in events['dribbles'].iterrows():
        if events['dribbles']['possession_team'][i] == away_team:
            x_w.append(shot['location'][0])
            y_w.append(shot['location'][1])

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color='gray', pitch_color='#22312b')
    bins = (6, 4)
    fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
    fig_text(s=f'{away_team} Passes: {len(x_w)}',
             x=.49, y=.67, fontsize=14, color='yellow')
    fig.text(.22, .14, f'@ahmedtarek26 / Github', fontstyle='italic', fontsize=12, color='yellow')
    fig.set_facecolor('#22312b')
    bs_heatmap = pitch.bin_statistic(x_w, y_w, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Reds')
    plt.savefig(f'graphs/{away_team}passes-{match_id}.png', dpi=300, bbox_inches='tight')
    st.image(f'graphs/{away_team}passes-{match_id}.png')
"""

if __name__ == "__main__":
    main()