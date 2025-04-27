#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer.pitch import Pitch, VerticalPitch
from highlight_text import fig_text
import plotly.express as px
from dash import dcc
from mplsoccer import FontManager
import matplotlib.patheffects as path_effects

# ========== STYLING SETTINGS ==========
# Custom font setup
URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/raleway/Raleway%5Bwght%5D.ttf"
custom_font = FontManager(URL)

# Color palette
PALETTE = {
    'background': '#f8f9fa',
    'text': '#212529',
    'primary': '#4361ee',
    'secondary': '#3a0ca3',
    'accent': '#7209b7',
    'success': '#4cc9f0',
    'danger': '#f72585',
    'warning': '#f8961e',
    'info': '#4895ef'
}

# Pitch styling
PITCH_STYLE = {
    'pitch_color': '#f8f9fa',
    'line_color': '#495057',
    'linewidth': 1.5,
    'spot_scale': 0.005
}

# ========== HELPER FUNCTIONS ==========
def setup_plot(title="", subtitle="", figsize=(12, 8)):
    """Set up a standardized plot with consistent styling"""
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(PALETTE['background'])
    ax.patch.set_facecolor(PALETTE['background'])
    
    if title:
        fig.suptitle(title, 
                    fontproperties=custom_font.prop, 
                    fontsize=24, 
                    color=PALETTE['text'],
                    y=0.95)
    if subtitle:
        ax.set_title(subtitle, 
                    fontproperties=custom_font.prop,
                    fontsize=16,
                    color=PALETTE['text'],
                    pad=20)
    
    return fig, ax

def add_credit(fig):
    """Add consistent credit line to plots"""
    fig.text(0.1, 0.02, 
            '@ahmedtarek26 / Github', 
            fontstyle='italic', 
            fontsize=10, 
            color=PALETTE['text'],
            fontproperties=custom_font.prop)

# ========== VISUALIZATION FUNCTIONS ==========
def pass_network(events, team_name, match_id, color=PALETTE['primary']):
    """Enhanced pass network visualization with modern styling"""
    try:
        # Data processing
        passes = events['passes']
        team_passes = passes[passes['team'] == team_name]
        successful_passes = team_passes[team_passes['pass_outcome'].isna()].copy()
        
        # Extract coordinates
        locations = successful_passes['location'].apply(lambda x: pd.Series(x, index=['x', 'y']))
        successful_passes[['x', 'y']] = locations
        
        # Calculate metrics
        avg_locations = successful_passes.groupby('player')[['x', 'y']].mean()
        pass_counts = successful_passes['player'].value_counts()
        avg_locations['pass_count'] = avg_locations.index.map(pass_counts)
        avg_locations['marker_size'] = 400 + (1600 * (avg_locations['pass_count'] / pass_counts.max()))
        
        # Pass connections
        pass_connections = successful_passes.groupby(['player', 'pass_recipient']).size().reset_index(name='count')
        pass_connections = pass_connections.merge(avg_locations[['x', 'y']], left_on='player', right_index=True)
        pass_connections = pass_connections.merge(avg_locations[['x', 'y']], left_on='pass_recipient', right_index=True, suffixes=['', '_end'])
        pass_connections['width'] = 1 + (4 * (pass_connections['count'] / pass_connections['count'].max()))
        
        # Setup plot
        pitch = Pitch(pitch_type="statsbomb", **PITCH_STYLE)
        fig, ax = setup_plot(
            title=f"{team_name} Pass Network",
            subtitle=f"Total Successful Passes: {len(successful_passes)}",
            figsize=(14, 9)
        )
        pitch.draw(ax=ax)
        
        # Heatmap layer
        bs_heatmap = pitch.bin_statistic(successful_passes['x'], successful_passes['y'], statistic='count', bins=(12, 8))
        pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues' if color == PALETTE['primary'] else 'Reds', alpha=0.25, zorder=1)
        
        # Pass connections
        pitch.lines(
            pass_connections.x, pass_connections.y,
            pass_connections.x_end, pass_connections.y_end,
            lw=pass_connections.width,
            color=color,
            zorder=2,
            ax=ax,
            comet=True,
            alpha=0.7
        )
        
        # Player nodes
        pitch.scatter(
            avg_locations.x, avg_locations.y,
            s=avg_locations.marker_size,
            color=color,
            edgecolors=PALETTE['text'],
            linewidth=1,
            alpha=1,
            ax=ax,
            zorder=3
        )
        
        # Player labels
        for index, row in avg_locations.iterrows():
            text = pitch.annotate(
                index.split()[-1],
                xy=(row.x, row.y),
                color=PALETTE['text'],
                va='center',
                ha='center',
                size=12,
                weight='bold',
                ax=ax,
                zorder=4,
                fontproperties=custom_font.prop
            )
            text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
        
        add_credit(fig)
        plt.tight_layout()
        plt.savefig(f'graphs/pass_network_{team_name}_{match_id}.png', dpi=300, bbox_inches='tight')
        st.image(f'graphs/pass_network_{team_name}_{match_id}.png')
        
    except Exception as e:
        st.error(f"Error creating pass network: {str(e)}")

def shots_goal(shots, h, w, match_id):
    """Modernized shot visualization"""
    pitch = Pitch(pitch_type='statsbomb', **PITCH_STYLE)
    fig, ax = setup_plot(
        title=f"{h} vs {w} Shots",
        subtitle=f"Total Shots: {len(shots)}",
        figsize=(12, 8)
    )
    pitch.draw(ax=ax)
    
    for i, shot in shots.iterrows():
        x, y = shot['location']
        goal = shot['shot_outcome'] == 'Goal'
        team_name = shot['team']
        
        size = np.sqrt(shot['shot_statsbomb_xg']) * 8
        color = PALETTE['primary'] if team_name == h else PALETTE['secondary']
        
        if team_name == h:
            y = 80 - y  # Flip y-axis for home team
            
        # Plot shot
        shot_circle = plt.Circle((x, y), size, color=color, alpha=0.7 if not goal else 1)
        ax.add_patch(shot_circle)
        
        # Annotate goals
        if goal:
            text = pitch.annotate(
                f"{shot['player']} ({shot['shot_body_part']}, xG: {shot['shot_statsbomb_xg']:.2f})",
                xy=(x, y),
                xytext=(0, 15),
                textcoords='offset points',
                ha='center',
                color=PALETTE['text'],
                fontproperties=custom_font.prop,
                size=10
            )
            text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    add_credit(fig)
    plt.tight_layout()
    plt.savefig(f'graphs/shots-{match_id}.png', dpi=300)
    st.image(f'graphs/shots-{match_id}.png')

# ========== STREAMLIT APP ==========
def main():
    # App config
    st.set_page_config(
        page_title="Football Analytics Dashboard",
        page_icon="⚽",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown(f"""
        <style>
            .stApp {{
                background-color: {PALETTE['background']};
            }}
            .css-1d391kg {{
                padding-top: 3.5rem;
            }}
            h1 {{
                color: {PALETTE['primary']};
                font-family: 'Raleway';
            }}
            .stSelectbox label {{
                font-weight: bold;
                color: {PALETTE['text']};
            }}
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title('⚽ Advanced Football Analytics')
    st.markdown("""
        <style>
            div[data-testid="stMarkdownContainer"] > p {
                font-family: 'Raleway';
                color: #495057;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Data loading
    com = sb.competitions()
    com_dict = dict(zip(com['competition_name'], com['competition_id']))
    season_dict = dict(zip(com['season_name'], com['season_id']))
    
    # UI Elements
    col1, col2 = st.columns(2)
    with col1:
        competition = st.selectbox('Select Competition', com_dict.keys(), 
                                 help="Choose the competition to analyze")
    with col2:
        season = st.selectbox('Select Season', season_dict.keys(),
                            help="Choose the season to analyze")
    
    data = sb.matches(competition_id=com_dict[competition], season_id=season_dict[season])
    matches_names, matches_idx, matches_id = matches_id(data)
    
    match = st.selectbox('Select Match', matches_names,
                        help="Choose a specific match to analyze")
    
    if st.button('Analyze Match', type="primary"):
        with st.spinner('Processing match data...'):
            analyze_match(data, matches_idx[match], matches_id[match])

def analyze_match(data, match_idx, match_id):
    """Main analysis function with improved layout"""
    # Match info
    home_team, away_team, home_score, away_score, stadium, home_manager, away_manager, comp_stats = match_data(data, match_idx)
    home_lineup, away_lineup = lineups(home_team, away_team, data=sb.lineups(match_id=match_id))
    events = sb.events(match_id=match_id, split=True)
    
    # Match header
    st.header(f"{home_team} {home_score} - {away_score} {away_team}")
    st.caption(f"{stadium} | {comp_stats}")
    
    # Lineups
    with st.expander("📋 Team Lineups", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.subheader(home_team)
            st.markdown(f"**Manager:** {home_manager}")
            st.markdown("**Starting XI:**")
            for player in home_lineup:
                st.markdown(f"- {player}")
        with col2:
            st.metric("Score", f"{home_score} - {away_score}")
        with col3:
            st.subheader(away_team)
            st.markdown(f"**Manager:** {away_manager}")
            st.markdown("**Starting XI:**")
            for player in away_lineup:
                st.markdown(f"- {player}")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Shots & Goals", "🔀 Passing Networks", "⚔️ Defensive Actions", "🏃 Player Movements"])
    
    with tab1:
        st.subheader("Shot Map")
        shots_goal(events['shots'], home_team, away_team, match_id)
        
        st.subheader("Goals Analysis")
        goals(events['shots'], home_team, away_team, match_id)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{home_team} Passing Network")
            pass_network(events, home_team, match_id, color=PALETTE['primary'])
        with col2:
            st.subheader(f"{away_team} Passing Network")
            pass_network(events, away_team, match_id, color=PALETTE['secondary'])
    
    # ... (other tabs and visualizations)