import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NBA BPM Dashboard", layout="wide")
st.title("🏀 NBA BPM Impact Dashboard")
st.markdown("---")

# Constants
TOTAL_TEAM_MINUTES = 240  # 5 players × 48 minutes
IMPACT_MULTIPLIER = 2.083

# Function to load data and normalize minutes
@st.cache_data(ttl=86400)
def load_nba_data():
    try:
        url = "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html"
        tables = pd.read_html(url)
        df = tables[0]
        
        df = df[df['Rk'] != 'Rk'].copy()
        df.columns = df.columns.str.strip()
        
        # Find BPM column
        bpm_column = None
        for col in df.columns:
            if 'BPM' in col:
                bpm_column = col
                break
        
        if bpm_column is None:
            st.error("Could not find BPM column!")
            return None
        
        df = df[['Player', 'Team', 'G', 'MP', bpm_column]].copy()
        df = df.rename(columns={bpm_column: 'BPM'})
        
        df['G'] = pd.to_numeric(df['G'], errors='coerce')
        df['MP'] = pd.to_numeric(df['MP'], errors='coerce')
        df['BPM'] = pd.to_numeric(df['BPM'], errors='coerce')
        df = df.dropna()
        
        # Calculate actual MPG from data
        df['Actual_MPG'] = df['MP'] / df['G']
        
        # For each team, normalize MPG to percentage of 240 minutes
        normalized_data = []
        
        for team in df['Team'].unique():
            team_df = df[df['Team'] == team].copy()
            
            # Get total minutes for the team
            total_team_minutes = team_df['Actual_MPG'].sum()
            
            if total_team_minutes > 0:
                # Calculate each player's percentage of team minutes
                team_df['Minutes_Percent'] = team_df['Actual_MPG'] / total_team_minutes
                
                # Scale to 240 total minutes (5 players × 48 minutes)
                team_df['Normalized_MPG'] = team_df['Minutes_Percent'] * TOTAL_TEAM_MINUTES
            else:
                team_df['Minutes_Percent'] = 0
                team_df['Normalized_MPG'] = 0
            
            normalized_data.append(team_df)
        
        # Combine all teams
        df = pd.concat(normalized_data, ignore_index=True)
        
        # Use normalized MPG for impact calculation
        df['MPG'] = df['Normalized_MPG'].round(1)
        df['Impact'] = (df['BPM'] / 100) * df['MPG'] * IMPACT_MULTIPLIER
        df['Impact'] = df['Impact'].round(3)
        
        # Clean up columns
        df = df[['Player', 'Team', 'G', 'Actual_MPG', 'MPG', 'BPM', 'Impact']]
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to redistribute minutes when players are removed
def redistribute_minutes_after_removal(data, removed_players, team):
    """
    Remove injured players and redistribute their minutes to remaining players
    proportionally based on current normalized MPG.
    """
    # Get all players from the team
    team_data = data[data['Team'] == team].copy()
    
    if len(team_data) == 0:
        return data
    
    # Remove injured players
    healthy_players = team_data[~team_data['Player'].isin(removed_players)].copy()
    
    if len(healthy_players) == 0:
        # All players are injured - return empty
        return data
    
    if len(removed_players) == 0:
        # No injuries on this team
        return data
    
    # Calculate total minutes to redistribute from removed players
    removed_minutes = team_data[team_data['Player'].isin(removed_players)]['MPG'].sum()
    
    # If no minutes to redistribute, just remove players
    if removed_minutes == 0:
        for player in removed_players:
            data = data[~((data['Player'] == player) & (data['Team'] == team))]
        return data
    
    # Calculate redistribution factors based on current MPG
    total_healthy_mpg = healthy_players['MPG'].sum()
    
    if total_healthy_mpg > 0:
        # Calculate new total (should equal 240 after redistribution)
        new_total_minutes = TOTAL_TEAM_MINUTES
        
        # Calculate how much each healthy player's share increases
        # First, calculate their current percentage of healthy minutes
        healthy_percentages = healthy_players['MPG'] / total_healthy_mpg
        
        # Their new MPG = (their percentage) × TOTAL_TEAM_MINUTES
        new_mpg_values = healthy_percentages * TOTAL_TEAM_MINUTES
        
        # Update MPG for healthy players
        for idx, player in healthy_players.iterrows():
            player_name = player['Player']
            new_mpg = new_mpg_values.loc[idx]
            
            # Update the data
            data.loc[(data['Player'] == player_name) & (data['Team'] == team), 'MPG'] = new_mpg
            data.loc[(data['Player'] == player_name) & (data['Team'] == team), 'Impact'] = (
                (data.loc[(data['Player'] == player_name) & (data['Team'] == team), 'BPM'].values[0] / 100) 
                * new_mpg 
                * IMPACT_MULTIPLIER
            )
    
    # Remove injured players from the dataset entirely
    for player in removed_players:
        data = data[~((data['Player'] == player) & (data['Team'] == team))]
    
    return data

# Load the data
with st.spinner('Loading and normalizing NBA data...'):
    nba_data = load_nba_data()

if nba_data is None:
    st.stop()

# Sidebar for controls
st.sidebar.header("📊 Dashboard Controls")

# Team selection FIRST (needed for injury selection)
st.sidebar.subheader("🏀 Matchup Selection")
all_teams = sorted(nba_data['Team'].unique())

col1, col2 = st.sidebar.columns(2)
with col1:
    team1 = st.selectbox("Team 1", all_teams, index=0 if 'LAL' in all_teams else 0)
with col2:
    team2_default = 'BOS' if 'BOS' in all_teams else all_teams[1] if len(all_teams) > 1 else all_teams[0]
    team2 = st.selectbox("Team 2", all_teams, index=all_teams.index(team2_default) if team2_default in all_teams else 0)

# Injury management - show only players from selected teams
st.sidebar.subheader("🚫 Remove Injured Players")

# Get players from both teams for injury selection
team1_players = nba_data[nba_data['Team'] == team1]['Player'].tolist()
team2_players = nba_data[nba_data['Team'] == team2]['Player'].tolist()
all_matchup_players = team1_players + team2_players

# Multi-select for injuries with team grouping
removed_players = st.sidebar.multiselect(
    "Select injured players to remove:",
    options=all_matchup_players,
    help="Selected players will be removed entirely and their minutes redistributed"
)

# Initialize these variables BEFORE using them
team1_removed = []
team2_removed = []

if removed_players:
    # Separate removed players by team
    team1_removed = [p for p in removed_players if p in team1_players]
    team2_removed = [p for p in removed_players if p in team2_players]

# Show team breakdown of injuries
if removed_players:
    if team1_removed:
        minutes_removed_team1 = nba_data[(nba_data['Player'].isin(team1_removed)) & 
                                        (nba_data['Team'] == team1)]['MPG'].sum()
        st.sidebar.info(f"**{team1} removed:** {len(team1_removed)} players ({minutes_removed_team1:.1f} MPG)")
    
    if team2_removed:
        minutes_removed_team2 = nba_data[(nba_data['Player'].isin(team2_removed)) & 
                                        (nba_data['Team'] == team2)]['MPG'].sum()
        st.sidebar.info(f"**{team2} removed:** {len(team2_removed)} players ({minutes_removed_team2:.1f} MPG)")

# Filter options
st.sidebar.subheader("🔍 Filters")
min_games = st.sidebar.slider("Minimum games played:", 1, 82, 20)
filtered_data = nba_data[nba_data['G'] >= min_games].copy()

# Apply injury removal and redistribution
working_data = filtered_data.copy()

if removed_players:
    # Apply removal and redistribution for each team
    if team1_removed:
        working_data = redistribute_minutes_after_removal(working_data, team1_removed, team1)
    
    if team2_removed:
        working_data = redistribute_minutes_after_removal(working_data, team2_removed, team2)

# Main content area
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader(f"📈 {team1} vs {team2} Matchup")
    
    # Get team data after removal
    team1_data = working_data[working_data['Team'] == team1].copy()
    team2_data = working_data[working_data['Team'] == team2].copy()
    
    # Calculate team totals
    team1_impact = team1_data['Impact'].sum()
    team2_impact = team2_data['Impact'].sum()
    advantage = team1_impact - team2_impact
    
    # Calculate total team minutes (should be close to 240)
    team1_total_mpg = team1_data['MPG'].sum()
    team2_total_mpg = team2_data['MPG'].sum()
    
    # Original totals for comparison
    original_team1_data = filtered_data[filtered_data['Team'] == team1]
    original_team2_data = filtered_data[filtered_data['Team'] == team2]
    original_team1_impact = original_team1_data['Impact'].sum()
    original_team2_impact = original_team2_data['Impact'].sum()
    
    # Display matchup metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        impact_change = team1_impact - original_team1_impact
        delta_sign = "+" if impact_change > 0 else ""
        delta_color = "normal" if impact_change >= 0 else "inverse"
        st.metric(f"{team1} Total Impact", 
                 f"{team1_impact:.2f}", 
                 delta=f"{delta_sign}{impact_change:.2f}",
                 delta_color=delta_color)
        st.caption(f"Total MPG: {team1_total_mpg:.1f}/240")
    
    with metric_col2:
        impact_change = team2_impact - original_team2_impact
        delta_sign = "+" if impact_change > 0 else ""
        delta_color = "normal" if impact_change >= 0 else "inverse"
        st.metric(f"{team2} Total Impact", 
                 f"{team2_impact:.2f}",
                 delta=f"{delta_sign}{impact_change:.2f}",
                 delta_color=delta_color)
        st.caption(f"Total MPG: {team2_total_mpg:.1f}/240")
    
    with metric_col3:
        original_advantage = original_team1_impact - original_team2_impact
        advantage_change = advantage - original_advantage
        delta_sign = "+" if advantage_change > 0 else ""
        delta_color = "normal" if advantage_change >= 0 else "inverse"
        st.metric("Projected Advantage", 
                 f"{advantage:.2f}",
                 delta=f"{delta_sign}{advantage_change:.2f}",
                 delta_color=delta_color)

with col2:
    st.subheader("📊 Team Impact Comparison")
    
    # Create comparison data
    impact_data = pd.DataFrame({
        'Team': [team1, team2],
        'Impact': [team1_impact, team2_impact]
    })
    
    st.bar_chart(impact_data.set_index('Team'))

with col3:
    st.subheader("🎯 Prediction")
    
    if advantage > 3:
        st.success(f"**{team1} favored** by {advantage:.2f} points")
    elif advantage < -3:
        st.error(f"**{team2} favored** by {abs(advantage):.2f} points")
    else:
        st.info("**Close matchup** - within 3 points")

# Show team rosters with changes
st.markdown("---")
st.subheader("👥 Team Rosters After Injury Adjustments")

col1, col2 = st.columns(2)

with col1:
    st.write(f"### {team1} Roster ({len(team1_data)} players)")
    
    if team1_removed:
        st.warning(f"**Removed:** {', '.join(team1_removed)}")
    
    # Display team 1 players sorted by Impact
    team1_display = team1_data.sort_values('Impact', ascending=False)[['Player', 'MPG', 'BPM', 'Impact']].copy()
    
    # Add indicators for players who gained minutes
    if team1_removed:
        for idx, row in team1_display.iterrows():
            player = row['Player']
            original_mpg = original_team1_data[original_team1_data['Player'] == player]['MPG'].values
            if len(original_mpg) > 0:
                mpg_change = row['MPG'] - original_mpg[0]
                if mpg_change > 0.1:  # Only show if change is significant
                    team1_display.at[idx, 'MPG'] = f"{row['MPG']:.1f} (+{mpg_change:.1f})"
    
    st.dataframe(team1_display, use_container_width=True)

with col2:
    st.write(f"### {team2} Roster ({len(team2_data)} players)")
    
    if team2_removed:
        st.warning(f"**Removed:** {', '.join(team2_removed)}")
    
    # Display team 2 players sorted by Impact
    team2_display = team2_data.sort_values('Impact', ascending=False)[['Player', 'MPG', 'BPM', 'Impact']].copy()
    
    # Add indicators for players who gained minutes
    if team2_removed:
        for idx, row in team2_display.iterrows():
            player = row['Player']
            original_mpg = original_team2_data[original_team2_data['Player'] == player]['MPG'].values
            if len(original_mpg) > 0:
                mpg_change = row['MPG'] - original_mpg[0]
                if mpg_change > 0.1:  # Only show if change is significant
                    team2_display.at[idx, 'MPG'] = f"{row['MPG']:.1f} (+{mpg_change:.1f})"
    
    st.dataframe(team2_display, use_container_width=True)

# Show minute distribution
st.markdown("---")
st.subheader("⏱️ Minute Distribution")

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    st.write(f"**{team1} Minute Allocation**")
    if len(team1_data) > 0:
        team1_minutes = team1_data[['Player', 'MPG']].sort_values('MPG', ascending=False)
        team1_minutes['Percentage'] = (team1_minutes['MPG'] / TOTAL_TEAM_MINUTES * 100).round(1)
        team1_minutes['MPG'] = team1_minutes['MPG'].round(1)
        st.dataframe(team1_minutes, use_container_width=True)

with dist_col2:
    st.write(f"**{team2} Minute Allocation**")
    if len(team2_data) > 0:
        team2_minutes = team2_data[['Player', 'MPG']].sort_values('MPG', ascending=False)
        team2_minutes['Percentage'] = (team2_minutes['MPG'] / TOTAL_TEAM_MINUTES * 100).round(1)
        team2_minutes['MPG'] = team2_minutes['MPG'].round(1)
        st.dataframe(team2_minutes, use_container_width=True)

# Combined player view
st.markdown("---")
st.subheader("🔍 Combined Player View")

sort_col1, sort_col2 = st.columns([1, 2])
with sort_col1:
    sort_by = st.selectbox(
        "Sort all players by:",
        ['Team', 'Player', 'Impact', 'BPM', 'MPG', 'G']
    )
with sort_col2:
    sort_order = st.radio(
        "Sort order:",
        ['Descending', 'Ascending'],
        horizontal=True,
        key='combined_sort'
    )

# Combine data for display
combined_data = pd.concat([team1_data, team2_data])

# Apply sorting
if len(combined_data) > 0:
    sorted_data = combined_data.sort_values(
        by=sort_by,
        ascending=(sort_order == 'Ascending')
    )
    
    # Display the table
    display_columns = ['Team', 'Player', 'G', 'MPG', 'BPM', 'Impact']
    st.dataframe(
        sorted_data[display_columns].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "G": st.column_config.NumberColumn("Games", format="%d"),
            "MPG": st.column_config.NumberColumn("MPG", format="%.1f"),
            "BPM": st.column_config.NumberColumn("BPM", format="%.1f"),
            "Impact": st.column_config.NumberColumn("Impact", format="%.3f")
        }
    )

# Summary stats
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Players", len(combined_data) if 'combined_data' in locals() else 0)
with col2:
    st.metric("Removed Players", len(removed_players))
with col3:
    st.metric(f"{team1} Active", len(team1_data))
with col4:
    st.metric(f"{team2} Active", len(team2_data))

# Data source and explanation
st.markdown("---")
st.caption("📊 **Data sourced from Basketball-Reference.com**")
st.caption("📈 **Impact Formula:** (BPM/100) × Normalized MPG × 2.083")
st.caption("⏱️ **Minute Normalization:** Each player's minutes scaled to percentage of 240 total team minutes")
st.caption("🔄 **Injury Logic:** Removed players' minutes redistributed proportionally to remaining teammates")
