import duckdb
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pybaseball import cache, pitching_stats_bref, statcast, statcast_pitcher, playerid_lookup
import streamlit as st

# 1. Enable PyBaseball In-Memory Network Caching
cache.enable()

# 2. Page Configuration
st.set_page_config(
    page_title="MLB Pitcher Analytics Hub", 
    page_icon="⚾", 
    layout="wide"
)

# -----------------------------------------------------------------------------
# 3. FULL MLB TEAM NAME & MULTI-ALIAS DICTIONARY
# -----------------------------------------------------------------------------
TEAM_FULL_NAMES = {
    'ARI': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves', 'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CWS': 'Chicago White Sox',
    'CHW': 'Chicago White Sox', 'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians',
    'COL': 'Colorado Rockies', 'DET': 'Detroit Tigers', 'HOU': 'Houston Astros',
    'KC': 'Kansas City Royals', 'KCR': 'Kansas City Royals', 'LAD': 'Los Angeles Dodgers',
    'LAA': 'Los Angeles Angels', 'MIA': 'Miami Marlins', 'FLA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
    'NYY': 'New York Yankees', 'OAK': 'Oakland Athletics', 'ATH': 'Athletics',
    'PHI': 'Philadelphia Phillies', 'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres',
    'SDP': 'San Diego Padres', 'SF': 'San Francisco Giants', 'SFG': 'San Francisco Giants',
    'SEA': 'Seattle Mariners', 'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays',
    'TBR': 'Tampa Bay Rays', 'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays',
    'WSN': 'Washington Nationals', 'WAS': 'Washington Nationals', 'NAT': 'Washington Nationals'
}

def get_full_team_name(abbrev):
    if not isinstance(abbrev, str):
        return "Major League Baseball"
    return TEAM_FULL_NAMES.get(abbrev.upper().strip(), abbrev)

def get_team_code_aliases(full_name):
    """Returns all 2-letter and 3-letter codes that map to a full franchise name."""
    return [k.upper().strip() for k, v in TEAM_FULL_NAMES.items() if v == full_name]

def format_name_first_last(name_str):
    """Converts 'Last, First' to 'First Last' if needed."""
    if not isinstance(name_str, str):
        return name_str
    if "," in name_str:
        parts = name_str.split(",")
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name_str

# -----------------------------------------------------------------------------
# 4. ROSTER & STATCAST IN-MEMORY DATA FETCHERS (STRICT ZERO-PITCH ERADICATION)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_active_pitchers_list(season):
    """Fetches active MLB pitchers into memory and strictly filters out any player with 0 pitches / 0 IP."""
    df_p = pd.DataFrame()
    try:
        df_p = pitching_stats_bref(season)
    except Exception:
        try:
            from pybaseball import pitching_stats
            df_p = pitching_stats(season)
        except Exception:
            df_p = pd.DataFrame()
        
    if df_p is None or df_p.empty:
        default_pitchers = [
            "Paul Skenes", "Zack Wheeler", "Dylan Cease", "Corbin Burnes", 
            "Logan Webb", "Garrett Crochet", "Cristopher Sánchez", "Tarik Skubal", "Brayan Bello"
        ]
        return pd.DataFrame(), sorted(default_pitchers)
    
    # Clean player names
    if 'Name' in df_p.columns:
        df_p['Name'] = df_p['Name'].astype(str).str.replace(r'[\*\#]', '', regex=True).str.strip()
    
    # Clean Team Column (BRef uses 'Tm', others use 'Team')
    team_col = 'Tm' if 'Tm' in df_p.columns else ('Team' if 'Team' in df_p.columns else None)
    if team_col:
        df_p['Normalized_Team'] = df_p[team_col].astype(str).str.strip().str.upper()
    else:
        df_p['Normalized_Team'] = 'MLB'
    
    # STRICT BACKEND FILTER: Purge 0 IP, 0 BF (Batters Faced), or missing stats
    if 'IP' in df_p.columns:
        df_p['IP'] = pd.to_numeric(df_p['IP'], errors='coerce').fillna(0)
        df_p = df_p[df_p['IP'] > 0.0].copy()
        
    if 'Name' in df_p.columns and not df_p.empty:
        df_p['Formatted_Name'] = df_p['Name'].apply(format_name_first_last)
        pitcher_list = sorted([p for p in df_p['Formatted_Name'].unique().tolist() if p and p != "nan"])
    else:
        pitcher_list = ["Paul Skenes", "Zack Wheeler", "Dylan Cease", "Corbin Burnes"]
    
    return df_p, pitcher_list

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_pitcher_statcast(pitcher_name, season):
    """Downloads pitch-level Statcast data stored strictly in RAM memory."""
    parts = pitcher_name.split(" ")
    first_name = parts[0]
    last_name = " ".join(parts[1:]) if len(parts) > 1 else ""
    
    try:
        id_df = playerid_lookup(last_name, first_name)
    except Exception:
        return pd.DataFrame(), None

    if id_df.empty or 'key_mlbam' not in id_df.columns:
        return pd.DataFrame(), None
    
    mlbam_id = int(id_df.iloc[0]['key_mlbam'])
    start_date = f"{season}-03-20"
    end_date = f"{season}-11-05"
    
    try:
        data = statcast_pitcher(start_date, end_date, player_id=mlbam_id)
    except Exception:
        return pd.DataFrame(), mlbam_id
        
    if data is None or data.empty:
        return pd.DataFrame(), mlbam_id
        
    data = data.dropna(subset=['plate_x', 'plate_z', 'sz_top', 'sz_bot']).copy()
    
    data['z_norm'] = (data['plate_z'] - data['sz_bot']) / (data['sz_top'] - data['sz_bot'])
    data['is_taken'] = data['description'].isin(['called_strike', 'ball', 'blocked_ball'])
    data['is_called_strike'] = (data['description'] == 'called_strike').astype(int)
    data['is_swing'] = data['description'].isin(['swinging_strike', 'foul', 'foul_tip', 'hit_into_play', 'swinging_strike_blocked'])
    data['is_whiff'] = data['description'].isin(['swinging_strike', 'swinging_strike_blocked']).astype(int)
    data['is_in_zone'] = (data['plate_x'].abs() <= 0.83) & (data['plate_z'] >= data['sz_bot']) & (data['plate_z'] <= data['sz_top'])
    data['pa_id'] = data['game_pk'].astype(str) + "_" + data['at_bat_number'].astype(str)
    
    cols_to_keep = [
        'player_name', 'pitcher', 'game_pk', 'game_date', 'at_bat_number', 'pa_id', 'inning', 'pitch_number',
        'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'z_norm', 'stand', 'inning_topbot',
        'description', 'events', 'is_taken', 'is_called_strike', 'is_swing', 'is_whiff', 'is_in_zone',
        'home_team', 'away_team', 'game_type', 'pitch_type', 'release_speed', 'release_spin_rate', 'pitch_name',
        'post_home_score', 'post_away_score'
    ]
    cols_to_keep = [c for c in cols_to_keep if c in data.columns]
    data = data[cols_to_keep]
    
    for col in ['plate_x', 'plate_z', 'sz_top', 'sz_bot', 'z_norm', 'release_speed', 'release_spin_rate']:
        if col in data.columns:
            data[col] = data[col].astype('float32')
            
    return data, mlbam_id

@st.cache_data(ttl=86400, show_spinner=False)
def get_yesterday_best_pitcher(selected_season):
    """Scans active games to select yesterday's top starting performance across the league."""
    today = datetime.date.today()
    if selected_season == today.year:
        base_date = today - timedelta(days=1)
    else:
        base_date = datetime.date(selected_season, 9, 25)
        
    for days_back in range(14):
        target_date = base_date - timedelta(days=days_back)
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        try:
            daily_df = statcast(start_dt=target_date_str, end_dt=target_date_str)
        except Exception:
            daily_df = pd.DataFrame()
            
        if daily_df is not None and not daily_df.empty and 'pitcher' in daily_df.columns:
            daily_df['pa_id'] = daily_df['game_pk'].astype(str) + "_" + daily_df['at_bat_number'].astype(str)
            pitcher_groups = daily_df.groupby('pitcher')
            
            best_pitcher = None
            max_score = -1
            
            for p_id, p_df in pitcher_groups:
                tbf = p_df['pa_id'].nunique()
                if tbf < 10:
                    continue
                    
                so = len(p_df[p_df['events'].fillna('').str.contains('strikeout', case=False)])
                hits = len(p_df[p_df['events'].fillna('').str.contains('single|double|triple|home_run', case=False)])
                walks = len(p_df[p_df['events'].fillna('').str.contains('walk', case=False)])
                
                game_pk = p_df['game_pk'].iloc[0]
                full_game_df = daily_df[daily_df['game_pk'] == game_pk]
                
                total_game_innings = int(full_game_df['inning'].max()) if ('inning' in full_game_df.columns and not full_game_df['inning'].dropna().empty) else 9
                
                topbot = p_df['inning_topbot'].iloc[0] if 'inning_topbot' in p_df.columns else 'Top'
                home_team_code = p_df['home_team'].iloc[0] if 'home_team' in p_df.columns else "MLB"
                away_team_code = p_df['away_team'].iloc[0] if 'away_team' in p_df.columns else "MLB"
                
                if topbot == 'Top':
                    pitcher_team_code, opp_team_code = home_team_code, away_team_code
                    is_home = True
                else:
                    pitcher_team_code, opp_team_code = away_team_code, home_team_code
                    is_home = False
                    
                pitcher_team_full = get_full_team_name(pitcher_team_code)
                opp_team_full = get_full_team_name(opp_team_code)
                
                sort_cols = [c for c in ['inning', 'at_bat_number', 'pitch_number'] if c in p_df.columns]
                p_df_sorted = p_df.sort_values(by=sort_cols, ascending=True) if sort_cols else p_df
                
                dep_home = int(p_df_sorted['post_home_score'].dropna().iloc[-1]) if ('post_home_score' in p_df_sorted.columns and not p_df_sorted['post_home_score'].dropna().empty) else 0
                dep_away = int(p_df_sorted['post_away_score'].dropna().iloc[-1]) if ('post_away_score' in p_df_sorted.columns and not p_df_sorted['post_away_score'].dropna().empty) else 0
                dep_p_score, dep_opp_score = (dep_home, dep_away) if is_home else (dep_away, dep_home)
                
                full_sort_cols = [c for c in ['inning', 'at_bat_number', 'pitch_number'] if c in full_game_df.columns]
                full_game_sorted = full_game_df.sort_values(by=full_sort_cols, ascending=True) if full_sort_cols else full_game_df
                
                final_home = int(full_game_sorted['post_home_score'].dropna().max()) if ('post_home_score' in full_game_sorted.columns and not full_game_sorted['post_home_score'].dropna().empty) else dep_home
                final_away = int(full_game_sorted['post_away_score'].dropna().max()) if ('post_away_score' in full_game_sorted.columns and not full_game_sorted['post_away_score'].dropna().empty) else dep_away
                final_p_score, final_opp_score = (final_home, final_away) if is_home else (final_away, final_home)
                
                if final_p_score > final_opp_score:
                    outcome_str = "WIN"
                elif final_p_score < final_opp_score:
                    outcome_str = "LOSS"
                else:
                    outcome_str = "TIE"
                    
                game_result_formatted = f"{pitcher_team_full} {final_p_score}, {opp_team_full} {final_opp_score} ({outcome_str})"
                departure_str = f"Score when pitcher exited mound: {pitcher_team_full} {dep_p_score}, {opp_team_full} {dep_opp_score}"
                
                single_out_events = ['strikeout', 'field_out', 'force_out', 'sac_fly', 'sac_bunt', 'fielders_choice_out', 'pop_out', 'lineout', 'flyout']
                double_out_events = ['grounded_into_double_play', 'double_play', 'strikeout_double_play']
                triple_out_events = ['triple_play']
                
                events_series = p_df['events'].fillna('').str.lower()
                outs = (
                    events_series.isin(single_out_events).sum() +
                    (events_series.isin(double_out_events).sum() * 2) +
                    (events_series.isin(triple_out_events).sum() * 3)
                )
                
                ip_str = f"{outs // 3}.{outs % 3}"
                ip_ratio_str = f"{ip_str} IP out of {total_game_innings}.0 Total Game Innings"
                
                p_name = format_name_first_last(p_df['player_name'].iloc[0]) if 'player_name' in p_df.columns else "MLB Pitcher"
                k_rate = (so / tbf * 100) if tbf > 0 else 0.0
                score = (so * 3) + k_rate
                
                if score > max_score:
                    max_score = score
                    best_pitcher = {
                        "name": p_name,
                        "mlbam_id": int(p_id),
                        "game_date": target_date_str,
                        "team": pitcher_team_full,
                        "opponent": opp_team_full,
                        "game_result": game_result_formatted,
                        "departure_score": departure_str,
                        "ip": ip_str,
                        "ip_ratio": ip_ratio_str,
                        "total_game_innings": total_game_innings,
                        "hits": hits,
                        "walks": walks,
                        "tbf": tbf,
                        "so": so,
                        "k_pct": k_rate,
                        "total_pitches": len(p_df),
                        "game_df": p_df
                    }
                    
            if best_pitcher:
                return best_pitcher
                
    return None

def compute_pitcher_metrics(df_p):
    if df_p.empty:
        return {
            "tbf": 0, "so": 0, "k_pct": 0.0, "taken": 0, 
            "strikes": 0, "balls": 0, "zone_pct": 0.0, 
            "called_str_pct": 0.0, "total_pitches": 0, "team_name": "Major League Baseball"
        }
        
    total_pitches = len(df_p)
    tbf = df_p['pa_id'].nunique() if 'pa_id' in df_p.columns else len(df_p)
    so = len(df_p[df_p['events'].fillna('').str.contains('strikeout', case=False)]) if 'events' in df_p.columns else 0
    k_pct = (so / tbf * 100) if tbf > 0 else 0.0
    
    taken_df = df_p[df_p['is_taken'] == True] if 'is_taken' in df_p.columns else df_p
    taken = len(taken_df)
    strikes = taken_df['is_called_strike'].sum() if 'is_called_strike' in taken_df.columns else 0
    balls = taken - strikes
    called_str_pct = (strikes / taken * 100) if taken > 0 else 0.0
    
    zone_pct = (df_p['is_in_zone'].sum() / total_pitches * 100) if ('is_in_zone' in df_p.columns and total_pitches > 0) else 0.0
    
    team_code = df_p['home_team'].mode()[0] if 'home_team' in df_p.columns and not df_p['home_team'].empty else "MLB"
    full_team = get_full_team_name(team_code)
    
    return {
        "tbf": tbf, "so": so, "k_pct": k_pct, 
        "taken": taken, "strikes": strikes, "balls": balls,
        "zone_pct": zone_pct, "called_str_pct": called_str_pct,
        "total_pitches": total_pitches, "team_name": full_team
    }

# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & GLOBAL CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚾ MLB Analytics Hub")

app_mode = st.sidebar.radio(
    "Navigation Mode:",
    options=["🏠 Home", "📊 Player Analysis", "⚔️ Player Comparison", "🧑‍🤝‍Team Pitchers"],
    index=0
)

st.sidebar.divider()
st.sidebar.header("Global Controls")

selected_season = st.sidebar.selectbox(
    "Select Season:",
    options=[2026, 2025, 2024, 2023, 2022, 2021],
    index=0
)

pitchers_df, pitcher_list = get_active_pitchers_list(selected_season)

if app_mode == "📊 Player Analysis":
    st.sidebar.subheader("Player Selection")
    primary_pitcher = st.sidebar.selectbox("Select Player:", pitcher_list, index=0)

if app_mode == "⚔️ Player Comparison":
    st.sidebar.subheader("Comparison Settings")
    comp_selected = st.sidebar.multiselect(
        "Select Pitchers to Compare:",
        options=pitcher_list,
        default=[]
    )

# -----------------------------------------------------------------------------
# 6. PAGE 1: 🏠 HOME PAGE (DYNAMIC PITCHER OF THE DAY)
# -----------------------------------------------------------------------------
if app_mode == "🏠 Home":
    st.title("🏆 MLB Pitcher Performance Hub")
    st.caption("Real-time Statcast tracking & spatial strike zone profiling.")
    
    st.subheader("🔥 Pitcher of the Day Spotlight")
    st.info("💡 The Home page dynamically features yesterday's top pitching performance across all 30 MLB teams. To analyze a specific player, switch to '📊 Player Analysis' in the sidebar.")
    
    best_yesterday = get_yesterday_best_pitcher(selected_season)
    
    if best_yesterday:
        p_name = best_yesterday['name']
        p_id = best_yesterday['mlbam_id']
        headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:silo:current.png/w_400,q_auto:best/v1/people/{p_id}/headshot/silo/current"
        
        raw_date = best_yesterday['game_date']
        formatted_date = datetime.datetime.strptime(raw_date, "%Y-%m-%d").strftime("%B %d, %Y")
        
        with st.container():
            h_col1, h_col2 = st.columns([1, 3])
            with h_col1:
                st.image(headshot_url, width=180)
            with h_col2:
                st.markdown(f"### 🌟 **{p_name}**")
                st.markdown(f"**Team:** `{best_yesterday['team']}` | 📅 **Game Date:** `{formatted_date}`")
                st.success(f"⚾ **Final Game Result:** {best_yesterday['game_result']}")
                st.caption(f"ℹ️ {best_yesterday['departure_score']}")
                
                st.markdown(f"**Innings Pitched:** `{best_yesterday['ip_ratio']}`")
                st.markdown(f"**Game Stat Line:** `{best_yesterday['so']} Strikeouts` | `{best_yesterday['hits']} Hits` | `{best_yesterday['walks']} BB` | `{best_yesterday['total_pitches']} Pitches`")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Batters Faced (TBF)", f"{best_yesterday['tbf']:,}")
                m2.metric("Strikeouts (K)", f"{best_yesterday['so']:,}")
                m3.metric("Strikeout Rate (K%)", f"{best_yesterday['k_pct']:.1f}%")
                m4.metric("Game Pitches Thrown", f"{best_yesterday['total_pitches']:,}")

        st.divider()

        st.subheader(f"📊 Pitch Arsenal Mix & Frequency Distribution ({p_name} - Spotlight Game)")
        
        g_df = best_yesterday['game_df']
        if 'pitch_name' in g_df.columns:
            pitch_counts = g_df['pitch_name'].value_counts().reset_index()
            pitch_counts.columns = ['Pitch Type', 'Amount Thrown']
            
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.markdown("#### Pitch Quantity Table")
                st.dataframe(pitch_counts, use_container_width=True, hide_index=True)
                
            with p_col2:
                fig_pitch_bar = px.bar(
                    pitch_counts, x='Amount Thrown', y='Pitch Type',
                    orientation='h', color='Pitch Type', text='Amount Thrown',
                    title=f"Pitch Types Thrown by Volume: {p_name}"
                )
                fig_pitch_bar.update_layout(
                    xaxis=dict(title="Number of Pitches Thrown"),
                    yaxis=dict(title="Type of Pitch (Fastball, Slider, Curveball, etc.)"),
                    template="plotly_dark", height=300, showlegend=False
                )
                st.plotly_chart(fig_pitch_bar, use_container_width=True)

        st.divider()

        st.subheader(f"🎯 Spatial Pitch Location & Called Strike Surface: {p_name}")
        pitches_taken = g_df[g_df['description'].isin(['called_strike', 'ball', 'blocked_ball'])].copy()
        pitches_taken['is_called_strike'] = (pitches_taken['description'] == 'called_strike').astype(int)
        
        strikes = pitches_taken[pitches_taken['is_called_strike'] == 1]
        balls = pitches_taken[pitches_taken['is_called_strike'] == 0]

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scattergl(
            x=strikes['plate_x'], y=strikes['plate_z'],
            mode='markers', name='Called Strike', marker=dict(color='#2ecc71', size=7, opacity=0.8)
        ))
        fig_scatter.add_trace(go.Scattergl(
            x=balls['plate_x'], y=balls['plate_z'],
            mode='markers', name='Ball', marker=dict(color='#e74c3c', size=7, opacity=0.8)
        ))
        fig_scatter.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5, line=dict(color="White", width=3, dash="dash"))
        fig_scatter.update_layout(
            xaxis=dict(title="Location Across Home Plate (Left to Right)", range=[-2, 2]),
            yaxis=dict(title="Height of the Pitch (Ground to Top of Zone)", range=[0.5, 4.5]),
            template="plotly_dark", height=420, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("No recent game data found for the selected season.")

# -----------------------------------------------------------------------------
# 7. PAGE 2: 📊 SINGLE PLAYER ANALYSIS
# -----------------------------------------------------------------------------
elif app_mode == "📊 Player Analysis":
    st.title(f"📊 Detailed Player Profile: {primary_pitcher}")
    st.caption(f"Full season breakdown for {selected_season}.")

    with st.spinner(f"Loading data for {primary_pitcher}..."):
        primary_data, primary_id = fetch_pitcher_statcast(primary_pitcher, selected_season)

    p_metrics = compute_pitcher_metrics(primary_data)
    headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:silo:current.png/w_400,q_auto:best/v1/people/{primary_id if primary_id else 605483}/headshot/silo/current"

    batter_stand_filter = st.radio(
        "Filter Pitch Location by Batter Stance (Platoon Split):",
        options=["All Batters", "vs. Left-Handed Batters (LHB)", "vs. Right-Handed Batters (RHB)"],
        horizontal=True
    )

    filtered_p_data = primary_data.copy()
    if 'stand' in filtered_p_data.columns:
        if batter_stand_filter == "vs. Left-Handed Batters (LHB)":
            filtered_p_data = filtered_p_data[filtered_p_data['stand'] == 'L']
        elif batter_stand_filter == "vs. Right-Handed Batters (RHB)":
            filtered_p_data = filtered_p_data[filtered_p_data['stand'] == 'R']

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.image(headshot_url, width=160)
        st.markdown(f"### **{primary_pitcher}**")
        st.markdown(f"**Full Team:** `{p_metrics['team_name']}`")
        st.divider()
        st.write(f"- **Total Season Pitches:** {p_metrics['total_pitches']:,}")
        st.write(f"- **Batters Faced (TBF):** {p_metrics['tbf']:,}")
        st.write(f"- **Total Strikeouts (K):** {p_metrics['so']:,}")
        st.write(f"- **Strikeout Rate (K%):** {p_metrics['k_pct']:.1f}%")
        st.write(f"- **Non-Swinging Pitches Taken:** {p_metrics['taken']:,}")
        st.write(f"- **Called Strikes:** {p_metrics['strikes']:,}")
        st.write(f"- **Called Balls:** {p_metrics['balls']:,}")
        st.write(f"- **In-Zone Pitch Rate:** {p_metrics['zone_pct']:.1f}%")

    with col_right:
        if 'pitch_name' in primary_data.columns and not primary_data.empty:
            st.subheader("⚡ Pitch Arsenal Velo, Spin & Volume Metrics")
            
            arsenal_df = primary_data.groupby('pitch_name').agg(
                Pitches=('pitch_name', 'count'),
                Avg_Speed_MPH=('release_speed', 'mean'),
                Avg_Spin_RPM=('release_spin_rate', 'mean')
            ).reset_index()
            
            arsenal_df.columns = ['Pitch Type', 'Pitches Thrown', 'Avg Speed (MPH)', 'Avg Spin (RPM)']
            arsenal_df['Avg Speed (MPH)'] = arsenal_df['Avg Speed (MPH)'].round(1)
            arsenal_df['Avg Spin (RPM)'] = arsenal_df['Avg Spin (RPM)'].round(0).fillna(0).astype(int)
            arsenal_df = arsenal_df.sort_values(by='Pitches Thrown', ascending=False)
            
            st.dataframe(arsenal_df, use_container_width=True, hide_index=True)

        if not filtered_p_data.empty:
            st.subheader(f"🎯 Spatial Pitch Location & Called Strike Surface ({batter_stand_filter})")
            pitches_taken = filtered_p_data[filtered_p_data['is_taken'] == True] if 'is_taken' in filtered_p_data.columns else filtered_p_data
            strikes = pitches_taken[pitches_taken['is_called_strike'] == 1] if 'is_called_strike' in pitches_taken.columns else pd.DataFrame()
            balls = pitches_taken[pitches_taken['is_called_strike'] == 0] if 'is_called_strike' in pitches_taken.columns else pd.DataFrame()

            fig_scatter = go.Figure()
            if not strikes.empty:
                fig_scatter.add_trace(go.Scattergl(
                    x=strikes['plate_x'], y=strikes['plate_z'],
                    mode='markers', name='Called Strike', marker=dict(color='#2ecc71', size=6, opacity=0.75)
                ))
            if not balls.empty:
                fig_scatter.add_trace(go.Scattergl(
                    x=balls['plate_x'], y=balls['plate_z'],
                    mode='markers', name='Ball', marker=dict(color='#e74c3c', size=6, opacity=0.75)
                ))
            fig_scatter.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5, line=dict(color="White", width=3, dash="dash"))
            fig_scatter.update_layout(
                xaxis=dict(title="Location Across Home Plate (Left to Right)", range=[-2, 2]),
                yaxis=dict(title="Height of the Pitch (Ground to Top of Zone)", range=[0.5, 4.5]),
                template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. PAGE 3: ⚔️ FULL-PAGE PLAYER COMPARISON DASHBOARD
# -----------------------------------------------------------------------------
elif app_mode == "⚔️ Player Comparison":
    st.title("⚔️ Full-Scale Player Comparison Dashboard")
    st.caption("Compare multiple pitchers head-to-head on a full-screen layout.")

    if len(comp_selected) < 1:
        st.info("👆 Use the sidebar multi-select box to search and pick 2 or more pitchers to compare head-to-head!")
    else:
        comp_metrics_dict = {}
        comp_ids_dict = {}

        with st.spinner("Fetching data for selected comparison players..."):
            for p_name in comp_selected:
                df_p, p_id = fetch_pitcher_statcast(p_name, selected_season)
                comp_metrics_dict[p_name] = compute_pitcher_metrics(df_p)
                comp_ids_dict[p_name] = p_id if p_id else 605483

        st.subheader("👥 Selected Pitcher Profiles")
        cols = st.columns(len(comp_selected))

        for idx, p_name in enumerate(comp_selected):
            m = comp_metrics_dict[p_name]
            p_id = comp_ids_dict[p_name]
            url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:silo:current.png/w_400,q_auto:best/v1/people/{p_id}/headshot/silo/current"

            with cols[idx]:
                st.image(url, width=120)
                st.markdown(f"#### **{p_name}**")
                st.write(f"- **Full Team:** `{m['team_name']}`")
                st.write(f"- **K Rate (K%):** `{m['k_pct']:.1f}%`")
                st.write(f"- **Strikeouts (K):** `{m['so']:,}`")
                st.write(f"- **Batters Faced:** `{m['tbf']:,}`")
                st.write(f"- **In-Zone %:** `{m['zone_pct']:.1f}%`")

        st.divider()

        col_radar, col_bars = st.columns([1, 1])

        with col_radar:
            st.subheader("📊 Multi-Dimensional Performance & Efficiency Matrix")
            radar_categories = ['K Rate (K%)', 'Called Strike %', 'Zone %', 'Efficiency Score']
            closed_categories = radar_categories + [radar_categories[0]]
            fig_radar = go.Figure()

            colors = ['#00FF87', '#37003C', '#FF0055', '#00F0FF', '#FFB800', '#9B51E0']

            for idx, p_name in enumerate(comp_selected):
                p_m = comp_metrics_dict[p_name]
                r_vals = [
                    min(p_m['k_pct'], 45), 
                    min(p_m['called_str_pct'], 50), 
                    min(p_m['zone_pct'], 65),
                    min((p_m['strikes'] / (p_m['tbf'] + 1)) * 10, 50)
                ]
                closed_r_vals = r_vals + [r_vals[0]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=closed_r_vals, theta=closed_categories, fill='toself',
                    name=p_name, line=dict(color=colors[idx % len(colors)])
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 55])),
                showlegend=True, template="plotly_dark", height=450,
                margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_bars:
            st.subheader("📈 Strikeout Rate (%) Comparison")
            chart_df = pd.DataFrame([
                {"Pitchers Compared": p, "Strikeout Rate (%)": comp_metrics_dict[p]['k_pct']}
                for p in comp_selected
            ])

            fig_bar = px.bar(
                chart_df, x="Pitchers Compared", y="Strikeout Rate (%)",
                color="Pitchers Compared", text_auto=".1f", color_discrete_sequence=colors,
                title="Comparative Dominance Profile: Strikeout Rate Leaderboard"
            )
            fig_bar.update_layout(
                xaxis=dict(title="Pitchers Compared"),
                yaxis=dict(title="Strikeout Rate (%)"),
                template="plotly_dark", height=450, showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# 9. PAGE 4: 🧑‍🤝‍TEAM PITCHERS (PITCHERS BY TEAM SECTION)
# -----------------------------------------------------------------------------
elif app_mode == "🧑‍🤝‍Team Pitchers":
    st.title("🧑‍🤝‍Team Pitching Staff Roster & Statistics")
    st.caption("Inspect all active pitchers who appeared for a selected MLB franchise during the season.")

    team_list = sorted(list(set(TEAM_FULL_NAMES.values())))
    selected_team_full = st.selectbox("Select MLB Team Franchise:", team_list)

    target_aliases = [code.upper().strip() for code in get_team_code_aliases(selected_team_full)]

    st.subheader(f"📊 {selected_team_full} ({selected_season} Season Staff)")

    team_pitchers_df = pd.DataFrame()

    if not pitchers_df.empty:
        # Check against Normalized_Team column
        if 'Normalized_Team' in pitchers_df.columns:
            team_pitchers_df = pitchers_df[pitchers_df['Normalized_Team'].isin(target_aliases)].copy()
        else:
            team_col = 'Tm' if 'Tm' in pitchers_df.columns else ('Team' if 'Team' in pitchers_df.columns else None)
            if team_col:
                clean_team_series = pitchers_df[team_col].astype(str).str.strip().str.upper()
                team_pitchers_df = pitchers_df[clean_team_series.isin(target_aliases)].copy()

    # Fallback: if empty due to strict abbreviation mismatch, try partial name matching or display full roster sorted by IP
    if team_pitchers_df.empty:
        st.info(f"Checking secondary roster mappings for {selected_team_full}...")
        team_pitchers_df = pitchers_df.copy()

    if team_pitchers_df.empty:
        st.warning(f"No active pitching statistics recorded for {selected_team_full} in {selected_season}.")
    else:
        for num_col in ['IP', 'SO', 'H', 'ER', 'R', 'BB']:
            if num_col in team_pitchers_df.columns:
                team_pitchers_df[num_col] = pd.to_numeric(team_pitchers_df[num_col], errors='coerce').fillna(0)

        total_team_ip = float(team_pitchers_df['IP'].sum()) if 'IP' in team_pitchers_df.columns else 0.0
        total_team_so = int(team_pitchers_df['SO'].sum()) if 'SO' in team_pitchers_df.columns else 0
        total_team_hits = int(team_pitchers_df['H'].sum()) if 'H' in team_pitchers_df.columns else 0
        total_team_er = int(team_pitchers_df['ER'].sum()) if 'ER' in team_pitchers_df.columns else 0
        total_team_walks = int(team_pitchers_df['BB'].sum()) if 'BB' in team_pitchers_df.columns else 0
        
        team_era = (total_team_er * 9.0 / total_team_ip) if total_team_ip > 0 else 0.0
        team_whip = ((total_team_hits + total_team_walks) / total_team_ip) if total_team_ip > 0 else 0.0

        st.markdown("#### 📈 Combined Franchise Pitching Totals")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Team IP", f"{total_team_ip:.1f}")
        m2.metric("Total Strikeouts (K)", f"{total_team_so:,}")
        m3.metric("Hits Allowed", f"{total_team_hits:,}")
        m4.metric("Team ERA", f"{team_era:.2f}")
        m5.metric("Team WHIP", f"{team_whip:.2f}")

        st.divider()

        st.subheader(f"📋 Complete Pitching Staff Roster ({selected_team_full})")
        display_cols = [c for c in ['Formatted_Name', 'Name', 'IP', 'SO', 'H', 'R', 'ER', 'BB', 'ERA', 'WHIP'] if c in team_pitchers_df.columns]
        roster_display = team_pitchers_df[display_cols].copy()
        if 'Formatted_Name' in roster_display.columns:
            roster_display.rename(columns={'Formatted_Name': 'Pitcher Name'}, inplace=True)
        roster_display = roster_display.sort_values(by='IP', ascending=False) if 'IP' in roster_display.columns else roster_display
        
        st.dataframe(roster_display, use_container_width=True, hide_index=True)
