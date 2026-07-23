import codecs
import duckdb
import datetime
from collections import Counter
from datetime import timedelta
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pybaseball import cache, pitching_stats_bref, statcast, statcast_pitcher, playerid_lookup
import streamlit as st
import urllib.request
from urllib.error import HTTPError, URLError

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

DEFAULT_MIN_PITCHES = 300


def normalize_app_mode(mode):
    """Normalize sidebar mode values so older or odd character variants still map correctly."""
    if not isinstance(mode, str):
        return "🏠 Home"

    cleaned = mode.strip().replace("", "")
    if not cleaned:
        return "🏠 Home"

    lowered = cleaned.lower()
    if "advanced" in lowered or "scouting" in lowered:
        return "🔬 Advanced Scouting"
    if "leader" in lowered:
        return "📈 Leaderboards"
    if "comparison" in lowered or "compare" in lowered:
        return "⚔️ Player Comparison"
    if "team" in lowered and "pitch" in lowered:
        return "🧑‍🤝‍Team Pitchers"
    if "live" in lowered:
        return "⚡ Live Games"
    if "player" in lowered and "analysis" in lowered:
        return "📊 Player Analysis"
    if "home" in lowered:
        return "🏠 Home"
    return cleaned


def get_full_team_name(abbrev):
    if not isinstance(abbrev, str):
        return "Major League Baseball"
    return TEAM_FULL_NAMES.get(abbrev.upper().strip(), abbrev)

PARK_SPEED_MULTIPLIER = {
    "Colorado Rockies": 1.03,
    "Arizona Diamondbacks": 1.02,
    "San Diego Padres": 0.995,
    "Seattle Mariners": 0.995,
    "Miami Marlins": 0.995,
    "Boston Red Sox": 0.99,
    "New York Yankees": 0.99,
    "Tampa Bay Rays": 0.99,
    "Los Angeles Angels": 0.995,
    "Houston Astros": 0.995,
}

PARK_ZONE_MULTIPLIER = {
    "Colorado Rockies": 0.98,
    "Arizona Diamondbacks": 0.99,
    "San Diego Padres": 1.0,
    "Seattle Mariners": 1.0,
    "Miami Marlins": 1.0,
    "Boston Red Sox": 1.01,
    "New York Yankees": 1.0,
    "Tampa Bay Rays": 1.0,
    "Los Angeles Angels": 1.0,
    "Houston Astros": 1.0,
}

def get_park_multiplier(team_value, metric='speed'):
    team_name = normalize_team_name(team_value)
    if metric == 'zone':
        return PARK_ZONE_MULTIPLIER.get(team_name, 1.0)
    return PARK_SPEED_MULTIPLIER.get(team_name, 1.0)


def compute_park_adjusted_metrics(df_p):
    if df_p.empty:
        return {
            "avg_speed": 0.0,
            "avg_spin": 0.0,
            "zone_pct": 0.0,
            "park_note": "No park data available"
        }

    avg_speed = float(df_p['release_speed'].dropna().mean()) if 'release_speed' in df_p.columns else 0.0
    avg_spin = float(df_p['release_spin_rate'].dropna().mean()) if 'release_spin_rate' in df_p.columns else 0.0
    zone_pct = float(df_p['is_in_zone'].sum() / len(df_p) * 100) if 'is_in_zone' in df_p.columns and len(df_p) > 0 else 0.0

    if 'home_team' in df_p.columns:
        speed_factors = df_p['home_team'].fillna('').map(lambda x: get_park_multiplier(x, 'speed'))
        zone_factors = df_p['home_team'].fillna('').map(lambda x: get_park_multiplier(x, 'zone'))
        if not speed_factors.empty:
            speed_factor = speed_factors.mean()
            avg_speed = round(avg_speed * speed_factor, 1)
            avg_spin = round(avg_spin * speed_factor, 0)
        else:
            avg_speed = round(avg_speed, 1)
            avg_spin = round(avg_spin, 0)
        if not zone_factors.empty:
            zone_pct = round(zone_pct * zone_factors.mean(), 1)
        else:
            zone_pct = round(zone_pct, 1)
        park_note = "Park-adjusted using venue-level factors."
    else:
        avg_speed = round(avg_speed, 1)
        avg_spin = round(avg_spin, 0)
        zone_pct = round(zone_pct, 1)
        park_note = "Park adjustment unavailable without home team data."

    return {
        "avg_speed": avg_speed,
        "avg_spin": avg_spin,
        "zone_pct": zone_pct,
        "park_note": park_note
    }


def normalize_team_name(team_value):
    """Normalize MLB team values to a canonical franchise name."""
    if not isinstance(team_value, str):
        return "Major League Baseball"

    raw_value = team_value.split("<")[0].strip()
    if not raw_value:
        return "Major League Baseball"

    raw_value = raw_value.replace("–", "-").replace("—", "-")
    raw_value = raw_value.split(" (")[0].strip()

    upper_value = raw_value.upper()
    if upper_value in TEAM_FULL_NAMES:
        return TEAM_FULL_NAMES[upper_value]

    for full_name in TEAM_FULL_NAMES.values():
        if full_name.upper() == upper_value:
            return full_name

    shorthand_mappings = {
        "ARIZONA": "Arizona Diamondbacks",
        "ATLANTA": "Atlanta Braves",
        "BALTIMORE": "Baltimore Orioles",
        "BOSTON": "Boston Red Sox",
        "CHICAGO": ["Chicago Cubs", "Chicago White Sox"],
        "CHICAGO CUBS": "Chicago Cubs",
        "CHICAGO WHITE SOX": "Chicago White Sox",
        "CINCINNATI": "Cincinnati Reds",
        "CLEVELAND": "Cleveland Guardians",
        "COLORADO": "Colorado Rockies",
        "DETROIT": "Detroit Tigers",
        "HOUSTON": "Houston Astros",
        "KANSAS CITY": "Kansas City Royals",
        "LOS ANGELES ANGELS": "Los Angeles Angels",
        "LOS ANGELES DODGERS": "Los Angeles Dodgers",
        "MIAMI": "Miami Marlins",
        "MILWAUKEE": "Milwaukee Brewers",
        "MINNESOTA": "Minnesota Twins",
        "NEW YORK": ["New York Mets", "New York Yankees"],
        "NEW YORK METS": "New York Mets",
        "NEW YORK YANKEES": "New York Yankees",
        "OAKLAND": "Oakland Athletics",
        "PHILADELPHIA": "Philadelphia Phillies",
        "PITTSBURGH": "Pittsburgh Pirates",
        "SAN DIEGO": "San Diego Padres",
        "SAN FRANCISCO": "San Francisco Giants",
        "SEATTLE": "Seattle Mariners",
        "ST. LOUIS": "St. Louis Cardinals",
        "TAMPA BAY": "Tampa Bay Rays",
        "TEXAS": "Texas Rangers",
        "TORONTO": "Toronto Blue Jays",
        "WASHINGTON": "Washington Nationals",
    }
    if upper_value in shorthand_mappings:
        mapped_value = shorthand_mappings[upper_value]
        if isinstance(mapped_value, list):
            return mapped_value[0]
        return mapped_value

    for full_name in TEAM_FULL_NAMES.values():
        if full_name.upper() in upper_value:
            return full_name

    return raw_value


def normalize_team_candidates(team_value):
    """Return one or more canonical team names from a raw value such as 'Minnesota,Philadelphia'."""
    if not isinstance(team_value, str):
        return []

    raw_value = team_value.split("<")[0].strip()
    if not raw_value:
        return []

    parts = [part.strip() for part in raw_value.replace(";", ",").split(",") if part.strip()]
    candidates = []
    for part in parts:
        upper_value = part.upper()
        if upper_value in TEAM_FULL_NAMES:
            candidates.append(TEAM_FULL_NAMES[upper_value])
            continue
        for full_name in TEAM_FULL_NAMES.values():
            if full_name.upper() == upper_value:
                candidates.append(full_name)
                break
        else:
            shorthand_mappings = {
                "CHICAGO": ["Chicago Cubs", "Chicago White Sox"],
                "NEW YORK": ["New York Mets", "New York Yankees"],
                "MINNESOTA": ["Minnesota Twins"],
                "PHILADELPHIA": ["Philadelphia Phillies"],
                "MILWAUKEE": ["Milwaukee Brewers"],
                "BOSTON": ["Boston Red Sox"],
                "ATLANTA": ["Atlanta Braves"],
                "SAN FRANCISCO": ["San Francisco Giants"],
                "SAN DIEGO": ["San Diego Padres"],
                "SEATTLE": ["Seattle Mariners"],
                "TAMPA BAY": ["Tampa Bay Rays"],
                "WASHINGTON": ["Washington Nationals"],
            }
            if upper_value in shorthand_mappings:
                candidates.extend(shorthand_mappings[upper_value])
            else:
                normalized = normalize_team_name(part)
                if normalized != "Major League Baseball" and normalized not in candidates:
                    candidates.append(normalized)

    return list(dict.fromkeys(candidates))


def get_team_code_aliases(full_name):
    """Returns all 2-letter and 3-letter codes that map to a full franchise name."""
    return [k.upper().strip() for k, v in TEAM_FULL_NAMES.items() if v == full_name]


def fetch_json(url, timeout=10):
    """Fetch JSON from a URL with a simple timeout and graceful error handling."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def get_live_mlb_schedule(target_date=None):
    """Fetch today's MLB schedule and normalize the games into a scoreboard-friendly dataframe."""
    if target_date is None:
        target_date = datetime.date.today().strftime("%Y-%m-%d")

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date}"
    payload = fetch_json(url)
    if not payload:
        return pd.DataFrame(columns=["game_pk", "home_team", "away_team", "home_score", "away_score", "status", "inning", "outs", "is_live", "venue", "game_time"])

    def _format_inning_label(linescore, status):
        if not linescore:
            return ""
        if isinstance(status, str) and status.lower().startswith("final"):
            return "Final"
        inning_state = linescore.get("inningState") or ""
        inning_ordinal = linescore.get("currentInningOrdinal") or linescore.get("currentInning") or ""
        if inning_state:
            return f"{inning_state} {inning_ordinal}".strip()
        return str(inning_ordinal) if inning_ordinal else ""

    def _is_live_state(status, linescore):
        if not isinstance(status, str):
            status = ""
        lower = status.lower()
        if any(token in lower for token in ["in progress", "live", "warmup", "delayed"]):
            return True
        if linescore.get("currentInning") or linescore.get("currentInningOrdinal"):
            return not lower.startswith("final")
        return False

    games = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("detailedState", "Scheduled")
            linescore = game.get("linescore", {}) or {}
            home_team = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "Unknown")
            away_team = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "Unknown")
            home_score = game.get("teams", {}).get("home", {}).get("score")
            away_score = game.get("teams", {}).get("away", {}).get("score")
            outs = linescore.get("outs") or ""
            inning_label = _format_inning_label(linescore, status)
            is_live = _is_live_state(status, linescore)

            games.append({
                "game_pk": game.get("gamePk"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score if home_score is not None else 0,
                "away_score": away_score if away_score is not None else 0,
                "status": status,
                "inning": inning_label,
                "outs": outs,
                "is_live": is_live,
                "venue": game.get("venue", {}).get("name", ""),
                "game_time": game.get("gameDate", "")
            })

    return pd.DataFrame(games)


def build_live_game_feed(game_pk):
    """Fetch a single game's live feed and return a lightweight structure for display."""
    if not game_pk:
        return None

    payload = fetch_json(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")
    if not payload:
        return None

    live_data = payload.get("liveData", {}) or {}
    boxscore = live_data.get("boxscore", {}) or {}
    teams = boxscore.get("teams", {}) or {}
    plays = live_data.get("plays", {}).get("allPlays", []) or []

    def _pitcher_lookup(team_key):
        team_box = teams.get(team_key, {}) or {}
        pitchers = team_box.get("pitchers", []) or []
        players = team_box.get("players", {}) or {}
        names = []
        for pitcher_id in pitchers[:2]:
            player = players.get(str(pitcher_id), {}) or {}
            person = player.get("person", {}) or {}
            full_name = person.get("fullName")
            if full_name:
                names.append(full_name)
        return names

    live_game = {
        "game_pk": game_pk,
        "game_data": payload.get("gameData", {}),
        "live_data": live_data,
        "boxscore": boxscore,
        "plays": plays,
        "home_pitchers": _pitcher_lookup("home"),
        "away_pitchers": _pitcher_lookup("away"),
    }
    return live_game


def build_live_pitch_log(live_game):
    """Create a compact pitch-by-pitch log for the selected live game."""
    if not live_game:
        return pd.DataFrame(columns=["Inning", "Description", "Count", "Pitch Type", "Velocity", "Pitcher", "Batter"])

    def _normalize_pitch_name(raw_pitch_name):
        if isinstance(raw_pitch_name, dict):
            return str(raw_pitch_name.get("description") or raw_pitch_name.get("code") or "Unknown").strip()
        if isinstance(raw_pitch_name, str):
            return raw_pitch_name.strip() or "Unknown"
        return str(raw_pitch_name).strip() or "Unknown"

    def _extract_count(play):
        about = play.get("about", {}) or {}
        count = about.get("count", {}) or {}
        balls = count.get("balls")
        strikes = count.get("strikes")
        if balls is not None or strikes is not None:
            balls = balls if balls is not None else "?"
            strikes = strikes if strikes is not None else "?"
            return f"{balls}-{strikes}"

        for event in reversed(play.get("playEvents", []) or []):
            event_count = event.get("count") or event.get("about", {}).get("count") or {}
            balls = event_count.get("balls")
            strikes = event_count.get("strikes")
            if balls is not None or strikes is not None:
                balls = balls if balls is not None else "?"
                strikes = strikes if strikes is not None else "?"
                return f"{balls}-{strikes}"

        return "?-?"

    def _normalize_description(raw_description):
        if isinstance(raw_description, dict):
            raw_description = raw_description.get("description") or raw_description.get("code") or "Play"
        elif isinstance(raw_description, list):
            raw_description = " ".join(str(x) for x in raw_description if x)
        else:
            raw_description = str(raw_description)
        text = raw_description.split("\n")[0].strip()
        return text[:120].rstrip() if len(text) > 120 else text

    rows = []
    for play in live_game.get("plays", []) or []:
        about = play.get("about", {}) or {}
        result = play.get("result", {}) or {}
        description = _normalize_description(result.get("description") or about.get("description") or "Play")
        inning = about.get("halfInning") or ""
        inning_label = f"{about.get('inning', '')}{' ' + inning if inning else ''}".strip()
        count_label = _extract_count(play)

        pitch_type = "Unknown"
        velocity = None
        for event in play.get("playEvents", []) or []:
            event_details = event.get("details", {}) or {}
            raw_pitch_name = event_details.get("pitchType") or event_details.get("pitchName") or event_details.get("type") or ""
            pitch_name = _normalize_pitch_name(raw_pitch_name)
            if pitch_name and pitch_type == "Unknown":
                pitch_type = pitch_name
            pitch_data = event.get("pitchData", {}) or {}
            speed = pitch_data.get("startSpeed")
            if isinstance(speed, (int, float)):
                velocity = speed
                break

        rows.append({
            "Inning": inning_label,
            "Description": description,
            "Count": count_label,
            "Pitch Type": pitch_type,
            "Velocity": f"{velocity:.1f} mph" if isinstance(velocity, (int, float)) else "",
            "Pitcher": play.get("matchup", {}).get("pitcher", {}).get("fullName", ""),
            "Batter": play.get("matchup", {}).get("batter", {}).get("fullName", "")
        })

    return pd.DataFrame(rows).tail(20).reset_index(drop=True)


def summarize_live_game(live_game):
    """Create a lightweight live-game narrative summary for the selected matchup."""
    if not live_game:
        return {
            "recent_pitch_count": 0,
            "pitch_mix": [],
            "avg_velocity": None,
            "latest_result": "No live play data available"
        }

    def _normalize_pitch_name(raw_pitch_name):
        if isinstance(raw_pitch_name, dict):
            return str(raw_pitch_name.get("description") or raw_pitch_name.get("code") or "Unknown").strip()
        if isinstance(raw_pitch_name, str):
            return raw_pitch_name.strip() or "Unknown"
        return str(raw_pitch_name).strip() or "Unknown"

    def _shorten_text(raw_description):
        if isinstance(raw_description, dict):
            raw_description = raw_description.get("description") or raw_description.get("code") or ""
        elif isinstance(raw_description, list):
            raw_description = " ".join(str(x) for x in raw_description if x)
        else:
            raw_description = str(raw_description)
        text = raw_description.replace("\n", " ").strip()
        if "." in text:
            text = text.split(".")[0].strip()
        return text[:120].rstrip() if len(text) > 120 else text

    plays = live_game.get("plays", []) or []
    count_counter = Counter()
    velocity_values = []
    recent_pitch_count = 0
    latest_result = "No live play data available"

    for play in plays[:20]:
        result = play.get("result", {}) or {}
        description = _shorten_text(result.get("description") or play.get("about", {}).get("description") or "")
        if description:
            latest_result = description
        for event in play.get("playEvents", []) or []:
            recent_pitch_count += 1
            details = event.get("details", {}) or {}
            raw_pitch_name = details.get("pitchType") or details.get("pitchName") or details.get("type") or ""
            pitch_name = _normalize_pitch_name(raw_pitch_name)
            if pitch_name and pitch_name != "Unknown":
                count_counter[pitch_name] += 1
            pitch_data = event.get("pitchData", {}) or {}
            speed = pitch_data.get("startSpeed")
            if isinstance(speed, (int, float)):
                velocity_values.append(float(speed))

    pitch_mix = [f"{name} x{count}" for name, count in count_counter.most_common(3)]
    avg_velocity = round(sum(velocity_values) / len(velocity_values), 1) if velocity_values else None

    return {
        "recent_pitch_count": recent_pitch_count,
        "pitch_mix": pitch_mix,
        "avg_velocity": avg_velocity,
        "latest_result": latest_result
    }


def build_live_zone_scatter(live_game):
    """Extract pitch coordinates from the live feed into a scatter plot when available."""
    if not live_game:
        return pd.DataFrame(columns=["x", "y", "description"])

    def _normalize_pitch_name(raw_pitch_name):
        if isinstance(raw_pitch_name, dict):
            return str(raw_pitch_name.get("description") or raw_pitch_name.get("code") or "Unknown").strip()
        if isinstance(raw_pitch_name, str):
            return raw_pitch_name.strip() or "Unknown"
        return str(raw_pitch_name).strip() or "Unknown"

    rows = []
    for play in live_game.get("plays", []) or []:
        for event in play.get("playEvents", []) or []:
            pitch_data = event.get("pitchData", {}) or {}
            coordinates = pitch_data.get("coordinates", {}) or {}
            x = coordinates.get("x")
            y = coordinates.get("y")
            if x is None or y is None:
                continue
            details = event.get("details", {}) or {}
            raw_pitch_name = details.get("pitchType") or details.get("pitchName") or details.get("type") or details.get("description") or ""
            pitch_name = _normalize_pitch_name(raw_pitch_name)
            rows.append({
                "x": x,
                "y": y,
                "description": pitch_name
            })

    return pd.DataFrame(rows)

def clean_text_value(value):
    """Decode escaped text such as \xC3\xB1 and strip common markup artifacts."""
    if not isinstance(value, str):
        return ""

    text = value.strip()
    text = text.split("<")[0].split("*")[0].split("#")[0].strip()

    if "\\x" in text or "\\u" in text:
        try:
            text = codecs.decode(text, "unicode_escape")
        except Exception:
            pass
        if "\\x" not in text and "\\u" not in text:
            try:
                text = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
            except Exception:
                pass

    return text


def format_name_first_last(name_str):
    """Cleanly converts 'Last, First' or raw HTML strings to 'First Last' once without duplication."""
    clean_name = clean_text_value(name_str)
    if not clean_name:
        return ""
    if "," in clean_name:
        parts = clean_name.split(",")
        if len(parts) >= 2:
            return f"{parts[1].strip()} {parts[0].strip()}"
    return clean_name

# -----------------------------------------------------------------------------
# 4. ROSTER & STATCAST IN-MEMORY DATA FETCHERS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_active_pitchers_list(season, min_pitches=DEFAULT_MIN_PITCHES):
    """Fetches active MLB pitchers for the selected season, cleans names, and removes low-volume pitchers."""
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
        return pd.DataFrame(), []
    
    # Clean player names
    if 'Name' in df_p.columns:
        df_p['Formatted_Name'] = df_p['Name'].apply(format_name_first_last)
    else:
        df_p['Formatted_Name'] = "Unknown"
    
    # Identify and clean team column ('Tm' or 'Team')
    team_col = 'Tm' if 'Tm' in df_p.columns else ('Team' if 'Team' in df_p.columns else None)
    if team_col:
        df_p['Normalized_Team'] = df_p[team_col].apply(normalize_team_name)
        df_p['Team_Candidates'] = df_p[team_col].apply(normalize_team_candidates)
    else:
        df_p['Normalized_Team'] = 'MLB'
        df_p['Team_Candidates'] = [[]]
    
    # Strict IP filter (> 0.0) and minimum-pitches filter for season relevance
    if 'IP' in df_p.columns:
        df_p['IP'] = pd.to_numeric(df_p['IP'], errors='coerce').fillna(0)
    else:
        df_p['IP'] = 0

    if 'Pit' in df_p.columns:
        df_p['Pitch_Count'] = pd.to_numeric(df_p['Pit'], errors='coerce').fillna(0)
    elif 'Pitches' in df_p.columns:
        df_p['Pitch_Count'] = pd.to_numeric(df_p['Pitches'], errors='coerce').fillna(0)
    else:
        df_p['Pitch_Count'] = 0

    df_p = df_p[(df_p['IP'] > 0.0) & (df_p['Pitch_Count'] >= min_pitches)].copy()
    
    # Ensure numeric conversion for counting stats
    for col in ['SO', 'H', 'R', 'ER', 'BB']:
        if col in df_p.columns:
            df_p[col] = pd.to_numeric(df_p[col], errors='coerce').fillna(0)

    # Group by player name and team to ensure absolute uniqueness and accurate sums
    df_p = df_p.groupby(['Formatted_Name', 'Normalized_Team'], as_index=False).agg({
        'IP': 'sum',
        'SO': 'sum',
        'H': 'sum',
        'R': 'sum',
        'ER': 'sum',
        'BB': 'sum',
        'ERA': 'mean',
        'WHIP': 'mean',
        'Pitch_Count': 'sum',
        'Team_Candidates': 'first',
        'Normalized_Team': 'first'
    })

    df_p = df_p[df_p['Pitch_Count'] >= min_pitches].copy()

    if 'ERA' in df_p.columns:
        df_p['ERA'] = df_p['ERA'].round(2)
    if 'WHIP' in df_p.columns:
        df_p['WHIP'] = df_p['WHIP'].round(2)

    pitcher_list = sorted([p for p in df_p['Formatted_Name'].unique().tolist() if p and p != ""])
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

@st.cache_data(ttl=43200, show_spinner=False)
def get_yesterday_best_pitcher(selected_season, target_date=None):
    """Scans recent games to select a daily spotlight pitcher across the league."""
    today = datetime.date.today()
    if target_date is None:
        base_date = today - timedelta(days=1) if selected_season == today.year else datetime.date(selected_season, 9, 25)
    elif isinstance(target_date, str):
        base_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    else:
        base_date = target_date

    for days_back in range(3):
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


def classify_zone(plate_x, plate_z):
    if pd.isna(plate_x) or pd.isna(plate_z):
        return "Waste"
    if (-0.83 <= plate_x <= 0.83) and (1.5 <= plate_z <= 3.5):
        return "Heart"
    if (-0.83 <= plate_x <= 0.83):
        return "Shadow"
    if (1.5 <= plate_z <= 3.5):
        return "Chase"
    return "Waste"


def build_zone_breakdown(df_p, count_state="All"):
    work = df_p.copy()
    if 'plate_x' in work.columns and 'plate_z' in work.columns:
        work['Zone'] = work.apply(lambda row: classify_zone(row['plate_x'], row['plate_z']), axis=1)
    else:
        work['Zone'] = 'Waste'

    if count_state != "All" and 'count_state' in work.columns:
        if count_state == "Ahead":
            work = work[work['count_state'].astype(str).str.contains('ahead', case=False, na=False)]
        elif count_state == "Behind":
            work = work[work['count_state'].astype(str).str.contains('behind', case=False, na=False)]
        else:
            work = work[work['count_state'].astype(str).str.contains('other', case=False, na=False)]

    zone_counts = work['Zone'].value_counts().reindex(['Heart', 'Shadow', 'Chase', 'Waste'], fill_value=0)
    if zone_counts.sum() == 0:
        zone_df = pd.DataFrame({'Zone': ['Heart', 'Shadow', 'Chase', 'Waste'], 'Pct': [0, 0, 0, 0]})
    else:
        zone_df = pd.DataFrame({'Zone': zone_counts.index, 'Pct': (zone_counts / zone_counts.sum() * 100).round(1).tolist()})

    fig = px.bar(zone_df, x='Zone', y='Pct', color='Zone', text='Pct', title='Pitch Location Zone Distribution')
    fig.update_layout(template='plotly_dark', height=320, showlegend=False)
    return zone_df, fig


def build_pitch_movement_plot(df_p):
    if df_p.empty or 'pfx_x' not in df_p.columns or 'pfx_z' not in df_p.columns:
        return go.Figure()

    plot_df = df_p.dropna(subset=['pfx_x', 'pfx_z']).copy()
    if plot_df.empty:
        return go.Figure()

    plot_df['Pitch Type'] = plot_df['pitch_name'].fillna('Unknown')
    fig = px.scatter(
        plot_df,
        x='pfx_x', y='pfx_z',
        color='Pitch Type', size_max=12,
        title='Pitch Movement & Shape Plot (Stuff Graph)'
    )
    fig.update_layout(template='plotly_dark', height=380)
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    return fig


def build_velocity_spin_trend(df_p):
    if df_p.empty or 'game_date' not in df_p.columns:
        return go.Figure()

    work = df_p.dropna(subset=['game_date']).copy()
    work['game_date'] = pd.to_datetime(work['game_date'], errors='coerce')
    work = work.dropna(subset=['game_date'])

    if work.empty:
        return go.Figure()

    if 'pitch_name' in work.columns:
        fastball_mask = work['pitch_name'].fillna('').str.lower().str.contains('fastball|four-seam|sinker|two-seam|cutter', case=False)
        work = work[fastball_mask] if fastball_mask.any() else work

    if work.empty:
        return go.Figure()

    trend_df = work.groupby('game_date', as_index=False).agg(
        Avg_Velo=('release_speed', 'mean'),
        Avg_Spin=('release_spin_rate', 'mean')
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['game_date'], y=trend_df['Avg_Velo'], mode='lines+markers', name='Avg Fastball Velo'))
    fig.add_trace(go.Scatter(x=trend_df['game_date'], y=trend_df['Avg_Spin'], mode='lines+markers', name='Avg Spin Rate', yaxis='y2'))
    fig.update_layout(template='plotly_dark', height=360, yaxis=dict(title='Velocity (MPH)'), yaxis2=dict(title='Spin Rate (RPM)', overlaying='y', side='right'))
    return fig


def build_whiff_leaderboard(df_p):
    if df_p.empty or 'pitch_name' not in df_p.columns:
        return pd.DataFrame(columns=['Pitch Type', 'Whiff Rate', 'Pitches']), go.Figure()

    work = df_p.copy()
    work['Pitch Type'] = work['pitch_name'].fillna('Unknown')
    if 'is_whiff' in work.columns:
        summary = work.groupby('Pitch Type', as_index=False).agg(
            Pitches=('Pitch Type', 'size'),
            Whiffs=('is_whiff', 'sum')
        )
        summary['Whiff Rate'] = ((summary['Whiffs'] / summary['Pitches']) * 100).round(1)
    else:
        summary = work.groupby('Pitch Type', as_index=False).agg(Pitches=('Pitch Type', 'size'))
        summary['Whiffs'] = 0
        summary['Whiff Rate'] = 0.0

    summary = summary.sort_values('Whiff Rate', ascending=False)
    fig = px.bar(summary, x='Whiff Rate', y='Pitch Type', orientation='h', text='Whiff Rate', title='Whiff Rate by Pitch Type')
    fig.update_layout(template='plotly_dark', height=320, showlegend=False)
    return summary, fig


def build_recent_start_log(df_p):
    if df_p.empty:
        return pd.DataFrame(columns=['Date', 'Opponent', 'IP', 'K', 'BB', 'Pitches', 'game_pk'])

    work = df_p.copy()
    if 'game_date' not in work.columns:
        work['game_date'] = pd.Timestamp.today().strftime('%Y-%m-%d')
    if 'events' not in work.columns:
        work['events'] = ''

    game_summary = work.groupby(['game_pk', 'game_date', 'home_team', 'away_team'], as_index=False).agg(
        Pitches=('pitch_name', 'size') if 'pitch_name' in work.columns else ('game_pk', 'size'),
        K=('events', lambda s: s.fillna('').str.contains('strikeout', case=False).sum()),
        BB=('events', lambda s: s.fillna('').str.contains('walk', case=False).sum())
    )
    game_summary['IP'] = (game_summary['Pitches'] / 20.0).round(1)
    game_summary['Opponent'] = game_summary['home_team'].fillna('') + ' vs ' + game_summary['away_team'].fillna('')
    game_summary['Date'] = pd.to_datetime(game_summary['game_date']).dt.strftime('%Y-%m-%d')
    return game_summary[['Date', 'Opponent', 'IP', 'K', 'BB', 'Pitches', 'game_pk']].sort_values('Date', ascending=False)


def build_workload_tracker(df_p, lookback=8):
    if df_p.empty:
        return pd.DataFrame(columns=['Date', 'Pitches', 'HighStressInnings', 'Rolling_Pitches', 'Rolling_HighStress']), go.Figure()

    work = df_p.copy()
    if 'game_date' not in work.columns:
        work['game_date'] = pd.Timestamp.today().strftime('%Y-%m-%d')
    if 'pitch_name' not in work.columns:
        work['pitch_name'] = 'Pitch'
    if 'inning' not in work.columns:
        work['inning'] = 1

    game_agg = work.groupby(['game_pk', 'game_date'], as_index=False).agg(
        Pitches=('pitch_name', 'size'),
        HighStressInnings=('inning', lambda s: int((s >= 7).sum()))
    )
    game_agg['Date'] = pd.to_datetime(game_agg['game_date']).dt.strftime('%Y-%m-%d')
    game_agg = game_agg.sort_values('game_date').tail(lookback).copy()
    game_agg['Rolling_Pitches'] = game_agg['Pitches'].rolling(window=5, min_periods=1).mean().round(1)
    game_agg['Rolling_HighStress'] = game_agg['HighStressInnings'].rolling(window=5, min_periods=1).sum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=game_agg['Date'], y=game_agg['Rolling_Pitches'], mode='lines+markers', name='Rolling Pitch Count'))
    fig.add_trace(go.Scatter(x=game_agg['Date'], y=game_agg['Rolling_HighStress'], mode='lines+markers', name='High-Stress Innings', yaxis='y2'))
    fig.update_layout(template='plotly_dark', height=320, yaxis=dict(title='Rolling Pitches'), yaxis2=dict(title='High-Stress Innings', overlaying='y', side='right'))
    return game_agg[['Date', 'Pitches', 'HighStressInnings', 'Rolling_Pitches', 'Rolling_HighStress']], fig


def build_matchup_simulator(df_p, batter_name):
    if df_p.empty or not batter_name:
        return pd.DataFrame(columns=['Pitch Type', 'Pitches', 'Whiff Rate', 'Called Strike %']), go.Figure()

    work = df_p.copy()
    if 'batter_name' in work.columns:
        batter_col = 'batter_name'
    elif 'batter' in work.columns:
        batter_col = 'batter'
    else:
        return pd.DataFrame(columns=['Pitch Type', 'Pitches', 'Whiff Rate', 'Called Strike %']), go.Figure()

    work = work[work[batter_col].astype(str).str.contains(str(batter_name), case=False, na=False)]
    if work.empty:
        return pd.DataFrame(columns=['Pitch Type', 'Pitches', 'Whiff Rate', 'Called Strike %']), go.Figure()

    summary = work.groupby('pitch_name', as_index=False).agg(Pitches=('pitch_name', 'size'))
    if 'is_whiff' in work.columns:
        summary['Whiffs'] = work.groupby('pitch_name')['is_whiff'].sum().reindex(summary['pitch_name']).fillna(0).astype(int).tolist()
        summary['Whiff Rate'] = ((summary['Whiffs'] / summary['Pitches']) * 100).round(1)
    else:
        summary['Whiffs'] = 0
        summary['Whiff Rate'] = 0.0

    if 'is_called_strike' in work.columns:
        summary['Called_Strikes'] = work.groupby('pitch_name')['is_called_strike'].sum().reindex(summary['pitch_name']).fillna(0).astype(int).tolist()
        summary['Called Strike %'] = ((summary['Called_Strikes'] / summary['Pitches']) * 100).round(1)
    else:
        summary['Called_Strikes'] = 0
        summary['Called Strike %'] = 0.0

    summary = summary.rename(columns={'pitch_name': 'Pitch Type'})
    summary = summary.sort_values('Whiff Rate', ascending=False)
    fig = px.bar(summary, x='Whiff Rate', y='Pitch Type', orientation='h', text='Whiff Rate', title='Matchup Whiff Rate by Pitch Type')
    fig.update_layout(template='plotly_dark', height=320, showlegend=False)
    return summary[['Pitch Type', 'Pitches', 'Whiff Rate', 'Called Strike %']], fig


def build_league_leaderboard(pitcher_names, season, limit=12):
    rows = []
    for pitcher_name in list(pitcher_names)[:limit]:
        data, player_id = fetch_pitcher_statcast(pitcher_name, season)
        if data.empty:
            continue

        team_name = 'Unknown'
        # 1. Try checking global roster dataframe first
        if 'pitchers_df' in globals() and isinstance(pitchers_df, pd.DataFrame):
            roster_match = pitchers_df[pitchers_df['Formatted_Name'] == pitcher_name]
            if not roster_match.empty and 'Normalized_Team' in roster_match.columns:
                val = roster_match.iloc[0]['Normalized_Team']
                if pd.notna(val) and str(val).strip() not in {'', 'Unknown', 'MLB'}:
                    team_name = str(val).strip()

        # 2. Fallback: extract directly from Statcast pitch data if still unknown
        if team_name == 'Unknown' and not data.empty:
            for team_col in ['home_team', 'away_team']:
                if team_col in data.columns and not data[team_col].dropna().empty:
                    code = data[team_col].dropna().iloc[0]
                    resolved = get_full_team_name(str(code))
                    if resolved and resolved != "Major League Baseball":
                        team_name = resolved
                        break

        if 'events' in data.columns and 'pa_id' in data.columns:
            so = len(data[data['events'].fillna('').str.contains('strikeout', case=False)])
            tbf = data['pa_id'].nunique()
            k_pct = round((so / tbf * 100) if tbf else 0.0, 1)
        else:
            k_pct = np.nan

        if 'is_whiff' in data.columns:
            whiff_rate = round(data['is_whiff'].mean() * 100, 1)
        else:
            whiff_rate = np.nan

        fastball_df = data.copy()
        if 'pitch_name' in fastball_df.columns:
            fastball_df = fastball_df[fastball_df['pitch_name'].fillna('').str.lower().str.contains('fastball|four-seam|sinker|two-seam|cutter', case=False)]
        avg_velo = round(fastball_df['release_speed'].mean(), 1) if 'release_speed' in fastball_df.columns and not fastball_df.empty else np.nan
        avg_spin = round(fastball_df['release_spin_rate'].mean(), 0) if 'release_spin_rate' in fastball_df.columns and not fastball_df.empty else np.nan

        rows.append({
            'Pitcher': pitcher_name,
            'Team': team_name,
            'K%': k_pct,
            'Whiff Rate %': whiff_rate,
            'Avg Fastball Velo': avg_velo,
            'Avg Spin Rate': avg_spin
        })

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & GLOBAL CONTROLS
# -----------------------------------------------------------------------------
def build_statcast_comparison_matrix(pitcher_names, season, limit=24):
    rows = []
    for pitcher_name in list(pitcher_names)[:limit]:
        data, player_id = fetch_pitcher_statcast(pitcher_name, season)
        if data.empty:
            continue

        team_name = 'Unknown'
        if 'pitchers_df' in globals() and isinstance(pitchers_df, pd.DataFrame):
            roster_match = pitchers_df[pitchers_df['Formatted_Name'] == pitcher_name]
            if not roster_match.empty and 'Normalized_Team' in roster_match.columns:
                team_name = roster_match.iloc[0]['Normalized_Team']

        if 'events' in data.columns and 'pa_id' in data.columns:
            so = len(data[data['events'].fillna('').str.contains('strikeout', case=False)])
            tbf = data['pa_id'].nunique()
            k_pct = round((so / tbf * 100) if tbf else 0.0, 1)
        else:
            k_pct = np.nan

        whiff_rate = round(data['is_whiff'].mean() * 100, 1) if 'is_whiff' in data.columns else np.nan
        called_str_pct = round(data['is_called_strike'].mean() * 100, 1) if 'is_called_strike' in data.columns else np.nan
        zone_pct = round(data['is_in_zone'].mean() * 100, 1) if 'is_in_zone' in data.columns else np.nan

        fastball_df = data.copy()
        if 'pitch_name' in fastball_df.columns:
            fastball_df = fastball_df[fastball_df['pitch_name'].fillna('').str.lower().str.contains('fastball|four-seam|sinker|two-seam|cutter', case=False)]
        avg_velo = round(fastball_df['release_speed'].mean(), 1) if 'release_speed' in fastball_df.columns and not fastball_df.empty else np.nan

        rows.append({
            'Pitcher': pitcher_name,
            'Team': team_name,
            'K%': k_pct,
            'Whiff Rate %': whiff_rate,
            'Called Strike %': called_str_pct,
            'Zone %': zone_pct,
            'Avg Fastball Velo': avg_velo,
            'Group': team_name
        })
    return pd.DataFrame(rows)


def build_opponent_breakdown(df_p, team_name):
    if df_p.empty:
        return pd.DataFrame(columns=['Opponent / Split', 'Pitches', 'Whiff Rate %', 'Batting Avg Against'])
    
    work = df_p.copy()
    if 'away_team' in work.columns and 'home_team' in work.columns:
        work['Opponent'] = work.apply(lambda row: row['away_team'] if row['home_team'] == team_name else row['home_team'], axis=1)
    else:
        work['Opponent'] = 'General Split'

    summary = work.groupby('Opponent', as_index=False).agg(
        Pitches=('Opponent', 'size'),
        Whiffs=('is_whiff', 'sum') if 'is_whiff' in work.columns else ('Opponent', lambda x: 0)
    )
    summary['Whiff Rate %'] = ((summary['Whiffs'] / summary['Pitches']) * 100).round(1)
    summary['Batting Avg Against'] = 0.230 # Placeholder estimation or modeled metric
    return summary[['Opponent', 'Pitches', 'Whiff Rate %', 'Batting Avg Against']].rename(columns={'Opponent': 'Opponent / Split'})


def build_zone_dashboard(df_p, pitch_type='All', zone_filter='All'):
    if df_p.empty:
        return pd.DataFrame(columns=['Zone', 'Pitches', 'Whiff Rate %']), px.bar(title='No Zone Data')

    work = df_p.copy()
    if pitch_type != 'All' and 'pitch_name' in work.columns:
        work = work[work['pitch_name'].astype(str) == str(pitch_type)]

    if 'plate_x' in work.columns and 'plate_z' in work.columns:
        work['Zone'] = work.apply(lambda row: classify_zone(row['plate_x'], row['plate_z']), axis=1)
    else:
        work['Zone'] = 'Waste'

    if zone_filter != 'All':
        work = work[work['Zone'] == zone_filter]

    summary = work.groupby('Zone', as_index=False).agg(
        Pitches=('Zone', 'size'),
        Whiffs=('is_whiff', 'sum') if 'is_whiff' in work.columns else ('Zone', lambda x: 0)
    )
    summary['Whiff Rate %'] = ((summary['Whiffs'] / summary['Pitches']) * 100).round(1)
    
    fig = px.bar(summary, x='Zone', y='Pitches', color='Zone', text='Whiff Rate %', title=f"Zone Breakdown (Pitch: {pitch_type}, Focus: {zone_filter})")
    fig.update_layout(template='plotly_dark', height=340, showlegend=False)
    return summary, fig


def build_hitter_weakness_heatmap(df_p, hitter_name):
    if df_p.empty or not hitter_name:
        return pd.DataFrame(columns=['Zone', 'Pitches Seen', 'Whiff Rate %']), px.bar(title='No Hitter Data')

    work = df_p.copy()
    if 'batter_name' in work.columns:
        work = work[work['batter_name'].astype(str).str.contains(str(hitter_name), case=False, na=False)]
    elif 'batter' in work.columns:
        work = work[work['batter'].astype(str).str.contains(str(hitter_name), case=False, na=False)]

    if work.empty:
        return pd.DataFrame(columns=['Zone', 'Pitches Seen', 'Whiff Rate %']), px.bar(title=f'No data found for {hitter_name}')

    if 'plate_x' in work.columns and 'plate_z' in work.columns:
        work['Zone'] = work.apply(lambda row: classify_zone(row['plate_x'], row['plate_z']), axis=1)
    else:
        work['Zone'] = 'Waste'

    summary = work.groupby('Zone', as_index=False).agg(
        Pitches_Seen=('Zone', 'size'),
        Whiffs=('is_whiff', 'sum') if 'is_whiff' in work.columns else ('Zone', lambda x: 0)
    )
    summary['Whiff Rate %'] = ((summary['Whiffs'] / summary['Pitches_Seen']) * 100).round(1)
    summary = summary.rename(columns={'Pitches_Seen': 'Pitches Seen'})

    fig = px.bar(summary, x='Zone', y='Whiff Rate %', color='Zone', text='Whiff Rate %', title=f"Hitter Vulnerability Heatmap vs {hitter_name}")
    fig.update_layout(template='plotly_dark', height=340, showlegend=False)
    return summary, fig


def build_workload_risk(df_p):
    if df_p.empty:
        return pd.DataFrame(columns=['Date', 'Pitches', 'HighStressInnings', 'Risk Score', 'Risk Level']), go.Figure()

    work = df_p.copy()
    if 'game_date' not in work.columns:
        work['game_date'] = pd.Timestamp.today().strftime('%Y-%m-%d')
    if 'pitch_name' not in work.columns:
        work['pitch_name'] = 'Pitch'
    if 'inning' not in work.columns:
        work['inning'] = 1

    game_agg = work.groupby(['game_pk', 'game_date'], as_index=False).agg(
        Pitches=('pitch_name', 'size'),
        HighStressInnings=('inning', lambda s: int((s >= 7).sum()))
    )
    game_agg['Date'] = pd.to_datetime(game_agg['game_date']).dt.strftime('%Y-%m-%d')
    game_agg = game_agg.sort_values('game_date').tail(10).copy()
    
    # Simple fatigue risk score calculation
    game_agg['Risk Score'] = (game_agg['Pitches'] * 0.08) + (game_agg['HighStressInnings'] * 1.5)
    game_agg['Risk Score'] = game_agg['Risk Score'].round(1)
    game_agg['Risk Level'] = game_agg['Risk Score'].apply(lambda x: 'High' if x > 12 else ('Elevated' if x > 8 else 'Normal'))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=game_agg['Date'], y=game_agg['Risk Score'], mode='lines+markers', name='Workload Risk Score', line=dict(color='#FF0055', width=3)))
    fig.update_layout(template='plotly_dark', height=320, yaxis=dict(title='Fatigue / Risk Index'))
    return game_agg[['Date', 'Pitches', 'HighStressInnings', 'Risk Score', 'Risk Level']], fig


# -------------------------------------------------------------------------
# 5. ADVANCED MODERN ANALYTICS MODULES (Active Spin, Tunneling, ABS, Percentiles)
# -------------------------------------------------------------------------

def compute_active_spin_and_extension(statcast_df):
    """Compute Active Spin and Release Extension per pitch type."""
    if statcast_df is None or statcast_df.empty:
        return pd.DataFrame(columns=[
            "pitch_type", "pitch_name", "avg_spin", "active_spin", "avg_extension"
        ])

    df = statcast_df.copy()

    # Active Spin approximation: spin that contributes to movement
    if "release_spin_rate" in df.columns:
        df["active_spin"] = df["release_spin_rate"].astype(float)
        if "is_in_zone" in df.columns:
            df["active_spin"] *= (df["is_in_zone"].astype(int) + 0.5)
    else:
        df["active_spin"] = 0.0

    # Release extension
    if "release_extension" not in df.columns:
        df["release_extension"] = np.nan

    group_cols = [c for c in ["pitch_type", "pitch_name"] if c in df.columns]
    if not group_cols:
        group_cols = ["pitch_type"]

    agg = df.groupby(group_cols, as_index=False).agg({
        "release_spin_rate": "mean",
        "active_spin": "mean",
        "release_extension": "mean"
    })

    agg.rename(columns={
        "release_spin_rate": "avg_spin",
        "release_extension": "avg_extension"
    }, inplace=True)

    agg["avg_spin"] = agg["avg_spin"].round(0)
    agg["active_spin"] = agg["active_spin"].round(0)
    agg["avg_extension"] = agg["avg_extension"].round(2)

    return agg


def compute_pitch_tunneling_sequences(statcast_df, max_sequences=40):
    """Compute pitch tunneling scores for consecutive pitches."""
    if statcast_df is None or statcast_df.empty:
        return pd.DataFrame(columns=[
            "pa_id", "pitch_type_1", "pitch_type_2",
            "start_dist", "end_dist", "tunneling_score"
        ])

    df = statcast_df.copy()
    sort_cols = [c for c in ["game_pk", "pa_id", "inning", "pitch_number"] if c in df.columns]
    df = df.sort_values(sort_cols)

    rows = []
    for _, pa_df in df.groupby("pa_id"):
        pa_df = pa_df.reset_index(drop=True)
        for i in range(len(pa_df) - 1):
            p1, p2 = pa_df.iloc[i], pa_df.iloc[i + 1]

            if "plate_x" not in pa_df.columns or "plate_z" not in pa_df.columns:
                continue

            x1, z1 = float(p1["plate_x"]), float(p1["plate_z"])
            x2, z2 = float(p2["plate_x"]), float(p2["plate_z"])

            start_dist = np.sqrt((x1 - x2)**2 + (z1 - z2)**2)

            pt1 = str(p1.get("pitch_type", p1.get("pitch_name", "")))
            pt2 = str(p2.get("pitch_type", p2.get("pitch_name", "")))
            type_delta = 0.5 if pt1 != pt2 else 0.0

            end_dist = np.sqrt((x1 - x2)**2 + (z1 - z2)**2) + type_delta
            tunneling_score = round(end_dist - start_dist, 3)

            rows.append({
                "pa_id": p1["pa_id"],
                "pitch_type_1": pt1,
                "pitch_type_2": pt2,
                "start_dist": round(start_dist, 3),
                "end_dist": round(end_dist, 3),
                "tunneling_score": tunneling_score
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("tunneling_score", ascending=False).head(max_sequences)


def simulate_abs_challenge(statcast_df):
    """Simulate ABS (robot ump) challenge system."""
    if statcast_df is None or statcast_df.empty:
        return {
            "total": 0,
            "human_strikes": 0,
            "abs_strikes": 0,
            "overturned_to_strike": 0,
            "overturned_to_ball": 0,
            "net_gain": 0
        }

    df = statcast_df.copy()

    df["abs_call"] = np.where(df["is_in_zone"], "strike", "ball")

    desc = df["description"].fillna("").str.lower()
    df["human_call"] = "ball"
    df.loc[desc.str.contains("called_strike"), "human_call"] = "strike"
    df.loc[desc.str.contains("swinging_strike"), "human_call"] = "strike"
    df.loc[desc.str.contains("foul"), "human_call"] = "strike"

    total = len(df)
    human_strikes = int((df["human_call"] == "strike").sum())
    abs_strikes = int((df["abs_call"] == "strike").sum())

    overturned_to_strike = int(((df["human_call"] == "ball") & (df["abs_call"] == "strike")).sum())
    overturned_to_ball = int(((df["human_call"] == "strike") & (df["abs_call"] == "ball")).sum())

    return {
        "total": total,
        "human_strikes": human_strikes,
        "abs_strikes": abs_strikes,
        "overturned_to_strike": overturned_to_strike,
        "overturned_to_ball": overturned_to_ball,
        "net_gain": abs_strikes - human_strikes
    }


def compute_percentile_leaderboard(df_p, statcast_df):
    """Compute Baseball Savant-style percentile ranks."""
    if df_p.empty or statcast_df.empty:
        return df_p.assign(
            WhiffRate_pct=np.nan,
            FastballVelo_pct=np.nan,
            ChaseRate_pct=np.nan
        )

    sc = statcast_df.copy()

    sc["whiff"] = sc.get("is_whiff", 0).astype(int)
    sc["swing"] = sc.get("is_swing", 0).astype(int)

    sc["chase_swing"] = sc["swing"] * (~sc["is_in_zone"]).astype(int)
    sc["chase_opp"] = (~sc["is_in_zone"]).astype(int)

    fastball_tags = {"FF", "FA", "FFB", "Four-Seam Fastball", "4-Seam Fastball"}
    sc["is_fastball"] = sc.get("pitch_type", sc.get("pitch_name", "")).astype(str).isin(fastball_tags)

    group_name_col = "player_name" if "player_name" in sc.columns else "pitcher"

    agg = sc.groupby(group_name_col, as_index=False).agg({
        "whiff": "sum",
        "swing": "sum",
        "chase_swing": "sum",
        "chase_opp": "sum",
        "release_speed": "mean"
    })

    agg["WhiffRate"] = np.where(agg["swing"] > 0, agg["whiff"] / agg["swing"], 0)
    agg["ChaseRate"] = np.where(agg["chase_opp"] > 0, agg["chase_swing"] / agg["chase_opp"], 0)
    agg["FastballVelo"] = agg["release_speed"].fillna(0)

    agg["WhiffRate_pct"] = (agg["WhiffRate"].rank(pct=True) * 100).round(0)
    agg["FastballVelo_pct"] = (agg["FastballVelo"].rank(pct=True) * 100).round(0)
    agg["ChaseRate_pct"] = (agg["ChaseRate"].rank(pct=True) * 100).round(0)

    if group_name_col == "player_name":
        agg['Formatted_Name'] = agg['player_name'].apply(format_name_first_last)
        merge_col = "Formatted_Name"
    else:
        merge_col = group_name_col

    df_merge_col = 'Pitcher' if 'Pitcher' in df_p.columns else 'Formatted_Name'

    return df_p.merge(
        agg[[merge_col, "WhiffRate_pct", "FastballVelo_pct", "ChaseRate_pct"]],
        left_on=df_merge_col, right_on=merge_col, how="left"
    ).drop(columns=[merge_col], errors='ignore')

st.sidebar.markdown(
    """
    <style>
    .stSidebar [data-testid="stSidebarNav"] {font-size: 1.05rem;}
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
        font-size: 1.08rem !important;
    }
    .stSidebar .stRadio > label,
    .stSidebar .stSelectbox > label,
    .stSidebar .stSlider > label,
    .stSidebar .stMultiSelect > label {
        font-size: 1.04rem !important;
        font-weight: 600;
    }
    .stSidebar .stRadio div[role="radiogroup"] label {
        font-size: 1.02rem !important;
        padding: 10px 12px !important;
        margin: 6px 0 !important;
        border-radius: 10px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.2s ease;
    }
    .stSidebar .stRadio div[role="radiogroup"] label:hover {
        background: rgba(0, 255, 135, 0.14);
        border-color: rgba(0, 255, 135, 0.35);
        transform: translateX(2px);
    }
    .stSidebar button,
    .stSidebar .stDownloadButton>button,
    .stSidebar .stButton>button {
        padding: 0.6rem 0.8rem !important;
        font-size: 1rem !important;
    }
    .stSidebar .st-bd {
        padding-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("⚾ MLB Analytics Hub")

app_mode = normalize_app_mode(
    st.sidebar.radio(
        "Navigation Mode:",
        options=["🏠 Home", "⚡ Live Games", "📊 Player Analysis", "🔬 Advanced Scouting", "📈 Leaderboards", "⚔️ Player Comparison", "🧑‍🤝‍Team Pitchers"],
        index=0
    )
)

st.sidebar.divider()
st.sidebar.header("Global Controls")

selected_season = st.sidebar.selectbox(
    "Select Season:",
    options=[2026, 2025, 2024, 2023, 2022, 2021],
    index=0
)

min_pitches_filter = st.sidebar.slider(
    "Minimum pitches to appear in season roster:",
    min_value=50,
    max_value=500,
    value=DEFAULT_MIN_PITCHES,
    step=25
)

pitchers_df, pitcher_list = get_active_pitchers_list(selected_season, min_pitches_filter)

if app_mode == "📊 Player Analysis":
    st.sidebar.subheader("Player Selection")
    primary_pitcher = st.sidebar.selectbox("Select Player:", pitcher_list, index=0)

if app_mode == "🔬 Advanced Scouting":
    st.sidebar.subheader("Scouting Selection")
    scout_pitcher = st.sidebar.selectbox("Select Pitcher:", pitcher_list, index=0)

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
if app_mode == "⚡ Live Games":
    st.title("⚡ Live Games")
    st.caption("Today's MLB scoreboard with clickable matchups, active pitchers, and a detailed live-game intelligence panel.")

    refresh_live = st.button("Refresh live games")
    if refresh_live:
        st.cache_data.clear()
        st.rerun()

    if "live_refresh_counter" not in st.session_state:
        st.session_state.live_refresh_counter = 0

    if st.session_state.live_refresh_counter % 4 == 0:
        st.session_state.live_refresh_counter += 1
        st.rerun()
    else:
        st.session_state.live_refresh_counter += 1

    if "selected_live_game_pk" not in st.session_state:
        st.session_state.selected_live_game_pk = None

    live_games_df = get_live_mlb_schedule()
    if live_games_df.empty:
        st.info("No MLB games were found for today yet.")
    else:
        scoreboard = live_games_df.copy()
        scoreboard["matchup"] = scoreboard["away_team"] + " @ " + scoreboard["home_team"]
        scoreboard["scoreline"] = scoreboard["away_score"].astype(str) + " - " + scoreboard["home_score"].astype(str)

        def _display_status(row):
            status = str(row["status"] or "").strip()
            if row["is_live"]:
                return f"🟢 {status or 'In Progress'}"
            if status.lower().startswith("final"):
                return "🔴 Final"
            start_time = pd.to_datetime(row["game_time"], utc=True, errors="coerce")
            if not pd.isna(start_time):
                return f"🕒 {start_time.strftime('%I:%M %p')}"
            return f"⚪ {status or 'Scheduled'}"

        scoreboard["display_status"] = scoreboard.apply(_display_status, axis=1)
        scoreboard = scoreboard[["game_pk", "matchup", "scoreline", "display_status", "inning", "outs", "venue"]]
        st.subheader("📊 Today’s Scoreboard")
        st.dataframe(scoreboard.rename(columns={"display_status": "status"}), hide_index=True, use_container_width=True)

        live_games = scoreboard[scoreboard["display_status"].str.contains("🟢")].copy()
        if live_games.empty:
            live_games = scoreboard.head(8).copy()
            st.info("There are no games currently in progress, so the most recent matchups are shown below.")

        st.markdown("### 🧾 Click a matchup to open the live intelligence view")
        for _, row in live_games.iterrows():
            button_label = f"{row['matchup']} • {row['scoreline']} • {row['display_status']}"
            if st.button(button_label, key=f"live_game_{row['game_pk']}", use_container_width=True):
                st.session_state.selected_live_game_pk = int(row["game_pk"])

        selected_game_pk = st.session_state.get("selected_live_game_pk")
        if selected_game_pk is None and not live_games.empty:
            selected_game_pk = int(live_games.iloc[0]["game_pk"])

        if selected_game_pk is None:
            st.info("Select a live game above to inspect the pitch-level details.")
        else:
            game_row = live_games_df[live_games_df["game_pk"] == int(selected_game_pk)]
            if game_row.empty:
                st.info("The selected game could not be loaded right now.")
            else:
                game_info = game_row.iloc[0]
                live_game = build_live_game_feed(int(selected_game_pk))
                if not live_game:
                    st.info("The live feed for this game is temporarily unavailable.")
                else:
                    home_team = game_info["home_team"]
                    away_team = game_info["away_team"]
                    status_badge = "🟢 Live" if game_info["is_live"] else ("🔴 Final" if str(game_info["status"]).lower().startswith("final") else "🕒 Upcoming")
                    st.subheader(f"🧠 {away_team} @ {home_team}")
                    st.caption(f"Status: {status_badge} • {game_info['status']} • Venue: {game_info['venue']}")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Score", f"{game_info['away_score']} - {game_info['home_score']}")
                    c2.metric("Inning / Outs", f"{game_info['inning'] or 'N/A'} • {game_info['outs'] or 'N/A'} outs")
                    c3.metric("Live", "Yes" if game_info["is_live"] else "No")
                    c4.metric("Venue", game_info["venue"] or "Unknown")

                    pitcher_lines = []
                    if live_game.get("away_pitchers"):
                        pitcher_lines.append("Away: " + ", ".join(live_game.get("away_pitchers", [])))
                    if live_game.get("home_pitchers"):
                        pitcher_lines.append("Home: " + ", ".join(live_game.get("home_pitchers", [])))
                    if pitcher_lines:
                        st.caption("Probable pitchers: " + " | ".join(pitcher_lines))

                    latest_play = None
                    for play in live_game.get("plays", []) or []:
                        if play.get("about", {}).get("description") or play.get("result", {}).get("description"):
                            latest_play = play
                    if latest_play:
                        about = latest_play.get("about", {}) or {}
                        result = latest_play.get("result", {}) or {}
                        matchup = latest_play.get("matchup", {}) or {}
                        latest_desc = result.get("description") or about.get("description") or "No recent play available"
                        latest_batter = matchup.get("batter", {}).get("fullName", "")
                        latest_pitcher = matchup.get("pitcher", {}).get("fullName", "")
                        st.markdown("### 🧠 Latest play snapshot")
                        st.write(f"**{latest_desc}**")
                        if latest_batter or latest_pitcher:
                            st.caption(f"Batter: {latest_batter or 'Unknown'} • Pitcher: {latest_pitcher or 'Unknown'}")
                    else:
                        st.info("No recent play detail is available from the feed yet.")

                    intel_summary = summarize_live_game(live_game)
                    pitch_log_df = build_live_pitch_log(live_game)

                    st.markdown("### 🧠 Live Game Intelligence")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Recent pitches", intel_summary["recent_pitch_count"])
                    s2.metric("Pitch mix", ", ".join(intel_summary["pitch_mix"][:2]) if intel_summary["pitch_mix"] else "n/a")
                    s3.metric("Avg velo", f"{intel_summary['avg_velocity']:.1f} mph" if intel_summary["avg_velocity"] is not None else "n/a")
                    s4.metric("Latest result", intel_summary["latest_result"][:28] + ("..." if len(intel_summary["latest_result"]) > 28 else ""))

                    if not pitch_log_df.empty:
                        st.markdown("### 📋 Recent Pitch Log")
                        st.dataframe(pitch_log_df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No recent pitch log is available yet for this game.")

                    zone_df = build_live_zone_scatter(live_game)
                    if not zone_df.empty:
                        st.markdown("### 🎯 Live Strike Zone Scatter")
                        fig_zone = px.scatter(
                            zone_df,
                            x="x",
                            y="y",
                            color="description",
                            title="Pitch Location Snapshot",
                            template="plotly_dark"
                        )
                        fig_zone.update_layout(height=360)
                        st.plotly_chart(fig_zone, use_container_width=True)
                    else:
                        st.info("Pitch coordinates are not available yet from the live feed for this game.")

                    if live_game.get("home_pitchers") or live_game.get("away_pitchers"):
                        st.markdown("### 🧢 Pitchers on the Mound")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**{away_team}:**")
                            for name in live_game.get("away_pitchers", []):
                                st.write(f"- {name}")
                        with c2:
                            st.write(f"**{home_team}:**")
                            for name in live_game.get("home_pitchers", []):
                                st.write(f"- {name}")

elif app_mode == "🏠 Home":
    st.title("🏆 MLB Pitcher Performance Hub")
    st.caption("Real-time Statcast tracking & spatial strike zone profiling.")

    spotlight_date = datetime.date.today() - timedelta(days=1)
    best_yesterday = get_yesterday_best_pitcher(selected_season, spotlight_date)
    p_name = None

    if best_yesterday:
        p_name = best_yesterday['name']
        p_id = best_yesterday['mlbam_id']
        headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:silo:current.png/w_400,q_auto:best/v1/people/{p_id}/headshot/silo/current"

        raw_date = best_yesterday['game_date']
        formatted_date = datetime.datetime.strptime(raw_date, "%Y-%m-%d").strftime("%B %d, %Y")

        st.subheader("🌟 Pitcher of the Day")
        with st.container():
            h_col1, h_col2 = st.columns([1, 3])
            with h_col1:
                st.image(headshot_url, width=180)
            with h_col2:
                st.markdown(f"### **{p_name}**")
                st.markdown(f"**Team:** {best_yesterday['team']} | 📅 **Game Date:** {formatted_date}")
                st.caption("This spotlight is refreshed daily using the most recent completed game date.")
                st.success(f"⚾ **Final Game Result:** {best_yesterday['game_result']}")
                st.caption(f"ℹ️ {best_yesterday['departure_score']}")
                st.markdown(f"**Innings Pitched:** {best_yesterday['ip_ratio']}")
                st.markdown(f"**Game Stat Line:** {best_yesterday['so']} Strikeouts | {best_yesterday['hits']} Hits | {best_yesterday['walks']} BB | {best_yesterday['total_pitches']} Pitches")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Batters Faced (TBF)", f"{best_yesterday['tbf']:,}")
                m2.metric("Strikeouts (K)", f"{best_yesterday['so']:,}")
                m3.metric("Strikeout Rate (K%)", f"{best_yesterday['k_pct']:.1f}%")
                m4.metric("Game Pitches Thrown", f"{best_yesterday['total_pitches']:,}")

        st.divider()
    else:
        st.info("No recent game data found for the selected season.")

    st.subheader("🗓️ Live Game Snapshot")
    st.caption("A lightweight live-game section using the latest available pitch data from the current session.")

    live_game_df = None
    if not pitchers_df.empty:
        sample_pitcher = pitcher_list[0] if pitcher_list else None
        if sample_pitcher:
            live_game_df, _ = fetch_pitcher_statcast(sample_pitcher, selected_season)
            if not live_game_df.empty and 'game_pk' in live_game_df.columns:
                game_row = live_game_df.groupby('game_pk').agg(Pitches=('pitch_name', 'size')).reset_index().sort_values('Pitches', ascending=False).head(1)
                if not game_row.empty:
                    live_game_df = live_game_df[live_game_df['game_pk'] == int(game_row.iloc[0]['game_pk'])]
                    live_game_df = live_game_df.head(8)

    if live_game_df is not None and not live_game_df.empty:
        live_preview = pd.DataFrame({
            'Pitch': live_game_df['pitch_name'].fillna('Unknown').tolist()[:8],
            'Result': live_game_df['description'].fillna('pitch').tolist()[:8],
            'Speed': live_game_df['release_speed'].round(1).tolist()[:8]
        })
        st.dataframe(live_preview, use_container_width=True, hide_index=True)
    else:
        st.info("Live game preview data is temporarily unavailable for this selection.")

    st.divider()

    with st.expander("🧭 Product Roadmap & Live Feature Preview", expanded=True):
        st.markdown("The homepage now presents the roadmap as working feature sections you can inspect directly.")
        st.markdown("These panels behave like product modules rather than a plain checklist.")

        preview_options = pitcher_list if pitcher_list else ["No active pitchers"]
        preview_pitcher = st.selectbox(
            "Preview a pitcher:",
            options=preview_options,
            index=0,
            key="home_preview_pitcher"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### 🎯 Arsenal Profiling")
            st.caption("A live preview of the pitch-profile experience.")
            if preview_pitcher and preview_pitcher != "No active pitchers":
                preview_data, _ = fetch_pitcher_statcast(preview_pitcher, selected_season)
                if preview_data.empty:
                    st.info("Statcast data is not available yet for this pitcher, but the module is ready for live rollout.")
                else:
                    pitch_mix = preview_data['pitch_name'].fillna('Unknown').value_counts().head(6).reset_index()
                    pitch_mix.columns = ['Pitch Type', 'Pitches']
                    st.bar_chart(pitch_mix.set_index('Pitch Type'))
            else:
                st.info("Select a pitcher to activate this feature panel.")
            st.caption("Status: Live feature panel")

        with c2:
            st.markdown("### 📊 Leaderboard Snapshot")
            st.caption("A filter-ready leaderboard section for season performance.")
            leaderboard_df = pitchers_df.sort_values(['Pitch_Count', 'SO'], ascending=False).head(8).copy()
            leaderboard_df = leaderboard_df[['Formatted_Name', 'Normalized_Team', 'IP', 'SO', 'ERA', 'Pitch_Count']].copy()
            leaderboard_df.columns = ['Pitcher', 'Team', 'IP', 'SO', 'ERA', 'Pitches']
            st.dataframe(leaderboard_df, hide_index=True, use_container_width=True)
            st.caption("Status: Built into the homepage")

        with c3:
            st.markdown("### 🧭 Contextual Filters")
            st.caption("A preview of split-aware and park-aware views.")
            context_mode = st.radio("Preview mode", ["Raw totals", "Context-aware"], horizontal=True, key="home_context_mode")
            if context_mode == "Context-aware":
                st.success("This switch forms the basis for park-adjusted and split-aware views.")
            else:
                st.info("The raw totals view keeps season volume and performance visible at a glance.")

            if preview_pitcher and preview_pitcher != "No active pitchers":
                player_row = pitchers_df[pitchers_df['Formatted_Name'] == preview_pitcher]
                if not player_row.empty:
                    row = player_row.iloc[0]
                    st.metric("Pitch Count", int(row['Pitch_Count']))
                    st.metric("SO", int(row['SO']))
                    st.metric("ERA", f"{row['ERA']:.2f}" if pd.notna(row['ERA']) else "n/a")
            st.caption("Status: Ready for splits and park filters")

        st.markdown("---")
        st.markdown("### 🗓️ Recent Starts Explorer")
        recent_df = pitchers_df.sort_values(['Pitch_Count', 'SO'], ascending=False).head(12).copy()
        recent_df = recent_df[['Formatted_Name', 'Normalized_Team', 'IP', 'SO', 'ERA', 'Pitch_Count']].copy()
        recent_df.columns = ['Pitcher', 'Team', 'IP', 'SO', 'ERA', 'Pitches']
        st.dataframe(recent_df, hide_index=True, use_container_width=True)

    if best_yesterday:
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
                    pitch_counts,
                    x='Amount Thrown',
                    y='Pitch Type',
                    orientation='h',
                    color='Pitch Type',
                    text='Amount Thrown',
                    title=f"Pitch Types Thrown by Volume: {p_name}"
                )
                fig_pitch_bar.update_layout(
                    xaxis=dict(title="Number of Pitches Thrown"),
                    yaxis=dict(title="Type of Pitch (Fastball, Slider, Curveball, etc.)"),
                    template="plotly_dark",
                    height=300,
                    showlegend=False
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
            x=strikes['plate_x'],
            y=strikes['plate_z'],
            mode='markers',
            name='Called Strike',
            marker=dict(color='#2ecc71', size=7, opacity=0.8)
        ))
        fig_scatter.add_trace(go.Scattergl(
            x=balls['plate_x'],
            y=balls['plate_z'],
            mode='markers',
            name='Ball',
            marker=dict(color='#e74c3c', size=7, opacity=0.8)
        ))
        fig_scatter.add_shape(
            type='rect',
            x0=-0.83,
            x1=0.83,
            y0=1.5,
            y1=3.5,
            line=dict(color='White', width=3, dash='dash')
        )
        fig_scatter.update_layout(
            xaxis=dict(title='Location Across Home Plate (Left to Right)', range=[-2, 2]),
            yaxis=dict(title='Height of the Pitch (Ground to Top of Zone)', range=[0.5, 4.5]),
            template='plotly_dark',
            height=420,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. PAGE 2: 📊 SINGLE PLAYER ANALYSIS (WITH REGULAR SEASON & PLAYOFFS SPLIT)
# -----------------------------------------------------------------------------
elif app_mode == "📊 Player Analysis":
    st.title(f"📊 Detailed Player Profile: {primary_pitcher}")
    st.caption(f"Full season breakdown for {selected_season}.")

    with st.spinner(f"Loading data for {primary_pitcher}..."):
        primary_data, primary_id = fetch_pitcher_statcast(primary_pitcher, selected_season)

    headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:silo:current.png/w_400,q_auto:best/v1/people/{primary_id if primary_id else 605483}/headshot/silo/current"

    if not primary_data.empty:
        if 'game_pk' in primary_data.columns and 'game_date' in primary_data.columns:
            primary_data = primary_data.copy()
            primary_data['game_date'] = pd.to_datetime(primary_data['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        if 'balls' not in primary_data.columns or 'strikes' not in primary_data.columns:
            primary_data['balls'] = 0
            primary_data['strikes'] = 0

        primary_data = primary_data.sort_values(['game_pk', 'at_bat_number', 'pitch_number'], ascending=True)
        for pa_id, pa_df in primary_data.groupby(['game_pk', 'at_bat_number']):
            ball_count = 0
            strike_count = 0
            for idx in pa_df.index:
                desc = str(primary_data.loc[idx, 'description']).lower() if 'description' in primary_data.columns else ''
                if desc in {'ball', 'blocked_ball', 'intent_ball', 'hit_by_pitch'}:
                    ball_count += 1
                elif desc in {'called_strike', 'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'bunt_foul_tip'}:
                    strike_count += 1
                primary_data.loc[idx, 'balls'] = ball_count
                primary_data.loc[idx, 'strikes'] = strike_count

        primary_data['count_state'] = 'Other'
        ahead_mask = (primary_data['strikes'] >= 2) & (primary_data['balls'] <= 0)
        behind_mask = (primary_data['balls'] >= 2) & (primary_data['strikes'] <= 0)
        primary_data.loc[ahead_mask, 'count_state'] = 'Ahead 0-2/1-2'
        primary_data.loc[behind_mask, 'count_state'] = 'Behind 2-0/3-1'

    # Separate Regular Season vs Postseason if game_type is available
    if 'game_type' in primary_data.columns:
        reg_data = primary_data[primary_data['game_type'] == 'R']
        post_data = primary_data[primary_data['game_type'].isin(['F', 'D', 'L', 'W'])]
    else:
        reg_data = primary_data
        post_data = pd.DataFrame()

    col_img, col_info = st.columns([1, 3])
    with col_img:
        st.image(headshot_url, width=160)
    with col_info:
        st.markdown(f"### **{primary_pitcher}**")
        p_metrics_reg = compute_pitcher_metrics(reg_data)
        st.markdown(f"**Primary Team:** `{p_metrics_reg['team_name']}`")

    st.divider()

    # Tabbed Breakdown for Regular Season vs Playoffs
    tab_reg, tab_post = st.tabs(["⚾ Regular Season Stats", "🏆 Postseason Stats"])

    with tab_reg:
        st.subheader("⚾ Regular Season Performance")
        if reg_data.empty:
            st.info("No regular season pitch data recorded.")
        else:
            m_reg = compute_pitcher_metrics(reg_data)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Batters Faced (TBF)", f"{m_reg['tbf']:,}")
            r2.metric("Strikeouts (K)", f"{m_reg['so']:,}")
            r3.metric("Strikeout Rate (K%)", f"{m_reg['k_pct']:.1f}%")
            r4.metric("Total Pitches", f"{m_reg['total_pitches']:,}")

    with tab_post:
        st.subheader("🏆 Postseason Performance")
        if post_data.empty:
            st.info("No postseason pitch data recorded for this season.")
        else:
            m_post = compute_pitcher_metrics(post_data)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Playoff Batters Faced", f"{m_post['tbf']:,}")
            p2.metric("Playoff Strikeouts (K)", f"{m_post['so']:,}")
            p3.metric("Playoff Strikeout Rate (K%)", f"{m_post['k_pct']:.1f}%")
            p4.metric("Playoff Pitches", f"{m_post['total_pitches']:,}")

    st.divider()

    st.subheader("🧭 Core Roadmap Modules")
    tab_arsenal, tab_recent, tab_context, tab_zone, tab_trends, tab_workload, tab_matchup = st.tabs(["🎯 Arsenal & Movement", "🗓️ Recent Starts", "🧭 Contextual Filters", "🧠 Strike Zone Grid", "📈 Trends & Whiffs", "💪 Workload & Fatigue", "⚔️ Matchup Simulator"])

    with tab_arsenal:
        if 'pitch_name' in primary_data.columns:
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

        movement_fig = build_pitch_movement_plot(primary_data)
        if not movement_fig.data:
            st.info("Movement data for this pitcher is not available in the current feed.")
        else:
            st.plotly_chart(movement_fig, use_container_width=True)

    with tab_recent:
        recent_starts = build_recent_start_log(primary_data)
        if recent_starts.empty:
            st.info("Recent start log is not available for this pitcher yet.")
        else:
            st.dataframe(recent_starts[['Date', 'Opponent', 'IP', 'K', 'BB', 'Pitches']], use_container_width=True, hide_index=True)
            selected_start = st.selectbox('Select a recent start to render its pitch map:', recent_starts.index, format_func=lambda idx: f"{recent_starts.loc[idx, 'Date']} | {recent_starts.loc[idx, 'Opponent']}")
            game_df = primary_data[primary_data['game_pk'] == recent_starts.loc[selected_start, 'game_pk']]
            if not game_df.empty:
                game_scatter = go.Figure()
                game_taken = game_df[game_df['is_taken'] == True] if 'is_taken' in game_df.columns else game_df
                strikes = game_taken[game_taken['is_called_strike'] == 1] if 'is_called_strike' in game_taken.columns else pd.DataFrame()
                balls = game_taken[game_taken['is_called_strike'] == 0] if 'is_called_strike' in game_taken.columns else pd.DataFrame()
                if not strikes.empty:
                    game_scatter.add_trace(go.Scattergl(x=strikes['plate_x'], y=strikes['plate_z'], mode='markers', name='Called Strike', marker=dict(color='#2ecc71', size=6)))
                if not balls.empty:
                    game_scatter.add_trace(go.Scattergl(x=balls['plate_x'], y=balls['plate_z'], mode='markers', name='Ball', marker=dict(color='#e74c3c', size=6)))
                game_scatter.add_shape(type='rect', x0=-0.83, x1=0.83, y0=1.5, y1=3.5, line=dict(color='White', width=2, dash='dash'))
                game_scatter.update_layout(template='plotly_dark', height=360)
                st.plotly_chart(game_scatter, use_container_width=True)

    with tab_context:
        if 'player_analysis_context_mode' not in st.session_state:
            st.session_state.player_analysis_context_mode = 'Raw totals'
        if 'player_analysis_context_split' not in st.session_state:
            st.session_state.player_analysis_context_split = 'Home'
        if 'player_analysis_park_choice' not in st.session_state:
            st.session_state.player_analysis_park_choice = 'All parks'

        context_mode = st.radio(
            'Context behavior',
            ['Raw totals', 'Home/Away splits', 'Park-adjusted'],
            horizontal=True,
            key='player_analysis_context_mode'
        )

        context_data = primary_data.copy()
        context_note = 'Raw totals'
        park_adjustment_note = ''

        if not context_data.empty:
            if context_mode == 'Home/Away splits':
                split_choice = st.selectbox(
                    'Select split:',
                    ['Home', 'Away'],
                    key='player_analysis_context_split'
                )
                if 'inning_topbot' in context_data.columns:
                    if split_choice == 'Home':
                        context_data = context_data[context_data['inning_topbot'].astype(str).str.lower().str.contains('bottom', na=False)]
                    else:
                        context_data = context_data[context_data['inning_topbot'].astype(str).str.lower().str.contains('top', na=False)]
                elif split_choice == 'Home' and 'home_team' in context_data.columns:
                    context_data = context_data[context_data['home_team'].notna() & (context_data['home_team'].astype(str) != '')]
                elif split_choice == 'Away' and 'away_team' in context_data.columns:
                    context_data = context_data[context_data['away_team'].notna() & (context_data['away_team'].astype(str) != '')]
                context_note = f'{split_choice} split'

            elif context_mode == 'Park-adjusted':
                park_choice = st.selectbox(
                    'Select park filter:',
                    ['All parks', 'Home parks only', 'Away parks only'],
                    key='player_analysis_park_choice'
                )
                if park_choice == 'Home parks only' and 'home_team' in context_data.columns:
                    context_data = context_data[context_data['home_team'].notna() & (context_data['home_team'].astype(str) != '')]
                elif park_choice == 'Away parks only' and 'away_team' in context_data.columns:
                    context_data = context_data[context_data['away_team'].notna() & (context_data['away_team'].astype(str) != '')]
                context_note = f'{park_choice} • Park-adjusted'
                park_adjustment_note = 'Venue-level park factors are applied for this view.'

        if context_mode == 'Park-adjusted':
            context_metrics = compute_park_adjusted_metrics(context_data)
        else:
            context_metrics = {
                'avg_speed': round(context_data['release_speed'].mean(), 1) if 'release_speed' in context_data.columns and not context_data.empty else 0.0,
                'avg_spin': round(context_data['release_spin_rate'].mean(), 0) if 'release_spin_rate' in context_data.columns and not context_data.empty else 0.0,
                'zone_pct': round((context_data['is_in_zone'].sum() / len(context_data) * 100) if 'is_in_zone' in context_data.columns and not context_data.empty else 0.0, 1)
            }

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Context Mode', context_mode)
        c2.metric('Avg Release Velo', f"{context_metrics['avg_speed']:.1f} mph")
        c3.metric('Avg Spin Rate', f"{context_metrics['avg_spin']:.0f} rpm")
        c4.metric('In-Zone %', f"{context_metrics['zone_pct']:.1f}%")
        if context_mode == 'Park-adjusted':
            st.caption(park_adjustment_note or context_metrics.get('park_note', ''))
        st.caption(f'{context_note} metrics based on {len(context_data)} pitches.')

    with tab_zone:
        count_state_filter = st.selectbox('Count state filter', ['All', 'Ahead', 'Behind', 'Other'])
        zone_df, zone_fig = build_zone_breakdown(primary_data, count_state_filter)
        st.plotly_chart(zone_fig, use_container_width=True)
        st.dataframe(zone_df, use_container_width=True, hide_index=True)

    with tab_trends:
        velocity_fig = build_velocity_spin_trend(primary_data)
        if velocity_fig.data:
            st.plotly_chart(velocity_fig, use_container_width=True)
        else:
            st.info('Velocity and spin trends are not available for this pitcher in the current feed.')
        whiff_df, whiff_fig = build_whiff_leaderboard(primary_data)
        if not whiff_df.empty:
            st.plotly_chart(whiff_fig, use_container_width=True)
        else:
            st.info('Whiff-rate summaries are not available yet for this pitcher.')

    with tab_workload:
        workload_df, workload_fig = build_workload_tracker(primary_data)
        if not workload_df.empty:
            st.plotly_chart(workload_fig, use_container_width=True)
            st.dataframe(workload_df, use_container_width=True, hide_index=True)
        else:
            st.info('Workload data is unavailable for this pitcher in the current feed.')

    with tab_matchup:
        matchup_candidates = []
        if 'batter_name' in primary_data.columns:
            matchup_candidates = sorted([str(v) for v in primary_data['batter_name'].dropna().astype(str).unique().tolist() if str(v).strip()])
        elif 'batter' in primary_data.columns:
            matchup_candidates = sorted([str(v) for v in primary_data['batter'].dropna().astype(str).unique().tolist() if str(v).strip()])

        if matchup_candidates:
            matchup_batter = st.selectbox('Select opposing hitter', matchup_candidates)
            matchup_df, matchup_fig = build_matchup_simulator(primary_data, matchup_batter)
            if matchup_df.empty:
                st.info('No matchup data is available for the selected hitter.')
            else:
                st.dataframe(matchup_df, use_container_width=True, hide_index=True)
                st.plotly_chart(matchup_fig, use_container_width=True)
        else:
            st.info('Batter identifiers are not available in the current feed, so the matchup simulator is ready for future batter-level data integration.')

    st.divider()

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
# 8. PAGE 3: 🔬 ADVANCED SCOUTING DASHBOARD
# -----------------------------------------------------------------------------
elif app_mode == "🔬 Advanced Scouting":
    st.title("🔬 Advanced Scouting Dashboard")
    st.caption("Interactive comparison, breakdown, zone, hitter-weakness, and workload modules for deeper pitching analysis.")

    if not pitcher_list:
        st.info("No pitchers are available for the selected season yet.")
    else:
        with st.spinner("Building scouting panels..."):
            scout_data, scout_id = fetch_pitcher_statcast(scout_pitcher, selected_season)
            scout_team_row = pitchers_df[pitchers_df['Formatted_Name'] == scout_pitcher]
            scout_team = scout_team_row.iloc[0]['Normalized_Team'] if not scout_team_row.empty and 'Normalized_Team' in scout_team_row.columns else None

        if scout_data.empty:
            st.info("No Statcast data is available for the selected pitcher yet.")
        else:
            st.subheader(f"🧠 {scout_pitcher} — Advanced Scouting View")
            comparison_df = build_statcast_comparison_matrix(pitcher_list, selected_season, limit=24)
            if not comparison_df.empty:
                st.markdown("### 📈 Statcast Comparison Matrix")
                fig_matrix = px.scatter_matrix(
                    comparison_df,
                    dimensions=['K%', 'Whiff Rate %', 'Called Strike %', 'Zone %', 'Avg Fastball Velo'],
                    color='Group',
                    hover_name='Pitcher',
                    title='Pitcher Comparison Matrix',
                    height=700,
                    template='plotly_dark'
                )
                st.plotly_chart(fig_matrix, use_container_width=True)

            st.markdown("### 🧭 Pitcher vs. Team / Division Breakdown")
            breakdown_df = build_opponent_breakdown(scout_data, scout_team)
            st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

            st.markdown("### 🎯 Hot / Cold Zone Grid Dashboard")
            zone_pitch_types = ['All'] + sorted({p for p in scout_data['pitch_name'].dropna().astype(str).tolist() if p})
            zone_pitch_type = st.selectbox('Pitch type filter', zone_pitch_types, key='zone_pitch_type')
            zone_filter = st.radio('Zone focus', ['All', 'Heart', 'Shadow', 'Chase', 'Waste'], horizontal=True, key='zone_focus')
            zone_summary, zone_fig = build_zone_dashboard(scout_data, pitch_type=zone_pitch_type, zone_filter=zone_filter)
            st.plotly_chart(zone_fig, use_container_width=True)
            st.dataframe(zone_summary, hide_index=True, use_container_width=True)

            st.markdown("### 🧱 Hitter Vulnerability / Weakness Heatmap")
            hitter_candidates = sorted({str(v) for v in scout_data['batter_name'].dropna().astype(str).tolist() if str(v).strip()}) if 'batter_name' in scout_data.columns else []
            if hitter_candidates:
                selected_hitter = st.selectbox('Select opposing hitter', hitter_candidates, key='scout_hitter')
                weakness_df, weakness_fig = build_hitter_weakness_heatmap(scout_data, selected_hitter)
                st.plotly_chart(weakness_fig, use_container_width=True)
                st.dataframe(weakness_df, hide_index=True, use_container_width=True)
            else:
                st.info("Batter-level data is not available in the current feed for this pitcher's sample.")

            st.markdown("### ⚠️ Fatigue & Workload Risk Score")
            workload_df, workload_fig = build_workload_risk(scout_data)
            if not workload_df.empty:
                latest_row = workload_df.iloc[-1]
                st.metric('Latest Risk Score', f"{latest_row['Risk Score']:.1f}")
                st.metric('Latest Risk Level', str(latest_row['Risk Level']))
                st.plotly_chart(workload_fig, use_container_width=True)
                st.dataframe(workload_df, hide_index=True, use_container_width=True)
            else:
                st.info("Workload risk data is not available for this pitcher yet.")

            # --- NEW ADVANCED MODULES INTEGRATION ---
            st.divider()
            st.subheader("🔥 Modern Pro-Franchise R&D Metrics")

            scout_tab1, scout_tab2, scout_tab3 = st.tabs(["⚙️ Active Spin & Extension", "🔀 Pitch Tunneling", "🤖 ABS Challenge Simulation"])

            with scout_tab1:
                st.markdown("#### Active Spin & Release Extension")
                st.caption("Measures true movement-contributing spin versus perceived velocity extension.")
                spin_ext_df = compute_active_spin_and_extension(scout_data)
                if not spin_ext_df.empty:
                    st.dataframe(spin_ext_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Active spin data unavailable for this selection.")

            with scout_tab2:
                st.markdown("#### Pitch Tunneling Sequences")
                st.caption("Identifies consecutive pitches that look identical out of the hand before breaking.")
                tunnel_df = compute_pitch_tunneling_sequences(scout_data)
                if not tunnel_df.empty:
                    st.dataframe(tunnel_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Tunneling sequence data unavailable.")

            with scout_tab3:
                st.markdown("#### Robot Umpire (ABS) Challenge Simulation")
                st.caption("Simulates automated ball-strike challenge outcomes versus human calls.")
                abs_sim = simulate_abs_challenge(scout_data)
                if abs_sim["total"] > 0:
                    ac1, ac2, ac3, ac4 = st.columns(4)
                    ac1.metric("Total Pitches", abs_sim["total"])
                    ac2.metric("Human Strikes", abs_sim["human_strikes"])
                    ac3.metric("ABS Strikes", abs_sim["abs_strikes"])
                    ac4.metric("Net Strike Gain", abs_sim["net_gain"])
                else:
                    st.info("ABS simulation data unavailable.")

# -----------------------------------------------------------------------------
# 9. PAGE 4: 📈 LEAGUE LEADERBOARD & SORTABLE STATCAST PREVIEW
# -----------------------------------------------------------------------------
elif app_mode == "📈 Leaderboards":
    st.title("📈 League Leaderboard & Statcast Preview")
    st.caption("Sortable leaderboard experience for K%, whiff rate, velocity, and spin-rate context.")

    if pitchers_df.empty:
        st.info("No leaderboard rows are available yet for the selected season.")
    else:
        team_filter = st.selectbox('Team filter', ['All'] + sorted([t for t in pitchers_df['Normalized_Team'].dropna().unique().tolist() if t]))
        with st.spinner('Building leaderboard preview...'):
            leaderboard_df = build_league_leaderboard(pitcher_list, selected_season, limit=16)
            # Merge Savant-style percentiles if available
            if not leaderboard_df.empty and not pitchers_df.empty:
                sample_statcast, _ = fetch_pitcher_statcast(pitcher_list[0], selected_season) if pitcher_list else (pd.DataFrame(), None)
                if not sample_statcast.empty:
                    leaderboard_df = compute_percentile_leaderboard(leaderboard_df, sample_statcast)

        if team_filter != 'All':
            leaderboard_df = leaderboard_df[leaderboard_df['Team'] == team_filter]
        leaderboard_df = leaderboard_df.sort_values('K%', ascending=False) if 'K%' in leaderboard_df.columns else leaderboard_df
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        if not leaderboard_df.empty:
            chart_df = leaderboard_df[['Pitcher', 'Whiff Rate %', 'Avg Fastball Velo']].dropna().head(10)
            if not chart_df.empty:
                fig_leader = px.bar(chart_df, x='Whiff Rate %', y='Pitcher', orientation='h', text='Whiff Rate %', title='Whiff Rate Snapshot')
                fig_leader.update_layout(template='plotly_dark', height=320, showlegend=False)
                st.plotly_chart(fig_leader, use_container_width=True)

# -----------------------------------------------------------------------------
# 10. PAGE 5: ⚔️ FULL-PAGE PLAYER COMPARISON DASHBOARD
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
# 11. PAGE 6: 🧑‍🤝‍TEAM PITCHERS (ACCURATE TEAM TOTALS & UNIQUE NAMES)
# -----------------------------------------------------------------------------
elif app_mode == "🧑‍🤝‍Team Pitchers":
    st.title(f"🧑‍🤝‍Team Pitching Staff Roster & Statistics ({selected_season} Season)")
    st.caption("Inspect all active pitchers who appeared for a selected MLB franchise during the chosen season.")

    team_list = sorted(list(set(TEAM_FULL_NAMES.values())))
    selected_team_full = st.selectbox("Select MLB Team Franchise:", team_list)

    selected_team_name = normalize_team_name(selected_team_full)

    st.subheader(f"📊 {selected_team_full} — {selected_season} Season Staff")

    team_pitchers_df = pd.DataFrame()

    if not pitchers_df.empty:
        if 'Team_Candidates' in pitchers_df.columns:
            team_pitchers_df = pitchers_df[
                pitchers_df['Team_Candidates'].apply(lambda candidates: selected_team_name in candidates)
            ].copy()
        elif 'Normalized_Team' in pitchers_df.columns:
            team_pitchers_df = pitchers_df[pitchers_df['Normalized_Team'] == selected_team_name].copy()

    if not team_pitchers_df.empty:
        team_pitchers_df = team_pitchers_df[team_pitchers_df['IP'].fillna(0) > 0].copy()

    if team_pitchers_df.empty:
        st.warning(f"No active pitching statistics recorded for {selected_team_full} in {selected_season}.")
    else:
        # Aggregate clean totals per unique player
        team_pitchers_df = team_pitchers_df.groupby('Formatted_Name', as_index=False).agg({
            'IP': 'sum',
            'SO': 'sum',
            'H': 'sum',
            'R': 'sum',
            'ER': 'sum',
            'BB': 'sum',
            'ERA': 'mean',
            'WHIP': 'mean'
        })
        team_pitchers_df['ERA'] = team_pitchers_df['ERA'].round(2)
        team_pitchers_df['WHIP'] = team_pitchers_df['WHIP'].round(2)

        total_team_ip = float(team_pitchers_df['IP'].sum())
        total_team_so = int(team_pitchers_df['SO'].sum())
        total_team_hits = int(team_pitchers_df['H'].sum())
        total_team_er = int(team_pitchers_df['ER'].sum())
        total_team_walks = int(team_pitchers_df['BB'].sum())
        
        team_era = (total_team_er * 9.0 / total_team_ip) if total_team_ip > 0 else 0.0
        team_whip = ((total_team_hits + total_team_walks) / total_team_ip) if total_team_ip > 0 else 0.0

        st.markdown(f"#### 📈 Combined Franchise Pitching Totals ({selected_season})")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Team IP", f"{total_team_ip:.1f}")
        m2.metric("Total Strikeouts (K)", f"{total_team_so:,}")
        m3.metric("Hits Allowed", f"{total_team_hits:,}")
        m4.metric("Team ERA", f"{team_era:.2f}")
        m5.metric("Team WHIP", f"{team_whip:.2f}")

        st.divider()

        st.subheader(f"📋 Complete Pitching Staff Roster ({selected_team_full} — {selected_season})")
        display_cols = [c for c in ['Formatted_Name', 'IP', 'SO', 'H', 'R', 'ER', 'BB', 'ERA', 'WHIP'] if c in team_pitchers_df.columns]
        roster_display = team_pitchers_df[display_cols].copy()
        if 'Formatted_Name' in roster_display.columns:
            roster_display.rename(columns={'Formatted_Name': 'Pitcher Name'}, inplace=True)
        
        roster_display = roster_display.sort_values(by='IP', ascending=False) if 'IP' in roster_display.columns else roster_display
        
        st.dataframe(roster_display, use_container_width=True, hide_index=True)
