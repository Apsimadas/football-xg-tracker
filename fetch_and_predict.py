"""
Football xG daily tracker.

Fetches season data (per-match xG history + fixture list) from understat.com
for a configurable set of leagues, computes each team's home/away attack and
defense strength relative to the league average (based on Expected Goals),
and uses a Poisson model to estimate, for each match scheduled "today":

  - a predicted final score
  - home/draw/away win probabilities (and whether there is a clear favorite)
  - Over/Under 2.5 goals probability

Output: a JSON file (matches.json) and a self-contained HTML dashboard
(dashboard.html) built from dashboard_template.html.

Data source note: understat.com only publishes data for Big-5-style European
leagues (see LEAGUES below) and only once a season's fixture list has been
released (typically ~1 month before a season starts). During the summer
close season there will be no matches to report for these leagues, which is
expected, not a bug.
"""

import argparse
import gzip
import json
import math
import os
import urllib.error
import urllib.request
from datetime import datetime, date
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
LOCAL_TZ = ZoneInfo("Europe/Athens")

# Slug -> display name. understat's classic "big 5 + Russia" league set.
LEAGUES = {
    "EPL": "Premier League",
    "La_Liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Accept-Encoding": "gzip",
}

# Minimum home/away matches played before we trust a team's xG averages.
MIN_SAMPLE_MATCHES = 3

# Win-probability threshold to flag a match as having a "clear favorite".
CLEAR_FAVORITE_THRESHOLD = 0.60
STRONG_FAVORITE_THRESHOLD = 0.70

OU_LINE = 2.5
OU_LEAN_THRESHOLD = 0.55

MAX_GOALS = 8  # Poisson matrix truncation


def season_for_date(d):
    """European season is named by its start year (Aug-May)."""
    return d.year if d.month >= 7 else d.year - 1


def fetch_league_data(league_slug, season):
    url = f"https://understat.com/getLeagueData/{league_slug}/{season}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return json.loads(raw.decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"  ! failed to fetch {league_slug}/{season}: {e}")
        return {"teams": {}, "players": [], "dates": []}


def compute_team_stats(teams, as_of_dt):
    """Per-team home/away attack & defense xG averages using matches
    strictly before as_of_dt (avoids leaking future data into a prediction).
    Also returns league-wide average home/away xG."""
    home_scored, away_scored = {}, {}
    home_conceded, away_conceded = {}, {}
    all_home_xg, all_away_xg = [], []

    for team_id, team in teams.items():
        for m in team.get("history", []):
            try:
                m_dt = datetime.strptime(m["date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            except (KeyError, ValueError):
                continue
            if m_dt >= as_of_dt:
                continue
            xg, xga = float(m["xG"]), float(m["xGA"])
            if m["h_a"] == "h":
                home_scored.setdefault(team_id, []).append(xg)
                home_conceded.setdefault(team_id, []).append(xga)
                all_home_xg.append(xg)
            else:
                away_scored.setdefault(team_id, []).append(xg)
                away_conceded.setdefault(team_id, []).append(xga)
                all_away_xg.append(xg)

    def avg(lst):
        return sum(lst) / len(lst) if lst else None

    league_avg_home_xg = avg(all_home_xg) or 1.4
    league_avg_away_xg = avg(all_away_xg) or 1.2

    stats = {}
    for team_id, team in teams.items():
        hs, hc = home_scored.get(team_id, []), home_conceded.get(team_id, [])
        as_, ac = away_scored.get(team_id, []), away_conceded.get(team_id, [])
        stats[team_id] = {
            "title": team["title"],
            "home_attack": avg(hs) or league_avg_home_xg,
            "home_defense": avg(hc) or league_avg_away_xg,
            "away_attack": avg(as_) or league_avg_away_xg,
            "away_defense": avg(ac) or league_avg_home_xg,
            "home_matches": len(hs),
            "away_matches": len(as_),
        }
    return stats, league_avg_home_xg, league_avg_away_xg


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def predict_match(home_stats, away_stats, league_avg_home_xg, league_avg_away_xg):
    home_attack_strength = home_stats["home_attack"] / league_avg_home_xg
    home_defense_strength = home_stats["home_defense"] / league_avg_away_xg
    away_attack_strength = away_stats["away_attack"] / league_avg_away_xg
    away_defense_strength = away_stats["away_defense"] / league_avg_home_xg

    lam_home = league_avg_home_xg * home_attack_strength * away_defense_strength
    lam_away = league_avg_away_xg * away_attack_strength * home_defense_strength
    # keep sane bounds
    lam_home = max(0.15, min(lam_home, 5.0))
    lam_away = max(0.15, min(lam_away, 5.0))

    home_pmf = [poisson_pmf(i, lam_home) for i in range(MAX_GOALS + 1)]
    away_pmf = [poisson_pmf(j, lam_away) for j in range(MAX_GOALS + 1)]

    best_score, best_p = (0, 0), -1.0
    home_win = draw = away_win = 0.0
    over = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = home_pmf[i] * away_pmf[j]
            if p > best_p:
                best_p, best_score = p, (i, j)
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
            if i + j > OU_LINE:
                over += p
    under = 1.0 - over

    return {
        "lambda_home": round(lam_home, 2),
        "lambda_away": round(lam_away, 2),
        "predicted_score": f"{best_score[0]}-{best_score[1]}",
        "home_win_pct": round(home_win * 100, 1),
        "draw_pct": round(draw * 100, 1),
        "away_win_pct": round(away_win * 100, 1),
        "over_2_5_pct": round(over * 100, 1),
        "under_2_5_pct": round(under * 100, 1),
    }


def classify_favorite(home_win_pct, away_win_pct, home_team, away_team):
    home_win, away_win = home_win_pct / 100, away_win_pct / 100
    fav_team, fav_pct = (home_team, home_win) if home_win >= away_win else (away_team, away_win)
    if fav_pct >= STRONG_FAVORITE_THRESHOLD:
        return fav_team, fav_pct, "clear"
    if fav_pct >= CLEAR_FAVORITE_THRESHOLD:
        return fav_team, fav_pct, "lean"
    return None, fav_pct, "even"


def classify_ou(over_pct):
    over = over_pct / 100
    if over >= OU_LEAN_THRESHOLD:
        return "Over", over
    if (1 - over) >= OU_LEAN_THRESHOLD:
        return "Under", 1 - over
    return "Toss-up", max(over, 1 - over)


def build_matches_for_date(target_date, leagues=LEAGUES):
    """target_date: a date (local Europe/Athens calendar day) to report on."""
    all_matches = []
    warnings = []

    for slug, display_name in leagues.items():
        season = season_for_date(target_date)
        print(f"Fetching {display_name} ({slug}), season {season}...")
        data = fetch_league_data(slug, season)
        teams = data.get("teams", {})
        fixtures = data.get("dates", [])
        if not teams or not fixtures:
            warnings.append(f"{display_name}: δεν υπάρχουν ακόμη δημοσιευμένα δεδομένα για τη σεζόν {season}/{season+1}.")
            continue

        todays_fixtures = []
        for fx in fixtures:
            try:
                fx_dt_utc = datetime.strptime(fx["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            except (KeyError, ValueError):
                continue
            fx_dt_local = fx_dt_utc.astimezone(LOCAL_TZ)
            if fx_dt_local.date() == target_date:
                todays_fixtures.append((fx, fx_dt_utc, fx_dt_local))

        if not todays_fixtures:
            continue

        # Use the earliest of today's kickoffs as the "as of" cutoff for stats,
        # so later matches the same day don't peek at earlier same-day results.
        cutoff = min(f[1] for f in todays_fixtures)
        stats, lg_home_xg, lg_away_xg = compute_team_stats(teams, cutoff)

        for fx, fx_dt_utc, fx_dt_local in todays_fixtures:
            home_id, away_id = fx["h"]["id"], fx["a"]["id"]
            home_stats = stats.get(home_id)
            away_stats = stats.get(away_id)
            if not home_stats or not away_stats:
                continue

            low_sample = (
                home_stats["home_matches"] < MIN_SAMPLE_MATCHES
                or away_stats["away_matches"] < MIN_SAMPLE_MATCHES
            )

            pred = predict_match(home_stats, away_stats, lg_home_xg, lg_away_xg)
            fav_team, fav_pct, fav_tier = classify_favorite(
                pred["home_win_pct"], pred["away_win_pct"], fx["h"]["title"], fx["a"]["title"]
            )
            ou_lean, ou_pct = classify_ou(pred["over_2_5_pct"])

            actual_score = None
            if fx.get("isResult") and fx.get("goals"):
                g = fx["goals"]
                if g.get("h") is not None and g.get("a") is not None:
                    actual_score = f"{g['h']}-{g['a']}"

            all_matches.append({
                "league": display_name,
                "kickoff_local": fx_dt_local.strftime("%H:%M"),
                "kickoff_iso": fx_dt_local.isoformat(),
                "home_team": fx["h"]["title"],
                "away_team": fx["a"]["title"],
                **pred,
                "favorite_team": fav_team,
                "favorite_pct": round(fav_pct * 100, 1),
                "favorite_tier": fav_tier,  # "clear" | "lean" | "even"
                "ou_lean": ou_lean,
                "ou_pct": round(ou_pct * 100, 1),
                "low_sample": low_sample,
                "actual_score": actual_score,
            })

    # Clear favorites first, then by kickoff time.
    tier_order = {"clear": 0, "lean": 1, "even": 2}
    all_matches.sort(key=lambda m: (tier_order[m["favorite_tier"]], m["kickoff_local"]))
    return all_matches, warnings


def render_html(matches, warnings, target_date, generated_at_local, template_path, output_path):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    payload = {
        "targetDate": target_date.isoformat(),
        "generatedAt": generated_at_local.isoformat(),
        "matches": matches,
        "warnings": warnings,
    }
    html = template.replace("__MATCH_DATA_JSON__", json.dumps(payload, ensure_ascii=False))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Daily football xG favorite/score tracker")
    parser.add_argument("--date", help="YYYY-MM-DD (local date) to report on; defaults to today", default=None)
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(LOCAL_TZ).date()

    print(f"Building report for {target_date.isoformat()}...")
    matches, warnings = build_matches_for_date(target_date)

    here = os.path.dirname(os.path.abspath(__file__))
    matches_json_path = os.path.join(here, "matches.json")
    with open(matches_json_path, "w", encoding="utf-8") as f:
        json.dump({"targetDate": target_date.isoformat(), "matches": matches, "warnings": warnings}, f,
                   ensure_ascii=False, indent=2)
    print(f"Wrote {matches_json_path} ({len(matches)} matches)")

    render_html(
        matches, warnings, target_date, datetime.now(LOCAL_TZ),
        template_path=os.path.join(here, "dashboard_template.html"),
        output_path=os.path.join(here, "dashboard.html"),
    )


if __name__ == "__main__":
    main()
