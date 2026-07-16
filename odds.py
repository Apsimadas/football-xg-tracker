"""
Market-odds favorite signal, sourced from The Odds API (the-odds-api.com).

Unlike the xG model (fetch_and_predict.py), which only covers 5 tracked
European football leagues, this pulls real bookmaker consensus odds across
football, basketball and tennis for whatever competitions are actually live
today, and derives a "market favorite" per match from the odds themselves
(no xG needed). Where a match is also covered by the xG model, we flag
whether the two independent signals agree.

Requires an ODDS_API_KEY environment variable (never hardcode the key —
this repo is public). Free tier is 500 requests/month, so the league
allow-lists below are deliberately curated rather than "every active sport"
to keep daily usage low (~1 sports-list call + a handful of per-league calls).
"""

import gzip
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
LOCAL_TZ = ZoneInfo("Europe/Athens")

API_BASE = "https://api.the-odds-api.com/v4"

# Curated allow-list per sport, checked against whatever the API reports as
# "active" (in season) today — off-season leagues/tournaments are skipped
# automatically, which keeps request usage low without needing manual
# seasonal updates. Dict order = priority: if more entries are active
# simultaneously than DAILY_LEAGUE_CAP allows (see below), the highest
# -priority ones (top of the dict) win and the rest are skipped with a
# logged warning rather than silently dropped.
#
# This list intentionally goes wide (major leagues on every continent +
# every standing international tournament) but skips second-tier domestic
# divisions, domestic cups and youth/reserve/futures markets, which keeps
# typical simultaneous-active counts inside the free-tier budget (500
# requests/month) even in peak season — see DAILY_LEAGUE_CAP.
SOCCER_ALLOW = {
    # International tournaments first (highest priority: these are the
    # "διεθνείς διοργανώσεις" explicitly asked for, and only run a few
    # weeks a year each, so they rarely all overlap at once).
    "soccer_fifa_world_cup": "FIFA World Cup",
    "soccer_uefa_champs_league": "Champions League",
    "soccer_uefa_europa_league": "Europa League",
    "soccer_uefa_europa_conference_league": "Conference League",
    "soccer_conmebol_copa_libertadores": "Copa Libertadores",
    "soccer_conmebol_copa_sudamericana": "Copa Sudamericana",
    "soccer_uefa_european_championship": "Euro",
    "soccer_conmebol_copa_america": "Copa América",
    "soccer_uefa_nations_league": "UEFA Nations League",
    "soccer_fifa_club_world_cup": "FIFA Club World Cup",
    "soccer_concacaf_gold_cup": "CONCACAF Gold Cup",
    "soccer_africa_cup_of_nations": "Africa Cup of Nations",
    "soccer_fifa_world_cup_qualifiers_europe": "Προκριματικά Mundial (Ευρώπη)",
    "soccer_fifa_world_cup_qualifiers_south_america": "Προκριματικά Mundial (Ν. Αμερική)",
    # Major domestic top flights, worldwide.
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
    "soccer_greece_super_league": "Super League Ελλάδας",
    "soccer_netherlands_eredivisie": "Eredivisie",
    "soccer_portugal_primeira_liga": "Primeira Liga",
    "soccer_turkey_super_league": "Τουρκικό Πρωτάθλημα",
    "soccer_russia_premier_league": "Ρωσικό Πρωτάθλημα",
    "soccer_saudi_arabia_pro_league": "Saudi Pro League",
    "soccer_brazil_campeonato": "Brazileirão",
    "soccer_argentina_primera_division": "Primera División Αργεντινής",
    "soccer_mexico_ligamx": "Liga MX",
    "soccer_usa_mls": "MLS",
    "soccer_japan_j_league": "J League",
    "soccer_korea_kleague1": "K League 1",
}
BASKETBALL_ALLOW = {
    "basketball_nba": "NBA",
    "basketball_euroleague": "Euroleague",
    "basketball_nba_summer_league": "NBA Summer League",
    "basketball_wnba": "WNBA",
    "basketball_ncaab": "NCAAB",
}
# Tennis tournament keys rotate all year (each Slam/Masters is its own key),
# so instead of a fixed list we just take whatever tennis_atp_*/tennis_wta_*
# keys the API reports as currently active.

# Hard ceiling on soccer leagues actually queried in a single run, so a
# freak day where everything overlaps can never blow the monthly quota by
# itself. SOCCER_ALLOW order decides who gets skipped when this is hit.
DAILY_LEAGUE_CAP = 20

CLEAR_FAVORITE_THRESHOLD = 0.70
LEAN_FAVORITE_THRESHOLD = 0.60


def _get_json(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept-Encoding": "gzip",
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def fetch_active_sport_keys(api_key):
    url = f"{API_BASE}/sports/?apiKey={api_key}"
    data = _get_json(url)
    keys = set()
    for s in data:
        if s.get("active") and s["group"] in ("Soccer", "Basketball", "Tennis"):
            keys.add((s["key"], s["title"], s["group"]))
    return keys


def select_target_keys(active_keys):
    """From today's active sports, keep only the leagues we've curated,
    plus any live tennis tournament (dynamic, since those rotate).
    Returns (targets, skipped_warnings)."""
    active_by_key = {key: (title, group) for key, title, group in active_keys}

    soccer_targets = [
        (key, SOCCER_ALLOW[key], "Ποδόσφαιρο")
        for key in SOCCER_ALLOW  # dict order = priority
        if key in active_by_key and active_by_key[key][1] == "Soccer"
    ]

    warnings = []
    if len(soccer_targets) > DAILY_LEAGUE_CAP:
        skipped = soccer_targets[DAILY_LEAGUE_CAP:]
        soccer_targets = soccer_targets[:DAILY_LEAGUE_CAP]
        skipped_names = ", ".join(name for _, name, _ in skipped)
        warnings.append(
            f"Παραλείφθηκαν {len(skipped)} διοργανώσεις ποδοσφαίρου σήμερα λόγω ημερήσιου ορίου "
            f"αιτημάτων ({DAILY_LEAGUE_CAP}): {skipped_names}."
        )

    basketball_targets = [
        (key, BASKETBALL_ALLOW[key], "Μπάσκετ")
        for key in BASKETBALL_ALLOW
        if key in active_by_key and active_by_key[key][1] == "Basketball"
    ]

    tennis_targets = [
        (key, title, "Τένις")
        for key, (title, group) in active_by_key.items()
        if group == "Tennis" and (key.startswith("tennis_atp_") or key.startswith("tennis_wta_"))
    ]

    return soccer_targets + basketball_targets + tennis_targets, warnings


def fetch_odds_for_league(api_key, league_key):
    url = (
        f"{API_BASE}/sports/{league_key}/odds/"
        f"?apiKey={api_key}&regions=eu&markets=h2h&oddsFormat=decimal&dateFormat=iso"
    )
    return _get_json(url)


def implied_probabilities(event):
    """Average, per-bookmaker-devigged implied win probability per outcome
    name, across every bookmaker offering an h2h market on this event."""
    totals = {}
    counts = {}
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market["key"] != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            raw = {o["name"]: 1.0 / o["price"] for o in outcomes if o.get("price")}
            overround = sum(raw.values())
            if overround <= 0:
                continue
            for name, p in raw.items():
                devigged = p / overround
                totals[name] = totals.get(name, 0.0) + devigged
                counts[name] = counts.get(name, 0) + 1
    return {name: totals[name] / counts[name] for name in totals}


def classify_tier(fav_pct):
    if fav_pct >= CLEAR_FAVORITE_THRESHOLD:
        return "clear"
    if fav_pct >= LEAN_FAVORITE_THRESHOLD:
        return "lean"
    return "even"


def _normalize_team(name):
    return name.lower().replace(".", "").replace("-", " ").strip()


def _names_match(a, b):
    a, b = _normalize_team(a), _normalize_team(b)
    return a == b or a in b or b in a


def find_xg_agreement(home, away, xg_matches):
    """Best-effort name match against the xG model's match list; returns
    True/False if this match is also xG-covered and favorites agree/disagree,
    or None if the match isn't covered by the xG model at all."""
    for m in xg_matches:
        if _names_match(home, m["home_team"]) and _names_match(away, m["away_team"]):
            if not m["favorite_team"]:
                return None
            return _names_match(m["favorite_team"], home) or _names_match(m["favorite_team"], away)
    return None


def build_odds_matches_for_date(target_date, xg_matches, api_key):
    warnings = []
    if not api_key:
        warnings.append("Δεν έχει οριστεί ODDS_API_KEY — η ενότητα φαβορί αγοράς παραλείπεται.")
        return [], warnings

    try:
        active_keys = fetch_active_sport_keys(api_key)
    except urllib.error.HTTPError as e:
        warnings.append(f"The Odds API επέστρεψε σφάλμα {e.code} (πιθανώς έληξε το μηνιαίο όριο αιτημάτων).")
        return [], warnings
    except urllib.error.URLError as e:
        warnings.append(f"Αποτυχία σύνδεσης με The Odds API: {e}")
        return [], warnings

    targets, cap_warnings = select_target_keys(active_keys)
    warnings.extend(cap_warnings)
    results = []

    for league_key, display_name, sport_group in targets:
        try:
            events = fetch_odds_for_league(api_key, league_key)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            warnings.append(f"{display_name}: αποτυχία λήψης odds ({e}).")
            continue

        for ev in events:
            try:
                commence_utc = datetime.strptime(
                    ev["commence_time"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=UTC)
            except (KeyError, ValueError):
                continue
            commence_local = commence_utc.astimezone(LOCAL_TZ)
            if commence_local.date() != target_date:
                continue

            probs = implied_probabilities(ev)
            if not probs:
                continue

            fav_name, fav_prob = max(probs.items(), key=lambda kv: kv[1])
            tier = classify_tier(fav_prob)
            agreement = find_xg_agreement(ev.get("home_team", ""), ev.get("away_team", ""), xg_matches)

            results.append({
                "sport": sport_group,
                "league": display_name,
                "kickoff_local": commence_local.strftime("%H:%M"),
                "home_team": ev.get("home_team", "?"),
                "away_team": ev.get("away_team", "?"),
                "favorite_team": fav_name,
                "favorite_pct": round(fav_prob * 100, 1),
                "favorite_tier": tier,
                "xg_agreement": agreement,  # True | False | None
                "bookmaker_count": len(ev.get("bookmakers", [])),
            })

    tier_order = {"clear": 0, "lean": 1, "even": 2}
    results.sort(key=lambda m: (tier_order[m["favorite_tier"]], m["sport"], m["kickoff_local"]))
    return results, warnings
