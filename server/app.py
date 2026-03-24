"""FastAPI application for the N-player poker frontend."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cfr.card_abstraction import CardAbstraction
from cfr.action_abstraction import set_num_players
from server.bot import Bot
from server.game_manager import GameManager

NUM_PLAYERS = int(os.environ.get('NUM_PLAYERS', '6'))

app = FastAPI(title="Poker — CFR Solver")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pick the right strategy file based on player count
_STRATEGY_FILES = {
    2: 'strategy_2max.json.gz',
    6: 'strategy_6max.json.gz',
    9: 'strategy_9max.json.gz',
}

# Cache loaded bots per player-count tier
_bot_cache = {}

def _get_strategy_path(num_players):
    env = os.environ.get('STRATEGY_PATH')
    if env:
        return env
    if num_players <= 2:
        key = 2
    elif num_players <= 6:
        key = 6
    else:
        key = 9
    return os.path.join(ROOT, _STRATEGY_FILES[key])


def _get_bot(num_players):
    """Get or create a bot for this player-count tier. Falls back gracefully."""
    if num_players <= 2:
        tier = 2
    elif num_players <= 6:
        tier = 6
    else:
        tier = 9

    if tier in _bot_cache:
        return _bot_cache[tier]

    path = _get_strategy_path(num_players)
    if os.path.exists(path):
        try:
            b = Bot(path, card_abstraction=CardAbstraction(num_players=num_players),
                    num_players=num_players)
            _bot_cache[tier] = b
            print(f"  Loaded strategy for {tier}-max from {path}", flush=True)
            return b
        except Exception as e:
            print(f"  Failed to load {path}: {e}", flush=True)

    # Fall back: try other tiers
    for fallback in [6, 2, 9]:
        if fallback == tier:
            continue
        if fallback in _bot_cache:
            print(f"  Using {fallback}-max strategy as fallback for {tier}-max", flush=True)
            return _bot_cache[fallback]
        fb_path = os.path.join(ROOT, _STRATEGY_FILES[fallback])
        if os.path.exists(fb_path):
            try:
                b = Bot(fb_path, card_abstraction=CardAbstraction(num_players=num_players),
                        num_players=num_players)
                _bot_cache[tier] = b
                print(f"  Using {fallback}-max strategy as fallback for {tier}-max", flush=True)
                return b
            except Exception as e:
                print(f"  Failed to load fallback {fb_path}: {e}", flush=True)

    # Last resort: empty strategy (uniform random)
    print(f"  No strategy files found — bot will play uniformly random", flush=True)
    b = Bot({}, card_abstraction=CardAbstraction(num_players=num_players),
            num_players=num_players)
    _bot_cache[tier] = b
    return b


set_num_players(NUM_PLAYERS)
bot = _get_bot(NUM_PLAYERS)
game_manager = GameManager(bot, num_players=NUM_PLAYERS)


class ActionReq(BaseModel):
    action: str


class NewGameReq(BaseModel):
    player_seat: int = 0
    num_players: int = 0


@app.post("/api/game/new")
def new_game(req: NewGameReq = NewGameReq()):
    global game_manager
    n = req.num_players if req.num_players >= 2 else NUM_PLAYERS
    set_num_players(n)
    new_bot = _get_bot(n)
    if new_bot is not game_manager.bot:
        old_model = game_manager.opponent_model
        old_stats = game_manager.stats
        game_manager = GameManager(new_bot, num_players=n)
        game_manager.opponent_model = old_model
        for seat, stat in old_stats.items():
            if seat < n:
                game_manager.stats[seat] = stat
    _, state = game_manager.new_game(player_seat=req.player_seat, num_players=n)
    return state


@app.post("/api/game/{sid}/action")
def action(sid: str, req: ActionReq):
    result = game_manager.player_action(sid, req.action)
    if result is None:
        raise HTTPException(404, "Game not found")
    if isinstance(result, dict) and 'error' in result:
        raise HTTPException(400, result['error'])
    return result


@app.get("/api/game/{sid}")
def state(sid: str):
    result = game_manager.get_state(sid)
    if result is None:
        raise HTTPException(404, "Game not found")
    return result


@app.get("/api/game/{sid}/range")
def range_strategy(sid: str):
    result = game_manager.get_range_strategy(sid)
    if result is None:
        raise HTTPException(404, "Game not found")
    return result


@app.get("/api/game/{sid}/live")
def live_recommendation(sid: str):
    """Get real-time solver recommendation for the player's current decision."""
    result = game_manager.get_live_recommendation(sid)
    return result


@app.get("/api/stats")
def player_stats():
    """Get VPIP/PFR stats for all seats."""
    return game_manager.get_player_stats()


@app.get("/api/exploit/{seat}")
def exploit_stats(seat: int):
    """Get opponent model stats and exploit adjustments for a seat."""
    stats = game_manager.opponent_model.get_stats(seat)
    adj = game_manager.opponent_model.get_exploit_adjustments(seat)
    return {"stats": stats, "adjustments": adj}


static_dir = os.path.join(ROOT, 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, 'index.html'))
