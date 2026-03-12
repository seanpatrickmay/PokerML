"""FastAPI application for the HUNL poker frontend."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cfr.card_abstraction import CardAbstraction
from server.bot import Bot
from server.game_manager import GameManager

app = FastAPI(title="HUNL Poker")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_PATH = os.environ.get(
    'STRATEGY_PATH', os.path.join(ROOT, 'strategy.json.gz'))

abstraction = CardAbstraction()
bot = Bot(STRATEGY_PATH, abstraction)
game_manager = GameManager(bot)


class ActionReq(BaseModel):
    action: str


@app.post("/api/game/new")
def new_game():
    _, state = game_manager.new_game()
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


static_dir = os.path.join(ROOT, 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, 'index.html'))
