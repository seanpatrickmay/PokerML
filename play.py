#!/usr/bin/env python3
"""CLI poker game — play heads-up against the CFR bot in your terminal."""

import os
import sys
import random
import time
import threading

import numpy as np

from cfr.card_abstraction import CardAbstraction
from cfr.game_state import GameState, STARTING_STACK, get_position_name
from cfr.strategy_store import load_strategy
from cfr.evaluator import determine_winners, enumerate_river_equity
from cfr.subgame_solver import SubgameSolver
from server.bot import Bot

# ── Card rendering ─────────────────────────────────────────────────

RANKS = "23456789TJQKA"
SUITS = "cdhs"
SUIT_SYM = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def card_str(c):
    r = RANKS[c // 4]
    s = SUITS[c % 4]
    sym = SUIT_SYM[s]
    if s in ("h", "d"):
        return f"{RED}{r}{sym}{RESET}"
    return f"{r}{sym}"


def hand_str(cards):
    return " ".join(card_str(c) for c in cards)


def board_str(board):
    if not board:
        return f"{DIM}(none){RESET}"
    return " ".join(card_str(c) for c in board)


# ── Display helpers ────────────────────────────────────────────────

def clear():
    print("\033[2J\033[H", end="")


def action_label(a, pot, to_call, stack):
    if a == "f":
        return "Fold"
    if a == "k":
        return "Check"
    if a == "c":
        return f"Call {min(to_call, stack):.1f}"
    if a == "a":
        return f"All-In {stack:.1f}"
    if a.startswith("b"):
        pct = int(a[1:])
        from cfr.action_abstraction import action_to_chips
        chips, _ = action_to_chips(a, pot, to_call, stack)
        return f"Bet {pct}% ({chips:.1f})"
    return a


def print_table(state, hands, hero_seat, street_names, show_opp=False):
    n = state.num_players
    vis = state.visible_board()
    pot = state.pot + sum(state.bets)
    street = street_names[min(state.street, 3)]

    print(f"\n  {BOLD}{'─' * 50}{RESET}")
    print(f"  {BOLD}{street}{RESET}  │  Pot: {YELLOW}{pot:.1f}{RESET} BB")
    print(f"  {BOLD}{'─' * 50}{RESET}")
    print(f"  Board: {board_str(vis)}")
    print()

    for i in range(n):
        pos = get_position_name(n, i)
        is_hero = i == hero_seat
        folded = state.folded[i] if isinstance(state.folded, (list, tuple)) else bool((state.folded >> i) & 1)
        all_in = state.all_in[i] if isinstance(state.all_in, (list, tuple)) else bool((state.all_in >> i) & 1)

        if is_hero:
            tag = f"{GREEN}YOU{RESET}"
            cards = hand_str(hands[i])
        elif show_opp or state.is_terminal:
            tag = f"{RED}BOT{RESET}"
            cards = hand_str(hands[i])
        else:
            tag = f"{RED}BOT{RESET}"
            cards = f"{DIM}?? ??{RESET}"

        status = ""
        if folded:
            status = f" {DIM}(folded){RESET}"
        elif all_in:
            status = f" {YELLOW}(all-in){RESET}"
        elif not state.is_terminal and state.current_player == i:
            status = f" {CYAN}← acting{RESET}"

        bet_str = f"  bet: {state.bets[i]:.1f}" if state.bets[i] > 0 else ""
        print(f"  {pos:>4} {tag}  {cards}  stack: {state.stacks[i]:.1f}{bet_str}{status}")

    print()


# ── Bot thinking display ───────────────────────────────────────────

def show_bot_thinking(bot, hand, state, seat):
    """Show real-time solver progress while bot thinks."""
    vis = state.visible_board()
    actions = state.get_actions()

    if state.street == 0:
        # Preflop: instant
        print(f"  {DIM}Bot using preflop ranges...{RESET}")
        action = bot.get_action(hand, vis, state.history, actions,
                                state=state, seat=seat)
        return action

    # Postflop: show solver progress
    solver = bot.solver
    ca = bot.card_abstraction
    bucket = ca.get_bucket(hand, vis)
    pos = get_position_name(state.num_players, seat)
    history_str = '/'.join(','.join(s) for s in state.history)
    info_key = f'{pos}:{bucket}|{history_str}'

    # Start solving in background
    solve_result = [None]
    solve_time = [0.0]

    def solve_thread():
        t0 = time.time()
        solved = solver.solve(state, seat)
        solve_time[0] = time.time() - t0
        solve_result[0] = solved

    t = threading.Thread(target=solve_thread, daemon=True)
    t.start()

    # Show progress while solving
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while t.is_alive():
        elapsed = time.time() - (time.time() - 0.1)
        frame = spinner[i % len(spinner)]
        print(f"\r  {CYAN}{frame} Bot solving... {i * 0.1:.1f}s{RESET}   ", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print(f"\r  {GREEN}✓ Solved in {solve_time[0]:.2f}s{RESET}              ")

    # Now get the action through the full bot pipeline
    action = bot.get_action(hand, vis, state.history, actions,
                            state=state, seat=seat)

    # Show the bot's strategy
    from cfr.game_state import lookup_with_fallback
    solved = solve_result[0]
    if solved:
        probs = lookup_with_fallback(solved, info_key, state.num_players)
        if probs and len(probs) == len(actions):
            print(f"  {DIM}Solver raw strategy:{RESET}")
            for j, a in enumerate(actions):
                bar_len = int(probs[j] * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                pct = probs[j] * 100
                lbl = action_label(a, state.pot + sum(state.bets),
                                   max(state.bets) - state.bets[seat],
                                   state.stacks[seat])
                highlight = f"{BOLD}" if a == action else ""
                end_hl = f"{RESET}" if a == action else ""
                print(f"    {highlight}{lbl:>20} {bar} {pct:5.1f}%{end_hl}")
            print()

    return action


# ── Equity display ─────────────────────────────────────────────────

def show_hero_equity(hand, board, ca):
    vis = tuple(board) if board else ()
    if len(vis) == 5:
        eq = enumerate_river_equity(hand, vis)
        print(f"  Your equity: {BOLD}{eq*100:.1f}%{RESET}")
    elif len(vis) >= 3:
        bucket = ca.get_bucket(hand, vis)
        eq_proxy = (bucket + 0.5) / ca.num_postflop_buckets * 100
        print(f"  Your equity (est): {BOLD}{eq_proxy:.0f}%{RESET}")


# ── Main game loop ─────────────────────────────────────────────────

def main():
    strategy_file = "strategy_2max.json.gz"
    if not os.path.exists(strategy_file):
        print(f"Strategy file not found: {strategy_file}")
        print("Run `python3 train.py` first.")
        sys.exit(1)

    print(f"{BOLD}Loading strategy...{RESET}", end=" ", flush=True)
    t0 = time.time()
    strategy = load_strategy(strategy_file)
    ca = CardAbstraction(num_players=2, use_emd=False)
    bot = Bot(strategy, card_abstraction=ca, num_players=2)
    print(f"{GREEN}done{RESET} ({time.time()-t0:.1f}s, {len(strategy):,} info sets)")

    street_names = ["Preflop", "Flop", "Turn", "River"]
    hero_seat = 1  # BB by default
    hands_played = 0
    total_profit = 0.0

    print(f"\n{BOLD}Heads-Up No-Limit Hold'em{RESET}")
    print(f"Starting stack: {STARTING_STACK} BB")
    print(f"You are {GREEN}BB (seat 1){RESET}, bot is {RED}BTN/SB (seat 0){RESET}")
    print(f"Type action number to play, 'q' to quit, 's' to switch seats\n")

    while True:
        # Deal
        deck = list(range(52))
        random.shuffle(deck)
        hands = ((deck[0], deck[1]), (deck[2], deck[3]))
        board = tuple(deck[4:9])
        state = GameState.new_hand(hands, board, num_players=2)

        hands_played += 1
        print(f"{BOLD}{'═' * 55}{RESET}")
        print(f"  {BOLD}Hand #{hands_played}{RESET}  │  "
              f"Profit: {'+' if total_profit >= 0 else ''}{total_profit:.1f} BB")
        print(f"{BOLD}{'═' * 55}{RESET}")

        # Clear bot's resolved cache for fresh hand
        if hasattr(bot, '_resolved_cache'):
            bot._resolved_cache.clear()

        hand_over = False
        last_street = -1

        while not state.is_terminal:
            # Show new street transition
            if state.street != last_street:
                last_street = state.street
                print_table(state, hands, hero_seat, street_names)
                if state.street > 0:
                    show_hero_equity(hands[hero_seat], state.visible_board(), ca)

            if state.current_player == hero_seat:
                # Hero's turn
                actions = state.get_actions()
                pot = state.pot + sum(state.bets)
                max_bet = max(state.bets)
                to_call = max_bet - state.bets[hero_seat]

                print(f"  {BOLD}Your action:{RESET}")
                for idx, a in enumerate(actions):
                    lbl = action_label(a, pot, to_call, state.stacks[hero_seat])
                    print(f"    {CYAN}{idx + 1}{RESET}) {lbl}")

                while True:
                    try:
                        raw = input(f"\n  {BOLD}> {RESET}").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\nGoodbye!")
                        return

                    if raw == "q":
                        print(f"\n{BOLD}Final: {hands_played} hands, "
                              f"{'+' if total_profit >= 0 else ''}{total_profit:.1f} BB{RESET}")
                        return
                    if raw == "s":
                        hero_seat = 1 - hero_seat
                        pos = "BB" if hero_seat == 1 else "BTN"
                        print(f"  Switched to {pos} (seat {hero_seat})")
                        break

                    try:
                        choice = int(raw) - 1
                        if 0 <= choice < len(actions):
                            chosen = actions[choice]
                            lbl = action_label(chosen, pot, to_call,
                                               state.stacks[hero_seat])
                            print(f"  → You: {GREEN}{lbl}{RESET}")
                            state = state.apply_action(chosen)
                            break
                        else:
                            print(f"  Pick 1-{len(actions)}")
                    except ValueError:
                        # Try direct action input
                        if raw in actions:
                            state = state.apply_action(raw)
                            break
                        print(f"  Pick 1-{len(actions)} or type action (f/k/c/b100/a)")
            else:
                # Bot's turn
                bot_seat = state.current_player
                bot_hand = hands[bot_seat]
                action = show_bot_thinking(bot, bot_hand, state, bot_seat)
                pot = state.pot + sum(state.bets)
                max_bet = max(state.bets)
                to_call = max_bet - state.bets[bot_seat]
                lbl = action_label(action, pot, to_call, state.stacks[bot_seat])
                print(f"  → Bot: {RED}{lbl}{RESET}")
                state = state.apply_action(action)

        # Hand over — show results
        print_table(state, hands, hero_seat, street_names, show_opp=True)

        profit = state.get_terminal_utility(hero_seat)
        total_profit += profit

        if state.terminal_type == "fold":
            active = [i for i in range(2) if not state.folded[i]]
            winner = active[0]
            if winner == hero_seat:
                print(f"  {GREEN}{BOLD}You win! Opponent folded. (+{profit:.1f} BB){RESET}")
            else:
                print(f"  {RED}Bot wins. You folded. ({profit:.1f} BB){RESET}")
        else:
            # Showdown
            winners = determine_winners(board[:5], hands, [0, 1])
            if hero_seat in winners:
                if len(winners) == 1:
                    print(f"  {GREEN}{BOLD}You win at showdown! (+{profit:.1f} BB){RESET}")
                else:
                    print(f"  {YELLOW}Split pot. ({profit:+.1f} BB){RESET}")
            else:
                print(f"  {RED}Bot wins at showdown. ({profit:.1f} BB){RESET}")

        print(f"  Running: {hands_played} hands, {'+' if total_profit >= 0 else ''}{total_profit:.1f} BB "
              f"({total_profit/hands_played:+.2f} BB/hand)")

        # Next hand prompt
        try:
            raw = input(f"\n  {DIM}Press Enter for next hand (q=quit, s=switch seats)...{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if raw == "q":
            break
        if raw == "s":
            hero_seat = 1 - hero_seat
            pos = "BB" if hero_seat == 1 else "BTN"
            print(f"  Switched to {pos}")

    print(f"\n{BOLD}Session: {hands_played} hands, "
          f"{'+' if total_profit >= 0 else ''}{total_profit:.1f} BB "
          f"({total_profit/hands_played:+.2f} BB/hand){RESET}")


if __name__ == "__main__":
    main()
