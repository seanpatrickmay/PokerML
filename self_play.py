#!/usr/bin/env python3
"""Self-play monitor — two bot instances play each other indefinitely.
Tracks strategy frequencies, detects imbalances, and logs anomalies.
Uses duplicate matching (play each deal from both seats) for lower-variance results."""

import os, sys, random, time, json, uuid
import numpy as np
from collections import defaultdict, Counter

from cfr.card_abstraction import CardAbstraction
from cfr.game_state import GameState, STARTING_STACK, get_position_name
from cfr.strategy_store import load_strategy
from cfr.evaluator import determine_winners
from cfr.action_abstraction import action_to_chips
from server.bot import Bot

RANKS = "23456789TJQKA"

def card_name(c):
    return RANKS[c // 4] + "cdhs"[c % 4]

class SelfPlayMonitor:
    def __init__(self, bot, ca):
        self.bot = bot
        self.ca = ca
        self.hands_played = 0
        self.deals_played = 0
        self.profit = [0.0, 0.0]  # per seat (raw)
        self.raw_profits = []  # per-hand seat-0 profit
        self.duplicate_profits = []  # averaged (seat-swapped) profit per deal
        self.street_actions = defaultdict(lambda: defaultdict(int))
        self.position_actions = defaultdict(lambda: defaultdict(int))
        self.action_log = []
        self.anomalies = []
        self.last_report = 0
        self.gto_targets = {
            ('flop', 'OOP', 'first_to_act'): {'check': (0.60, 0.80), 'bet': (0.20, 0.40)},
            ('flop', 'IP', 'first_to_act'): {'check': (0.30, 0.50), 'bet': (0.50, 0.70)},
            ('flop', 'OOP', 'facing_bet'): {'fold': (0.30, 0.50), 'call': (0.35, 0.55), 'raise': (0.05, 0.18)},
            ('turn', 'OOP', 'first_to_act'): {'check': (0.55, 0.75), 'bet': (0.25, 0.45)},
            ('turn', 'IP', 'first_to_act'): {'check': (0.30, 0.50), 'bet': (0.50, 0.70)},
            ('river', 'OOP', 'first_to_act'): {'check': (0.50, 0.75), 'bet': (0.25, 0.50)},
        }

    def _clear_bot_cache(self):
        """Clear the bot's board-specific caches between hands."""
        if hasattr(self.bot, '_resolved_cache'):
            self.bot._resolved_cache.clear()
        if hasattr(self.bot, '_cache_board'):
            self.bot._cache_board = None

    def _play_single_hand(self, hands, board):
        """Play one hand with the given card assignment. Returns (actions, final_state)."""
        state = GameState.new_hand(hands, board, num_players=2)
        self._clear_bot_cache()

        hand_actions = []
        moves = 0
        while not state.is_terminal and moves < 50:
            seat = state.current_player
            hand = hands[seat]
            actions = state.get_actions()
            if not actions:
                break

            action = self.bot.get_action(
                hand, state.visible_board(), state.history,
                actions, state=state, seat=seat)

            street_name = ['preflop', 'flop', 'turn', 'river'][min(state.street, 3)]
            pos = 'OOP' if seat == 1 else 'IP'
            if state.street == 0:
                pos = 'BTN' if seat == 0 else 'BB'

            facing_bet = 'f' in actions
            situation = 'facing_bet' if facing_bet else 'first_to_act'

            is_bet_action = action.startswith('b') or action == 'a'
            if action == 'f':
                act_type = 'fold'
            elif action == 'k':
                act_type = 'check'
            elif action == 'c':
                act_type = 'call'
            elif is_bet_action and facing_bet:
                act_type = 'raise'
            elif is_bet_action:
                act_type = 'bet'
            else:
                act_type = action

            key = (street_name, pos, situation)
            self.street_actions[key][act_type] += 1
            self.position_actions[(street_name, pos)][act_type] += 1

            hand_actions.append({
                'seat': seat, 'street': street_name,
                'pos': pos, 'action': action, 'act_type': act_type,
                'situation': situation,
            })

            state = state.apply_action(action)
            moves += 1

        return hand_actions, state

    def play_hand(self):
        """Deal cards, play from both seats (duplicate matching), return results."""
        deck = list(range(52))
        random.shuffle(deck)
        hand_a = (deck[0], deck[1])
        hand_b = (deck[2], deck[3])
        board = tuple(deck[4:9])
        dup_id = uuid.uuid4().hex[:12]

        # Play 1: original assignment (seat 0 = hand_a, seat 1 = hand_b)
        hands_original = (hand_a, hand_b)
        actions_1, state_1 = self._play_single_hand(hands_original, board)
        self.hands_played += 1

        profit_1_seat0 = 0.0
        if state_1.is_terminal:
            profit_1_seat0 = state_1.get_terminal_utility(0)
            self.profit[0] += profit_1_seat0
            self.profit[1] += state_1.get_terminal_utility(1)
        self.raw_profits.append(profit_1_seat0)

        # Play 2: swapped assignment (seat 0 = hand_b, seat 1 = hand_a)
        hands_swapped = (hand_b, hand_a)
        actions_2, state_2 = self._play_single_hand(hands_swapped, board)
        self.hands_played += 1

        profit_2_seat0 = 0.0
        if state_2.is_terminal:
            profit_2_seat0 = state_2.get_terminal_utility(0)
            self.profit[0] += profit_2_seat0
            self.profit[1] += state_2.get_terminal_utility(1)
        self.raw_profits.append(profit_2_seat0)

        # Duplicate-matched result: average seat-0 profit across both plays
        dup_profit = (profit_1_seat0 + profit_2_seat0) / 2.0
        self.duplicate_profits.append(dup_profit)
        self.deals_played += 1

        return (actions_1, state_1, actions_2, state_2, dup_id)

    def bootstrap_ci(self, n_bootstrap=1000):
        """Compute 95% CI on BTN profit via bootstrap resampling."""
        if len(self.duplicate_profits) < 2:
            return (0.0, 0.0)
        data = np.array(self.duplicate_profits)
        rng = np.random.default_rng()
        means = np.empty(n_bootstrap)
        n = len(data)
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            means[i] = sample.mean()
        lower = float(np.percentile(means, 2.5))
        upper = float(np.percentile(means, 97.5))
        return (lower, upper)

    def check_frequencies(self):
        """Check all tracked frequencies against GTO targets."""
        issues = []
        for key, targets in self.gto_targets.items():
            counts = self.street_actions[key]
            total = sum(counts.values())
            if total < 50:
                continue
            for act_type, (lo, hi) in targets.items():
                freq = counts.get(act_type, 0) / total
                if freq < lo - 0.05:
                    issues.append(f"LOW {key[0]} {key[1]} {key[2]} {act_type}: {freq:.1%} (target {lo:.0%}-{hi:.0%}, n={total})")
                elif freq > hi + 0.05:
                    issues.append(f"HIGH {key[0]} {key[1]} {key[2]} {act_type}: {freq:.1%} (target {lo:.0%}-{hi:.0%}, n={total})")
        return issues

    def report(self):
        """Print frequency report with raw and duplicate-matched win rates."""
        print(f"\n{'=' * 70}")
        print(f"  SELF-PLAY REPORT -- {self.hands_played} hands ({self.deals_played} deals)")

        # Raw win rate
        raw_rate = self.profit[0] / self.hands_played if self.hands_played > 0 else 0
        print(f"\n  RAW RESULTS:")
        print(f"    Seat 0 (BTN) profit: {self.profit[0]:+.1f} BB ({raw_rate:+.3f} BB/hand)")
        print(f"    Seat 1 (BB)  profit: {self.profit[1]:+.1f} BB ({self.profit[1]/self.hands_played:+.3f} BB/hand)")

        # Duplicate-matched win rate
        if self.deals_played > 0:
            dup_mean = np.mean(self.duplicate_profits)
            dup_std = np.std(self.duplicate_profits, ddof=1) if self.deals_played > 1 else 0
            raw_std = np.std(self.raw_profits, ddof=1) if len(self.raw_profits) > 1 else 0
            variance_reduction = (1 - (dup_std**2 / raw_std**2)) * 100 if raw_std > 0 else 0

            print(f"\n  DUPLICATE-MATCHED RESULTS ({self.deals_played} deals):")
            print(f"    BTN profit (avg of seat-swapped pairs): {dup_mean:+.3f} BB/hand")
            print(f"    Std dev:  raw={raw_std:.3f}  dup={dup_std:.3f}  (variance reduction: {variance_reduction:.0f}%)")

            ci_lo, ci_hi = self.bootstrap_ci()
            print(f"    95% CI (bootstrap): [{ci_lo:+.3f}, {ci_hi:+.3f}] BB/hand")

        print(f"\n{'=' * 70}")

        print(f"\n  {'Scenario':<40} {'Freq':>8} {'GTO':>12} {'Status':>8}")
        print(f"  {'-'*40} {'-'*8} {'-'*12} {'-'*8}")

        for key in sorted(self.street_actions.keys()):
            counts = self.street_actions[key]
            total = sum(counts.values())
            if total < 20:
                continue
            label = f"{key[0]} {key[1]} {key[2]}"
            for act_type in ['check', 'bet', 'call', 'fold', 'raise']:
                freq = counts.get(act_type, 0) / total
                if freq < 0.01:
                    continue
                targets = self.gto_targets.get(key, {})
                if act_type in targets:
                    lo, hi = targets[act_type]
                    if freq < lo - 0.05:
                        status = "LOW"
                    elif freq > hi + 0.05:
                        status = "HIGH"
                    else:
                        status = "OK"
                    gto_str = f"{lo:.0%}-{hi:.0%}"
                else:
                    status = ""
                    gto_str = ""
                print(f"  {label + ' ' + act_type:<40} {freq:>7.1%} {gto_str:>12} {status:>8}")

        issues = self.check_frequencies()
        if issues:
            print(f"\n  IMBALANCES DETECTED:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n  All frequencies within GTO ranges")
        print()


LOG_FILE = "self_play_log.jsonl"
RESULTS_FILE = "self_play_results.json"


def _load_cumulative():
    """Load cumulative results from prior runs."""
    import os
    if not os.path.exists(RESULTS_FILE):
        return None
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    print("Loading strategy...", end=" ", flush=True)
    t0 = time.time()
    strategy = load_strategy("strategy_2max.json.gz")
    ca = CardAbstraction(num_players=2, use_emd=False)
    bot = Bot(strategy, card_abstraction=ca, num_players=2)
    print(f"done ({time.time()-t0:.1f}s)")

    monitor = SelfPlayMonitor(bot, ca)
    report_interval = 100  # in deals (each deal = 2 hands)

    # Load cumulative state from prior runs
    prior = _load_cumulative()
    cumulative_hands = prior['hands'] if prior else 0
    cumulative_profit = list(prior['profit']) if prior else [0.0, 0.0]
    if prior and 'frequencies' in prior:
        for key_str, counts in prior['frequencies'].items():
            parts = key_str.strip('()').replace("'", '').split(', ')
            if len(parts) == 3:
                key = tuple(parts)
                for act, count in counts.items():
                    monitor.street_actions[key][act] += count
        monitor.hands_played = 0
        print(f"  Loaded {cumulative_hands} prior hands")

    log_f = open(LOG_FILE, "a")

    print(f"Starting self-play with duplicate matching (report every {report_interval} deals)...")
    print(f"Press Ctrl+C to stop and see final report.\n")

    try:
        while True:
            actions_1, state_1, actions_2, state_2, dup_id = monitor.play_hand()

            # Log both plays to JSONL with duplicate_id linking the pair
            for play_idx, (hand_actions, final_state) in enumerate(
                    [(actions_1, state_1), (actions_2, state_2)]):
                if final_state.is_terminal:
                    log_entry = {
                        'hand': monitor.hands_played - 1 + play_idx,
                        'deal': monitor.deals_played,
                        'duplicate_id': dup_id,
                        'play': play_idx,  # 0=original, 1=swapped
                        'cumulative': cumulative_hands + monitor.hands_played,
                        'profit_seat0': final_state.get_terminal_utility(0),
                        'actions': [
                            {'seat': a['seat'], 'street': a['street'],
                             'action': a['action'], 'act_type': a['act_type']}
                            for a in hand_actions
                        ],
                        'terminal_type': final_state.terminal_type,
                    }
                    log_f.write(json.dumps(log_entry) + "\n")

            if monitor.deals_played % 25 == 0:
                log_f.flush()

            # Progress dot every 5 deals (10 hands)
            if monitor.deals_played % 5 == 0:
                print(".", end="", flush=True)

            # Full report every N deals
            if monitor.deals_played % report_interval == 0:
                print()
                monitor.report()

    except KeyboardInterrupt:
        print("\n\nStopped.")
        monitor.report()
    finally:
        log_f.close()
        total_hands = cumulative_hands + monitor.hands_played
        total_profit = [
            cumulative_profit[0] + monitor.profit[0],
            cumulative_profit[1] + monitor.profit[1],
        ]
        with open(RESULTS_FILE, "w") as f:
            data = {
                'hands': total_hands,
                'deals': monitor.deals_played,
                'profit': total_profit,
                'session_hands': monitor.hands_played,
                'session_deals': monitor.deals_played,
                'frequencies': {
                    str(k): dict(v) for k, v in monitor.street_actions.items()
                }
            }
            json.dump(data, f, indent=2)
        print(f"  Cumulative: {total_hands} hands saved to {RESULTS_FILE}")
        print(f"  Hand log: {LOG_FILE}")


if __name__ == "__main__":
    main()
