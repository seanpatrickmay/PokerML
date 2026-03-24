"""Bot that plays using fixed preflop ranges + trained postflop CFR strategy.

Integrates:
  - Always-resolve on turn+river (Libratus-style real-time solving)
  - Safe subgame solving with gadget game for untrained nodes
  - Action translation for off-tree human bet sizes
  - Strategy thresholding to remove low-probability noise
  - Opponent model adjustments for exploitative play
"""

import numpy as np

from cfr.action_translation import translate_action, get_tree_fractions
from cfr.card_abstraction import CardAbstraction
from cfr.game_state import get_position_name, lookup_with_fallback
from cfr.preflop_ranges import (
    get_preflop_action, classify_scenario, has_trained_strategy, EQUITIES,
)
from cfr.preflop_solver import hand_to_class
from cfr.strategy_store import load_strategy
from cfr.evaluator import enumerate_river_equity
from cfr.strategy_utils import threshold_strategy, blend_strategies, adaptive_threshold
from cfr.subgame_solver import SubgameSolver
from config import ALWAYS_RESOLVE_FROM_STREET


class Bot:
    def __init__(self, strategy_path, card_abstraction=None,
                 num_players=6):
        if isinstance(strategy_path, dict):
            self.strategy = strategy_path
        else:
            self.strategy = load_strategy(strategy_path)
        self.num_players = num_players
        self.card_abstraction = card_abstraction or CardAbstraction(
            num_players=num_players)
        self.solver = SubgameSolver(
            self.strategy, self.card_abstraction)
        self._resolved_cache = {}  # Cache resolved subgame strategies
        self._cache_board = None   # Board state when cache was populated

    def get_action(self, hand, visible_board, history, legal_actions,
                   state=None, seat=None, exploit_adj=None):
        num_players = state.num_players if state else self.num_players
        pos = get_position_name(num_players, seat) if seat is not None else ''

        # ---- Preflop: use fixed ranges ----
        if state and state.street == 0:
            return self._preflop_action(hand, pos, history, legal_actions,
                                        seat, state)

        # ---- Postflop: use trained CFR strategy ----
        return self._postflop_action(hand, visible_board, history,
                                     legal_actions, state, seat, pos,
                                     exploit_adj)

    def _preflop_action(self, hand, pos, history, legal_actions, seat, state):
        # Try the main trained CFR strategy (uses bucket abstraction)
        bucket = self.card_abstraction.get_bucket(hand, ())
        history_str = '/'.join(','.join(s) for s in history)
        info_key = f'{pos}:{bucket}|{history_str}'
        probs = lookup_with_fallback(self.strategy, info_key, state.num_players)

        # Count raises to determine scenario depth
        preflop_actions_so_far = history[0] if history and history[0] else []
        num_raises = sum(1 for a in preflop_actions_so_far
                         if a.startswith('b') or a == 'a')

        # Use trained strategy for simple scenarios (opening, vs open)
        # Use heuristic for complex scenarios (vs 3-bet+) where buckets fail
        use_trained = (probs is not None and len(probs) == len(legal_actions)
                       and num_raises < 2)

        if use_trained:
            probs = list(probs)
        else:
            # Fall back to heuristic ranges (better for 3-bet+ scenarios)
            hand_key = hand_to_class(hand)
            scenario = classify_scenario(history, pos, state.num_players)
            action_probs = get_preflop_action(
                hand_key, pos, scenario, legal_actions,
                history=history, num_players=state.num_players)
            probs = [action_probs.get(a, 0.0) for a in legal_actions]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(legal_actions)] * len(legal_actions)

        # HU preflop correction: trained strategy is too tight for heads-up
        if state.num_players == 2 and 'f' in legal_actions:
            fold_idx = legal_actions.index('f')
            raise_indices = [i for i, a in enumerate(legal_actions)
                             if a.startswith('b') or a == 'a']
            call_idx = legal_actions.index('c') if 'c' in legal_actions else None

            hand_key = hand_to_class(hand)
            rank = EQUITIES.get(hand_key, 0.5)

            # Count raises so far to determine scenario
            preflop_actions = history[0] if history and history[0] else []
            raise_count = sum(1 for a in preflop_actions
                              if a.startswith('b') or a == 'a')

            if raise_count >= 2:
                # Facing 3-bet: HU ranges are wider than 6-max heuristic
                # BTN should defend ~45-50% vs 3-bet in HU
                if rank > 0.60:
                    max_fold = 0.05  # premium: never fold
                elif rank > 0.35:
                    max_fold = 0.40  # medium: fold less than half
                else:
                    max_fold = 0.80  # weak: fold most
            elif pos == 'BTN' and raise_count == 0:
                # BTN opening: should open ~70-80%
                max_fold = 0.30 if rank < 0.30 else 0.10 if rank < 0.50 else 0.0
            elif pos == 'BB' and raise_count == 1:
                # BB vs open: should defend ~55%
                max_fold = 0.70 if rank < 0.20 else 0.40 if rank < 0.40 else 0.15
            else:
                max_fold = 1.0  # other scenarios: no correction

            if probs[fold_idx] > max_fold:
                excess = probs[fold_idx] - max_fold
                probs[fold_idx] = max_fold
                # Premium hands: excess goes to raise. Medium: split raise/call
                if rank > 0.70 and raise_indices:
                    for i in raise_indices:
                        probs[i] += excess / len(raise_indices)
                elif call_idx is not None and raise_indices:
                    # Split between call and raise
                    probs[call_idx] += excess * 0.6
                    for i in raise_indices:
                        probs[i] += (excess * 0.4) / len(raise_indices)
                elif call_idx is not None:
                    probs[call_idx] += excess
                elif raise_indices:
                    for i in raise_indices:
                        probs[i] += excess / len(raise_indices)

            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

        return legal_actions[np.random.choice(len(legal_actions), p=probs)]

    def _postflop_action(self, hand, visible_board, history, legal_actions,
                         state, seat, pos, exploit_adj=None):
        num_players = state.num_players if state else self.num_players
        bucket = self.card_abstraction.get_bucket(hand, visible_board)
        history_str = '/'.join(','.join(s) for s in history)
        info_key = f'{pos}:{bucket}|{history_str}'

        # Always resolve on turn+river for superior play (Libratus approach)
        should_resolve = (state is not None and seat is not None
                          and state.street >= ALWAYS_RESOLVE_FROM_STREET)

        if should_resolve:
            # Invalidate cache when board changes (new hand or new street)
            current_board = state.visible_board()
            if current_board != self._cache_board:
                self._resolved_cache.clear()
                self._cache_board = current_board

            # Check cache first before re-solving
            cached_probs = lookup_with_fallback(self._resolved_cache, info_key, num_players)
            if cached_probs is not None and len(cached_probs) == len(legal_actions):
                resolved_probs = cached_probs
            else:
                solved = self.solver.solve(state, seat)
                self._resolved_cache.update(solved)
                resolved_probs = lookup_with_fallback(solved, info_key, num_players)

            blueprint_probs = lookup_with_fallback(self.strategy, info_key, num_players)

            if resolved_probs is not None and len(resolved_probs) == len(legal_actions):
                if blueprint_probs is not None and len(blueprint_probs) == len(legal_actions):
                    # Blend resolved with blueprint for robustness
                    probs = blend_strategies(
                        list(resolved_probs), list(blueprint_probs),
                        resolve_weight=0.90)
                else:
                    probs = list(resolved_probs)
            else:
                probs = blueprint_probs
        else:
            # Earlier streets: use blueprint with fallback chain
            probs = lookup_with_fallback(self.strategy, info_key, num_players)

            # Try action translation for off-tree bets
            if probs is None or len(probs) != len(legal_actions):
                probs = self._try_action_translation(
                    history, pos, bucket, legal_actions, num_players)

            if probs is not None and len(probs) == len(legal_actions):
                probs = list(probs)
            else:
                # Real-time subgame solve on miss
                if state is not None and seat is not None:
                    solved = self.solver.solve(state, seat)
                    self._resolved_cache.update(solved)

                    probs = lookup_with_fallback(solved, info_key, num_players)
                    if probs is not None and len(probs) == len(legal_actions):
                        probs = list(probs)

        if probs is not None and len(probs) == len(legal_actions):
            probs = list(probs)

        if probs is None or len(probs) != len(legal_actions):
            return np.random.choice(legal_actions)

        # Apply opponent model adjustments
        if exploit_adj is not None:
            probs = self._apply_adjustments(probs, legal_actions, exploit_adj)

        # Adaptive thresholding (applied early, before GTO corrections)
        num_actions = sum(len(s) for s in history) if history else 0
        probs = adaptive_threshold(probs, num_actions_taken=num_actions)

        # All GTO corrections applied LAST — they are the final authority
        probs = self._equity_corrections(
            probs, legal_actions, hand, visible_board, bucket, state)
        probs = self._check_raise_cap(probs, legal_actions, state, seat)
        probs = self._river_calling_correction(
            probs, legal_actions, hand, visible_board, bucket, state)
        probs = self._positional_check_correction(
            probs, legal_actions, state, seat, bucket)

        return legal_actions[np.random.choice(len(legal_actions), p=probs)]

    def _try_action_translation(self, history, pos, bucket, legal_actions,
                                num_players):
        """Try to find strategy by translating off-tree actions in history."""
        if not history:
            return None

        # Check if the last opponent action was off-tree
        for street_idx, street_actions in enumerate(history):
            for act_idx, action in enumerate(street_actions):
                if action.startswith('b'):
                    try:
                        pct = int(action[1:])
                        tree_fracs = get_tree_fractions(legal_actions)
                        if tree_fracs and pct / 100.0 not in tree_fracs:
                            # This action was off-tree; try translating
                            mappings = translate_action(pct / 100.0, tree_fracs)
                            if mappings:
                                return self._blend_translated_strategies(
                                    mappings, history, street_idx, act_idx,
                                    pos, bucket, legal_actions, num_players)
                    except (ValueError, IndexError):
                        continue
        return None

    def _blend_translated_strategies(self, mappings, history, street_idx,
                                      act_idx, pos, bucket, legal_actions,
                                      num_players):
        """Blend strategies from translated action nodes."""
        blended = [0.0] * len(legal_actions)
        found_any = False

        for frac, weight in mappings:
            pct = round(frac * 100)
            # Reconstruct history with translated action
            translated_history = [list(s) for s in history]
            translated_history[street_idx][act_idx] = f'b{pct}'
            history_str = '/'.join(','.join(s) for s in translated_history)
            info_key = f'{pos}:{bucket}|{history_str}'

            probs = lookup_with_fallback(self.strategy, info_key, num_players)
            if probs is not None and len(probs) == len(legal_actions):
                found_any = True
                for i in range(len(legal_actions)):
                    blended[i] += weight * probs[i]

        if not found_any:
            return None

        total = sum(blended)
        if total > 0:
            return [p / total for p in blended]
        return None

    def _equity_corrections(self, probs, actions, hand, board, bucket, state):
        """Post-processing corrections for bucket-abstraction blind spots.

        Bucket abstraction groups hands with similar but not identical
        strength. This causes two known issues:
          1. Nut hands fold because they share a bucket with weaker hands
          2. Air hands don't fold enough because they share a bucket
             with slightly better hands

        Fix: use actual hand equity (river) or bucket position to adjust.
        """
        if state is None or 'f' not in actions:
            return probs

        num_buckets = self.card_abstraction.num_postflop_buckets
        if num_buckets <= 1:
            return probs

        fold_idx = actions.index('f')
        facing_bet = 'f' in actions  # fold is only available when facing a bet

        # River: use exact equity for precise corrections
        if board and len(board) == 5:
            equity = enumerate_river_equity(hand, board)

            # Never fold with equity > 0.90 (nut-level hands)
            if equity > 0.90 and probs[fold_idx] > 0.01:
                fold_mass = probs[fold_idx]
                probs[fold_idx] = 0.0
                # Redistribute to raise actions if any, else call
                raise_indices = [i for i, a in enumerate(actions)
                                 if a.startswith('b') or a == 'a']
                if raise_indices:
                    for i in raise_indices:
                        probs[i] += fold_mass / len(raise_indices)
                elif 'c' in actions:
                    probs[actions.index('c')] += fold_mass

            # Air (equity < 0.15) facing a bet: fold at least 40%
            elif equity < 0.15 and facing_bet:
                min_fold = 0.40
                if probs[fold_idx] < min_fold:
                    deficit = min_fold - probs[fold_idx]
                    probs[fold_idx] = min_fold
                    # Reduce other actions proportionally
                    other_total = sum(probs[i] for i in range(len(probs))
                                     if i != fold_idx)
                    if other_total > 0:
                        scale = (other_total - deficit) / other_total
                        for i in range(len(probs)):
                            if i != fold_idx:
                                probs[i] *= max(scale, 0.0)
        else:
            # Pre-river: use bucket position as proxy
            equity_proxy = (bucket + 0.5) / num_buckets

            # Top 5% buckets: never fold
            if equity_proxy > 0.95 and probs[fold_idx] > 0.01:
                fold_mass = probs[fold_idx]
                probs[fold_idx] = 0.0
                raise_indices = [i for i, a in enumerate(actions)
                                 if a.startswith('b') or a == 'a']
                if raise_indices:
                    for i in raise_indices:
                        probs[i] += fold_mass / len(raise_indices)
                elif 'c' in actions:
                    probs[actions.index('c')] += fold_mass

            # Facing a bet: enforce minimum fold frequencies by hand strength
            # GTO defends ~55% and folds ~45% on average
            elif facing_bet:
                if equity_proxy < 0.20:
                    min_fold = 0.55  # very weak: fold most
                elif equity_proxy < 0.40:
                    min_fold = 0.30  # weak-medium: fold a fair amount
                elif equity_proxy < 0.55:
                    min_fold = 0.10  # medium: fold occasionally
                else:
                    min_fold = 0.0  # strong+: no floor
                if probs[fold_idx] < min_fold:
                    deficit = min_fold - probs[fold_idx]
                    probs[fold_idx] = min_fold
                    other_total = sum(probs[i] for i in range(len(probs))
                                     if i != fold_idx)
                    if other_total > 0:
                        scale = (other_total - deficit) / other_total
                        for i in range(len(probs)):
                            if i != fold_idx:
                                probs[i] *= max(scale, 0.0)

        # Ensure valid distribution
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        return probs

    def _check_raise_cap(self, probs, actions, state, seat):
        """Cap check-raise frequency to GTO-realistic levels.

        The blueprint massively over-check-raises (75%+ vs GTO 8-12%).
        This applies when facing a bet after checking — cap raise actions
        and redistribute to calling.
        """
        if state is None or 'f' not in actions or 'c' not in actions:
            return probs

        raise_indices = [i for i, a in enumerate(actions)
                         if a.startswith('b') or a == 'a']
        if not raise_indices:
            return probs

        # Max raise frequency depends on street
        if state.street == 1:    # flop x/r: 8-15%
            max_raise = 0.15
        elif state.street == 2:  # turn x/r: 8-12%
            max_raise = 0.12
        else:                    # river x/r: 10-15%
            max_raise = 0.15

        total_raise = sum(probs[i] for i in raise_indices)
        if total_raise <= max_raise:
            return probs

        call_idx = actions.index('c')
        fold_idx = actions.index('f')
        excess = total_raise - max_raise
        scale = max_raise / total_raise if total_raise > 0 else 0
        for i in raise_indices:
            probs[i] *= scale
        # Distribute excess proportionally between fold and call
        fold_call_total = probs[fold_idx] + probs[call_idx]
        if fold_call_total > 0:
            probs[fold_idx] += excess * (probs[fold_idx] / fold_call_total)
            probs[call_idx] += excess * (probs[call_idx] / fold_call_total)
        else:
            probs[call_idx] += excess

        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        return probs

    def _river_calling_correction(self, probs, actions, hand, board, bucket, state):
        """Fix over-raising with medium hands on the river.

        The blueprint tends to turn medium-strength hands into raises
        instead of making correct calls. GTO river play with medium
        hands facing a bet should primarily call, not raise.
        """
        if state is None or state.street != 3:
            return probs
        if 'c' not in actions or 'f' not in actions:
            return probs  # not facing a bet

        num_buckets = self.card_abstraction.num_postflop_buckets
        equity_proxy = (bucket + 0.5) / max(num_buckets, 1)

        # Medium hands (30-70% equity) facing a bet should mostly call
        if 0.25 < equity_proxy < 0.75:
            call_idx = actions.index('c')
            raise_indices = [i for i, a in enumerate(actions)
                             if a.startswith('b') or a == 'a']
            if not raise_indices:
                return probs

            # Cap total raise probability for medium hands
            max_raise = 0.20
            total_raise = sum(probs[i] for i in raise_indices)
            if total_raise > max_raise:
                excess = total_raise - max_raise
                scale = max_raise / total_raise if total_raise > 0 else 0
                for i in raise_indices:
                    probs[i] *= scale
                probs[call_idx] += excess

            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

        return probs

    def _positional_check_correction(self, probs, actions, state, seat, bucket):
        """Increase check frequency when OOP on turn/river.

        GTO play requires the OOP player to check frequently to protect
        their checking range. Blueprint strategies trained with bucket
        abstraction tend to under-check because positional disadvantage
        is not fully captured in hand-strength buckets alone.

        Uses a soft correction: boosts check probability toward a floor
        that depends on hand strength (weak hands check more, strong
        hands still check for protection but less).
        """
        if state is None or 'k' not in actions:
            return probs
        if state.street < 1:  # preflop handled separately
            return probs
        if state.num_players != 2:
            return probs

        from cfr.game_state import action_order
        order = action_order(state.num_players, state.street)
        is_oop = (order[0] == seat)

        check_idx = actions.index('k')
        num_buckets = self.card_abstraction.num_postflop_buckets
        equity_proxy = (bucket + 0.5) / max(num_buckets, 1)

        # History adjustment: if we checked the prior street, our range
        # is weaker → check more on this street (delayed aggression penalty)
        prior_street = state.street - 1
        checked_prior = False
        if prior_street >= 1 and len(state.history) > prior_street:
            prior_actions = state.history[prior_street]
            # Find our action on the prior street
            from cfr.game_state import action_order
            prior_order = action_order(state.num_players, prior_street)
            my_position_in_order = prior_order.index(seat) if seat in prior_order else -1
            if my_position_in_order >= 0 and my_position_in_order < len(prior_actions):
                checked_prior = (prior_actions[my_position_in_order] == 'k')
        history_boost = 0.08 if checked_prior else 0.0

        # Board wetness adjustment for flop
        wet_boost = 0.0
        if state.street == 1:
            vis = state.visible_board()
            if len(vis) >= 3:
                suits = [c % 4 for c in vis[:3]]
                ranks = sorted([c // 4 for c in vis[:3]])
                flush_draw = len(set(suits)) <= 2
                connected = sum(1 for i in range(2) if ranks[i+1] - ranks[i] <= 2)
                if flush_draw and connected >= 1:
                    wet_boost = 0.08  # wet board: check more
                elif flush_draw or connected >= 2:
                    wet_boost = 0.04  # semi-wet

        if is_oop:
            # OOP check floors: GTO requires frequent checking to protect range
            if state.street == 1:  # flop OOP: check ~65-75%
                if equity_proxy < 0.3:
                    min_check = 0.80 + wet_boost
                elif equity_proxy < 0.7:
                    min_check = 0.65 + wet_boost
                else:
                    min_check = 0.45 + wet_boost
            else:  # turn/river OOP: check ~55-70%
                if equity_proxy < 0.3:
                    min_check = 0.75 + history_boost
                elif equity_proxy < 0.7:
                    min_check = 0.55 + history_boost
                else:
                    min_check = 0.40 + history_boost
        else:
            # IP check-back floors: protect checking range
            if state.street == 1:  # flop IP: check ~35-45%
                if equity_proxy < 0.25:
                    min_check = 0.65 + wet_boost
                elif equity_proxy < 0.5:
                    min_check = 0.40 + wet_boost
                else:
                    min_check = 0.20 + wet_boost
            else:  # turn/river IP: check-back protection
                if equity_proxy < 0.25:
                    min_check = 0.55 + history_boost
                elif equity_proxy < 0.5:
                    min_check = 0.35 + history_boost
                else:
                    min_check = 0.15 + history_boost

        current_check = probs[check_idx]
        if current_check >= min_check:
            return probs

        # Boost check probability, reduce bet probabilities proportionally
        deficit = min_check - current_check
        probs[check_idx] = min_check
        bet_total = sum(probs[i] for i in range(len(probs)) if i != check_idx)
        if bet_total > 0:
            scale = (bet_total - deficit) / bet_total
            for i in range(len(probs)):
                if i != check_idx:
                    probs[i] *= max(scale, 0.0)

        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        return probs

    def _apply_adjustments(self, probs, actions, adj):
        """Apply exploit adjustments bounded by confidence.

        Adjustments are clamped so the bot never deviates more than
        max_deviation from its GTO strategy, preventing counter-exploitation.
        """
        max_dev = adj.get('max_deviation', 0.15)
        adjusted = list(probs)

        for i, action in enumerate(actions):
            original = probs[i]
            if action.startswith('b') or action == 'a':
                # Bet/raise: scale by bet_frequency and bluff multipliers
                bet_mult = adj.get('bet_frequency_mult', 1.0)
                adjusted[i] *= bet_mult
            elif action == 'f':
                # Fold: inverse of bet frequency (fold less when we bet more)
                bet_mult = adj.get('bet_frequency_mult', 1.0)
                adjusted[i] /= max(bet_mult, 0.3)
            elif action == 'c':
                # Call: adjust by call threshold
                call_mult = adj.get('call_threshold_mult', 1.0)
                adjusted[i] *= call_mult

        total = sum(adjusted)
        if total > 0:
            adjusted = [p / total for p in adjusted]

        # Bound deviation from GTO: no action prob changes more than max_dev
        # Uses iterative projection to maintain sum=1 within box constraints
        if max_dev < 1.0:
            lo = [max(0, probs[i] - max_dev) for i in range(len(adjusted))]
            hi = [probs[i] + max_dev for i in range(len(adjusted))]

            for _ in range(20):
                for i in range(len(adjusted)):
                    adjusted[i] = max(lo[i], min(hi[i], adjusted[i]))
                deficit = 1.0 - sum(adjusted)
                if abs(deficit) < 1e-12:
                    break
                if deficit > 0:
                    room = [(i, hi[i] - adjusted[i]) for i in range(len(adjusted))
                            if adjusted[i] < hi[i] - 1e-12]
                else:
                    room = [(i, adjusted[i] - lo[i]) for i in range(len(adjusted))
                            if adjusted[i] > lo[i] + 1e-12]
                if not room:
                    break
                total_room = sum(r for _, r in room)
                if total_room < 1e-12:
                    break
                for i, r in room:
                    adjusted[i] += deficit * (r / total_room)

        return adjusted
