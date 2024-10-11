For evaluating equity from preflop, the naive solution would be as following:
Pick 3 cards, then pick 1, then pick 1, having 3 nested loops.
This is much much slower than optimal, however, as this is the same as choose(50, 3) * 47 * 46 = 4.2e10^7
Instead, we do choose(50, 5) = 2.1e10^6

