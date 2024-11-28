PokerStrategy:
- Lots of ideas with choosing bluffs. Pick from bottom of range (outside of calling range)?
- Same with mixing? How do I determine what % a nutted hand should check to uncap checking range?
- Currently assuming 1 bet size. I think this is best for now.

Range.getEquityAgainstHand:
- For monte-carlo, currently randomly grabs board from set. This makes it so it can grab the same board multiple times. Should I change this?
- Currently has option to get equity squared. It does this by squaring the eqiuty after each board for each hand. I should instead do each board, then for each hand, add the value to a list, and then for the hand, compare every single hand for each board, square that, and get average of all of those. Perhaps?


MiniMax:
- Bucket hands from range into certain # of buckets, use equity to determine bet sizing.

GetEquityFromBetSizing:
- This method should recursively calculate the equity if opponent calls/folds + equity if opponent raises (recurse)
- This method should also take into account stack sizes, 10x pot, even if super polarized, is not best with an SPR of 1. I think bet sizes should be a list up to and including SPR.
- If spr is 0, no need to recurse.


How do I represent a bet when I am calling the getBestSizing function? I need to find the EV of calling first. Then I can compare the EV of calling to that of raising





GENERAL MINIMAX ROADMAP:

Given node:
Calculate best sizing for range
Done by:
Calculating EV of bet + 


