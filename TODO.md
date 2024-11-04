PokerStrategy:
- Figure out nut equity cutoff
- Figure out range vs range nut ratio for bet sizing
- Figure out equity ratio for bet frequency
- Figure out hand equity cutoff for bet or check
- Lots of ideas with choosing bluffs. Pick from bottom of range (outside of calling range)?
- Same with mixing? How do I determine what % a nutted hand should check to uncap checking range?

- Is the idea of giving an opponent range and self range in the constructor valid? Maybe that should be fed dynamically. This way the strategy doesn't have to update the range itself.

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
