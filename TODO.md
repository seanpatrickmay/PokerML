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
