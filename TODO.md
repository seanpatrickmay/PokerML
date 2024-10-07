PokerStrategy:
- Figure out nut equity cutoff
- Figure out range vs range nut ratio for bet sizing
- Figure out equity ratio for bet frequency
- Figure out hand equity cutoff for bet or check
- Lots of ideas with choosing bluffs. Pick from bottom of range (outside of calling range)?
- Same with mixing? How do I determine what % a nutted hand should check to uncap checking range?

- Is the idea of giving an opponent range and self range in the constructor valid? Maybe that should be fed dynamically. This way the strategy doesn't have to update the range itself.

-Currently assuming 1 bet size. I think this is best for now.

Range.getEquityAgainstHand:
- Implement preflop ranges using Monte-Carlo sim to try to find equity. (Will redo this later for greater accuracy)
- Implement range equity vs hand on flop
- Fix to use a copy of self.hands. It currently removes cards from self.hands, and doesn't add them back after calculation.

Range.abstractify:
- Currently takes in a comparator, but there are issues with this, as sets aren't sorted, so gives somewhat random compression.
- Maybe I should change abstractify not to take in a comparator, but a function that takes in a set.

