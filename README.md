# PokerML

In this project, I am currently trying to implement a computer poker strategy.
Specifically, this strategy will be for Heads-Up NLHE.

TODO:
Implement preflop ranges using Monte-Carlo sim to try to find equity. (Will redo this later for greater accuracy)

Implement range equity vs hand on flop

Abstractify function:
Currently takes in a comparator, but there are issues with this, as sets aren't sorted, so gives somewhat random compression.
Maybe I should change abstractify not to take in a comparator, but a function that takes in a set.
