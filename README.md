# Investigating-the-Fate-of-the-Crew
Combined use of basic game theory concepts and scikit-learn neural networks for predicting the probability of winning a game of Among Us. Was able to predict the winner with over 70% accuracy.

Planned to create a model that predicts whether a crewmate or impostor will win a given game of Among Us, taking factors such as game time, tasks completed, murders that happened, etc. into account.

Intially unsure about how to process the fields. First step was to graph the datasets. If closely related to bell curve, then standardization is a good fit. If more closely representing two different extremes, then normalization is a better fit. 

Significant fields considered:

For crewmate:
* 0th col: Tasks Completed
* 1st col: All Tasks Completed?
* 2nd col: Murdered?
* 3rd col: Crewmate Game Length
* 4th col: Ejected?
* 5th col: Sabotages Fixed
* 6th col: Time to Complete All Tasks
* 7th col: NA Player Region
* 8th col: Europe Player Region
* 9th col: Outcome (Win = 1; Loss = 0)

For impostor:
* 0th col: Imposter Kills
* 1st col: Imposter Game Length
* 2nd col: Ejected?
* 3rd col: NA Player Region
* 4th col: Europe Player Region
* 5th col: Outcome (Win = 1; Loss = 0)

Neural network like one that was used in MNIST, but with tweaks to the amount of nodes,layers, epochs, and batch sizes.

The probability of the given player, whether that may be a crewmate or impostor, has of winning the game, and it is presented as a number between 0 and 1, and calculated using a Sigmoid or maybe SoftMax function. 1 would be that they would certainly win and 0 would mean that they would certainly lose.


