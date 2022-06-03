# nba-regression-models
Hand built regression models for predicting a players points scored in an nba game.

Playertojson.py:

- After downloading a player json file using the https://github.com/MaxymWojnowskyj/Scrape-player-from-basketball-reference script playertojson.py allows the user to download prediction variables and predictee values from the players dataset. 

- We start by downloading a mins vs. pts scored scatterplot so the user can visually observe the distribution of the data. 

- Then we prompt the user to decide wether they would like to remove any outliers, if the user selects yes, the script automatically removes outliers calculated via the empirical rule (removing any points beyond 2 standard deviations from the mean) if the players pts scored mean is greater than its standard deviation. If the standard deviation is greater than the mean we remove any outliers outside of the  the inter quartile range. 

- The user is then prompted with a chance to download a scatterplot of any prediction variable vs the players points scored so they can observe the distribution of any prediction variable.

- Once the user knows the distribution of the prediction variables they can then input the list of prediction variables they would like to download with any transformation applied to the prediction variables. Wether it be exponential, logarithmic, interactive, division or multiplication. The user can then name the prediction variables file and a json file will be downloaded holding the users inputted variables list.

- Finally, the user can select wether they would like to download a json file which is an array containing the values of the players points scored (the value we are trying to predict). This prompt is added in case the user wants to download multiple prediction variable files to try and has already downloaded the predictee variables file (points scored).  
