Data extraction details

Overall: we  make a list of the users, divide them into seen/unseen, and for each of them divide their examples in test/train. The split is random, controled with a seed. 
While the code is not available here, in their paper they reproduce this experiment (by varying the seeds?)  for 20 random splitting. 
Otherwise (for one split) the result is in average over the users on their training data.


PRISM: 
- the dataset conversations contains 8k sequences of choices between two answers given by two different models to a same prompt 


#### 
Commencer sans changer la seed; mais au moment de changer ça veut dire que je dois recharger toutes les données? c'est bizarre.. à faire attention. De plus, c'est valable que pour PRISM?

