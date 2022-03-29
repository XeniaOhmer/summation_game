# summation_game


In this project, I train a sender and a receiver agent on a "summation game". The sender receives to integers 
(s<sub>1</sub>, s<sub>2</sub>) with 0 <= s<sub>i</sub> <= N. It generates a single symbol (discretization is achieved 
by using Gumbel-Softmax). Based on that symbol, the receiver tries to predict the sum of the two integers, 
s<sub>1</sub> + s<sub>2</sub>.

The agents' architectures can be found in 'architectures.py' and the training procedure in 'train.py'. The latter 
contains explanations for all command line parameters.

For example, to train the agents with *N=20* and *|V|=41* using the same hyperparameters as in the report, run: 

    python train.py --N 20 --n_symbols 41 --lr 0.001 --n_epochs 250 

The folder 'grid_search/' contains the json file used to run the grid search (with nest_local.py), as well as the text 
files recording the console output of each run, and a jupyter notebook analyzing the results: 
'analyze_grid_search.ipynb'.

The folder 'results/' contains the console output files for all experiments. I did not upload the interaction 
files due to their large size. The folder also contains the json files used to run 
the experiments (with nest_local.py): sweep_N20, sweep_N40, and sweep_N80, plus sweep_N80_alt_splits (for the experiments with different train/test ratios).

All analyses and figures in the report can be found in the notebook 'analysis.ipynb'. 


 