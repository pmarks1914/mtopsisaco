

## NOTES
- The combination of `TOPSIS` and `ACO` for feature selection aims to leverage the strengths of both methods to improve multi-label classification performance.

#### 1. Role of TOPSIS:
```sh
- TOPSIS is used to evaluate and rank the importance of each feature based on multiple criteria (e.g., relevance to each label, contribution to classification accuracy).
- It helps identify features that are closer to the ideal solution, which are considered more relevant and useful for classification.
```
#### 2. Role of ACO:

```sh
- ACO is used to search for the optimal subset of features. It simulates the behavior of ants to explore various combinations of features.
Artificial ants select features based on pheromone levels and `heuristic information (e.g., feature rankings from TOPSIS).`
- Over time, pheromone trails guide the search towards the best feature subsets, balancing exploration (trying new combinations) and exploitation (using known good combinations).
```

#### Execution
```sh
## Initialization:

Evaluate all features using TOPSIS to determine their initial rankings based on their importance for multi-label classification.
Initialize pheromone levels for all features.

## Ant Colony Optimization:
Iteratively, artificial ants construct feature subsets based on pheromone levels and heuristic information (TOPSIS rankings).
Evaluate the performance of each feature subset using a multi-label classification model.
Update pheromone levels based on the performance of feature subsets (better subsets receive more pheromones).

## Feature Selection:
After a certain number of iterations or when convergence is achieved, select the feature subset with the highest performance.
Use this subset for the final multi-label classification model.
```
## SET UP

```sh
# Remove the Old Host Key:
ssh-keygen -f "/home/tiaspaces/.ssh/known_hosts" -R "webserve.appatechlab.com"

# Force SSH to Use Password Authentication:
ssh -o 'proxycommand socat - PROXY:18.157.151.201:%h:%p,proxyport=6000' -o PreferredAuthentications=password disal@webserve.appatechlab.com -v

```

## Server Access
```sh
# OLd server
 ssh -o 'proxycommand socat - PROXY:18.157.151.201:%h:%p,proxyport=6000' -o PreferredAuthentications=password disal@webserve.appatechlab.com 

 # New server
  ssh -o 'proxycommand socat - PROXY:18.157.151.201:%h:%p,proxyport=6000' -o PreferredAuthentications=password disal2@mlserve.appatechlab.com 
```

## Helper Commands
```sh
# Check Memory & CPU consumption
top -l 1 | grep -E "^CPU|^Phys"

```