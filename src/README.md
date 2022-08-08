---
# Vertex Pipelines for training Main Event Propensity Models


Using Custom Container Training and Develop and Upload a DM propensity model via Vertex Pipelines.

---

This is a barebones solution to run model training via Python command.

The Python module syntax is:\
\
`python -m proptrainer.train --segment <SEGMENT> --dataset <DATASET>`\
\
where
- "SEGMENT" is the propensity segment to be trained
- "DATASET" is the BQ table for training and evaluation

Alternatively, if the dataset is already split into train and eval:\
\
`python -m proptrainer.train --segment <SEGMENT> --train_data <TRAIN> --eval_data <EVAL>`


If training with Vertex Managed Dataset, dataset is passed as an argument directly to the 
CustomContainerTrainingJob or CustomContainerTrainingJobRunOp (Vertex Pipeline) and neither of
dataset, train_data or eval_data args is passed allowing the training script to pick the data
from default environment variables.

Optional hyperparameters:
- --max_depth: Maximum depth of the base learners
- --min_child_weight: Minimum sum of instance weight in a child tree
- --max_delta_step: Maximum delta step allowed for each tree's weight estimation
- --reg_lambda: L2 regularization
- --reg_alpha: L1 regularization
- --gamma: Pruning strength
- --lr: Learning rate
- --rounds: Number of rounds without improvement for early stopping

Hyperparameter tuning (using cloudml-hypertune) enabled by passing the "--hypertune" parameter:

`python -m proptrainer.train --segment=<SEGMENT> --hypertune [hyperparameter arguments]`