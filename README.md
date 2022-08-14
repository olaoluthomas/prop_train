---
# Vertex Pipelines for training Main Event Propensity Models


Custom Container Propensity model training in Vertex AI.

---

This is a barebones solution to run model training in Vertex AI with training data
either with a BQ table or a Vertex Managed Dataset.

The Python module syntax is:\
\
`python -m vertex_proptrainer.train --segment <SEGMENT> --dataset <DATASET> --model_dir <DIR>`\
\
where
- "SEGMENT" is the propensity segment to be trained
- "DATASET" is the BQ table for training and evaluation
- "DIR" is the GCS URI/folder path to save model artifacts

Alternatively, if the dataset is already split into train and eval:\
\
`python -m vertex_proptrainer.train --segment <SEGMENT> --train_data <TRAIN> --eval_data <EVAL> --model_dir <DIR>`


If training with Vertex Managed Dataset, dataset is passed as an argument directly to the 
CustomContainerTrainingJob or CustomContainerTrainingJobRunOp (Vertex Pipeline) and neither of
dataset, train_data or eval_data args is passed allowing the training script to pick the data
from default environment variables of Vertex AI.

If training segments 115, 116 or prospects, a linear model is trained by default.
Otherwise, model is tree-based.

## Optional arguments
### Tree-based model hyperparemeters
- --dataset: BQ dataset with data for training and testing
- --max_depth: Maximum depth of the base learners
- --min_child_weight: Minimum sum of instance weight in a child tree
- --max_delta_step: Maximum delta step allowed for each tree's weight estimation
- --reg_lambda: L2 regularization
- --reg_alpha: L1 regularization
- --gamma: Pruning strength
- --lr: Learning rate

### Linear model hyperparameters
- --loss: Model loss function ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron')
- --penalty: Regularization term ('l2', 'l1', 'elasticnet')
- --l1_ratio: L1 & L2 Regularization mixing parameter if penalty is elasticnet
- --alpha: Regularization strength
- --tol: Stopping criterion (loss > best_loss - tol)
- --epsilon: Measure of epsilon-insensitive loss if loss is 'huber', 'epsilon_insensitive' or 'squared_epsilon_insensitive'
- --linear_lr: Learning rate schedule ('constant', 'optimal', 'invscaling', 'adaptive')
- --eta0: Initial learning rate. Not used if learning rate schedule is 'optimal'
- --no_intercept: Fit model without an intercept

### Early stopping
- --rounds: Number of rounds without evaluation improvement to trigger early stopping

Hyperparameter tuning (using cloudml-hypertune package) enabled by passing the "--hypertune" argument:

`python -m vertex_proptrainer.train --segment=<SEGMENT> --dataset <DATASET> --hypertune <ARGS>`

Build package within src folder:

`cd src`\
`python setup.py sdist` OR `python setup.py sdist bdist_wheel`