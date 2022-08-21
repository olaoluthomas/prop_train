---
# Vertex Pipelines for training Main Event Propensity Models


Custom Container Propensity model training in Vertex AI.

---

This is a barebones solution to run model training in Vertex AI with training data
either with a BQ table or a Vertex Managed Dataset.

The Python module syntax is:\
\
`python -m vertex_proptrainer.train --segment <SEGMENT> --dataset <DATASET>`\
\
where
- "SEGMENT" is the propensity segment to be trained
- "DATASET" is the BQ table with data for training and evaluation

Alternatively, if the dataset is already split into train and eval:\
\
`python -m vertex_proptrainer.train --segment <SEGMENT> --train_data <TRAIN> --eval_data <EVAL>`


If training with Vertex Managed Dataset in Vertex Kubeflow Pipelines, dataset is passed as an
argument directly to CustomContainerTrainingJobRunOp and neither of dataset, train_data or 
eval_data args is passed allowing the trainer package to pick the data from default
environment variables - `os.environ["AIP_TRAINING_DATA_URI"]`, `os.environ["AIP_EVALUATION_DATA_URI"]`,
and `os.environ["AIP_TESTING_DATA_URI"]` - set by Vertex AI.

---

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

### Hyperparameter tuning
Tuning (using cloudml-hypertune package) is enabled by passing the "--hypertune" argument:

`python -m vertex_proptrainer.train --segment=<SEGMENT> --dataset <DATASET> --hypertune`

Tuning does not save artifacts at this time; functionality to be added in subsequent release(s)
to allow checkpointing, training resumption and tuning artifact saving to GCS.

### Early stopping
- --rounds: Number of rounds without evaluation improvement to trigger early stopping

### Model artifact saving
- --model_dir: GCS URI to save model artifacts. If not provided, package will use Vertex AI 
default environment variable `os.environ["AIP_MODEL_DIR"]`, and, if provided, must end with
"/model"
i.e.\
`... --model_dir gs://<bucket>/<gcs_path>/model`

---
## Building from Source

Build package within src folder:

`cd src`\
`python setup.py sdist` OR `python setup.py sdist bdist_wheel`