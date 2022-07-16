"""
A task utility and argument parser to initiate Propensity model training and hyperparameter tuning.
"""

from proptrainer import model
import argparse
import logging
import sys
from sklearn.metrics import roc_auc_score

from hypertune import HyperTune

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bucket_name = 'ai-ml-vertex-d01'
gcs_path = 'dm-propensity/pc'


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment',
                        help='Propensity segment to train.',
                        required=True)
    parser.add_argument('--table_name',
                        help='BQ table name with training data.',
                        required=True)
    parser.add_argument('--timestamp',
                        help='time of model training.',
                        required=True)

    # hyper parameters
    parser.add_argument('--hypertune',
                        help="Perform hyperparameter tuning.",
                        action='store_true',
                        default=False)
    parser.add_argument('--max_depth',
                        help="Max. depth of tree",
                        default=3,
                        type=int,
                        required=False)
    parser.add_argument('--min_child_weight',
                        help="hyperparameter (min_child_weight)",
                        default=5,
                        type=int,
                        required=False)
    parser.add_argument('--max_delta_step',
                        help="hyperparameter (max_delta_step)",
                        default=0.5,
                        type=float,
                        required=False)
    parser.add_argument('--reg_lambda',
                        help="hyperparameter (L2 regularization)",
                        default=0.25,
                        type=float,
                        required=False)
    parser.add_argument('--reg_alpha',
                        help="hyperparameter (L1 regularization)",
                        default=0.1,
                        type=float,
                        required=False)
    parser.add_argument('--gamma',
                        help="Strength of pruning.",
                        default=0.025,
                        type=float,
                        required=False)
    parser.add_argument('--rounds',
                        help="Max # of training rounds before early stopping.",
                        default=10,
                        required=False)
    parser.add_argument('--lr',
                        help="Learning rate.",
                        default=0.1,
                        type=float,
                        required=False)
    return parser


def main():
    args = create_parser().parse_args()
    logging.info(f'Python version: {sys.version}')

    if args.hypertune:
        train_data, eval_data = model.create_dataframe(args.segment,
                                                       args.table_name)
        train_x, train_y = model.input_fn(args.segment, train_data)
        eval_x, eval_y = model.input_fn(args.segment, eval_data)
        train_x, eval_x, scaler = model.scale_inputs(args.segment, train_x,
                                                     eval_x)

        estimator = model.get_estimator(train_y,
                                        max_depth=args.max_depth,
                                        min_child_weight=args.min_child_weight,
                                        max_delta_step=args.max_delta_step,
                                        reg_lambda=args.reg_lambda,
                                        reg_alpha=args.reg_alpha,
                                        gamma=args.gamma,
                                        learning_rate=args.lr)

        logging.info(f'Max. depth: {estimator.max_depth}')
        logging.info(f'Min. child weight: {estimator.min_child_weight}')
        logging.info(f'Pruning strength: {estimator.gamma}')
        logging.info(f'Learning rate: {estimator.learning_rate}')
        logging.info(f'L2 regularization: {estimator.reg_lambda}')
        logging.info(f'L1 regularization: {estimator.reg_alpha}')

        estimator.fit(train_x,
                      train_y,
                      early_stopping_rounds=args.rounds,
                      eval_set=[(train_x, train_y), (eval_x, eval_y)],
                      eval_metric='auc',
                      verbose=False)

        logging.info(f'Best iteration: {estimator.best_iteration}')
        train_score = roc_auc_score(train_y,
                                    estimator.predict_proba(train_x)[:, 1])
        logging.info(f'Training score = {train_score}')

        # evaluate model
        eval_score = roc_auc_score(eval_y,
                                   estimator.predict_proba(eval_x)[:, 1])
        logging.info(f'Eval score = {eval_score}')

        hpt = HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='eval_auc',
            metric_value=eval_score,
            global_step=0)

    else:
        estimator, scaler = model.train_and_evaluate(
            args.segment,
            args.table_name,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            max_delta_step=args.max_delta_step,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            gamma=args.gamma,
            learning_rate=args.lr)

        model.save_model(args.segment, estimator, scaler, gcs_path,
                         bucket_name, args.timestamp)


if __name__ == "__main__":
    main()