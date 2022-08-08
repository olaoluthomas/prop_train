"""
A task utility and argument parser to initiate Propensity model training and hyperparameter tuning.

https://github.bedbath.com/Advanced-Analytics/propensity_training_pipeline

Author: Simeon Thomas
"""

from proptrainer import model
from sklearn.metrics import roc_auc_score
import argparse
import logging
import os

from hypertune import HyperTune

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment',
                        help='Propensity segment to train.',
                        required=True)
    parser.add_argument("--dataset",
                        help="BQ table with entire data for splitting.",
                        type=str,
                        required=False)
    parser.add_argument('--train_data',
                        help='BQ table name with training data.',
                        type=str,
                        required=False)
    parser.add_argument('--eval_data',
                        help='BQ table name with evaluation data.',
                        type=str,
                        required=False)
    parser.add_argument("--model_dir",
                        help="GCS uri to save model artifact(s)",
                        type=str,
                        required=False)
    parser.add_argument('--rounds',
                        help="Max # of training rounds before early stopping.",
                        default=20,
                        required=False)
    parser.add_argument('--hypertune',
                        help="Perform hyperparameter tuning.",
                        action='store_true',
                        default=False)

    # tree hyperparameters
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
                        default=0.15,
                        type=float,
                        required=False)
    parser.add_argument('--reg_alpha',
                        help="hyperparameter (L1 regularization)",
                        default=0.29,
                        type=float,
                        required=False)
    parser.add_argument('--gamma',
                        help="Strength of pruning.",
                        default=0.005,
                        type=float,
                        required=False)
    parser.add_argument('--lr',
                        help="Learning rate.",
                        default=0.285,
                        type=float,
                        required=False)
    parser.add_argument("--subsamp",
                        help="Portion of features used in training",
                        default=1.0,
                        type=float,
                        required=False)
    parser.add_argument("--col_tree",
                        help="colsample_bytree",
                        default=1.0,
                        type=float,
                        required=False)
    parser.add_argument("--col_level",
                        help="colsample_bylevel",
                        default=1.0,
                        type=float,
                        required=False)
    parser.add_argument("--col_node",
                        help="colsample_bynode",
                        default=1.0,
                        type=float,
                        required=False)

    # linear hyperparameters
    parser.add_argument("--l1_ratio",
                        help="Elastic Net mixing parameter",
                        default=0.75,
                        type=float,
                        required=False)
    parser.add_argument("--alpha",
                        help="regularization strength",
                        default=0.0001,
                        type=float,
                        required=False)
    parser.add_argument("--epsilon",
                        help="epsilon-insensitive loss",
                        default=0.1,
                        type=float,
                        required=False)
    parser.add_argument("--no_intercept",
                        help="fit linear model without an intercept",
                        action='store_true',
                        required=False)

    return parser


def main():
    try:
        args = create_parser().parse_args()

        if not args.dataset and not args.train_data and not args.eval_data:
            try:
                args.train_data = os.environ["AIP_TRAINING_DATA_URI"]
                args.eval_data = os.environ["AIP_VALIDATION_DATA_URI"]
            except Exception as e:
                logging.exception(
                    """If neither of <dataset> or (<train_data> and <eval_data>) is passed as args to the training package,
                use a Vertex Managed Dataset as an argument to the CustomContainerTrainingJob."""
                )

        if args.no_intercept:
            fit_intercept = False
        else:
            fit_intercept = True

        if int(args.segment) in [0, 115, 116]:
            model_type = 'linear'
            kwargs = {
                "l1_ratio": args.l1_ratio,
                "alpha": args.alpha,
                "epsilon": args.epsilon,
                "fit_intercept": fit_intercept,
            }

        else:
            model_type = 'tree'
            kwargs = {
                "max_depth": args.max_depth,
                "min_child_weight": args.min_child_weight,
                "max_delta_step": args.max_delta_step,
                "reg_lambda": args.reg_lambda,
                "reg_alpha": args.reg_alpha,
                "gamma": args.gamma,
                "learning_rate": args.lr,
                "subsample": args.subsamp,
                "colsample_bytree": args.col_tree,
                "colsample_bylevel": args.col_level,
                "colsample_bynode": args.col_node
            }

        if args.hypertune:
            if args.dataset:
                train_data, eval_data = model.create_dataframe(
                    segment=args.segment, dataset=args.dataset)
            else:
                train_data, eval_data = model.create_dataframe(
                    segment=args.segment,
                    train_data=args.train_data,
                    eval_data=args.eval_data)

            train_x, train_y = model.input_fn(args.segment, train_data)
            eval_x, eval_y = model.input_fn(args.segment, eval_data)
            train_x, eval_x, scaler = model.scale_inputs(train_x, eval_x)

            estimator = model.get_estimator(train_y, model_type, args.rounds,
                                            **kwargs)

            if model_type == 'linear':
                logging.info(
                    f"Elastic Net mixing parameter: {estimator.l1_ratio}")
                logging.info(
                    f"Fitting with intercept: {estimator.fit_intercept}")
                logging.info(f"Regularization multiplier: {estimator.alpha}")
                logging.info(f"Max. # of epochs: {estimator.max_iter}")
                logging.info(f"Initial learning rate: {estimator.eta0}")
                logging.info(
                    f"Learning rate schedule: {estimator.learning_rate}")
                logging.info(
                    f"Reuse solution of previous call? {estimator.warm_start}")
                logging.info(f"Early stopping: {estimator.early_stopping}")
                logging.info(f"Tolerance to early stopping: {estimator.tol}")
                logging.info(
                    f"# of epochs without improvement prior to early stopping: {estimator.n_iter_no_change}"
                )

                estimator.fit(train_x, train_y)

            elif model_type == 'tree':
                logging.info(f"Max. depth: {estimator.max_depth}")
                logging.info(f"Max. delta step: {estimator.max_delta_step}")
                logging.info(
                    f"Min. child weight: {estimator.min_child_weight}")
                logging.info(f"Pruning strength: {estimator.gamma}")
                logging.info(f"Learning rate: {estimator.learning_rate}")
                logging.info(f"L2 regularization: {estimator.reg_lambda}")
                logging.info(f"L1 regularization: {estimator.reg_alpha}")
                logging.info(f"Subsample ratio: {estimator.subsample}")
                logging.info(
                    f"Colsample_bytree ratio: {estimator.colsample_bytree}")
                logging.info(
                    f"Colsample_bylevel ratio: {estimator.colsample_bylevel}")
                logging.info(
                    f"Colsample_bynode ratio: {estimator.colsample_bynode}")
                logging.info(
                    f"# of rounds prior to early stopping: {args.rounds}")

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
            if args.dataset:
                estimator, scaler = model.train_and_evaluate(
                    segment=args.segment,
                    dataset=args.dataset,
                    n_rounds=args.rounds,
                    **kwargs)
            else:
                estimator, scaler = model.train_and_evaluate(
                    segment=args.segment,
                    train_data=args.train_data,
                    eval_data=args.eval_data,
                    n_rounds=args.rounds,
                    **kwargs)

            if not args.model_dir:
                args.model_dir = os.environ["AIP_MODEL_DIR"]
            model.save_model(estimator, scaler, args.model_dir)

    except Exception as e:
        logger.exception("Fatal error in main()")


if __name__ == "__main__":
    main()
