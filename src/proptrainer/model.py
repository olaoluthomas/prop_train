# Copyright 2022 Bed Bath & Beyond Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A utility to facilitate DM Propensity model training in Vertex AI.

https://github.bedbath.com/Advanced-Analytics/propensity_training_pipeline

Author: Simeon Thomas
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import compute_class_weight
import joblib
import pickle
import logging
import os

from proptrainer import features


def get_credentials():
    """
    Function to load GCP service account credentials
    using package key file.
    """
    from importlib_resources import files
    from google.oauth2 import service_account
    from proptrainer import key

    datakey = files(key).joinpath('datakey.json')
    credentials = service_account.Credentials.from_service_account_file(
        datakey,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    return credentials


def get_bq_clients():
    """
    Function to instantiate BQ & BQ storage client.

    Returns
    -------
    bq_client: bigquery.Client
    bqstorageclient: bigquery_storage.BigQueryReadClient
    """
    from google.cloud import bigquery
    from google.cloud import bigquery_storage

    credentials = get_credentials()
    bq_client = bigquery.Client(credentials=credentials,
                                project=credentials.project_id)
    bqstorageclient = bigquery_storage.BigQueryReadClient(
        credentials=credentials)

    return bq_client, bqstorageclient


def create_queries(segment, dataset=None, train_data=None, eval_data=None):
    """
    Function to build queries for training and evaluation sets.

    Parameters
    ----------
    segment: int/str
        Propensity model segment.
        One of [104, 106, 108, 109, 110, 112, 114, 115, 116, prospects]
    table_name: str
        BQ table with training data.
    dataset: str
        BQ dataset where table_name resides.
    """
    time_condition = " AND IN_HOME_DT >= DATE'2021-01-01'"
    if int(segment) in [115, 116]:
        seg_condition = f""" WHERE em_segment = {segment}
        AND BBB_R_2Y IS NOT NULL"""
    elif int(segment) == 0:
        seg_condition = f""" WHERE em_segment IN (115, 116)
        AND BBB_R_2Y IS NULL"""
    else:
        seg_condition = f" WHERE em_segment = {segment}"

    if dataset:
        query = f"""SELECT *
        FROM {dataset}{seg_condition}{time_condition}"""
        train_query = f"""SELECT * FROM ({query})
        WHERE MOD(ABS(FARM_FINGERPRINT(CAST(COUPON_BARCODE AS STRING))), 10) < 8"""
        eval_query = f"""SELECT * FROM ({query})
        WHERE MOD(ABS(FARM_FINGERPRINT(CAST(COUPON_BARCODE AS STRING))), 10) >= 8"""
    else:
        train_query = f"""SELECT *
        FROM {train_data}{seg_condition}{time_condition}"""
        eval_query = f"""SELECT *
        FROM {eval_data}{seg_condition}{time_condition}"""

    return train_query, eval_query


def preprocess(dataframe):
    """
    Function to convert data types and null imputation.

    Parameters
    ----------
    dataframe: pandas DataFrame.
    logs: list
        features for log transformation.
    eps: float
        added value to avoid zero-value log transformation error.

    Returns
    -------
    dataframe: pandas DataFrame
        preprocessed dataframe
    """
    columns = [col for col in dataframe.columns]
    if 'BBB_R_2Y' in columns:
        dataframe['BBB_R_2Y'].fillna(0, inplace=True)
    for column in columns:
        if column in [
                'BBB_F_DECILE_2Y', 'BBB_ONCOUPON_R_DECILE_2Y',
                'A_A9350N_ECONOMIC_STB_01_10', 'BBB_INSTORE_F',
                'BBB_INSTORE_F_2Y', 'TOTAL_TXNS_L12M', 'BBB_R_2Y'
        ]:
            dataframe[column] = dataframe[column].astype('int')
        if column in [
                'COUPON_ANY_AMT', 'COUPON_SALES_Q_05', 'COUPON_SALES_Q_08',
                'HARMON_SALES_L12M', 'TOTAL_SALES_L12M',
                'Z_AVG_PCT_SALES_RETURNED'
        ]:
            dataframe[column] = dataframe[column].astype('float')
        if column in [
                'COUPON_SALES_Q_05', 'COUPON_SALES_Q_08', 'TOTAL_SALES_L12M',
                'NUM_TOTAL_ITEMS'
        ]:
            dataframe.loc[(dataframe[column] < 0), column] = 0

    return dataframe


def get_data_from_bq(query):
    """
    To convert data from BQ to Pandas DataFrame.

    Parameters
    ----------
    query: str
        BQ query to reference propensity data.

    Returns
    -------
    data: pandas DataFrame
    """
    bq_client, bqstorageclient = get_bq_clients()
    dataframe = bq_client.query(query).result().to_dataframe(
        bqstorage_client=bqstorageclient)
    bq_client.close()
    dataframe = preprocess(dataframe)

    return dataframe


def create_dataframe(segment, dataset=None, train_data=None, eval_data=None):
    """
    Create training and evaluation DataFrames.

    Parameters
    ----------
    segment: int/str
    table_name: str

    Returns
    -------
    train_df: dataframe for training estimator.
    eval_df: dataframe for model evaluation.
    """
    if dataset:
        train_query, eval_query = create_queries(segment=segment,
                                                 dataset=dataset)
    else:
        train_query, eval_query = create_queries(segment=segment,
                                                 train_data=train_data,
                                                 eval_data=eval_data)

    logging.info('Preprocessing training data...')
    train_df = get_data_from_bq(train_query)
    logging.info('Preprocessing testing data...')
    eval_df = get_data_from_bq(eval_query)

    return train_df, eval_df


def input_fn(segment, df, target='TARGET_14'):
    """
    Function to separate features from the target.

    Parameters
    ----------
    segment: int/str
    df: input dataframe
    target: str

    Returns
    -------
    inputs: pandas DataFrame
    target: pandas Series
    """
    columns = features.feature_lookup[str(segment)]
    inputs = df[columns]
    target = df[target]

    return inputs, target


def scale_inputs(train_data, eval_data, outliers=False):
    """
    Function to generate a scaler on training data and transform both training
    and evaluation data to zero mean and unit variance.

    Parameters
    ----------
    train_data: DataFrame.
    eval_data: DataFrame.

    Returns
    -------
    scaled_train: Scaled training dataframe.
    scaled_eval: Scaled evaluation dataframe.
    scaler: Scaling and centering preprocessor.
    """
    if outliers == True:
        scaler = RobustScaler(unit_variance=True)
    else:
        scaler = StandardScaler()

    scaled_train = scaler.fit_transform(train_data)
    scaled_eval = scaler.transform(eval_data)
    logging.info(f'Training and evaluation features scaled with {scaler}.')

    return scaled_train, scaled_eval, scaler


def get_class_weights(y_train):
    classes = np.unique(y_train)
    return dict(
        zip(
            classes,
            compute_class_weight(class_weight="balanced",
                                 classes=classes,
                                 y=y_train)))


def get_base_score(y_train):
    """
    Compute global bias

    Parameters
    ----------
    y_train: array or pandas Series.
        Collection of classes to be predicted.

    Returns
    -------
    float
        Global bias of positive class (target). 
    """
    return round(y_train.value_counts()[1] / len(y_train), 5)


def get_pos_weight(y_train):
    """
    Compute the ratio of negative class to positive class.

    Parameters
    ----------
    y_train: array or pandas Series.
        Collection of classes to be predicted.

    Returns
    -------
    float
        Ratio of negative class to positive class.
    """
    return round(y_train.value_counts()[0] / y_train.value_counts()[1], 5)


def get_estimator(y_train, model_type, n_rounds, random_state=777, **kwargs):
    """
    Instantiate the estimator.
    Options: XGBoostClassifier, SGDClassifier, LogisticRegression.

    Parameters
    ----------
    y_train: array or pandas Series.
    model: str
        If 'tree', model will be XGBoost. If 'linear', model will
        be a Stochastic Gradient Descent Classifier.
    random_state: 
        random number seed.

    Returns
    -------
    Estimator with non-tuned hyperparameters initialized.
    """
    if model_type == 'tree':
        estimator = XGBClassifier(n_estimators=500,
                                  objective='binary:logistic',
                                  booster='gbtree',
                                  use_label_encoder=False,
                                  tree_method='approx',
                                  base_score=get_base_score(y_train),
                                  scale_pos_weight=get_pos_weight(y_train),
                                  importance_type='total_gain',
                                  verbosity=0,
                                  random_state=random_state,
                                  **kwargs)
        logging.info('Using XGBoost Classifier...')

    elif model_type == 'linear':
        estimator = SGDClassifier(loss='modified_huber',
                                  penalty='elasticnet',
                                  early_stopping=True,
                                  tol=0.005,
                                  validation_fraction=0.15,
                                  eta0=0.1,
                                  learning_rate='adaptive',
                                  class_weight=get_class_weights(y_train),
                                  warm_start=True,
                                  n_iter_no_change=n_rounds,
                                  n_jobs=os.cpu_count(),
                                  random_state=random_state,
                                  verbose=0,
                                  **kwargs)
        logging.info('Using SGDClassifier...')

    else:
        logging.info(
            "Invalid model_type used. Please select one of 'tree' or 'linear'."
        )

    return estimator


def calibrate(estimator, X_test, y_test):
    """
    """
    calibrated_estimator = CalibratedClassifierCV(base_estimator=estimator,
                                                  method='isotonic',
                                                  cv='prefit')
    logging.info('Calibrating estimator...')
    calibrated_estimator.fit(X_test, y_test)

    return calibrated_estimator


def train_and_evaluate(segment,
                       dataset=None,
                       train_data=None,
                       eval_data=None,
                       n_rounds,
                       **kwargs):
    """
    Train and evaluate the estimator

    Parameters
    ----------
    segment: int/str
        model segment
    table_name: str

    Returns
    -------
    estimator: ML estimator (XGBoosted Tree or SGDClassifier).
    scaler: Preprocessor to scale and center input data.
    """
    if dataset:
        train_df, eval_df = create_dataframe(segment=segment, dataset=dataset)
    else:
        train_df, eval_df = create_dataframe(segment=segment,
                                             train_data=train_data,
                                             eval_data=eval_data)

    train_x, train_y = input_fn(segment, train_df)
    eval_x, eval_y = input_fn(segment, eval_df)
    train_x, eval_x, scaler = scale_inputs(train_x, eval_x)

    if int(segment) in [0, 115, 116]:
        model_type = 'linear'
    else:
        model_type = 'tree'
    try:
        estimator = get_estimator(train_y,
                                  model_type=model_type,
                                  n_rounds=n_rounds,
                                  **kwargs)

        # train model
        if model_type == 'tree':
            logging.info(f"Max. depth: {estimator.max_depth}")
            logging.info(f"Max. delta step: {estimator.max_delta_step}")
            logging.info(f"Min. child weight: {estimator.min_child_weight}")
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
            logging.info(f"# of rounds prior to early stopping: {n_rounds}")
            estimator.fit(train_x,
                          train_y,
                          early_stopping_rounds=n_rounds,
                          eval_set=[(train_x, train_y), (eval_x, eval_y)],
                          eval_metric='auc',
                          verbose=False)
            logging.info(f'Best iteration: {estimator.best_iteration}')

        elif model_type == 'linear':
            logging.info(f"Elastic Net mixing parameter: {estimator.l1_ratio}")
            logging.info(f"Fitting with intercept: {estimator.fit_intercept}")
            logging.info(f"Regularization multiplier: {estimator.alpha}")
            logging.info(f"Max. # of epochs: {estimator.max_iter}")
            logging.info(f"Initial learning rate: {estimator.eta0}")
            logging.info(f"Learning rate schedule: {estimator.learning_rate}")
            logging.info(
                f"Reuse solution of previous call? {estimator.warm_start}")
            logging.info(f"Early stopping: {estimator.early_stopping}")
            logging.info(f"Tolerance to early stopping: {estimator.tol}")
            logging.info(
                f"# of epochs without improvement prior to early stopping: {estimator.n_iter_no_change}"
            )

            estimator.fit(train_x, train_y)

    except Exception as e:
        logging.exception(
            "Fatal error in training. Evaluation cannot continue...")

    train_score = roc_auc_score(train_y,
                                estimator.predict_proba(train_x)[:, 1])
    logging.info(f'Training score = {train_score}')

    # evaluate model
    eval_score = roc_auc_score(eval_y, estimator.predict_proba(eval_x)[:, 1])
    logging.info(f'Test score = {eval_score}')

    # calibrate model
    calibrated_estimator = calibrate(estimator, eval_x, eval_y)

    return calibrated_estimator, scaler


def save_model(estimator, scaler, model_dir):
    """
    Function to save model and scaler artifacts.

    Parameters
    ----------
    """
    from google.cloud import storage
    import os

    model, scal = 'model.joblib', 'scaler.pkl'
    with open(model, 'wb') as m:
        joblib.dump(estimator, m)
    with open(scal, 'wb') as s:
        pickle.dump(scaler, s)

    model_storage_path = os.path.join(model_dir, model)
    scaler_storage_path = os.path.join(model_dir, scal)
    storage_client = storage.Client()
    model_blob = storage.blob.Blob.from_string(model_storage_path,
                                               client=storage_client)
    model_blob.upload_from_filename(model)
    scaler_blob = storage.blob.Blob.from_string(scaler_storage_path,
                                                client=storage_client)
    scaler_blob.upload_from_filename(scal)
    logging.info(f"Saved scaler and model to {model_dir}")