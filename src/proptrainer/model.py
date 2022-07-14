"""
A utility to facilitate DM Propensity model training in Vertex AI
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import compute_class_weight
import joblib
import pickle
import logging

from proptrainer import features, clients


def create_queries(segment, table_name, dataset='SANDBOX_ANALYTICS'):
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
    project = 'dw-bq-data-d00'
    time_condition = " AND IN_HOME_DT >= DATE'2021-01-01'"
    if segment in [115, 116]:
        seg_condition = f""" WHERE em_segment = {segment}
        AND BBB_R_2Y IS NOT NULL"""
    elif segment == 'prospects':
        seg_condition = f""" WHERE em_segment IN (115, 116)
        AND BBB_R_2Y IS NULL"""
    else:
        seg_condition = f" WHERE em_segment = {segment}"
    query = f"""SELECT *
    FROM {project}.{dataset}.{table_name}{seg_condition}{time_condition}"""
    train_query = f"""SELECT * FROM ({query}) 
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(COUPON_BARCODE AS STRING))), 10) < 8"""
    eval_query = f"""SELECT * FROM ({query})
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(COUPON_BARCODE AS STRING))), 10) >= 8"""

    return train_query, eval_query


def preprocess(segment, dataframe):
    """
    Function to convert data types and null imputation.

    Parameters
    ----------
    segment: int/str.
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
                'A_A9350N_ECONOMIC_STB_01_10', 'BBB_INSTORE_F', 'BBB_INSTORE_F_2Y',
                'TOTAL_TXNS_L12M', 'BBB_R_2Y'
        ]:
            dataframe[column] = dataframe[column].astype('int')
        if column in [
                'COUPON_ANY_AMT', 'COUPON_SALES_Q_05', 'COUPON_SALES_Q_08',
                'HARMON_SALES_L12M', 'TOTAL_SALES_L12M', 'Z_AVG_PCT_SALES_RETURNED'
        ]:
            dataframe[column] = dataframe[column].astype('float')
        if column in [
                'COUPON_SALES_Q_05', 'COUPON_SALES_Q_08', 'TOTAL_SALES_L12M',
                'NUM_TOTAL_ITEMS'
        ]:
            dataframe.loc[(dataframe[column] < 0), column] = 0

    return dataframe


def get_data_from_bq(segment, query):
    """
    To convert data from BQ to Pandas DataFrame.

    Parameters
    ----------
    query: str
        BQ query to reference training data of segment.

    Returns
    -------
    data: pandas DataFrame
    """
    bq_client, bqstorageclient = clients.get_bq_clients()
    dataframe = bq_client.query(query).result().to_dataframe(
        bqstorage_client=bqstorageclient)
    bq_client.close()
    dataframe = preprocess(segment, dataframe)

    return dataframe


def create_dataframe(segment, table_name):
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
    train_query, eval_query = create_queries(segment, table_name)
    logging.info('Preprocessing training data...')
    train_df = get_data_from_bq(segment, train_query)
    logging.info('Preprocessing evaluation data...')
    eval_df = get_data_from_bq(segment, eval_query)

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


def scale_inputs(segment, train_data, eval_data, outliers=False):
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


def get_estimator(y_train, random_state=777, **kwargs):
    """
    Instantiate the estimator.
    Options: XGBoostClassifier, SGDClassifier, LogisticRegression.

    Parameters
    ----------
    y_train: array or pandas Series.
    model: str
        If 'tree', model will be XGBoost. If 'sgd', model will
        be a Stochastic Gradient Descent Classifier. Else, Logistic Regression.
    penalty: str
        For Logistic Regression, either of l1, l2 and elasticnet
    random_state: 
        random number seed.

    Returns
    -------
    Estimator with non-tuned hyperparameters initialized.
    """
    estimator = XGBClassifier(n_estimators=200,
                              objective='binary:logistic',
                              booster='gbtree',
                              use_label_encoder=False,
                              tree_method='approx',
                              base_score=get_base_score(y_train),
                              scale_pos_weight=get_pos_weight(y_train),
                              importance_type='total_gain',
                              verbosity=0,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              colsample_bylevel=0.8,
                              colsample_bynode=0.8,
                              random_state=random_state,
                              **kwargs)
    logging.info('Using XGBoost Classifier...')

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


def train_and_evaluate(segment, table_name, n_rounds=10, **kwargs):
    """
    Train and evaluate the estimator

    Parameters
    ----------
    segment: int/str
        model segment
    table_name: str

    Returns
    -------
    estimator: ML estimator (sklearn OR xgboost).
    scaler: Preprocessor to scale and center input data.
    """
    train_df, eval_df = create_dataframe(segment, table_name)
    train_x, train_y = input_fn(segment, train_df)
    eval_x, eval_y = input_fn(segment, eval_df)
    train_x, eval_x, scaler = scale_inputs(segment, train_x, eval_x)
    estimator = get_estimator(train_y, **kwargs)

    # train model
    logging.info(f'Max. depth: {estimator.max_depth}')
    logging.info(f'Min. child weight: {estimator.min_child_weight}')
    logging.info(f'L2 regularization: {estimator.reg_lambda}')
    logging.info(f'L1 regularization: {estimator.reg_alpha}')
    logging.info(f'Pruning strength: {estimator.gamma}')
    estimator.fit(train_x,
                  train_y,
                  early_stopping_rounds=n_rounds,
                  eval_set=[(train_x, train_y), (eval_x, eval_y)],
                  eval_metric='auc',
                  verbose=False)
    logging.info(f'Best iteration: {estimator.best_iteration}')

    train_score = roc_auc_score(train_y,
                                estimator.predict_proba(train_x)[:, 1])
    logging.info(f'Training score = {train_score}')

    # evaluate model
    eval_score = roc_auc_score(eval_y, estimator.predict_proba(eval_x)[:, 1])
    logging.info(f'Eval score = {eval_score}')

    # calibrate model
    calibrated_estimator = calibrate(estimator, eval_x, eval_y)

    return calibrated_estimator, scaler


def save_model(segment, estimator, scaler, gcs_path, bucket_name, timestamp):
    """
    Function to save model and scaler artifacts.

    Parameters
    ----------
    """
    model, scal = 'model.joblib', 'scaler.pkl'
    with open(model, 'wb') as m:
        joblib.dump(estimator, m)
    with open(scal, 'wb') as s:
        pickle.dump(scaler, s)
    gcs_path = f'{gcs_path}/{segment}/{timestamp}/model'
    model_path = f'{gcs_path}/{model}'
    scaler_path = f'{gcs_path}/{scal}'
    storage_client = clients.get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    model_blob = bucket.blob(model_path)
    model_blob.upload_from_filename(model)
    scaler_blob = bucket.blob(scaler_path)
    scaler_blob.upload_from_filename(scal)
    logging.info(f'Saved scaler and model to {bucket_name}/{gcs_path}')