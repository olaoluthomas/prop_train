"""
A utility to facilitate DM Propensity model training in Vertex AI.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import compute_class_weight
import os, cloudpickle, logging, datetime

from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.cloud import storage
from google.oauth2 import service_account


def get_credentials(key_path):
    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return credentials


def get_project_id(key_path):
    credentials = get_credentials(key_path)
    return credentials.project_id


def get_bq_clients(key_path):
    credentials = get_credentials(key_path)
    bq_client = bigquery.Client(credentials=credentials,
                                project=credentials.project_id)
    bqstorageclient = bigquery_storage.BigQueryReadClient(
        credentials=credentials)
    return bq_client, bqstorageclient


def get_storage_client(key_path):
    credentials = get_credentials(key_path)
    return storage.Client(credentials=credentials,
                          project=credentials.project_id)


def create_queries(segment, key_path, table_name, dataset='SANDBOX_ANALYTICS'):
    project = get_project_id(key_path)
    if segment in [115, 116]:
        seg_condition = f""" WHERE em_segment = {segment}
        AND BBB_R_2Y IS NOT NULL"""
    elif segment == 'prospects':
        seg_condition = f""" WHERE em_segment IN (115, 116)
        AND BBB_R_2Y IS NULL"""
    else:
        seg_condition = f" WHERE em_segment = {segment}"
    query = f"""SELECT *
    FROM {project}.{dataset}.{table_name}{seg_condition}"""
    train_query = f"""SELECT * FROM ({query}) 
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(COUPON_BARCODE AS STRING))), 10) < 8"""
    eval_query = f"""SELECT * FROM ({query})
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(COUPON_BARCODE AS STRING))), 10) >= 8"""
    return train_query, eval_query


def get_data_from_bq(key_path, query):
    bq_client, bqstorageclient = get_bq_clients(key_path)
    dataframe = bq_client.query(query).result().to_dataframe(
        bqstorage_client=bqstorageclient)
    bq_client.close()
    if 'BBB_R_2Y' in dataframe.columns:
        dataframe['BBB_R_2Y'].fillna(0, inplace=True)
    return dataframe


def create_dataframe(segment, key_path, table_name):
    train_query, eval_query = create_queries(segment, key_path, table_name)
    train_df = get_data_from_bq(key_path, train_query)
    eval_df = get_data_from_bq(key_path, eval_query)
    return train_df, eval_df


def scale_inputs(train_data, eval_data, cat_features):
    features = [col for col in train_data.columns]
    cat_features = [feat for feat in cat_features if feat in features]
    cont_features = [feat for feat in features if feat not in cat_features]
    scaler = RobustScaler(unit_variance=True)
    scaled_train = pd.DataFrame(
        scaler.fit_transform(train_data[cont_features]),
        columns=[col for col in train_data[cont_features].columns])
    scaled_train[cat_features] = train_data[cat_features]
    scaled_eval = pd.DataFrame(
        scaler.transform(eval_data[cont_features]),
        columns=[col for col in eval_data[cont_features].columns])
    scaled_eval[cat_features] = eval_data[cat_features]
    return scaled_train, scaled_eval, scaler


def input_fn(segment, df, target='TARGET_14'):
    columns = feature_lookup[str(segment)]
    features = df[columns]
    target = df[target]
    return features, target


def get_class_weights(y_train):
    return dict(
        zip(
            np.unique(y_train),
            compute_class_weight(class_weight='balanced',
                                 classes=np.unique(y_train),
                                 y=y_train)))


def get_estimator(y_train, penalty='l1', random_state=777):
    # have hypertune handle hyperparameter tuning of reg. strength
    # create ability to get other types of estimators
    if penalty != 'elasticnet':
        l1_ratio = None
    else:
        l1_ratio = 0.7
    return LogisticRegression(penalty=penalty,
                              C=0.001,
                              solver='saga',
                              warm_start=True,
                              max_iter=2000,
                              l1_ratio=l1_ratio,
                              random_state=random_state,
                              class_weight=get_class_weights(y_train))


def train_and_evaluate(segment, table_name, key_path):
    # train model
    train_df, eval_df = create_dataframe(segment, key_path, table_name)
    train_x, train_y = input_fn(segment, train_df)
    eval_x, eval_y = input_fn(segment, eval_df)
    train_x, eval_x, scaler = scale_inputs(train_x, eval_x, cat_features)
    estimator = get_estimator(train_y)
    estimator.fit(train_x, train_y)
    train_proba = estimator.predict_proba(train_x)[:, 1]
    train_score = roc_auc_score(train_y, train_proba)
    logging.info(f'Training score = {train_score}')
    # evaluate model
    eval_proba = estimator.predict_proba(eval_x)[:, 1]
    eval_score = roc_auc_score(eval_y, eval_proba)
    logging.info(f'Eval score = {eval_score}')
    return estimator, eval_score, scaler


def get_scaler_path(segment, scaler, scaler_path):
    scaler_name = f'scaler_{segment}.pkl'
    with open(scaler_name, 'wb') as s:
        cloudpickle.dump(scaler, s)
    return scaler_name, f'{scaler_path}/scaler/{scaler_name}'


def save_model(segment, estimator, scaler, key_path, gcs_path, bucket_name):
    model = f'model_{segment}.pkl'
    with open(model, 'wb') as m:
        cloudpickle.dump(estimator, m)
    gcs_path = f'{gcs_path}/{segment}'
    export_path = f"{gcs_path}/{datetime.datetime.now().strftime('export_%Y%m%d_%H%M%S')}"
    model_path = f'{export_path}/{model}'
    scaler_name, scaler_path = get_scaler_path(segment, scaler, export_path)
    storage_client = get_storage_client(key_path)
    bucket = storage_client.bucket(bucket_name)
    model_blob = bucket.blob(model_path)
    model_blob.upload_from_filename(model)
    logging.info(f'Saved model to {bucket_name}/{export_path}')
    scaler_blob = bucket.blob(scaler_path)
    scaler_blob.upload_from_filename(scaler_name)
    logging.info(f'Saved data scaler to {bucket_name}/{export_path}')