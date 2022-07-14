"""
A utility to facilitate DM Propensity model training in Vertex AI
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import compute_class_weight
import pickle, logging
import argparse

from google.cloud import bigquery
from google.cloud import storage

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bucket_name = 'dev_dw_npii_adhoc'
gcs_path = 'ml/dm-propensity'
table_name = 'dm_pc_tiny_data'
target = 'TARGET_14'
cat_features = ['is_in_trade_area', 'A_A9350N_ECONOMIC_STB_01_10']
data_project = 'dw-bq-data-d00'

COLUMN_NAMES_104 = [
    'BBB_R_2Y', 'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BUYS_Q_08',
    'NUM_TXNS', 'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D', 'TOTAL_TXNS_L12M',
    'Z_AVG_PCT_SALES_RETURNED', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10',
]

COLUMN_NAMES_106 = [
    'BBB_F_DECILE_2Y', 'BUYS_M_01', 'BUYS_Q_08', 'COUPON_SALES_Q_05',
    'NUM_TXNS', 'PH_CSTACK_90D', 'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D',
    'PH_PFREQ365D', 'Z_AVG_PCT_SALES_RETURNED', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10',
]

COLUMN_NAMES_108 = [
    'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BUYS_Q_01', 'BUYS_Q_02',
    'BUYS_Q_04', 'BUYS_Q_08', 'NUM_PERIODS', 'PH_MREDEEM730D_PERC',
    'PH_NMFREQ182D', 'PH_PFREQ365D', 'RECENCY', 'TOTAL_TXNS_L12M',
    'Z_AVG_PCT_SALES_RETURNED', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10',
]

COLUMN_NAMES_109 = [
    'BBB_R_2Y', 'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BUYS_M_24',
    'COUPON_SALES_Q_08', 'NUM_PERIODS', 'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D',
    'TOTAL_TXNS_L12M', 'Z_ONCOUPON_F_DEC_4_2Y', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10',
]

COLUMN_NAMES_110 = [
    'A_A8642_HM_MKT_VAL', 'BUYS_Q_01', 'BUYS_Q_04', 'BUYS_Q_08', 'NUM_PERIODS',
    'PH_MREDEEM730D_PERC', 'PH_PFREQ365D', 'TOTAL_TXNS_L12M',
    'is_in_trade_area', 'A_A9350N_ECONOMIC_STB_01_10',
]

COLUMN_NAMES_112 = [
    'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BBB_INSTORE_F_2Y', 'BUYS_Q_01',
    'BUYS_Q_02', 'BUYS_M_01', 'BUYS_Q_04', 'BUYS_Q_08', 'NUM_PERIODS',
    'PH_CREDEEM365D', 'PH_MREDEEM182D_PERC', 'PH_MREDEEM730D_PERC',
    'PH_NMFREQ182D', 'PH_PFREQ365D', 'PH_PREDEEM548D', 'RECENCY',
    'Z_AVG_PCT_SALES_RETURNED', 'Z_MARKDOWNS', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10',
]

COLUMN_NAMES_114 = [
    'BBB_R_2Y', 'A_A8642_HM_MKT_VAL', 'BBB_INSTORE_F_2Y', 'BUYS_Q_01',
    'BUYS_Q_02', 'BUYS_M_01', 'BUYS_Q_04', 'BUYS_Q_08', 'COUPON_ANY_AMT',
    'NUM_PERIODS', 'NUM_TOTAL_ITEMS', 'PH_MREDEEM182D_PERC',
    'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D', 'Z_AVG_PCT_SALES_RETURNED',
    'is_in_trade_area', 'A_A9350N_ECONOMIC_STB_01_10',
]

COLUMN_NAMES_115 = [
    'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BBB_ONCOUPON_R_DECILE_2Y',
    'BUYS_Q_01', 'BUYS_Q_02', 'BUYS_M_01', 'BUYS_Q_04', 'COUPON_Q_01',
    'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D', 'PH_PFREQ365D', 'PH_PREDEEM365D',
    'PH_PREDEEM730D', 'TOTAL_TXNS_L12M', 'Z_AVG_PCT_SALES_RETURNED',
    'Z_INSTORE_M_DEC_10_2Y', 'Z_ONCOUPON_F_DEC_4_2Y', 'is_in_trade_area',
]

COLUMN_NAMES_116 = [
    'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BBB_ONCOUPON_R_DECILE_2Y',
    'BUYS_Q_02', 'BUYS_M_01', 'BUYS_Q_04', 'BUYS_Q_08', 'COUPON_Q_01',
    'NUM_TOTAL_ITEMS', 'PH_MREDEEM182D_PERC', 'PH_MREDEEM730D_PERC',
    'PH_NMFREQ182D', 'PH_PFREQ365D', 'RECENCY', 'TOTAL_TXNS_L12M',
    'Z_AVG_PCT_SALES_RETURNED', 'Z_AVG_RECENCY', 'Z_INSTORE_M_DEC_10_2Y',
    'Z_ONCOUPON_F_DEC_4_2Y', 'Z_MARKDOWNS', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10',
]

COLUMN_NAMES_P = [
    'BUYS_Q_01', 'BUYS_M_06', 'BUYS_M_24', 'BUYS_Q_04', 'COUPON_SALES_Q_05',
    'NUM_TOTAL_ITEMS', 'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D',
    'TOTAL_TXNS_L12M', 'Z_INSTORE_M_DEC_10_2Y', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10',
]

feature_lookup = {
    '104': COLUMN_NAMES_104,
    '106': COLUMN_NAMES_106,
    '108': COLUMN_NAMES_108,
    '109': COLUMN_NAMES_109,
    '110': COLUMN_NAMES_110,
    '112': COLUMN_NAMES_112,
    '114': COLUMN_NAMES_114,
    '115': COLUMN_NAMES_115,
    '116': COLUMN_NAMES_116,
    'prospects': COLUMN_NAMES_P
}


def create_queries(segment, table_name, dataset='SANDBOX_ANALYTICS'):
    if segment in [115, 116]:
        seg_condition = f""" WHERE em_segment = {segment}
        AND BBB_R_2Y IS NOT NULL"""
    elif segment == 'prospects':
        seg_condition = f""" WHERE em_segment IN (115, 116)
        AND BBB_R_2Y IS NULL"""
    else:
        seg_condition = f" WHERE em_segment = {segment}"
    query = f"""SELECT *
    FROM {data_project}.{dataset}.{table_name}{seg_condition}"""
    train_query = f"""SELECT * FROM ({query}) 
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(COUPON_BARCODE AS STRING))), 10) < 8"""
    eval_query = f"""SELECT * FROM ({query})
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(COUPON_BARCODE AS STRING))), 10) >= 8"""
    return train_query, eval_query


def get_data_from_bq(query):
    bq_client = bigquery.Client(project=data_project)
    dataframe = bq_client.query(query).result().to_dataframe()
    bq_client.close()
    if 'BBB_R_2Y' in dataframe.columns:
        dataframe['BBB_R_2Y'].fillna(0, inplace=True)
    return dataframe


def create_dataframe(segment, table_name):
    train_query, eval_query = create_queries(segment, table_name)
    train_df = get_data_from_bq(train_query)
    eval_df = get_data_from_bq(eval_query)
    return train_df, eval_df


def input_fn(segment, df, target=target):
    columns = feature_lookup[str(segment)]
    features = df[columns]
    target = df[target]
    return features, target


def scale_inputs(train_data, eval_data, cat_features):
    features = [col for col in train_data.columns]
    cat_features = [feat for feat in cat_features if feat in features]
    cont_features = [feat for feat in features if feat not in cat_features]
    scaler = StandardScaler()
    scaled_train = pd.DataFrame(
        scaler.fit_transform(train_data[cont_features]),
        columns=[col for col in train_data[cont_features].columns])
    scaled_train[cat_features] = train_data[cat_features]
    scaled_eval = pd.DataFrame(
        scaler.transform(eval_data[cont_features]),
        columns=[col for col in eval_data[cont_features].columns])
    scaled_eval[cat_features] = eval_data[cat_features]
    return scaled_train, scaled_eval, scaler


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
        l1_ratio = 0.2
    return LogisticRegression(penalty=penalty,
                              C=0.01,
                              solver='saga',
                              warm_start=True,
                              max_iter=2000,
                              l1_ratio=l1_ratio,
                              random_state=random_state,
                              class_weight=get_class_weights(y_train))


def train_and_evaluate(segment, table_name):
    # train model
    train_df, eval_df = create_dataframe(segment, table_name)
    train_x, train_y = input_fn(segment, train_df)
    eval_x, eval_y = input_fn(segment, eval_df)
    train_x, eval_x, scaler = scale_inputs(train_x, eval_x, cat_features)
    estimator = get_estimator(train_y, penalty='elasticnet')
    estimator.fit(train_x, train_y)
    train_proba = estimator.predict_proba(train_x)[:, 1]
    train_score = roc_auc_score(train_y, train_proba)
    logging.info(f'Training score = {train_score}')
    # evaluate model
    eval_proba = estimator.predict_proba(eval_x)[:, 1]
    eval_score = roc_auc_score(eval_y, eval_proba)
    logging.info(f'Eval score = {eval_score}')
    return estimator, eval_score, scaler


def get_scaler(segment, scaler, scaler_path):
    scaler_name = f'scaler_{segment}.pkl'
    with open(scaler_name, 'wb') as s:
        pickle.dump(scaler, s)
    return scaler_name, f'{scaler_path}/scalers/{segment}/vertex_ai/{scaler_name}'


def save_model(segment, estimator, scaler, gcs_path, bucket_name):
    model = 'model.pkl'
    with open(model, 'wb') as m:
        pickle.dump(estimator, m)
    model_path = f'{gcs_path}/models/{segment}/vertex_ai/model'
    scaler_name, scaler_path = get_scaler(segment, scaler, gcs_path)
    storage_client = storage.Client(project=data_project)
    bucket = storage_client.bucket(bucket_name)
    model_blob = bucket.blob(f"{model_path}/{model}")
    model_blob.upload_from_filename(model)
    logging.info(f'Saved model to {bucket_name}/{model_path}')
    scaler_blob = bucket.blob(scaler_path)
    scaler_blob.upload_from_filename(scaler_name)
    logging.info(f'Saved data scaler as {bucket_name}/{scaler_path}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment',
                        help='Propensity segment to train.',
                        required=True)

    args = parser.parse_args()
    arguments = args.__dict__
    estimator, eval_score, scaler = train_and_evaluate(
        arguments['segment'], table_name)
    save_model(arguments['segment'], estimator, scaler, gcs_path,
               bucket_name)