"""
An argument parser to initiate Propensity model training and hyperparameter tuning.
"""

import argparse
# import hypertune
import model
import os, logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

key_path = 'datakey.json'
bucket_name = 'dev_dw_npii_adhoc'
gcs_path = 'analytics/dm_propensity/model_export'
table_name = 'dm_pc_tiny_data'
model.cat_features = ['is_in_trade_area', 'A_A9350N_ECONOMIC_STB_01_10']

COLUMN_NAMES_104 = [
    'BBB_R_2Y', 'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BUYS_Q_08',
    'NUM_TXNS', 'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D', 'TOTAL_TXNS_L12M',
    'Z_AVG_PCT_SALES_RETURNED', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10'
]

COLUMN_NAMES_106 = [
    'BBB_F_DECILE_2Y', 'BUYS_M_01', 'BUYS_Q_08', 'COUPON_SALES_Q_05',
    'NUM_TXNS', 'PH_CSTACK_90D', 'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D',
    'PH_PFREQ365D', 'Z_AVG_PCT_SALES_RETURNED', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10'
]

COLUMN_NAMES_108 = [
    'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BUYS_Q_01', 'BUYS_Q_02',
    'BUYS_Q_04', 'BUYS_Q_08', 'NUM_PERIODS', 'PH_MREDEEM730D_PERC',
    'PH_NMFREQ182D', 'PH_PFREQ365D', 'RECENCY', 'TOTAL_TXNS_L12M',
    'Z_AVG_PCT_SALES_RETURNED', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10'
]

COLUMN_NAMES_109 = [
    'BBB_R_2Y', 'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BUYS_M_24',
    'COUPON_SALES_Q_08', 'NUM_PERIODS', 'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D',
    'TOTAL_TXNS_L12M', 'Z_ONCOUPON_F_DEC_4_2Y', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10'
]

COLUMN_NAMES_110 = [
    'A_A8642_HM_MKT_VAL', 'BUYS_Q_01', 'BUYS_Q_04', 'BUYS_Q_08', 'NUM_PERIODS',
    'PH_MREDEEM730D_PERC', 'PH_PFREQ365D', 'TOTAL_TXNS_L12M',
    'is_in_trade_area', 'A_A9350N_ECONOMIC_STB_01_10'
]

COLUMN_NAMES_112 = [
    'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BBB_INSTORE_F_2Y', 'BUYS_Q_01',
    'BUYS_Q_02', 'BUYS_M_01', 'BUYS_Q_04', 'BUYS_Q_08', 'NUM_PERIODS',
    'PH_CREDEEM365D', 'PH_MREDEEM182D_PERC', 'PH_MREDEEM730D_PERC',
    'PH_NMFREQ182D', 'PH_PFREQ365D', 'PH_PREDEEM548D', 'RECENCY',
    'Z_AVG_PCT_SALES_RETURNED', 'Z_MARKDOWNS', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10'
]

COLUMN_NAMES_114 = [
    'BBB_R_2Y', 'A_A8642_HM_MKT_VAL', 'BBB_INSTORE_F_2Y', 'BUYS_Q_01',
    'BUYS_Q_02', 'BUYS_M_01', 'BUYS_Q_04', 'BUYS_Q_08', 'COUPON_ANY_AMT',
    'NUM_PERIODS', 'NUM_TOTAL_ITEMS', 'PH_MREDEEM182D_PERC',
    'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D', 'Z_AVG_PCT_SALES_RETURNED',
    'is_in_trade_area', 'A_A9350N_ECONOMIC_STB_01_10'
]

COLUMN_NAMES_115 = [
    'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BBB_ONCOUPON_R_DECILE_2Y',
    'BUYS_Q_01', 'BUYS_Q_02', 'BUYS_M_01', 'BUYS_Q_04', 'COUPON_Q_01',
    'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D', 'PH_PFREQ365D', 'PH_PREDEEM365D',
    'PH_PREDEEM730D', 'TOTAL_TXNS_L12M', 'Z_AVG_PCT_SALES_RETURNED',
    'Z_INSTORE_M_DEC_10_2Y', 'Z_ONCOUPON_F_DEC_4_2Y', 'is_in_trade_area'
]

COLUMN_NAMES_116 = [
    'A_A8642_HM_MKT_VAL', 'BBB_F_DECILE_2Y', 'BBB_ONCOUPON_R_DECILE_2Y',
    'BUYS_Q_02', 'BUYS_M_01', 'BUYS_Q_04', 'BUYS_Q_08', 'COUPON_Q_01',
    'NUM_TOTAL_ITEMS', 'PH_MREDEEM182D_PERC', 'PH_MREDEEM730D_PERC',
    'PH_NMFREQ182D', 'PH_PFREQ365D', 'RECENCY', 'TOTAL_TXNS_L12M',
    'Z_AVG_PCT_SALES_RETURNED', 'Z_AVG_RECENCY', 'Z_INSTORE_M_DEC_10_2Y',
    'Z_ONCOUPON_F_DEC_4_2Y', 'Z_MARKDOWNS', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10'
]

COLUMN_NAMES_P = [
    'BUYS_Q_01', 'BUYS_M_06', 'BUYS_M_24', 'BUYS_Q_04', 'COUPON_SALES_Q_05',
    'NUM_TOTAL_ITEMS', 'PH_MREDEEM730D_PERC', 'PH_NMFREQ182D',
    'TOTAL_TXNS_L12M', 'Z_INSTORE_M_DEC_10_2Y', 'is_in_trade_area',
    'A_A9350N_ECONOMIC_STB_01_10'
]

model.feature_lookup = {
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment',
                        help='Propensity segment to train.',
                        required=True)

    args = parser.parse_args()
    arguments = args.__dict__
    estimator, eval_score, scaler = model.train_and_evaluate(
        arguments['segment'], table_name, key_path)
    model.save_model(arguments['segment'], estimator, scaler, key_path,
                     gcs_path, bucket_name)