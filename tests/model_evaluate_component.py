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

from typing import NamedTuple

@component(
    # base_image="gcr.io/dw-analytics-d01/propimage:0.1-pipe",
    packages_to_install=["pandas==1.1.5", "sklearn==0.24.2", "xgboost==1.5.2",
                         "google-cloud-storage", "google-cloud-bigquery",
                         "google-cloud-bigquery-storage"]
)
def evaluate_model(
    dataset: Input[Artifact],
    segment: int,
    threshold: float,
    # model: Input[Model],
    gcs_artifact_uri: str,
    metrics: Output[Metrics],
    class_metrics: Output[ClassificationMetrics],
) -> NamedTuple("output", [("passed", str)]):
    
    from proptrainer import features
    from proptrainer.model import preprocess
    from google.cloud import bigquery
    from google.cloud import storage
    from google.cloud import aiplatform
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    from sklearn.utils import compute_class_weight
    import joblib
    import pickle
    # import typing
    
    def threshold_check(value, threshold):
        condition = "false"
        if value > threshold:
            condition = "true"

        return condition
    
    class CprPredictor(object):
        def __init__(self):
            return

        def load(self, gcs_artifacts_uri: str):
            gcs_client = storage.Client()
            with open('model.joblib', 'wb') as gcs_model, open('scaler.pkl', 'wb') as gcs_scaler:
                gcs_client.download_blob_to_file(
                    f"{gcs_artifacts_uri}/model.joblib", gcs_model)
                gcs_client.download_blob_to_file(f"{gcs_artifacts_uri}/scaler.pkl",
                                                 gcs_scaler)
            with open('scaler.pkl', 'rb') as scal:
                scaler = pickle.load(scal)

            self._model = joblib.load("model.joblib")
            self._scaler = scaler

        def predict(self, instances):
            scaled_inputs = self._scaler.transform(instances)
            predictions = self._model.predict(scaled_inputs)
            return predictions
        
        def predict_proba(self, instances):
            scaled_inputs = self._scaler.transform(instances)
            probabilities = self._model.predict_proba(scaled_inputs)[:, 1]
            return probabilities
            
        
    dataset = aiplatform.TabularDataset('projects/' + 
                                        dataset.uri.split('projects/')[-1])
    table_id = dataset._gca_resource.metadata.get("inputConfig").get(
        "bigquerySource").get("uri").split('//')[-1]
    
    bqclient = bigquery.Client()
    bqstorageclient = bigquery_storage.BigQueryReadClient()
        
    query = f"""
    SELECT
    *
    FROM
    (SELECT
    * 
    FROM {table_id}
    WHERE em_segment = {segment}
    AND IN_HOME_DT >= DATE'2021-06-01')
    WHERE MOD(ABS(FARM_FINGERPRINT(CASE(COUPON_BARCODE AS STRING))), 100) < 2
    """
    
    eval_d = bqclient.query(query).result().to_dataframe(
        bqstorage_client=bqstorageclient)
    eval_d = preprocess(eval_d)
    
    columns = features.feature_lookup[str(segment)]
    inputs = eval_d[columns]
    target = eval_d["TARGET_14"]
        
    predictor = CprPredictor()
    predictor.load(gcs_artifact_uri)
    probabilities = predictor.predict_proba(inputs)
    
    # evaluate predictions
    prior = round(target.value_counts()[1] / len(target), 5)
    classes = np.unique(target)
    class_weights = dict(
        zip(
            classes,
            compute_class_weight(class_weight="balanced",
                                 classes=classes,
                                 y=target)))
    cm = confusion_matrix(target,
                          probabilities > prior,
                          labels=classes,
                          sample_weight=class_weights)
    categories = ["0", "1"]
    class_metrics.log_confusion_matrix(categories=categories, matrix=cm.tolist())
    
    fpr, tpr, thresholds = roc_curve(target, probabilities)
    class_metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())
    
    test_auc = roc_auc_score(target, probabilities)
    metrics.log_metric("auc", test_auc)
    passed = threshold_check(test_auc, threshold)
    
    return (passed,) # this would ideally be the precursor to model upload but CustomContainerTraining uploads the model already...