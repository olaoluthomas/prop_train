---
# ML Training Pipeline for Main Event Propensity Models


Deploying DM propensity model training and serving in Vertex AI.

---

This is a barebones solution to run model training via Python command.

The Python module syntax is:\
\
`python -m proptrainer.train --segment=<SEGMENT> --table_name=<TABLENAME> --timestamp=<TIMESTAMP>`\
\
where
- "SEGMENT" is the propensity segment to be trained
- "TABLENAME" holds the training data.
- "TIMESTAMP" is an identifier of training job execution time (used as part of GCS URI/folder path to save and retrieve model artifacts

Suggested timestamp:\
`import datetime`\
`timestamp = datetime.datetime.now().strftime('export_%Y%m%d_%H%M%S')`

Hyperparameter tuning enabled by passing the "--hypertune" argument

`python -m proptrainer.train --segment=<SEGMENT> --table_name=<TABLENAME> --timestamp=<TIMESTAMP> --hypertune`
