"""
A utility to build Google Cloud service clients.
"""

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


def get_project_id():
    """
    Function to get project ID from service account credentials.
    TODO* Ideally should be passed as an environment variable.

    Returns
    -------
    PROJECT_ID: str
    """
    return get_credentials().project_id


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


def get_storage_client():
    """
    Function to instantiate Cloud storage client.

    Returns
    -------
    storage.Client
    """
    from google.cloud import storage
    
    # you should not explitly declare project ID
    # project = os.environ["CLOUD_ML_PROJECT_ID"] not working locally
    return storage.Client(project='dw-analytics-d01')