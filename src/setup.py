from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
VERSION = '0.2.2'

REQUIRED_PACKAGES = [
    'pandas==1.1.5', 'xgboost==1.5.2', 'numpy>=1.21', 'scikit-learn==0.24.2',
    'pandas-gbq==0.14.1', 'google-cloud-bigquery==2.34.2',
    'google-cloud-bigquery-storage==2.13.0', 'google-cloud-storage==1.44.0',
    'importlib-resources==5.4.0', 'cloudml-hypertune==0.1.0.dev6',
    'google-cloud-aiplatform==1.12.1',
]

setup(
    name='vertex_proptrainer',
    version=VERSION,
    author="Simeon Thomas",
    author_email="thomasolaoluwa@gmail.com",
    description="A Python trainer package for Propensity modeling in Vertex AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/olaoluthomas/vertex_prop_train",
    project_urls={
        "Bug Tracker": "https://github.com/olaoluthomas/vertex_prop_train/issues",
    },
    license="Apache2",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={'vertex_proptrainer': ['key/datakey.json']},
    python_requires=">=3.6",
    install_requires=REQUIRED_PACKAGES,
    zip_safe=True)
