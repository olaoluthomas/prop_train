from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
VERSION = '0.3.0'

REQUIRED_PACKAGES = [
    'pandas==2.2.2', 'xgboost==2.1.1', 'numpy==2.0.2', 'scikit-learn==1.5.1',
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
    package_data={'proptrainer': ['key/datakey.json']},
    python_requires=">=3.8",
    install_requires=REQUIRED_PACKAGES,
    zip_safe=True)