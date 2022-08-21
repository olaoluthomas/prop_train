FROM python:3.7.13-slim

LABEL version="0.2.3"
LABEL image-desc="A docker image for building Propensity models via Vertex AI"
LABEL maintainer="Simeon Thomas"
LABEL org.opencontainers.image.authors="thomasolaoluwa@gmail.com"
LABEL org.opencontainers.image.source="https://github.com/olaoluthomas/vertex_prop_train"
LABEL org.opencontainers.image.description="This version expands the list of hyperparameters for SGDClassifier training of lowest segments"

# to show print statements and logs to display in Knative logs
ENV PYTHONUNBUFFERED True

# copy package installer
WORKDIR /
COPY ./src/dist/vertex_proptrainer-0.2.3-py3-none-any.whl ./

# make RUN commands use `bash --login`
SHELL ["/bin/bash", "--login", "-c"]

# install package
RUN pip install ./vertex_proptrainer-0.2.3-py3-none-any.whl

# establish entrypoint
ENTRYPOINT ["python", "-m", "vertex_proptrainer.train"]
