FROM python:3.7.13-slim

LABEL version="v0.1-dev1"
LABEL maintainer="Simeon Thomas"
LABEL org.opencontainers.image.source https://github.com/olaoluthomas/prop_train

# to show print statements and logs to display in Knative logs
ENV PYTHONUNBUFFERED True

# copy package installer
WORKDIR /
COPY ./src/dist/prop_trainer-0.1.dev2-py3-none-any.whl ./

# make RUN commands use `bash --login`
SHELL ["/bin/bash", "--login", "-c"]

# install package
# RUN pip install --upgrade pip
RUN pip install ./prop_trainer-0.1.dev2-py3-none-any.whl

# execute training script
ENTRYPOINT ["python", "-m", "proptrainer.train"]