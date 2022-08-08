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

FROM python:3.7.13-slim

LABEL version="v0.2"
LABEL image-desc="A docker image for building Propensity models via Vertex Pipelines"
LABEL maintainer="Simeon Thomas"
LABEL maintainer-email="thomasolaoluwa@gmail.com"
LABEL org.opencontainers.image.source="https://github.com/olaoluthomas/vertex_prop_train"

# to show print statements and logs to display in Knative logs
ENV PYTHONUNBUFFERED True

# copy package installer
WORKDIR /
COPY ./src/dist/prop_trainer-0.2-py3-none-any.whl ./

# make RUN commands use `bash --login`
SHELL ["/bin/bash", "--login", "-c"]

# install package
# RUN pip install --upgrade pip
RUN pip install ./prop_trainer-0.2-py3-none-any.whl
RUN pip install google-cloud-aiplatform==1.12.1

# establish entrypoint
ENTRYPOINT ["python", "-m", "proptrainer.train"]
