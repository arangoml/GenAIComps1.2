# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0

FROM python:3.11-slim

# Set environment variables
ENV LANG=en_US.UTF-8

COPY comps /home/comps

# Install system dependencies
RUN apt-get update \
    && apt-get install -y curl

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r /home/comps/lvms/src/integrations/dependency/predictionguard/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home

WORKDIR /home/comps/lvms/src/integrations/dependency/predictionguard

ENTRYPOINT ["python", "lvm.py"]
