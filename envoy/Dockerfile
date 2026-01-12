FROM python:3.12-slim

RUN useradd -m envoy_admin -s /bin/bash

WORKDIR /home/envoy_admin/app

COPY envoy/requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt

COPY dependencies ./dependencies
COPY logger ./logger
COPY schemas ./schemas
COPY utils ./utils


COPY envoy ./envoy

RUN mkdir -p ./envoy/logs && chown -R envoy_admin:envoy_admin ./envoy/logs

RUN apt-get update && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/home/envoy_admin/app:$PYTHONPATH

USER envoy_admin

WORKDIR /home/envoy_admin/app

CMD ["python", "-m", "envoy.bot"]