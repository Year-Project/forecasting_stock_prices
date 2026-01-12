FROM python:3.12-slim

RUN useradd -m magician_admin -s /bin/bash

WORKDIR /home/magician_admin/app

COPY magician/requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt

COPY db ./db
COPY dependencies ./dependencies
COPY exceptions ./exceptions
COPY kafka ./kafka
COPY logger ./logger
COPY schemas ./schemas
COPY utils ./utils


COPY magician ./magician

RUN mkdir -p ./magician/logs && chown -R magician_admin:magician_admin ./magician/logs

RUN chmod +x ./magician/entrypoint.sh

RUN apt-get update && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/home/magician_admin/app:$PYTHONPATH

USER magician_admin

WORKDIR /home/magician_admin/app

EXPOSE 8000

ENTRYPOINT ["./magician/entrypoint.sh"]
