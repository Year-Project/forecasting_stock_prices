FROM python:3.12-slim

RUN useradd -m guard_admin -s /bin/bash

WORKDIR /home/guard_admin/app

COPY guard/requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt

COPY db ./db
COPY dependencies ./dependencies
COPY exceptions ./exceptions
COPY kafka ./kafka
COPY logger ./logger
COPY schemas ./schemas
COPY utils ./utils

COPY guard ./guard

RUN mkdir -p ./guard/logs && chown -R guard_admin:guard_admin ./guard/logs

RUN chmod +x ./guard/entrypoint.sh

RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/home/guard_admin/app:$PYTHONPATH

USER guard_admin

WORKDIR /home/guard_admin/app/guard

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]

