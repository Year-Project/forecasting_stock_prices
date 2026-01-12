FROM python:3.12-slim

RUN useradd -m postman_admin -s /bin/bash

WORKDIR /home/postman_admin/app

COPY postman/requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt

COPY db ./db
COPY dependencies ./dependencies
COPY exceptions ./exceptions
COPY kafka ./kafka
COPY logger ./logger
COPY schemas ./schemas
COPY utils ./utils


COPY postman ./postman

RUN mkdir -p ./postman/logs && chown -R postman_admin:postman_admin ./postman/logs

RUN chmod +x ./postman/entrypoint.sh

RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/home/postman_admin/app:$PYTHONPATH

USER postman_admin

WORKDIR /home/postman_admin/app

EXPOSE 8000

ENTRYPOINT ["./postman/entrypoint.sh"]