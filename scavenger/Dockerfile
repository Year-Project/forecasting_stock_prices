FROM python:3.12-slim

RUN useradd -m scavenger_admin -s /bin/bash

WORKDIR /home/scavenger_admin/app

COPY scavenger/requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt

COPY db ./db
COPY dependencies ./dependencies
COPY exceptions ./exceptions
COPY kafka ./kafka
COPY logger ./logger
COPY schemas ./schemas
COPY utils ./utils

COPY scavenger ./scavenger

RUN mkdir -p ./scavenger/logs && chown -R scavenger_admin:scavenger_admin ./scavenger/logs

RUN chmod +x ./scavenger/entrypoint.sh
RUN apt-get update && rm -rf /var/lib/apt/lists/*


ENV PYTHONPATH=/home/scavenger_admin/app:$PYTHONPATH

USER scavenger_admin

WORKDIR /home/scavenger_admin/app/scavenger

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
