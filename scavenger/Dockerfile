FROM python:3.12-slim

RUN useradd -m appuser -s /bin/bash

WORKDIR /home/appuser/app
COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ./logs && chown -R appuser:appuser ./logs
RUN chmod +x ./entrypoint.sh
RUN apt-get update && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/home/appuser/app:$PYTHONPATH

USER appuser

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
