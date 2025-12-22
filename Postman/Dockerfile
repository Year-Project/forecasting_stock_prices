FROM python:3.12-slim

RUN useradd -m postman_admin -s /bin/bash

WORKDIR /home/postman_admin/postman-sevice

COPY requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt

COPY /entrypoint.sh .
COPY /src ./src
COPY alembic.ini .
COPY /migrations ./migrations

RUN chmod +x entrypoint.sh

RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

USER postman_admin

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]