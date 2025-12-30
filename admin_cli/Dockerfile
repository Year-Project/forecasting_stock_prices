FROM python:3.12-slim

WORKDIR /app

COPY  admin_cli/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ./logs

COPY logger ./logger
COPY utils/config_utils.py ./utils/config_utils.py

COPY admin_cli/main.py ./main.py

ENTRYPOINT ["python", "main.py"]
