
FROM python:3.9 AS base

WORKDIR /app

COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


WORKDIR /app

COPY asr/asr_api.py .

EXPOSE 8001

CMD ["python", "asr_api.py"]
