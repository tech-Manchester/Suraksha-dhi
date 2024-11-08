FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV SECRET_KEY='your_secret_key'

CMD ["python", "new.py"]
