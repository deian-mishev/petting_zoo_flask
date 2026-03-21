FROM python:3.9-slim-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    cmake \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]
