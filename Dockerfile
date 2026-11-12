FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    cmake \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check
RUN AutoROM --accept-license
COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "16", "--access-logfile", "-", "--error-logfile", "-", "wsgi:application"]
