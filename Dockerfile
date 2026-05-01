# Python 3.11 (not 3.12) because basic-pitch 0.3.0 requires tensorflow<2.16
# on Linux + Python>=3.11, and only TF 2.16+ Linux wheels exist for Python
# 3.12.  TF 2.15.x has Python 3.11 wheels and satisfies the constraint.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade "yt-dlp[default]"

COPY app ./app
COPY deploy ./deploy
COPY entrypoint.sh ./

RUN mkdir -p /app/data/storage /app/data/tmp

EXPOSE 8080

CMD ["./entrypoint.sh"]
