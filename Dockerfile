# Pin to bookworm (Debian 12 stable) to avoid trixie proxy issues
FROM python:3.12-slim-bookworm

WORKDIR /app

# Force bookworm sources before any apt calls
RUN rm -f /etc/apt/sources.list.d/* && \
    echo "deb http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main" >> /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nodejs \
    npm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod -R 555 /app/model && \
    chmod -R 555 /app/api && \
    chmod -R 755 /app/instance && \
    chmod 700 /proc 2>/dev/null || true

USER appuser

ENV FLASK_ENV=production \
    GUNICORN_CMD_ARGS="--workers=5 --threads=2 --bind=0.0.0.0:8009 --timeout=30 --access-logfile -"

EXPOSE 8009

CMD ["gunicorn", "main:app"]
