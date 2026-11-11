FROM python:3.11-slim
ARG REF=main
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /tmp/build /app \
    && curl -fsSL "https://github.com/alex-rs/solar-cast/archive/${REF}.tar.gz" -o /tmp/src.tar.gz \
    && tar -xzf /tmp/src.tar.gz -C /tmp/build \
    && cp -r /tmp/build/*/backend/app /app/app \
    && cp -r /tmp/build/*/backend/config /app/config \
    && pip install --no-cache-dir -r /tmp/build/*/backend/requirements.lock \
    && rm -rf /tmp/build /tmp/src.tar.gz
COPY run.sh /run.sh
RUN chmod +x /run.sh
WORKDIR /app
ENV DATA_DIR=/config \
    SYSTEMS_FILE=/config/systems.yaml
EXPOSE 8000
CMD ["/run.sh"]
