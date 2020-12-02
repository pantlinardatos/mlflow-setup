ARG PYTHON_TAG=3.9.0-slim
FROM python:${PYTHON_TAG}
ENV PYTHONUNBUFFERED=1

ARG TINI_VERSION=0.19.0
ADD https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini /tini
RUN chmod 755 /tini
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod 755 /docker-entrypoint.sh
# Python exits with code 143 for SIGTERM; remap it to 0
ENTRYPOINT ["/tini", "-e", "143", "--", "/docker-entrypoint.sh"]

ARG USER=mlflow
RUN useradd -m ${USER}
WORKDIR /home/${USER}
COPY --chown=${USER}:${USER} requirements.txt ./
RUN PSYCOPG2_BDEPS="gcc libpq-dev python3-dev" && \
    PSYCOPG2_RDEPS="libpq5" && \
    apt-get update && \
    apt-get install --assume-yes --no-install-recommends ${PSYCOPG2_BDEPS} ${PSYCOPG2_RDEPS} && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get autoremove --purge -y ${PSYCOPG2_BDEPS} && \
    rm -rf /var/lib/apt/lists/*
USER ${USER}
