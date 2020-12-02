ARG PYTHON_TAG=3.9.0-slim
FROM python:${PYTHON_TAG}
ENV PYTHONUNBUFFERED=1

ARG TINI_VERSION=0.19.0
ADD https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini /tini
RUN chmod 755 /tini
# Python exits with code 143 for SIGTERM; remap it to 0
ENTRYPOINT ["/tini", "-e", "143", "--"]

ARG USER=training
RUN useradd -m ${USER}
WORKDIR /home/${USER}
COPY --chown=${USER}:${USER} requirements.txt ./
RUN SCIKIT_LEARN_BDEPS="g++ python3-dev" && \
    SCIKIT_LEARN_RDEPS="libgomp1" && \
    apt-get update && \
    apt-get install --assume-yes --no-install-recommends ${SCIKIT_LEARN_BDEPS} ${SCIKIT_LEARN_RDEPS} && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get autoremove --purge -y ${SCIKIT_LEARN_BDEPS} && \
    rm -rf /var/lib/apt/lists/*
USER ${USER}
COPY --chown=${USER}:${USER} ./ ./
CMD ["python", "main.py"]
