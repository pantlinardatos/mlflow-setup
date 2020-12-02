#!/bin/sh

exec mlflow server \
    --backend-store-uri "${DATABASE_URL}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --host "${MLFLOW_HOST}" \
    --port "${MLFLOW_PORT}" \
    "$@"
