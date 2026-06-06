## slancha-mesh registry image — Apache 2.0
## The mesh control plane: `uvicorn mesh.registry_app:app` on :8088.
## In-memory + event-sourced (rebuilds from heartbeats in ~5s); no persistence.
##
## Build: docker build -t slancha-registry -f docker/Dockerfile .
## Run:   docker run --rm -p 8088:8088 slancha-registry
##        (set -e SLANCHA_NODE_TOKEN=... to require Bearer auth on all but /health)

# Base image digest-pinned (#129) so a silently-republished `python:3.12-slim` tag
# can't change the build. Bump deliberately: docker pull python:3.12-slim &&
# docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim
FROM python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203 AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential covers source builds for psutil / pydantic-core on arches
# without a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md /build/
COPY mesh /build/mesh

RUN pip install --upgrade pip && pip install hatchling
RUN python -m hatchling build && ls -la dist/

# ---

FROM python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203 AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MESH_REGISTRY_HOST=0.0.0.0 \
    MESH_REGISTRY_PORT=8088

COPY --from=build /build/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

EXPOSE 8088

# /health needs no auth — safe as a liveness probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,httpx; httpx.get(f\"http://127.0.0.1:{os.environ['MESH_REGISTRY_PORT']}/health\", timeout=2.0).raise_for_status()" || exit 1

# exec-form sh -c: env vars expand AND `exec` hands PID 1 to uvicorn so it
# receives SIGTERM directly for graceful shutdown (uvicorn drains in-flight).
CMD ["sh", "-c", "exec uvicorn mesh.registry_app:app --host ${MESH_REGISTRY_HOST} --port ${MESH_REGISTRY_PORT} --log-level info --no-access-log"]
