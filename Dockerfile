# ---- UI -------------------------------------------------------------------
FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ---- API + scoring ----------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    DB_PATH=/data/selector.sqlite CACHE_DIR=/data/thumbs
# CPU-only torch keeps the image a few GB smaller than the default CUDA wheels
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/ backend/
RUN pip install "./backend[aesthetic]"
COPY --from=ui /ui/dist frontend/dist
COPY entrypoint.sh /entrypoint.sh
WORKDIR /app/backend
VOLUME /data
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
