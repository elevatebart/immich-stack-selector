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
# CPU-only torch keeps the image a few GB smaller than the default CUDA wheels.
# torchvision must come from the same index and version pair, otherwise open_clip
# loads a PyPI torchvision built against another torch ("torchvision::nms does not exist").
RUN pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
COPY backend/ backend/
RUN pip install "./backend[aesthetic]" && python -c "import torchvision, open_clip; torchvision.ops.nms"
COPY --from=ui /ui/dist frontend/dist
COPY entrypoint.sh /entrypoint.sh
WORKDIR /app/backend
VOLUME /data
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
