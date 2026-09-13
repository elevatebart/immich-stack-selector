#!/bin/sh
# serve (default) runs the API + UI; build / sync run one-shot and exit
set -e
case "${1:-serve}" in
  serve) exec uvicorn app.main:app --host 0.0.0.0 --port 8000 ;;
  build) shift; exec python -m app.build "$@" ;;
  sync)  shift; exec python -m app.sync "$@" ;;
  train) shift; exec python -m app.train "$@" ;;
  *)     exec "$@" ;;
esac
