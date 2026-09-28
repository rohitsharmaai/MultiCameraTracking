#!/usr/bin/env bash
# Multi-Camera Person ReID - one-command launcher.
#
#   ./start.sh            # mount check + start PostgreSQL + start the web app
#   ./start.sh status     # show containers + app health
#   ./start.sh down       # stop the app and PostgreSQL
#   ./start.sh mount      # mount the NTFS "Data" drive (asks for sudo)
#   ./start.sh db         # start only PostgreSQL
#   ./start.sh app        # start only the web app
#   ./start.sh logs       # follow PostgreSQL logs
#   ./start.sh install    # refresh python deps + download models
#
# Configurable via .env or the environment:
#   CONDA_ENV (default multicam-reid), API_PORT (8000),
#   DATA_MOUNT (/media/parmpal/Data), DATA_DEV (/dev/nvme0n1p5)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

CONDA_ENV="${CONDA_ENV:-multicam-reid}"
API_PORT="${API_PORT:-8000}"
DATA_MOUNT="${DATA_MOUNT:-/media/parmpal/Data}"
DATA_DEV="${DATA_DEV:-/dev/nvme0n1p5}"
PG_CONTAINER="${PG_CONTAINER:-mct-postgres}"

c_info(){ printf '\033[1;36m[start]\033[0m %s\n' "$*"; }
c_err(){ printf '\033[1;31m[start]\033[0m %s\n' "$*" >&2; }
require(){ command -v "$1" >/dev/null 2>&1 || { c_err "required command not found: $1"; exit 1; }; }

resolve_python(){
  local p="$HOME/anaconda3/envs/$CONDA_ENV/bin/python"
  if [ -x "$p" ]; then echo "$p"; return; fi
  if command -v conda >/dev/null 2>&1; then
    p="$(conda run -n "$CONDA_ENV" python -c 'import sys;print(sys.executable)' 2>/dev/null || true)"
    if [ -n "$p" ]; then echo "$p"; return; fi
  fi
  command -v python3
}

ensure_mount(){
  if mountpoint -q "$DATA_MOUNT" 2>/dev/null; then return 0; fi
  c_err "The Data drive is NOT mounted at $DATA_MOUNT."
  c_err "Fix it with:   ./start.sh mount"
  return 1
}

cmd_mount(){
  require sudo
  c_info "mounting $DATA_DEV at $DATA_MOUNT (may ask for your password)"
  sudo mkdir -p "$DATA_MOUNT"
  sudo mount -t ntfs-3g -o "uid=$(id -u),gid=$(id -g)" "$DATA_DEV" "$DATA_MOUNT" \
    && c_info "mounted OK" \
    || c_err "mount failed (NTFS may be dirty -> disable Windows Fast Startup)"
}

start_db(){
  require docker
  c_info "starting PostgreSQL (docker compose) ..."
  docker compose up -d postgres
  local i status
  for i in $(seq 1 60); do
    status="$(docker inspect -f '{{.State.Health.Status}}' "$PG_CONTAINER" 2>/dev/null || echo missing)"
    if [ "$status" = "healthy" ]; then c_info "PostgreSQL is healthy"; return 0; fi
    sleep 2
  done
  c_err "PostgreSQL did not become healthy"
  docker logs --tail 40 "$PG_CONTAINER" || true
  return 1
}

start_app(){
  local py; py="$(resolve_python)"
  c_info "starting web app on http://127.0.0.1:$API_PORT  (python: $py)"
  c_info "press Ctrl+C to stop"
  exec "$py" -m app.web
}

cmd_up(){
  ensure_mount || exit 1
  start_db
  start_app
}

cmd_down(){
  require docker
  c_info "stopping app + PostgreSQL ..."
  pkill -f "python -m app.web" 2>/dev/null || true
  docker compose down
}

cmd_status(){
  echo "== containers =="
  docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null || true
  echo
  echo "== app =="
  curl -s --max-time 5 "http://127.0.0.1:$API_PORT/health" && echo || echo "(app not responding on :$API_PORT)"
}

cmd_install(){
  ensure_mount || exit 1
  local py; py="$(resolve_python)"
  c_info "installing python dependencies into $CONDA_ENV"
  "$py" -m pip install -U pip
  "$py" -m pip install -e ".[dev]"
  c_info "downloading model weights (skips ones already present)"
  "$py" scripts/download_models.py || c_err "model download step failed"
}

case "${1:-up}" in
  up|start|"")  cmd_up ;;
  mount)        cmd_mount ;;
  db)           ensure_mount && start_db ;;
  app)          ensure_mount && start_app ;;
  down|stop)    cmd_down ;;
  status)       cmd_status ;;
  logs)         docker compose logs -f postgres ;;
  install)      cmd_install ;;
  *) echo "usage: $0 [up|down|status|mount|db|app|logs|install]"; exit 2 ;;
esac
