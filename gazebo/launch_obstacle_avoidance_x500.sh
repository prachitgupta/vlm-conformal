#!/usr/bin/env bash
set -euo pipefail

PX4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CFG_DIR="${PX4_DIR}/Tools/simulation/gz/config"
WORLD_FILE="${PX4_DIR}/Tools/simulation/gz/worlds/obstacle_avoidance.sdf"
GUI_CFG="${CFG_DIR}/obstacle_avoidance_freelook.gui.config"
ENV_CFG="${CFG_DIR}/obstacle_avoidance_x500.env"
ENABLE_GZ_VIDEO_RECORDING="${ENABLE_GZ_VIDEO_RECORDING:-1}"
GZ_VIDEO_RECORD_DIR="${GZ_VIDEO_RECORD_DIR:-${HOME}/test_run}"
GZ_VIDEO_RECORD_FORMAT="${GZ_VIDEO_RECORD_FORMAT:-mp4}"
GZ_GUI_LOG_FILE="${GZ_GUI_LOG_FILE:-/tmp/gz_obstacle_avoidance_gui.log}"
VIDEO_RECORD_SERVICE=""
VIDEO_RECORD_FILE=""

if [[ ! -f "${WORLD_FILE}" ]]; then
  echo "World file not found: ${WORLD_FILE}"
  exit 1
fi

if [[ ! -f "${GUI_CFG}" ]]; then
  echo "GUI config file not found: ${GUI_CFG}"
  exit 1
fi

if [[ ! -f "${ENV_CFG}" ]]; then
  echo "Environment config file not found: ${ENV_CFG}"
  exit 1
fi

# Load PX4's Gazebo env if available (resource/plugin paths).
if [[ -f "${PX4_DIR}/build/px4_sitl_default/rootfs/gz_env.sh" ]]; then
  # Prevent nounset failures in gz_env.sh when these vars are unset.
  export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:-}"
  export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
  # shellcheck disable=SC1091
  source "${PX4_DIR}/build/px4_sitl_default/rootfs/gz_env.sh"
fi

# Ensure local world/model search paths are available.
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:-}:${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds"

# This world embeds the required PX4 sensor/world plugins directly to avoid
# startup failures when a server config path is ignored or points elsewhere.
unset GZ_SIM_SERVER_CONFIG_PATH

# shellcheck disable=SC1091
source "${ENV_CFG}"

echo "Starting Gazebo server with world: ${WORLD_FILE}"
echo "Using embedded world plugins (GZ_SIM_SERVER_CONFIG_PATH unset)"
echo "Using GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}"
gz sim -r -s "${WORLD_FILE}" &
GZ_SERVER_PID=$!

sleep 1

echo "Starting Gazebo GUI with config: ${GUI_CFG}"
echo "Gazebo GUI log: ${GZ_GUI_LOG_FILE}"
gz sim -g --gui-config "${GUI_CFG}" >"${GZ_GUI_LOG_FILE}" 2>&1 &
GZ_GUI_PID=$!

sleep 1

start_video_recording() {
  if [[ "${ENABLE_GZ_VIDEO_RECORDING}" != "1" ]]; then
    echo "Gazebo video recording disabled (ENABLE_GZ_VIDEO_RECORDING=${ENABLE_GZ_VIDEO_RECORDING})"
    return 0
  fi

  if ! mkdir -p "${GZ_VIDEO_RECORD_DIR}"; then
    echo "Warning: unable to create video output dir: ${GZ_VIDEO_RECORD_DIR}"
    return 0
  fi

  local attempt services record_candidates
  for attempt in $(seq 1 20); do
    services="$(gz service -l 2>/dev/null || true)"
    VIDEO_RECORD_SERVICE="$(grep -E '/record_video$' <<< "${services}" | head -n1 || true)"
    [[ -n "${VIDEO_RECORD_SERVICE}" ]] && break
    sleep 0.25
  done

  if [[ -z "${VIDEO_RECORD_SERVICE}" ]]; then
    echo "Warning: video recording service not found; skipping auto-record."
    record_candidates="$(grep -E 'record|video' <<< "${services}" || true)"
    if [[ -n "${record_candidates}" ]]; then
      echo "Services containing record/video:"
      echo "${record_candidates}"
    else
      echo "No services containing record/video were discovered."
    fi
    if [[ -f "${GZ_GUI_LOG_FILE}" ]]; then
      echo "Last GUI log lines (${GZ_GUI_LOG_FILE}):"
      tail -n 30 "${GZ_GUI_LOG_FILE}" || true
    fi
    return 0
  fi

  VIDEO_RECORD_FILE="${GZ_VIDEO_RECORD_DIR}/gazebo_$(date +%Y%m%d_%H%M%S).${GZ_VIDEO_RECORD_FORMAT}"
  local req_save_filename req_filename
  req_save_filename="$(printf 'start: true\nformat: "%s"\nsave_filename: "%s"' "${GZ_VIDEO_RECORD_FORMAT}" "${VIDEO_RECORD_FILE}")"
  req_filename="$(printf 'start: true\nformat: "%s"\nfilename: "%s"' "${GZ_VIDEO_RECORD_FORMAT}" "${VIDEO_RECORD_FILE}")"

  if gz service -s "${VIDEO_RECORD_SERVICE}" \
      --reqtype gz.msgs.VideoRecord \
      --reptype gz.msgs.Boolean \
      --timeout 3000 \
      --req "${req_save_filename}" >/dev/null 2>&1; then
    echo "Gazebo video recording started: ${VIDEO_RECORD_FILE}"
  elif gz service -s "${VIDEO_RECORD_SERVICE}" \
      --reqtype gz.msgs.VideoRecord \
      --reptype gz.msgs.Boolean \
      --timeout 3000 \
      --req "${req_filename}" >/dev/null 2>&1; then
    echo "Gazebo video recording started (filename field): ${VIDEO_RECORD_FILE}"
  else
    echo "Warning: failed to start Gazebo video recording."
    echo "Service used: ${VIDEO_RECORD_SERVICE}"
    if [[ -f "${GZ_GUI_LOG_FILE}" ]]; then
      echo "Last GUI log lines (${GZ_GUI_LOG_FILE}):"
      tail -n 30 "${GZ_GUI_LOG_FILE}" || true
    fi
    VIDEO_RECORD_FILE=""
  fi
}

stop_video_recording() {
  if [[ -z "${VIDEO_RECORD_SERVICE}" ]]; then
    return 0
  fi

  gz service -s "${VIDEO_RECORD_SERVICE}" \
    --reqtype gz.msgs.VideoRecord \
    --reptype gz.msgs.Boolean \
    --timeout 2000 \
    --req 'stop: true' >/dev/null 2>&1 || true
}

# Best-effort: switch view mode to orbit (free-look style controls).
gz service -s /gui/camera/view_control \
  --reqtype gz.msgs.StringMsg \
  --reptype gz.msgs.Boolean \
  --timeout 1000 \
  --req 'data: "orbit"' >/dev/null 2>&1 || true

start_video_recording

cleanup() {
  stop_video_recording
  kill "${GZ_GUI_PID}" "${GZ_SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Launching PX4 SITL in standalone mode (${PX4_SIM_MODEL} in obstacle_avoidance world)"
echo "Using PX4_SIM_MODEL=${PX4_SIM_MODEL}"
PX4_GZ_STANDALONE=1 make -C "${PX4_DIR}" px4_sitl "${PX4_SIM_MODEL}" &
PX4_PID=$!

# If Gazebo exits first, stop PX4 so cleanup can run and finalize recording.
while kill -0 "${PX4_PID}" >/dev/null 2>&1; do
  if ! kill -0 "${GZ_SERVER_PID}" >/dev/null 2>&1 || ! kill -0 "${GZ_GUI_PID}" >/dev/null 2>&1; then
    echo "Gazebo exited; stopping PX4 SITL and cleaning up."
    kill "${PX4_PID}" >/dev/null 2>&1 || true
    break
  fi
  sleep 1
done

wait "${PX4_PID}"
