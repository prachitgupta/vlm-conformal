#!/usr/bin/env bash
set -euo pipefail

PX4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CFG_DIR="${PX4_DIR}/Tools/simulation/gz/config"
WORLD_FILE="${PX4_DIR}/Tools/simulation/gz/worlds/obstacle_avoidance.sdf"
GUI_CFG="${CFG_DIR}/obstacle_avoidance_freelook.gui.config"
ENV_CFG="${CFG_DIR}/obstacle_avoidance_x500.env"

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
  # shellcheck disable=SC1091
  source "${PX4_DIR}/build/px4_sitl_default/rootfs/gz_env.sh"
fi

# Ensure local world/model search paths are available.
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:-}:${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds"

# shellcheck disable=SC1091
source "${ENV_CFG}"

echo "Starting Gazebo server with world: ${WORLD_FILE}"
gz sim -r -s "${WORLD_FILE}" &
GZ_SERVER_PID=$!

sleep 1

echo "Starting Gazebo GUI with config: ${GUI_CFG}"
gz sim -g --gui-config "${GUI_CFG}" >/dev/null 2>&1 &
GZ_GUI_PID=$!

sleep 1

# Best-effort: switch view mode to orbit (free-look style controls).
gz service -s /gui/camera/view_control \
  --reqtype gz.msgs.StringMsg \
  --reptype gz.msgs.Boolean \
  --timeout 1000 \
  --req 'data: "orbit"' >/dev/null 2>&1 || true

cleanup() {
  kill "${GZ_GUI_PID}" "${GZ_SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Launching PX4 SITL in standalone mode (x500 in obstacle_avoidance world)"
PX4_GZ_STANDALONE=1 make -C "${PX4_DIR}" px4_sitl gz_x500
