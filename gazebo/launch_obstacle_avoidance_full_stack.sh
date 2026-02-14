#!/usr/bin/env bash
set -euo pipefail

PX4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORLD_NAME="${WORLD_NAME:-obstacle_avoidance}"
MODEL_INSTANCE="${MODEL_INSTANCE:-x500_depth_0}"

# Wait settings for Gazebo readiness before starting bridge.
GAZEBO_READY_TIMEOUT="${GAZEBO_READY_TIMEOUT:-90}"
GAZEBO_READY_POLL_INTERVAL="${GAZEBO_READY_POLL_INTERVAL:-1}"
GAZEBO_ECHO_TIMEOUT="${GAZEBO_ECHO_TIMEOUT:-3}"
DELAY_BEFORE_AGENT="${DELAY_BEFORE_AGENT:-2}"
DELAY_BEFORE_RQT="${DELAY_BEFORE_RQT:-2}"
CAMERA_GZ_TOPIC="${CAMERA_GZ_TOPIC:-/world/${WORLD_NAME}/model/${MODEL_INSTANCE}/link/camera_link/sensor/IMX214/image}"
GAZEBO_CONTROL_SERVICE="${GAZEBO_CONTROL_SERVICE:-/world/${WORLD_NAME}/control}"

detect_terminal() {
  if command -v gnome-terminal >/dev/null 2>&1; then
    echo "gnome-terminal"
    return
  fi
  if command -v konsole >/dev/null 2>&1; then
    echo "konsole"
    return
  fi
  if command -v xterm >/dev/null 2>&1; then
    echo "xterm"
    return
  fi
  return 1
}

open_term() {
  local title="$1"
  local cmd="$2"
  local tail_cmd='status=$?; echo; echo "[process exited: ${status}]"; read -r -p "Press Enter to close..." _'

  case "${TERM_APP}" in
    gnome-terminal)
      gnome-terminal --title="${title}" -- bash -lc "${cmd}; ${tail_cmd}" &
      ;;
    konsole)
      konsole --new-tab -p tabtitle="${title}" -e bash -lc "${cmd}; ${tail_cmd}" &
      ;;
    xterm)
      xterm -T "${title}" -hold -e bash -lc "${cmd}" &
      ;;
    *)
      echo "Unsupported terminal app: ${TERM_APP}"
      exit 1
      ;;
  esac
}

wait_for_gazebo_data() {
  local deadline=$((SECONDS + GAZEBO_READY_TIMEOUT))

  echo "Waiting for Gazebo server and camera data..."
  while (( SECONDS < deadline )); do
    if ! gz service -l 2>/dev/null | grep -Fxq "${GAZEBO_CONTROL_SERVICE}"; then
      sleep "${GAZEBO_READY_POLL_INTERVAL}"
      continue
    fi

    if ! gz topic -l 2>/dev/null | grep -Fxq "${CAMERA_GZ_TOPIC}"; then
      sleep "${GAZEBO_READY_POLL_INTERVAL}"
      continue
    fi

    if timeout "${GAZEBO_ECHO_TIMEOUT}" gz topic -e -n 1 -t "${CAMERA_GZ_TOPIC}" >/dev/null 2>&1; then
      echo "Gazebo is active and publishing on ${CAMERA_GZ_TOPIC}."
      return 0
    fi

    sleep "${GAZEBO_READY_POLL_INTERVAL}"
  done

  echo "Timed out after ${GAZEBO_READY_TIMEOUT}s waiting for Gazebo readiness and data."
  return 1
}

TERM_APP="$(detect_terminal || true)"
if [[ -z "${TERM_APP}" ]]; then
  echo "No supported terminal found. Install one of: gnome-terminal, konsole, xterm."
  exit 1
fi

BRIDGE_ARGS=(
  "/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image"
  "/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
  "/world/${WORLD_NAME}/model/${MODEL_INSTANCE}/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image"
  "/world/${WORLD_NAME}/model/${MODEL_INSTANCE}/link/camera_link/sensor/IMX214/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
  "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
  "${GAZEBO_CONTROL_SERVICE}@ros_gz_interfaces/srv/ControlWorld@gz.msgs.WorldControl@gz.msgs.Boolean"
)

BRIDGE_CMD=$(cat <<EOF
source /opt/ros/humble/setup.bash
mkdir -p /tmp/roslog
export ROS_LOG_DIR=/tmp/roslog
ros2 run ros_gz_bridge parameter_bridge $(printf '%q ' "${BRIDGE_ARGS[@]}")
EOF
)

echo "Opening Terminal 1: PX4 + Gazebo"
open_term "PX4 + Gazebo" "cd '${PX4_DIR}' && ./Tools/simulation/gz/launch_obstacle_avoidance_x500.sh"

# Keep waiting in this terminal; do not launch the bridge terminal until data is flowing.
wait_for_gazebo_data
echo "Opening Terminal 2: ROS-GZ bridge"
open_term "ROS-GZ Bridge" "${BRIDGE_CMD}"

sleep "${DELAY_BEFORE_AGENT}"
echo "Opening Terminal 3: MicroXRCEAgent"
open_term "MicroXRCEAgent" "MicroXRCEAgent udp4 -p 8888"

sleep "${DELAY_BEFORE_RQT}"
echo "Opening Terminal 4: rqt_image_view"
open_term "rqt_image_view" "source /opt/ros/humble/setup.bash && ros2 run rqt_image_view rqt_image_view"

echo "All terminals launched."
