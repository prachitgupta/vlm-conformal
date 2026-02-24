#!/usr/bin/env bash
set -euo pipefail

PX4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORLD_NAME="${WORLD_NAME:-obstacle_avoidance}"
MODEL_INSTANCE="${MODEL_INSTANCE:-x500_depth_0}"

# Fixed startup buffer before launching full bridges.
BRIDGE_STARTUP_BUFFER_SEC="${BRIDGE_STARTUP_BUFFER_SEC:-12}"
DELAY_BEFORE_AGENT="${DELAY_BEFORE_AGENT:-2}"
DELAY_BEFORE_RQT="${DELAY_BEFORE_RQT:-2}"
GAZEBO_CONTROL_SERVICE="${GAZEBO_CONTROL_SERVICE:-/world/${WORLD_NAME}/control}"

add_bridge_arg() {
  local arg="$1"
  local existing
  for existing in "${BRIDGE_ARGS[@]:-}"; do
    [[ "${existing}" == "${arg}" ]] && return 0
  done
  BRIDGE_ARGS+=("${arg}")
}

discover_sensor_bridge_args() {
  local topic_list
  local rgb_image_topic=""
  local rgb_info_topic=""
  local exact_rgb_prefix="/world/${WORLD_NAME}/model/${MODEL_INSTANCE}/link/camera_link/sensor/IMX214"

  if ! command -v gz >/dev/null 2>&1; then
    echo "Warning: gz CLI not found, skipping topic discovery and using static bridge args."
    return 0
  fi

  topic_list="$(gz topic -l 2>/dev/null || true)"
  if [[ -z "${topic_list}" ]]; then
    echo "Warning: unable to query Gazebo topics yet; using static bridge args."
    return 0
  fi

  if grep -qx "${exact_rgb_prefix}/image" <<< "${topic_list}"; then
    rgb_image_topic="${exact_rgb_prefix}/image"
    rgb_info_topic="${exact_rgb_prefix}/camera_info"
  else
    rgb_image_topic="$(grep -E "^/world/${WORLD_NAME}/model/[^/]+/link/camera_link/sensor/IMX214/image$" <<< "${topic_list}" | head -n1 || true)"
    if [[ -n "${rgb_image_topic}" ]]; then
      rgb_info_topic="${rgb_image_topic%/image}/camera_info"
      echo "Detected camera topic on model instance: ${rgb_image_topic}"
    fi
  fi

  if [[ -n "${rgb_image_topic}" ]]; then
    add_bridge_arg "${rgb_image_topic}@sensor_msgs/msg/Image[gz.msgs.Image"
  fi

  if [[ -n "${rgb_info_topic}" ]] && grep -qx "${rgb_info_topic}" <<< "${topic_list}"; then
    add_bridge_arg "${rgb_info_topic}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
  fi

  # Some Gazebo versions scope depth camera topics under the sensor path instead of
  # honoring the unscoped <topic>depth_camera</topic> name. Bridge both if present.
  local scoped_depth_image
  local scoped_depth_points
  scoped_depth_image="$(grep -E "^/world/${WORLD_NAME}/model/[^/]+/link/camera_link/sensor/StereoOV7251(/.*)?$" <<< "${topic_list}" | grep -E "(/depth_image|/image)$" | head -n1 || true)"
  scoped_depth_points="$(grep -E "^/world/${WORLD_NAME}/model/[^/]+/link/camera_link/sensor/StereoOV7251(/.*)?/points$" <<< "${topic_list}" | head -n1 || true)"

  if [[ -n "${scoped_depth_image}" ]]; then
    echo "Detected scoped depth image topic: ${scoped_depth_image}"
    add_bridge_arg "${scoped_depth_image}@sensor_msgs/msg/Image[gz.msgs.Image"
  fi

  if [[ -n "${scoped_depth_points}" ]]; then
    echo "Detected scoped point cloud topic: ${scoped_depth_points}"
    add_bridge_arg "${scoped_depth_points}@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
  fi
}

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

echo "Opening Terminal 1: MicroXRCEAgent"
open_term "MicroXRCEAgent" "echo ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0}; MicroXRCEAgent udp4 -p 8888"

sleep "${DELAY_BEFORE_AGENT}"

echo "Opening Terminal 2: PX4 + Gazebo"
open_term "PX4 + Gazebo" "echo ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0}; cd '${PX4_DIR}' && ./Tools/simulation/gz/launch_obstacle_avoidance_x500.sh"

echo "Waiting ${BRIDGE_STARTUP_BUFFER_SEC}s before launching ROS-GZ bridge..."
sleep "${BRIDGE_STARTUP_BUFFER_SEC}"
discover_sensor_bridge_args

echo "ROS-GZ bridge topics:"
printf '  %s\n' "${BRIDGE_ARGS[@]}"

BRIDGE_CMD=$(cat <<EOF
source /opt/ros/humble/setup.bash
mkdir -p /tmp/roslog
export ROS_LOG_DIR=/tmp/roslog
ros2 run ros_gz_bridge parameter_bridge $(printf '%q ' "${BRIDGE_ARGS[@]}")
EOF
)

echo "Opening Terminal 3: ROS-GZ bridge"
open_term "ROS-GZ Bridge" "${BRIDGE_CMD}"

sleep "${DELAY_BEFORE_RQT}"
echo "Opening Terminal 4: rqt_image_view"
open_term "rqt_image_view" "source /opt/ros/humble/setup.bash && ros2 run rqt_image_view rqt_image_view"

echo "All terminals launched."
