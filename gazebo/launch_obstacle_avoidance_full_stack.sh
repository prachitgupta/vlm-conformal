#!/usr/bin/env bash
set -euo pipefail

PX4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORLD_NAME="${WORLD_NAME:-obstacle_avoidance}"
MODEL_INSTANCE="${MODEL_INSTANCE:-x500_depth_0}"

# Ordered startup delays and timeouts.
GZ_PUBLISH_TIMEOUT_SEC="${GZ_PUBLISH_TIMEOUT_SEC:-40}"
DELAY_BEFORE_AGENT="${DELAY_BEFORE_AGENT:-3}"
DELAY_BEFORE_RQT="${DELAY_BEFORE_RQT:-2}"
DELAY_BEFORE_QGC="${DELAY_BEFORE_QGC:-2}"
GAZEBO_CONTROL_SERVICE="${GAZEBO_CONTROL_SERVICE:-/world/${WORLD_NAME}/control}"
export ENABLE_GZ_VIDEO_RECORDING="${ENABLE_GZ_VIDEO_RECORDING:-0}"
export GZ_VIDEO_RECORD_DIR="${GZ_VIDEO_RECORD_DIR:-${HOME}/test_run}"
export GZ_VIDEO_RECORD_FORMAT="${GZ_VIDEO_RECORD_FORMAT:-mp4}"
X500_ENABLE_GZ_VIDEO_RECORDING="${X500_ENABLE_GZ_VIDEO_RECORDING:-1}"
ENABLE_FINAL_GZ_VIDEO_RECORDING="${ENABLE_FINAL_GZ_VIDEO_RECORDING:-0}"
QGC_APPIMAGE="${QGC_APPIMAGE:-${HOME}/Downloads/QGroundControl.AppImage}"

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

wait_for_gz_publishing() {
  local deadline now topics
  deadline=$((SECONDS + GZ_PUBLISH_TIMEOUT_SEC))
  while true; do
    topics="$(gz topic -l 2>/dev/null || true)"
    if grep -qE "^/world/${WORLD_NAME}/clock$|^/clock$" <<< "${topics}"; then
      echo "Gazebo topic check passed."
      return 0
    fi

    now=$SECONDS
    if (( now >= deadline )); then
      echo "Error: Gazebo topics are not being published within ${GZ_PUBLISH_TIMEOUT_SEC}s."
      return 1
    fi
    sleep 1
  done
}

resolve_qgc_appimage() {
  if [[ -f "${QGC_APPIMAGE}" ]]; then
    echo "${QGC_APPIMAGE}"
    return 0
  fi

  local detected
  detected="$(find "${HOME}/Downloads" -maxdepth 1 -type f \( -iname "QGroundControl*.AppImage" -o -iname "*qgc*.AppImage" \) | head -n1 || true)"
  if [[ -n "${detected}" ]]; then
    echo "${detected}"
    return 0
  fi

  return 1
}

start_video_recording() {
  if [[ "${ENABLE_FINAL_GZ_VIDEO_RECORDING}" != "1" ]]; then
    echo "Final video recording step disabled (ENABLE_FINAL_GZ_VIDEO_RECORDING=${ENABLE_FINAL_GZ_VIDEO_RECORDING})."
    return 0
  fi

  if ! mkdir -p "${GZ_VIDEO_RECORD_DIR}"; then
    echo "Warning: unable to create video output dir: ${GZ_VIDEO_RECORD_DIR}"
    return 0
  fi

  local services record_service req video_file
  services="$(gz service -l 2>/dev/null || true)"
  record_service="$(grep -E '^/gui/record_video$|/record_video$' <<< "${services}" | head -n1 || true)"
  if [[ -z "${record_service}" ]]; then
    echo "Warning: video recording service not found; skipping recording start."
    return 0
  fi

  video_file="${GZ_VIDEO_RECORD_DIR}/gazebo_$(date +%Y%m%d_%H%M%S).${GZ_VIDEO_RECORD_FORMAT}"
  req="$(printf 'start: true\nformat: "%s"\nsave_filename: "%s"' "${GZ_VIDEO_RECORD_FORMAT}" "${video_file}")"

  if gz service -s "${record_service}" \
      --reqtype gz.msgs.VideoRecord \
      --reptype gz.msgs.Boolean \
      --timeout 3000 \
      --req "${req}" >/dev/null 2>&1; then
    echo "Gazebo video recording started: ${video_file}"
  else
    echo "Warning: failed to start Gazebo video recording."
  fi
}

get_depth_image_topic() {
  local bridge_arg topic
  for bridge_arg in "${BRIDGE_ARGS[@]}"; do
    topic="${bridge_arg%%@*}"
    if [[ "${bridge_arg}" == *"@sensor_msgs/msg/Image["* ]] && [[ "${topic}" == *depth* ]]; then
      echo "${topic}"
      return 0
    fi
  done

  echo "/depth_camera"
  return 0
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

echo "Opening Terminal 1: PX4 + Gazebo"
echo "Passing ENABLE_GZ_VIDEO_RECORDING=${X500_ENABLE_GZ_VIDEO_RECORDING} to x500 launcher"
open_term "PX4 + Gazebo" "echo ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0}; cd '${PX4_DIR}' && ENABLE_GZ_VIDEO_RECORDING=${X500_ENABLE_GZ_VIDEO_RECORDING} GZ_VIDEO_RECORD_DIR='${GZ_VIDEO_RECORD_DIR}' GZ_VIDEO_RECORD_FORMAT='${GZ_VIDEO_RECORD_FORMAT}' ./Tools/simulation/gz/launch_obstacle_avoidance_x500.sh"

echo "Waiting for Gazebo to publish topics..."
wait_for_gz_publishing

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

echo "Opening Terminal 2: ROS-GZ bridge"
open_term "ROS-GZ Bridge" "${BRIDGE_CMD}"

sleep "${DELAY_BEFORE_AGENT}"
echo "Opening Terminal 3: MicroXRCEAgent"
open_term "MicroXRCEAgent" "echo ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0}; MicroXRCEAgent udp4 -p 8888"

sleep "${DELAY_BEFORE_RQT}"
DEPTH_IMAGE_TOPIC="$(get_depth_image_topic)"
echo "Opening Terminal 4: rqt_image_view (${DEPTH_IMAGE_TOPIC})"
open_term "rqt_image_view" "source /opt/ros/humble/setup.bash && ros2 run rqt_image_view rqt_image_view ${DEPTH_IMAGE_TOPIC}"

sleep "${DELAY_BEFORE_QGC}"
QGC_PATH="$(resolve_qgc_appimage || true)"
if [[ -n "${QGC_PATH}" ]]; then
  echo "Opening Terminal 5: QGroundControl (${QGC_PATH})"
  QGC_PATH_Q="$(printf '%q' "${QGC_PATH}")"
  open_term "QGroundControl" "chmod +x ${QGC_PATH_Q} && ${QGC_PATH_Q}"
else
  echo "Warning: QGroundControl AppImage not found. Checked default: ${QGC_APPIMAGE} and ~/Downloads/*.AppImage"
fi

echo "Starting Gazebo video recording as final step..."
start_video_recording

echo "All components launched in requested order."
