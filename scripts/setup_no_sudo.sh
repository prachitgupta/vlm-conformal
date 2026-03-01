#!/usr/bin/env bash
set -euo pipefail

# This script intentionally avoids sudo.
# Assumes Ubuntu 22.04 with ROS 2 Humble and Gazebo already installed.

PX4_COMMIT="684ba28fbf6f7462559620e410b0d9b6d87162f6"
VLM_COMMIT="8cbe87456e370390c4fd77067721d25770e0b2be"
XRCE_COMMIT="155cfaaf8b7abac2e85d4a62d3649b09ace0be55"

if [[ "$(lsb_release -rs)" != "22.04" ]]; then
  echo "This script is for Ubuntu 22.04. Current: $(lsb_release -rs)"
  exit 1
fi

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble not found at /opt/ros/humble/setup.bash"
  echo "Install ROS 2 Humble first (requires sudo), then rerun."
  exit 1
fi

for cmd in git cmake python3 colcon; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    echo "Install it with apt first (requires sudo), then rerun."
    exit 1
  fi
done

mkdir -p "$HOME/.local/bin" "$HOME/.local/lib" "$HOME/ros2_ws/src"

# PX4 repo
cd "$HOME"
if [[ ! -d PX4-Autopilot/.git ]]; then
  git clone https://github.com/PX4/PX4-Autopilot.git
fi
git -C PX4-Autopilot fetch --all --tags
git -C PX4-Autopilot checkout "$PX4_COMMIT"

# PX4 Ubuntu dependency script requires sudo/apt, so skip here.
echo "Skipping PX4 apt dependency installer (requires sudo):"
echo "  $HOME/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx"

# Micro XRCE DDS Agent (user-local install)
cd "$HOME"
if [[ ! -d Micro-XRCE-DDS-Agent/.git ]]; then
  git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
fi
git -C Micro-XRCE-DDS-Agent fetch --all --tags
git -C Micro-XRCE-DDS-Agent checkout "$XRCE_COMMIT"
cmake -S "$HOME/Micro-XRCE-DDS-Agent" -B "$HOME/Micro-XRCE-DDS-Agent/build" \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build "$HOME/Micro-XRCE-DDS-Agent/build" -j"$(nproc)"
cmake --install "$HOME/Micro-XRCE-DDS-Agent/build"

# vlm-conformal repo
cd "$HOME"
if [[ ! -d vlm-conformal/.git ]]; then
  git clone https://github.com/prachitgupta/vlm-conformal.git
fi
git -C vlm-conformal fetch --all
git -C vlm-conformal checkout "$VLM_COMMIT"

# Python extras used by llm_drone
python3 -m pip install --user \
  numpy==1.26.4 scipy==1.15.3 cvxpy==1.7.2 matplotlib==3.10.5 \
  opencv-python==4.11.0.86 openai==2.16.0 mavsdk==3.10.2

# px4_msgs workspace
if [[ ! -d "$HOME/ros2_ws/src/px4_msgs" ]]; then
  git clone https://github.com/PX4/px4_msgs.git "$HOME/ros2_ws/src/px4_msgs"
fi
source /opt/ros/humble/setup.bash
cd "$HOME/ros2_ws"
colcon build --packages-select px4_msgs

# Build llm_drone
cd "$HOME/vlm-conformal/px4_ws"
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
colcon build --packages-select llm_drone

# Persist environment sourcing
grep -qxF 'source /opt/ros/humble/setup.bash' ~/.bashrc || echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
grep -qxF 'source ~/ros2_ws/install/setup.bash' ~/.bashrc || echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
grep -qxF 'source ~/vlm-conformal/px4_ws/install/setup.bash' ~/.bashrc || echo 'source ~/vlm-conformal/px4_ws/install/setup.bash' >> ~/.bashrc
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
grep -qxF 'export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"' ~/.bashrc || echo 'export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc

echo
echo "Non-sudo setup complete."
echo "If PX4 SITL build fails, install missing OS packages with sudo, then rerun."
echo "Run full stack with:"
echo "  cd ~/PX4-Autopilot"
echo "  Tools/simulation/gz/launch_obstacle_avoidance_full_stack.sh"
