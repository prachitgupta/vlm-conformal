 #!/usr/bin/env bash
  set -euo pipefail

  # Pinned repos/commits from your current working machine
  PX4_COMMIT="684ba28fbf6f7462559620e410b0d9b6d87162f6"
  VLM_COMMIT="8cbe87456e370390c4fd77067721d25770e0b2be"
  XRCE_COMMIT="155cfaaf8b7abac2e85d4a62d3649b09ace0be55"

  if [[ "$(lsb_release -rs)" != "22.04" ]]; then
    echo "This script is for Ubuntu 22.04. Current: $(lsb_release -rs)"
    exit 1
  fi

  sudo -v

  # Locale (ROS 2 requirement)
  sudo apt update
  sudo apt install -y locales
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8

  # ROS 2 apt repo
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key
  \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ro
  s-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs)
  main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null

  # Gazebo apt repo
  sudo wget -q https://packages.osrfoundation.org/gazebo.gpg \
    -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pk
  gs-osrf-archive-keyring.gpg]
  http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) mai
  n" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null

  sudo apt update
  sudo apt install -y \
    curl gnupg2 lsb-release ca-certificates software-properties-common wget \
    git build-essential cmake ninja-build ccache pkg-config \
    python3 python3-dev python3-pip python3-setuptools python3-wheel \
    python3-colcon-common-extensions python3-rosdep python3-vcstool \
    ros-humble-desktop \
    ros-humble-ros-gzharmonic ros-humble-ros-gzharmonic-bridge ros-humble-ros-
  gzharmonic-interfaces \
    ros-humble-rqt-image-view ros-humble-cv-bridge ros-humble-sensor-msgs-py \
    gz-harmonic \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-plugins-good
  \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev libeigen3-dev libopencv-dev \
    protobuf-compiler libxml2-utils dmidecode bc \
    libasio-dev libtinyxml2-dev \
    gnome-terminal xterm

  # rosdep
  sudo rosdep init || true
  rosdep update

  # PX4
  cd "$HOME"
  if [[ ! -d PX4-Autopilot/.git ]]; then
    git clone https://github.com/PX4/PX4-Autopilot.git
  fi
  git -C PX4-Autopilot fetch --all --tags
  git -C PX4-Autopilot checkout "$PX4_COMMIT"

  # PX4 dependencies (SITL-focused)
  bash "$HOME/PX4-Autopilot/Tools/setup/ubuntu.sh" --no-nuttx

  # Micro XRCE DDS Agent
  cd "$HOME"
  if [[ ! -d Micro-XRCE-DDS-Agent/.git ]]; then
    git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
  fi
  git -C Micro-XRCE-DDS-Agent fetch --all --tags
  git -C Micro-XRCE-DDS-Agent checkout "$XRCE_COMMIT"
  cmake -S "$HOME/Micro-XRCE-DDS-Agent" -B "$HOME/Micro-XRCE-DDS-Agent/build"
  cmake --build "$HOME/Micro-XRCE-DDS-Agent/build" -j"$(nproc)"
  sudo cmake --install "$HOME/Micro-XRCE-DDS-Agent/build"

  # Your repo
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

  # Optional px4_msgs workspace (recommended for llm_drone nodes using px4_msgs)
  mkdir -p "$HOME/ros2_ws/src"
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
  grep -qxF 'source /opt/ros/humble/setup.bash' ~/.bashrc || echo 'source /opt/
  ros/humble/setup.bash' >> ~/.bashrc
  grep -qxF 'source ~/ros2_ws/install/setup.bash' ~/.bashrc || echo 'source ~/
  ros2_ws/install/setup.bash' >> ~/.bashrc
  grep -qxF 'source ~/vlm-conformal/px4_ws/install/setup.bash' ~/.bashrc || echo
  'source ~/vlm-conformal/px4_ws/install/setup.bash' >> ~/.bashrc

  echo
  echo "Installation complete."
  echo "Run full stack with:"
  echo "  cd ~/PX4-Autopilot"
  echo "  Tools/simulation/gz/launch_obstacle_avoidance_full_stack.sh"
  echo
  echo "Note: full_stack script opens GUI terminals/rqt; requires desktop/X11."
  EOF
