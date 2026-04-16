cmake_minimum_required(VERSION 3.8)
project(llm_drone)

file(GLOB_RECURSE PYTHON_PACKAGE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  llm_drone/*.py
)
file(GLOB PYTHON_SCRIPT_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  scripts/*
)
set_property(
  DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${PYTHON_PACKAGE_FILES}
  ${PYTHON_SCRIPT_FILES}
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(px4_ros2_cpp REQUIRED)

ament_python_install_package(${PROJECT_NAME})
ament_get_python_install_dir(PYTHON_INSTALL_DIR)

pybind11_add_module(_mpc_native src/mpc_native.cpp)
target_compile_features(_mpc_native PUBLIC cxx_std_17)
target_link_libraries(_mpc_native PRIVATE dl)

install(TARGETS
  _mpc_native
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
)

add_executable(mpc_optimal_planner_cpp src/mpc_optimal_planner_cpp.cpp)
target_compile_features(mpc_optimal_planner_cpp PUBLIC cxx_std_17)
ament_target_dependencies(
  mpc_optimal_planner_cpp
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  cv_bridge
  px4_msgs
)
target_link_libraries(mpc_optimal_planner_cpp ${OpenCV_LIBRARIES})

add_executable(llm_planner_goto_cpp src/llm_planner_goto_cpp.cpp)
target_compile_features(llm_planner_goto_cpp PUBLIC cxx_std_17)
ament_target_dependencies(
  llm_planner_goto_cpp
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  cv_bridge
  px4_msgs
)
target_link_libraries(
  llm_planner_goto_cpp
  ${OpenCV_LIBRARIES}
  CURL::libcurl
  px4_ros2_cpp::px4_ros2_cpp
)

install(TARGETS
  mpc_optimal_planner_cpp
  llm_planner_goto_cpp
  DESTINATION lib/${PROJECT_NAME}
)

set(PYTHON_SCRIPT_WRAPPERS
  scripts/mpc
  scripts/mpc_local_planner
  scripts/mpc_optimal_planner
  scripts/mpc_sim
  scripts/mission_executor
  scripts/llm
  scripts/mpc_voxl
  scripts/llm_voxl
  scripts/performance_analyzer
  scripts/dataset_generator
  scripts/dataset_generator_sync
  scripts/dataset_generator_executor
  scripts/debug_pointcloud_obstacles
  scripts/eval_llm
  scripts/trajectory_goto_debug
  scripts/trajectory_goto_mode_debug
  scripts/generate_synthetic_prompts
  scripts/ground_truth_generator
  scripts/offline_dataset_batch_runner
  scripts/start_llm_planner_goto_cpp_stack.py
)

# Remove stale wrappers from older typo'd names so a rebuild doesn't leave both
# the broken and corrected executables visible in the install space.
install(CODE
  "file(REMOVE
    \"\${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/dataset_genrator_executer\"
    \"\${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/ground_truth_genrator\"
    \"\${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/llm/dataset_genrator_executer.py\"
    \"\${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/llm/ground_truth_genrator.py\"
  )"
)

install(PROGRAMS
  ${PYTHON_SCRIPT_WRAPPERS}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
