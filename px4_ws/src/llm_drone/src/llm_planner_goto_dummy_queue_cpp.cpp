#include <Eigen/Core>

#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/components/mode_executor.hpp>
#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/control/setpoint_types/multicopter/goto.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <px4_ros2/third_party/nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;
using json = nlohmann::json;

namespace
{

constexpr char kNodeName[] = "llm_planner_goto_dummy_queue_cpp";
constexpr char kModeName[] = "LLM Goto Dummy Queue";
constexpr int kWaypointCount = 5;
constexpr float kGoalReachedToleranceM = 0.5f;

struct Waypoint
{
  float x{};
  float y{};
  float z{};
};

struct DummyRollout
{
  int rollout_id{0};
  std::vector<Waypoint> waypoints;
};

struct SharedState
{
  std::mutex mutex;
  std::optional<DummyRollout> latest_rollout;
  int next_rollout_id{1};
};

static float distance3(const Eigen::Vector3f & a, const Eigen::Vector3f & b)
{
  // Small helper used throughout the file to keep all distance checks in one
  // place and avoid repeating the same Eigen norm expression.
  return (a - b).norm();
}

static std::vector<std::string> splitConcatenatedJsonObjects(const std::string & raw_text)
{
  // The user may paste one JSON object or several objects back-to-back. This
  // helper extracts each top-level {...} block without assuming newlines.
  std::vector<std::string> objects;
  int depth = 0;
  size_t object_start = std::string::npos;
  bool in_string = false;
  bool escaping = false;

  for (size_t i = 0; i < raw_text.size(); ++i) {
    const char ch = raw_text[i];

    if (escaping) {
      escaping = false;
      continue;
    }
    if (ch == '\\' && in_string) {
      escaping = true;
      continue;
    }
    if (ch == '"') {
      in_string = !in_string;
      continue;
    }
    if (in_string) {
      continue;
    }
    if (ch == '{') {
      if (depth == 0) {
        object_start = i;
      }
      ++depth;
    } else if (ch == '}') {
      --depth;
      if (depth == 0 && object_start != std::string::npos) {
        objects.push_back(raw_text.substr(object_start, i - object_start + 1));
        object_start = std::string::npos;
      }
    }
  }

  return objects;
}

class DummyQueueGotoMode : public px4_ros2::ModeBase
{
public:
  explicit DummyQueueGotoMode(rclcpp::Node & node)
  : ModeBase(node, Settings{kModeName}.preventArming(false)),
    node_(node),
    goto_setpoint_(std::make_shared<px4_ros2::MulticopterGotoSetpointType>(*this)),
    local_position_(std::make_shared<px4_ros2::OdometryLocalPosition>(*this))
  {
    // Read all debug/control parameters at startup so the node behavior can be
    // tuned from the command line without editing code.
    goal_x_ = static_cast<float>(node_.declare_parameter<double>("goal_x", 35.0));
    goal_y_ = static_cast<float>(node_.declare_parameter<double>("goal_y", 3.0));
    goal_z_ = static_cast<float>(node_.declare_parameter<double>("goal_z", -2.5));
    goal_frame_ = node_.declare_parameter<std::string>("goal_frame", "ned");
    dummy_update_rate_hz_ = node_.declare_parameter<double>("dummy_update_rate_hz", 1.0);
    injected_waypoint_json_ = node_.declare_parameter<std::string>("injected_waypoint_json", "");
    max_horizontal_speed_mps_ = static_cast<float>(
      node_.declare_parameter<double>("max_horizontal_speed_mps", 6.0));
    max_vertical_speed_mps_ = static_cast<float>(
      node_.declare_parameter<double>("max_vertical_speed_mps", 2.0));
    waypoint_acceptance_radius_m_ = static_cast<float>(
      node_.declare_parameter<double>("waypoint_acceptance_radius_m", 0.75));
    rollout_step_xy_m_ = static_cast<float>(
      node_.declare_parameter<double>("rollout_step_xy_m", 2.0));
    rollout_lateral_amplitude_m_ = static_cast<float>(
      node_.declare_parameter<double>("rollout_lateral_amplitude_m", 1.0));
    heading_rad_ = static_cast<float>(
      node_.declare_parameter<double>("heading_rad", std::numeric_limits<double>::quiet_NaN()));

    goal_ned_ = convertGoalToNed(Eigen::Vector3f{goal_x_, goal_y_, goal_z_}, goal_frame_);

    // These publishers mirror the real planner and make it easy to visualize
    // both the current active target and the full queued rollout in RViz.
    traj_pub_ = node_.create_publisher<geometry_msgs::msg::PoseStamped>("/llm/trajectory", 10);
    path_pub_ = node_.create_publisher<nav_msgs::msg::Path>("/llm/trajectory_sequence", 10);

    // This thread plays the role of the LLM thread in the real planner:
    // it periodically generates a fresh 5-waypoint rollout and publishes it
    // into shared state for the tracking loop to consume.
    dummy_worker_thread_ = std::thread(&DummyQueueGotoMode::dummyProducerLoop, this);

    RCLCPP_INFO(
      node_.get_logger(),
      "Dummy queue goto mode initialized. goal_ned=(%.2f, %.2f, %.2f), dummy_update_rate_hz=%.2f",
      goal_ned_.x(), goal_ned_.y(), goal_ned_.z(), dummy_update_rate_hz_);
  }

  ~DummyQueueGotoMode() override
  {
    // Stop the producer thread cleanly before destructing the node. This keeps
    // shutdown deterministic and avoids dangling background work.
    stop_worker_.store(true);
    if (dummy_worker_thread_.joinable()) {
      dummy_worker_thread_.join();
    }
  }

  void onActivate() override
  {
    // Reset all execution-side state every time the PX4 mode becomes active so
    // stale queue entries from an earlier run cannot leak into a new flight.
    std::lock_guard<std::mutex> lock(execution_mutex_);
    applied_rollout_id_ = 0;
    pending_waypoints_.clear();
    active_waypoint_.reset();
    hold_position_ned_.reset();
    RCLCPP_INFO(node_.get_logger(), "Dummy queue goto mode activated");
  }

  void onDeactivate() override
  {
    // Nothing special needs to happen here yet, but the log is useful for
    // tracing ownership handoffs between PX4 modes.
    RCLCPP_INFO(node_.get_logger(), "Dummy queue goto mode deactivated");
  }

  void updateSetpoint(float dt_s) override
  {
    // This is the main control loop for the PX4 custom mode. PX4 calls this
    // continuously while the mode is active, and this function decides whether
    // to hold position or command the next queued waypoint.
    (void)dt_s;

    if (!local_position_->positionXYValid() || !local_position_->positionZValid()) {
      RCLCPP_WARN_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 2000, "Waiting for valid local position");
      return;
    }

    const Eigen::Vector3f current_position = local_position_->positionNed();

    if (goalReached(current_position)) {
      publishHoldPosition(current_position);
      return;
    }

    // Load a newly generated rollout if the producer thread has published one.
    {
      std::lock_guard<std::mutex> state_lock(shared_state_.mutex);
      if (shared_state_.latest_rollout &&
          shared_state_.latest_rollout->rollout_id > applied_rollout_id_) {
        pending_waypoints_.clear();
        for (const auto & waypoint : shared_state_.latest_rollout->waypoints) {
          pending_waypoints_.push_back(waypoint);
        }
        active_waypoint_.reset();
        applied_rollout_id_ = shared_state_.latest_rollout->rollout_id;
        publishWaypointPath(shared_state_.latest_rollout->waypoints);
        RCLCPP_INFO(
          node_.get_logger(),
          "Loaded dummy rollout %d with %zu waypoints",
          applied_rollout_id_, pending_waypoints_.size());
      }
    }

    // When the active waypoint is reached, drop it and advance to the next one.
    if (active_waypoint_ && waypointReached(current_position, *active_waypoint_)) {
      RCLCPP_INFO(
        node_.get_logger(),
        "Reached dummy rollout %d waypoint at (%.2f, %.2f, %.2f)",
        applied_rollout_id_, active_waypoint_->x, active_waypoint_->y, active_waypoint_->z);
      active_waypoint_.reset();
    }

    // If there is no current target, consume the next queued waypoint.
    if (!active_waypoint_ && !pending_waypoints_.empty()) {
      active_waypoint_ = pending_waypoints_.front();
      pending_waypoints_.pop_front();
      RCLCPP_INFO(
        node_.get_logger(),
        "Tracking dummy rollout %d waypoint (%.2f, %.2f, %.2f)",
        applied_rollout_id_, active_waypoint_->x, active_waypoint_->y, active_waypoint_->z);
    }

    // Same behavior as the real planner: hold current position when queue is empty.
    if (!active_waypoint_) {
      publishHoldPosition(current_position);
      return;
    }

    const Eigen::Vector3f target = toEigen(*active_waypoint_);
    const std::optional<float> heading =
      std::isfinite(heading_rad_) ? std::optional<float>(heading_rad_) : std::nullopt;
    goto_setpoint_->update(target, heading, max_horizontal_speed_mps_, max_vertical_speed_mps_);
    publishActiveWaypoint(*active_waypoint_);

    RCLCPP_INFO_THROTTLE(
      node_.get_logger(),
      *node_.get_clock(),
      1000,
      "Dummy target=(%.2f, %.2f, %.2f) current=(%.2f, %.2f, %.2f)",
      target.x(), target.y(), target.z(),
      current_position.x(), current_position.y(), current_position.z());
  }

private:
  static Eigen::Vector3f toEigen(const Waypoint & waypoint)
  {
    // Convert the lightweight queue representation into Eigen so the rest of
    // the motion logic can use vector arithmetic directly.
    return Eigen::Vector3f{waypoint.x, waypoint.y, waypoint.z};
  }

  static Eigen::Vector3f convertGoalToNed(const Eigen::Vector3f & goal_xyz, const std::string & goal_frame)
  {
    // Match the real planner behavior: user goals may be supplied in Gazebo ENU
    // coordinates, but the PX4 goto API expects local NED.
    std::string lower = goal_frame;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (lower == "gazebo" || lower == "gazebo_enu" || lower == "enu" || lower == "map") {
      return Eigen::Vector3f{goal_xyz.y(), goal_xyz.x(), -goal_xyz.z()};
    }
    return goal_xyz;
  }

  bool goalReached(const Eigen::Vector3f & current_position) const
  {
    // Once the vehicle is close enough to the mission goal, stop consuming the
    // queue and simply hold position at the current location.
    return distance3(current_position, goal_ned_) <= kGoalReachedToleranceM;
  }

  bool waypointReached(const Eigen::Vector3f & current_position, const Waypoint & waypoint) const
  {
    // Use the same basic acceptance test as the real goto planner: a waypoint
    // counts as complete when XY error is small and altitude error is bounded.
    const Eigen::Vector3f target = toEigen(waypoint);
    const float xy_error = (target.head<2>() - current_position.head<2>()).norm();
    const float z_error = std::fabs(target.z() - current_position.z());
    return xy_error <= waypoint_acceptance_radius_m_ && z_error <= 0.5f;
  }

  void publishHoldPosition(const Eigen::Vector3f & current_position)
  {
    // Holding is implemented by freezing the first hold point we see and
    // repeatedly commanding PX4's goto controller to remain there.
    if (!hold_position_ned_) {
      hold_position_ned_ = current_position;
    }
    const std::optional<float> heading =
      std::isfinite(heading_rad_) ? std::optional<float>(heading_rad_) : std::nullopt;
    goto_setpoint_->update(
      *hold_position_ned_,
      heading,
      std::min(max_horizontal_speed_mps_, 1.5f),
      std::min(max_vertical_speed_mps_, 1.0f));
  }

  void publishActiveWaypoint(const Waypoint & waypoint)
  {
    // Publish only the currently tracked waypoint so debugging tools can show
    // exactly what point PX4 is trying to reach right now.
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = node_.now();
    pose_msg.header.frame_id = "ned";
    pose_msg.pose.position.x = waypoint.x;
    pose_msg.pose.position.y = waypoint.y;
    pose_msg.pose.position.z = waypoint.z;
    pose_msg.pose.orientation.w = 1.0;
    traj_pub_->publish(pose_msg);
  }

  void publishWaypointPath(const std::vector<Waypoint> & waypoints)
  {
    // Publish the whole queued rollout as a Path message to make queue refreshes
    // visible during testing.
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = node_.now();
    path_msg.header.frame_id = "ned";
    for (const auto & waypoint : waypoints) {
      geometry_msgs::msg::PoseStamped pose_msg;
      pose_msg.header = path_msg.header;
      pose_msg.pose.position.x = waypoint.x;
      pose_msg.pose.position.y = waypoint.y;
      pose_msg.pose.position.z = waypoint.z;
      pose_msg.pose.orientation.w = 1.0;
      path_msg.poses.push_back(pose_msg);
    }
    path_pub_->publish(path_msg);
  }

  // Generate a smooth synthetic rollout that progresses toward the goal while
  // alternating a small lateral offset. This makes it easy to see queue updates
  // in logs and RViz without involving any LLM or verifier.
  DummyRollout makeDummyRollout(const Eigen::Vector3f & current_position, int rollout_id) const
  {
    // Synthesize a new five-point rollout that generally moves toward the goal.
    // The alternating lateral offset makes rollout replacement obvious in logs
    // and visualizations.
    DummyRollout rollout;
    rollout.rollout_id = rollout_id;

    Eigen::Vector3f direction = goal_ned_ - current_position;
    direction.z() = 0.0f;
    const float norm = direction.norm();
    if (norm < 1e-4f) {
      direction = Eigen::Vector3f{1.0f, 0.0f, 0.0f};
    } else {
      direction /= norm;
    }

    const Eigen::Vector3f lateral{-direction.y(), direction.x(), 0.0f};

    for (int i = 0; i < kWaypointCount; ++i) {
      const float along = rollout_step_xy_m_ * static_cast<float>(i + 1);
      const float lateral_sign = ((rollout_id + i) % 2 == 0) ? 1.0f : -1.0f;
      const float lateral_scale = (i == kWaypointCount - 1) ? 0.0f : rollout_lateral_amplitude_m_;
      const Eigen::Vector3f point =
        current_position + along * direction + lateral_sign * lateral_scale * lateral;
      rollout.waypoints.push_back(Waypoint{point.x(), point.y(), goal_ned_.z()});
    }

    return rollout;
  }

  std::optional<DummyRollout> parseInjectedRollout(
    const std::string & raw_text,
    const Eigen::Vector3f & current_position,
    int rollout_id) const
  {
    // Parse user-supplied waypoint JSON in the same general shape as the real
    // planner output. If several JSON objects are concatenated together, the
    // newest one wins because that best matches "latest rollout" semantics.
    if (raw_text.empty()) {
      return std::nullopt;
    }

    const std::vector<std::string> objects = splitConcatenatedJsonObjects(raw_text);
    if (objects.empty()) {
      throw std::runtime_error("No JSON object found in injected_waypoint_json");
    }

    const json parsed = json::parse(objects.back());
    if (!parsed.contains("waypoints") || !parsed.at("waypoints").is_array()) {
      throw std::runtime_error("Injected JSON must contain a 'waypoints' array");
    }

    const auto & waypoints_json = parsed.at("waypoints");
    if (waypoints_json.empty()) {
      throw std::runtime_error("Injected 'waypoints' array is empty");
    }

    DummyRollout rollout;
    rollout.rollout_id = rollout_id;
    rollout.waypoints.reserve(waypoints_json.size());

    // Interpret x/y as local offsets from the current vehicle position. This
    // makes pasted planner outputs immediately usable even if they were written
    // as relative "next waypoint" commands rather than absolute map points.
    for (const auto & waypoint_json : waypoints_json) {
      if (!waypoint_json.contains("x") || !waypoint_json.contains("y")) {
        throw std::runtime_error("Each injected waypoint must contain x and y");
      }

      const float dx = waypoint_json.at("x").get<float>();
      const float dy = waypoint_json.at("y").get<float>();
      const float z = waypoint_json.contains("z")
        ? waypoint_json.at("z").get<float>()
        : goal_ned_.z();

      const Eigen::Vector3f point_ned{
        current_position.x() + dx,
        current_position.y() + dy,
        z,
      };
      rollout.waypoints.push_back(Waypoint{point_ned.x(), point_ned.y(), point_ned.z()});
    }

    return rollout;
  }

  void dummyProducerLoop()
  {
    // Background producer thread. This deliberately mirrors the real planner's
    // "LLM thread updates the latest rollout" architecture, but replaces the
    // LLM call with deterministic synthetic waypoints.
    const auto sleep_period =
      std::chrono::duration<double>(1.0 / std::max(0.1, dummy_update_rate_hz_));

    while (!stop_worker_.load()) {
      try {
        if (!local_position_->positionXYValid() || !local_position_->positionZValid()) {
          std::this_thread::sleep_for(500ms);
          continue;
        }

        const Eigen::Vector3f current_position = local_position_->positionNed();
        if (goalReached(current_position)) {
          std::this_thread::sleep_for(sleep_period);
          continue;
        }

        DummyRollout rollout;
        {
          std::lock_guard<std::mutex> lock(shared_state_.mutex);
          const int rollout_id = shared_state_.next_rollout_id++;
          const std::string injected_waypoint_json =
            node_.get_parameter("injected_waypoint_json").as_string();

          // When the user supplies manual waypoint JSON, parse and execute that.
          // Otherwise keep the original synthetic rollout generator as fallback.
          const auto injected_rollout =
            parseInjectedRollout(injected_waypoint_json, current_position, rollout_id);
          rollout = injected_rollout.has_value()
            ? *injected_rollout
            : makeDummyRollout(current_position, rollout_id);
          shared_state_.latest_rollout = rollout;
        }

        RCLCPP_INFO(
          node_.get_logger(),
          "Published dummy rollout %d with %zu waypoints at %.2f Hz",
          rollout.rollout_id, rollout.waypoints.size(), dummy_update_rate_hz_);
      } catch (const std::exception & exc) {
        RCLCPP_ERROR_THROTTLE(
          node_.get_logger(),
          *node_.get_clock(),
          2000,
          "Failed to build injected dummy rollout: %s",
          exc.what());
      }

      std::this_thread::sleep_for(sleep_period);
    }
  }

  rclcpp::Node & node_;
  std::shared_ptr<px4_ros2::MulticopterGotoSetpointType> goto_setpoint_;
  std::shared_ptr<px4_ros2::OdometryLocalPosition> local_position_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr traj_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

  SharedState shared_state_;
  std::mutex execution_mutex_;
  std::deque<Waypoint> pending_waypoints_;
  std::optional<Waypoint> active_waypoint_;
  std::optional<Eigen::Vector3f> hold_position_ned_;
  int applied_rollout_id_{0};

  std::thread dummy_worker_thread_;
  std::atomic<bool> stop_worker_{false};

  float goal_x_{};
  float goal_y_{};
  float goal_z_{};
  Eigen::Vector3f goal_ned_{};
  std::string goal_frame_{"ned"};
  double dummy_update_rate_hz_{1.0};
  std::string injected_waypoint_json_;
  float max_horizontal_speed_mps_{6.0f};
  float max_vertical_speed_mps_{2.0f};
  float waypoint_acceptance_radius_m_{0.75f};
  float rollout_step_xy_m_{2.0f};
  float rollout_lateral_amplitude_m_{1.0f};
  float heading_rad_{std::numeric_limits<float>::quiet_NaN()};
};

class DummyQueueGotoExecutor : public px4_ros2::ModeExecutorBase
{
public:
  explicit DummyQueueGotoExecutor(px4_ros2::ModeBase & owned_mode)
  : ModeExecutorBase(
      []() {
        // ActivateImmediately lets this executor take charge as soon as the
        // node registers with PX4, which is what we want for an autonomous
        // startup test.
        px4_ros2::ModeExecutorBase::Settings settings;
        settings.activate(px4_ros2::ModeExecutorBase::Settings::Activation::ActivateImmediately);
        return settings;
      }(),
      owned_mode),
    node_(owned_mode.node())
  {
  }

  void onActivate() override
  {
    // Entry point for the startup state machine. Wait for PX4 arming checks to
    // pass, then arm, then use PX4's built-in takeoff action.
    RCLCPP_INFO(
      node_.get_logger(),
      "Dummy executor activated, waiting for arming checks before arm and takeoff");
    waitReadyToArm([this](px4_ros2::Result ready_result) { onReadyToArm(ready_result); });
  }

  void onDeactivate(DeactivateReason reason) override
  {
    // This is mainly a trace hook: if PX4 takes control away from the executor,
    // the log tells us when that happened.
    RCLCPP_INFO(node_.get_logger(), "Dummy executor deactivated (%d)", static_cast<int>(reason));
  }

private:
  void onReadyToArm(px4_ros2::Result result)
  {
    // PX4's arming checks must pass before we send an arm command. This is the
    // gate that prevents takeoff from starting too early.
    if (result != px4_ros2::Result::Success) {
      RCLCPP_ERROR(node_.get_logger(), "Vehicle not ready to arm: %s", resultToString(result));
      return;
    }
    RCLCPP_INFO(node_.get_logger(), "Arming checks passed. Arming vehicle.");
    arm([this](px4_ros2::Result arm_result) { onArmed(arm_result); });
  }

  void onArmed(px4_ros2::Result result)
  {
    // Once the vehicle is armed successfully, transition immediately to PX4's
    // built-in takeoff action.
    if (result != px4_ros2::Result::Success) {
      RCLCPP_ERROR(node_.get_logger(), "Arm failed: %s", resultToString(result));
      return;
    }
    RCLCPP_INFO(node_.get_logger(), "Armed. Starting takeoff.");
    takeoff([this](px4_ros2::Result takeoff_result) { onTakeoffCompleted(takeoff_result); });
  }

  void onTakeoffCompleted(px4_ros2::Result result)
  {
    // After PX4 confirms takeoff, hand over control to the custom dummy queue
    // goto mode, which then starts consuming generated waypoints.
    if (result != px4_ros2::Result::Success) {
      RCLCPP_ERROR(node_.get_logger(), "Takeoff failed: %s", resultToString(result));
      return;
    }
    RCLCPP_INFO(node_.get_logger(), "Takeoff complete. Handing control to dummy queue goto mode.");
    scheduleMode(ownedMode().id(), [this](px4_ros2::Result mode_result) {
      RCLCPP_INFO(
        node_.get_logger(),
        "Dummy queue goto mode finished with result: %s",
        resultToString(mode_result));
    });
  }

  rclcpp::Node & node_;
};

}  // namespace

int main(int argc, char ** argv)
{
  // Standard ROS 2 entrypoint: initialize ROS, create the combined
  // mode-executor node, spin until shutdown, then clean up.
  rclcpp::init(argc, argv);
  using DummyPlannerNode = px4_ros2::NodeWithModeExecutor<DummyQueueGotoExecutor, DummyQueueGotoMode>;
  rclcpp::spin(std::make_shared<DummyPlannerNode>(kNodeName, true));
  rclcpp::shutdown();
  return 0;
}
