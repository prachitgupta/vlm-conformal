#include <Eigen/Core>
#include <curl/curl.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/components/mode_executor.hpp>
#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/control/setpoint_types/multicopter/goto.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <px4_ros2/third_party/nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

using json = nlohmann::json;
using namespace std::chrono_literals;

namespace
{

constexpr char kNodeName[] = "llm_planner_goto_cpp";
constexpr char kModeName[] = "LLM Goto Planner";
constexpr int kWaypointCount = 5;
constexpr float kGoalReachedToleranceM = 0.5f;
constexpr float kWaypointVelocityDtS = 1.0f;
constexpr float kDepthMinM = 0.2f;
constexpr float kDepthMaxM = 19.1f;
constexpr float kDepthHfovDeg = 72.995f;
constexpr float kDepthVfovDeg = 58.053f;
constexpr int kDepthSampleCountDefault = 500;
constexpr size_t kObstacleHistoryMax = 200;
constexpr char kDefaultPromptFile[] =
  "/home/prachit/Desktop/vlm-conformal/px4_ws/src/llm_drone/config/variant_X.txt";
constexpr char kDefaultOpenAiUrl[] = "https://api.openai.com/v1/chat/completions";
constexpr char kDefaultOpenAiModel[] = "gpt-4o-mini";

struct Waypoint
{
  float x{};
  float y{};
  float z{};
};

struct PlannerRollout
{
  int rollout_id{0};
  std::vector<Waypoint> waypoints;
  std::string reasoning;
  int selected_index{0};
};

struct VerificationResult
{
  bool passed{false};
  std::string summary;
};

struct SharedState
{
  std::mutex mutex;
  std::optional<cv::Mat> depth_image_m;
  std::vector<Eigen::Vector3f> obstacle_points_ned;
  std::optional<PlannerRollout> latest_rollout;
  int next_rollout_id{1};
};

static size_t curlWriteCallback(void * contents, size_t size, size_t nmemb, void * userp)
{
  const size_t bytes = size * nmemb;
  auto * output = static_cast<std::string *>(userp);
  output->append(static_cast<const char *>(contents), bytes);
  return bytes;
}

static float clampFloat(float value, float lo, float hi)
{
  return std::max(lo, std::min(value, hi));
}

static float distance3(const Eigen::Vector3f & a, const Eigen::Vector3f & b)
{
  return (a - b).norm();
}

static float segmentPointDistance(const Eigen::Vector3f & a, const Eigen::Vector3f & b, const Eigen::Vector3f & p)
{
  const Eigen::Vector3f ab = b - a;
  const float denom = ab.squaredNorm();
  if (denom < 1e-9f) {
    return (p - a).norm();
  }
  const float t = clampFloat((p - a).dot(ab) / denom, 0.0f, 1.0f);
  const Eigen::Vector3f proj = a + t * ab;
  return (p - proj).norm();
}

static std::string loadTextFile(const std::string & path)
{
  std::ifstream file(path);
  if (!file.is_open()) {
    return {};
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

static std::string getenvOrEmpty(const char * name)
{
  const char * value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string{};
}

static std::string extractJsonObject(const std::string & text)
{
  const auto start = text.find('{');
  const auto end = text.rfind('}');
  if (start == std::string::npos || end == std::string::npos || end < start) {
    return {};
  }
  return text.substr(start, end - start + 1);
}

static std::vector<std::string> splitConcatenatedJsonObjects(const std::string & raw_text)
{
  // Accept either a single JSON object or several back-to-back objects and
  // return each top-level {...} block. When multiple are present, the newest
  // one is used later to match "latest rollout wins" semantics.
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

class LlmGotoPlannerMode : public px4_ros2::ModeBase
{
public:
  explicit LlmGotoPlannerMode(rclcpp::Node & node)
  : ModeBase(node, Settings{kModeName}.preventArming(false)),
    node_(node),
    goto_setpoint_(std::make_shared<px4_ros2::MulticopterGotoSetpointType>(*this)),
    local_position_(std::make_shared<px4_ros2::OdometryLocalPosition>(*this))
  {
    goal_x_ = static_cast<float>(node_.declare_parameter<double>("goal_x", 35.0));
    goal_y_ = static_cast<float>(node_.declare_parameter<double>("goal_y", 3.0));
    goal_z_ = static_cast<float>(node_.declare_parameter<double>("goal_z", -2.5));
    update_rate_hz_ = node_.declare_parameter<double>("update_rate", 0.5);
    max_horizontal_speed_mps_ = static_cast<float>(
      node_.declare_parameter<double>("max_horizontal_speed_mps", 6.0));
    max_vertical_speed_mps_ = static_cast<float>(
      node_.declare_parameter<double>("max_vertical_speed_mps", 2.0));
    waypoint_acceptance_radius_m_ = static_cast<float>(
      node_.declare_parameter<double>("waypoint_acceptance_radius_m", 0.75));
    verification_safety_radius_m_ = static_cast<float>(
      node_.declare_parameter<double>("verification_safety_radius_m", 0.60));
    verification_max_velocity_mps_ = static_cast<float>(
      node_.declare_parameter<double>("verification_max_velocity_mps", 15.0));
    verification_max_accel_mps2_ = static_cast<float>(
      node_.declare_parameter<double>("verification_max_accel_mps2", 12.0));
    depth_obstacle_samples_ = node_.declare_parameter<int>(
      "depth_obstacle_samples", kDepthSampleCountDefault);
    goal_frame_ = node_.declare_parameter<std::string>("goal_frame", "ned");
    llm_backend_ = node_.declare_parameter<std::string>("llm_backend", "vllm");
    vllm_url_ = node_.declare_parameter<std::string>(
      "vllm_url", "http://172.22.224.93:8000/v1/chat/completions");
    vllm_model_ = node_.declare_parameter<std::string>(
      "vllm_model", "drone_planner_gt");
    vllm_api_key_ = node_.declare_parameter<std::string>("vllm_api_key", "token-abc123");
    openai_url_ = node_.declare_parameter<std::string>("openai_url", kDefaultOpenAiUrl);
    openai_model_ = node_.declare_parameter<std::string>("openai_model", kDefaultOpenAiModel);
    openai_api_key_ = node_.declare_parameter<std::string>("openai_api_key", "");
    vllm_temperature_ = node_.declare_parameter<double>("vllm_temperature", 0.3);
    vllm_max_tokens_ = node_.declare_parameter<int>("vllm_max_tokens", 256);
    prompt_file_ = node_.declare_parameter<std::string>("prompt_file", kDefaultPromptFile);
    heading_rad_ = static_cast<float>(node_.declare_parameter<double>("heading_rad", std::numeric_limits<double>::quiet_NaN()));

    if (openai_api_key_.empty()) {
      openai_api_key_ = getenvOrEmpty("OPENAI_API_KEY");
    }

    goal_ned_ = convertGoalToNed(Eigen::Vector3f{goal_x_, goal_y_, goal_z_}, goal_frame_);

    const auto sensor_qos = rclcpp::SensorDataQoS();
    depth_sub_ = node_.create_subscription<sensor_msgs::msg::Image>(
      "/depth_camera",
      sensor_qos,
      std::bind(&LlmGotoPlannerMode::onDepthImage, this, std::placeholders::_1));

    llm_traj_pub_ = node_.create_publisher<geometry_msgs::msg::PoseStamped>("/llm/trajectory", 10);
    llm_path_pub_ = node_.create_publisher<nav_msgs::msg::Path>("/llm/trajectory_sequence", 10);

    system_prompt_ = loadSystemPrompt();
    worker_thread_ = std::thread(&LlmGotoPlannerMode::llmWorkerLoop, this);

    RCLCPP_INFO(
      node_.get_logger(),
      "LLM goto planner mode initialized. goal_ned=(%.2f, %.2f, %.2f) backend=%s model=%s",
      goal_ned_.x(), goal_ned_.y(), goal_ned_.z(), llm_backend_.c_str(), activeModelName().c_str());
  }

  ~LlmGotoPlannerMode() override
  {
    stop_worker_.store(true);
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }

  void onActivate() override
  {
    std::lock_guard<std::mutex> lock(execution_mutex_);
    applied_rollout_id_ = 0;
    pending_waypoints_.clear();
    active_waypoint_.reset();
    hold_position_ned_.reset();
    RCLCPP_INFO(node_.get_logger(), "LLM goto planner mode activated");
  }

  void onDeactivate() override
  {
    RCLCPP_INFO(node_.get_logger(), "LLM goto planner mode deactivated");
  }

  void updateSetpoint(float dt_s) override
  {
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
          "Loaded rollout %d with %zu verified waypoints",
          applied_rollout_id_, pending_waypoints_.size());
      }
    }

    if (active_waypoint_ && waypointReached(current_position, *active_waypoint_)) {
      RCLCPP_INFO(
        node_.get_logger(),
        "Reached rollout %d waypoint at (%.2f, %.2f, %.2f)",
        applied_rollout_id_, active_waypoint_->x, active_waypoint_->y, active_waypoint_->z);
      active_waypoint_.reset();
    }

    if (!active_waypoint_ && !pending_waypoints_.empty()) {
      active_waypoint_ = pending_waypoints_.front();
      pending_waypoints_.pop_front();
      RCLCPP_INFO(
        node_.get_logger(),
        "Tracking rollout %d waypoint (%.2f, %.2f, %.2f)",
        applied_rollout_id_, active_waypoint_->x, active_waypoint_->y, active_waypoint_->z);
    }

    if (!active_waypoint_) {
      publishHoldPosition(current_position);
      return;
    }

    const Eigen::Vector3f target = toEigen(*active_waypoint_);
    const std::optional<float> heading = std::isfinite(heading_rad_) ? std::optional<float>(heading_rad_) : std::nullopt;
    goto_setpoint_->update(
      target,
      heading,
      max_horizontal_speed_mps_,
      max_vertical_speed_mps_);
    publishActiveWaypoint(*active_waypoint_);

    RCLCPP_INFO_THROTTLE(
      node_.get_logger(),
      *node_.get_clock(),
      1000,
      "Active waypoint: target=(%.2f, %.2f, %.2f) current=(%.2f, %.2f, %.2f)",
      target.x(), target.y(), target.z(),
      current_position.x(), current_position.y(), current_position.z());
  }

private:
  static Eigen::Vector3f toEigen(const Waypoint & waypoint)
  {
    return Eigen::Vector3f{waypoint.x, waypoint.y, waypoint.z};
  }

  static Eigen::Vector3f convertGoalToNed(const Eigen::Vector3f & goal_xyz, const std::string & goal_frame)
  {
    const std::string lower = [&goal_frame]() {
      std::string out = goal_frame;
      std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      return out;
    }();
    if (lower == "gazebo" || lower == "gazebo_enu" || lower == "enu" || lower == "map") {
      return Eigen::Vector3f{goal_xyz.y(), goal_xyz.x(), -goal_xyz.z()};
    }
    return goal_xyz;
  }

  bool goalReached(const Eigen::Vector3f & position_ned) const
  {
    return distance3(position_ned, goal_ned_) <= kGoalReachedToleranceM;
  }

  bool waypointReached(const Eigen::Vector3f & current_position, const Waypoint & waypoint) const
  {
    const Eigen::Vector3f target = toEigen(waypoint);
    const float xy_error = (target.head<2>() - current_position.head<2>()).norm();
    const float z_error = std::fabs(target.z() - current_position.z());
    return xy_error <= waypoint_acceptance_radius_m_ && z_error <= 0.5f;
  }

  void publishHoldPosition(const Eigen::Vector3f & current_position)
  {
    if (!hold_position_ned_) {
      hold_position_ned_ = current_position;
    }
    const std::optional<float> heading = std::isfinite(heading_rad_) ? std::optional<float>(heading_rad_) : std::nullopt;
    goto_setpoint_->update(
      *hold_position_ned_,
      heading,
      std::min(max_horizontal_speed_mps_, 1.5f),
      std::min(max_vertical_speed_mps_, 1.0f));
  }

  void publishActiveWaypoint(const Waypoint & waypoint)
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = node_.now();
    pose_msg.header.frame_id = "ned";
    pose_msg.pose.position.x = waypoint.x;
    pose_msg.pose.position.y = waypoint.y;
    pose_msg.pose.position.z = waypoint.z;
    pose_msg.pose.orientation.w = 1.0;
    llm_traj_pub_->publish(pose_msg);
  }

  void publishWaypointPath(const std::vector<Waypoint> & waypoints)
  {
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
    llm_path_pub_->publish(path_msg);
  }

  std::string loadSystemPrompt() const
  {
    std::string prompt = loadTextFile(prompt_file_);
    if (!prompt.empty()) {
      return prompt;
    }
    return
      "You are a PX4 multicopter planner. Return exactly one JSON object with keys "
      "\"waypoints\", \"selected_waypoint_index\", and \"reasoning\". "
      "\"waypoints\" must contain exactly 5 waypoint objects. "
      "Each waypoint object must include numeric x and y fields in local NED meters. "
      "z is optional; if omitted the controller will use the mission goal altitude. "
      "Choose safe, smooth waypoints that move closer to the goal and avoid obstacles.";
  }

  void onDepthImage(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat depth_m;
    try {
      if (msg->encoding == "16UC1" || msg->encoding == "mono16") {
        depth_m = cv_bridge::toCvCopy(msg, msg->encoding)->image;
        depth_m.convertTo(depth_m, CV_32FC1, 1.0 / 1000.0);
      } else {
        depth_m = cv_bridge::toCvCopy(msg, "32FC1")->image;
      }
    } catch (const std::exception & exc) {
      RCLCPP_ERROR_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 2000, "Depth conversion failed: %s", exc.what());
      return;
    }

    if (!local_position_->positionXYValid() || !local_position_->positionZValid()) {
      return;
    }

    std::vector<Eigen::Vector3f> points_ned = depthToObstaclePoints(depth_m, local_position_->positionNed());
    std::lock_guard<std::mutex> lock(shared_state_.mutex);
    shared_state_.depth_image_m = depth_m.clone();
    shared_state_.obstacle_points_ned = std::move(points_ned);
    if (shared_state_.obstacle_points_ned.size() > kObstacleHistoryMax) {
      shared_state_.obstacle_points_ned.erase(
        shared_state_.obstacle_points_ned.begin(),
        shared_state_.obstacle_points_ned.end() - static_cast<std::ptrdiff_t>(kObstacleHistoryMax));
    }
  }

  std::vector<Eigen::Vector3f> depthToObstaclePoints(const cv::Mat & depth_m, const Eigen::Vector3f & position_ned) const
  {
    std::vector<Eigen::Vector3f> points;
    if (depth_m.empty() || depth_m.rows <= 0 || depth_m.cols <= 0) {
      return points;
    }

    const float hfov = kDepthHfovDeg * static_cast<float>(M_PI) / 180.0f;
    const float vfov = kDepthVfovDeg * static_cast<float>(M_PI) / 180.0f;
    const float fx = (static_cast<float>(depth_m.cols) / 2.0f) / std::tan(hfov / 2.0f);
    const float fy = (static_cast<float>(depth_m.rows) / 2.0f) / std::tan(vfov / 2.0f);
    const float cx = static_cast<float>(depth_m.cols) / 2.0f;
    const float cy = static_cast<float>(depth_m.rows) / 2.0f;
    const Eigen::Matrix3f rotation_body_camera =
      (Eigen::Matrix3f() << 0.0f, 0.0f, 1.0f,
                            -1.0f, 0.0f, 0.0f,
                            0.0f, -1.0f, 0.0f)
        .finished();
    const Eigen::Vector3f translation_body_camera{0.13233f, 0.0f, 0.26078f};

    const int row_stride = std::max(1, depth_m.rows / std::max(1, static_cast<int>(std::sqrt(depth_obstacle_samples_))));
    const int col_stride = std::max(1, depth_m.cols / std::max(1, static_cast<int>(std::sqrt(depth_obstacle_samples_))));

    for (int r = 0; r < depth_m.rows; r += row_stride) {
      for (int c = 0; c < depth_m.cols; c += col_stride) {
        const float depth = depth_m.at<float>(r, c);
        if (!std::isfinite(depth) || depth < kDepthMinM || depth > kDepthMaxM) {
          continue;
        }
        const float xc = (static_cast<float>(c) - cx) / fx * depth;
        const float yc = (static_cast<float>(r) - cy) / fy * depth;
        const float zc = depth;
        const Eigen::Vector3f point_cam{xc, yc, zc};
        const Eigen::Vector3f point_body = rotation_body_camera * point_cam + translation_body_camera;
        if (point_body.x() < 0.35f || std::fabs(point_body.y()) > 4.0f || point_body.z() < -0.25f || point_body.z() > 2.5f) {
          continue;
        }
        points.push_back(position_ned + point_body);
      }
    }
    return points;
  }

  std::string buildUserPrompt(
    const Eigen::Vector3f & current_position,
    const Eigen::Vector3f & current_velocity,
    const std::vector<Eigen::Vector3f> & obstacle_points) const
  {
    std::ostringstream prompt;
    prompt << "Plan the next 5 waypoints for a PX4 multicopter.\n";
    prompt << "Current position NED: [" << current_position.x() << ", " << current_position.y() << ", " << current_position.z() << "]\n";
    prompt << "Current velocity NED: [" << current_velocity.x() << ", " << current_velocity.y() << ", " << current_velocity.z() << "]\n";
    prompt << "Goal position NED: [" << goal_ned_.x() << ", " << goal_ned_.y() << ", " << goal_ned_.z() << "]\n";
    prompt << "Provide exactly 5 waypoint objects with x/y (and optional z).\n";
    prompt << "Obstacle samples NED:\n";
    const size_t limit = std::min<size_t>(obstacle_points.size(), 30);
    for (size_t i = 0; i < limit; ++i) {
      const auto & point = obstacle_points[i];
      prompt << "- [" << point.x() << ", " << point.y() << ", " << point.z() << "]\n";
    }
    if (limit == 0) {
      prompt << "- none\n";
    }
    return prompt.str();
  }

  std::optional<std::string> requestPlannerResponse(const std::string & user_prompt) const
  {
    CURL * curl = curl_easy_init();
    if (curl == nullptr) {
      RCLCPP_ERROR(node_.get_logger(), "Failed to initialize curl");
      return std::nullopt;
    }

    const std::string request_url = activeRequestUrl();
    const std::string model_name = activeModelName();
    const std::string api_key = activeApiKey();

    if (request_url.empty() || model_name.empty()) {
      RCLCPP_ERROR(node_.get_logger(), "LLM backend is missing URL or model configuration");
      curl_easy_cleanup(curl);
      return std::nullopt;
    }
    if (usingOpenAiBackend() && api_key.empty()) {
      RCLCPP_ERROR(
        node_.get_logger(),
        "OpenAI backend selected but openai_api_key and OPENAI_API_KEY are empty");
      curl_easy_cleanup(curl);
      return std::nullopt;
    }

    json payload = {
      {"model", model_name},
      {"temperature", vllm_temperature_},
      {"max_tokens", vllm_max_tokens_},
      {"messages", {
        {{"role", "system"}, {"content", system_prompt_}},
        {{"role", "user"}, {"content", user_prompt}}
      }}
    };

    std::string response;
    struct curl_slist * headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!api_key.empty()) {
      const std::string auth_header = "Authorization: Bearer " + api_key;
      headers = curl_slist_append(headers, auth_header.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, request_url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    const std::string body = payload.dump();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    const CURLcode result = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (result != CURLE_OK) {
      RCLCPP_ERROR(node_.get_logger(), "LLM request failed: %s", curl_easy_strerror(result));
      return std::nullopt;
    }
    if (http_code < 200 || http_code >= 300) {
      RCLCPP_ERROR(node_.get_logger(), "LLM backend returned HTTP %ld: %s", http_code, response.c_str());
      return std::nullopt;
    }

    try {
      const auto outer = json::parse(response);
      return outer.at("choices").at(0).at("message").at("content").get<std::string>();
    } catch (const std::exception & exc) {
      RCLCPP_ERROR(node_.get_logger(), "Failed to parse LLM response envelope: %s", exc.what());
      return std::nullopt;
    }
  }

  bool usingOpenAiBackend() const
  {
    std::string backend = llm_backend_;
    std::transform(
      backend.begin(), backend.end(), backend.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return backend == "openai" || backend == "chatgpt";
  }

  std::string activeRequestUrl() const
  {
    return usingOpenAiBackend() ? openai_url_ : vllm_url_;
  }

  std::string activeModelName() const
  {
    return usingOpenAiBackend() ? openai_model_ : vllm_model_;
  }

  std::string activeApiKey() const
  {
    return usingOpenAiBackend() ? openai_api_key_ : vllm_api_key_;
  }

  std::optional<PlannerRollout> parseRollout(
    const std::string & raw_content,
    const Eigen::Vector3f & current_position) const
  {
    const std::vector<std::string> objects = splitConcatenatedJsonObjects(raw_content);
    if (objects.empty()) {
      RCLCPP_ERROR(node_.get_logger(), "LLM response does not contain a JSON object");
      return std::nullopt;
    }

    try {
      const auto parsed = json::parse(objects.back());
      const auto & waypoints_json = parsed.at("waypoints");
      if (!waypoints_json.is_array() || waypoints_json.size() != kWaypointCount) {
        throw std::runtime_error("waypoints must be an array of length 5");
      }

      PlannerRollout rollout;
      rollout.reasoning = parsed.value("reasoning", "");
      rollout.selected_index = parsed.value("selected_waypoint_index", 0);
      if (rollout.selected_index < 0 || rollout.selected_index >= kWaypointCount) {
        throw std::runtime_error("selected_waypoint_index out of range");
      }

      for (const auto & waypoint_json : waypoints_json) {
        Waypoint waypoint;
        const float dx = waypoint_json.at("x").get<float>();
        const float dy = waypoint_json.at("y").get<float>();
        waypoint.x = current_position.x() + dx;
        waypoint.y = current_position.y() + dy;
        waypoint.z = waypoint_json.contains("z") ? waypoint_json.at("z").get<float>() : goal_ned_.z();
        rollout.waypoints.push_back(waypoint);
      }
      return rollout;
    } catch (const std::exception & exc) {
      RCLCPP_ERROR(node_.get_logger(), "Failed to parse planner rollout: %s", exc.what());
      return std::nullopt;
    }
  }

  void llmWorkerLoop()
  {
    const auto sleep_period = std::chrono::duration<double>(1.0 / std::max(0.1, update_rate_hz_));
    while (!stop_worker_.load()) {
      if (!local_position_->positionXYValid() || !local_position_->positionZValid()) {
        std::this_thread::sleep_for(500ms);
        continue;
      }

      const Eigen::Vector3f current_position = local_position_->positionNed();
      const Eigen::Vector3f current_velocity = local_position_->velocityNed();

      if (goalReached(current_position)) {
        std::this_thread::sleep_for(sleep_period);
        continue;
      }

      std::vector<Eigen::Vector3f> obstacle_points;
      {
        std::lock_guard<std::mutex> lock(shared_state_.mutex);
        obstacle_points = shared_state_.obstacle_points_ned;
      }

      const std::string user_prompt = buildUserPrompt(current_position, current_velocity, obstacle_points);
      const auto raw_response = requestPlannerResponse(user_prompt);
      if (!raw_response) {
        std::this_thread::sleep_for(sleep_period);
        continue;
      }

      const auto rollout = parseRollout(*raw_response, current_position);
      if (!rollout) {
        std::this_thread::sleep_for(sleep_period);
        continue;
      }

      PlannerRollout accepted = *rollout;
      {
        std::lock_guard<std::mutex> lock(shared_state_.mutex);
        accepted.rollout_id = shared_state_.next_rollout_id++;
        shared_state_.latest_rollout = accepted;
      }
      RCLCPP_INFO(
        node_.get_logger(),
        "Accepted rollout %d from LLM with %zu waypoints",
        accepted.rollout_id, accepted.waypoints.size());
      if (!accepted.reasoning.empty()) {
        RCLCPP_INFO(node_.get_logger(), "LLM reasoning: %s", accepted.reasoning.c_str());
      }

      std::this_thread::sleep_for(sleep_period);
    }
  }

  rclcpp::Node & node_;
  std::shared_ptr<px4_ros2::MulticopterGotoSetpointType> goto_setpoint_;
  std::shared_ptr<px4_ros2::OdometryLocalPosition> local_position_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr llm_traj_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr llm_path_pub_;

  SharedState shared_state_;
  std::mutex execution_mutex_;
  std::deque<Waypoint> pending_waypoints_;
  std::optional<Waypoint> active_waypoint_;
  std::optional<Eigen::Vector3f> hold_position_ned_;
  int applied_rollout_id_{0};

  std::thread worker_thread_;
  std::atomic<bool> stop_worker_{false};

  float goal_x_{};
  float goal_y_{};
  float goal_z_{};
  Eigen::Vector3f goal_ned_{};
  double update_rate_hz_{1.0};
  float max_horizontal_speed_mps_{6.0f};
  float max_vertical_speed_mps_{2.0f};
  float waypoint_acceptance_radius_m_{0.75f};
  float verification_safety_radius_m_{0.60f};
  float verification_max_velocity_mps_{15.0f};
  float verification_max_accel_mps2_{12.0f};
  int depth_obstacle_samples_{kDepthSampleCountDefault};
  std::string goal_frame_{"ned"};
  std::string llm_backend_{"vllm"};
  std::string vllm_url_;
  std::string vllm_model_;
  std::string vllm_api_key_;
  std::string openai_url_;
  std::string openai_model_;
  std::string openai_api_key_;
  double vllm_temperature_{0.3};
  int vllm_max_tokens_{256};
  std::string prompt_file_;
  std::string system_prompt_;
  float heading_rad_{std::numeric_limits<float>::quiet_NaN()};
};

class LlmGotoPlannerExecutor : public px4_ros2::ModeExecutorBase
{
public:
  explicit LlmGotoPlannerExecutor(px4_ros2::ModeBase & owned_mode)
  : ModeExecutorBase(
      []() {
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
    RCLCPP_INFO(
      node_.get_logger(),
      "Executor activated, waiting for arming checks before arm and takeoff");
    waitReadyToArm([this](px4_ros2::Result ready_result) { onReadyToArm(ready_result); });
  }

  void onDeactivate(DeactivateReason reason) override
  {
    RCLCPP_INFO(node_.get_logger(), "Executor deactivated (%d)", static_cast<int>(reason));
  }

private:
  void onReadyToArm(px4_ros2::Result result)
  {
    if (result != px4_ros2::Result::Success) {
      RCLCPP_ERROR(node_.get_logger(), "Vehicle not ready to arm: %s", resultToString(result));
      return;
    }
    RCLCPP_INFO(node_.get_logger(), "Arming checks passed. Arming vehicle.");
    arm([this](px4_ros2::Result arm_result) { onArmed(arm_result); });
  }

  void onArmed(px4_ros2::Result result)
  {
    if (result != px4_ros2::Result::Success) {
      RCLCPP_ERROR(node_.get_logger(), "Arm failed: %s", resultToString(result));
      return;
    }
    RCLCPP_INFO(node_.get_logger(), "Armed. Starting takeoff.");
    takeoff([this](px4_ros2::Result takeoff_result) { onTakeoffCompleted(takeoff_result); });
  }

  void onTakeoffCompleted(px4_ros2::Result result)
  {
    if (result != px4_ros2::Result::Success) {
      RCLCPP_ERROR(node_.get_logger(), "Takeoff failed: %s", resultToString(result));
      return;
    }
    RCLCPP_INFO(node_.get_logger(), "Takeoff complete. Handing control to LLM goto mode.");
    scheduleMode(ownedMode().id(), [this](px4_ros2::Result mode_result) {
      RCLCPP_INFO(node_.get_logger(), "LLM goto mode finished with result: %s", resultToString(mode_result));
    });
  }

  rclcpp::Node & node_;
};

}  // namespace

int main(int argc, char ** argv)
{
  curl_global_init(CURL_GLOBAL_DEFAULT);
  rclcpp::init(argc, argv);
  using PlannerNode = px4_ros2::NodeWithModeExecutor<LlmGotoPlannerExecutor, LlmGotoPlannerMode>;
  rclcpp::spin(std::make_shared<PlannerNode>(kNodeName, true));
  rclcpp::shutdown();
  curl_global_cleanup();
  return 0;
}
