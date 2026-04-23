#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${OAK_VENV_DIR:-$ROOT_DIR/.venv}"
NATIVE_DIR="$ROOT_DIR/.build/oak-native"
SRC_DIR="$NATIVE_DIR/envpool-src"
REQS="$ROOT_DIR/requirements.txt"

BOOT="$(command -v python3.11 || command -v python3 || true)"
[ -n "$BOOT" ] || { echo "Python 3.11 not found." >&2; exit 1; }

[ -f "$REQS" ] || { echo "Missing $REQS" >&2; exit 1; }
[ -x "$VENV_DIR/bin/python" ] || "$BOOT" -m venv "$VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install -r "$REQS"
MUJOCO_DIR="$("$PYTHON" -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).resolve().parent)' )"
MUJOCO_LIB="$("$PYTHON" -c 'import mujoco, pathlib; libs=sorted(pathlib.Path(mujoco.__file__).resolve().parent.glob("libmujoco*.dylib")); print(libs[0] if libs else "")' )"
[ -n "$MUJOCO_LIB" ] || { echo "Missing libmujoco dylib in $MUJOCO_DIR" >&2; exit 1; }

# Pip wheels ship libmujoco with LC_ID_DYLIB @rpath/mujoco.framework/... but no framework on disk.
# Copy next to liboakenv and retag so @loader_path resolves at runtime.
mkdir -p "$NATIVE_DIR"
MUJOCO_LIB_LOCAL="$NATIVE_DIR/$(basename "$MUJOCO_LIB")"
cp -f "$MUJOCO_LIB" "$MUJOCO_LIB_LOCAL"
install_name_tool -id "@loader_path/$(basename "$MUJOCO_LIB")" "$MUJOCO_LIB_LOCAL"

mkdir -p "$ROOT_DIR/.build" "$NATIVE_DIR"/{src,generated,deps,include/absl/log}
[ -d "$SRC_DIR/.git" ] || git clone --depth 1 --branch v1.2.0 https://github.com/sail-sg/envpool.git "$SRC_DIR"

"$PYTHON" - "$NATIVE_DIR" <<'PY'
import os, sys, pathlib, textwrap
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import envpool.minigrid.registration, envpool.registration as reg
native = pathlib.Path(sys.argv[1]); generated = native / "generated"; src = native / "src"
files = {
    native / "deps/ThreadPool.h": r'''#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
class ThreadPool { std::vector<std::thread> workers; std::queue<std::function<void()>> tasks; std::mutex queue_mutex; std::condition_variable condition; bool stop; public: ThreadPool(size_t); template<class F, class... Args> auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>; ~ThreadPool(); };
inline ThreadPool::ThreadPool(size_t threads) : stop(false) { for (size_t i = 0; i < threads; ++i) workers.emplace_back([this] { for (;;) { std::function<void()> task; { std::unique_lock<std::mutex> lock(this->queue_mutex); this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); }); if (this->stop && this->tasks.empty()) return; task = std::move(this->tasks.front()); this->tasks.pop(); } task(); } }); }
template<class F, class... Args> auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> { using return_type = std::invoke_result_t<F, Args...>; auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...)); std::future<return_type> res = task->get_future(); { std::unique_lock<std::mutex> lock(queue_mutex); if (stop) throw std::runtime_error("enqueue on stopped ThreadPool"); tasks.emplace([task]() { (*task)(); }); } condition.notify_one(); return res; }
inline ThreadPool::~ThreadPool() { { std::unique_lock<std::mutex> lock(queue_mutex); stop = true; } condition.notify_all(); for (std::thread& worker : workers) worker.join(); }
#endif
''',
    native / "deps/lightweightsemaphore.h": r'''#pragma once
#include <condition_variable>
#include <cstddef>
#include <mutex>
namespace moodycamel { class LightweightSemaphore { std::mutex mutex_; std::condition_variable condition_; std::ptrdiff_t count_; public: explicit LightweightSemaphore(std::ptrdiff_t initial = 0) : count_(initial) {} bool wait() { std::unique_lock<std::mutex> lock(mutex_); condition_.wait(lock, [&] { return count_ > 0; }); --count_; return true; } bool tryWait() { std::lock_guard<std::mutex> lock(mutex_); if (count_ <= 0) return false; --count_; return true; } void signal(std::ptrdiff_t count = 1) { std::lock_guard<std::mutex> lock(mutex_); count_ += count; for (std::ptrdiff_t i = 0; i < count; ++i) condition_.notify_one(); } }; }
''',
    native / "include/absl/log/check.h": r'''#pragma once
#include <cstdlib>
#include <iostream>
namespace oak_native { class CheckStream { bool ok_; public: explicit CheckStream(bool ok): ok_(ok) {} ~CheckStream() { if (!ok_) { std::cerr << std::endl; std::abort(); } } template <typename T> CheckStream& operator<<(const T& value) { if (!ok_) std::cerr << value; return *this; } }; }
#define CHECK(expr) ::oak_native::CheckStream(static_cast<bool>(expr))
#define DCHECK(expr) CHECK(expr)
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
''',
    native / "include/absl/log/log.h": r'''#pragma once
#include <iostream>
namespace oak_native { class LogStream { public: template <typename T> LogStream& operator<<(const T& value) { std::cerr << value; return *this; } ~LogStream() { std::cerr << std::endl; } }; }
#define LOG(level) ::oak_native::LogStream()
#define DLOG(level) ::oak_native::LogStream()
#define INFO info
'''
}
for path, text in files.items(): path.write_text(text.strip() + "\n")
specs = reg.registry.specs
minigrids = sorted(k for k in specs if k.startswith("MiniGrid-"))
def cpp(value):
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, int): return str(value)
    if isinstance(value, float): return repr(value) + "f"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'std::string("{escaped}")'
    if isinstance(value, tuple) and len(value) == 2: return f"std::pair<int, int>({value[0]}, {value[1]})"
    raise TypeError((type(value), value))
lines = ['inline bool ApplyMiniGridTaskConfig(const std::string& env_id, minigrid::MiniGridEnvSpec::Config* config) {']
for env_id in minigrids:
    lines.append(f'  if (env_id == "{env_id}") {{')
    lines.append('    (*config)["agent_view_size"_] = 7;')
    lines.append('    (*config)["mission_bytes"_] = 96;')
    for key, value in specs[env_id][2].items():
        if key != "base_path": lines.append(f'    (*config)["{key}"_] = {cpp(value)};')
    lines += ['    return true;', '  }']
lines += ['  return false;', '}', 'inline std::vector<std::string> BenchmarkMiniGridEnvIds() { return {']
lines += [f'  "{env_id}",' for env_id in minigrids]
lines += ['}; }']
(generated / "minigrid_registry.inc").write_text("\n".join(lines) + "\n")
(src / "render_stubs.cc").write_text(textwrap.dedent(r'''
  #include <algorithm>
  #include <cstring>
  #include <memory>
  #include <stdexcept>
  #include <utility>
  #include <vector>
  #include "envpool/minigrid/impl/minigrid_env.h"
  #include "envpool/mujoco/offscreen_renderer.h"
  namespace minigrid {
  namespace { inline int GridOffset(int x, int y, int height) { return (x * height + y) * 3; } }
  void MiniGridTask::GenImage(const Array& obs) const {
    obs.Zero(); int top_x = 0, top_y = 0;
    if (agent_dir_ == 0) { top_x = agent_pos_.first; top_y = agent_pos_.second - agent_view_size_ / 2; }
    else if (agent_dir_ == 1) { top_x = agent_pos_.first - agent_view_size_ / 2; top_y = agent_pos_.second; }
    else if (agent_dir_ == 2) { top_x = agent_pos_.first - agent_view_size_ + 1; top_y = agent_pos_.second - agent_view_size_ / 2; }
    else { top_x = agent_pos_.first - agent_view_size_ / 2; top_y = agent_pos_.second - agent_view_size_ + 1; }
    std::vector<std::vector<WorldObj>> view(agent_view_size_, std::vector<WorldObj>(agent_view_size_, WorldObj(kWall)));
    for (int y = 0; y < agent_view_size_; ++y) for (int x = 0; x < agent_view_size_; ++x) {
      int gx = top_x + x, gy = top_y + y; if (InBounds(gx, gy)) view[y][x] = GetCell(gx, gy);
    }
    for (int i = 0; i < agent_dir_ + 1; ++i) {
      std::vector<std::vector<WorldObj>> rotated = view;
      for (int y = 0; y < agent_view_size_; ++y) for (int x = 0; x < agent_view_size_; ++x) rotated[agent_view_size_ - 1 - x][y] = view[y][x];
      view = std::move(rotated);
    }
    std::vector<std::vector<bool>> vis_mask(agent_view_size_, std::vector<bool>(agent_view_size_, false));
    int agent_x = agent_view_size_ / 2, agent_y = agent_view_size_ - 1;
    if (see_through_walls_) for (auto& col : vis_mask) std::fill(col.begin(), col.end(), true);
    else {
      vis_mask[agent_x][agent_y] = true;
      for (int y = agent_view_size_ - 1; y >= 0; --y) {
        for (int x = 0; x < agent_view_size_ - 1; ++x) if (vis_mask[x][y] && view[y][x].CanSeeBehind()) {
          vis_mask[x + 1][y] = true; if (y > 0) { vis_mask[x + 1][y - 1] = true; vis_mask[x][y - 1] = true; }
        }
        for (int x = agent_view_size_ - 1; x >= 1; --x) if (vis_mask[x][y] && view[y][x].CanSeeBehind()) {
          vis_mask[x - 1][y] = true; if (y > 0) { vis_mask[x - 1][y - 1] = true; vis_mask[x][y - 1] = true; }
        }
      }
    }
    view[agent_y][agent_x] = carrying_.GetType() == kEmpty ? WorldObj(kEmpty) : carrying_;
    for (int y = 0; y < agent_view_size_; ++y) for (int x = 0; x < agent_view_size_; ++x) if (vis_mask[x][y]) {
      const auto enc = view[y][x].Encode(); obs(x, y, 0) = enc[0]; obs(x, y, 1) = enc[1]; obs(x, y, 2) = enc[2];
    }
  }
  void MiniGridTask::WriteMission(const Array& obs) const {
    obs.Zero(); auto* data = reinterpret_cast<uint8_t*>(obs.Data());
    int n = std::min(static_cast<int>(mission_.size()), mission_bytes_ - 1); std::memcpy(data, mission_.data(), n);
  }
  MiniGridDebugState MiniGridTask::DebugState() const {
    MiniGridDebugState state; state.env_name = env_name_; state.mission = mission_; state.mission_id = mission_id_; state.width = width_; state.height = height_;
    state.action_max = action_max_; state.max_steps = max_steps_; state.grid.resize(width_ * height_ * 3); state.grid_contains.resize(width_ * height_ * 3, 0);
    for (int x = 0; x < width_; ++x) for (int y = 0; y < height_; ++y) {
      const WorldObj& cell = GetCell(x, y); const auto enc = cell.Encode(); int offset = GridOffset(x, y, height_);
      state.grid[offset + 0] = enc[0]; state.grid[offset + 1] = enc[1]; state.grid[offset + 2] = enc[2];
      if (const WorldObj* contains = cell.GetContains(); contains != nullptr) {
        const auto contains_enc = contains->Encode();
        state.grid_contains[offset + 0] = contains_enc[0]; state.grid_contains[offset + 1] = contains_enc[1]; state.grid_contains[offset + 2] = contains_enc[2];
      }
    }
    state.agent_pos = agent_pos_; state.agent_dir = agent_dir_; state.has_carrying = carrying_.GetType() != kEmpty;
    state.carrying_type = static_cast<int>(carrying_.GetType()); state.carrying_color = static_cast<int>(carrying_.GetColor()); state.carrying_state = carrying_.GetState();
    if (const WorldObj* contains = carrying_.GetContains(); contains != nullptr) {
      state.carrying_has_contains = true; state.carrying_contains_type = static_cast<int>(contains->GetType());
      state.carrying_contains_color = static_cast<int>(contains->GetColor()); state.carrying_contains_state = contains->GetState();
    }
    state.target_pos = target_pos_; state.target_type = static_cast<int>(target_type_); state.target_color = static_cast<int>(target_color_);
    state.move_pos = move_pos_; state.move_type = static_cast<int>(move_type_); state.move_color = static_cast<int>(move_color_);
    state.success_pos = success_pos_; state.failure_pos = failure_pos_; state.goal_pos = goal_pos_; return state;
  }
  std::pair<int, int> MiniGridTask::RenderSize(int width, int height) const { return {width > 0 ? width : 1, height > 0 ? height : 1}; }
  void MiniGridTask::Render(int width, int height, unsigned char* rgb) const { std::memset(rgb, 0, static_cast<size_t>(width) * static_cast<size_t>(height) * 3); }
  }  // namespace minigrid
  namespace envpool::mujoco {
  std::shared_ptr<GlContext> CreateGlContext() { return nullptr; }
  OffscreenRenderer::OffscreenRenderer(CameraPolicy camera_policy) : camera_policy_(camera_policy) {}
  OffscreenRenderer::~OffscreenRenderer() = default;
  void OffscreenRenderer::Initialize(const mjModel* model) { (void)model; initialized_ = true; }
  void OffscreenRenderer::UpdateCamera(const mjModel* model, const mjData* data, int camera_id, const mjvCamera* camera_override) {
    (void)model; (void)data; (void)camera_id; (void)camera_override;
  }
  void OffscreenRenderer::Render(const mjModel* model, mjData* data, int width, int height, int camera_id, unsigned char* rgb, const mjvCamera* camera_override) {
    (void)model; (void)data; (void)camera_id; (void)camera_override;
    if (rgb == nullptr) throw std::runtime_error("rgb buffer is null");
    std::memset(rgb, 0, static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
  }
  }  // namespace envpool::mujoco
  ''').strip() + "\n")
(src / "oak_env_bridge.cc").write_text(textwrap.dedent(r'''
  #include <algorithm>
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <memory>
  #include <stdexcept>
  #include <string>
  #include <utility>
  #include <vector>
  #include <mujoco/mujoco.h>
  #include "envpool/core/array.h"
  #include "envpool/minigrid/minigrid.h"
  #include "envpool/mujoco/gym/half_cheetah.h"
  #include "envpool/mujoco/gym/humanoid.h"
  #include "envpool/mujoco/gym/walker2d.h"
  #include "minigrid_registry.inc"
  namespace {
  constexpr int kMiniGridImageBytes = 7 * 7 * 3;
  constexpr int kMiniGridMissionBytes = 96;
  constexpr int kMiniGridObservationDim = 1 + kMiniGridImageBytes + kMiniGridMissionBytes;
  void CopyError(const std::string& error, char* buffer, int capacity) { if (buffer == nullptr || capacity <= 0) return; std::size_t n = std::min<std::size_t>(error.size(), static_cast<std::size_t>(capacity - 1)); std::memcpy(buffer, error.data(), n); buffer[n] = '\0'; }
  std::string& MujocoWarnings() { static thread_local std::string warnings; return warnings; }
  void OakMujocoWarning(const char* message) { if (message && *message) { auto& warnings = MujocoWarnings(); if (!warnings.empty()) warnings.push_back('\n'); warnings += message; } }
  template <typename T> Array MakeArray(std::size_t size) { return Array(Spec<T>({static_cast<int>(size)})); }
  template <typename T> Array MakeArray(std::size_t rows, std::size_t cols) { return Array(Spec<T>({static_cast<int>(rows), static_cast<int>(cols)})); }
  struct NativeEnv { virtual ~NativeEnv() = default; virtual int obs_dim() const = 0; virtual int act_dim() const = 0; virtual int discrete_count() const = 0; virtual const std::vector<float>& action_low() const = 0; virtual const std::vector<float>& action_high() const = 0; virtual void Reset(float* obs) = 0; virtual void StepContinuous(const float* action, float* obs, float* reward, uint8_t* done) = 0; virtual void StepDiscrete(const int32_t* action, float* obs, float* reward, uint8_t* done) = 0; };
  template <typename Pool> class MujocoWrapper final : public NativeEnv { Pool pool_; int num_envs_, act_dim_; Array env_ids_, player_ids_, action_; std::vector<float> low_, high_; void CopyObs(const std::vector<Array>& recv, float* obs) { auto src = TArray<mjtNum>(recv[8]); for (int row = 0; row < num_envs_; ++row) for (int col = 0; col < obs_dim(); ++col) obs[row * obs_dim() + col] = static_cast<float>(src(row, col)); } void CopyRewardDone(const std::vector<Array>& recv, float* reward, uint8_t* done) { auto rewards = TArray<float>(recv[4]); auto dones = TArray<bool>(recv[3]); for (int row = 0; row < num_envs_; ++row) { reward[row] = rewards[row]; done[row] = dones[row] ? 1 : 0; } } public: explicit MujocoWrapper(const typename Pool::Spec& spec) : pool_(spec), num_envs_(pool_.spec.config["num_envs"_]), act_dim_(pool_.spec.action_spec["action"_].shape.back()), env_ids_(MakeArray<int>(num_envs_)), player_ids_(MakeArray<int>(num_envs_)), action_(MakeArray<mjtNum>(num_envs_, act_dim_)) { auto bounds = pool_.spec.action_spec["action"_].bounds; low_.assign(act_dim_, static_cast<float>(std::get<0>(bounds))); high_.assign(act_dim_, static_cast<float>(std::get<1>(bounds))); auto* ids = static_cast<int*>(env_ids_.Data()); auto* players = static_cast<int*>(player_ids_.Data()); for (int i = 0; i < num_envs_; ++i) ids[i] = players[i] = i; } int obs_dim() const override { return pool_.spec.state_spec["obs"_].shape.back(); } int act_dim() const override { return act_dim_; } int discrete_count() const override { return 0; } const std::vector<float>& action_low() const override { return low_; } const std::vector<float>& action_high() const override { return high_; } void Reset(float* obs) override { pool_.Reset(env_ids_); CopyObs(pool_.Recv(), obs); } void StepContinuous(const float* action, float* obs, float* reward, uint8_t* done) override { auto* values = static_cast<mjtNum*>(action_.Data()); for (int i = 0; i < num_envs_ * act_dim_; ++i) values[i] = static_cast<mjtNum>(action[i]); pool_.Send(std::vector<Array>{env_ids_, player_ids_, action_}); auto recv = pool_.Recv(); CopyObs(recv, obs); CopyRewardDone(recv, reward, done); } void StepDiscrete(const int32_t*, float*, float*, uint8_t*) override { throw std::runtime_error("continuous env received discrete action"); } };
  class MiniGridWrapper final : public NativeEnv {
    minigrid::MiniGridEnvPool pool_;
    int num_envs_, obs_dim_, discrete_count_;
    Array env_ids_, player_ids_, action_;
    std::vector<float> empty_;
    void CopyObs(const std::vector<Array>& recv, float* obs) {
      auto direction = TArray<int>(recv[8]);
      auto image = TArray<uint8_t>(recv[9]);
      auto mission = TArray<uint8_t>(recv[10]);
      if (image.ndim < 2 || mission.ndim < 2) throw std::runtime_error("MiniGrid observation contract drifted: expected batched image and mission arrays.");
      if (static_cast<int>(image[0].size) != kMiniGridImageBytes || static_cast<int>(mission[0].size) != kMiniGridMissionBytes) {
        throw std::runtime_error("MiniGrid observation contract drifted: expected direction + 7x7x3 image + 96 mission bytes.");
      }
      for (int row = 0; row < num_envs_; ++row) {
        float* dst = obs + row * obs_dim_;
        std::fill(dst, dst + obs_dim_, 0.0f);
        dst[0] = static_cast<float>(direction[row]);
        auto image_row = image[row];
        auto mission_row = mission[row];
        auto* image_ptr = static_cast<uint8_t*>(image_row.Data());
        auto* mission_ptr = static_cast<uint8_t*>(mission_row.Data());
        for (int index = 0; index < kMiniGridImageBytes; ++index) dst[1 + index] = static_cast<float>(image_ptr[index]);
        for (int index = 0; index < kMiniGridMissionBytes; ++index) dst[1 + kMiniGridImageBytes + index] = static_cast<float>(mission_ptr[index]);
      }
    }
    void CopyRewardDone(const std::vector<Array>& recv, float* reward, uint8_t* done) {
      auto rewards = TArray<float>(recv[4]);
      auto dones = TArray<bool>(recv[3]);
      for (int row = 0; row < num_envs_; ++row) { reward[row] = rewards[row]; done[row] = dones[row] ? 1 : 0; }
    }
   public:
    explicit MiniGridWrapper(const minigrid::MiniGridEnvSpec& spec)
      : pool_(spec), num_envs_(pool_.spec.config["num_envs"_]), obs_dim_(kMiniGridObservationDim), discrete_count_(pool_.spec.config["action_max"_] + 1), env_ids_(MakeArray<int>(num_envs_)), player_ids_(MakeArray<int>(num_envs_)), action_(MakeArray<int>(num_envs_)) {
      auto* ids = static_cast<int*>(env_ids_.Data());
      auto* players = static_cast<int*>(player_ids_.Data());
      for (int i = 0; i < num_envs_; ++i) ids[i] = players[i] = i;
    }
    int obs_dim() const override { return obs_dim_; }
    int act_dim() const override { return discrete_count_; }
    int discrete_count() const override { return discrete_count_; }
    const std::vector<float>& action_low() const override { return empty_; }
    const std::vector<float>& action_high() const override { return empty_; }
    void Reset(float* obs) override { pool_.Reset(env_ids_); CopyObs(pool_.Recv(), obs); }
    void StepContinuous(const float*, float*, float*, uint8_t*) override { throw std::runtime_error("discrete env received continuous action"); }
    void StepDiscrete(const int32_t* action_values, float* obs, float* reward, uint8_t* done) override {
      auto* values = static_cast<int*>(action_.Data());
      for (int row = 0; row < num_envs_; ++row) values[row] = action_values[row];
      pool_.Send(std::vector<Array>{env_ids_, player_ids_, action_});
      auto recv = pool_.Recv();
      CopyObs(recv, obs);
      CopyRewardDone(recv, reward, done);
    }
  };
  template <typename Spec, typename Pool> std::unique_ptr<NativeEnv> OpenMujoco(int num_envs, int seed, const std::string& base_path) { auto config = Spec::kDefaultConfig; config["num_envs"_] = num_envs; config["batch_size"_] = num_envs; config["seed"_] = seed; config["base_path"_] = base_path; return std::make_unique<MujocoWrapper<Pool>>(Spec(config.AllValues())); }
  std::unique_ptr<NativeEnv> OpenEnv(const std::string& env_id, int num_envs, int seed, const std::string& base_path) { mju_user_warning = OakMujocoWarning; if (env_id == "HalfCheetah-v4") return OpenMujoco<mujoco_gym::HalfCheetahEnvSpec, mujoco_gym::HalfCheetahEnvPool>(num_envs, seed, base_path); if (env_id == "Humanoid-v4") return OpenMujoco<mujoco_gym::HumanoidEnvSpec, mujoco_gym::HumanoidEnvPool>(num_envs, seed, base_path); if (env_id == "Walker2d-v4") return OpenMujoco<mujoco_gym::Walker2dEnvSpec, mujoco_gym::Walker2dEnvPool>(num_envs, seed, base_path); auto config = minigrid::MiniGridEnvSpec::kDefaultConfig; if (!ApplyMiniGridTaskConfig(env_id, &config)) throw std::runtime_error("unsupported env id: " + env_id); config["num_envs"_] = num_envs; config["batch_size"_] = num_envs; config["seed"_] = seed; config["base_path"_] = base_path; return std::make_unique<MiniGridWrapper>(minigrid::MiniGridEnvSpec(config.AllValues())); }
  struct OakEnvHandle { std::unique_ptr<NativeEnv> env; };
  const std::vector<std::string>& BenchmarkEnvIds() { static const std::vector<std::string> ids = [] { auto ids = BenchmarkMiniGridEnvIds(); ids.push_back("HalfCheetah-v4"); ids.push_back("Humanoid-v4"); ids.push_back("Walker2d-v4"); return ids; }(); return ids; }
  }  // namespace
  extern "C" {
  void* oak_env_open(const char* env_id, int num_envs, int seed, const char* base_path, char* error, int error_capacity) { try { if (!env_id || !base_path) throw std::runtime_error("env_id and base_path are required"); auto* handle = new OakEnvHandle(); handle->env = OpenEnv(env_id, num_envs, seed, base_path); return handle; } catch (const std::exception& ex) { CopyError(ex.what(), error, error_capacity); return nullptr; } }
  void oak_env_close(void* raw) { delete static_cast<OakEnvHandle*>(raw); }
  int oak_env_obs_dim(void* raw) { return static_cast<OakEnvHandle*>(raw)->env->obs_dim(); }
  int oak_env_act_dim(void* raw) { return static_cast<OakEnvHandle*>(raw)->env->act_dim(); }
  int oak_env_discrete_count(void* raw) { return static_cast<OakEnvHandle*>(raw)->env->discrete_count(); }
  int oak_env_action_bounds(void* raw, float* low, float* high) { auto* env = static_cast<OakEnvHandle*>(raw)->env.get(); const auto& lows = env->action_low(); const auto& highs = env->action_high(); if (lows.empty() || highs.empty()) return 0; std::memcpy(low, lows.data(), sizeof(float) * lows.size()); std::memcpy(high, highs.data(), sizeof(float) * highs.size()); return static_cast<int>(lows.size()); }
  int oak_env_reset(void* raw, float* obs, char* error, int error_capacity) { try { static_cast<OakEnvHandle*>(raw)->env->Reset(obs); return 1; } catch (const std::exception& ex) { CopyError(ex.what(), error, error_capacity); return 0; } }
  int oak_env_step_continuous(void* raw, const float* action, float* obs, float* reward, uint8_t* done, char* error, int error_capacity) { try { static_cast<OakEnvHandle*>(raw)->env->StepContinuous(action, obs, reward, done); return 1; } catch (const std::exception& ex) { CopyError(ex.what(), error, error_capacity); return 0; } }
  int oak_env_step_discrete(void* raw, const int32_t* action, float* obs, float* reward, uint8_t* done, char* error, int error_capacity) { try { static_cast<OakEnvHandle*>(raw)->env->StepDiscrete(action, obs, reward, done); return 1; } catch (const std::exception& ex) { CopyError(ex.what(), error, error_capacity); return 0; } }
  int oak_env_take_warning(char* error, int error_capacity) { auto& warnings = MujocoWarnings(); if (warnings.empty()) return 0; CopyError(warnings, error, error_capacity); warnings.clear(); return 1; }
  int oak_benchmark_env_count() { return static_cast<int>(BenchmarkEnvIds().size()); }
  const char* oak_benchmark_env_id(int index) { const auto& ids = BenchmarkEnvIds(); return index >= 0 && index < static_cast<int>(ids.size()) ? ids[index].c_str() : nullptr; }
  }
  ''').strip() + "\n")
PY

ln -sfn "$("$PYTHON" -c 'import envpool, pathlib; print(pathlib.Path(envpool.__file__).resolve().parent)')" "$NATIVE_DIR/env-assets"
MINIGRID_SRCS=()
for src in "$SRC_DIR"/envpool/minigrid/impl/*.cc; do case "$(basename "$src")" in minigrid_render.cc|minigrid_empty_env.cc) ;; *) MINIGRID_SRCS+=("$src") ;; esac; done
/usr/bin/clang++ -std=c++20 -O3 -fPIC -dynamiclib "$NATIVE_DIR/src/oak_env_bridge.cc" "$NATIVE_DIR/src/render_stubs.cc" "${MINIGRID_SRCS[@]}" \
  -I"$SRC_DIR" -I"$NATIVE_DIR/generated" -I"$NATIVE_DIR/include" -I"$NATIVE_DIR/deps" \
  -I"$MUJOCO_DIR/include" -I"$MUJOCO_DIR/include/mujoco" \
  -I/opt/homebrew/include/eigen3 -I/opt/homebrew/include \
  "$MUJOCO_LIB_LOCAL" \
  -Wno-deprecated-this-capture -Wno-gnu-string-literal-operator-template -Wno-return-type \
  -Wl,-rpath,@loader_path -Wl,-rpath,"$MUJOCO_DIR" \
  -o "$NATIVE_DIR/liboakenv.dylib"

# Single-source default library. Pass 15 Tier 0: -fno-fast-math forces the Metal compiler
# to preserve isfinite() / isnan() / isinf() checks at -O. Per IREE issue #23914 (March 2026),
# the default fast-math mode silently elides these checks, making 182M+ NaN/Inf events at
# deep workerDepth invisible to the runtime telemetry.  We pay <5% throughput for provable
# numerical-safety on infinite-horizon runs (BWH-orthodox, no clamps, no bias).
xcrun -sdk macosx metal -fno-fast-math -c "$ROOT_DIR/Sources/OakKernel.metal" -o "$ROOT_DIR/.build/OakKernel.air"
xcrun -sdk macosx metallib "$ROOT_DIR/.build/OakKernel.air" -o "$ROOT_DIR/.build/OakKernel.metallib"
swift build -c release

echo "Setup complete. ./benchmark.sh"
