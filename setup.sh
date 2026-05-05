#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${OAK_VENV_DIR:-$ROOT/.venv}"
NATIVE="$ROOT/.build/oak-native"
SRC="$NATIVE/envpool-src"
PYBOOT="$(command -v python3.11 || command -v python3 || true)"

[ -n "$PYBOOT" ] || { echo "Python 3.11 not found." >&2; exit 1; }
[ -x "$VENV/bin/python" ] || "$PYBOOT" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install -r "$ROOT/requirements.txt"

MUJOCO_DIR="$("$PY" -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).resolve().parent)')"
MUJOCO_LIB="$("$PY" -c 'import mujoco, pathlib; libs=sorted(pathlib.Path(mujoco.__file__).resolve().parent.glob("libmujoco*.dylib")); print(libs[0] if libs else "")')"
[ -n "$MUJOCO_LIB" ] || { echo "Missing MuJoCo dylib." >&2; exit 1; }

mkdir -p "$ROOT/.build" "$NATIVE/src" "$NATIVE/deps" "$NATIVE/include/absl/log"
[ -d "$SRC/.git" ] || git clone --depth 1 --branch v1.2.0 https://github.com/sail-sg/envpool.git "$SRC"

MUJOCO_LOCAL="$NATIVE/$(basename "$MUJOCO_LIB")"
cp -f "$MUJOCO_LIB" "$MUJOCO_LOCAL"
install_name_tool -id "@loader_path/$(basename "$MUJOCO_LIB")" "$MUJOCO_LOCAL"

"$PY" - "$NATIVE" <<'PY'
import pathlib, sys, textwrap
n = pathlib.Path(sys.argv[1])
(n / "deps/ThreadPool.h").write_text(r'''#pragma once
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
class ThreadPool { std::vector<std::thread> workers; std::queue<std::function<void()>> tasks; std::mutex mutex; std::condition_variable cv; bool stop=false; public: explicit ThreadPool(size_t n){ for(size_t i=0;i<n;++i) workers.emplace_back([this]{ for(;;){ std::function<void()> task; { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock,[this]{return stop||!tasks.empty();}); if(stop&&tasks.empty()) return; task=std::move(tasks.front()); tasks.pop(); } task(); }}); } template<class F,class... A> auto enqueue(F&& f,A&&... a)->std::future<std::invoke_result_t<F,A...>>{ using R=std::invoke_result_t<F,A...>; auto task=std::make_shared<std::packaged_task<R()>>(std::bind(std::forward<F>(f),std::forward<A>(a)...)); auto out=task->get_future(); { std::lock_guard<std::mutex> lock(mutex); if(stop) throw std::runtime_error("enqueue on stopped ThreadPool"); tasks.emplace([task]{(*task)();}); } cv.notify_one(); return out; } ~ThreadPool(){ { std::lock_guard<std::mutex> lock(mutex); stop=true; } cv.notify_all(); for(auto& w:workers) w.join(); } };
''')
(n / "deps/lightweightsemaphore.h").write_text(r'''#pragma once
#include <condition_variable>
#include <cstddef>
#include <mutex>
namespace moodycamel { class LightweightSemaphore { std::mutex m; std::condition_variable cv; std::ptrdiff_t count; public: explicit LightweightSemaphore(std::ptrdiff_t c=0):count(c){} bool wait(){ std::unique_lock<std::mutex> lock(m); cv.wait(lock,[&]{return count>0;}); --count; return true; } bool tryWait(){ std::lock_guard<std::mutex> lock(m); if(count<=0) return false; --count; return true; } void signal(std::ptrdiff_t n=1){ std::lock_guard<std::mutex> lock(m); count+=n; for(std::ptrdiff_t i=0;i<n;++i) cv.notify_one(); } }; }
''')
(n / "include/absl/log/check.h").write_text(r'''#pragma once
#include <cstdlib>
#include <iostream>
namespace oak_native { class CheckStream { bool ok; public: explicit CheckStream(bool v):ok(v){} ~CheckStream(){ if(!ok){ std::cerr<<std::endl; std::abort(); } } template<class T> CheckStream& operator<<(const T& x){ if(!ok) std::cerr<<x; return *this; } }; }
#define CHECK(x) ::oak_native::CheckStream(static_cast<bool>(x))
#define DCHECK(x) CHECK(x)
#define CHECK_EQ(a,b) CHECK((a)==(b))
#define CHECK_NE(a,b) CHECK((a)!=(b))
#define CHECK_LT(a,b) CHECK((a)<(b))
#define CHECK_LE(a,b) CHECK((a)<=(b))
#define CHECK_GT(a,b) CHECK((a)>(b))
#define CHECK_GE(a,b) CHECK((a)>=(b))
#define DCHECK_EQ(a,b) CHECK_EQ(a,b)
#define DCHECK_NE(a,b) CHECK_NE(a,b)
#define DCHECK_LT(a,b) CHECK_LT(a,b)
#define DCHECK_LE(a,b) CHECK_LE(a,b)
#define DCHECK_GT(a,b) CHECK_GT(a,b)
#define DCHECK_GE(a,b) CHECK_GE(a,b)
''')
(n / "include/absl/log/log.h").write_text(r'''#pragma once
#include <iostream>
namespace oak_native { class LogStream { public: template<class T> LogStream& operator<<(const T& x){ std::cerr<<x; return *this; } ~LogStream(){ std::cerr<<std::endl; } }; }
#define LOG(level) ::oak_native::LogStream()
#define DLOG(level) ::oak_native::LogStream()
#define INFO info
''')
(n / "src/oak_env_bridge.cc").write_text(textwrap.dedent(r'''
  #include <algorithm>
  #include <cmath>
  #include <cstdint>
  #include <cstring>
  #include <memory>
  #include <random>
  #include <stdexcept>
  #include <string>
  #include <vector>
  #include <mujoco/mujoco.h>
  #include "envpool/core/array.h"
  #include "envpool/mujoco/gym/half_cheetah.h"
  #include "envpool/mujoco/offscreen_renderer.h"

  namespace envpool::mujoco {
  std::shared_ptr<GlContext> CreateGlContext() { return nullptr; }
  OffscreenRenderer::OffscreenRenderer(CameraPolicy camera_policy) : camera_policy_(camera_policy) {}
  OffscreenRenderer::~OffscreenRenderer() = default;
  void OffscreenRenderer::Initialize(const mjModel*) { initialized_ = true; }
  void OffscreenRenderer::UpdateCamera(const mjModel*, const mjData*, int, const mjvCamera*) {}
  void OffscreenRenderer::Render(const mjModel*, mjData*, int w, int h, int, unsigned char* rgb, const mjvCamera*) { if (rgb) std::memset(rgb, 0, static_cast<size_t>(w) * static_cast<size_t>(h) * 3); }
  }

  namespace {
  template <typename T> Array A(std::size_t n) { return Array(Spec<T>({static_cast<int>(n)})); }
  template <typename T> Array A(std::size_t r, std::size_t c) { return Array(Spec<T>({static_cast<int>(r), static_cast<int>(c)})); }
  std::string& GlobalError() { static thread_local std::string e; return e; }
  std::string& Warnings() { static thread_local std::string w; return w; }
  void OakWarn(const char* m) { if (m && *m) { auto& w = Warnings(); if (!w.empty()) w.push_back('\n'); w += m; } }
  float safe(float x) { return std::isfinite(x) ? x : 0.0f; }
  mujoco_gym::HalfCheetahEnvSpec Spec(int seed, const std::string& base) {
    auto c = mujoco_gym::HalfCheetahEnvSpec::kDefaultConfig;
    c["num_envs"_] = 1; c["batch_size"_] = 1; c["seed"_] = seed; c["base_path"_] = base;
    return mujoco_gym::HalfCheetahEnvSpec(c.AllValues());
  }
  class HCEnv final {
    mujoco_gym::HalfCheetahEnvPool pool_;
    int act_dim_, d_policy_, option_count_ = 0, steps_ = 0, done_ = 0;
    float reward_ = 0.0f; double delta_sum_ = 0.0;
    Array env_ids_, player_ids_, action_;
    std::vector<const char*> info_names_;
    std::vector<double> info_values_;
    std::vector<float> low_, high_, obs_, env_action_, action_signal_, option_credit_, importance_, logp_, last_policy_, descriptor_;
    std::mt19937 rng_;
    std::normal_distribution<float> normal_{0.0f, 1.0f};
    std::uniform_real_distribution<float> uniform_{0.0f, 1.0f};
    void copy_obs(const std::vector<Array>& recv) { auto src = TArray<mjtNum>(recv[8]); for (int i = 0; i < obs_dim(); ++i) obs_[i] = static_cast<float>(src(0, i)); }
    void copy_reward_done(const std::vector<Array>& recv) { auto r = TArray<float>(recv[4]); auto d = TArray<bool>(recv[3]); reward_ = r[0]; done_ = d[0] ? 1 : 0; }
    void copy_info(const std::vector<Array>& recv) { auto run = TArray<mjtNum>(recv[9]); auto ctrl = TArray<mjtNum>(recv[10]); auto pos = TArray<mjtNum>(recv[11]); auto vel = TArray<mjtNum>(recv[12]); info_values_[0] = run[0]; info_values_[1] = ctrl[0]; info_values_[2] = pos[0]; info_values_[3] = vel[0]; }
    void send() { auto* dst = static_cast<mjtNum*>(action_.Data()); for (int i = 0; i < act_dim_; ++i) dst[i] = static_cast<mjtNum>(env_action_[i]); pool_.Send(std::vector<Array>{env_ids_, player_ids_, action_}); auto recv = pool_.Recv(); copy_obs(recv); copy_reward_done(recv); copy_info(recv); }
    void delta(const std::vector<float>& x) { double d = 0.0; for (int i = 0; i < d_policy_; ++i) { double v = double(x[i]) - double(last_policy_[i]); d += v * v; last_policy_[i] = x[i]; } delta_sum_ += d / double(std::max(d_policy_, 1)); }
   public:
    HCEnv(int seed, const std::string& base) : pool_(Spec(seed, base)), act_dim_(pool_.spec.action_spec["action"_].shape.back()), d_policy_(2 * act_dim_), env_ids_(A<int>(1)), player_ids_(A<int>(1)), action_(A<mjtNum>(1, act_dim_)), info_names_{"info:reward_run", "info:reward_ctrl", "info:x_position", "info:x_velocity"}, info_values_(info_names_.size()), obs_(obs_dim()), env_action_(act_dim_), action_signal_(d_policy_), last_policy_(d_policy_), descriptor_(d_policy_), rng_(static_cast<uint32_t>(seed)) {
      static_cast<int*>(env_ids_.Data())[0] = 0; static_cast<int*>(player_ids_.Data())[0] = 0;
      auto b = pool_.spec.action_spec["action"_].bounds; low_.assign(act_dim_, static_cast<float>(std::get<0>(b))); high_.assign(act_dim_, static_cast<float>(std::get<1>(b)));
    }
    int obs_dim() const { return pool_.spec.state_spec["obs"_].shape.back(); }
    int act_dim() const { return act_dim_; }
    int d_policy() const { return d_policy_; }
    const float* obs() const { return obs_.data(); }
    const float* action_signal() const { return action_signal_.data(); }
    const float* option_credit() const { return option_credit_.data(); }
    const float* importance() const { return importance_.data(); }
    int info_count() const { return static_cast<int>(info_names_.size()); }
    const char* const* info_names() const { return info_names_.data(); }
    const double* info_values() const { return info_values_.data(); }
    float reward() const { return reward_; }
    int done() const { return done_; }
    double action_delta_mean() const { return steps_ > 0 ? delta_sum_ / steps_ : 0.0; }
    void prepare(int k) { option_count_ = std::max(k, 0); option_credit_.assign(option_count_ * d_policy_, 0.0f); importance_.assign(option_count_, 1.0f); logp_.assign(option_count_, 0.0f); }
    void reset() { pool_.Reset(env_ids_); auto recv = pool_.Recv(); copy_obs(recv); copy_info(recv); reward_ = 0.0f; done_ = 0; steps_ = 0; delta_sum_ = 0.0f; std::fill(last_policy_.begin(), last_policy_.end(), 0.0f); std::fill(action_signal_.begin(), action_signal_.end(), 0.0f); std::fill(option_credit_.begin(), option_credit_.end(), 0.0f); std::fill(importance_.begin(), importance_.end(), 1.0f); }
    void step_zero() { std::fill(env_action_.begin(), env_action_.end(), 0.0f); std::fill(action_signal_.begin(), action_signal_.end(), 0.0f); std::fill(descriptor_.begin(), descriptor_.end(), 0.0f); delta(descriptor_); ++steps_; send(); }
    void step_random() { for (int d = 0; d < act_dim_; ++d) { float x = low_[d] + (high_[d] - low_[d]) * uniform_(rng_); env_action_[d] = x; action_signal_[d] = x; action_signal_[act_dim_ + d] = 0.0f; descriptor_[d] = x; descriptor_[act_dim_ + d] = 0.0f; } delta(descriptor_); ++steps_; send(); }
    void step_oak(const float* policy, const float* options, int k) {
      if (k != option_count_) prepare(k);
      float sd = 1.0f / std::max(d_policy_, 1), log_sigma = -std::log(sd), behavior = 0.0f;
      std::fill(logp_.begin(), logp_.end(), 0.0f); std::fill(importance_.begin(), importance_.end(), 0.0f);
      for (int i = 0; i < d_policy_; ++i) descriptor_[i] = safe(policy ? policy[i] : 0.0f);
      for (int d = 0; d < act_dim_; ++d) {
        float mu = descriptor_[d], z = normal_(rng_), sample = safe(mu + sd * z);
        env_action_[d] = std::max(low_[d], std::min(high_[d], sample)); action_signal_[d] = env_action_[d]; action_signal_[act_dim_ + d] = 0.0f; behavior += log_sigma - 0.5f * z * z;
        for (int o = 0; o < option_count_; ++o) { int off = o * d_policy_; float om = safe(options ? options[off + d] : 0.0f), oz = (sample - om) / sd; logp_[o] += log_sigma - 0.5f * oz * oz; option_credit_[off + d] = sample - om; option_credit_[off + act_dim_ + d] = 0.0f; }
      }
      for (int o = 0; o < option_count_; ++o) importance_[o] = std::exp(logp_[o] - behavior);
      delta(descriptor_); ++steps_; send();
    }
  };
  struct Handle { std::unique_ptr<HCEnv> env; std::string error; };
  struct View { int obs_dim, act_dim, d_policy, done; float reward; double action_delta_mean; int info_count; const char* const* info_names; const double* info_values; const float *obs, *action_signal, *option_credit, *importance; };
  template <class F> int call(void* raw, F f) { auto* h = static_cast<Handle*>(raw); try { h->error.clear(); f(*h->env); if (!Warnings().empty()) { h->error = Warnings(); Warnings().clear(); return 0; } return 1; } catch (const std::exception& e) { h->error = e.what(); return 0; } }
  }
  extern "C" {
  void* oak_env_open(int seed, const char* env_id, const char* base) { try { if (!env_id || std::string(env_id) != "HalfCheetah-v4") throw std::runtime_error("unsupported env id"); if (!base) throw std::runtime_error("base_path is required"); mju_user_warning = OakWarn; auto* h = new Handle(); h->env = std::make_unique<HCEnv>(seed, base); return h; } catch (const std::exception& e) { GlobalError() = e.what(); return nullptr; } }
  void oak_env_close(void* raw) { delete static_cast<Handle*>(raw); }
  const char* oak_env_error(void* raw) { return raw ? static_cast<Handle*>(raw)->error.c_str() : GlobalError().c_str(); }
  int oak_env_prepare(void* raw, int k) { return call(raw, [k](HCEnv& e){ e.prepare(k); }); }
  int oak_env_reset(void* raw) { return call(raw, [](HCEnv& e){ e.reset(); }); }
  int oak_env_step_random(void* raw) { return call(raw, [](HCEnv& e){ e.step_random(); }); }
  int oak_env_step_zero(void* raw) { return call(raw, [](HCEnv& e){ e.step_zero(); }); }
  int oak_env_step_oak(void* raw, const float* p, const float* o, int k) { return call(raw, [p,o,k](HCEnv& e){ e.step_oak(p, o, k); }); }
  int oak_env_view(void* raw, View* v) { if (!raw || !v) return 0; auto& e = *static_cast<Handle*>(raw)->env; *v = View{e.obs_dim(), e.act_dim(), e.d_policy(), e.done(), e.reward(), e.action_delta_mean(), e.info_count(), e.info_names(), e.info_values(), e.obs(), e.action_signal(), e.option_credit(), e.importance()}; return 1; }
  }
  ''').strip() + "\n")
PY

ln -sfn "$("$PY" -c 'import envpool, pathlib; print(pathlib.Path(envpool.__file__).resolve().parent)')" "$NATIVE/env-assets"
/usr/bin/clang++ -std=c++20 -O3 -fPIC -dynamiclib "$NATIVE/src/oak_env_bridge.cc" \
  -I"$SRC" -I"$NATIVE/include" -I"$NATIVE/deps" \
  -I"$MUJOCO_DIR/include" -I"$MUJOCO_DIR/include/mujoco" \
  -I/opt/homebrew/include/eigen3 -I/opt/homebrew/include \
  "$MUJOCO_LOCAL" -Wno-deprecated-this-capture -Wno-gnu-string-literal-operator-template -Wno-return-type \
  -Wl,-rpath,@loader_path -Wl,-rpath,"$MUJOCO_DIR" -o "$NATIVE/liboakenv.dylib"

xcrun -sdk macosx metal -fno-fast-math -c "$ROOT/Sources/OakKernel.metal" -o "$ROOT/.build/OakKernel.air"
xcrun -sdk macosx metallib "$ROOT/.build/OakKernel.air" -o "$ROOT/.build/OakKernel.metallib"
swift build -c release
echo "Setup complete. ./benchmark.sh"
