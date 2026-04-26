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
  #include <cstdint>
  #include <cstring>
  #include <memory>
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
  template <typename T> Array MakeArray(std::size_t n) { return Array(Spec<T>({static_cast<int>(n)})); }
  template <typename T> Array MakeArray(std::size_t r, std::size_t c) { return Array(Spec<T>({static_cast<int>(r), static_cast<int>(c)})); }
  void CopyError(const std::string& e, char* b, int cap) { if (!b || cap <= 0) return; auto n = std::min<std::size_t>(e.size(), static_cast<std::size_t>(cap - 1)); std::memcpy(b, e.data(), n); b[n] = 0; }
  std::string& Warnings() { static thread_local std::string w; return w; }
  void OakWarn(const char* m) { if (m && *m) { auto& w = Warnings(); if (!w.empty()) w.push_back('\n'); w += m; } }
  mujoco_gym::HalfCheetahEnvSpec MakeSpec(int seed, const std::string& base) {
    auto c = mujoco_gym::HalfCheetahEnvSpec::kDefaultConfig;
    c["num_envs"_] = 1; c["batch_size"_] = 1; c["seed"_] = seed; c["base_path"_] = base;
    return mujoco_gym::HalfCheetahEnvSpec(c.AllValues());
  }
  class HCEnv final {
    mujoco_gym::HalfCheetahEnvPool pool_;
    int act_dim_;
    Array env_ids_, player_ids_, action_;
    std::vector<float> low_, high_;
    void CopyObs(const std::vector<Array>& recv, float* obs) { auto src = TArray<mjtNum>(recv[8]); for (int i = 0; i < obs_dim(); ++i) obs[i] = static_cast<float>(src(0, i)); }
    void CopyRewardDone(const std::vector<Array>& recv, float* reward, uint8_t* done) { auto r = TArray<float>(recv[4]); auto d = TArray<bool>(recv[3]); reward[0] = r[0]; done[0] = d[0] ? 1 : 0; }
   public:
    HCEnv(int seed, const std::string& base) : pool_(MakeSpec(seed, base)), act_dim_(pool_.spec.action_spec["action"_].shape.back()), env_ids_(MakeArray<int>(1)), player_ids_(MakeArray<int>(1)), action_(MakeArray<mjtNum>(1, act_dim_)) {
      static_cast<int*>(env_ids_.Data())[0] = 0; static_cast<int*>(player_ids_.Data())[0] = 0;
      auto bounds = pool_.spec.action_spec["action"_].bounds; low_.assign(act_dim_, static_cast<float>(std::get<0>(bounds))); high_.assign(act_dim_, static_cast<float>(std::get<1>(bounds)));
    }
    int obs_dim() const { return pool_.spec.state_spec["obs"_].shape.back(); }
    int act_dim() const { return act_dim_; }
    const std::vector<float>& low() const { return low_; }
    const std::vector<float>& high() const { return high_; }
    void Reset(float* obs) { pool_.Reset(env_ids_); CopyObs(pool_.Recv(), obs); }
    void Step(const float* action, float* obs, float* reward, uint8_t* done) { auto* dst = static_cast<mjtNum*>(action_.Data()); for (int i = 0; i < act_dim_; ++i) dst[i] = static_cast<mjtNum>(action[i]); pool_.Send(std::vector<Array>{env_ids_, player_ids_, action_}); auto recv = pool_.Recv(); CopyObs(recv, obs); CopyRewardDone(recv, reward, done); }
  };
  struct Handle { std::unique_ptr<HCEnv> env; };
  }
  extern "C" {
  void* oak_env_open(int seed, const char* base, char* error, int cap) { try { if (!base) throw std::runtime_error("base_path is required"); mju_user_warning = OakWarn; auto* h = new Handle(); h->env = std::make_unique<HCEnv>(seed, base); return h; } catch (const std::exception& e) { CopyError(e.what(), error, cap); return nullptr; } }
  void oak_env_close(void* raw) { delete static_cast<Handle*>(raw); }
  int oak_env_obs_dim(void* raw) { return static_cast<Handle*>(raw)->env->obs_dim(); }
  int oak_env_act_dim(void* raw) { return static_cast<Handle*>(raw)->env->act_dim(); }
  int oak_env_action_bounds(void* raw, float* low, float* high) { auto* e = static_cast<Handle*>(raw)->env.get(); std::memcpy(low, e->low().data(), sizeof(float) * e->low().size()); std::memcpy(high, e->high().data(), sizeof(float) * e->high().size()); return e->act_dim(); }
  int oak_env_reset(void* raw, float* obs, char* error, int cap) { try { static_cast<Handle*>(raw)->env->Reset(obs); return 1; } catch (const std::exception& e) { CopyError(e.what(), error, cap); return 0; } }
  int oak_env_step(void* raw, const float* action, float* obs, float* reward, uint8_t* done, char* error, int cap) { try { static_cast<Handle*>(raw)->env->Step(action, obs, reward, done); return 1; } catch (const std::exception& e) { CopyError(e.what(), error, cap); return 0; } }
  int oak_env_take_warning(char* error, int cap) { auto& w = Warnings(); if (w.empty()) return 0; CopyError(w, error, cap); w.clear(); return 1; }
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
