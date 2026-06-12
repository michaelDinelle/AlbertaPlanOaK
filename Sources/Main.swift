import CryptoKit
import Darwin
import Foundation
import Metal

struct OakError: Error, CustomStringConvertible { let description: String }
func err(_ s: String) -> OakError { OakError(description: s) }
enum Policy: String { case oak, zero, random }
let defaultEnv = "HalfCheetah-v4"
let lockName = ".oak-benchmark.lock"
typealias Raw = UnsafeMutableRawPointer


struct KernelConfig {
  var obsDim, rank, optionCount, actionParamCount, obsTraceLevels, linearAnchorCount: UInt32
  var invRank, invOptionCount, metaEta: Float
  var agentCount, paramStride, stateStride, uintStride: UInt32
}
struct KernelScalars {
  var stepCount: UInt32 = 0, stateSlot: UInt32 = 0
  var rho: Float = 0
}
struct KernelStats {
  var encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount: Float
  var oakOptionReinitCount, oakFeatureReinitCount, terminationRate, optionDurationMean: Float
  var importanceEss, actorUpdateNormMean, actorSupportEntropy, modelPriorityMax: Float
  var selectedOptionAge, selectedOptionDuration, selectedOptionPriority, keyboardL1, actorAdvantageL1: Float
  var selectedOptionIndex, keyboardOrigin: Float
  var selectedOptionFeature, selectedContinuation, selectedKeyboardWeight, mainTd, planTdValue: Float
  var keyboardSketchDim, keyboardEffectiveL1, searchControlUseMean, generateTestEligibleCount: Float
  var learningMature: Float = 0
}

// ---- Phase F3 GPU fleet mirrors (must match OakKernel.metal layouts) ----
struct FleetEnvCfg {
  var laneCount, episodeLen, policyMode, recordCapacity: UInt32
  var invObsFloor: Float
}
struct SnapInfo {
  var count: UInt32 = 0, finiteCount: UInt32 = 0
  var sum: Float = 0, sumC: Float = 0, first: Float = 0, last: Float = 0
}
// Mirror of Metal OakLaneSnapshot (all 4-byte scalars; stride 276).
struct LaneSnapshot {
  var scalars: KernelScalars
  var stats: KernelStats
  var paramAbsMax: Float
  var paramNonFinite: UInt32
  var steps, doneCount, resets: UInt32
  var rewardSum, rho, deltaSum, deltaPeak: Float
  var deathStep, deathDetector: UInt32
  var deathRewardSum, deathRho: Float
  var info0, info1, info2, info3: SnapInfo
}
let fleetLaneStride = 356  // Metal OakFleetLane (static_assert'd kernel-side)

struct AblationSpec {
  static let ordered = [
    "feature-construction", "gvf-prediction", "rr-subtasks", "option-policies",
    "option-models", "planning", "option-keyboard", "model-to-keyboard",
    "keyboard-to-policy", "search-control", "search-to-generate-test", "generate-and-test",
  ]
  let names: [String]
  var label: String { names.isEmpty ? "none" : names.joined(separator: ",") }
  static func fromEnv(_ env: [String: String] = ProcessInfo.processInfo.environment) throws
    -> AblationSpec
  {
    let raw = env["OAK_ABLATE"] ?? "none"
    var seen = Set<String>()
    for item in raw.split(separator: ",") {
      let name = String(item).trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      if name.isEmpty || name == "none" { continue }
      guard ordered.contains(name) else {
        throw err("Unknown OAK_ABLATE entry '\(name)'. Known: none,\(ordered.joined(separator: ","))")
      }
      seen.insert(name)
    }
    return AblationSpec(names: Array(seen).sorted())
  }
  func apply(to constants: MTLFunctionConstantValues) {
    for (idx, name) in Self.ordered.enumerated() {
      var enabled = names.contains(name)
      constants.setConstantValue(&enabled, type: .bool, index: idx)
    }
  }
}

func root(_ p: String) -> URL {
  URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(p)
}
func require(_ p: String, _ what: String) throws -> URL {
  let u = root(p)
  guard FileManager.default.fileExists(atPath: u.path) else {
    throw err("Missing \(what). Run ./setup.sh first.")
  }
  return u
}
func requireFreshMetallib() throws -> URL {
  let lib = try require(".build/OakKernel.metallib", "non-empty Metal library")
  let src = root("Sources/OakKernel.metal")
  let fm = FileManager.default
  let libTime = try fm.attributesOfItem(atPath: lib.path)[.modificationDate] as? Date
  let srcTime = try fm.attributesOfItem(atPath: src.path)[.modificationDate] as? Date
  if let libTime, let srcTime, libTime < srcTime {
    throw err("Stale Metal library. Rebuild with `xcrun -sdk macosx metal -fno-fast-math -c Sources/OakKernel.metal -o .build/OakKernel.air && xcrun -sdk macosx metallib .build/OakKernel.air -o .build/OakKernel.metallib`.")
  }
  return lib
}
func mtlSize(_ width: Int) -> MTLSize {
  MTLSize(width: width, height: 1, depth: 1)
}
func sha256Hex(_ buf: UnsafeRawBufferPointer) -> String {
  var h = SHA256()
  h.update(bufferPointer: buf)
  return h.finalize().map { String(format: "%02x", $0) }.joined()
}

struct Arena {
  let obs, policy, options, action, credit, imp, reward, utilOpt, bonus, alpha, beta, alphaPsi,
    betaPsi, alphaModel, betaModel, total: Int
  static func traces(_ o: Int) -> Int {
    Int(log2(Double(max(o, 1)))) + 1
  }
  static func input(_ o: Int, _ d: Int) -> Int { traces(o) * max(o, 1) + max(d, 0) + 2 }
  init(_ o: Int, _ r: Int, _ d: Int) {
    let traceExtra = max(Self.traces(o) - 1, 0)
    obs = 0
    policy = 3 * o + 2 * r
    action = policy + d
    credit = action + d
    options = credit + r * d
    imp = options + r * d
    reward = imp + r
    utilOpt = reward + 3 + 2 * r
    bonus = utilOpt + r
    alpha = bonus + 3 * r
    beta = alpha + r
    alphaPsi = alpha + 4 * r
    betaPsi = alpha + 5 * r
    alphaModel = alpha + 8 * r
    betaModel = alpha + 9 * r
    // Base layout ends with `+ 1` (valueGauge slot). The trailing `+ 2 * r`
    // is for tdPsiFast and tdPsiPeak — per-option learning-progress EMAs.
    total = alpha + r * r + 21 * r + traceExtra * max(o, 1) + 1 + 2 * r
  }
}
final class Shared<T> {
  let b: MTLBuffer
  let p: UnsafeMutablePointer<T>
  init(_ d: MTLDevice, _ n: Int, _ z: T) {
    b = d.makeBuffer(length: max(n, 1) * MemoryLayout<T>.stride, options: [.storageModeShared])!
    p = b.contents().bindMemory(to: T.self, capacity: max(n, 1))
    p.initialize(repeating: z, count: max(n, 1))
  }
}
final class Engine {
  let rank, d, tg, learnTotal, actionGroups, batch, obsDim: Int
  let a: Arena
  let paramTotal, uintTotal: Int
  let q: MTLCommandQueue, pipes: [MTLComputePipelineState], params, state: Shared<Float>,
    uints: Shared<UInt32>, res: [MTLBuffer]
  let scalarsB, statsB, configB: MTLBuffer
  let scalars: UnsafeMutablePointer<KernelScalars>, stats: UnsafeMutablePointer<KernelStats>,
    tgBytes: Int
  var policy: UnsafePointer<Float> { policy(0) }
  var options: UnsafePointer<Float> { options(0) }
  func policy(_ lane: Int) -> UnsafePointer<Float> {
    UnsafePointer(state.p + lane * a.total + a.policy)
  }
  func options(_ lane: Int) -> UnsafePointer<Float> {
    UnsafePointer(state.p + lane * a.total + a.options)
  }
  func stats(_ lane: Int) -> KernelStats { stats[lane] }
  init(obsDim: Int, dPolicy: Int, batch b: Int, dev: MTLDevice, ablations: AblationSpec) throws {
    precondition(b >= 1, "Engine batch must be >= 1")
    batch = b
    self.obsDim = obsDim
    rank = max(obsDim, dPolicy * 2, 2)
    d = dPolicy
    a = Arena(obsDim, rank, d)
    let k = rank
    paramTotal = rank + k * rank + k * d * rank + k * rank + k * rank * rank + k * rank
    uintTotal = 9 + 8 * rank
    learnTotal =
      rank + rank * rank + rank * d * rank + rank * rank + rank * rank * rank + rank * rank
    let anchors = min(rank, max(obsDim, d, 1))
    let cfgValue = KernelConfig(
      obsDim: UInt32(obsDim), rank: UInt32(rank), optionCount: UInt32(rank),
      actionParamCount: UInt32(d), obsTraceLevels: UInt32(max(Arena.traces(obsDim) - 1, 0)),
      linearAnchorCount: UInt32(anchors), invRank: 1 / Float(rank),
      invOptionCount: 1 / Float(rank), metaEta: 1 / Float(rank) / Float(max(rank + d, 1)),
      agentCount: UInt32(batch), paramStride: UInt32(paramTotal),
      stateStride: UInt32(a.total), uintStride: UInt32(uintTotal))
    guard let cq = dev.makeCommandQueue() else {
      throw err("Metal command queue unavailable.")
    }
    q = cq
    let lib = try dev.makeLibrary(
      URL: try requireFreshMetallib())
    let constants = MTLFunctionConstantValues()
    ablations.apply(to: constants)
    Self.applyShapeConstants(constants, cfgValue)
    pipes = try ["oak_encode", "oak_plan", "oak_learn_heads", "oak_replace", "oak_action_emit"].map {
      let fn = try lib.makeFunction(name: $0, constantValues: constants)
      return try dev.makeComputePipelineState(function: fn)
    }
    let maxTG = pipes.map(\.maxTotalThreadsPerThreadgroup).min() ?? rank
    guard rank <= maxTG else {
      throw err("Rank \(rank) exceeds device threadgroup capacity \(maxTG).")
    }
    tg = min(max(rank, pipes.map { Int($0.threadExecutionWidth) }.max() ?? 1), maxTG)
    tgBytes = tg * MemoryLayout<Float>.stride
    actionGroups = max(1, (rank * d + tg - 1) / tg)
    params = Shared(dev, batch * paramTotal, 0)
    state = Shared(dev, batch * a.total, 0)
    uints = Shared(dev, batch * uintTotal, 0)
    scalarsB = dev.makeBuffer(
      length: batch * MemoryLayout<KernelScalars>.stride, options: [.storageModeShared])!
    statsB = dev.makeBuffer(
      length: batch * MemoryLayout<KernelStats>.stride, options: [.storageModeShared])!
    configB = dev.makeBuffer(
      length: MemoryLayout<KernelConfig>.stride, options: [.storageModeShared])!
    res = [params.b, state.b, uints.b, scalarsB, statsB, configB]
    scalars = scalarsB.contents().bindMemory(to: KernelScalars.self, capacity: batch)
    stats = statsB.contents().bindMemory(to: KernelStats.self, capacity: batch)
    memset(stats, 0, batch * MemoryLayout<KernelStats>.stride)
    let psi = rank
    let optPolicy = psi + rank * rank
    let inv = 1 / sqrt(Float(rank))
    let alpha = 1 / Float(max(obsDim, 1))
    for lane in 0..<batch {
      scalars[lane] = KernelScalars()
      let lp = params.p + lane * paramTotal
      let ls = state.p + lane * a.total
      let lu = uints.p + lane * uintTotal
      for i in 0..<rank {
        lp[psi + i * rank + i] = inv
        lu[7 + i] = UInt32(i)
        lu[7 + 3 * rank + i] = UInt32(i % max(anchors, 1))
        lu[7 + 4 * rank + i] = UInt32((i * 7 + 3) % max(anchors, 1))
        lu[7 + 5 * rank + i] = UInt32(i % 3)
        ls[a.utilOpt + i] = Float(rank)
        [a.alpha, a.alphaPsi, a.alphaModel].forEach { ls[$0 + i] = alpha }
        [a.beta, a.betaPsi, a.betaModel].forEach { ls[$0 + i] = log(alpha) }
      }
      let basis = 1 / Float(max(d, 1))
      for k in 0..<rank {
        let slot = k % max(d, 1)
        let sign: Float = ((k / max(d, 1)) % 2 == 0) ? 1 : -1
        lp[optPolicy + (k * d + slot) * rank] = sign * basis
      }
    }
    configB.contents().bindMemory(to: KernelConfig.self, capacity: 1).initialize(to: cfgValue)
  }
  // Shape specialization: bind the env dims and derived strides as Metal
  // function constants 13..24, sourced from the exact KernelConfig values the
  // kernels otherwise load from the cfg buffer — a specialized pipeline is
  // bit-identical to the generic one by construction (-fno-fast-math forbids
  // reassociation, so unrolled/constant-folded code keeps the fma sequences).
  static func applyShapeConstants(_ constants: MTLFunctionConstantValues, _ cfg: KernelConfig) {
    var u: [UInt32] = [
      cfg.obsDim, cfg.rank, cfg.optionCount, cfg.actionParamCount,
      cfg.obsTraceLevels, cfg.linearAnchorCount,
      cfg.paramStride, cfg.stateStride, cfg.uintStride,
    ]
    for i in u.indices { constants.setConstantValue(&u[i], type: .uint, index: 13 + i) }
    var f: [Float] = [cfg.invRank, cfg.invOptionCount, cfg.metaEta]
    for i in f.indices { constants.setConstantValue(&f[i], type: .float, index: 22 + i) }
  }
  func applyShapeConstants(_ constants: MTLFunctionConstantValues) {
    Self.applyShapeConstants(
      constants, configB.contents().bindMemory(to: KernelConfig.self, capacity: 1).pointee)
  }
  // Validation instrumentation: SHA-256 over a lane's trajectory-bearing slices
  // (params | state arena | uints | scalars), excluding diagnostics (stats).
  func stateHashes(_ lane: Int) -> [String: String] {
    func hex(_ base: UnsafeRawPointer, _ bytes: Int) -> String {
      sha256Hex(UnsafeRawBufferPointer(start: base, count: bytes))
    }
    return [
      "params": hex(params.p + lane * paramTotal, paramTotal * MemoryLayout<Float>.stride),
      "state": hex(state.p + lane * a.total, a.total * MemoryLayout<Float>.stride),
      "uints": hex(uints.p + lane * uintTotal, uintTotal * MemoryLayout<UInt32>.stride),
      "scalars": hex(scalars + lane, MemoryLayout<KernelScalars>.stride),
    ]
  }
}

// Fleet lanes are disjoint by construction (per-lane state, per-lane env handle);
// this box passes the shared handles into the env-step shards.
func jsonLine(_ obj: [String: Any]) throws -> Data {
  var d = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
  d.append(0x0A)
  return d
}
func metricsRoot() -> URL {
  if let r = ProcessInfo.processInfo.environment["OAK_METRICS_STORAGE"],
    !r.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  {
    return URL(fileURLWithPath: (r as NSString).expandingTildeInPath, isDirectory: true)
  }
  let id =
    ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
    + "-\(ProcessInfo.processInfo.processIdentifier)"
  return URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(
    ".oak-benchmarks/\(id)", isDirectory: true)
}

@main struct Main {
  static func main() {
    _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    do {
      switch CommandLine.arguments.dropFirst().first {
      case "benchmark": try benchmark()
      case "physics-probe": try physicsProbe()
      case "physics-rollout": try physicsRollout()
      default: throw err("Use `OaK benchmark`.")
      }
    } catch {
      fputs("\(error)\n", stderr)
      Foundation.exit(1)
    }
  }
  static func pol(_ s: String) throws -> Policy {
    guard let p = Policy(rawValue: s.lowercased()) else {
      throw err("Unknown policy '\(s)'.")
    }
    return p
  }
  static func fleet(
    _ p: Policy, _ seeds: [Int], _ steps: Int, _ root: URL, _ envId: String,
    _ ablations: AblationSpec
  ) throws {
    try fleetGPU(p, seeds, steps, root, envId, ablations)
  }
  // GPU-resident fleet (Phase F3): agent kernels + action sampling + physics
  // all live in multi-step command buffers; the CPU only drains heartbeat
  // snapshots into the same fleet.jsonl/deaths.jsonl/aggregate.jsonl contract.
  //   - OAK_EPISODE_STEPS: 0 (default) = continuing stream (CPU-era bridge
  //     semantics, docs/roadmap.md EPISODE-SEMANTICS DISCOVERY); 1000 = gym.
  //   - OAK_STEPS_PER_CB: step-iterations per command buffer (default 32,
  //     auto-tuned to keep CB GPU time <= ~80 ms when unset); 2 CBs in flight.
  //   - Heartbeats/finals/state-hash dumps always land on CB boundaries
  //     (OAK_DUMP_STATE_EVERY therefore forces a boundary + GPU drain at each
  //     multiple — validation mode, serializes the pipeline at dump steps).
  static func fleetGPU(
    _ p: Policy, _ seeds: [Int], _ steps: Int, _ root: URL, _ envId: String,
    _ ablations: AblationSpec
  ) throws {
    guard envId == defaultEnv else {
      throw err(
        "Only \(defaultEnv) is wired through the GPU physics kernel (got \(envId)).")
    }
    let e = ProcessInfo.processInfo.environment
    let fm = FileManager.default
    let B = seeds.count
    let obsDim = 17, actDim = 6, dPolicy = 12  // HalfCheetah-v4 (matches the GPU physics port)
    guard let dev = MTLCreateSystemDefaultDevice() else { throw err("Metal unavailable.") }
    let engine =
      p == .oak
      ? try Engine(obsDim: obsDim, dPolicy: dPolicy, batch: B, dev: dev, ablations: ablations)
      : nil
    if let eng = engine, eng.rank > 32 {
      throw err(
        "Megastep lane stages are simdgroup-wide (32 threads, 32-float tg slots); rank \(eng.rank) exceeds 32.")
    }
    guard let q = engine?.q ?? dev.makeCommandQueue() else {
      throw err("Metal command queue unavailable.")
    }
    let lib = try dev.makeLibrary(URL: try requireFreshMetallib())
    // make_ctx references the optional OAK_SPEC_* function constants, so every
    // kernel touching it must come from the specialized-function path (plain
    // makeFunction(name:) cannot build a pipeline once constants are present).
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let fn = try lib.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
      return try dev.makeComputePipelineState(function: fn)
    }
    let initPipe = try pipeline("oak_fleet_init")
    let snapPipe = try pipeline("oak_fleet_snapshot")
    // Megastep pipeline: the stage functions reference the OAK_ABLATE function
    // constants, so it must be specialized with the same constant values as the
    // Engine's agent pipelines. OAK_MEGA_G (constant 12) sets the v2 lane
    // packing: one threadgroup of 32*G threads owns G consecutive lanes
    // (OAK_MEGA_LANES_PER_TG; default = best measured; G=1 reproduces the v1
    // one-lane-per-threadgroup geometry). Threadgroup memory bounds G: nine
    // slots x G x 32 floats = G x 1152 bytes (G=24 -> 27.6 KB < 32 KB).
    // Megastep geometry law (measured 2026-06-11, M4 Max 32 cores): G
    // (lanes per threadgroup) and the learn helper simdgroup (OAK_MEGA_WIDE2X,
    // kernel constant 26) trade per-lane chain latency against saturated
    // occupancy. Measured oak fleet: B=64 wants G=2 + wide2x (1822 -> 1251
    // us/step per lane vs the old G=16 default); B=256 wants G=8 + wide2x
    // (153K vs 133K agg); B=512 is the crossover (~equal); B>=1024 wants
    // G=16 without wide2x (351K agg vs 266K with wide2x). Geometry only —
    // lane values are bitwise identical across G/wide2x (gated). Both env
    // vars override the law for experiments.
    let autoG = max(2, min(16, B / 32))
    let megaG = e["OAK_MEGA_LANES_PER_TG"].flatMap(Int.init) ?? autoG
    let wide2x = (e["OAK_MEGA_WIDE2X"].flatMap(Int.init).map { $0 != 0 }) ?? (B <= 512)
    let megaW = 32 * megaG * (wide2x ? 2 : 1)
    let megaConstants = MTLFunctionConstantValues()
    ablations.apply(to: megaConstants)
    var megaG32 = UInt32(megaG)
    megaConstants.setConstantValue(&megaG32, type: .uint, index: 12)
    if wide2x {
      var b = true
      megaConstants.setConstantValue(&b, type: .bool, index: 26)
    }
    // Shape specialization: oak runs bind the env dims (and derived strides)
    // as compile-time constants 13..24 — bit-identical by construction under
    // -fno-fast-math. zero/random runs have no engine (agent stages are
    // skipped) and keep the generic cfg path.
    if let eng = engine { eng.applyShapeConstants(megaConstants) }
    // OAK_MEGA_STAGE_MASK (DEBUG ONLY): stage-resolved timing instrument.
    // Bitmask gates megastep stages (encode 1, plan 2, learn 4, replace 8,
    // emit 16, sample 32, env 64); skipped stages still hit their barriers.
    // Semantics are intentionally broken when any bit is off — read only
    // aggStepsPerSecond. Unset = no constant bound = the certified all-on path.
    if let mask = e["OAK_MEGA_STAGE_MASK"].flatMap(UInt32.init) {
      var m = mask
      megaConstants.setConstantValue(&m, type: .uint, index: 25)
      fputs("OAK_MEGA_STAGE_MASK=\(mask): timing instrument, trajectories are garbage\n", stderr)
    }
    let megaPipe = try dev.makeComputePipelineState(
      function: lib.makeFunction(name: "oak_megastep", constantValues: megaConstants))
    let megaSlotBytes = 32 * megaG * MemoryLayout<Float>.stride  // per-lane 32-float slots
    guard megaG >= 1, megaW <= megaPipe.maxTotalThreadsPerThreadgroup,
      9 * megaSlotBytes <= dev.maxThreadgroupMemoryLength
    else {
      throw err(
        "OAK_MEGA_LANES_PER_TG=\(megaG) needs \(megaW) threads/threadgroup (pipeline max \(megaPipe.maxTotalThreadsPerThreadgroup)) and \(9 * megaSlotBytes) tg bytes (device max \(dev.maxThreadgroupMemoryLength)).")
    }
    func makeBuf(_ len: Int) throws -> MTLBuffer {
      guard let b = dev.makeBuffer(length: max(len, 16), options: [.storageModeShared]) else {
        throw err("Metal buffer allocation failed (\(len) bytes).")
      }
      return b
    }
    let laneBuf = try makeBuf(B * fleetLaneStride)
    let seedsBuf = try makeBuf(B * 4)
    do {
      let sp = seedsBuf.contents().bindMemory(to: UInt32.self, capacity: B)
      for (i, s) in seeds.enumerated() { sp[i] = UInt32(truncatingIfNeeded: s) }
    }
    let res: [MTLBuffer]
    if let eng = engine {
      res = eng.res
    } else {
      let dummy = try makeBuf(16)
      res = [dummy, dummy, dummy, dummy, dummy, try makeBuf(MemoryLayout<KernelConfig>.stride)]
    }
    let episodeLen = max(e["OAK_EPISODE_STEPS"].flatMap(Int.init) ?? 0, 0)
    let hb = max(e["OAK_BENCH_HEARTBEAT_STEPS"].flatMap(Int.init) ?? steps / 10, 1)
    if e["OAK_FLEET_FREEZE_DEAD"] == "1" {
      fputs("OAK_FLEET_FREEZE_DEAD is CPU-backend only; GPU lanes always keep stepping.\n", stderr)
    }
    if (e["OAK_RECORD_STATS_EVERY"].flatMap(Int.init) ?? 0) > 0 {
      throw err("OAK_RECORD_STATS_EVERY (stats.bin) is not supported by the GPU fleet.")
    }
    func seedDir(_ s: Int) throws -> URL {
      let dir = root.appendingPathComponent(envId, isDirectory: true)
        .appendingPathComponent("seed\(s)", isDirectory: true)
      try fm.createDirectory(at: dir, withIntermediateDirectories: true)
      return dir
    }
    // Action recording: GPU ring (2 heartbeat intervals deep), drained at
    // heartbeats into the CPU-era actions.bin format (actDim f32 + done u8).
    let recordAll = e["OAK_RECORD_ACTIONS"] == "1"
    let recordSeeds = Set(
      (e["OAK_RECORD_SEEDS"] ?? "").split(separator: ",").compactMap {
        Int(String($0).trimmingCharacters(in: .whitespaces))
      })
    var recordLanes: [Int] = []
    var recordHandles: [FileHandle] = []
    for (lane, s) in seeds.enumerated() where recordAll || recordSeeds.contains(s) {
      let dir = try seedDir(s)
      let meta: [String: Any] = ["envId": envId, "seed": s, "actDim": actDim, "recordVersion": 2]
      try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        .write(to: dir.appendingPathComponent("meta.json"), options: .atomic)
      let url = dir.appendingPathComponent("actions.bin")
      fm.createFile(atPath: url.path, contents: nil)
      recordHandles.append(try FileHandle(forWritingTo: url))
      recordLanes.append(lane)
    }
    let recordCapacity = recordLanes.isEmpty ? 0 : 2 * hb
    let recordSlotBuf = try makeBuf(B * 4)
    do {
      let rp = recordSlotBuf.contents().bindMemory(to: Int32.self, capacity: B)
      for i in 0..<B { rp[i] = -1 }
      for (slot, lane) in recordLanes.enumerated() { rp[lane] = Int32(slot) }
    }
    let recordBuf = try makeBuf(recordLanes.count * recordCapacity * 7 * 4)
    let snapStride = MemoryLayout<LaneSnapshot>.stride
    precondition(snapStride == 276, "LaneSnapshot layout drifted from OakLaneSnapshot")
    let snapBuf = try makeBuf(2 * B * snapStride)
    let snapPtr = snapBuf.contents().bindMemory(to: LaneSnapshot.self, capacity: 2 * B)
    let dumpEvery = max(e["OAK_DUMP_STATE_EVERY"].flatMap(Int.init) ?? 0, 0)
    var hashHandles: [FileHandle?] = Array(repeating: nil, count: B)
    if dumpEvery > 0, engine != nil {
      for (lane, s) in seeds.enumerated() {
        let url = try seedDir(s).appendingPathComponent("state-hashes.jsonl")
        fm.createFile(atPath: url.path, contents: nil)
        hashHandles[lane] = try FileHandle(forWritingTo: url)
      }
    }
    func appender(_ name: String) throws -> FileHandle {
      let url = root.appendingPathComponent(name)
      if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil) }
      let h = try FileHandle(forWritingTo: url)
      try h.seekToEnd()
      return h
    }
    let fleetOut = try appender("fleet.jsonl")
    let deathsOut = try appender("deaths.jsonl")
    let aggOut = try appender("aggregate.jsonl")
    defer {
      hashHandles.forEach { try? $0?.close() }
      recordHandles.forEach { try? $0.close() }
      try? fleetOut.close()
      try? deathsOut.close()
      try? aggOut.close()
    }
    let timing = e["OAK_FLEET_TIMING"] == "1"
    let ecfg = FleetEnvCfg(
      laneCount: UInt32(B), episodeLen: UInt32(episodeLen),
      policyMode: p == .oak ? 0 : (p == .zero ? 1 : 2),
      recordCapacity: UInt32(recordCapacity), invObsFloor: 1 / Float(obsDim))
    let laneTG = mtlSize(
      min(64, initPipe.maxTotalThreadsPerThreadgroup, snapPipe.maxTotalThreadsPerThreadgroup))
    func bindFleet(_ enc: MTLComputeCommandEncoder, _ pipe: MTLComputePipelineState) {
      enc.setComputePipelineState(pipe)
      for j in res.indices { enc.setBuffer(res[j], offset: 0, index: j) }
      enc.setBuffer(laneBuf, offset: 0, index: 6)
      enc.setBuffer(seedsBuf, offset: 0, index: 7)
      var c = ecfg
      enc.setBytes(&c, length: MemoryLayout<FleetEnvCfg>.stride, index: 8)
    }
    // Megastep v2: one persistent dispatch runs n whole step-iterations on
    // ceil(B/G) threadgroups x 32*G threads (each threadgroup owns G
    // consecutive lanes; threadgroup slots 0..8 mirror oak_encode's nine at 32
    // floats per lane). Bitwise-certified equal to v1 (one lane per
    // threadgroup), which was certified equal to the retired 7-dispatch
    // per-step loop; the per-stage wrapper kernels in OakKernel.metal remain
    // as the bisect instrument.
    let megaTGBytes = megaSlotBytes
    func encodeMegastep(_ enc: MTLComputeCommandEncoder, _ n: Int) {
      bindFleet(enc, megaPipe)
      enc.setBuffer(recordSlotBuf, offset: 0, index: 9)
      enc.setBuffer(recordBuf, offset: 0, index: 10)
      var n32 = UInt32(n)
      enc.setBytes(&n32, length: 4, index: 11)
      for m in 0..<9 { enc.setThreadgroupMemoryLength(megaTGBytes, index: m) }
      enc.dispatchThreadgroups(
        MTLSize(width: (B + megaG - 1) / megaG, height: 1, depth: 1),
        threadsPerThreadgroup: mtlSize(megaW))
      enc.memoryBarrier(scope: [.buffers])
    }
    // Lane init (env reset + arena seeding), synchronous.
    do {
      guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
        throw err("Failed to encode Metal command.")
      }
      bindFleet(enc, initPipe)
      enc.dispatchThreads(mtlSize(B), threadsPerThreadgroup: laneTG)
      enc.endEncoding()
      cb.commit()
      cb.waitUntilCompleted()
      if let e = cb.error { throw err("Metal command failed: \(e)") }
    }
    // ---- host writer (heartbeat snapshot -> jsonl rows) ----
    final class WriterState: @unchecked Sendable {
      var deadStep: [Int?]
      var deadDetector: [String?]
      var lastDrainStep = 0
      var error: Error?
      var gpuSeconds = 0.0
      // Previous-heartbeat rewardSums for recent-window delivered rates
      // (the live "is the population learning NOW" signal).
      var prevRewardSum: [Double]
      var prevStep = 0
      init(_ b: Int) {
        deadStep = Array(repeating: nil, count: b)
        deadDetector = Array(repeating: nil, count: b)
        prevRewardSum = Array(repeating: 0, count: b)
      }
    }
    let st = WriterState(B)
    let writer = DispatchQueue(label: "oak.fleet.gpu.writer")
    let detectorNames: [UInt32: String] = [1: "envReward", 2: "policyMean", 3: "envObs"]
    let wallStart = DispatchTime.now().uptimeNanoseconds
    func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1e9 }
    func infoJSON(_ a: SnapInfo) -> [String: Any] {
      let finite = Int(a.finiteCount)
      return [
        "count": Int(a.count), "finiteCount": finite,
        "nonFiniteCount": Int(a.count) - finite,
        "sum": finite > 0 ? Double(a.sum) : NSNull() as Any,
        "mean": finite > 0 ? Double(a.sum) / Double(finite) : NSNull() as Any,
        "last": finite > 0 ? Double(a.last) as Any : NSNull(),
        "delta": finite > 0 ? (Double(a.last) - Double(a.first)) as Any : NSNull(),
      ]
    }
    func health(_ snap: LaneSnapshot) -> [String: Any] {
      let adm = Double(snap.deltaSum) / Double(max(Int(snap.steps), 1))
      var h: [String: Any] = [
        "rewardRate": snap.rho.isFinite ? Double(snap.rho) : NSNull(),
        "actionDeltaL2Mean": adm.isFinite ? adm : NSNull(),
        "actionDeltaPeak": snap.deltaPeak.isFinite ? Double(snap.deltaPeak) : NSNull(),
        "continuingRestartCount": Double(snap.doneCount),
      ]
      if engine != nil {
        let s = snap.stats
        func f(_ x: Float) -> Any { x.isFinite ? Double(x) : NSNull() }
        h["oakOptionReinitCountFinal"] = Int(s.oakOptionReinitCount)
        h["oakFeatureReinitCountFinal"] = Int(s.oakFeatureReinitCount)
        h["nonFiniteByPhase"] = [
          "encode": Int(s.encodeNonFiniteCount), "learn": Int(s.learnNonFiniteCount),
          "plan": Int(s.planNonFiniteCount), "tdPsi": Int(s.tdPsiNonFiniteCount),
        ]
        h["terminationRate"] = f(s.terminationRate)
        h["optionDurationMean"] = f(s.optionDurationMean)
        h["importanceEss"] = f(s.importanceEss)
        h["actorUpdateNormMean"] = f(s.actorUpdateNormMean)
        h["actorSupportEntropy"] = f(s.actorSupportEntropy)
        h["modelPriorityMax"] = f(s.modelPriorityMax)
        h["selectedOptionAge"] = f(s.selectedOptionAge)
        h["selectedOptionDuration"] = f(s.selectedOptionDuration)
        h["selectedOptionPriority"] = f(s.selectedOptionPriority)
        h["keyboardL1"] = f(s.keyboardL1)
        h["actorAdvantageL1"] = f(s.actorAdvantageL1)
        h["selectedOptionIndex"] = f(s.selectedOptionIndex)
        h["keyboardOrigin"] = f(s.keyboardOrigin)
        h["selectedOptionFeature"] = f(s.selectedOptionFeature)
        h["selectedContinuation"] = f(s.selectedContinuation)
        h["selectedKeyboardWeight"] = f(s.selectedKeyboardWeight)
        h["mainTd"] = f(s.mainTd)
        h["planTd"] = f(s.planTdValue)
        h["keyboardSketchDim"] = f(s.keyboardSketchDim)
        h["keyboardEffectiveL1"] = f(s.keyboardEffectiveL1)
        h["searchControlUseMean"] = f(s.searchControlUseMean)
        h["generateTestEligibleCount"] = f(s.generateTestEligibleCount)
        h["learningMature"] = f(s.learningMature)
      }
      h["envInfo"] = [
        "info:reward_run": infoJSON(snap.info0), "info:reward_ctrl": infoJSON(snap.info1),
        "info:x_position": infoJSON(snap.info2), "info:x_velocity": infoJSON(snap.info3),
      ]
      return h
    }
    func writeSnapshot(half: Int, step: Int, kind: String) throws {
      let wall = now()
      var deathLines = Data()
      var lines = Data()
      for lane in 0..<B {
        let snap = snapPtr[half * B + lane]
        if st.deadStep[lane] == nil {
          var detector: String? = nil
          var dStep = step
          var dRewardSum = Double(snap.rewardSum)
          var dRho = Double(snap.rho)
          if snap.deathStep != 0xFFFF_FFFF {
            detector = detectorNames[snap.deathDetector] ?? "envError"
            dStep = Int(snap.deathStep)
            dRewardSum = Double(snap.deathRewardSum)
            dRho = Double(snap.deathRho)
          } else if engine != nil {
            if snap.paramNonFinite != 0 {
              detector = "paramNonFinite"
            } else if Double(snap.paramAbsMax) > 1e8 {
              detector = "paramExplosion"
            } else if !snap.scalars.rho.isFinite {
              detector = "kernelRho"
            }
          }
          if let det = detector {
            st.deadStep[lane] = dStep
            st.deadDetector[lane] = det
            var row: [String: Any] = [
              "seed": seeds[lane], "lane": lane, "step": dStep, "detector": det,
              "policy": p.rawValue, "rewardSumAtDeath": dRewardSum,
              "lastGoodRewardRate": dRho.isFinite ? dRho : NSNull(),
            ]
            if engine != nil {
              let s = snap.stats
              row["nfByPhase"] = [
                "encode": Int(s.encodeNonFiniteCount), "learn": Int(s.learnNonFiniteCount),
                "plan": Int(s.planNonFiniteCount), "tdPsi": Int(s.tdPsiNonFiniteCount),
              ]
              row["actorSupportEntropy"] =
                s.actorSupportEntropy.isFinite ? Double(s.actorSupportEntropy) : NSNull()
              row["terminationRate"] =
                s.terminationRate.isFinite ? Double(s.terminationRate) : NSNull()
              row["optionDurationMean"] =
                s.optionDurationMean.isFinite ? Double(s.optionDurationMean) : NSNull()
              row["rhoAtDeath"] =
                snap.scalars.rho.isFinite ? Double(snap.scalars.rho) : NSNull()
              let pm = snap.paramNonFinite != 0 ? Double.infinity : Double(snap.paramAbsMax)
              row["paramAbsMax"] = pm.isFinite ? pm : NSNull()
            }
            deathLines.append(try jsonLine(row))
          }
        }
        var m: [String: Any] = [
          "v": 1, "kind": kind, "envId": envId, "policy": p.rawValue,
          "ablationLabel": ablations.label, "lane": lane, "seed": seeds[lane],
          "step": Int(snap.steps), "wallSeconds": wall, "rewardSum": Double(snap.rewardSum),
          "doneCount": Int(snap.doneCount), "health": health(snap),
          "firstNonFiniteStep": st.deadStep[lane] ?? NSNull(),
          "firstNonFiniteDetector": st.deadDetector[lane] ?? NSNull(),
        ]
        if wall > 0 { m["stepsPerSecond"] = Double(snap.steps) / wall }
        if engine != nil {
          m["kernelRho"] = snap.scalars.rho.isFinite ? Double(snap.scalars.rho) : NSNull()
          let pm = snap.paramNonFinite != 0 ? Double.infinity : Double(snap.paramAbsMax)
          m["paramAbsMax"] = pm.isFinite ? pm : NSNull()
        }
        lines.append(try jsonLine(m))
      }
      if !deathLines.isEmpty { try deathsOut.write(contentsOf: deathLines) }
      try fleetOut.write(contentsOf: lines)
      let alive = st.deadStep.filter { $0 == nil }.count
      let unsorted = (0..<B).map { l -> Double in
        let sn = snapPtr[half * B + l]
        return sn.steps > 0 ? Double(sn.rewardSum) / Double(sn.steps) : 0
      }
      let rates = unsorted.sorted()
      func quantile(_ s: [Double], _ x: Double) -> Double {
        s[min(Int((Double(s.count - 1) * x).rounded()), s.count - 1)]
      }
      // Recent-window per-lane rates from the previous heartbeat's rewardSums.
      let dStep = step - st.prevStep
      let recent = (0..<B).map { l -> Double in
        let sn = snapPtr[half * B + l]
        return dStep > 0
          ? (Double(sn.rewardSum) - st.prevRewardSum[l]) / Double(dStep) : unsorted[l]
      }
      for l in 0..<B { st.prevRewardSum[l] = Double(snapPtr[half * B + l].rewardSum) }
      st.prevStep = step
      let recentSorted = recent.sorted()
      let bestLane = unsorted.indices.max(by: { unsorted[$0] < unsorted[$1] }) ?? 0
      try aggOut.write(
        contentsOf: jsonLine([
          "step": step, "kind": kind, "policy": p.rawValue, "alive": alive, "dead": B - alive,
          "deliveredRateP10": quantile(rates, 0.1), "deliveredRateP50": quantile(rates, 0.5),
          "deliveredRateP90": quantile(rates, 0.9), "wallSeconds": wall,
          "aggStepsPerSecond": wall > 0 ? Double(step) * Double(B) / wall : 0,
          "bestLaneSeed": seeds[bestLane], "bestLaneDeliveredRate": unsorted[bestLane],
          "bestLaneRewardSum": Double(snapPtr[half * B + bestLane].rewardSum),
          "positiveRateLaneCount": unsorted.filter { $0 > 0 }.count,
          "positiveRateFraction": Double(unsorted.filter { $0 > 0 }.count) / Double(B),
          "recentRateP10": quantile(recentSorted, 0.1),
          "recentRateP50": quantile(recentSorted, 0.5),
          "recentRateP90": quantile(recentSorted, 0.9),
          "recentRateMax": recentSorted[recentSorted.count - 1],
        ]))
      if !recordLanes.isEmpty {
        let rp = recordBuf.contents().bindMemory(
          to: Float.self, capacity: recordLanes.count * recordCapacity * 7)
        for (slot, _) in recordLanes.enumerated() {
          var data = Data(capacity: (step - st.lastDrainStep) * (actDim * 4 + 1))
          for r in st.lastDrainStep..<step {
            let base = (slot * recordCapacity + (r % recordCapacity)) * 7
            for u in 0..<actDim {
              var f = rp[base + u]
              withUnsafeBytes(of: &f) { data.append(contentsOf: $0) }
            }
            data.append(rp[base + 6] != 0 ? 1 : 0)
          }
          try recordHandles[slot].write(contentsOf: data)
        }
        st.lastDrainStep = step
      }
      if timing {
        fputs(
          String(
            format: "[fleet %@ gpu] step %d alive %d/%d | gpu %.1fs wall %.1fs | %.0f agg steps/s\n",
            p.rawValue, step, alive, B, st.gpuSeconds, wall,
            wall > 0 ? Double(step) * Double(B) / wall : 0), stderr)
      }
    }
    func writeHashes(step: Int) throws {
      guard let eng = engine else { return }
      for lane in 0..<B {
        guard let h = hashHandles[lane] else { continue }
        let hs = eng.stateHashes(lane)
        let line =
          "{\"seed\":\(seeds[lane]),\"step\":\(step),\"params\":\"\(hs["params"]!)\",\"state\":\"\(hs["state"]!)\",\"uints\":\"\(hs["uints"]!)\",\"scalars\":\"\(hs["scalars"]!)\"}\n"
        try h.write(contentsOf: Data(line.utf8))
      }
    }
    // ---- main loop: multi-step command buffers, 2 in flight ----
    let explicitN = e["OAK_STEPS_PER_CB"].flatMap(Int.init)
    let tuneLock = NSLock()
    var tunedN = max(explicitN ?? 32, 1)
    let inflight = DispatchSemaphore(value: 2)
    // With recording or hash dumps active the writer must finish draining a
    // heartbeat interval before the GPU can start the interval after next
    // (record ring is 2 intervals deep) — cap outstanding snapshots at 1.
    let snapSlots = DispatchSemaphore(value: (recordLanes.isEmpty && dumpEvery == 0) ? 2 : 1)
    var snapHalf = 0
    var step = 0
    while step < steps {
      tuneLock.lock()
      let n = tunedN
      tuneLock.unlock()
      let nextHb = min(((step / hb) + 1) * hb, steps)
      var boundary = nextHb
      if dumpEvery > 0 { boundary = min(boundary, ((step / dumpEvery) + 1) * dumpEvery) }
      let target = min(boundary, step + n)
      let isSnap = target == nextHb
      let isDump = dumpEvery > 0 && target % dumpEvery == 0
      inflight.wait()
      if let err = writer.sync(execute: { st.error }) { throw err }
      if isSnap { snapSlots.wait() }
      guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
        throw err("Failed to encode Metal command.")
      }
      encodeMegastep(enc, target - step)
      if isSnap {
        bindFleet(enc, snapPipe)
        enc.setBuffer(snapBuf, offset: 0, index: 9)
        var base = UInt32(snapHalf * B)
        enc.setBytes(&base, length: 4, index: 10)
        enc.dispatchThreads(mtlSize(B), threadsPerThreadgroup: laneTG)
      }
      enc.endEncoding()
      let thisHalf = snapHalf
      if isSnap { snapHalf ^= 1 }
      let kind = target == steps ? "final" : "hb"
      let tStep = target
      let nSteps = target - step
      cb.addCompletedHandler { cb in
        let gpuT = cb.gpuEndTime - cb.gpuStartTime
        if explicitN == nil, gpuT > 0, nSteps > 0 {
          let fit = max(1, min(Int(0.08 / (gpuT / Double(nSteps))), 8192))
          tuneLock.lock()
          tunedN = max(1, min(fit, tunedN * 4))
          tuneLock.unlock()
        }
        let cbError = cb.error
        writer.async {
          st.gpuSeconds += gpuT
          if let cbError, st.error == nil {
            st.error = err("Metal command failed: \(cbError)")
          }
          if isSnap {
            if st.error == nil {
              do { try writeSnapshot(half: thisHalf, step: tStep, kind: kind) } catch {
                st.error = error
              }
            }
            snapSlots.signal()
          }
          inflight.signal()
        }
      }
      cb.commit()
      if isDump {
        cb.waitUntilCompleted()
        writer.sync {}  // snapshot rows for this boundary are flushed first
        try writeHashes(step: target)
      }
      step = target
    }
    inflight.wait()
    inflight.wait()
    writer.sync {}
    inflight.signal()
    inflight.signal()
    if let err = st.error { throw err }
  }
  // fp32 Metal physics probe: argv[2] = input bin (uint32 count + count*33 f32:
  // qpos[9], qvel[9], ctrl[6], warmstart[9]), argv[3] = output bin (count*18 f32:
  // qpos'[9], qvel'[9]). One thread per case, exactly one hc_mj_step each.
  static func physicsProbe() throws {
    let args = CommandLine.arguments
    guard args.count >= 4 else {
      throw err("Usage: OaK physics-probe <input.bin> <output.bin>")
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: args[2]))
    guard data.count >= 4 else { throw err("Probe input too small.") }
    let n = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) })
    let inFloats = n * 33
    guard n > 0, data.count == 4 + inFloats * 4 else {
      throw err("Probe input size mismatch: \(data.count) bytes for \(n) cases.")
    }
    // Optional argv[4] == "debug": dump per-case solver intermediates (904 floats)
    // via oak_physics_probe_debug instead of the 18-float qpos'/qvel' output.
    let debug = args.count > 4 && args[4] == "debug"
    let outStride = debug ? 904 : 18
    let kernelName = debug ? "oak_physics_probe_debug" : "oak_physics_probe"
    guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
      throw err("Metal unavailable.")
    }
    let lib = try dev.makeLibrary(URL: requireFreshMetallib())
    guard let fn = lib.makeFunction(name: kernelName) else {
      throw err("Missing \(kernelName) kernel.")
    }
    let pipe = try dev.makeComputePipelineState(function: fn)
    guard let inBuf = dev.makeBuffer(length: inFloats * 4, options: [.storageModeShared]),
      let outBuf = dev.makeBuffer(length: n * outStride * 4, options: [.storageModeShared])
    else { throw err("Metal buffer allocation failed.") }
    data.withUnsafeBytes { raw in
      memcpy(inBuf.contents(), raw.baseAddress!.advanced(by: 4), inFloats * 4)
    }
    guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
      throw err("Failed to encode Metal command.")
    }
    enc.setComputePipelineState(pipe)
    enc.setBuffer(inBuf, offset: 0, index: 0)
    enc.setBuffer(outBuf, offset: 0, index: 1)
    var count = UInt32(n)
    enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)
    let tg = min(64, pipe.maxTotalThreadsPerThreadgroup)
    enc.dispatchThreads(mtlSize(n), threadsPerThreadgroup: mtlSize(tg))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw err("Metal command failed: \(e)") }
    try Data(bytes: outBuf.contents(), count: n * outStride * 4)
      .write(to: URL(fileURLWithPath: args[3]))
  }
  // Env-step rollout host (Phase F2). Forms:
  //   OaK physics-rollout <input.bin> <output.bin>
  //     input: u32 header [lanes, steps, mode, seedBase, episodeLen], then
  //     initial states (lanes*18 f32, modes 0/3), then ctrl (lanes*steps*6 f32,
  //     mode 0). Output: mode 0/1 -> lanes*steps*37 f32, mode 3 -> lanes*steps*2.
  //   OaK physics-rollout --bench
  //     mode 2 (Philox reset + random ctrl, no trajectory): B from
  //     OAK_ROLLOUT_LANES (4096), steps from OAK_ROLLOUT_STEPS (2000); reports
  //     env-steps/s + mj-steps/s + output checksum to stderr.
  //   OaK physics-rollout --rng-probe <out.bin>
  //     mode 4 sampling probe: OAK_RNG_LANES (200000) lanes x 24 f32
  //     (reset qpos[9]/qvel[9] + step-0 ctrl[6]).
  static let rolloutLaneStride = 35 * 4  // sizeof(HCEnvLane): HCState 33 f + 2 u32
  static func physicsRollout() throws {
    let args = CommandLine.arguments
    let e = ProcessInfo.processInfo.environment
    guard args.count >= 3 else {
      throw err("Usage: OaK physics-rollout <input.bin> <output.bin> | --bench | --rng-probe <out.bin>")
    }
    if args[2] == "--bench" {
      let lanes = e["OAK_ROLLOUT_LANES"].flatMap(Int.init) ?? 4096
      let steps = e["OAK_ROLLOUT_STEPS"].flatMap(Int.init) ?? 2000
      let seedBase = e["OAK_ROLLOUT_SEED_BASE"].flatMap(Int.init) ?? 0
      let episodeLen = e["OAK_ROLLOUT_EPISODE_LEN"].flatMap(Int.init) ?? 1000
      _ = try runRollout(
        header: [UInt32(lanes), UInt32(steps), 2, UInt32(seedBase), UInt32(episodeLen)],
        initStates: nil, ctrl: nil, outURL: nil, bench: true)
      return
    }
    if args[2] == "--rng-probe" {
      guard args.count >= 4 else { throw err("Usage: OaK physics-rollout --rng-probe <out.bin>") }
      let lanes = e["OAK_RNG_LANES"].flatMap(Int.init) ?? 200_000
      _ = try runRollout(
        header: [UInt32(lanes), 1, 4, 0, 1000], initStates: nil, ctrl: nil,
        outURL: URL(fileURLWithPath: args[3]), bench: false)
      return
    }
    guard args.count >= 4 else { throw err("Usage: OaK physics-rollout <input.bin> <output.bin>") }
    let data = try Data(contentsOf: URL(fileURLWithPath: args[2]))
    guard data.count >= 20 else { throw err("Rollout input too small.") }
    let header: [UInt32] = data.withUnsafeBytes { raw in
      (0..<5).map { raw.load(fromByteOffset: $0 * 4, as: UInt32.self) }
    }
    let lanes = Int(header[0]), steps = Int(header[1]), mode = Int(header[2])
    guard lanes > 0, steps > 0, [0, 1, 3].contains(mode) else {
      throw err("Rollout header invalid: lanes=\(lanes) steps=\(steps) mode=\(mode).")
    }
    let initBytes = (mode == 0 || mode == 3) ? lanes * 18 * 4 : 0
    let ctrlBytes = mode == 0 ? lanes * steps * 6 * 4 : 0
    guard data.count == 20 + initBytes + ctrlBytes else {
      throw err("Rollout input size mismatch: \(data.count) bytes, expected \(20 + initBytes + ctrlBytes).")
    }
    let initStates = initBytes > 0 ? data.subdata(in: 20..<(20 + initBytes)) : nil
    let ctrl = ctrlBytes > 0 ? data.subdata(in: (20 + initBytes)..<data.count) : nil
    _ = try runRollout(
      header: header, initStates: initStates, ctrl: ctrl,
      outURL: URL(fileURLWithPath: args[3]), bench: false)
  }
  @discardableResult
  static func runRollout(
    header: [UInt32], initStates: Data?, ctrl: Data?, outURL: URL?, bench: Bool
  ) throws -> Double {
    let lanes = Int(header[0]), steps = Int(header[1]), mode = Int(header[2])
    let outFloats: Int
    switch mode {
    case 0, 1: outFloats = lanes * steps * 37
    case 2: outFloats = lanes * 18
    case 3: outFloats = lanes * steps * 2
    case 4: outFloats = lanes * 24
    default: throw err("Unknown rollout mode \(mode).")
    }
    guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
      throw err("Metal unavailable.")
    }
    let lib = try dev.makeLibrary(URL: requireFreshMetallib())
    guard let fn = lib.makeFunction(name: "oak_physics_rollout") else {
      throw err("Missing oak_physics_rollout kernel.")
    }
    let pipe = try dev.makeComputePipelineState(function: fn)
    func buf(_ data: Data?, _ minBytes: Int, shared: Bool = true) throws -> MTLBuffer {
      let n = max(data?.count ?? 0, max(minBytes, 16))
      guard let b = dev.makeBuffer(length: n, options: shared ? [.storageModeShared] : [.storageModePrivate])
      else { throw err("Metal buffer allocation failed (\(n) bytes).") }
      if let d = data, shared {
        d.withUnsafeBytes { memcpy(b.contents(), $0.baseAddress!, d.count) }
      }
      return b
    }
    var hdr = header
    let hdrBuf = try buf(Data(bytes: &hdr, count: 20), 20)
    let initBuf = try buf(initStates, 16)
    let ctrlBuf = try buf(ctrl, 16)
    let outBuf = try buf(nil, outFloats * 4)
    let laneBuf = try buf(nil, lanes * rolloutLaneStride, shared: false)
    // mode 3 steps are raw mj-steps; env-step chunks cost 5 mj-steps each.
    let chunkSteps = mode == 4 ? 1 : (mode == 3 ? 500 : 100)
    let tg = min(64, pipe.maxTotalThreadsPerThreadgroup)
    var gpuSeconds = 0.0
    var start = 0
    while start < (mode == 4 ? 1 : steps) {
      guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
        throw err("Failed to encode Metal command.")
      }
      enc.setComputePipelineState(pipe)
      enc.setBuffer(hdrBuf, offset: 0, index: 0)
      enc.setBuffer(initBuf, offset: 0, index: 1)
      enc.setBuffer(ctrlBuf, offset: 0, index: 2)
      enc.setBuffer(outBuf, offset: 0, index: 3)
      enc.setBuffer(laneBuf, offset: 0, index: 4)
      var chunk = SIMD2<UInt32>(UInt32(start), UInt32(chunkSteps))
      enc.setBytes(&chunk, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 5)
      enc.dispatchThreads(mtlSize(lanes), threadsPerThreadgroup: mtlSize(tg))
      enc.endEncoding()
      cb.commit()
      cb.waitUntilCompleted()
      if let e = cb.error { throw err("Metal command failed: \(e)") }
      gpuSeconds += cb.gpuEndTime - cb.gpuStartTime
      start += chunkSteps
    }
    if let url = outURL {
      try Data(bytes: outBuf.contents(), count: outFloats * 4).write(to: url)
    }
    if bench {
      let p = outBuf.contents().bindMemory(to: Float.self, capacity: outFloats)
      var sum = 0.0
      for i in 0..<outFloats { sum += Double(p[i]) }
      let sha = sha256Hex(UnsafeRawBufferPointer(start: outBuf.contents(), count: outFloats * 4))
      let envSteps = Double(lanes) * Double(steps)
      fputs(
        String(
          format:
            "physics-rollout bench: B=%d steps=%d gpu %.3fs | %.0f env-steps/s | %.0f mj-steps/s | checksum sum=%.6e sha=%@\n",
          lanes, steps, gpuSeconds, envSteps / gpuSeconds, 5 * envSteps / gpuSeconds, sum,
          String(sha.prefix(16))), stderr)
    }
    return gpuSeconds
  }
  static func benchmark() throws {
    let env = ProcessInfo.processInfo.environment
    let ablations = try AblationSpec.fromEnv(env)
    let fm = FileManager.default
    let envId = env["OAK_BENCH_ENV"] ?? defaultEnv
    let seed0 = env["OAK_BENCH_SEED_START"].flatMap(Int.init) ?? 0
    let seedValues: [Int]
    if let explicitSeeds = env["OAK_BENCH_SEEDS"] {
      let parsed = explicitSeeds.split(separator: ",").compactMap {
        Int(String($0).trimmingCharacters(in: .whitespaces))
      }
      seedValues = parsed.isEmpty
        ? Array(seed0..<(seed0 + (env["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 1024)))
        : parsed
    } else {
      seedValues = Array(seed0..<(seed0 + (env["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 1024)))
    }
    let steps = max(env["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 500_000, 1)
    let policies = try (env["OAK_BENCH_POLICIES"] ?? "oak,zero,random").split(separator: ",").map {
      try pol(String($0))
    }
    var dir = metricsRoot()
    let lockDir = URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent(
      ".oak-benchmarks", isDirectory: true)
    try fm.createDirectory(at: lockDir, withIntermediateDirectories: true)
    let fd = open(
      lockDir.appendingPathComponent(lockName).path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
    guard fd >= 0, flock(fd, LOCK_EX) == 0 else {
      throw err("Cannot lock benchmark workspace.")
    }
    defer {
      _ = flock(fd, LOCK_UN)
      _ = close(fd)
    }
    if fm.fileExists(atPath: dir.path),
      !(try fm.contentsOfDirectory(atPath: dir.path).filter { $0 != ".DS_Store" }).isEmpty
    {
      dir = URL(
        fileURLWithPath: "\(dir.path)-\(ProcessInfo.processInfo.processIdentifier)",
        isDirectory: true)
    }
    try fm.createDirectory(at: dir, withIntermediateDirectories: true)
    // Static FMA-class op count per lane-step (build constant; the per-FLOP
    // denominator for reward-rate-per-FLOP reporting). Derivation from the
    // kernel loop bounds (HalfCheetah: O=17, D=12, R=K=24, learnTotal=22,488):
    // encode option-model query + plan transition TDs ~ 2*K*R^2; learn ~ 2 ops
    // per element; emit blend K*D*R; RR/psi encode + features ~ 4*K*R + 6*R^2;
    // GPU physics ~ 20K (5 fp32 mj_steps: FK/CRB/RNE/contacts/Newton/Euler).
    let mR = max(17, 12 * 2, 2)
    let mLearnTotal = mR + mR * mR + mR * 12 * mR + mR * mR + mR * mR * mR + mR * mR
    let flopsPerLaneStep =
      2 * mR * mR * mR + 2 * mLearnTotal + mR * 12 * mR + 4 * mR * mR + 6 * mR * mR + 20_000
    try JSONSerialization.data(
      withJSONObject: [
        "fleetSchemaVersion": 1, "command": "benchmark", "envId": envId,
        "batch": seedValues.count, "seeds": seedValues, "steps": steps,
        "policies": policies.map(\.rawValue), "ablation": ablations.names,
        "ablationLabel": ablations.label,
        "flopsPerLaneStep": flopsPerLaneStep,
        "startedAt": ISO8601DateFormatter().string(from: Date()),
      ] as [String: Any], options: [.prettyPrinted, .sortedKeys]
    ).write(to: dir.appendingPathComponent("meta.json"), options: .atomic)
    fputs(
      "OaK fleet: B=\(seedValues.count) lanes × \(steps) steps × \(policies.count) policies on \(envId) (ablation=\(ablations.label), dir=\(dir.path))\n",
      stderr)
    for p in policies {
      try fleet(p, seedValues, steps, dir, envId, ablations)
    }
  }
}
