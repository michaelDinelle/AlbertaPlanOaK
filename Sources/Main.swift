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
  var nurseryCount: UInt32
  // Host-precomputed constants (identical IEEE values — Float division is
  // correctly rounded on both sides with fast-math off).
  var keyboardSketchDim: UInt32  // max(1, min(D, K))
  var sampleSigma: Float  // 1 / max(D, 1)
  var sampleInv2S2: Float  // 1 / (2 * sampleSigma * sampleSigma)
  var actorAlphaDenom: Float  // Float(max(D * 32, 1)) — divisor, NOT a reciprocal
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
  var dynaBackupMean: Float = 0
}

struct FleetEnvCfg {
  var laneCount, episodeLen, policyMode, recordCapacity: UInt32
  var invObsFloor: Float
}
struct SnapInfo {
  var count: UInt32 = 0, finiteCount: UInt32 = 0
  var sum: Float = 0, sumC: Float = 0, first: Float = 0, last: Float = 0
}
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
let fleetLaneStride = 400

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
  let sources = [
    "Sources/OakKernel.metal",
    "Sources/OakPhysics.metal",
    "Sources/HalfCheetahModel.metal",
    "Sources/Gridworld.metal",
    "Sources/ClassicControl.metal",
  ].map(root)
  let fm = FileManager.default
  let libTime = try fm.attributesOfItem(atPath: lib.path)[.modificationDate] as? Date
  let staleSource = try sources.first { src in
    guard let libTime,
      let srcTime = try fm.attributesOfItem(atPath: src.path)[.modificationDate] as? Date
    else { return false }
    return libTime < srcTime
  }
  if let staleSource {
    throw err("Stale Metal library: \(staleSource.path) is newer than .build/OakKernel.metallib. Rebuild with `./setup.sh`.")
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
  let obs, policy, options, utilOpt, alpha, beta, alphaPsi, betaPsi, alphaModel, betaModel,
    alphaAct, betaAct, total: Int
  static func traces(_ o: Int) -> Int {
    Int(log2(Double(max(o, 1)))) + 1
  }
  static func input(_ o: Int, _ d: Int) -> Int { traces(o) * max(o, 1) + max(d, 0) + 2 }
  init(_ o: Int, _ r: Int, _ k: Int, _ d: Int, _ rn: Int) {
    let tl = max(Self.traces(o) - 1, 0)
    var s = 0
    func take(_ n: Int) -> Int {
      let at = s
      s += n
      return at
    }
    obs = take(3 * o)
    _ = take(2 * r)  // state slots
    policy = take(d)
    _ = take(d)  // policyMeanRaw (raw-acc blend for discrete logits)
    _ = take(d)  // actionSignal
    _ = take(k * d)  // optionActorCredit
    options = take(k * d)  // optionPolicyMean
    _ = take(k)  // importance
    _ = take(3)  // reward, td, planTd
    _ = take(k)  // tdPsi
    _ = take(r)  // utilFeat
    utilOpt = take(k)
    _ = take(3 * r)  // bonusW, featureMean, featureSM
    alpha = take(r)
    beta = take(r)
    _ = take(2 * r)  // hMeta, gradEnergy
    alphaPsi = take(r)
    betaPsi = take(r)
    _ = take(2 * r)  // hPsi, gradEnergyPsi
    alphaModel = take(r)
    betaModel = take(r)
    _ = take(2 * r)  // hModel, gradEnergyModel
    _ = take(5 * k)  // actorAdv, optRewardTd + 3 reserved (headDenom/durationTd/durationHat deleted)
    _ = take(tl * max(o, 1))  // obsTraceMulti
    _ = take(4 * k)  // prioOpt, continuation, keyboard + 1 reserved (modelDenom deleted)
    _ = take(1)  // valueGauge
    _ = take(2 * k)  // reserved (tdPsiFast/tdPsiPeak deleted)
    _ = take(2 * k)  // headScale + 1 reserved (modelScale deleted)
    _ = take(2)  // valueScale, xNormPrev
    _ = take(5 * k)  // hDotPsi, psiAuxScale, psiRho, optRho, optFit
    alphaAct = take(k)  // actor IDBD multiplier (init 1)
    betaAct = take(k)
    _ = take(2 * k)  // hAct, energyAct
    _ = take(k)  // impStage (sample->next-step importance staging)
    _ = take(k * d)  // creditStage (sample->next-step actor-credit staging)
    _ = take(8)  // sampleScr
    _ = take(4 * k)  // rHatC, lRawC, vsNextC, twinDis (encode->plan dot cache)
    _ = take(4 * k)  // vsTd, vsDot, vsScale, vsAuxScale
    _ = take(1)  // planScale
    s = (s + 3) & ~3  // 16B-align nurStats/muRaw/polCoef (float4 access)
    _ = take(4 * rn)  // nursery mean/SM/util/utilSM
    _ = take(k * d)  // muRaw (emit blend scratch)
    _ = take(k * d)  // polCoef (actor coefficient hoist)
    _ = take(r)  // optUseByFeat (curation scan scatter)
    _ = take(rn)  // lcbScr (parallel LCB precompute)
    s = (s + 3) & ~3  // align pad
    _ = take(4 * r)  // eligScr (parallel eligibility tuples)
    _ = take(k)  // fcG (tg-diet: keyboard scores scratch)
    _ = take(16)  // fcS (tg-diet: fc scalars + sample exchange)
    _ = take(64 * r)  // ring values (R slots x 64 active tuples: core + gated)
    _ = take(r)  // dynaTd (per-ring-slot backup priorities)
    _ = take(32)  // dynaD (per-thread backup deltas)
    _ = take(r)  // planU (planner-owned ring value table, reclaims dynaAcc)
    _ = take(k)  // dynaC (copy-2 per-option update coefficients)
    _ = take(4)  // planScr (reward snapshot; slot 2 = rsqrt(R) cache; 1/3 reserved)
    _ = take(r)  // ringReward (outbound reward per ring slot, for the relay)
    total = s
  }
}
final class Shared<T> {
  let b: MTLBuffer
  let p: UnsafeMutablePointer<T>
  init(_ d: MTLDevice, _ n: Int, _ z: T) {
    b = d.makeBuffer(
      length: max(n, 1) * MemoryLayout<T>.stride,
      options: [.storageModeShared])!
    p = b.contents().bindMemory(to: T.self, capacity: max(n, 1))
    p.initialize(repeating: z, count: max(n, 1))
  }
}
final class Engine {
  let rank, d, batch, obsDim: Int
  let a: Arena
  let paramTotal, uintTotal: Int
  let q: MTLCommandQueue, params, state: Shared<Float>,
    uints: Shared<UInt32>, res: [MTLBuffer]
  let scalarsB, statsB, configB: MTLBuffer
  let scalars: UnsafeMutablePointer<KernelScalars>, stats: UnsafeMutablePointer<KernelStats>
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
    // Sparse-wide: capacity = smallest power of two >= 2x linear input width
    // (stream-shape-derived, Inv-9); active width = 32 (execution width) via
    // banked WTA; K = active width.
    let inputW = Arena.input(obsDim, dPolicy)
    var cap = 32
    while cap < 2 * inputW { cap *= 2 }
    rank = cap
    d = dPolicy
    // Sutton-test instrument: scales the nursery capacity (canonical runs: 1).
    let nurseryMult = max(
      ProcessInfo.processInfo.environment["OAK_SUTTON_MULT"].flatMap(Int.init) ?? 1, 1)
    let rn = rank * nurseryMult
    let k = 32
    a = Arena(obsDim, rank, k, d, rn)
    paramTotal = rank + 8 * k * rank + k * (d / 2) * rank
    // uints: 11 scalars + optFeature/optAge/lastStopAge (3k) + 5 rank arrays
    // + 4 nursery arrays + actIdx (2*64) + dynaSlot (32) + ringIdx (32*rank)
    uintTotal = 11 + 3 * k + 5 * rank + 4 * rn + 128 + 32 + 32 * rank
    // Core region: 17 std obs + 12 prev-action + reward + bias + spare = 32
    // always-active slots (Mark III law: all raw inputs visible every step).
    let anchors = 32
    let sampleSigma = 1 / Float(max(d, 1))
    let cfgValue = KernelConfig(
      obsDim: UInt32(obsDim), rank: UInt32(rank), optionCount: UInt32(k),
      actionParamCount: UInt32(d), obsTraceLevels: UInt32(max(Arena.traces(obsDim) - 1, 0)),
      linearAnchorCount: UInt32(anchors), invRank: 1 / Float(rank),
      invOptionCount: 1 / Float(k), metaEta: 1 / Float(rank) / Float(max(rank + d, 1)),
      agentCount: UInt32(batch), paramStride: UInt32(paramTotal),
      stateStride: UInt32(a.total), uintStride: UInt32(uintTotal),
      nurseryCount: UInt32(rn),
      keyboardSketchDim: UInt32(max(1, min(d, k))),
      sampleSigma: sampleSigma,
      sampleInv2S2: 1 / (2 * sampleSigma * sampleSigma),
      actorAlphaDenom: Float(max(d * 32, 1)))
    guard let cq = dev.makeCommandQueue() else {
      throw err("Metal command queue unavailable.")
    }
    q = cq
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
    let inv = 1 / sqrt(Float(rank))
    let alpha0 = 1 / Float(max(obsDim, 1))
    // uint layout offsets (mirror make_ctx)
    let uFeatAge = 7 + k
    let uParentA = 7 + 2 * k + rank
    let uParentB = 7 + 2 * k + 2 * rank
    let uKind = 7 + 2 * k + 3 * rank
    let uExplStep = 9 + 2 * k + 5 * rank
    let uNur = 11 + 3 * k + 5 * rank
    _ = uFeatAge
    for lane in 0..<batch {
      scalars[lane] = KernelScalars()
      let lp = params.p + lane * paramTotal
      let ls = state.p + lane * a.total
      let lu = uints.p + lane * uintTotal
      lu[uExplStep] = UInt32.max
      lu[uExplStep + 1] = UInt32.max
      let traceChannels = max((Arena.traces(obsDim) - 1) * obsDim, 1)  // 4 levels x 17
      for n in 0..<rn {
        // Generator seeds: kinds cycle product/sum/residual/trace; parents are
        // core slots (always active) or trace channels for kind 3.
        lu[uNur + n] = UInt32(n % ((n % 4 == 3) ? traceChannels : anchors))
        lu[uNur + rn + n] = UInt32((n * 7 + 3) % anchors)
        lu[uNur + 2 * rn + n] = UInt32(n % 4)
      }
      for i in 0..<rank {
        if i >= anchors {
          // Gated region: first 68 slots are the multi-timescale obs traces
          // (kind 3); the rest are products/sums/residuals of core channels.
          let idx = i - anchors
          if idx < traceChannels {
            lu[uParentA + i] = UInt32(idx)
            lu[uParentB + i] = 0
            lu[uKind + i] = 3
          } else {
            lu[uParentA + i] = UInt32(idx % anchors)
            lu[uParentB + i] = UInt32((idx * 7 + 3) % anchors)
            lu[uKind + i] = UInt32(idx % 3)
          }
        } else {
          lu[uParentA + i] = 0
          lu[uParentB + i] = 0
          lu[uKind + i] = 0
        }
        [a.alpha, a.alphaPsi, a.alphaModel].forEach { ls[$0 + i] = alpha0 }
        [a.beta, a.betaPsi, a.betaModel].forEach { ls[$0 + i] = log(alpha0) }
      }
      let basis = 1 / Float(max(d, 1))
      // Ensemble copy-2 decorrelation: vStop2 gets a shifted diagonal (same
      // magnitude as psi's diagonal, different index pattern); reward/duration
      // copies start at zero like copy 1 and decorrelate through vStop2's
      // different bootstrap plus the semi-gradient update geometry.
      let vStop2 = rank + 7 * k * rank + k * (d / 2) * rank
      for kk in 0..<k {
        lu[7 + kk] = UInt32(kk)
        lp[psi + kk * k + kk] = inv
        lp[vStop2 + ((kk * 7 + 5) % rank) * k + kk] = inv
        ls[a.utilOpt + kk] = Float(rank)
        ls[a.alphaAct + kk] = 1
        ls[a.betaAct + kk] = 0
        // Per-lane init diversity: every lane previously started at the SAME
        // symmetric near-zero saddle, making ignition a pure noise-stream
        // lottery. A cheap integer hash permutes each option's seeded action
        // slot and sign per lane (deterministic per (lane, option)).
        let h = UInt32(truncatingIfNeeded: lane &* 2_654_435_761 &+ kk &* 40_503 &+ 977)
        let slot = Int(h % UInt32(max(d / 2, 1)))
        let sign: Float = ((h >> 16) & 1) == 0 ? 1 : -1
        // optPolicy offset: rank(value) + k*rank(psi); live-half stride d/2
        lp[rank + k * rank + kk * (d / 2) + slot] = sign * basis
      }
    }
    configB.contents().bindMemory(to: KernelConfig.self, capacity: 1).initialize(to: cfgValue)
  }
  static func applyShapeConstants(_ constants: MTLFunctionConstantValues, _ cfg: KernelConfig) {
    var u: [UInt32] = [
      cfg.obsDim, cfg.rank, cfg.optionCount, cfg.actionParamCount,
      cfg.obsTraceLevels, cfg.linearAnchorCount,
      cfg.paramStride, cfg.stateStride, cfg.uintStride,
    ]
    for i in u.indices { constants.setConstantValue(&u[i], type: .uint, index: 13 + i) }
    var f: [Float] = [cfg.invRank, cfg.invOptionCount, cfg.metaEta]
    for i in f.indices { constants.setConstantValue(&f[i], type: .float, index: 22 + i) }
    var nc = cfg.nurseryCount
    constants.setConstantValue(&nc, type: .uint, index: 28)
  }
  func applyShapeConstants(_ constants: MTLFunctionConstantValues) {
    Self.applyShapeConstants(
      constants, configB.contents().bindMemory(to: KernelConfig.self, capacity: 1).pointee)
  }
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
      case "gridworld-probe": try gridworldProbe()
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
    _ p: Policy, _ seeds: [Int], _ steps: Int, _ root: URL, _ laneEnvs: [String],
    _ ablations: AblationSpec
  ) throws {
    try fleetGPU(p, seeds, steps, root, laneEnvs, ablations)
  }
  // Universal env contract: envId -> kernel envKind (the ONLY env-identity
  // mapping; the agent kernel stays env-blind, Inv-1).
  static func envKind(_ envId: String) throws -> UInt32 {
    switch envId {
    case "HalfCheetah-v4": return 0
    case "FourRooms-cont-v0": return 1
    case "DeepSea-cont-v0": return 2
    case "KeyCorridor-cont-v0": return 3
    case "Acrobot-cont-v0": return 4
    case "MountainCar-cont-v0": return 5
    case "CartPole-cont-v0": return 6
    case "Pendulum-cont-v0": return 7
    case "Chain-cont-v0": return 8
    default:
      throw err("Unknown env '\(envId)'. Known: HalfCheetah-v4, FourRooms-cont-v0, DeepSea-cont-v0, KeyCorridor-cont-v0, Acrobot-cont-v0, MountainCar-cont-v0, CartPole-cont-v0, Pendulum-cont-v0, Chain-cont-v0.")
    }
  }
  static func fleetGPU(
    _ p: Policy, _ seeds: [Int], _ steps: Int, _ root: URL, _ laneEnvs: [String],
    _ ablations: AblationSpec
  ) throws {
    guard laneEnvs.count == seeds.count else {
      throw err("laneEnvs (\(laneEnvs.count)) must match seed count (\(seeds.count)).")
    }
    let laneEnvKinds = try laneEnvs.map(envKind)
    let e = ProcessInfo.processInfo.environment
    let fm = FileManager.default
    let B = seeds.count
    let obsDim = 17, actDim = 6, dPolicy = 12
    guard let dev = MTLCreateSystemDefaultDevice() else { throw err("Metal unavailable.") }
    let engine =
      p == .oak
      ? try Engine(obsDim: obsDim, dPolicy: dPolicy, batch: B, dev: dev, ablations: ablations)
      : nil

    guard let q = engine?.q ?? dev.makeCommandQueue() else {
      throw err("Metal command queue unavailable.")
    }
    let lib = try dev.makeLibrary(URL: try requireFreshMetallib())
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let fn = try lib.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
      return try dev.makeComputePipelineState(function: fn)
    }
    let initPipe = try pipeline("oak_fleet_init")
    let snapPipe = try pipeline("oak_fleet_snapshot")
    // G=32 with scalar env measured fastest at B=1024 (mk5 geometry sweep).
    let autoG = max(2, min(32, B / 32))
    let megaG = e["OAK_MEGA_LANES_PER_TG"].flatMap(Int.init) ?? autoG
    // Scalar-env mode measured 3x faster than coop at high G (mk5 attribution);
    // the coop simd-env path (OAK_MEGA_ENV_COOP, physRows/tgPhys) was deleted
    // in the kernel-slimming pass.
    let megaW = 32 * megaG
    let megaConstants = MTLFunctionConstantValues()
    ablations.apply(to: megaConstants)
    var megaG32 = UInt32(megaG)
    megaConstants.setConstantValue(&megaG32, type: .uint, index: 12)
    if let eng = engine { eng.applyShapeConstants(megaConstants) }
    if e["OAK_PROBE_FTZ"] == "1" {
      var b = true
      megaConstants.setConstantValue(&b, type: .bool, index: 31)
      fputs("OAK_PROBE_FTZ=1: timing instrument, trajectories are garbage\n", stderr)
    }
    if e["OAK_PROBE_FIXEDIDX"] == "1" {
      var b = true
      megaConstants.setConstantValue(&b, type: .bool, index: 30)
      fputs("OAK_PROBE_FIXEDIDX=1: timing instrument, trajectories are garbage\n", stderr)
    }
    let explicitMask = e["OAK_MEGA_STAGE_MASK"].flatMap(UInt32.init)
    if let mask = explicitMask {
      var m = mask
      megaConstants.setConstantValue(&m, type: .uint, index: 25)
      fputs("OAK_MEGA_STAGE_MASK=\(mask): timing instrument, trajectories are garbage\n", stderr)
    }
    let megaPipe = try dev.makeComputePipelineState(
      function: lib.makeFunction(name: "oak_megastep", constantValues: megaConstants))
    // tg: per lane, the K-wide continuation vector plus a 4K prev-dot stash
    // (encode gathers plan's prev-dots; plan reads them gather-free) and a
    // stash-dirty flag (feature replacement forces a plan re-gather).
    // K = 32, Engine's hardcoded option count.
    let megaSlotBytes = (5 * 32 + 1) * megaG * MemoryLayout<Float>.stride
    guard megaG >= 1, megaW <= megaPipe.maxTotalThreadsPerThreadgroup,
      megaSlotBytes <= dev.maxThreadgroupMemoryLength
    else {
      throw err(
        "OAK_MEGA_LANES_PER_TG=\(megaG) needs \(megaW) threads/threadgroup (pipeline max \(megaPipe.maxTotalThreadsPerThreadgroup)) and \(megaSlotBytes) tg bytes (device max \(dev.maxThreadgroupMemoryLength)).")
    }
    func makeBuf(_ len: Int) throws -> MTLBuffer {
      guard let b = dev.makeBuffer(
        length: max(len, 16), options: [.storageModeShared])
      else {
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
    let envKindsBuf = try makeBuf(B * 4)
    do {
      let kp = envKindsBuf.contents().bindMemory(to: UInt32.self, capacity: B)
      for i in 0..<B { kp[i] = laneEnvKinds[i] }
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
    func seedDir(_ lane: Int) throws -> URL {
      let dir = root.appendingPathComponent(laneEnvs[lane], isDirectory: true)
        .appendingPathComponent("seed\(seeds[lane])", isDirectory: true)
      try fm.createDirectory(at: dir, withIntermediateDirectories: true)
      return dir
    }
    let recordAll = e["OAK_RECORD_ACTIONS"] == "1"
    let recordSeeds = Set(
      (e["OAK_RECORD_SEEDS"] ?? "").split(separator: ",").compactMap {
        Int(String($0).trimmingCharacters(in: .whitespaces))
      })
    var recordLanes: [Int] = []
    var recordHandles: [FileHandle] = []
    for (lane, s) in seeds.enumerated() where recordAll || recordSeeds.contains(s) {
      let dir = try seedDir(lane)
      // Record v3: actions.bin records are 16 f32 — ctrl[6], done, pose[9].
      let meta: [String: Any] = ["envId": laneEnvs[lane], "seed": s, "actDim": actDim, "recordVersion": 3]
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
    let recordBuf = try makeBuf(recordLanes.count * recordCapacity * 16 * 4)
    let snapStride = MemoryLayout<LaneSnapshot>.stride
    precondition(snapStride == 280, "LaneSnapshot layout drifted from OakLaneSnapshot")
    let snapBuf = try makeBuf(2 * B * snapStride)
    let snapPtr = snapBuf.contents().bindMemory(to: LaneSnapshot.self, capacity: 2 * B)
    let dumpEvery = max(e["OAK_DUMP_STATE_EVERY"].flatMap(Int.init) ?? 0, 0)
    var hashHandles: [FileHandle?] = Array(repeating: nil, count: B)
    if dumpEvery > 0, engine != nil {
      for (lane, _) in seeds.enumerated() {
        let url = try seedDir(lane).appendingPathComponent("state-hashes.jsonl")
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
    let megaTGBytes = megaSlotBytes
    func encodeMegastep(_ enc: MTLComputeCommandEncoder, _ n: Int) {
      bindFleet(enc, megaPipe)
      enc.setBuffer(recordSlotBuf, offset: 0, index: 9)
      enc.setBuffer(recordBuf, offset: 0, index: 10)
      var n32 = UInt32(n)
      enc.setBytes(&n32, length: 4, index: 11)
      enc.setThreadgroupMemoryLength(megaTGBytes, index: 0)
      enc.dispatchThreadgroups(
        MTLSize(width: (B + megaG - 1) / megaG, height: 1, depth: 1),
        threadsPerThreadgroup: mtlSize(megaW))
      enc.memoryBarrier(scope: [.buffers])
    }
    do {
      guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
        throw err("Failed to encode Metal command.")
      }
      bindFleet(enc, initPipe)
      enc.setBuffer(envKindsBuf, offset: 0, index: 9)
      enc.dispatchThreads(mtlSize(B), threadsPerThreadgroup: laneTG)
      enc.endEncoding()
      cb.commit()
      cb.waitUntilCompleted()
      if let e = cb.error { throw err("Metal command failed: \(e)") }
    }
    final class WriterState: @unchecked Sendable {
      var deadStep: [Int?]
      var deadDetector: [String?]
      var lastDrainStep = 0
      var error: Error?
      var gpuSeconds = 0.0
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
    func health(_ snap: LaneSnapshot, lane: Int) -> [String: Any] {
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
        h["dynaBackupMean"] = f(s.dynaBackupMean)
        h["generateTestEligibleCount"] = f(s.generateTestEligibleCount)
        h["learningMature"] = f(s.learningMature)
      }
      // envInfo channel names are env-conditional (per lane's env): cheetah
      // lanes keep the legacy MuJoCo names; gridworld lanes report their own.
      let infoNames: [String]
      switch laneEnvs[lane] {
      case "FourRooms-cont-v0":
        infoNames = ["info:goal", "info:x", "info:y", "info:since_goal"]
      case "DeepSea-cont-v0":
        infoNames = ["info:reward", "info:row", "info:col", "info:last_action"]
      case "KeyCorridor-cont-v0":
        infoNames = ["info:goal", "info:x", "info:y", "info:since_goal"]
      case "Chain-cont-v0":
        infoNames = ["info:reward", "info:pos", "info:done", "info:unused"]
      case "Acrobot-cont-v0":
        infoNames = ["info:reward", "info:theta1", "info:theta2", "info:height"]
      case "MountainCar-cont-v0":
        infoNames = ["info:reward", "info:pos", "info:vel", "info:done"]
      case "CartPole-cont-v0":
        infoNames = ["info:reward", "info:x", "info:theta", "info:done"]
      case "Pendulum-cont-v0":
        infoNames = ["info:reward", "info:theta", "info:theta_dot", "info:torque"]
      default:
        infoNames = ["info:reward_run", "info:reward_ctrl", "info:x_position", "info:x_velocity"]
      }
      h["envInfo"] = [
        infoNames[0]: infoJSON(snap.info0), infoNames[1]: infoJSON(snap.info1),
        infoNames[2]: infoJSON(snap.info2), infoNames[3]: infoJSON(snap.info3),
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
            } else if Double(snap.paramAbsMax) > 1e16 {
              detector = "paramExplosion"
            } else if !snap.scalars.rho.isFinite {
              detector = "kernelRho"
            }
          }
          if let eng = engine, detector == "paramNonFinite" || detector == "paramExplosion" {
            // Kernel-dated explosion step (rotating-stripe scan, exact to 32 steps)
            // beats the heartbeat-cadence snapshot date.
            let ex = eng.uints.p[lane * eng.uintTotal + 9 + 2 * 32 + 5 * eng.rank]
            if ex != UInt32.max { dStep = Int(ex) }
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
              if let eng = engine {
                let base = lane * eng.uintTotal
                let ex = eng.uints.p[base + 9 + 2 * 32 + 5 * eng.rank]
                let el = eng.uints.p[base + 10 + 2 * 32 + 5 * eng.rank]
                row["explosionStep"] = ex != UInt32.max ? Int(ex) : NSNull()
                row["explosionElem"] = el != UInt32.max ? Int(el) : NSNull()
              }
            }
            deathLines.append(try jsonLine(row))
          }
        }
        var m: [String: Any] = [
          "v": 1, "kind": kind, "envId": laneEnvs[lane], "policy": p.rawValue,
          "ablationLabel": ablations.label, "lane": lane, "seed": seeds[lane],
          "step": Int(snap.steps), "wallSeconds": wall, "rewardSum": Double(snap.rewardSum),
          "doneCount": Int(snap.doneCount), "health": health(snap, lane: lane),
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
          to: Float.self, capacity: recordLanes.count * recordCapacity * 16)
        for (slot, _) in recordLanes.enumerated() {
          var data = Data(capacity: (step - st.lastDrainStep) * (16 * 4))
          for r in st.lastDrainStep..<step {
            let base = (slot * recordCapacity + (r % recordCapacity)) * 16
            for u in 0..<16 {
              var f = rp[base + u]
              withUnsafeBytes(of: &f) { data.append(contentsOf: $0) }
            }
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
      if let rawDir = e["OAK_DEBUG_RAW"] {
        for lane in 0..<min(8, B) {
          let sp = eng.state.p + lane * eng.a.total
          let d = Data(bytes: sp, count: eng.a.total * MemoryLayout<Float>.stride)
          try d.write(to: URL(fileURLWithPath: "\(rawDir)/state-l\(lane)-s\(step).bin"))
          let up = eng.uints.p + lane * eng.uintTotal
          let du = Data(bytes: up, count: eng.uintTotal * MemoryLayout<UInt32>.stride)
          try du.write(to: URL(fileURLWithPath: "\(rawDir)/uints-l\(lane)-s\(step).bin"))
          let pp = eng.params.p + lane * eng.paramTotal
          let dp = Data(bytes: pp, count: eng.paramTotal * MemoryLayout<Float>.stride)
          try dp.write(to: URL(fileURLWithPath: "\(rawDir)/params-l\(lane)-s\(step).bin"))
        }
      }
      for lane in 0..<B {
        guard let h = hashHandles[lane] else { continue }
        let hs = eng.stateHashes(lane)
        let line =
          "{\"seed\":\(seeds[lane]),\"step\":\(step),\"params\":\"\(hs["params"]!)\",\"state\":\"\(hs["state"]!)\",\"uints\":\"\(hs["uints"]!)\",\"scalars\":\"\(hs["scalars"]!)\"}\n"
        try h.write(contentsOf: Data(line.utf8))
      }
    }
    let explicitN = e["OAK_STEPS_PER_CB"].flatMap(Int.init)
    let tuneLock = NSLock()
    var tunedN = max(explicitN ?? 32, 1)
    // Untracked buffers disable automatic hazard ordering BETWEEN command
    // buffers; with 2 chunks in flight the GPU may overlap them. An event
    // chain restores strict chunk order (async submission preserved).
    guard let chunkOrder = dev.makeEvent() else { throw err("Failed to create Metal event.") }
    var chunkSeq: UInt64 = 0
    let inflight = DispatchSemaphore(value: 2)
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
      guard let cb = q.makeCommandBuffer() else {
        throw err("Failed to encode Metal command.")
      }
      cb.encodeWaitForEvent(chunkOrder, value: chunkSeq)
      guard let enc = cb.makeComputeCommandEncoder() else {
        throw err("Failed to encode Metal command.")
      }
      encodeMegastep(enc, target - step)
      if isSnap {
        bindFleet(enc, snapPipe)
        enc.setBuffer(snapBuf, offset: 0, index: 9)
        var base = UInt32(snapHalf * B)
        enc.setBytes(&base, length: 4, index: 10)
        // One threadgroup (= one simdgroup) per lane: the param scan is
        // warp-parallel in-kernel.
        enc.dispatchThreadgroups(mtlSize(B), threadsPerThreadgroup: mtlSize(32))
      }
      enc.endEncoding()
      chunkSeq += 1
      cb.encodeSignalEvent(chunkOrder, value: chunkSeq)
      let thisHalf = snapHalf
      if isSnap { snapHalf ^= 1 }
      let kind = target == steps ? "final" : "hb"
      let tStep = target
      let nSteps = target - step
      cb.addCompletedHandler { cb in
        let gpuT = cb.gpuEndTime - cb.gpuStartTime
        if explicitN == nil, gpuT > 0, nSteps > 0 {
          let fit = max(1, min(Int(0.08 / (gpuT / Double(nSteps))), 1024))
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
        writer.sync {}
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
  // Case-gate probe for the gridworld env contract: reads a
  // gridworld_ref.py case file (header uint32 x3 = n, inW, outW; then n in
  // records, then n golden out records), runs the in-records through the
  // Metal env step (the SAME functions the fleet env window calls), and
  // writes n out records for `gridworld_ref.py --compare cases out`.
  static func gridworldProbe() throws {
    let args = CommandLine.arguments
    guard args.count >= 4 else {
      throw err("Usage: OaK gridworld-probe <cases.bin> <out.bin>")
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: args[2]))
    guard data.count >= 12 else { throw err("Case file too small.") }
    let (n, wi, wo): (Int, Int, Int) = data.withUnsafeBytes { raw in
      (
        Int(raw.load(fromByteOffset: 0, as: UInt32.self)),
        Int(raw.load(fromByteOffset: 4, as: UInt32.self)),
        Int(raw.load(fromByteOffset: 8, as: UInt32.self))
      )
    }
    let isGrid = (wi == 8 && wo == 24) || (wi == 6 && wo == 21) || (wi == 11 && wo == 28)
      || (wi == 2 && wo == 20)
    let isClassic = (wi == 3 && wo == 4) || (wi == 5 && wo == 6) || (wi == 7 && wo == 6)
      || (wi == 4 && wo == 4)
    guard n > 0, isGrid || isClassic else {
      throw err(
        "Bad case header: n=\(n) in=\(wi) out=\(wo) (expected FourRooms 8/24, DeepSea 6/21, KeyCorridor 11/28, Chain 2/20, MountainCar 3/4, Acrobot 5/6, CartPole 7/6, or Pendulum 4/4).")
    }
    guard data.count == 12 + n * wi * 4 + n * wo * 4 else {
      throw err("Case file size mismatch: \(data.count) bytes for n=\(n) in=\(wi) out=\(wo).")
    }
    guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
      throw err("Metal unavailable.")
    }
    let lib = try dev.makeLibrary(URL: requireFreshMetallib())
    let probeName = isGrid ? "oak_gridworld_probe" : "oak_classic_probe"
    guard let fn = lib.makeFunction(name: probeName) else {
      throw err("Missing \(probeName) kernel.")
    }
    let pipe = try dev.makeComputePipelineState(function: fn)
    guard let inBuf = dev.makeBuffer(length: n * wi * 4, options: [.storageModeShared]),
      let outBuf = dev.makeBuffer(length: n * wo * 4, options: [.storageModeShared]),
      let hdrBuf = dev.makeBuffer(length: 12, options: [.storageModeShared])
    else { throw err("Metal buffer allocation failed.") }
    data.withUnsafeBytes { raw in
      memcpy(inBuf.contents(), raw.baseAddress!.advanced(by: 12), n * wi * 4)
    }
    do {
      let hp = hdrBuf.contents().bindMemory(to: UInt32.self, capacity: 3)
      hp[0] = UInt32(n)
      hp[1] = UInt32(wi)
      hp[2] = UInt32(wo)
    }
    guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
      throw err("Failed to encode Metal command.")
    }
    enc.setComputePipelineState(pipe)
    enc.setBuffer(inBuf, offset: 0, index: 0)
    enc.setBuffer(outBuf, offset: 0, index: 1)
    enc.setBuffer(hdrBuf, offset: 0, index: 2)
    let tg = min(64, pipe.maxTotalThreadsPerThreadgroup)
    enc.dispatchThreads(mtlSize(n), threadsPerThreadgroup: mtlSize(tg))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw err("Metal command failed: \(e)") }
    try Data(bytes: outBuf.contents(), count: n * wo * 4)
      .write(to: URL(fileURLWithPath: args[3]))
  }
  static let rolloutLaneStride = 35 * 4
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
        ? Array(seed0..<(seed0 + (env["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 1408)))
        : parsed
    } else {
      seedValues = Array(seed0..<(seed0 + (env["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 1408)))
    }
    let steps = max(env["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 500_000, 1)
    let policies = try (env["OAK_BENCH_POLICIES"] ?? "oak,zero,random").split(separator: ",").map {
      try pol(String($0))
    }
    // Seed-deterministic env schedule (universal env contract). Multi-env is
    // the MANDATORY default (user law 2026-07-19): every run is a superset of
    // the keeper — 1024 cheetah lanes (same seeds as the keeper) plus
    // FourRooms/DeepSea lanes riding free on the GPU's spare capacity.
    // OAK_BENCH_ENVS overrides the schedule; OAK_BENCH_ENV single-env is a
    // hard error (no opt-out of multi-env science).
    let defaultSchedule = "HalfCheetah-v4:1024,KeyCorridor-cont-v0:128,FourRooms-cont-v0:64,DeepSea-cont-v0:64,Acrobot-cont-v0:64,MountainCar-cont-v0:64"
    var raw = env["OAK_BENCH_ENVS"] ?? defaultSchedule
    if raw.trimmingCharacters(in: .whitespaces).isEmpty { raw = defaultSchedule }
    if env["OAK_BENCH_ENV"] != nil {
      throw err("OAK_BENCH_ENV single-env is deleted (user law 2026-07-19): every run is multi-env. Use OAK_BENCH_ENVS to change the schedule.")
    }
    var laneEnvs = [String](repeating: envId, count: seedValues.count)
    var envSchedule: [(envId: String, seedStart: Int, seedEnd: Int)]? = nil
    if !raw.trimmingCharacters(in: .whitespaces).isEmpty {
      var entries: [(String, Int, Int)] = []
      var offset = 0
      for item in raw.split(separator: ",") {
        let parts = item.split(separator: ":")
        guard parts.count == 2, let count = Int(parts[1].trimmingCharacters(in: .whitespaces)),
          count > 0
        else {
          throw err("Bad OAK_BENCH_ENVS entry '\(item)' (expected EnvId:count).")
        }
        let name = String(parts[0]).trimmingCharacters(in: .whitespaces)
        _ = try Main.envKind(name)
        guard offset + count <= seedValues.count else {
          throw err(
            "OAK_BENCH_ENVS schedule overflows the fleet: \(offset + count) lanes > B=\(seedValues.count).")
        }
        for i in 0..<count { laneEnvs[offset + i] = name }
        entries.append((name, seedValues[offset], seedValues[offset + count - 1]))
        offset += count
      }
      guard offset == seedValues.count else {
        throw err("OAK_BENCH_ENVS schedule covers \(offset) of B=\(seedValues.count) lanes.")
      }
      envSchedule = entries
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
    // flopsPerLaneStep: honest per-lane-step multiply+add count (fma = 2
    // flops) of the post-slimming megakernel, by stage, computed from the
    // real config — R = rank, K = 32 options, D = 12, DL = D/2 live dims,
    // A = 64 active rows, Rn nursery. Counts the fma-dominated terms:
    //   encode  value dots 2A | psi dots 2KA | hDot KA | model dots 3KA | norms 6A
    //   plan    prev-side model dots 4KA | polCoef K*DL
    //   learn   policy apply KA*DL | psi/vStop+aux head updates 4KA | value banks 2A
    //   emit    option means KA*DL | blend D*K
    //   dyna    ring vX+xNorm 2A per warp thread | 4 candidates x 3 dots 12A per
    //           thread | copy-2 4KA | dynaAcc apply 2K per thread
    //   window  utility AK | consensus DL*K dir + DL*A apply | nursery Rn/2 | eligScr R
    //   misc    serial scalar/keyboard/curation/stats loops ~4e3
    //   env     fp32 HalfCheetah physics chain ~2e4 (F4-port family estimate)
    // Ablation-aware: each term is counted only when the corresponding
    // function-constant-guarded loop actually compiles in. Unguarded loops
    // (value/psi/hDot dots, norms) run — and are counted — under every
    // ablation. Canonical case: OAK_ABLATE=planning removes all of dyna
    // (oak_dyna_backups early-returns and the dynaAcc apply compiles out).
    let fObs = 17, fD = 12  // HalfCheetah config (same constants fleetGPU uses)
    let fInputW = Arena.input(fObs, fD)
    var fR = 32
    while fR < 2 * fInputW { fR *= 2 }
    let fK = 32, fA = 64, fDL = max(fD / 2, 1)
    let fRn = fR * max(
      ProcessInfo.processInfo.environment["OAK_SUTTON_MULT"].flatMap(Int.init) ?? 1, 1)
    let ablated = Set(ablations.names)
    var fEncode = 2 * fA + 2 * fK * fA + fK * fA + 6 * fA
    if !ablated.contains("option-models") { fEncode += 3 * fK * fA }
    var fPlan = fK * fDL
    if !ablated.contains("option-models") { fPlan += 4 * fK * fA }
    var fLearn = 0
    if !ablated.contains("option-policies") { fLearn += fK * fA * fDL }
    if !ablated.contains("rr-subtasks") { fLearn += 2 * fK * fA }
    if !ablated.contains("option-models") { fLearn += 2 * fK * fA }
    if !ablated.contains("gvf-prediction") { fLearn += 2 * fA }
    var fEmit = fD * fK
    if !ablated.contains("option-policies") { fEmit += fK * fA * fDL }
    let fDyna = ablated.contains("planning") ? 0 : 14 * fA * 32 + 4 * fK * fA + 2 * fK * 32
    var fWindow = fA * fK
    if !ablated.contains("option-policies"), !ablated.contains("generate-and-test") {
      fWindow += fDL * fK + fDL * fA
    }
    if !ablated.contains("generate-and-test") { fWindow += fRn / 2 + fR }
    let fMisc = 4_000
    // Env-window flops are env-conditional per lane's env: the fp32
    // HalfCheetah physics chain ~2e4 (F4-port family estimate); the integer
    // gridworld steps are near-free (obs formulas dominate).
    let agentFlops =
      2 * (fEncode + fPlan + fLearn + fEmit + fDyna + fWindow + fMisc)
    let envFlops: [String: Int] = [
      "HalfCheetah-v4": 20_000, "FourRooms-cont-v0": 40, "DeepSea-cont-v0": 20,
      "KeyCorridor-cont-v0": 60, "Acrobot-cont-v0": 200, "MountainCar-cont-v0": 40,
      "CartPole-cont-v0": 60, "Pendulum-cont-v0": 50, "Chain-cont-v0": 10,
    ]
    let flopsPerLaneStep = agentFlops + (envFlops[envId] ?? 20_000)
    var runMeta: [String: Any] = [
      "fleetSchemaVersion": 1, "command": "benchmark", "envId": envId,
      "batch": seedValues.count, "seeds": seedValues, "steps": steps,
      "policies": policies.map(\.rawValue), "ablation": ablations.names,
      "ablationLabel": ablations.label,
      "flopsPerLaneStep": flopsPerLaneStep,
      "startedAt": ISO8601DateFormatter().string(from: Date()),
    ]
    if let envSchedule {
      runMeta["envSchedule"] = envSchedule.map { e in
        [
          "envId": e.envId, "seedStart": e.seedStart, "seedEnd": e.seedEnd,
          "flopsPerLaneStep": agentFlops + (envFlops[e.envId] ?? 0),
        ] as [String: Any]
      }
    }
    try JSONSerialization.data(
      withJSONObject: runMeta, options: [.prettyPrinted, .sortedKeys]
    ).write(to: dir.appendingPathComponent("meta.json"), options: .atomic)
    let envDesc =
      envSchedule?.map { "\($0.envId):\($0.seedEnd - $0.seedStart + 1)" }.joined(
        separator: ",") ?? envId
    fputs(
      "OaK fleet: B=\(seedValues.count) lanes × \(steps) steps × \(policies.count) policies on \(envDesc) (ablation=\(ablations.label), dir=\(dir.path))\n",
      stderr)
    for p in policies {
      try fleet(p, seedValues, steps, dir, laneEnvs, ablations)
    }
  }
}
