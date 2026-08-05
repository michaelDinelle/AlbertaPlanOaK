import CryptoKit
import Darwin
import Foundation
import Metal
import zlib

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
  var oakOptionReinitCount, oakFeatureReinitCount, terminationRate: Float
  var importanceEss, actorUpdateNormMean, actorSupportEntropy, modelPriorityMax: Float
  var selectedOptionAge, selectedOptionPriority, keyboardL1, actorAdvantageL1: Float
  var selectedOptionIndex, selectedContinuation, selectedKeyboardWeight, mainTd, planTdValue: Float
  var keyboardEffectiveL1, searchControlUseMean, generateTestEligibleCount: Float
}

struct FleetEnvCfg {
  var laneCount, episodeLen, policyMode, recordCapacity: UInt32
  var invObsFloor: Float
}
struct LaneSnapshot {
  var scalars: KernelScalars
  var stats: KernelStats
  var paramAbsMax: Float
  var paramNonFinite: UInt32
  var steps, doneCount: UInt32
  var rewardSum, rho, deltaSum, deltaPeak: Float
  var deathStep, deathDetector: UInt32
  var deathRewardSum, deathRho: Float
}
let fleetLaneStride = 312

struct AblationSpec {
  static let ordered = [
    "feature-construction", "gvf-prediction", "rr-subtasks", "option-policies",
    "option-models", "planning", "option-keyboard",
    "keyboard-to-policy", "search-control", "search-to-generate-test", "generate-and-test",
    "derived-core", "joint-geometry",
  ]
  // Metal function-constant index per organ. The first seven map identity
  // (0-6); the greedy-Q-selection organ ("model-to-keyboard", was index 7) is
  // EXCISED (Phase 7.5, convicted toxic in Phase 6) — removed from the list, so
  // the next four keep their original indices explicitly to avoid shifting.
  // The Phase-7 representation-geometry organs take the free slots 26/27 —
  // 12-25/28-31 are OAK_MEGA_G, OAK_SPEC_*, the stage mask, OAK_SPEC_NURSERY,
  // and the probes (Sources/OakKernel.metal:16-40).
  static let functionConstantIndex = [
    "keyboard-to-policy": 8,
    "search-control": 9,
    "search-to-generate-test": 10,
    "generate-and-test": 11,
    "derived-core": 26,
    "joint-geometry": 27,
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
      constants.setConstantValue(
        &enabled, type: .bool, index: Self.functionConstantIndex[name] ?? idx)
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
    // + planTree (rank, Phase 9.1 planning priority index)
    // + parked (k, Phase 9.3 K-parking hibernation flags)
    uintTotal = 11 + 3 * k + 5 * rank + 4 * rn + 128 + 32 + 32 * rank + rank + k
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

// ---------------------------------------------------------------------------
// Record format v4 — the FROZEN codec of frontend/src/sim/RECORD_V4_SPEC.md.
//
// This encoder and the TS decoder (frontend/src/sim/record4.ts) both implement
// that document verbatim; frontend/scripts/verify-record-v4.ts proves agreement
// against a paired v3 run. Any change to the layout bumps codecVersion.
//
// The kernel ring write is UNTOUCHED (Inv-10 bit-identity of the agent stream):
// recording stays passive, the host drain transcodes v3 rows into v4 chunks.
// ---------------------------------------------------------------------------

enum V4Enc: String { case u8, i16, dx32 }

struct V4Field {
  let idx: Int
  let enc: V4Enc
  let scale: Double
  let wrap: Bool
  init(_ idx: Int, _ enc: V4Enc, _ scale: Double = 0, _ wrap: Bool = false) {
    self.idx = idx
    self.enc = enc
    self.scale = scale
    self.wrap = wrap
  }
  /// Bytes this column occupies for `n` rows.
  func bytes(_ n: Int) -> Int {
    switch enc {
    case .u8: return n
    case .i16: return 2 * n
    case .dx32: return 4 * n
    }
  }
  var json: [String: Any] {
    var o: [String: Any] = ["idx": idx, "enc": enc.rawValue]
    if enc != .u8 { o["scale"] = scale }
    if wrap { o["wrap"] = true }
    return o
  }
}

struct V4Codec {
  let category: String
  /// 0 = ctrl "index" (discrete envs, action in hdr bits 0-3); else the number
  /// of u8-quantized Box ctrl columns.
  let ctrlBox: Int
  let fields: [V4Field]
  let poseConst: [Int: Double]
  var hasDx: Bool { fields.contains { $0.enc == .dx32 } }
  var json: [String: Any] {
    var pc: [String: Any] = [:]
    for (k, v) in poseConst { pc[String(k)] = v }
    return [
      "category": category,
      "ctrl": ctrlBox == 0 ? "index" : ["kind": "u8box", "n": ctrlBox] as [String: Any],
      "poseFields": fields.map(\.json),
      "poseConst": pc,
    ]
  }
  /// The static codec table, verbatim from RECORD_V4_SPEC.md. envKind numbering
  /// is Main.envKind(_:) — the ONLY env-identity mapping (Inv-1 keeps it out of
  /// the agent; the recorder is a host-side wrapper concern like the env window).
  static func table(_ envKind: UInt32) throws -> V4Codec {
    func grid(_ idxs: [Int], _ consts: [Int: Double] = [:]) -> V4Codec {
      V4Codec(
        category: "grid", ctrlBox: 0, fields: idxs.map { V4Field($0, .u8) },
        poseConst: consts)
    }
    switch envKind {
    case 0:  // HalfCheetah — dx-chained rootx, wrapped angles, 6 u8 ctrl columns
      var f = [V4Field(0, .dx32, 0.01), V4Field(1, .i16, 1e-4)]
      for i in 2...8 { f.append(V4Field(i, .i16, 1e-4, true)) }
      return V4Codec(category: "cheetah", ctrlBox: 6, fields: f, poseConst: [:])
    case 1: return grid([0, 1, 2, 3])  // FourRooms
    case 2: return grid([0, 1], [2: 7, 3: 7])  // DeepSea
    case 3: return grid([0, 1, 2, 3, 4, 5, 6, 7])  // KeyCorridor
    case 4:  // Acrobot
      return V4Codec(
        category: "classic", ctrlBox: 0,
        fields: [
          V4Field(0, .i16, 1e-4, true), V4Field(1, .i16, 1e-4, true),
          V4Field(2, .i16, 1e-3), V4Field(3, .i16, 1e-3),
        ], poseConst: [:])
    case 5:  // MountainCar
      return V4Codec(
        category: "classic", ctrlBox: 0,
        fields: [V4Field(0, .i16, 1e-4), V4Field(1, .i16, 1e-5)], poseConst: [2: 0.5])
    case 6:  // CartPole
      return V4Codec(
        category: "classic", ctrlBox: 0,
        fields: [
          V4Field(0, .i16, 1e-4), V4Field(1, .i16, 1e-3),
          V4Field(2, .i16, 1e-4, true), V4Field(3, .i16, 1e-3),
        ], poseConst: [:])
    case 7:  // Pendulum (Box 1 -> one u8 ctrl column)
      return V4Codec(
        category: "classic", ctrlBox: 1,
        fields: [
          V4Field(0, .i16, 1e-4, true), V4Field(1, .i16, 1e-3), V4Field(2, .i16, 1e-3),
        ], poseConst: [:])
    case 8: return grid([0], [1: 15])  // Chain
    case 9: return grid([0, 1], [2: 15, 3: 15])  // LargeDeepSea
    case 10: return grid([0, 1, 2, 3, 4, 5, 6, 7])  // MultiKeyCorridor
    case 11: return grid([0, 2], [1: 31])  // MemoryChain
    case 12: return grid([0, 1, 2, 3])  // ProceduralMaze
    case 13: return grid([0, 1, 2, 3, 4])  // HighDimMaze
    case 14: return grid([0, 1, 2, 3])  // POMaze
    case 15: return grid([0, 2], [1: 31])  // DelayedChain
    default: throw err("No record-v4 codec for envKind \(envKind).")
    }
  }
}

/// One recorded lane's v4 file. Chunk-per-drain, append-only; a torn tail (killed
/// run) is detectable by the reader (short header / missing bytes / crc).
/// Instances are touched by exactly one drain worker at a time (the drain is
/// serialized on the writer queue, then fanned out one lane per iteration).
final class RecordV4Writer: @unchecked Sendable {
  static let fileMagic: UInt32 = 0x344B_414F  // "OAK4"
  static let chunkMagic: UInt32 = 0x344B_4843  // "CHK4"
  static let codecVersion = 1

  let handle: FileHandle
  let codec: V4Codec
  let metaURL: URL
  let envId: String
  let seed: Int
  let actDim: Int
  private var buf: [UInt8] = []
  private var chunks: [[Int]] = []
  private var fileOffset = 0
  private var totalSteps = 0
  private var deflateVerified = false

  init(
    handle: FileHandle, envKind: UInt32, metaURL: URL, envId: String, seed: Int, actDim: Int
  ) throws {
    self.handle = handle
    self.codec = try V4Codec.table(envKind)
    self.metaURL = metaURL
    self.envId = envId
    self.seed = seed
    self.actDim = actDim
    var head = Data(capacity: 16)
    for v: UInt32 in [Self.fileMagic, 4, envKind, UInt32(Self.codecVersion)] {
      var x = v.littleEndian
      withUnsafeBytes(of: &x) { head.append(contentsOf: $0) }
    }
    try handle.write(contentsOf: head)
    fileOffset = head.count
    try writeMeta(final: false)
  }

  func writeMeta(final: Bool) throws {
    var meta: [String: Any] = [
      "envId": envId, "seed": seed, "actDim": actDim, "recordVersion": 4,
      "codecVersion": Self.codecVersion, "codec": codec.json,
    ]
    if final {
      meta["steps"] = totalSteps
      meta["chunks"] = chunks
    }
    try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
      .write(to: metaURL, options: .atomic)
  }

  /// Transcode `[stepStart, stepStart+stepCount)` of the kernel ring into one
  /// compressed chunk and append it. `rewardDelta` is the lane's rewardSum delta
  /// over THIS CHUNK (NaN when the caller cannot measure exactly that interval).
  func encodeChunk(
    _ rp: UnsafePointer<Float>, slot: Int, capacity: Int, stepStart: Int, stepCount: Int,
    rewardDelta: Float
  ) throws {
    guard stepCount > 0 else { return }
    let nBox = codec.ctrlBox
    // Column offsets: hdr | box ctrl columns | chunkBaseX (f64, cheetah) | pose
    // columns in descriptor order | u32 escape count | escape rows.
    let baseOff = stepCount * (1 + nBox)
    var cursor = baseOff + (codec.hasDx ? 8 : 0)
    var off: [Int] = []
    off.reserveCapacity(codec.fields.count)
    for f in codec.fields {
      off.append(cursor)
      cursor += f.bytes(stepCount)
    }
    let fixed = cursor + 4
    if buf.count < fixed { buf = [UInt8](repeating: 0, count: fixed) }

    var escRows: [Int] = []
    var doneCount = 0
    let fields = codec.fields
    buf.withUnsafeMutableBufferPointer { bp in
      let b = bp.baseAddress!
      memset(b, 0, fixed)
      var recon = 0.0
      if codec.hasDx, let dxIdx = fields.first(where: { $0.enc == .dx32 })?.idx {
        // chunkBaseX: the EXACT f64 promotion of the first row's pose value.
        let r0 = (slot * capacity + (stepStart % capacity)) * 16
        recon = Double(rp[r0 + 7 + dxIdx])
        var bits = recon.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { memcpy(b + baseOff, $0.baseAddress!, 8) }
      }
      for i in 0..<stepCount {
        let base = (slot * capacity + ((stepStart + i) % capacity)) * 16
        var esc = false
        for u in 0..<16 where !rp[base + u].isFinite { esc = true }
        var hdr: UInt8 = 0
        // done is BIT-EXACT by contract: anything that is not exactly 0 or 1
        // escapes rather than being rounded into the flag.
        let doneV = rp[base + 6]
        if doneV == 1 {
          hdr |= 0x10
          doneCount += 1
        } else if doneV != 0 {
          esc = true
        }
        if nBox == 0 {
          let a = Double(rp[base])
          if a.isFinite, a == a.rounded(), a >= 0, a <= 15 {
            hdr |= UInt8(a) & 0x0F
          } else {
            esc = true
          }
        } else {
          for c in 0..<nBox {
            let v = Double(rp[base + c])
            let q = v.isFinite ? ((min(max(v, -1), 1) + 1) * 127.5).rounded() : 0
            b[stepCount * (1 + c) + i] = UInt8(min(max(q, 0), 255))
          }
        }
        for (j, f) in fields.enumerated() {
          let v = Double(rp[base + 7 + f.idx])
          switch f.enc {
          case .u8:
            if !(v.isFinite && v == v.rounded() && v >= 0 && v <= 255) { esc = true }
            b[off[j] + i] = UInt8(v.isFinite ? min(max(v.rounded(), 0), 255) : 0)
          case .i16:
            var w = v
            if f.wrap, w.isFinite {
              // floor-based positive wrap into [-pi, pi) — identical in Swift and JS.
              w -= 2 * Double.pi * ((w + Double.pi) / (2 * Double.pi)).rounded(.down)
            }
            var q = w.isFinite ? (w / f.scale).rounded() : 0
            if !(q >= -32767 && q <= 32767) { esc = true }
            q = min(max(q, -32767), 32767)
            var qi = Int16(q).littleEndian
            withUnsafeBytes(of: &qi) { memcpy(b + off[j] + 2 * i, $0.baseAddress!, 2) }
          case .dx32:
            var q = 0.0
            if v.isFinite, recon.isFinite {
              q = ((v - recon) / f.scale).rounded()
              if !(q >= -2_147_483_647 && q <= 2_147_483_647) { esc = true }
              q = min(max(q, -2_147_483_647), 2_147_483_647)
            } else {
              esc = true  // a poisoned chain cannot reconstruct: escape verbatim
            }
            // The chain uses the CLAMPED value, so encoder and decoder stay in
            // lockstep even across escaped rows.
            recon += q * f.scale
            var qi = Int32(q).littleEndian
            withUnsafeBytes(of: &qi) { memcpy(b + off[j] + 4 * i, $0.baseAddress!, 4) }
          }
        }
        if esc {
          hdr |= 0x20
          escRows.append(i)
        }
        b[i] = hdr
      }
      var cnt = UInt32(escRows.count).littleEndian
      withUnsafeBytes(of: &cnt) { memcpy(b + fixed - 4, $0.baseAddress!, 4) }
    }

    var payload = Data(capacity: fixed + escRows.count * 68)
    buf.withUnsafeBufferPointer { payload.append($0.baseAddress!, count: fixed) }
    for i in escRows {
      var idx = UInt32(i).littleEndian
      withUnsafeBytes(of: &idx) { payload.append(contentsOf: $0) }
      let base = (slot * capacity + ((stepStart + i) % capacity)) * 16
      payload.append(UnsafeRawPointer(rp + base).assumingMemoryBound(to: UInt8.self), count: 64)
    }

    // Compression's COMPRESSION_ZLIB is RFC-1951 RAW DEFLATE (no zlib wrapper),
    // which is what the TS side inflates (inflateRawSync / 'deflate-raw').
    // Verified at runtime on this lane's first chunk rather than trusted.
    let comp = try (payload as NSData).compressed(using: .zlib) as Data
    if !deflateVerified {
      deflateVerified = true
      let wrapped = comp.count >= 2
        && comp.withUnsafeBytes { p -> Bool in
          let b0 = Int(p[0]), b1 = Int(p[1])
          return b0 == 0x78 && ((b0 << 8) | b1) % 31 == 0
        }
      if wrapped {
        throw err("record v4: Compression .zlib emitted a ZLIB-WRAPPED stream; the spec requires raw DEFLATE (frontend/src/sim/RECORD_V4_SPEC.md).")
      }
      let back = try (comp as NSData).decompressed(using: .zlib) as Data
      guard back == payload else {
        throw err("record v4: deflate round-trip mismatch (\(back.count) vs \(payload.count) bytes).")
      }
    }
    let crc = comp.withUnsafeBytes { p in
      UInt32(truncatingIfNeeded: crc32(0, p.bindMemory(to: UInt8.self).baseAddress, uInt(comp.count)))
    }
    var out = Data(capacity: 32 + comp.count)
    for v: UInt32 in [
      Self.chunkMagic, UInt32(stepStart), UInt32(stepCount), UInt32(payload.count),
      UInt32(comp.count), crc, UInt32(doneCount), rewardDelta.bitPattern,
    ] {
      var x = v.littleEndian
      withUnsafeBytes(of: &x) { out.append(contentsOf: $0) }
    }
    out.append(comp)
    try handle.write(contentsOf: out)
    chunks.append([stepStart, stepCount, fileOffset, 32 + comp.count])
    fileOffset += 32 + comp.count
    totalSteps += stepCount
  }

  /// Compressed bytes written so far (chunk headers included) — drain telemetry.
  var bytesWritten: Int { fileOffset }
}

/// Continuous metrics tape (OAKT v1) — the replacement for fleet.jsonl's
/// per-lane heartbeat snapshots (METRICS LAW 2026-08-05). One file per run
/// root per policy (`metrics.<policy>.oakt`); one row per lane per
/// command-buffer window, so the on-disk curve resolution is the CB cadence
/// (~1024 steps), not the heartbeat. Self-describing: the header carries the
/// field table as JSON, so readers never hard-code the schema. Windows are
/// buffered in memory and flushed as one raw-DEFLATE segment per heartbeat
/// (same cadence, same writer queue, same codec family as record v4 — zero
/// new I/O rhythm). Torn tails drop segment-by-segment via crc, exactly like
/// v4 chunks. Layout, all little-endian:
///   FileHeader: u32 magic "OAKT", u32 version=1, u32 laneCount,
///               u32 jsonLen, jsonLen bytes of {"fields":[...names]}
///   Segment:    u32 magic "SEGT", u32 windowCount, u32 rawBytes,
///               u32 compBytes, u32 crc32(compressed), then compBytes of
///               raw DEFLATE. Inflated payload = windowCount windows, each:
///               u32 stepStart, u32 stepCount, then fields × f32[laneCount].
final class MetricsTapeWriter {
  static let fileMagic: UInt32 = 0x544B_414F  // "OAKT"
  static let segMagic: UInt32 = 0x54_4745_53  // "SEGT"
  /// The curated continuous set (everything else was removed as dead weight
  /// or lives only in the final fleet.jsonl row / deaths.jsonl certificate).
  static let fields: [String] = [
    "rewardSum", "doneCount", "rewardRate", "kernelRho", "deltaSum",
    "deltaPeak", "mainTd", "actorSupportEntropy", "terminationRate",
    "keyboardEffectiveL1", "keyboardL1", "actorAdvantageL1",
    "modelPriorityMax", "importanceEss", "actorUpdateNormMean",
    "searchControlUseMean", "selectedOptionPriority", "selectedContinuation",
    "generateTestEligibleCount", "featureReinitCount", "optionReinitCount",
    "nfEncode", "nfLearn", "nfPlan", "nfTdPsi",
    // -1 sentinel on per-CB windows ("not scanned"); the real O(paramStride)
    // scan value on heartbeat windows only.
    "paramAbsMax",
  ]
  private let handle: FileHandle
  private let laneCount: Int
  private var pending = Data()
  private var pendingWindows = 0
  private var deflateVerified = false

  init(url: URL, laneCount: Int) throws {
    FileManager.default.createFile(atPath: url.path, contents: nil)
    self.handle = try FileHandle(forWritingTo: url)
    self.laneCount = laneCount
    let json = try JSONSerialization.data(withJSONObject: ["fields": Self.fields])
    var head = Data()
    for v: UInt32 in [Self.fileMagic, 1, UInt32(laneCount), UInt32(json.count)] {
      var x = v.littleEndian
      withUnsafeBytes(of: &x) { head.append(contentsOf: $0) }
    }
    head.append(json)
    try handle.write(contentsOf: head)
  }

  /// One window row per lane from the CB-boundary snapshot. Writer queue only.
  func append(snap: UnsafeMutablePointer<LaneSnapshot>, half: Int,
              stepStart: Int, stepCount: Int) {
    for v: UInt32 in [UInt32(stepStart), UInt32(stepCount)] {
      var x = v.littleEndian
      withUnsafeBytes(of: &x) { pending.append(contentsOf: $0) }
    }
    var col = [Float](repeating: 0, count: laneCount)
    func push(_ get: (LaneSnapshot) -> Float) {
      for l in 0..<laneCount { col[l] = get(snap[half * laneCount + l]) }
      col.withUnsafeBufferPointer {
        pending.append(UnsafeRawPointer($0.baseAddress!).assumingMemoryBound(to: UInt8.self),
                       count: laneCount * 4)
      }
    }
    push { $0.rewardSum }
    push { Float($0.doneCount) }
    push { $0.rho }
    push { $0.scalars.rho }
    push { $0.deltaSum }
    push { $0.deltaPeak }
    push { $0.stats.mainTd }
    push { $0.stats.actorSupportEntropy }
    push { $0.stats.terminationRate }
    push { $0.stats.keyboardEffectiveL1 }
    push { $0.stats.keyboardL1 }
    push { $0.stats.actorAdvantageL1 }
    push { $0.stats.modelPriorityMax }
    push { $0.stats.importanceEss }
    push { $0.stats.actorUpdateNormMean }
    push { $0.stats.searchControlUseMean }
    push { $0.stats.selectedOptionPriority }
    push { $0.stats.selectedContinuation }
    push { $0.stats.generateTestEligibleCount }
    push { $0.stats.oakFeatureReinitCount }
    push { $0.stats.oakOptionReinitCount }
    push { $0.stats.encodeNonFiniteCount }
    push { $0.stats.learnNonFiniteCount }
    push { $0.stats.planNonFiniteCount }
    push { $0.stats.tdPsiNonFiniteCount }
    push { $0.paramNonFinite != 0 ? Float.infinity : $0.paramAbsMax }
    pendingWindows += 1
  }

  /// Compress and write all buffered windows as one segment. Writer queue only.
  func flush() throws {
    guard pendingWindows > 0 else { return }
    let comp = try (pending as NSData).compressed(using: .zlib) as Data
    if !deflateVerified {
      deflateVerified = true
      let back = try (comp as NSData).decompressed(using: .zlib) as Data
      guard back == pending else {
        throw err("metrics tape: deflate round-trip mismatch.")
      }
    }
    let crc = comp.withUnsafeBytes { p in
      UInt32(truncatingIfNeeded: crc32(0, p.bindMemory(to: UInt8.self).baseAddress, uInt(comp.count)))
    }
    var out = Data(capacity: 20 + comp.count)
    for v: UInt32 in [
      Self.segMagic, UInt32(pendingWindows), UInt32(pending.count), UInt32(comp.count), crc,
    ] {
      var x = v.littleEndian
      withUnsafeBytes(of: &x) { out.append(contentsOf: $0) }
    }
    out.append(comp)
    try handle.write(contentsOf: out)
    pending = Data()
    pendingWindows = 0
  }
}

@main struct Main {
  /// Set at launch when a multi-policy workload makes default-on recording
  /// unsafe (all policies share one <envId>/seed<N>/actions.bin — the
  /// truncation trap). An EXPLICIT OAK_RECORD_ACTIONS=1 request still throws.
  nonisolated(unsafe) static var recordingDefaultSuppressed = false

  static func main() {
    _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    do {
      switch CommandLine.arguments.dropFirst().first {
      case "benchmark": try benchmark()
      case "physics-probe": try physicsProbe()
      case "gridworld-probe": try gridworldProbe()
      case "record-v4-probe": try recordV4Probe()
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
  // Universal env contract: envId <-> kernel envKind (the ONLY env-identity
  // mapping; the agent kernel stays env-blind, Inv-1). Index = envKind.
  static let envIdsByKind: [String] = [
    "HalfCheetah-v4", "FourRooms-cont-v0", "DeepSea-cont-v0", "KeyCorridor-cont-v0",
    "Acrobot-cont-v0", "MountainCar-cont-v0", "CartPole-cont-v0", "Pendulum-cont-v0",
    "Chain-cont-v0", "LargeDeepSea-cont-v0", "MultiKeyCorridor-cont-v0", "MemoryChain-cont-v0",
    "ProceduralMaze-cont-v0", "HighDimMaze-cont-v0", "POMaze-cont-v0", "DelayedChain-cont-v0",
  ]
  static func envKind(_ envId: String) throws -> UInt32 {
    guard let i = envIdsByKind.firstIndex(of: envId) else {
      throw err("Unknown env '\(envId)'. Known: \(envIdsByKind.joined(separator: ", ")).")
    }
    return UInt32(i)
  }
  static func envId(kind: UInt32) throws -> String {
    guard Int(kind) < envIdsByKind.count else {
      throw err("Unknown envKind \(kind) (0..\(envIdsByKind.count - 1)).")
    }
    return envIdsByKind[Int(kind)]
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
    // RECORDING LAW (2026-08-05): every fleet run records every lane by
    // default (v4). Opt out with OAK_RECORD_ACTIONS=0 (OAK_RECORD_SEEDS then
    // still selects individual lanes); multi-policy workloads auto-suppress
    // at the launch guard via recordingDefaultSuppressed.
    let recordAll =
      e["OAK_RECORD_ACTIONS"] != "0" && !Main.recordingDefaultSuppressed
    let recordSeeds = Set(
      (e["OAK_RECORD_SEEDS"] ?? "").split(separator: ",").compactMap {
        Int(String($0).trimmingCharacters(in: .whitespaces))
      })
    // Recorded stream format: 4 = the columnar/quantized codec (DEFAULT —
    // frontend/src/sim/RECORD_V4_SPEC.md; RECORDING LAW 2026-08-05: ~22x
    // smaller — 2.7 GiB vs 61 GiB for a 1024x1M fleet — at measured SPS
    // parity), 3 = flat f32[16] rows, the EXPLICIT certification format for
    // gates whose tolerances sit below the v4 quantization contract:
    // `tools/physics_parity.py --classic-tost` (1e-4 one-step parity) and
    // `frontend/test/fixtures/gen_classic.py` (1e-6 cheetah replay) REFUSE
    // v4 lanes with a clear error. Every other Python consumer reads both
    // via `tools/record4.py::load_rows` (dispatches on recordVersion); the
    // frontend theater has always read both.
    let recordFormat = e["OAK_RECORD_FORMAT"] ?? "4"
    guard recordFormat == "3" || recordFormat == "4" else {
      throw err("OAK_RECORD_FORMAT must be 3 or 4 (got '\(recordFormat)').")
    }
    var recordLanes: [Int] = []
    var recordHandles: [FileHandle] = []
    var recordV4: [RecordV4Writer] = []
    for (lane, s) in seeds.enumerated() where recordAll || recordSeeds.contains(s) {
      let dir = try seedDir(lane)
      let url = dir.appendingPathComponent("actions.bin")
      let metaURL = dir.appendingPathComponent("meta.json")
      fm.createFile(atPath: url.path, contents: nil)
      let h = try FileHandle(forWritingTo: url)
      if recordFormat == "4" {
        recordV4.append(
          try RecordV4Writer(
            handle: h, envKind: laneEnvKinds[lane], metaURL: metaURL, envId: laneEnvs[lane],
            seed: s, actDim: actDim))
      } else {
        // Record v3: actions.bin records are 16 f32 — ctrl[6], done, pose[9].
        let meta: [String: Any] = ["envId": laneEnvs[lane], "seed": s, "actDim": actDim, "recordVersion": 3]
        try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
          .write(to: metaURL, options: .atomic)
      }
      recordHandles.append(h)
      recordLanes.append(lane)
    }
    // Ring flush cadence. Default = the heartbeat (the historical drain point);
    // OAK_RECORD_DRAIN_STEPS adds intermediate flush-only boundaries, which is
    // what keeps the ring (and its allocation) small at long horizons.
    let drainEvery = max(e["OAK_RECORD_DRAIN_STEPS"].flatMap(Int.init) ?? hb, 1)
    let recordCapacity = recordLanes.isEmpty ? 0 : 2 * drainEvery
    let recordSlotBuf = try makeBuf(B * 4)
    do {
      let rp = recordSlotBuf.contents().bindMemory(to: Int32.self, capacity: B)
      for i in 0..<B { rp[i] = -1 }
      for (slot, lane) in recordLanes.enumerated() { rp[lane] = Int32(slot) }
    }
    let recordBuf = try makeBuf(recordLanes.count * recordCapacity * 16 * 4)
    let snapStride = MemoryLayout<LaneSnapshot>.stride
    precondition(snapStride == 152, "LaneSnapshot layout drifted from OakLaneSnapshot")
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
    let tape = try MetricsTapeWriter(
      url: root.appendingPathComponent("metrics.\(p.rawValue).oakt"), laneCount: B)
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
      /// Per-RECORDED-lane rewardSum at the last snapshot drain, and the step it
      /// was taken at — the v4 chunk header's rewardDelta baseline. A chunk may
      /// carry a finite delta ONLY when `recPrevStep` is its own first step.
      var recPrevReward: [Double]
      var recPrevStep = 0
      init(_ b: Int, _ rec: Int) {
        deadStep = Array(repeating: nil, count: b)
        deadDetector = Array(repeating: nil, count: b)
        prevRewardSum = Array(repeating: 0, count: b)
        recPrevReward = Array(repeating: 0, count: rec)
      }
    }
    let st = WriterState(B, recordLanes.count)
    let writer = DispatchQueue(label: "oak.fleet.gpu.writer")
    let detectorNames: [UInt32: String] = [1: "envReward", 2: "policyMean", 3: "envObs"]
    let wallStart = DispatchTime.now().uptimeNanoseconds
    func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1e9 }
    // METRICS LAW (2026-08-05): the continuous series live in the metrics
    // tape; this block survives only on the FINAL fleet.jsonl row. envInfo,
    // the structurally-zero stats, and the never-read fields were deleted at
    // the source (kernel + lane struct), not just here.
    func health(_ snap: LaneSnapshot, lane: Int) -> [String: Any] {
      let adm = Double(snap.deltaSum) / Double(max(Int(snap.steps), 1))
      var h: [String: Any] = [
        "rewardRate": snap.rho.isFinite ? Double(snap.rho) : NSNull(),
        "actionDeltaL2Mean": adm.isFinite ? adm : NSNull(),
        "actionDeltaPeak": snap.deltaPeak.isFinite ? Double(snap.deltaPeak) : NSNull(),
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
        h["importanceEss"] = f(s.importanceEss)
        h["actorUpdateNormMean"] = f(s.actorUpdateNormMean)
        h["actorSupportEntropy"] = f(s.actorSupportEntropy)
        h["modelPriorityMax"] = f(s.modelPriorityMax)
        h["selectedOptionAge"] = f(s.selectedOptionAge)
        h["selectedOptionPriority"] = f(s.selectedOptionPriority)
        h["keyboardL1"] = f(s.keyboardL1)
        h["actorAdvantageL1"] = f(s.actorAdvantageL1)
        h["selectedOptionIndex"] = f(s.selectedOptionIndex)
        h["selectedContinuation"] = f(s.selectedContinuation)
        h["selectedKeyboardWeight"] = f(s.selectedKeyboardWeight)
        h["mainTd"] = f(s.mainTd)
        h["planTd"] = f(s.planTdValue)
        h["keyboardEffectiveL1"] = f(s.keyboardEffectiveL1)
        h["searchControlUseMean"] = f(s.searchControlUseMean)
        h["generateTestEligibleCount"] = f(s.generateTestEligibleCount)
      }
      return h
    }
    // Errors out of the per-lane fan-out (DispatchQueue.concurrentPerform can
    // neither throw nor safely write a captured Array from several threads).
    final class DrainErr: @unchecked Sendable {
      private let lock = NSLock()
      private var first: Error?
      func record(_ e: Error) {
        lock.lock()
        if first == nil { first = e }
        lock.unlock()
      }
      func take() -> Error? {
        lock.lock()
        defer { lock.unlock() }
        return first
      }
    }
    /// Flush ring rows [lastDrainStep, step) to disk. `half` is the snapshot
    /// buffer half when this drain rides a heartbeat (the only place a lane
    /// rewardSum exists, so the only place the v4 chunk header's rewardDelta can
    /// be finite); nil on flush-only boundaries.
    func drainRecords(step: Int, half: Int?) throws {
      guard !recordLanes.isEmpty, step > st.lastDrainStep else { return }
      let t0 = DispatchTime.now().uptimeNanoseconds
      let start = st.lastDrainStep
      let count = step - start
      let rp = recordBuf.contents().bindMemory(
        to: Float.self, capacity: recordLanes.count * recordCapacity * 16)
      if recordV4.isEmpty {
        for (slot, _) in recordLanes.enumerated() {
          var data = Data(capacity: count * (16 * 4))
          for r in start..<step {
            let base = (slot * recordCapacity + (r % recordCapacity)) * 16
            for u in 0..<16 {
              var f = rp[base + u]
              withUnsafeBytes(of: &f) { data.append(contentsOf: $0) }
            }
          }
          try recordHandles[slot].write(contentsOf: data)
        }
        st.lastDrainStep = step
        if timing {
          let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
          fputs(
            String(
              format: "[record v3] step %d | %d lanes x %d rows | %.1f ms | %.2f MiB\n",
              step, recordLanes.count, count, ms,
              Double(recordLanes.count * count * 64) / 1_048_576), stderr)
        }
        return
      }
      // rewardDelta is the delta over THIS CHUNK (RECORD_V4_SPEC.md), or NaN.
      // The host only holds a lane rewardSum at heartbeat snapshots, so a chunk
      // owns a real delta only when the previous snapshot landed exactly on the
      // chunk's first step — true for every chunk at the default cadence
      // (drainEvery == heartbeat). With OAK_RECORD_DRAIN_STEPS < heartbeat the
      // interval spans several chunks and NONE of them owns it, so they all
      // carry NaN ("unavailable"); an interval total is never mislabelled as a
      // chunk total on the chunk that happens to close the interval.
      var deltas = [Float](repeating: .nan, count: recordLanes.count)
      if let half {
        let exact = st.recPrevStep == start
        for (slot, lane) in recordLanes.enumerated() {
          let rs = Double(snapPtr[half * B + lane].rewardSum)
          if exact { deltas[slot] = Float(rs - st.recPrevReward[slot]) }
          st.recPrevReward[slot] = rs
        }
        st.recPrevStep = step
      }
      let before = recordV4.reduce(0) { $0 + $1.bytesWritten }
      let box = DrainErr()
      let writers = recordV4
      let cap = recordCapacity
      DispatchQueue.concurrentPerform(iterations: writers.count) { slot in
        do {
          try writers[slot].encodeChunk(
            rp, slot: slot, capacity: cap, stepStart: start, stepCount: count,
            rewardDelta: deltas[slot])
        } catch {
          box.record(error)
        }
      }
      if let e = box.take() { throw e }
      st.lastDrainStep = step
      if timing {
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let wrote = recordV4.reduce(0) { $0 + $1.bytesWritten } - before
        let raw = recordLanes.count * count * 64
        fputs(
          String(
            format: "[record v4] step %d | %d lanes x %d rows | %.1f ms | %.2f MiB -> %.2f MiB (%.1fx)\n",
            step, recordLanes.count, count, ms, Double(raw) / 1_048_576,
            Double(wrote) / 1_048_576, wrote > 0 ? Double(raw) / Double(wrote) : 0), stderr)
      }
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
        // METRICS LAW (2026-08-05): per-lane heartbeat rows are gone — the
        // continuous series live in the metrics tape. fleet.jsonl keeps only
        // the FINAL certificate row per lane (plus deaths.jsonl, unchanged).
        if kind == "final" {
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
      }
      if !deathLines.isEmpty { try deathsOut.write(contentsOf: deathLines) }
      if !lines.isEmpty { try fleetOut.write(contentsOf: lines) }
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
      try drainRecords(step: step, half: half)
      try tape.flush()
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
    // Tape window: the autotuner can settle on tiny CBs (~33 steps at
    // B=1024, 80 ms GPU each), so the tape coalesces to >= this many steps
    // per row instead of one row per CB — resolution stays ~1k steps while
    // the snapshot dispatch is skipped entirely on non-tape CBs.
    let tapeWindow = max(e["OAK_TAPE_WINDOW_STEPS"].flatMap(Int.init) ?? 1024, 1)
    var lastTapeStep = 0
    var step = 0
    while step < steps {
      tuneLock.lock()
      let n = tunedN
      tuneLock.unlock()
      let nextHb = min(((step / hb) + 1) * hb, steps)
      var boundary = nextHb
      if dumpEvery > 0 { boundary = min(boundary, ((step / dumpEvery) + 1) * dumpEvery) }
      if !recordLanes.isEmpty {
        boundary = min(boundary, ((step / drainEvery) + 1) * drainEvery)
      }
      let target = min(boundary, step + n)
      let isSnap = target == nextHb
      // Flush-only ring boundary: same snapSlots gating as a heartbeat (the ring
      // holds 2 x drainEvery rows, so the next chunk may not be encoded until the
      // previous drain has copied its rows out), no fleet/aggregate rows.
      let isDrain = !recordLanes.isEmpty && !isSnap && target % drainEvery == 0
      let isDump = dumpEvery > 0 && target % dumpEvery == 0
      // METRICS LAW (2026-08-05): a tape row lands when >= tapeWindow steps
      // have elapsed since the last one (and on every heartbeat). Only those
      // CBs carry the snapshot dispatch and take a snapshot half.
      let isTape = isSnap || target - lastTapeStep >= tapeWindow
      inflight.wait()
      if let err = writer.sync(execute: { st.error }) { throw err }
      if isTape || isDrain { snapSlots.wait() }
      guard let cb = q.makeCommandBuffer() else {
        throw err("Failed to encode Metal command.")
      }
      cb.encodeWaitForEvent(chunkOrder, value: chunkSeq)
      guard let enc = cb.makeComputeCommandEncoder() else {
        throw err("Failed to encode Metal command.")
      }
      encodeMegastep(enc, target - step)
      if isTape {
        bindFleet(enc, snapPipe)
        enc.setBuffer(snapBuf, offset: 0, index: 9)
        var base = UInt32(snapHalf * B)
        enc.setBytes(&base, length: 4, index: 10)
        // The O(paramStride) paramAbsMax scan stays at heartbeat cadence; the
        // tape-window snapshot skips it (fullScan = 0 -> sentinel -1).
        var fullScan: UInt32 = isSnap ? 1 : 0
        enc.setBytes(&fullScan, length: 4, index: 11)
        // One threadgroup (= one simdgroup) per lane: the param scan is
        // warp-parallel in-kernel.
        enc.dispatchThreadgroups(mtlSize(B), threadsPerThreadgroup: mtlSize(32))
      }
      enc.endEncoding()
      chunkSeq += 1
      cb.encodeSignalEvent(chunkOrder, value: chunkSeq)
      let thisHalf = snapHalf
      if isTape { snapHalf ^= 1 }
      let kind = target == steps ? "final" : "hb"
      let tStep = target
      let nSteps = target - step
      let wStart = lastTapeStep
      if isTape { lastTapeStep = target }
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
          if isTape, st.error == nil {
            tape.append(snap: snapPtr, half: thisHalf, stepStart: wStart, stepCount: tStep - wStart)
          }
          if isSnap {
            if st.error == nil {
              do { try writeSnapshot(half: thisHalf, step: tStep, kind: kind) } catch {
                st.error = error
              }
            }
          } else if isDrain {
            if st.error == nil {
              do { try drainRecords(step: tStep, half: nil) } catch { st.error = error }
            }
          }
          if isTape || isDrain { snapSlots.signal() }
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
    // Finalize: the chunk table + total step count are a reader HINT written at
    // close (an in-flight or torn file is still readable by sequential scan).
    for w in recordV4 { try w.writeMeta(final: true) }
  }
  // Case-gate probe for the record-v4 ENCODER. `RecordV4Writer.encodeChunk`'s
  // escape branch is unreachable from a healthy fleet (it needs a non-finite row
  // or a field past its integer ceiling), so a fleet A/B can only ever prove the
  // happy path. This feeds the REAL encoder a hand-built ring — the caller's
  // escape zoo — and writes a real <out-dir>/{actions.bin,meta.json} pair for
  // frontend/scripts/verify-record-v4.ts to decode and compare.
  //
  // Case file: u32 envKind, u32 rowCount, u32 chunkSteps, then rowCount x 16 f32
  // in v3 row order (ctrl[6], done, pose[9]).
  static func recordV4Probe() throws {
    let args = CommandLine.arguments
    guard args.count >= 4 else {
      throw err("Usage: OaK record-v4-probe <cases.bin> <out-dir>")
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: args[2]))
    guard data.count >= 12 else { throw err("record-v4-probe: case file too small.") }
    let (kind, rowCount, chunkSteps): (UInt32, Int, Int) = data.withUnsafeBytes { raw in
      (
        raw.load(fromByteOffset: 0, as: UInt32.self),
        Int(raw.load(fromByteOffset: 4, as: UInt32.self)),
        Int(raw.load(fromByteOffset: 8, as: UInt32.self))
      )
    }
    guard rowCount > 0, chunkSteps > 0, data.count == 12 + rowCount * 16 * 4 else {
      throw err(
        "record-v4-probe: case size mismatch — \(data.count) bytes for \(rowCount) rows (want \(12 + rowCount * 16 * 4)), chunkSteps \(chunkSteps).")
    }
    let envId = try envId(kind: kind)
    let dir = URL(fileURLWithPath: args[3], isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appendingPathComponent("actions.bin")
    FileManager.default.createFile(atPath: url.path, contents: nil)
    let h = try FileHandle(forWritingTo: url)
    defer { try? h.close() }
    // actDim is the fleet's constant 6 (Box half of D_policy = 12).
    let w = try RecordV4Writer(
      handle: h, envKind: kind, metaURL: dir.appendingPathComponent("meta.json"),
      envId: envId, seed: 0, actDim: 6)
    // The rows ARE the ring: slot 0, capacity = rowCount, so the writer's
    // `(stepStart + i) % capacity` indexes them absolutely.
    try data.withUnsafeBytes { raw in
      let rp = raw.baseAddress!.advanced(by: 12).assumingMemoryBound(to: Float.self)
      var start = 0
      while start < rowCount {
        let n = min(chunkSteps, rowCount - start)
        try w.encodeChunk(
          rp, slot: 0, capacity: rowCount, stepStart: start, stepCount: n, rewardDelta: .nan)
        start += n
      }
    }
    try w.writeMeta(final: true)
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
      || (wi == 2 && wo == 20) || (wi == 12 && wo == 21) || (wi == 15 && wo == 32)
      || (wi == 7 && wo == 23) || (wi == 13 && wo == 25) || (wi == 1 && wo == 6)
      || (wi == 9 && wo == 26) || (wi == 10 && wo == 25) || (wi == 3 && wo == 21)
    let isClassic = (wi == 3 && wo == 4) || (wi == 5 && wo == 6) || (wi == 7 && wo == 6)
      || (wi == 4 && wo == 4)
    guard n > 0, isGrid || isClassic else {
      throw err(
        "Bad case header: n=\(n) in=\(wi) out=\(wo) (expected FourRooms 8/24, DeepSea 6/21, KeyCorridor 11/28, Chain 2/20, LargeDeepSea 12/21, MultiKeyCorridor 15/32, MemoryChain 7/23, ProceduralMaze 13/25, ProceduralMazeGen 1/6, HighDimMaze 9/26, POMaze 10/25, DelayedChain 3/21, MountainCar 3/4, Acrobot 5/6, CartPole 7/6, or Pendulum 4/4).")
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
    // Recording + multiple policies is the TRUNCATION TRAP: every policy runs the
    // same lane set and writes the SAME <envId>/seed<N>/actions.bin, so each
    // policy truncates the previous one's stream and the surviving meta.json
    // describes a file that was written by a different policy. The DEFAULT
    // OAK_BENCH_POLICIES is "oak,zero,random", so this fires on the obvious
    // `OAK_RECORD_ACTIONS=1` invocation — refuse instead of silently shipping
    // mislabeled recordings.
    let recordingRequested =
      env["OAK_RECORD_ACTIONS"] == "1"
      || !(env["OAK_RECORD_SEEDS"] ?? "").trimmingCharacters(in: .whitespaces).isEmpty
    if recordingRequested, policies.count > 1 {
      throw err("Recording (OAK_RECORD_ACTIONS / OAK_RECORD_SEEDS) with \(policies.count) policies (\(policies.map(\.rawValue).joined(separator: ","))) would TRUNCATE: all policies share one <envId>/seed<N>/actions.bin, so each overwrites the last and meta.json ends up describing another policy's stream. Set OAK_BENCH_POLICIES=oak (the default is oak,zero,random).")
    }
    // RECORDING LAW (2026-08-05): recording is default-ON, but the default
    // cannot apply to a multi-policy workload (the truncation trap above) —
    // suppress with a visible line instead of throwing; an EXPLICIT request
    // still throws.
    if policies.count > 1, env["OAK_RECORD_ACTIONS"] == nil {
      recordingDefaultSuppressed = true
      print("[fleet] default recording disabled: \(policies.count) policies share lane dirs (set OAK_BENCH_POLICIES=oak to record)")
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
    // organFlops / flopsPerLaneStep (organCostVersion 1): honest per-lane-step
    // multiply+add count (fma = 2 flops) of the post-slimming megakernel,
    // binned by FC-STOMP ORGAN (not pipeline stage), computed from the real
    // config — R = rank, K = 32 options, D = 12, DL = D/2 live dims, A = 64
    // active rows, Rn nursery, traceLevels from Arena. Each organ maps to a
    // set of (stage, loop, trip-count) terms; this is the old per-stage model
    // re-binned, plus three corrections found by reading the actual loops
    // (docs/research/allocation-and-instrument-design.md §1):
    //   1. twin-dot undercount: the encode K-gather computes 11 dots/option-
    //      step (psiPrev/psiNext/hDot = 3A; rHat/lRaw/vModel/vModel2 = 4A;
    //      pdVs/pdVsA/pdR/pdL stash = 4A — OakKernel.metal:624-643); the stage
    //      model counted 10KA — the second twin vModel2 dot was uncounted.
    //      +KA lands in OptionModels.
    //   2. FeatureConstruction was absent: obs stats + trace EMAs
    //      (OakKernel.metal:877-888), core write (:920-959), gate/tile eval
    //      (:962-987), gated-bank eval 32 banks x 7 slots (:988-1010).
    //   3. ActionEmission (sample stage) was absent: Philox + Box-Muller per
    //      dim (:2536-2546), behavior-credit staging ~3/(option,live-dim) +
    //      importance ratio K (:2570-2586); Gumbel-max discrete (:2460-2515).
    // Ablation-aware: each term is counted only when the corresponding
    // function-constant-guarded loop actually compiles in (guard placement
    // identical to the old stage model). Unguarded loops (value/psi/hDot
    // dots, norms, polCoef, keyboard, utility scan, sample) run — and are
    // counted — under every ablation. Canonical case: OAK_ABLATE=planning
    // removes all of dyna (oak_dyna_backups early-returns and the dynaAcc
    // apply compiles out). K-parking (Phase 9.3) is DYNAMIC — parked options
    // skip their A-gathers at runtime; the static model counts full-width
    // trip counts regardless (parking's refund shows up in wall-clock/SPS,
    // not here).
    // CONSUME AS RELATIVE SHARES (or per-organ deltas across ablations),
    // NEVER ABSOLUTE SILICON FLOPS: static trip counts over-attribute ~3-5x
    // vs measured wall (design §1 calibration). organCostVersion gates
    // cross-version absolute comparisons (pre-organ roots are version 0).
    let fObs = 17, fD = 12  // universal I/O contract (same constants fleetGPU uses)
    let fInputW = Arena.input(fObs, fD)
    var fR = 32
    while fR < 2 * fInputW { fR *= 2 }
    let fK = 32, fA = 64, fDL = max(fD / 2, 1)
    let fTraceLevels = max(Arena.traces(fObs) - 1, 0)
    let fRn = fR * max(
      ProcessInfo.processInfo.environment["OAK_SUTTON_MULT"].flatMap(Int.init) ?? 1, 1)
    let ablated = Set(ablations.names)
    var organFma: [String: Int] = [
      "FeatureConstruction": 0, "GVFPrediction": 0, "RewardRespectingSubtasks": 0,
      "OptionPolicies": 0, "OptionModels": 0, "Planning": 0, "OptionKeyboard": 0,
      "SearchControl": 0, "GenerateAndTest": 0, "ActionEmission": 0, "misc": 0,
    ]
    // FeatureConstruction (encode): core write 32 slots + gate/tile eval 32
    // banks run under every ablation; obs stats/traces and the gated-bank
    // eval compile out under feature-construction.
    organFma["FeatureConstruction"]! += 16 + 3 * 32
    if !ablated.contains("feature-construction") {
      organFma["FeatureConstruction"]! += fObs * (2 + fTraceLevels) + 32 * 7 * 5
    }
    // GVFPrediction: value dots 2A + norms 6A (encode, unguarded) + value
    // banks 2A (learn).
    organFma["GVFPrediction"]! += 2 * fA + 6 * fA
    if !ablated.contains("gvf-prediction") { organFma["GVFPrediction"]! += 2 * fA }
    // RewardRespectingSubtasks: encode psi dots 2KA + hDot KA (unguarded) +
    // learn 2KA + headScale/psiRho K.
    organFma["RewardRespectingSubtasks"]! += 2 * fK * fA + fK * fA
    if !ablated.contains("rr-subtasks") {
      organFma["RewardRespectingSubtasks"]! += 2 * fK * fA + fK
    }
    // OptionPolicies: polCoef K*DL (plan, unguarded) + learn apply KA*DL +
    // emit option means KA*DL + consensus DL*K dir + DL*A apply (window).
    organFma["OptionPolicies"]! += fK * fDL
    if !ablated.contains("option-policies") {
      organFma["OptionPolicies"]! += 2 * fK * fA * fDL
      if !ablated.contains("generate-and-test") {
        organFma["OptionPolicies"]! += fDL * fK + fDL * fA
      }
    }
    // OptionModels (correction 1 — encode gather 4KA now includes the vModel2
    // twin dot): encode 4KA + plan-stage prev-dot stash 4KA + learn 2KA +
    // oak_plan_one scalars ~16/option (OakKernel.metal:1086-1126).
    if !ablated.contains("option-models") {
      organFma["OptionModels"]! += 4 * fK * fA + 4 * fK * fA + 2 * fK * fA + 16 * fK
    }
    // Planning (dyna): ring vX+xNorm 2A per warp thread | 4 candidates x 3
    // dots 12A per thread | copy-2 4KA | dynaAcc apply 2K per thread.
    if !ablated.contains("planning") {
      organFma["Planning"]! += 14 * fA * 32 + 4 * fK * fA + 2 * fK * 32
    }
    // OptionKeyboard: softmax refresh K + chord blend D*K + GPI scans K.
    organFma["OptionKeyboard"]! += fK + fD * fK + fK
    // SearchControl: window utility scan A*K + prioOpt K + ESS/entropy ~1/4-step.
    organFma["SearchControl"]! += fA * fK + fK + fK / 2
    // GenerateAndTest: nursery Rn/2 + eligScr R + LCB scan K.
    if !ablated.contains("generate-and-test") {
      organFma["GenerateAndTest"]! += fRn / 2 + fR + fK
    }
    // ActionEmission (correction 3, sample stage): Philox + Box-Muller ~5/dim
    // + behavior-credit staging ~3 per (option, live dim) + importance ratio K.
    organFma["ActionEmission"]! += 5 * fD + 3 * fK * fDL + fK
    // misc: serial scalar/keyboard/curation/stats loops (unattributed
    // overhead no single organ owns).
    organFma["misc"]! += 4_000
    let organCostVersion = 1
    let organFlops = organFma.mapValues { 2 * $0 }
    // Env-window flops are env-conditional per lane's env: the fp32
    // HalfCheetah physics chain ~2e4 (F4-port family estimate); the integer
    // gridworld steps are near-free (obs formulas dominate).
    let agentFlops = organFlops.values.reduce(0, +)
    let envFlops: [String: Int] = [
      "HalfCheetah-v4": 20_000, "FourRooms-cont-v0": 40, "DeepSea-cont-v0": 20,
      "KeyCorridor-cont-v0": 60, "Acrobot-cont-v0": 200, "MountainCar-cont-v0": 40,
      "CartPole-cont-v0": 60, "Pendulum-cont-v0": 50, "Chain-cont-v0": 10,
      "LargeDeepSea-cont-v0": 25, "MultiKeyCorridor-cont-v0": 60,
      "MemoryChain-cont-v0": 15,
      "ProceduralMaze-cont-v0": 50, "HighDimMaze-cont-v0": 55,
      "POMaze-cont-v0": 55, "DelayedChain-cont-v0": 8,
    ]
    let flopsPerLaneStep = agentFlops + (envFlops[envId] ?? 20_000)
    var runMeta: [String: Any] = [
      "fleetSchemaVersion": 1, "command": "benchmark", "envId": envId,
      "batch": seedValues.count, "seeds": seedValues, "steps": steps,
      "policies": policies.map(\.rawValue), "ablation": ablations.names,
      "ablationLabel": ablations.label,
      "flopsPerLaneStep": flopsPerLaneStep,
      "organFlops": organFlops,
      "organCostVersion": organCostVersion,
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
