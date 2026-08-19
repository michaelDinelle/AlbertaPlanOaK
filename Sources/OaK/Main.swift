import Accelerate
import CryptoKit
import Darwin
import Foundation
import Metal
import zlib

struct OakError: Error, CustomStringConvertible { let description: String }
func err(_ s: String) -> OakError { OakError(description: s) }
let lockName = ".oak-benchmark.lock"
typealias Raw = UnsafeMutableRawPointer

struct KernelConfig {
  var obsDim, rank, optionCount, actionParamCount, obsTraceLevels, linearAnchorCount: UInt32
  var invRank, invOptionCount, metaEta: Float
  var agentCount, paramStride, stateStride, uintStride: UInt32
  var nurseryCount: UInt32
  var sampleSigma: Float
  var sampleInv2S2: Float
  var actorAlphaDenom: Float
}
struct KernelScalars {
  var stepCount: UInt32 = 0, stateSlot: UInt32 = 0
  var averageReward: Float = 0
}

struct FleetEnvCfg {
  var laneCount, episodeLen, recordCapacity: UInt32
  var invObsFloor: Float
}
struct LaneSnapshot {
  var scalars: KernelScalars
  var paramAbsMax: Float
  var paramNonFinite: UInt32
  var rewardSum, rho: Float
  var deathStep, deathDetector: UInt32
}
let fleetLaneStride = 240

enum OakConst: Int {
  case ablateFeatureConstruction = 0
  case ablateGvfPrediction = 1
  case ablateRrSubtasks = 2
  case ablateOptionPolicies = 3
  case ablateOptionModels = 4
  case ablateOptionKeyboard = 6
  case ablateKeyboardToPolicy = 8
  case ablateSearchControl = 9
  case ablateSearchToGenerateTest = 10
  case ablateGenerateAndTest = 11
  case megaG = 12
  case specObsDim = 13
  case specRank = 14
  case specOptionCount = 15
  case specActionParams = 16
  case specTraceLevels = 17
  case specAnchorCount = 18
  case specParamStride = 19
  case specStateStride = 20
  case specUintStride = 21
  case specInvRank = 22
  case specInvOptionCount = 23
  case specMetaEta = 24
  case megaStageMask = 25
  case ablateJointGeometry = 27
  case specNursery = 28
  case specEnvKind = 29
  case probeFixedIdx = 30
  case probeFtz = 31
}

enum OakBuf: Int {
  case params = 0
  case state = 1
  case uints = 2
  case scalars = 3
  case config = 4
  case lanes = 6
  case seeds = 7
  case envCfg = 8
  case aux0 = 9
  case aux1 = 10
  case aux2 = 11
  case diagAccum = 12
  case diagOut = 13
}

enum Diag {
  static let rank = 480
  static let k = 64
  static let gram = k * k
  static let ppoParams = 2768
  static let accumGram = 0
  static let accumSumAbsHiddenActor = accumGram + gram
  static let accumSumAbsHiddenCritic = accumSumAbsHiddenActor + k
  static let accumFloats = accumSumAbsHiddenCritic + k
  static let oakFeatureAge = 0
  static let oakFeatureUtility = oakFeatureAge + rank
  static let oakValueAlpha = oakFeatureUtility + rank
  static let oakOptionValueAlpha = oakValueAlpha + rank
  static let oakModelAlpha = oakOptionValueAlpha + rank
  static let oakFeatureBirths = oakModelAlpha + rank
  static let oakWeightAbsMean = oakFeatureBirths + 1
  static let oakWeightAbsMeanValue = oakWeightAbsMean + 1
  static let oakGram = oakWeightAbsMeanValue + 1
  static let oakFloats = oakGram + gram
  static let ppoSumAbsHiddenActor = 0
  static let ppoSumAbsHiddenCritic = ppoSumAbsHiddenActor + k
  static let ppoWeightAbsMean = ppoSumAbsHiddenCritic + k
  static let ppoAdamT = ppoWeightAbsMean + 1
  static let ppoAdamV = ppoAdamT + 1
  static let ppoGram = ppoAdamV + ppoParams
  static let ppoFloats = ppoGram + gram
  static let outFloats = ppoFloats
}

struct AblationSpec {
  static let ordered = [
    "feature-construction", "gvf-prediction", "rr-subtasks", "option-policies",
    "option-models", "option-keyboard",
    "keyboard-to-policy", "search-control", "search-to-generate-test", "generate-and-test",
    "joint-geometry",
  ]
  static let functionConstantIndex = [
    "feature-construction": OakConst.ablateFeatureConstruction.rawValue,
    "gvf-prediction": OakConst.ablateGvfPrediction.rawValue,
    "rr-subtasks": OakConst.ablateRrSubtasks.rawValue,
    "option-policies": OakConst.ablateOptionPolicies.rawValue,
    "option-models": OakConst.ablateOptionModels.rawValue,
    "option-keyboard": OakConst.ablateOptionKeyboard.rawValue,
    "keyboard-to-policy": OakConst.ablateKeyboardToPolicy.rawValue,
    "search-control": OakConst.ablateSearchControl.rawValue,
    "search-to-generate-test": OakConst.ablateSearchToGenerateTest.rawValue,
    "generate-and-test": OakConst.ablateGenerateAndTest.rawValue,
    "joint-geometry": OakConst.ablateJointGeometry.rawValue,
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
    for name in Self.ordered {
      var enabled = names.contains(name)
      constants.setConstantValue(
        &enabled, type: .bool, index: Self.functionConstantIndex[name]!)
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
  let fm = FileManager.default
  let sourcesDir = root("Sources")
  let sources = try fm.subpathsOfDirectory(atPath: sourcesDir.path)
    .filter { $0.hasSuffix(".metal") }
    .map { sourcesDir.appendingPathComponent($0) }
  guard !sources.isEmpty else { throw err("No .metal sources under \(sourcesDir.path).") }
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

struct ShapeProbe {
  var rank, optionCount, nursery, traceLevels, anchorCount: UInt32
  var paramStride, stateStride, uintStride: UInt32
  var pPsi, pOptPolicy, stUtilOpt: UInt32
  var stAlpha, stBeta, stAlphaPsi, stBetaPsi, stAlphaModel, stBetaModel: UInt32
  var stAlphaAct, stBetaAct, uOptFeature, uParentA, uParentB, uKind, uNurParentA: UInt32
  var invRank, invOptionCount, metaEta, sampleSigma, sampleInv2S2, actorAlphaDenom: Float
  var stKeep0, stKeep0n, stKeep1, stKeep1n, uKeep0, uKeep0n: UInt32
}
struct RewardSlices {
  let state: [(Int, Int)]
  let uints: [(Int, Int)]
  init(_ sp: ShapeProbe) {
    state = [(Int(sp.stKeep0), Int(sp.stKeep0n)), (Int(sp.stKeep1), Int(sp.stKeep1n))]
    uints = [(Int(sp.uKeep0), Int(sp.uKeep0n))]
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
struct EnvSpec {
  static let poseLen = 9
  static let stride = 7 + poseLen
  static let slotNames = [
    "none", "x", "y", "gx", "gy", "kx", "ky", "carry", "gates", "tokens", "cue", "heading",
    "effort",
  ]
  let nameHash: UInt32
  let naturalO: Int
  let actionKind: Int
  let actionCount: Int
  let gridW: Int
  let gridH: Int
  let controlHz: Int
  let slots: [Int]
  static func slotName(_ code: Int) -> String {
    if code >= 16 { return "q\(code - 16)" }
    guard code >= 0, code < slotNames.count else { return "none" }
    return slotNames[code]
  }
}

final class Engine {
  struct Kind {
    let kind: UInt32
    let naturalO, d, count, firstLane: Int
    let off: ShapeProbe
    let paramTotal, stateTotal, uintTotal: Int
    let paramBase, stateBase, uintBase: Int
    let configB: MTLBuffer
    var cfg: KernelConfig {
      configB.contents().bindMemory(to: KernelConfig.self, capacity: 1).pointee
    }
  }
  let batch: Int
  let kinds: [Kind]
  let laneKindIdx: [Int]
  let envTable: [EnvSpec]
  let q: MTLCommandQueue, params, state: Shared<Float>,
    uints: Shared<UInt32>, res: [MTLBuffer]
  let scalarsB: MTLBuffer
  let scalars: UnsafeMutablePointer<KernelScalars>
  func kind(of lane: Int) -> Kind { kinds[laneKindIdx[lane]] }
  func paramPtr(_ lane: Int) -> UnsafeMutablePointer<Float> {
    let kd = kind(of: lane)
    return params.p + kd.paramBase + (lane - kd.firstLane) * kd.paramTotal
  }
  func statePtr(_ lane: Int) -> UnsafeMutablePointer<Float> {
    let kd = kind(of: lane)
    return state.p + kd.stateBase + (lane - kd.firstLane) * kd.stateTotal
  }
  func uintPtr(_ lane: Int) -> UnsafeMutablePointer<UInt32> {
    let kd = kind(of: lane)
    return uints.p + kd.uintBase + (lane - kd.firstLane) * kd.uintTotal
  }
  static func runShapeProbe(_ dev: MTLDevice, _ lib: MTLLibrary, obsDim: Int, dPolicy: Int)
    throws -> ShapeProbe
  {
    precondition(
      MemoryLayout<ShapeProbe>.stride == 144, "ShapeProbe layout drifted from OakShapeProbe")
    let constants = MTLFunctionConstantValues()
    var o32 = UInt32(obsDim)
    constants.setConstantValue(&o32, type: .uint, index: OakConst.specObsDim.rawValue)
    var d32 = UInt32(dPolicy)
    constants.setConstantValue(&d32, type: .uint, index: OakConst.specActionParams.rawValue)
    let fn = try lib.makeFunction(name: "oak_shape_probe", constantValues: constants)
    let pipe = try dev.makeComputePipelineState(function: fn)
    guard let outB = dev.makeBuffer(length: 144, options: [.storageModeShared]),
      let dummy = dev.makeBuffer(length: 16, options: [.storageModeShared]),
      let pq = dev.makeCommandQueue(), let cb = pq.makeCommandBuffer(),
      let enc = cb.makeComputeCommandEncoder()
    else { throw err("Shape probe dispatch failed.") }
    enc.setComputePipelineState(pipe)
    enc.setBuffer(outB, offset: 0, index: OakBuf.params.rawValue)
    for j in OakBuf.state.rawValue...OakBuf.config.rawValue {
      enc.setBuffer(dummy, offset: 0, index: j)
    }
    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return outB.contents().bindMemory(to: ShapeProbe.self, capacity: 1).pointee
  }
  static func runEnvTableProbe(_ dev: MTLDevice, _ lib: MTLLibrary) throws -> [EnvSpec] {
    let fn = lib.makeFunction(name: "oak_env_table_probe")!
    let pipe = try dev.makeComputePipelineState(function: fn)
    let words = 1 + Main.envIdsByKind.count * EnvSpec.stride
    guard let outB = dev.makeBuffer(length: words * 4, options: [.storageModeShared]),
      let pq = dev.makeCommandQueue(), let cb = pq.makeCommandBuffer(),
      let enc = cb.makeComputeCommandEncoder()
    else { throw err("Env table probe dispatch failed.") }
    enc.setComputePipelineState(pipe)
    enc.setBuffer(outB, offset: 0, index: OakBuf.params.rawValue)
    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let tbl = outB.contents().bindMemory(to: UInt32.self, capacity: words)
    guard tbl[0] == UInt32(Main.envIdsByKind.count) else {
      throw err("Env table count drift: kernel \(tbl[0]) vs host \(Main.envIdsByKind.count).")
    }
    return (0..<Main.envIdsByKind.count).map { i in
      let row = 1 + i * EnvSpec.stride
      return EnvSpec(
        nameHash: tbl[row], naturalO: Int(tbl[row + 1]), actionKind: Int(tbl[row + 2]),
        actionCount: Int(tbl[row + 3]), gridW: Int(tbl[row + 4]), gridH: Int(tbl[row + 5]),
        controlHz: Int(tbl[row + 6]),
        slots: (0..<EnvSpec.poseLen).map { Int(tbl[row + 7 + $0]) })
    }
  }
  init(laneEnvs: [String], batch b: Int, dev: MTLDevice) throws {
    precondition(b >= 1 && laneEnvs.count == b, "Engine batch/laneEnvs mismatch")
    batch = b
    let lib = try dev.makeLibrary(URL: try requireFreshMetallib())
    let table = try Engine.runEnvTableProbe(dev, lib)
    var runs: [(kind: UInt32, firstLane: Int, count: Int)] = []
    var seen = Set<UInt32>()
    for (i, env) in laneEnvs.enumerated() {
      let kd = try Main.envKind(env)
      if let last = runs.last, last.kind == kd {
        runs[runs.count - 1].count += 1
      } else {
        guard !seen.contains(kd) else {
          throw err("Schedule must keep each env contiguous; \(env) repeats.")
        }
        seen.insert(kd)
        runs.append((kd, i, 1))
      }
    }
    var built: [Kind] = []
    var probeCache: [UInt32: ShapeProbe] = [:]
    var pBase = 0, sBase = 0, uBase = 0
    var laneIdx = [Int](repeating: 0, count: b)
    let fleetD = 2 * table.map(\.actionCount).max()!
    for r in runs {
      let spec = table[Int(r.kind)]
      let naturalO = spec.naturalO
      let dk = fleetD
      let sp = try probeCache[r.kind] ?? Engine.runShapeProbe(
        dev, lib, obsDim: naturalO, dPolicy: dk)
      probeCache[r.kind] = sp
      let anchors = Int(sp.anchorCount)
      precondition(Int(sp.optionCount) == anchors, "K equals core")
      precondition(Int(sp.rank) == anchors + anchors * 14, "gate-bank identity broken")
      precondition(anchors >= naturalO + dk + 2, "core slots cannot hold obs+action+reward+bias")
      precondition(Int(sp.stateStride) % 4 == 0, "stateStride must stay 16B-aligned")
      let cfgB = dev.makeBuffer(
        length: MemoryLayout<KernelConfig>.stride, options: [.storageModeShared])!
      cfgB.contents().bindMemory(to: KernelConfig.self, capacity: 1).initialize(
        to: KernelConfig(
          obsDim: UInt32(naturalO), rank: sp.rank, optionCount: sp.optionCount,
          actionParamCount: UInt32(dk), obsTraceLevels: sp.traceLevels,
          linearAnchorCount: sp.anchorCount, invRank: sp.invRank,
          invOptionCount: sp.invOptionCount, metaEta: sp.metaEta,
          agentCount: UInt32(b), paramStride: sp.paramStride,
          stateStride: sp.stateStride, uintStride: sp.uintStride,
          nurseryCount: sp.nursery,
          sampleSigma: sp.sampleSigma,
          sampleInv2S2: sp.sampleInv2S2,
          actorAlphaDenom: sp.actorAlphaDenom))
      let kd = Kind(
        kind: r.kind, naturalO: naturalO, d: dk, count: r.count, firstLane: r.firstLane,
        off: sp, paramTotal: Int(sp.paramStride), stateTotal: Int(sp.stateStride),
        uintTotal: Int(sp.uintStride), paramBase: pBase, stateBase: sBase, uintBase: uBase,
        configB: cfgB)
      built.append(kd)
      for l in r.firstLane..<(r.firstLane + r.count) { laneIdx[l] = built.count - 1 }
      pBase += r.count * kd.paramTotal
      sBase += r.count * kd.stateTotal
      uBase += r.count * kd.uintTotal
    }
    kinds = built
    laneKindIdx = laneIdx
    envTable = table
    guard let cq = dev.makeCommandQueue() else {
      throw err("Metal command queue unavailable.")
    }
    q = cq
    params = Shared(dev, pBase, 0)
    state = Shared(dev, sBase, 0)
    uints = Shared(dev, uBase, 0)
    scalarsB = dev.makeBuffer(
      length: batch * MemoryLayout<KernelScalars>.stride, options: [.storageModeShared])!
    res = [params.b, state.b, uints.b, scalarsB, kinds[0].configB]
    scalars = scalarsB.contents().bindMemory(to: KernelScalars.self, capacity: batch)
    for kd in kinds {
      let sp = kd.off
      let rank = Int(sp.rank)
      let k = Int(sp.optionCount)
      let rn = Int(sp.nursery)
      let anchors = Int(sp.anchorCount)
      let psi = Int(sp.pPsi)
      let inv = 1 / sqrt(Float(rank))
      let alpha0 = 1 / Float(2 * Int(sp.anchorCount))
      let traceChannels = max(Int(sp.traceLevels) * kd.naturalO, 1)
      let basis = 1 / Float(max(kd.d, 1))
      for local in 0..<kd.count {
        let lane = kd.firstLane + local
        scalars[lane] = KernelScalars()
        let lp = params.p + kd.paramBase + local * kd.paramTotal
        let ls = state.p + kd.stateBase + local * kd.stateTotal
        let lu = uints.p + kd.uintBase + local * kd.uintTotal
        for n in 0..<rn {
          lu[Int(sp.uNurParentA) + n] = UInt32(n % ((n % 4 == 3) ? traceChannels : anchors))
          lu[Int(sp.uNurParentA) + rn + n] = UInt32((n * 7 + 3) % anchors)
          lu[Int(sp.uNurParentA) + 2 * rn + n] = UInt32(n % 4)
        }
        for i in 0..<rank {
          if i >= anchors {
            let idx = i - anchors
            if idx < traceChannels {
              lu[Int(sp.uParentA) + i] = UInt32(idx)
              lu[Int(sp.uParentB) + i] = 0
              lu[Int(sp.uKind) + i] = 3
            } else {
              lu[Int(sp.uParentA) + i] = UInt32(idx % anchors)
              lu[Int(sp.uParentB) + i] = UInt32((idx * 7 + 3) % anchors)
              lu[Int(sp.uKind) + i] = UInt32(idx % 3)
            }
          } else {
            lu[Int(sp.uParentA) + i] = 0
            lu[Int(sp.uParentB) + i] = 0
            lu[Int(sp.uKind) + i] = 0
          }
          [sp.stAlpha, sp.stAlphaPsi, sp.stAlphaModel].forEach { ls[Int($0) + i] = alpha0 }
          [sp.stBeta, sp.stBetaPsi, sp.stBetaModel].forEach { ls[Int($0) + i] = log(alpha0) }
        }
        for kk in 0..<k {
          lu[Int(sp.uOptFeature) + kk] = UInt32(kk)
          lp[psi + kk * k + kk] = inv
          ls[Int(sp.stUtilOpt) + kk] = Float(rank)
          ls[Int(sp.stAlphaAct) + kk] = 1
          ls[Int(sp.stBetaAct) + kk] = 0
          let h = UInt32(truncatingIfNeeded: lane &* 2_654_435_761 &+ kk &* 40_503 &+ 977)
          let slot = Int(h % UInt32(max(kd.d / 2, 1)))
          let sign: Float = ((h >> 16) & 1) == 0 ? 1 : -1
          lp[Int(sp.pOptPolicy) + kk * (kd.d / 2) + slot] = sign * basis
        }
      }
    }
  }
  static func applyShapeConstants(_ constants: MTLFunctionConstantValues, _ cfg: KernelConfig) {
    let us: [(OakConst, UInt32)] = [
      (.specObsDim, cfg.obsDim), (.specRank, cfg.rank), (.specOptionCount, cfg.optionCount),
      (.specActionParams, cfg.actionParamCount), (.specTraceLevels, cfg.obsTraceLevels),
      (.specAnchorCount, cfg.linearAnchorCount), (.specParamStride, cfg.paramStride),
      (.specStateStride, cfg.stateStride), (.specUintStride, cfg.uintStride),
      (.specNursery, cfg.nurseryCount),
    ]
    for (fc, val) in us {
      var v = val
      constants.setConstantValue(&v, type: .uint, index: fc.rawValue)
    }
    let fs: [(OakConst, Float)] = [
      (.specInvRank, cfg.invRank), (.specInvOptionCount, cfg.invOptionCount),
      (.specMetaEta, cfg.metaEta),
    ]
    for (fc, val) in fs {
      var v = val
      constants.setConstantValue(&v, type: .float, index: fc.rawValue)
    }
  }
  func stateHashes(_ lane: Int, laneBytes: UnsafeRawPointer) -> [String: String] {
    func hex(_ base: UnsafeRawPointer, _ bytes: Int) -> String {
      sha256Hex(UnsafeRawBufferPointer(start: base, count: bytes))
    }
    func hexSlices(_ base: UnsafeRawPointer, _ elem: Int, _ ranges: [(Int, Int)]) -> String {
      var h = SHA256()
      for (off, n) in ranges {
        h.update(
          bufferPointer: UnsafeRawBufferPointer(start: base + off * elem, count: n * elem))
      }
      return h.finalize().map { String(format: "%02x", $0) }.joined()
    }
    let kd = kind(of: lane)
    let slices = RewardSlices(kd.off)
    return [
      "params": hex(paramPtr(lane), kd.paramTotal * MemoryLayout<Float>.stride),
      "state": hex(statePtr(lane), kd.stateTotal * MemoryLayout<Float>.stride),
      "uints": hex(uintPtr(lane), kd.uintTotal * MemoryLayout<UInt32>.stride),
      "scalars": hex(scalars + lane, MemoryLayout<KernelScalars>.stride),
      "stateReward": hexSlices(statePtr(lane), MemoryLayout<Float>.stride, slices.state),
      "uintsReward": hexSlices(uintPtr(lane), MemoryLayout<UInt32>.stride, slices.uints),
      "lane": hex(laneBytes, fleetLaneStride),
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


struct RecordField {
  let idx: Int
  let scale: Double
  let wrap: Bool
  let anchor: Bool
  init(_ idx: Int, _ scale: Double = 1, _ wrap: Bool = false, anchor: Bool = false) {
    self.idx = idx
    self.scale = scale
    self.wrap = wrap
    self.anchor = anchor
  }
}

struct RecordCodec {
  let category: String
  let fields: [RecordField]
  static func grid(_ idxs: [Int]) -> RecordCodec {
    RecordCodec(category: "grid", fields: idxs.map { RecordField($0) })
  }
  static let byKind: [RecordCodec] = {
    var cheetah = [RecordField(0, 0.01, false, anchor: true), RecordField(1, 1e-4)]
    for i in 2...8 { cheetah.append(RecordField(i, 1e-4, true)) }
    return [
      RecordCodec(category: "cheetah", fields: cheetah),
      grid([0, 1, 2, 3]),
      grid([0, 1, 2, 3, 4]),
      grid([0, 1, 2, 3, 4, 5, 6, 7]),
      RecordCodec(category: "classic", fields: [
        RecordField(0, 1e-4, true), RecordField(1, 1e-4, true),
        RecordField(2, 1e-3), RecordField(3, 1e-3), RecordField(4),
      ]),
      RecordCodec(category: "classic", fields: [
        RecordField(0, 1e-4), RecordField(1, 1e-5), RecordField(3),
      ]),
      RecordCodec(category: "classic", fields: [
        RecordField(0, 1e-4), RecordField(1, 1e-3),
        RecordField(2, 1e-4, true), RecordField(3, 1e-3), RecordField(4),
      ]),
      RecordCodec(category: "classic", fields: [
        RecordField(0, 1e-4, true), RecordField(1, 1e-3), RecordField(2, 1e-3),
      ]),
      grid([0]),
      grid([0, 1, 2, 3, 4]),
      grid([0, 1, 2, 3, 4, 5, 6]),
      grid([0, 2, 3]),
      grid([0, 1, 2, 3]),
      grid([0, 1, 2, 3, 4, 5]),
      grid([0, 1, 2, 3]),
      grid([0, 2]),
    ]
  }()
  static func table(_ envKind: UInt32) throws -> RecordCodec {
    guard Int(envKind) < byKind.count else {
      throw err("No record codec for envKind \(envKind).")
    }
    return byKind[Int(envKind)]
  }
}

let recordFloats = 1 + EnvSpec.poseLen
let quantizeLimit = 4.6e18

func putVarint(_ out: inout [UInt8], _ value: Int64) {
  var z = UInt64(bitPattern: (value << 1) ^ (value >> 63))
  while z >= 0x80 {
    out.append(UInt8(z & 0x7F) | 0x80)
    z >>= 7
  }
  out.append(UInt8(z))
}

final class RecordWriter: @unchecked Sendable {
  static let fileMagic: UInt32 = 0x524B_414F
  static let chunkMagic: UInt32 = 0x524B_4843

  let handle: FileHandle
  let codec: RecordCodec
  private var buf: [UInt8] = []
  private var fileOffset = 0
  private(set) var closed = false
  private var deflateVerified = false

  init(handle: FileHandle, envKind: UInt32) throws {
    self.handle = handle
    self.codec = try RecordCodec.table(envKind)
    var head = Data(capacity: 8)
    for v: UInt32 in [Self.fileMagic, envKind] {
      var x = v.littleEndian
      withUnsafeBytes(of: &x) { head.append(contentsOf: $0) }
    }
    try handle.write(contentsOf: head)
    fileOffset = head.count
  }

  func encodeChunk(
    _ rp: UnsafePointer<Float>, slot: Int, capacity: Int, laneCount: Int,
    stepStart: Int, stepCount: Int
  ) throws {
    guard !closed, stepCount > 0 else { return }
    let fields = codec.fields
    let rowStride = recordFloats * laneCount
    func cell(_ i: Int, _ field: Int) -> Double {
      Double(rp[((stepStart + i) % capacity) * rowStride + field * laneCount + slot])
    }
    func value(_ i: Int, _ f: RecordField) -> Double? {
      var w = cell(i, 1 + f.idx)
      guard w.isFinite else { return nil }
      if f.wrap { w -= 2 * Double.pi * ((w + Double.pi) / (2 * Double.pi)).rounded(.down) }
      return w
    }
    func quantized(_ i: Int, _ f: RecordField) -> Int64? {
      guard let w = value(i, f) else { return nil }
      let q = (w / f.scale).rounded()
      guard q.magnitude < quantizeLimit else { return nil }
      return Int64(q)
    }

    var live = stepCount
    scan: for i in 0..<stepCount {
      guard cell(i, 0).isFinite else { live = i; closed = true; break scan }
      for f in fields where quantized(i, f) == nil {
        live = i
        closed = true
        break scan
      }
    }
    guard live > 0 else { return }

    buf.removeAll(keepingCapacity: true)
    buf.append(contentsOf: repeatElement(0, count: (live + 7) / 8))
    for i in 0..<live where cell(i, 0) != 0 { buf[i >> 3] |= UInt8(1 << (i & 7)) }

    var predictorBits: UInt32 = 0
    var column = 0
    for f in fields {
      if f.anchor {
        var recon = value(0, f)!
        withUnsafeBytes(of: recon.bitPattern.littleEndian) { buf.append(contentsOf: $0) }
        for i in 0..<live {
          let q = ((value(i, f)! - recon) / f.scale).rounded()
          recon += q * f.scale
          putVarint(&buf, Int64(q))
        }
        column += 1
        continue
      }
      var q = [Int64](repeating: 0, count: live)
      for i in 0..<live { q[i] = quantized(i, f)! }
      var first: [UInt8] = []
      var second: [UInt8] = []
      var previous: Int64 = 0
      var previousDelta: Int64 = 0
      for i in 0..<live {
        let delta = i == 0 ? q[i] : q[i] &- previous
        putVarint(&first, delta)
        putVarint(&second, i < 2 ? delta : delta &- previousDelta)
        previous = q[i]
        previousDelta = delta
      }
      if second.count < first.count {
        predictorBits |= UInt32(1) << UInt32(column)
        buf.append(contentsOf: second)
      } else {
        buf.append(contentsOf: first)
      }
      column += 1
    }

    let payload = Data(buf)
    let comp = try (payload as NSData).compressed(using: .zlib) as Data
    if !deflateVerified {
      deflateVerified = true
      let wrapped = comp.count >= 2
        && comp.withUnsafeBytes { p -> Bool in
          let b0 = Int(p[0]), b1 = Int(p[1])
          return b0 == 0x78 && ((b0 << 8) | b1) % 31 == 0
        }
      if wrapped {
        throw err("record: Compression .zlib emitted a ZLIB-WRAPPED stream; the spec requires raw DEFLATE.")
      }
      let back = try (comp as NSData).decompressed(using: .zlib) as Data
      guard back == payload else {
        throw err("record: deflate round-trip mismatch (\(back.count) vs \(payload.count) bytes).")
      }
    }
    var out = Data(capacity: 20 + comp.count)
    for v: UInt32 in [
      Self.chunkMagic, UInt32(live), UInt32(payload.count), UInt32(comp.count), predictorBits,
    ] {
      var x = v.littleEndian
      withUnsafeBytes(of: &x) { out.append(contentsOf: $0) }
    }
    out.append(comp)
    try handle.write(contentsOf: out)
    fileOffset += 20 + comp.count
  }

  var bytesWritten: Int { fileOffset }
}

enum DiagMath {
  static func quantile(_ sorted: [Double], _ p: Double) -> Double {
    let n = sorted.count
    guard n > 0 else { return .nan }
    let pos = p * Double(n - 1)
    let lo = Int(pos.rounded(.down))
    let hi = min(lo + 1, n - 1)
    let frac = pos - Double(lo)
    return sorted[lo] * (1 - frac) + sorted[hi] * frac
  }
  static func logQuantiles(_ values: [Double], _ ps: [Double]) -> [Float] {
    guard !values.isEmpty, values.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      return ps.map { _ in Float.nan }
    }
    let sorted = values.map { log10($0) }.sorted()
    return ps.map { Float(quantile(sorted, $0)) }
  }
  static let dormantTau = 0.025
  static func dormantFraction(_ activity: [Double]) -> Float {
    let finite = activity.filter(\.isFinite)
    let mean = finite.isEmpty ? 0 : finite.reduce(0, +) / Double(finite.count)
    guard mean > 0, mean.isFinite else { return .nan }
    var dormant = 0
    for a in activity where a.isFinite && a / mean < dormantTau { dormant += 1 }
    return Float(dormant) / Float(activity.count)
  }
  static func erank(_ gram: UnsafePointer<Float>) -> Float {
    var a = [Float](repeating: 0, count: Diag.gram)
    for i in 0..<Diag.gram {
      let v = gram[i]
      guard v.isFinite else { return .nan }
      a[i] = v
    }
    var jobz = CChar(UInt8(ascii: "N"))
    var uplo = CChar(UInt8(ascii: "U"))
    var n = __LAPACK_int(Diag.k)
    var lda = __LAPACK_int(Diag.k)
    var eig = [Float](repeating: 0, count: Diag.k)
    var lwork = __LAPACK_int(64 * Diag.k)
    var work = [Float](repeating: 0, count: Int(lwork))
    var info: __LAPACK_int = 0
    a.withUnsafeMutableBufferPointer { ap in
      eig.withUnsafeMutableBufferPointer { wp in
        work.withUnsafeMutableBufferPointer { kp in
          ssyev_(&jobz, &uplo, &n, ap.baseAddress!, &lda, wp.baseAddress!, kp.baseAddress!, &lwork, &info)
        }
      }
    }
    guard info == 0 else { return .nan }
    let sigma = eig.map { Double(max($0, 0)).squareRoot() }
    let total = sigma.reduce(0, +)
    guard total > 0, total.isFinite else { return .nan }
    var entropy = 0.0
    for s in sigma where s > 0 {
      let p = s / total
      entropy -= p * log(p)
    }
    return Float(exp(entropy))
  }
}

final class OakDiagHost: @unchecked Sendable {
  let prevAge: UnsafeMutablePointer<UInt32>
  let prevBirths: UnsafeMutablePointer<Float>
  let hasPrev: UnsafeMutablePointer<Bool>
  init(_ laneCount: Int) {
    prevAge = .allocate(capacity: laneCount * Diag.rank)
    prevAge.initialize(repeating: 0, count: laneCount * Diag.rank)
    prevBirths = .allocate(capacity: laneCount)
    prevBirths.initialize(repeating: 0, count: laneCount)
    hasPrev = .allocate(capacity: laneCount)
    hasPrev.initialize(repeating: false, count: laneCount)
  }
}

final class TapeWriter {
  static let fileMagic: UInt32 = 0x544B_414F
  static let baseFields = ["rewardSum", "rho", "deathStep", "deathDetector"]
  let fields: [String]
  private let handle: FileHandle
  private let laneCount: Int
  private let diagCount: Int
  private var col: [Float]

  init(url: URL, laneCount: Int, diagFields: [String]) throws {
    FileManager.default.createFile(atPath: url.path, contents: nil)
    self.handle = try FileHandle(forWritingTo: url)
    self.laneCount = laneCount
    self.diagCount = diagFields.count
    self.fields = Self.baseFields + diagFields
    self.col = [Float](repeating: 0, count: laneCount)
    let json = try JSONSerialization.data(withJSONObject: ["fields": fields])
    var head = Data()
    for v: UInt32 in [Self.fileMagic, UInt32(laneCount), UInt32(json.count)] {
      var x = v.littleEndian
      withUnsafeBytes(of: &x) { head.append(contentsOf: $0) }
    }
    head.append(json)
    try handle.write(contentsOf: head)
  }

  func append(snap: UnsafeMutablePointer<LaneSnapshot>, half: Int,
              deadStep: [Int?], deadDetector: [UInt32?], diag: [Float],
              stepStart: Int, stepCount: Int, wallSeconds: Double) throws {
    precondition(diag.count == diagCount * laneCount, "diag matrix shape drifted from the field list")
    var out = Data(capacity: 12 + fields.count * laneCount * 4)
    for v: UInt32 in [UInt32(stepStart), UInt32(stepCount)] {
      var x = v.littleEndian
      withUnsafeBytes(of: &x) { out.append(contentsOf: $0) }
    }
    var w = Float(wallSeconds).bitPattern.littleEndian
    withUnsafeBytes(of: &w) { out.append(contentsOf: $0) }
    func push(_ get: (Int, LaneSnapshot) -> Float) {
      for l in 0..<laneCount { col[l] = get(l, snap[half * laneCount + l]) }
      col.withUnsafeBufferPointer {
        out.append(UnsafeRawPointer($0.baseAddress!).assumingMemoryBound(to: UInt8.self),
                   count: laneCount * 4)
      }
    }
    push { _, s in s.rewardSum }
    push { _, s in s.rho }
    push { l, _ in deadStep[l].map { Float(bitPattern: UInt32($0)) } ?? Float.nan }
    push { l, _ in deadDetector[l].map { Float($0) } ?? Float.nan }
    for f in 0..<diagCount {
      push { l, _ in diag[l * diagCount + f] }
    }
    try handle.write(contentsOf: out)
  }
}

@main struct Main {
  static func main() {
    _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    do {
      switch CommandLine.arguments.dropFirst().first {
      case "benchmark": try benchmark()
      case "compare-hashes": try compareHashes()
      default: throw err("Use `OaK benchmark`.")
      }
    } catch {
      fputs("\(error)\n", stderr)
      Foundation.exit(1)
    }
  }
  nonisolated(unsafe) static var pendingMeta: [String: Any] = [:]
  static func writeRunMeta(_ dir: URL, _ table: [EnvSpec]) throws {
    var meta = pendingMeta
    meta["envCatalog"] = envIdsByKind.enumerated().map { kind, envId -> [String: Any] in
      let spec = table[kind]
      let codec = RecordCodec.byKind[kind]
      return [
        "envId": envId, "kind": kind,
        "actionKind": spec.actionKind == 0 ? "box" : "discrete",
        "actionCount": spec.actionCount,
        "controlHz": spec.controlHz,
        "gridW": spec.gridW, "gridH": spec.gridH,
        "slots": spec.slots.map(EnvSpec.slotName),
        "category": codec.category,
        "fields": codec.fields.map { f in
          ["slot": f.idx, "scale": f.scale, "wrap": f.wrap, "anchor": f.anchor] as [String: Any]
        },
      ]
    }
    try JSONSerialization.data(
      withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]
    ).write(to: dir.appendingPathComponent("meta.json"), options: .atomic)
  }
  static let envIdsByKind: [String] = [
    "HalfCheetah-v4", "FourRooms-cont-v0", "DeepSea-cont-v0", "KeyCorridor-cont-v0",
    "Acrobot-cont-v0", "MountainCar-cont-v0", "CartPole-cont-v0", "Pendulum-cont-v0",
    "Chain-cont-v0", "LargeDeepSea-cont-v0", "MultiKeyCorridor-cont-v0", "MemoryChain-cont-v0",
    "ProceduralMaze-cont-v0", "HighDimMaze-cont-v0", "POMaze-cont-v0", "DelayedChain-cont-v0",
  ]
  static let envStepFlops: [Double] = [
    150_000, 60, 30, 70, 400, 40, 60, 60,
    25, 30, 80, 40, 90, 70, 70, 30,
  ]
  static func policyFlopsPerStep(
    _ policy: String, kind: Int, spec: EnvSpec, off: ShapeProbe?, d: Int
  ) -> Double {
    let env = envStepFlops[kind]
    let obs = Double(spec.naturalO)
    let actions = Double(spec.actionCount)
    switch policy {
    case "oak":
      let sp = off!
      let rank = Double(sp.rank)
      let options = Double(sp.optionCount)
      let anchors = Double(sp.anchorCount)
      let levels = Double(sp.traceLevels)
      let gateGroups = (rank / anchors - 1) / levels
      let active = anchors * (1 + gateGroups)
      let nursery = Double(sp.nursery)
      let policyN = Double(max(d / 2, 1))
      return options * active * (56 + 2 * policyN) + active * 256 + 16 * (rank - options)
        + 13 * nursery + 4 * policyN * active + obs * (3 * levels + 10) + env
    case "ppo":
      let hidden = 64.0
      let rolloutSteps = 128.0
      let epochs = 4.0
      let minibatch = 32.0
      let adamFlopsPerParam = 11.0
      let params = 2 * obs * hidden + 3 * hidden + hidden * actions + actions + 1
        + (spec.actionKind == 0 ? actions : 0)
      let forward = 2 * params
      let backward = 2 * forward
      let updatesPerStep = epochs / minibatch
      return forward + epochs * (forward + backward) + forward / rolloutSteps
        + adamFlopsPerParam * params * updatesPerStep + env
    default:
      return 120 + 2 * actions + env
    }
  }
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
  struct FleetPipes {
    let label: String
    let encodeInit: (MTLComputeCommandEncoder) -> Void
    let encodeMegastep: (MTLComputeCommandEncoder, Int, Int) -> Void
    let encodeSnapshot: (MTLComputeCommandEncoder, Int) -> Void
    let writeDebugRaw: (String, Int) throws -> Void
    let laneHashes: (Int, UnsafeRawPointer) -> [String: String]
    let diagFields: [String]
    let diagValues: (Int, Int, Int) -> [Float]
  }
  struct FleetBuffers {
    let B: Int
    let maxChunkSteps: Int
    let windowSteps: Int
    let recordCapacity: Int
    let laneBuf: MTLBuffer
    let seedsBuf: MTLBuffer
    let envKindsBuf: MTLBuffer
    let recordSlotBuf: MTLBuffer
    let recordBuf: MTLBuffer
    let snapBuf: MTLBuffer
    let diagAccumBuf: MTLBuffer
    let diagOutBuf: MTLBuffer
    let ecfg: FleetEnvCfg
    init(_ dev: MTLDevice, seeds: [Int], laneEnvKinds: [UInt32], episodeLen: Int) throws {
      func makeBuf(_ len: Int) throws -> MTLBuffer {
        guard let b = dev.makeBuffer(
          length: max(len, 16), options: [.storageModeShared])
        else {
          throw err("Metal buffer allocation failed (\(len) bytes).")
        }
        return b
      }
      B = seeds.count
      maxChunkSteps = 1024
      windowSteps = maxChunkSteps
      recordCapacity = 2 * maxChunkSteps
      laneBuf = try makeBuf(B * fleetLaneStride)
      seedsBuf = try makeBuf(B * 4)
      do {
        let sp = seedsBuf.contents().bindMemory(to: UInt32.self, capacity: B)
        for (i, s) in seeds.enumerated() { sp[i] = UInt32(truncatingIfNeeded: s) }
      }
      envKindsBuf = try makeBuf(B * 4)
      do {
        let kp = envKindsBuf.contents().bindMemory(to: UInt32.self, capacity: B)
        for i in 0..<B { kp[i] = laneEnvKinds[i] }
      }
      recordSlotBuf = try makeBuf(B * 4)
      do {
        let rp = recordSlotBuf.contents().bindMemory(to: Int32.self, capacity: B)
        for i in 0..<B { rp[i] = Int32(i) }
      }
      recordBuf = try makeBuf(B * recordCapacity * recordFloats * 4)
      let snapStride = MemoryLayout<LaneSnapshot>.stride
      precondition(snapStride == 36, "LaneSnapshot layout drifted from OakLaneSnapshot")
      snapBuf = try makeBuf(2 * B * snapStride)
      diagAccumBuf = try makeBuf(B * Diag.accumFloats * 4)
      diagOutBuf = try makeBuf(2 * B * Diag.outFloats * 4)
      memset(diagAccumBuf.contents(), 0, diagAccumBuf.length)
      memset(diagOutBuf.contents(), 0, diagOutBuf.length)
      ecfg = FleetEnvCfg(
        laneCount: UInt32(B), episodeLen: UInt32(episodeLen),
        recordCapacity: UInt32(recordCapacity), invObsFloor: 1 / Float(64))
    }
  }
  static func oakPipes(
    _ engine: Engine, _ lib: MTLLibrary, _ dev: MTLDevice, _ bufs: FleetBuffers,
    _ ablations: AblationSpec, _ e: [String: String]
  ) throws -> FleetPipes {
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let fn = try lib.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
      return try dev.makeComputePipelineState(function: fn)
    }
    let initPipe = try pipeline("oak_fleet_init")
    let snapPipe = try pipeline("oak_fleet_snapshot")
    let B = bufs.B
    let core = Int(engine.kinds[0].off.optionCount)
    let autoG = max(2, min(core, B / max(core, 1)))
    let megaG = e["OAK_MEGA_LANES_PER_TG"].flatMap(Int.init) ?? autoG
    let megaW = core * megaG
    func baseConstants() -> MTLFunctionConstantValues {
      let c = MTLFunctionConstantValues()
      ablations.apply(to: c)
      var megaG32 = UInt32(megaG)
      c.setConstantValue(&megaG32, type: .uint, index: OakConst.megaG.rawValue)
      if e["OAK_PROBE_FTZ"] == "1" {
        var b = true
        c.setConstantValue(&b, type: .bool, index: OakConst.probeFtz.rawValue)
      }
      if e["OAK_PROBE_FIXEDIDX"] == "1" {
        var b = true
        c.setConstantValue(&b, type: .bool, index: OakConst.probeFixedIdx.rawValue)
      }
      if let mask = e["OAK_MEGA_STAGE_MASK"].flatMap(UInt32.init) {
        var m = mask
        c.setConstantValue(&m, type: .uint, index: OakConst.megaStageMask.rawValue)
      }
      return c
    }
    var envRuns: [(kindIdx: Int, kind: UInt32, tgStart: Int, tgCount: Int)] = []
    var variantPipes: [MTLComputePipelineState] = []
    guard engine.kinds.allSatisfy({ $0.firstLane % megaG == 0 }) else {
      throw err(
        "Env schedule block starts must be multiples of megaG=\(megaG) for variant dispatch.")
    }
    for (ki, kd) in engine.kinds.enumerated() {
      let c = baseConstants()
      Engine.applyShapeConstants(c, kd.cfg)
      var k32 = kd.kind
      c.setConstantValue(&k32, type: .uint, index: OakConst.specEnvKind.rawValue)
      variantPipes.append(try dev.makeComputePipelineState(
        function: lib.makeFunction(name: "oak_megastep", constantValues: c)))
      envRuns.append((ki, kd.kind, kd.firstLane / megaG, (kd.count + megaG - 1) / megaG))
    }
    guard variantPipes.allSatisfy({ $0.threadExecutionWidth == core }) else {
      throw err("optionCount \(core) must equal simd width \(variantPipes[0].threadExecutionWidth).")
    }
    let pipeMaxThreads = variantPipes.map(\.maxTotalThreadsPerThreadgroup).min() ?? 0
    let tgOptionCount = Int(engine.kinds[0].off.optionCount)
    let megaSlotBytes = (5 * tgOptionCount + 1) * megaG * MemoryLayout<Float>.stride
    guard megaG >= 2, megaW <= pipeMaxThreads,
      megaSlotBytes <= dev.maxThreadgroupMemoryLength
    else {
      throw err(
        "OAK_MEGA_LANES_PER_TG=\(megaG) needs \(megaW) threads/threadgroup (pipeline max \(pipeMaxThreads)) and \(megaSlotBytes) tg bytes (device max \(dev.maxThreadgroupMemoryLength)).")
    }
    let res = engine.res
    let ecfg = bufs.ecfg
    func bindFleet(_ enc: MTLComputeCommandEncoder, _ pipe: MTLComputePipelineState) {
      enc.setComputePipelineState(pipe)
      for j in res.indices { enc.setBuffer(res[j], offset: 0, index: j) }
      enc.setBuffer(bufs.laneBuf, offset: 0, index: OakBuf.lanes.rawValue)
      enc.setBuffer(bufs.seedsBuf, offset: 0, index: OakBuf.seeds.rawValue)
      var c = ecfg
      enc.setBytes(&c, length: MemoryLayout<FleetEnvCfg>.stride, index: OakBuf.envCfg.rawValue)
    }
    let laneTG = mtlSize(
      min(64, initPipe.maxTotalThreadsPerThreadgroup, snapPipe.maxTotalThreadsPerThreadgroup))
    let diagHost = OakDiagHost(B)
    let diagPtr = bufs.diagOutBuf.contents().bindMemory(
      to: Float.self, capacity: 2 * B * Diag.outFloats)
    return FleetPipes(
      label: "oak",
      encodeInit: { enc in
        for kd in engine.kinds {
          bindFleet(enc, initPipe)
          enc.setBuffer(engine.params.b, offset: kd.paramBase * 4, index: OakBuf.params.rawValue)
          enc.setBuffer(engine.state.b, offset: kd.stateBase * 4, index: OakBuf.state.rawValue)
          enc.setBuffer(engine.uints.b, offset: kd.uintBase * 4, index: OakBuf.uints.rawValue)
          enc.setBuffer(kd.configB, offset: 0, index: OakBuf.config.rawValue)
          var ce = ecfg
          ce.invObsFloor = 1 / Float(2 * Int(kd.off.anchorCount))
          enc.setBytes(&ce, length: MemoryLayout<FleetEnvCfg>.stride, index: OakBuf.envCfg.rawValue)
          enc.setBuffer(bufs.envKindsBuf, offset: 0, index: OakBuf.aux0.rawValue)
          var off = UInt32(kd.firstLane)
          enc.setBytes(&off, length: 4, index: OakBuf.aux1.rawValue)
          enc.dispatchThreads(mtlSize(kd.count), threadsPerThreadgroup: laneTG)
        }
      },
      encodeMegastep: { enc, n, base in
        for run in envRuns {
          let kd = engine.kinds[run.kindIdx]
          bindFleet(enc, variantPipes[run.kindIdx])
          enc.setBuffer(engine.params.b, offset: kd.paramBase * 4, index: OakBuf.params.rawValue)
          enc.setBuffer(engine.state.b, offset: kd.stateBase * 4, index: OakBuf.state.rawValue)
          enc.setBuffer(engine.uints.b, offset: kd.uintBase * 4, index: OakBuf.uints.rawValue)
          enc.setBuffer(kd.configB, offset: 0, index: OakBuf.config.rawValue)
          var ce = ecfg
          ce.invObsFloor = 1 / Float(2 * Int(kd.off.anchorCount))
          enc.setBytes(&ce, length: MemoryLayout<FleetEnvCfg>.stride, index: OakBuf.envCfg.rawValue)
          enc.setBuffer(bufs.recordSlotBuf, offset: 0, index: OakBuf.aux0.rawValue)
          enc.setBuffer(bufs.recordBuf, offset: 0, index: OakBuf.aux1.rawValue)
          var span = SIMD3<UInt32>(UInt32(n), UInt32(base), UInt32(run.tgStart))
          enc.setBytes(&span, length: MemoryLayout<SIMD3<UInt32>>.stride, index: OakBuf.aux2.rawValue)
          enc.setBuffer(bufs.diagAccumBuf, offset: 0, index: OakBuf.diagAccum.rawValue)
          enc.setThreadgroupMemoryLength(megaSlotBytes, index: 0)
          enc.dispatchThreadgroups(
            MTLSize(width: run.tgCount, height: 1, depth: 1),
            threadsPerThreadgroup: mtlSize(megaW))
        }
        enc.memoryBarrier(scope: [.buffers])
      },
      encodeSnapshot: { enc, half in
        var fullScan: UInt32 = 1
        for kd in engine.kinds {
          bindFleet(enc, snapPipe)
          enc.setBuffer(engine.params.b, offset: kd.paramBase * 4, index: OakBuf.params.rawValue)
          enc.setBuffer(engine.state.b, offset: kd.stateBase * 4, index: OakBuf.state.rawValue)
          enc.setBuffer(engine.uints.b, offset: kd.uintBase * 4, index: OakBuf.uints.rawValue)
          enc.setBuffer(kd.configB, offset: 0, index: OakBuf.config.rawValue)
          enc.setBuffer(bufs.snapBuf, offset: 0, index: OakBuf.aux0.rawValue)
          var ctl = SIMD2<UInt32>(UInt32(half * B), UInt32(kd.firstLane))
          enc.setBytes(&ctl, length: MemoryLayout<SIMD2<UInt32>>.stride, index: OakBuf.aux1.rawValue)
          enc.setBytes(&fullScan, length: 4, index: OakBuf.aux2.rawValue)
          enc.setBuffer(bufs.diagAccumBuf, offset: 0, index: OakBuf.diagAccum.rawValue)
          enc.setBuffer(bufs.diagOutBuf, offset: 0, index: OakBuf.diagOut.rawValue)
          enc.dispatchThreadgroups(mtlSize(kd.count), threadsPerThreadgroup: mtlSize(core))
        }
      },
      writeDebugRaw: { rawDir, step in
        for lane in 0..<min(64, B) {
          let kd = engine.kind(of: lane)
          let d = Data(
            bytes: engine.statePtr(lane), count: kd.stateTotal * MemoryLayout<Float>.stride)
          try d.write(to: URL(fileURLWithPath: "\(rawDir)/state-l\(lane)-s\(step).bin"))
          let du = Data(
            bytes: engine.uintPtr(lane), count: kd.uintTotal * MemoryLayout<UInt32>.stride)
          try du.write(to: URL(fileURLWithPath: "\(rawDir)/uints-l\(lane)-s\(step).bin"))
          let dp = Data(
            bytes: engine.paramPtr(lane), count: kd.paramTotal * MemoryLayout<Float>.stride)
          try dp.write(to: URL(fileURLWithPath: "\(rawDir)/params-l\(lane)-s\(step).bin"))
        }
      },
      laneHashes: { lane, laneBytes in engine.stateHashes(lane, laneBytes: laneBytes) },
      diagFields: [
        "dormantFraction", "utilityDormantFraction", "featureBirths",
        "weightAbsMean", "weightAbsMeanValue", "erank",
        "logAlphaValueP10", "logAlphaValueP50", "logAlphaValueP90",
        "logAlphaOptionValueP50", "logAlphaModelP50",
      ],
      diagValues: { half, lane, stepCount in
        let rec = diagPtr + (half * B + lane) * Diag.outFloats
        let prevAge = diagHost.prevAge + lane * Diag.rank
        var ages = [UInt32](repeating: 0, count: Diag.rank)
        for i in 0..<Diag.rank { ages[i] = rec[Diag.oakFeatureAge + i].bitPattern }
        var dormantFraction = Float.nan
        if diagHost.hasPrev[lane] {
          var delta = [Double](repeating: .nan, count: Diag.rank)
          var sum = 0.0
          var validCount = 0
          for i in 0..<Diag.rank {
            let d = ages[i] &- prevAge[i]
            if d <= UInt32(stepCount) {
              delta[i] = Double(d)
              sum += Double(d)
              validCount += 1
            }
          }
          let mean = validCount > 0 ? sum / Double(validCount) : 0
          if mean > 0 {
            var dormant = 0
            for i in 0..<Diag.rank where delta[i].isFinite && delta[i] / mean < DiagMath.dormantTau {
              dormant += 1
            }
            dormantFraction = Float(dormant) / Float(Diag.rank)
          }
        }
        let utility = (0..<Diag.rank).map { Double(abs(rec[Diag.oakFeatureUtility + $0])) }
        let births = rec[Diag.oakFeatureBirths]
        let birthsPerStep: Float = diagHost.hasPrev[lane] && stepCount > 0
          ? (births - diagHost.prevBirths[lane]) / Float(stepCount) : .nan
        func alphas(_ off: Int) -> [Double] { (0..<Diag.rank).map { Double(rec[off + $0]) } }
        let valueQ = DiagMath.logQuantiles(alphas(Diag.oakValueAlpha), [0.1, 0.5, 0.9])
        let optionQ = DiagMath.logQuantiles(alphas(Diag.oakOptionValueAlpha), [0.5])
        let modelQ = DiagMath.logQuantiles(alphas(Diag.oakModelAlpha), [0.5])
        for i in 0..<Diag.rank { prevAge[i] = ages[i] }
        diagHost.prevBirths[lane] = births
        diagHost.hasPrev[lane] = true
        return [
          dormantFraction, DiagMath.dormantFraction(utility), birthsPerStep,
          rec[Diag.oakWeightAbsMean], rec[Diag.oakWeightAbsMeanValue],
          DiagMath.erank(rec + Diag.oakGram),
          valueQ[0], valueQ[1], valueQ[2], optionQ[0], modelQ[0],
        ]
      }
    )
  }
  static func fleetGPU(
    _ policy: String, _ seeds: [Int], _ steps: Int, _ root: URL, _ laneEnvs: [String],
    _ ablations: AblationSpec
  ) throws {
    guard laneEnvs.count == seeds.count else {
      throw err("laneEnvs (\(laneEnvs.count)) must match seed count (\(seeds.count)).")
    }
    let laneEnvKinds = try laneEnvs.map(envKind)
    let e = ProcessInfo.processInfo.environment
    guard let dev = MTLCreateSystemDefaultDevice() else { throw err("Metal unavailable.") }
    let lib = try dev.makeLibrary(URL: try requireFreshMetallib())
    let engine: Engine?
    let q: MTLCommandQueue
    let envTable: [EnvSpec]
    if policy == "oak" {
      let eng = try Engine(laneEnvs: laneEnvs, batch: seeds.count, dev: dev)
      engine = eng
      q = eng.q
      envTable = eng.envTable
    } else {
      engine = nil
      guard let queue = dev.makeCommandQueue() else { throw err("Metal unavailable.") }
      q = queue
      envTable = try Engine.runEnvTableProbe(dev, lib)
    }
    do {
      precondition(
        envStepFlops.count == envIdsByKind.count, "envStepFlops drifted from envIdsByKind")
      func fnv1a(_ s: String) -> UInt32 {
        var h: UInt32 = 0x811c_9dc5
        for b in Array(s.utf8) { h = (h ^ UInt32(b)) &* 16_777_619 }
        return h
      }
      for (i, name) in Self.envIdsByKind.enumerated() {
        guard envTable[i].nameHash == fnv1a(name) else {
          throw err("Env table drift at kind \(i): kernel hash != fnv1a(\(name)).")
        }
        let declared = Set(envTable[i].slots.indices.filter { envTable[i].slots[$0] != 0 })
        let recorded = Set(RecordCodec.byKind[i].fields.map(\.idx))
        guard declared == recorded else {
          throw err(
            "\(name): kernel declares slots \(declared.sorted()) but the codec records \(recorded.sorted())."
          )
        }
      }
    }
    var offByKind: [UInt32: ShapeProbe] = [:]
    engine?.kinds.forEach { offByKind[$0.kind] = $0.off }
    let fleetD = engine?.kinds.first?.d ?? 0
    var envFlopsPerStep: [String: Double] = [:]
    var flopsSum = 0.0
    for kindU in laneEnvKinds {
      let kind = Int(kindU)
      let envId = envIdsByKind[kind]
      let flops = envFlopsPerStep[envId] ?? policyFlopsPerStep(
        policy, kind: kind, spec: envTable[kind], off: offByKind[kindU], d: fleetD)
      envFlopsPerStep[envId] = flops
      flopsSum += flops
    }
    Self.pendingMeta["envFlopsPerStep"] = envFlopsPerStep
    Self.pendingMeta["flopsPerStep"] = flopsSum / Double(laneEnvKinds.count)
    try Self.writeRunMeta(root, envTable)
    for flag in ["OAK_PROBE_FTZ", "OAK_PROBE_FIXEDIDX", "OAK_MEGA_STAGE_MASK"] where e[flag] != nil {
      fputs("\(flag)=\(e[flag]!): timing instrument, trajectories are garbage\n", stderr)
    }
    let episodeLen = max(e["OAK_EPISODE_STEPS"].flatMap(Int.init) ?? 0, 0)
    let bufs = try FleetBuffers(dev, seeds: seeds, laneEnvKinds: laneEnvKinds, episodeLen: episodeLen)
    let pipes: FleetPipes
    if let engine {
      pipes = try oakPipes(engine, lib, dev, bufs, ablations, e)
    } else {
      pipes = try baselinePipes(policy, lib, dev, bufs, envTable, laneEnvKinds, e)
    }
    let gpuSeconds = try runFleetLoop(
      q, bufs, pipes, seeds: seeds, steps: steps, root: root, laneEnvs: laneEnvs,
      laneEnvKinds: laneEnvKinds)
    Self.pendingMeta["gpuSeconds"] = gpuSeconds
    Self.pendingMeta["finishedAt"] = ISO8601DateFormatter().string(from: Date())
    try Self.writeRunMeta(root, envTable)
  }
  static let ppoParamStride = 2768
  static let ppoRollStride = 3744
  static let ppoScratchStride = 12064
  static let ppoStateStride = 48
  static let ppoLR = 3e-4
  static let ppoAdamB2 = 0.999
  static let ppoAdamEps = 1e-5
  static func baselinePipes(
    _ policy: String, _ lib: MTLLibrary, _ dev: MTLDevice, _ bufs: FleetBuffers,
    _ envTable: [EnvSpec], _ laneEnvKinds: [UInt32], _ e: [String: String]
  ) throws -> FleetPipes {
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let constants = MTLFunctionConstantValues()
      if let mask = e["OAK_MEGA_STAGE_MASK"].flatMap(UInt32.init) {
        var m = mask
        constants.setConstantValue(&m, type: .uint, index: OakConst.megaStageMask.rawValue)
      }
      let fn = try lib.makeFunction(name: name, constantValues: constants)
      return try dev.makeComputePipelineState(function: fn)
    }
    func makeBuf(_ len: Int) throws -> MTLBuffer {
      guard let b = dev.makeBuffer(length: max(len, 16), options: [.storageModeShared]) else {
        throw err("Metal buffer allocation failed (\(len) bytes).")
      }
      return b
    }
    let isPpo = policy == "ppo"
    let initPipe = try pipeline(isPpo ? "ppo_fleet_init" : "baseline_fleet_init")
    let stepPipe = try pipeline("\(policy)_megastep")
    let snapPipe = try pipeline(isPpo ? "ppo_fleet_snapshot" : "baseline_fleet_snapshot")
    let B = bufs.B
    let ecfg = bufs.ecfg
    let ppoBufs: [MTLBuffer]
    if isPpo {
      guard stepPipe.maxTotalThreadsPerThreadgroup >= 32 else {
        throw err("ppo_megastep supports only \(stepPipe.maxTotalThreadsPerThreadgroup) threads/threadgroup.")
      }
      ppoBufs = [
        try makeBuf(B * ppoParamStride * 4),
        try makeBuf(B * 2 * ppoParamStride * 4),
        try makeBuf(B * ppoRollStride * 4),
        try makeBuf(B * ppoScratchStride * 4),
        try makeBuf(B * ppoStateStride * 4),
      ]
    } else {
      ppoBufs = []
    }
    func bindCommon(_ enc: MTLComputeCommandEncoder, _ pipe: MTLComputePipelineState) {
      enc.setComputePipelineState(pipe)
      for (j, b) in ppoBufs.enumerated() { enc.setBuffer(b, offset: 0, index: j) }
      enc.setBuffer(bufs.laneBuf, offset: 0, index: OakBuf.lanes.rawValue)
      enc.setBuffer(bufs.seedsBuf, offset: 0, index: OakBuf.seeds.rawValue)
      var c = ecfg
      enc.setBytes(&c, length: MemoryLayout<FleetEnvCfg>.stride, index: OakBuf.envCfg.rawValue)
    }
    let laneTG = mtlSize(min(
      64, initPipe.maxTotalThreadsPerThreadgroup, stepPipe.maxTotalThreadsPerThreadgroup,
      snapPipe.maxTotalThreadsPerThreadgroup))
    let stepThreads = isPpo ? B * 32 : B
    let stepTG = isPpo ? mtlSize(32) : laneTG
    let diagPtr = bufs.diagOutBuf.contents().bindMemory(
      to: Float.self, capacity: 2 * B * Diag.outFloats)
    let ppoParamCounts: [Int] = laneEnvKinds.map { kind in
      let spec = envTable[Int(kind)]
      let box = spec.actionKind == 0
      return 2 * spec.naturalO * Diag.k + 3 * Diag.k + Diag.k * spec.actionCount
        + spec.actionCount + 1 + (box ? spec.actionCount : 0)
    }
    return FleetPipes(
      label: policy,
      encodeInit: { enc in
        bindCommon(enc, initPipe)
        enc.setBuffer(bufs.envKindsBuf, offset: 0, index: OakBuf.aux0.rawValue)
        enc.dispatchThreads(mtlSize(stepThreads), threadsPerThreadgroup: stepTG)
      },
      encodeMegastep: { enc, n, base in
        bindCommon(enc, stepPipe)
        enc.setBuffer(bufs.recordSlotBuf, offset: 0, index: OakBuf.aux0.rawValue)
        enc.setBuffer(bufs.recordBuf, offset: 0, index: OakBuf.aux1.rawValue)
        var span = SIMD3<UInt32>(UInt32(n), UInt32(base), 0)
        enc.setBytes(&span, length: MemoryLayout<SIMD3<UInt32>>.stride, index: OakBuf.aux2.rawValue)
        if isPpo {
          enc.setBuffer(bufs.diagAccumBuf, offset: 0, index: OakBuf.diagAccum.rawValue)
        }
        enc.dispatchThreads(mtlSize(stepThreads), threadsPerThreadgroup: stepTG)
        enc.memoryBarrier(scope: [.buffers])
      },
      encodeSnapshot: { enc, half in
        bindCommon(enc, snapPipe)
        enc.setBuffer(bufs.snapBuf, offset: 0, index: OakBuf.aux0.rawValue)
        var snapBase = UInt32(half * B)
        enc.setBytes(&snapBase, length: 4, index: OakBuf.aux1.rawValue)
        if isPpo {
          enc.setBuffer(bufs.diagAccumBuf, offset: 0, index: OakBuf.diagAccum.rawValue)
          enc.setBuffer(bufs.diagOutBuf, offset: 0, index: OakBuf.diagOut.rawValue)
        }
        enc.dispatchThreads(mtlSize(B), threadsPerThreadgroup: laneTG)
      },
      writeDebugRaw: { _, _ in },
      laneHashes: { lane, laneBytes in
        var hs = ["lane": sha256Hex(UnsafeRawBufferPointer(start: laneBytes, count: fleetLaneStride))]
        if isPpo {
          let base = ppoBufs[0].contents() + lane * ppoParamStride * 4
          hs["params"] = sha256Hex(UnsafeRawBufferPointer(start: base, count: ppoParamStride * 4))
        }
        return hs
      },
      diagFields: isPpo
        ? [
          "dormantFraction", "dormantFractionCritic", "weightAbsMean", "erank",
          "logStepP10", "logStepP50", "logStepP90",
        ] : [],
      diagValues: { half, lane, _ in
        guard isPpo else { return [] }
        let rec = diagPtr + (half * B + lane) * Diag.outFloats
        let actor = (0..<Diag.k).map { Double(rec[Diag.ppoSumAbsHiddenActor + $0]) }
        let critic = (0..<Diag.k).map { Double(rec[Diag.ppoSumAbsHiddenCritic + $0]) }
        let adamT = Double(rec[Diag.ppoAdamT])
        var stepQ: [Float] = [.nan, .nan, .nan]
        if adamT > 0 {
          let bc2 = 1 - pow(ppoAdamB2, adamT)
          let effs = (0..<ppoParamCounts[lane]).map { p in
            ppoLR / ((Double(rec[Diag.ppoAdamV + p]) / bc2).squareRoot() + ppoAdamEps)
          }
          stepQ = DiagMath.logQuantiles(effs, [0.1, 0.5, 0.9])
        }
        return [
          DiagMath.dormantFraction(actor), DiagMath.dormantFraction(critic),
          rec[Diag.ppoWeightAbsMean], DiagMath.erank(rec + Diag.ppoGram),
          stepQ[0], stepQ[1], stepQ[2],
        ]
      }
    )
  }
  static func runFleetLoop(
    _ q: MTLCommandQueue, _ bufs: FleetBuffers, _ pipes: FleetPipes,
    seeds: [Int], steps: Int, root: URL, laneEnvs: [String], laneEnvKinds: [UInt32]
  ) throws -> Double {
    let e = ProcessInfo.processInfo.environment
    let fm = FileManager.default
    let B = bufs.B
    let dev = q.device
    func seedDir(_ lane: Int) throws -> URL {
      let dir = root.appendingPathComponent(laneEnvs[lane], isDirectory: true)
        .appendingPathComponent("seed\(seeds[lane])", isDirectory: true)
      try fm.createDirectory(at: dir, withIntermediateDirectories: true)
      return dir
    }
    var recordHandles: [FileHandle] = []
    var records: [RecordWriter] = []
    for lane in 0..<B {
      let dir = try seedDir(lane)
      let url = dir.appendingPathComponent("pose.bin")
      fm.createFile(atPath: url.path, contents: nil)
      let h = try FileHandle(forWritingTo: url)
      records.append(try RecordWriter(handle: h, envKind: laneEnvKinds[lane]))
      recordHandles.append(h)
    }
    let maxChunkSteps = bufs.maxChunkSteps
    let windowSteps = bufs.windowSteps
    let recordCapacity = bufs.recordCapacity
    let snapPtr = bufs.snapBuf.contents().bindMemory(to: LaneSnapshot.self, capacity: 2 * B)
    let dumpEvery = max(e["OAK_DUMP_STATE_EVERY"].flatMap(Int.init) ?? 0, 0)
    var hashHandles: [FileHandle?] = Array(repeating: nil, count: B)
    if dumpEvery > 0 {
      for (lane, _) in seeds.enumerated() {
        let url = try seedDir(lane).appendingPathComponent("state-hashes.jsonl")
        fm.createFile(atPath: url.path, contents: nil)
        hashHandles[lane] = try FileHandle(forWritingTo: url)
      }
    }
    let tape = try TapeWriter(
      url: root.appendingPathComponent("tape.bin"), laneCount: B, diagFields: pipes.diagFields)
    defer {
      hashHandles.forEach { try? $0?.close() }
      recordHandles.forEach { try? $0.close() }
    }
    let timing = e["OAK_FLEET_TIMING"] == "1"
    do {
      guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
        throw err("Failed to encode Metal command.")
      }
      pipes.encodeInit(enc)
      enc.endEncoding()
      cb.commit()
      cb.waitUntilCompleted()
      if let e = cb.error { throw err("Metal command failed: \(e)") }
    }
    final class WriterState: @unchecked Sendable {
      var deadStep: [Int?]
      var deadDetector: [UInt32?]
      var deadCount = 0
      var lastDrainStep = 0
      var gpuSeconds = 0.0
      private let errLock = NSLock()
      private var failure: Error?
      private var failed = false
      var error: Error? {
        get { errLock.lock(); defer { errLock.unlock() }; return failure }
        set {
          errLock.lock()
          defer { errLock.unlock() }
          if failure == nil { failure = newValue }
          failed = newValue != nil
        }
      }
      var didFail: Bool { errLock.lock(); defer { errLock.unlock() }; return failed }
      init(_ b: Int) {
        deadStep = Array(repeating: nil, count: b)
        deadDetector = Array(repeating: nil, count: b)
      }
    }
    let st = WriterState(B)
    let writer = DispatchQueue(label: "oak.fleet.gpu.writer")
    let wallStart = DispatchTime.now().uptimeNanoseconds
    func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1e9 }
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
    func drainRecords(step: Int) throws {
      guard step > st.lastDrainStep else { return }
      let t0 = DispatchTime.now().uptimeNanoseconds
      let start = st.lastDrainStep
      let count = step - start
      let rp = bufs.recordBuf.contents().bindMemory(
        to: Float.self, capacity: B * recordCapacity * recordFloats)
      let before = timing ? records.reduce(0) { $0 + $1.bytesWritten } : 0
      let box = DrainErr()
      let writers = records
      let cap = recordCapacity
      let lanes = B
      DispatchQueue.concurrentPerform(iterations: writers.count) { slot in
        do {
          try writers[slot].encodeChunk(
            rp, slot: slot, capacity: cap, laneCount: lanes, stepStart: start, stepCount: count)
        } catch {
          box.record(error)
        }
      }
      if let e = box.take() { throw e }
      st.lastDrainStep = step
      if timing {
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let wrote = records.reduce(0) { $0 + $1.bytesWritten } - before
        let raw = B * count * recordFloats * 4
        fputs(
          String(
            format: "[record] step %d | %d lanes x %d rows | %.1f ms | %.2f MiB -> %.2f MiB (%.1fx)\n",
            step, B, count, ms, Double(raw) / 1_048_576,
            Double(wrote) / 1_048_576, wrote > 0 ? Double(raw) / Double(wrote) : 0), stderr)
      }
    }
    func latchDeaths(half: Int, step: Int) {
      for lane in 0..<B where st.deadStep[lane] == nil {
        defer { if st.deadStep[lane] != nil { st.deadCount += 1 } }
        let snap = snapPtr[half * B + lane]
        var detector: UInt32? = nil
        var dStep = step
        if snap.deathStep != 0xFFFF_FFFF {
          detector = (1...3).contains(snap.deathDetector) ? snap.deathDetector : 7
          dStep = Int(snap.deathStep)
        } else if snap.paramNonFinite != 0 {
          detector = 4
        } else if Double(snap.paramAbsMax) > 1e16 {
          detector = 5
        } else if !snap.scalars.averageReward.isFinite {
          detector = 6
        }
        if let detector {
          st.deadStep[lane] = dStep
          st.deadDetector[lane] = detector
        }
      }
    }
    func writeWindow(half: Int, stepStart: Int, step: Int) throws {
      latchDeaths(half: half, step: step)
      let diagCount = pipes.diagFields.count
      var diag = [Float](repeating: .nan, count: B * diagCount)
      if diagCount > 0 {
        let stepCount = step - stepStart
        diag.withUnsafeMutableBufferPointer { buf in
          let base = buf.baseAddress!
          DispatchQueue.concurrentPerform(iterations: B) { lane in
            let values = pipes.diagValues(half, lane, stepCount)
            for f in 0..<diagCount { base[lane * diagCount + f] = values[f] }
          }
        }
      }
      try tape.append(
        snap: snapPtr, half: half, deadStep: st.deadStep, deadDetector: st.deadDetector,
        diag: diag, stepStart: stepStart, stepCount: step - stepStart, wallSeconds: now())
      try drainRecords(step: step)
      if timing {
        let alive = B - st.deadCount
        let wall = now()
        fputs(
          "[fleet \(pipes.label) gpu]" + String(
            format: " step %d alive %d/%d | gpu %.1fs wall %.1fs | %.0f agg steps/s\n",
            step, alive, B, st.gpuSeconds, wall,
            wall > 0 ? Double(step) * Double(B) / wall : 0), stderr)
      }
    }
    func writeHashes(step: Int) throws {
      if let rawDir = e["OAK_DEBUG_RAW"] {
        try pipes.writeDebugRaw(rawDir, step)
      }
      for lane in 0..<B {
        guard let h = hashHandles[lane] else { continue }
        let laneBytes = bufs.laneBuf.contents() + lane * fleetLaneStride
        let hs = pipes.laneHashes(lane, laneBytes)
        var obj: [String: Any] = ["seed": seeds[lane], "step": step]
        for (k, v) in hs { obj[k] = v }
        try h.write(contentsOf: jsonLine(obj))
      }
    }
    let explicitN = e["OAK_STEPS_PER_CB"].flatMap(Int.init)
    let tuneLock = NSLock()
    var tunedN = max(explicitN ?? 32, 1)
    guard let chunkOrder = dev.makeEvent() else { throw err("Failed to create Metal event.") }
    var chunkSeq: UInt64 = 0
    let inflight = DispatchSemaphore(value: 2)
    let windowSlots = DispatchSemaphore(value: 1)
    var snapHalf = 0
    var lastWindowStep = 0
    var step = 0
    while step < steps {
      tuneLock.lock()
      let n = tunedN
      tuneLock.unlock()
      var boundary = min(((step / windowSteps) + 1) * windowSteps, steps)
      if dumpEvery > 0 { boundary = min(boundary, ((step / dumpEvery) + 1) * dumpEvery) }
      let target = min(boundary, step + n)
      let isWindow = target % windowSteps == 0 || target == steps
      let isDump = dumpEvery > 0 && target % dumpEvery == 0
      inflight.wait()
      if st.didFail, let err = st.error { throw err }
      if isWindow { windowSlots.wait() }
      guard let cb = q.makeCommandBuffer() else {
        throw err("Failed to encode Metal command.")
      }
      cb.encodeWaitForEvent(chunkOrder, value: chunkSeq)
      guard let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else {
        throw err("Failed to encode Metal command.")
      }
      pipes.encodeMegastep(enc, target - step, step)
      if isWindow {
        pipes.encodeSnapshot(enc, snapHalf)
      }
      enc.endEncoding()
      chunkSeq += 1
      cb.encodeSignalEvent(chunkOrder, value: chunkSeq)
      let thisHalf = snapHalf
      if isWindow { snapHalf ^= 1 }
      let tStep = target
      let nSteps = target - step
      let wStart = lastWindowStep
      if isWindow { lastWindowStep = target }
      cb.addCompletedHandler { cb in
        let gpuT = cb.gpuEndTime - cb.gpuStartTime
        if explicitN == nil, gpuT > 0, nSteps > 0 {
          let fit = max(1, min(Int(0.08 / (gpuT / Double(nSteps))), maxChunkSteps))
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
          if isWindow, st.error == nil {
            do { try writeWindow(half: thisHalf, stepStart: wStart, step: tStep) } catch {
              st.error = error
            }
          }
          if isWindow { windowSlots.signal() }
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
    return st.gpuSeconds
  }
  static func compareHashes() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count == 3 else {
      throw err("Usage: OaK compare-hashes <left.jsonl|dir> <right.jsonl|dir>")
    }
    func asInt(_ v: Any?) -> Int? {
      if let i = v as? Int { return i }
      if let n = v as? NSNumber { return n.intValue }
      return nil
    }
    func files(_ path: String) throws -> [URL] {
      var isDir: ObjCBool = false
      guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir) else {
        throw err("compare-hashes: missing \(path)")
      }
      if !isDir.boolValue { return [URL(fileURLWithPath: path)] }
      let it = FileManager.default.enumerator(atPath: path)
      var urls: [URL] = []
      while let rel = it?.nextObject() as? String {
        if rel.hasSuffix("/state-hashes.jsonl") || rel == "state-hashes.jsonl" {
          urls.append(URL(fileURLWithPath: path).appendingPathComponent(rel))
        }
      }
      guard !urls.isEmpty else { throw err("compare-hashes: no state-hashes.jsonl under \(path)") }
      return urls
    }
    func load(_ path: String) throws -> [String: [String: String]] {
      var out: [String: [String: String]] = [:]
      for url in try files(path) {
        let text = try String(contentsOf: url, encoding: .utf8)
        for line in text.split(whereSeparator: \.isNewline) {
          guard let data = String(line).data(using: .utf8),
            let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let seed = asInt(obj["seed"]), let step = asInt(obj["step"])
          else {
            throw err("compare-hashes: bad line in \(url.path)")
          }
          var hashes: [String: String] = [:]
          for (k, v) in obj where k != "seed" && k != "step" {
            guard let s = v as? String else {
              throw err("compare-hashes: non-string field '\(k)' in \(url.path)")
            }
            hashes[k] = s
          }
          out["\(seed)@\(step)"] = hashes
        }
      }
      return out
    }
    let left = try load(args[1])
    let right = try load(args[2])
    let required = ["params", "scalars", "stateReward", "uintsReward", "lane"]
    var missing = 0
    var failed = 0
    var extra = 0
    let keys = Set(left.keys).union(right.keys).sorted()
    for key in keys {
      guard let l = left[key], let r = right[key] else {
        missing += 1
        fputs("missing \(key) left=\(left[key] != nil) right=\(right[key] != nil)\n", stderr)
        continue
      }
      var bad: [String] = []
      for field in required {
        let lv = l[field]
        let rv = r[field]
        if lv == nil, rv == nil { continue }
        guard let lv, let rv else {
          bad.append("\(field):absent")
          continue
        }
        if lv != rv { bad.append(field) }
      }
      let optional = Set(l.keys).union(r.keys).subtracting(required)
      var optBad: [String] = []
      for field in optional.sorted() {
        if l[field] != r[field] { optBad.append(field) }
      }
      extra += optBad.count
      if !bad.isEmpty {
        failed += 1
        fputs("FAIL \(key) reward=[\(bad.joined(separator: ","))] other=[\(optBad.joined(separator: ","))]\n", stderr)
      } else if !optBad.isEmpty {
        fputs("ok \(key) other=[\(optBad.joined(separator: ","))]\n", stderr)
      }
    }
    if missing > 0 || failed > 0 {
      throw err(
        "compare-hashes: \(failed) reward-state mismatches, \(missing) missing rows, \(extra) whole-buffer diffs.")
    }
    print(
      "compare-hashes: \(keys.count) rows match on \(required.joined(separator: ",")).")
  }
  static func benchmark() throws {
    let env = ProcessInfo.processInfo.environment
    let ablations = try AblationSpec.fromEnv(env)
    let policy = env["OAK_BENCH_POLICY"] ?? "oak"
    guard ["oak", "random", "ppo"].contains(policy) else {
      throw err("OAK_BENCH_POLICY must be one of oak,random,ppo (got \(policy)).")
    }
    guard policy == "oak" || ablations.label == "none" else {
      throw err("Ablations apply only to the oak policy.")
    }
    let fm = FileManager.default
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
    let raw = (env["OAK_BENCH_ENVS"] ?? "").trimmingCharacters(in: .whitespaces)
    guard !raw.isEmpty else {
      throw err("OAK_BENCH_ENVS is required (EnvId:count,... covering every lane).")
    }
    var laneEnvs = [String](repeating: "", count: seedValues.count)
    var entries: [(envId: String, seedStart: Int, seedEnd: Int)] = []
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
          "OAK_BENCH_ENVS schedule overflows the fleet: \(offset + count) lanes > B=\(seedValues.count)."
        )
      }
      for i in 0..<count { laneEnvs[offset + i] = name }
      entries.append((name, seedValues[offset], seedValues[offset + count - 1]))
      offset += count
    }
    guard offset == seedValues.count else {
      throw err("OAK_BENCH_ENVS schedule covers \(offset) of B=\(seedValues.count) lanes.")
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
      if env["OAK_METRICS_STORAGE"] != nil {
        throw err("OAK_METRICS_STORAGE points at non-empty \(dir.path); refusing to write beside existing data.")
      }
      dir = URL(
        fileURLWithPath: "\(dir.path)-\(ProcessInfo.processInfo.processIdentifier)",
        isDirectory: true)
    }
    try fm.createDirectory(at: dir, withIntermediateDirectories: true)
    Self.pendingMeta = [
      "batch": seedValues.count, "seeds": seedValues, "steps": steps,
      "policy": policy,
      "ablationLabel": ablations.label,
      "startedAt": ISO8601DateFormatter().string(from: Date()),
      "envSchedule": entries.map { e in
        ["envId": e.envId, "seedStart": e.seedStart, "seedEnd": e.seedEnd] as [String: Any]
      },
    ]
    let envDesc = entries.map { "\($0.envId):\($0.seedEnd - $0.seedStart + 1)" }.joined(
      separator: ",")
    fputs(
      "OaK fleet: policy=\(policy) B=\(seedValues.count) lanes × \(steps) steps on \(envDesc) (ablation=\(ablations.label), dir=\(dir.path))\n",
      stderr)
    try fleetGPU(policy, seedValues, steps, dir, laneEnvs, ablations)
  }
}
