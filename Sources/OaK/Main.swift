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
  var laneCount, recordCapacity, persistentMask: UInt32
  var invObsFloor: Float
}
struct LaneSnapshot {
  var scalars: KernelScalars
  var paramAbsMax: Float
  var paramNonFinite: UInt32
  var rewardSum, rho: Float
  var deathStep, deathDetector: UInt32
}
let fleetLaneStride = 44

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
  static let k = 64
  static let gram = k * k
  static let accumGram = 0
  static let accumSumAbsHiddenActor = accumGram + gram
  static let accumSumAbsHiddenCritic = accumSumAbsHiddenActor + k
  static let accumFloats = accumSumAbsHiddenCritic + k
  static func oakFloats(_ rank: Int) -> Int { 5 * rank + 3 + gram }
  static func ppoParams(obsMax: Int, actMax: Int) -> Int {
    (2 * obsMax * k + 2 * k + k * actMax + 2 * actMax + k + 1 + 3) & ~3
  }
  static let ppoSumAbsHiddenActor = 0
  static let ppoSumAbsHiddenCritic = ppoSumAbsHiddenActor + k
  static let ppoWeightAbsMean = ppoSumAbsHiddenCritic + k
  static let ppoAdamT = ppoWeightAbsMean + 1
  static let ppoAdamV = ppoAdamT + 1
  static func ppoGram(_ params: Int) -> Int { ppoAdamV + params }
  static func ppoFloats(_ params: Int) -> Int { ppoAdamV + params + gram }
  static func oakRank(_ spec: EnvSpec) -> Int {
    let anchors = max(32, (spec.naturalO + spec.actionParams + 2 + 31) / 32 * 32)
    return anchors * 15
  }
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
  static let stride = 12 + poseLen
  static let slotNames = [
    "none", "x", "y", "gx", "gy", "kx", "ky", "carry", "gates", "tokens", "cue", "heading",
    "effort", "banana", "onion", "jellyBean", "truffle", "depletionLo", "depletionHi",
  ]
  let nameHash: UInt32
  let naturalO: Int
  let actionKind: Int
  let actionCount: Int
  let actionParams: Int
  let gridW: Int
  let gridH: Int
  let controlHz: Int
  let envStateWords: Int
  let mjIndex: Int
  let laneThreads: Int
  let slots: [Int]
  let persistentEntryWords: Int
  static func slotName(_ code: Int) -> String {
    if code >= 32 { return "q\(code - 32)" }
    guard code >= 0, code < slotNames.count else { return "none" }
    return slotNames[code]
  }
}

struct EnvTable {
  let specs: [EnvSpec]
  let constantWords: Int
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
  static let simdWidth = 32
  static func runShapeProbe(
    _ dev: MTLDevice, _ lib: MTLLibrary, obsDim: Int, dPolicy: Int
  ) throws -> ShapeProbe {
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
  static func runEnvTableProbe(_ dev: MTLDevice, _ lib: MTLLibrary) throws -> EnvTable {
    let fn = lib.makeFunction(name: "oak_env_table_probe")!
    let pipe = try dev.makeComputePipelineState(function: fn)
    let words = 5 + Main.envIdsByKind.count * EnvSpec.stride
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
    let specs = (0..<Main.envIdsByKind.count).map { i in
      let row = 5 + i * EnvSpec.stride
      return EnvSpec(
        nameHash: tbl[row], naturalO: Int(tbl[row + 1]), actionKind: Int(tbl[row + 2]),
        actionCount: Int(tbl[row + 3]), actionParams: Int(tbl[row + 4]),
        gridW: Int(tbl[row + 5]), gridH: Int(tbl[row + 6]),
        controlHz: Int(tbl[row + 7]), envStateWords: Int(tbl[row + 8]),
        mjIndex: tbl[row + 9] == 0xffff_ffff ? -1 : Int(tbl[row + 9]),
        laneThreads: Int(tbl[row + 10]),
        slots: (0..<EnvSpec.poseLen).map { Int(tbl[row + 11 + $0]) },
        persistentEntryWords: Int(tbl[row + 11 + EnvSpec.poseLen]))
    }
    if let bad = specs.first(where: { $0.mjIndex >= 0 && $0.laneThreads <= 1 }) {
      throw err(
        "Env \(bad.nameHash) uses the physics engine but declares laneThreads "
          + "\(bad.laneThreads); the engine requires a full simdgroup per lane.")
    }
    if let bad = specs.first(where: { $0.laneThreads != 1 && $0.laneThreads != 32 }) {
      throw err("Env \(bad.nameHash) declares laneThreads \(bad.laneThreads); expected 1 or 32.")
    }
    guard tbl[1] == UInt32(specs.map(\.naturalO).max()!),
      tbl[2] == UInt32(specs.map(\.actionCount).max()!),
      tbl[3] >= UInt32(specs.filter { $0.mjIndex < 0 }.map(\.envStateWords).max() ?? 0)
    else {
      throw err(
        "Env maxima drift: kernel obs/act max \(tbl[1])/\(tbl[2]) vs table "
          + "\(specs.map(\.naturalO).max()!)/\(specs.map(\.actionCount).max()!).")
    }
    return EnvTable(specs: specs, constantWords: Int(tbl[4]))
  }
  init(
    laneEnvs: [String], batch b: Int, dev: MTLDevice, lib: MTLLibrary,
    envTable table: [EnvSpec]
  ) throws {
    precondition(b >= 1 && laneEnvs.count == b, "Engine batch/laneEnvs mismatch")
    batch = b
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
    for r in runs {
      let spec = table[Int(r.kind)]
      let naturalO = spec.naturalO
      let dk = spec.actionParams
      let sp = try probeCache[r.kind] ?? Engine.runShapeProbe(
        dev, lib, obsDim: naturalO, dPolicy: dk)
      probeCache[r.kind] = sp
      let anchors = Int(sp.anchorCount)
      precondition(anchors % Engine.simdWidth == 0, "perception budget must tile the simd width")
      precondition(Int(sp.rank) == anchors * 15, "gate-bank identity broken")
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
  func stateHashes(_ lane: Int, laneBytes: UnsafeRawPointer, laneByteCount: Int)
    -> [String: String]
  {
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
      "lane": hex(laneBytes, laneByteCount),
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
      RecordCodec(category: "jellyBean", fields: (0..<7).map { RecordField($0) }),
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

extension Data {
  mutating func appendLE<T: FixedWidthInteger>(_ v: T) {
    Swift.withUnsafeBytes(of: v.littleEndian) { append(contentsOf: $0) }
  }
}

final class RecordWriter: @unchecked Sendable {
  static let chunkMagic: UInt32 = 0x524B_4843

  let codec: RecordCodec
  private var buf: [UInt8] = []
  private(set) var closed = false

  init(envKind: UInt32) throws {
    self.codec = try RecordCodec.table(envKind)
  }

  func encodeChunk(
    _ rp: UnsafePointer<Float>, slot: Int, capacity: Int, laneCount: Int,
    stepStart: Int, stepCount: Int
  ) throws -> Data? {
    guard !closed, stepCount > 0 else { return nil }
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
    guard live > 0 else { return nil }

    buf.removeAll(keepingCapacity: true)
    buf.append(contentsOf: repeatElement(0, count: (live + 7) / 8))
    for i in 0..<live where cell(i, 0) != 0 { buf[i >> 3] |= UInt8(1 << (i & 7)) }

    var plans: [(varint: ColumnPlan, full: ColumnPlan, seed: UInt64?)] = []
    for f in fields {
      var q = [Int64](repeating: 0, count: live)
      var seed: UInt64? = nil
      if f.anchor {
        var recon = value(0, f)!
        seed = recon.bitPattern
        var total: Int64 = 0
        for i in 0..<live {
          let d = Int64(((value(i, f)! - recon) / f.scale).rounded())
          recon += Double(d) * f.scale
          total &+= d
          q[i] = total
        }
      } else {
        for i in 0..<live { q[i] = quantized(i, f)! }
      }
      let (varint, full) = Self.encodeColumn(q)
      plans.append((varint, full, seed))
    }
    func assemble(_ pick: (ColumnPlan, ColumnPlan, UInt64?) -> ColumnPlan) -> [UInt8] {
      var out = buf
      for plan in plans {
        let chosen = pick(plan.varint, plan.full, plan.seed)
        out.append(UInt8(chosen.pred | (chosen.coder << 2)))
      }
      for plan in plans {
        let chosen = pick(plan.varint, plan.full, plan.seed)
        if let seed = plan.seed {
          withUnsafeBytes(of: seed.littleEndian) { out.append(contentsOf: $0) }
        }
        out.append(contentsOf: chosen.bytes)
      }
      return out
    }
    let payloadA = Data(assemble { varint, _, _ in varint })
    let payloadB = Data(assemble { _, full, _ in full })
    let compA = try (payloadA as NSData).compressed(using: .zlib) as Data
    let compB = try (payloadB as NSData).compressed(using: .zlib) as Data
    let optionA: (payload: Data, stored: Data, flags: UInt32) = (
      payloadA, compA.count < payloadA.count ? compA : payloadA,
      compA.count < payloadA.count ? 1 << 31 : 0)
    let optionB: (payload: Data, stored: Data, flags: UInt32) = (
      payloadB, compB.count < payloadB.count ? compB : payloadB,
      compB.count < payloadB.count ? 1 << 31 : 0)
    let pick = optionB.stored.count < optionA.stored.count ? optionB : optionA
    var out = Data(capacity: 20 + pick.stored.count)
    for v: UInt32 in [
      Self.chunkMagic, UInt32(live), UInt32(pick.payload.count), UInt32(pick.stored.count), pick.flags,
    ] { out.appendLE(v) }
    out.append(pick.stored)
    return out
  }

  static let lpcOrder = 8
  static let lpcShift = 14

  static func lpcResiduals(_ q: [Int64], _ d1: [Int64]) -> (coefs: [Int32], res: [Int64])? {
    let n = q.count
    let m = lpcOrder
    guard n > 4 * m, q.allSatisfy({ $0.magnitude < 1 << 24 }) else { return nil }
    var ata = [Double](repeating: 0, count: m * m)
    var atb = [Double](repeating: 0, count: m)
    for i in m..<n {
      let y = Double(q[i])
      for j in 0..<m {
        let xj = Double(q[i - 1 - j])
        atb[j] += xj * y
        for l in j..<m { ata[j * m + l] += xj * Double(q[i - 1 - l]) }
      }
    }
    for j in 0..<m {
      for l in 0..<j { ata[j * m + l] = ata[l * m + j] }
      ata[j * m + j] += 1e-6
    }
    var c = atb
    for p in 0..<m {
      let pivot = ata[p * m + p]
      guard pivot.magnitude > 1e-12 else { return nil }
      for r in (p + 1)..<m {
        let factor = ata[r * m + p] / pivot
        for l in p..<m { ata[r * m + l] -= factor * ata[p * m + l] }
        c[r] -= factor * c[p]
      }
    }
    for p in stride(from: m - 1, through: 0, by: -1) {
      var acc = c[p]
      for l in (p + 1)..<m { acc -= ata[p * m + l] * c[l] }
      c[p] = acc / ata[p * m + p]
    }
    let limit = Double(1 << 20)
    let coefs = c.map { Int32(max(-limit, min(limit, ($0 * Double(1 << lpcShift)).rounded()))) }
    var res = [Int64](repeating: 0, count: n)
    for i in 0..<m { res[i] = d1[i] }
    for i in m..<n {
      var acc: Int64 = 0
      for j in 0..<m { acc &+= Int64(coefs[j]) &* q[i - 1 - j] }
      res[i] = q[i] &- (acc >> Int64(lpcShift))
    }
    return (coefs, res)
  }

  static func zig(_ v: Int64) -> UInt64 { UInt64(bitPattern: (v << 1) ^ (v >> 63)) }

  static func varintSize(_ res: [Int64]) -> Int {
    var total = 0
    for v in res {
      var z = zig(v)
      repeat { total += 1; z >>= 7 } while z != 0
    }
    return total
  }

  static func riceSize(_ res: [Int64]) -> (k: Int, bits: Int) {
    var best = (k: 0, bits: Int.max)
    for k in 0..<20 {
      var bits = 0
      for v in res {
        let unary = Int(clamping: zig(v) >> UInt64(k))
        let (step, stepOver) = unary.addingReportingOverflow(1 + k)
        let (sum, sumOver) = bits.addingReportingOverflow(step)
        if stepOver || sumOver {
          bits = Int.max
          break
        }
        bits = sum
      }
      if bits < best.bits { best = (k, bits) }
    }
    return best
  }

  static func riceBytes(_ res: [Int64], _ k: Int) -> [UInt8] {
    var out: [UInt8] = [UInt8(k)]
    var acc: UInt64 = 0
    var nbits = 0
    func push(_ value: UInt64, _ count: Int) {
      var value = value
      var count = count
      while count > 0 {
        let take = min(count, 64 - nbits)
        acc |= (value & ((take == 64 ? 0 : 1 << UInt64(take)) &- 1)) << UInt64(nbits)
        nbits += take
        value >>= UInt64(take)
        count -= take
        while nbits >= 8 {
          out.append(UInt8(acc & 0xFF))
          acc >>= 8
          nbits -= 8
        }
      }
    }
    for v in res {
      let z = zig(v)
      var unary = Int(z >> UInt64(k))
      while unary >= 32 { push(0, 32); unary -= 32 }
      push(1 << UInt64(unary), unary + 1)
      if k > 0 { push(z & ((1 << UInt64(k)) - 1), k) }
    }
    if nbits > 0 { out.append(UInt8(acc & 0xFF)) }
    return out
  }

  typealias ColumnPlan = (pred: UInt32, coder: UInt32, bytes: [UInt8])

  static func encodeColumn(_ q: [Int64]) -> (varint: ColumnPlan, full: ColumnPlan) {
    var d1 = q
    for i in stride(from: q.count - 1, through: 1, by: -1) { d1[i] &-= d1[i - 1] }
    var d2 = d1
    for i in stride(from: d2.count - 1, through: 2, by: -1) { d2[i] &-= d2[i - 1] }
    var candidates: [(pred: UInt32, extra: [UInt8], res: [Int64])] = [(0, [], d1), (1, [], d2)]
    if let lpc = lpcResiduals(q, d1) {
      var extra: [UInt8] = []
      for c in lpc.coefs {
        withUnsafeBytes(of: c.littleEndian) { extra.append(contentsOf: $0) }
      }
      candidates.append((2, extra, lpc.res))
    }
    var bestVarint: (plan: ColumnPlan, size: Int)? = nil
    var best: (plan: ColumnPlan, size: Int)? = nil
    for cand in candidates {
      let vs = cand.extra.count + varintSize(cand.res)
      if bestVarint == nil || vs < bestVarint!.size {
        var b = cand.extra
        for v in cand.res { putVarint(&b, v) }
        if cand.pred < 2 { bestVarint = ((cand.pred, 0, b), vs) }
        if best == nil || vs < best!.size { best = ((cand.pred, 0, b), vs) }
      } else if best == nil || vs < best!.size {
        var b = cand.extra
        for v in cand.res { putVarint(&b, v) }
        best = ((cand.pred, 0, b), vs)
      }
      let rice = riceSize(cand.res)
      if rice.bits <= Int.max - 7 {
        let rs = cand.extra.count + 1 + (rice.bits + 7) / 8
        if rs < best!.size {
          best = ((cand.pred, 1, cand.extra + riceBytes(cand.res, rice.k)), rs)
        }
      }
    }
    return (bestVarint!.plan, best!.plan)
  }
}

final class PoseTapeWriter {
  static let packMagic: UInt32 = 0x504B_414F
  static let idxMagic: UInt32 = 0x494B_414F
  private let pack: FileHandle
  private let idx: FileHandle
  private let encoders: [RecordWriter]
  private(set) var bytesWritten: Int

  init(root: URL, laneEnvKinds: [UInt32], seeds: [Int]) throws {
    encoders = try laneEnvKinds.map { try RecordWriter(envKind: $0) }
    let packURL = root.appendingPathComponent("pose.pack")
    let idxURL = root.appendingPathComponent("pose.idx")
    FileManager.default.createFile(atPath: packURL.path, contents: nil)
    FileManager.default.createFile(atPath: idxURL.path, contents: nil)
    pack = try FileHandle(forWritingTo: packURL)
    idx = try FileHandle(forWritingTo: idxURL)
    var head = Data()
    for v: UInt32 in [Self.packMagic, UInt32(laneEnvKinds.count)] { head.appendLE(v) }
    for lane in 0..<laneEnvKinds.count {
      head.appendLE(laneEnvKinds[lane])
      head.appendLE(Int64(seeds[lane]))
    }
    try pack.write(contentsOf: head)
    bytesWritten = head.count
    var ih = Data()
    for v: UInt32 in [Self.idxMagic, UInt32(laneEnvKinds.count)] { ih.appendLE(v) }
    try idx.write(contentsOf: ih)
  }

  func encodeWindow(
    _ rp: UnsafePointer<Float>, capacity: Int, stepStart: Int, stepCount: Int,
    onError: (Error) -> Void
  ) throws {
    let lanes = encoders.count
    var chunks: [Data?] = Array(repeating: nil, count: lanes)
    chunks.withUnsafeMutableBufferPointer { out in
      let base = out.baseAddress!
      DispatchQueue.concurrentPerform(iterations: lanes) { slot in
        do {
          base[slot] = try encoders[slot].encodeChunk(
            rp, slot: slot, capacity: capacity, laneCount: lanes,
            stepStart: stepStart, stepCount: stepCount)
        } catch {
          onError(error)
        }
      }
    }
    var body = Data()
    var rows = Data()
    var chunkCount: UInt32 = 0
    for lane in 0..<chunks.count {
      guard let c = chunks[lane] else { continue }
      chunkCount += 1
      body.append(c)
      for v: UInt32 in [UInt32(lane), UInt32(c.count)] { rows.appendLE(v) }
    }
    var head = Data()
    for v: UInt32 in [UInt32(stepStart), chunkCount] { head.appendLE(v) }
    head.appendLE(UInt64(bytesWritten))
    head.append(rows)
    try pack.write(contentsOf: body)
    bytesWritten += body.count
    try idx.write(contentsOf: head)
  }

  func close() throws {
    try pack.close()
    try idx.close()
  }
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
  let ageOffsets: [Int]
  init(laneRanks: [Int]) {
    var offsets: [Int] = []
    var cursor = 0
    for rank in laneRanks {
      offsets.append(cursor)
      cursor += rank
    }
    ageOffsets = offsets
    prevAge = .allocate(capacity: max(cursor, 1))
    prevAge.initialize(repeating: 0, count: max(cursor, 1))
    prevBirths = .allocate(capacity: laneRanks.count)
    prevBirths.initialize(repeating: 0, count: laneRanks.count)
    hasPrev = .allocate(capacity: laneRanks.count)
    hasPrev.initialize(repeating: false, count: laneRanks.count)
  }
}

final class TapeWriter {
  static let fileMagic: UInt32 = 0x544B_414F
  static let baseFields = ["rewardSum", "rho", "deathStep", "deathDetector"]
  let fields: [String]
  private let handle: FileHandle
  private let laneCount: Int
  private let diagCount: Int
  private var block: [Float]

  init(url: URL, laneCount: Int, diagFields: [String]) throws {
    FileManager.default.createFile(atPath: url.path, contents: nil)
    self.handle = try FileHandle(forWritingTo: url)
    self.laneCount = laneCount
    self.diagCount = diagFields.count
    self.fields = Self.baseFields + diagFields
    self.block = [Float](repeating: 0, count: (Self.baseFields.count + diagCount) * laneCount)
    let json = try JSONSerialization.data(withJSONObject: ["fields": fields])
    var head = Data()
    for v: UInt32 in [Self.fileMagic, UInt32(laneCount), UInt32(json.count)] { head.appendLE(v) }
    head.append(json)
    try handle.write(contentsOf: head)
  }

  func append(snap: UnsafeMutablePointer<LaneSnapshot>, half: Int,
              deadStep: [Int?], deadDetector: [UInt32?], diag: [Float],
              stepStart: Int, stepCount: Int, wallSeconds: Double) throws {
    precondition(diag.count == diagCount * laneCount, "diag matrix shape drifted from the field list")
    var out = Data(capacity: 12 + fields.count * laneCount * 4)
    for v: UInt32 in [UInt32(stepStart), UInt32(stepCount)] { out.appendLE(v) }
    out.appendLE(Float(wallSeconds).bitPattern)
    var field = 0
    func push(_ get: (Int, LaneSnapshot) -> Float) {
      for l in 0..<laneCount { block[field * laneCount + l] = get(l, snap[half * laneCount + l]) }
      field += 1
    }
    push { _, s in s.rewardSum }
    push { _, s in s.rho }
    push { l, _ in deadStep[l].map { Float(bitPattern: UInt32($0)) } ?? Float.nan }
    push { l, _ in deadDetector[l].map { Float($0) } ?? Float.nan }
    for f in 0..<diagCount {
      push { l, _ in diag[l * diagCount + f] }
    }
    let comp = try block.withUnsafeBufferPointer {
      try (Data(buffer: $0) as NSData).compressed(using: .zlib) as Data
    }
    out.appendLE(UInt32(comp.count))
    out.append(comp)
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
  static func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "" }
    return String(cString: buf)
  }
  static func hardwareInfo(_ dev: MTLDevice) -> [String: Any] {
    [
      "gpu": dev.name,
      "machine": sysctlString("hw.model"),
      "cpuCores": ProcessInfo.processInfo.processorCount,
      "memoryBytes": ProcessInfo.processInfo.physicalMemory,
      "os": ProcessInfo.processInfo.operatingSystemVersionString,
    ]
  }
  static func hardwareKey(_ dev: MTLDevice) -> String {
    let parts = "\(sysctlString("hw.model"))|\(dev.name)|\(ProcessInfo.processInfo.processorCount)"
    return String(sha256Hex(Array(parts.utf8).withUnsafeBytes { UnsafeRawBufferPointer($0) }).prefix(16))
  }
  static let envIdsByKind: [String] = [
    "HalfCheetah-v4", "FourRooms-cont-v0", "DeepSea-cont-v0", "KeyCorridor-cont-v0",
    "Acrobot-cont-v0", "MountainCar-cont-v0", "CartPole-cont-v0", "Pendulum-cont-v0",
    "Chain-cont-v0", "LargeDeepSea-cont-v0", "MultiKeyCorridor-cont-v0", "MemoryChain-cont-v0",
    "ProceduralMaze-cont-v0", "HighDimMaze-cont-v0", "POMaze-cont-v0", "DelayedChain-cont-v0",
    "JellyBeanWorld-cont-v0",
  ]
  static func fnv1a(_ s: String) -> UInt32 {
    var h: UInt32 = 0x811c_9dc5
    for b in Array(s.utf8) { h = (h ^ UInt32(b)) &* 16_777_619 }
    return h
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
    let envStateBuf: MTLBuffer
    let envOffBuf: MTLBuffer
    let envOffsets: [Int]
    let mjModelsBuf: MTLBuffer
    let mjScratchBuf: MTLBuffer
    let mjScratchOffBuf: MTLBuffer
    let envConstantsBuf: MTLBuffer
    let ecfg: FleetEnvCfg
    init(
      _ dev: MTLDevice, lib: MTLLibrary, seeds: [Int], laneEnvKinds: [UInt32],
      envTable: [EnvSpec], envConstantWords: Int, steps: Int
    ) throws {
      func makeBuf(_ len: Int) throws -> MTLBuffer {
        guard let b = dev.makeBuffer(
          length: max(len, 16), options: [.storageModeShared])
        else {
          throw err("Metal buffer allocation failed (\(len) bytes).")
        }
        return b
      }
      B = seeds.count
      guard steps > 0, steps <= Int(UInt32.max), steps <= Int.max / 2 else {
        throw err("Declared run horizon \(steps) cannot be represented exactly.")
      }
      var persistentSlots = 2
      while persistentSlots < 2 * steps {
        guard persistentSlots <= Int(UInt32.max) / 2 else {
          throw err("Declared run horizon \(steps) exceeds persistent coordinate capacity.")
        }
        persistentSlots *= 2
      }
      windowSteps = 1024
      maxChunkSteps = windowSteps / 4
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
      var diagFloats = 0
      let ppoP = Diag.ppoParams(
        obsMax: envTable.map(\.naturalO).max()!, actMax: envTable.map(\.actionCount).max()!)
      for kind in laneEnvKinds {
        diagFloats += max(Diag.oakFloats(Diag.oakRank(envTable[Int(kind)])), Diag.ppoFloats(ppoP))
      }
      diagOutBuf = try makeBuf(2 * diagFloats * 4)
      memset(diagAccumBuf.contents(), 0, diagAccumBuf.length)
      memset(diagOutBuf.contents(), 0, diagOutBuf.length)
      var offsets: [Int] = []
      var envCursor = 0
      for kind in laneEnvKinds {
        let spec = envTable[Int(kind)]
        offsets.append(envCursor)
        let fixed = (spec.actionCount + spec.envStateWords + spec.naturalO + 3) / 4 * 4
        let (persistent, persistentOverflow) = spec.persistentEntryWords.multipliedReportingOverflow(
          by: persistentSlots)
        let (next, cursorOverflow) = envCursor.addingReportingOverflow(fixed + persistent)
        guard !persistentOverflow, !cursorOverflow else {
          throw err("Declared run horizon \(steps) overflows the environment arena.")
        }
        envCursor = next
      }
      envOffsets = offsets
      envStateBuf = try makeBuf(envCursor * 4)
      memset(envStateBuf.contents(), 0, envStateBuf.length)
      envOffBuf = try makeBuf(B * 4)
      do {
        let op = envOffBuf.contents().bindMemory(to: UInt32.self, capacity: B)
        for i in 0..<B { op[i] = UInt32(offsets[i]) }
      }
      let blob = try MjBlob.load()
      var mjEntryByKind: [Int: MjEntry] = [:]
      for (kind, spec) in envTable.enumerated() where spec.mjIndex >= 0 {
        let envId = Main.envIdsByKind[kind]
        guard spec.mjIndex < blob.entries.count,
          blob.entries[spec.mjIndex].nameHash == Main.fnv1a(envId)
        else { throw err("\(envId): mjIndex \(spec.mjIndex) does not match mjmodels.bin.") }
        let entry = blob.entries[spec.mjIndex]
        guard entry.stateWords == spec.envStateWords else {
          throw err(
            "\(envId): blob state words \(entry.stateWords) != spec envStateWords \(spec.envStateWords)."
          )
        }
        mjEntryByKind[kind] = entry
      }
      let modelsBuf = try makeBuf(blob.data.count)
      blob.data.withUnsafeBytes { raw in
        modelsBuf.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
      }
      mjModelsBuf = modelsBuf
      var scratchOffsets = [UInt32](repeating: 0, count: B)
      var scratchCursor = 0
      for i in 0..<B {
        if let entry = mjEntryByKind[Int(laneEnvKinds[i])] {
          scratchOffsets[i] = UInt32(scratchCursor)
          scratchCursor += entry.scratchWords
        }
      }
      mjScratchBuf = try makeBuf(max(scratchCursor, 1) * 4)
      mjScratchOffBuf = try makeBuf(B * 4)
      do {
        let sp = mjScratchOffBuf.contents().bindMemory(to: UInt32.self, capacity: B)
        for i in 0..<B { sp[i] = scratchOffsets[i] }
      }
      let constants = try makeBuf(envConstantWords * 4)
      guard let fn = lib.makeFunction(name: "oak_env_constants_init"),
        let queue = dev.makeCommandQueue(), let cb = queue.makeCommandBuffer(),
        let enc = cb.makeComputeCommandEncoder()
      else { throw err("Environment constants initialization failed.") }
      let pipe = try dev.makeComputePipelineState(function: fn)
      enc.setComputePipelineState(pipe)
      enc.setBuffer(constants, offset: 0, index: 0)
      enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
      enc.endEncoding()
      cb.commit()
      cb.waitUntilCompleted()
      guard cb.status == .completed else {
        let message = cb.error?.localizedDescription ?? "unknown Metal error"
        throw err("Environment constants initialization failed: \(message).")
      }
      envConstantsBuf = constants
      ecfg = FleetEnvCfg(
        laneCount: UInt32(B), recordCapacity: UInt32(recordCapacity),
        persistentMask: UInt32(persistentSlots - 1),
        invObsFloor: 1 / Float(64))
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
    let core = snapPipe.threadExecutionWidth
    guard core == Engine.simdWidth else {
      throw err("simd width \(core) is not the \(Engine.simdWidth) the kernel is compiled for.")
    }
    let megaG = [32, 16, 8, 4, 2, 1].first {
      candidate in engine.kinds.allSatisfy {
        $0.count % candidate == 0 && $0.firstLane % candidate == 0
      }
    }!
    let stageMask = try stageMaskEnv(e)
    func baseConstants() -> MTLFunctionConstantValues {
      let c = MTLFunctionConstantValues()
      ablations.apply(to: c)
      var megaG32 = UInt32(megaG)
      c.setConstantValue(&megaG32, type: .uint, index: OakConst.megaG.rawValue)
      if e["OAK_PROBE_FIXEDIDX"] == "1" {
        var b = true
        c.setConstantValue(&b, type: .bool, index: OakConst.probeFixedIdx.rawValue)
      }
      if let mask = stageMask {
        var m = mask
        c.setConstantValue(&m, type: .uint, index: OakConst.megaStageMask.rawValue)
      }
      return c
    }
    var envRuns: [(kindIdx: Int, kind: UInt32, tgStart: Int, tgCount: Int)] = []
    var variantPipes: [MTLComputePipelineState] = []
    var kindMegaG: [Int] = []
    let tgOptionCount = Int(engine.kinds[0].off.optionCount)
    for (ki, kd) in engine.kinds.enumerated() {
      var gk = megaG
      while gk > 1, gk * Int(kd.off.anchorCount) > core * core { gk /= 2 }
      guard kd.firstLane % gk == 0, kd.count % gk == 0 else {
        throw err(
          "Env block [\(kd.firstLane), +\(kd.count)) must start and size on multiples of megaG=\(gk)."
        )
      }
      let c = baseConstants()
      Engine.applyShapeConstants(c, kd.cfg)
      var k32 = kd.kind
      c.setConstantValue(&k32, type: .uint, index: OakConst.specEnvKind.rawValue)
      var g32 = UInt32(gk)
      c.setConstantValue(&g32, type: .uint, index: OakConst.megaG.rawValue)
      let desc = MTLComputePipelineDescriptor()
      desc.computeFunction = try lib.makeFunction(name: "oak_megastep", constantValues: c)
      desc.label = "oak_megastep \(Main.envIdsByKind[Int(kd.kind)])"
      variantPipes.append(try dev.makeComputePipelineState(descriptor: desc, options: []).0)
      kindMegaG.append(gk)
      envRuns.append((ki, kd.kind, kd.firstLane / gk, (kd.count + gk - 1) / gk))
    }
    guard variantPipes.allSatisfy({ $0.threadExecutionWidth == core }) else {
      throw err("simd width \(core) differs across specialized pipelines.")
    }
    func laneSlotBytes(_ gk: Int) -> Int {
      (5 * tgOptionCount + 1) * gk * MemoryLayout<Float>.stride
    }
    for (ki, pipe) in variantPipes.enumerated() {
      let gk = kindMegaG[ki]
      let slotBytes = laneSlotBytes(gk)
      guard gk >= 1, core * gk <= pipe.maxTotalThreadsPerThreadgroup,
        slotBytes <= dev.maxThreadgroupMemoryLength
      else {
        throw err(
          "megaG=\(gk) needs \(core * gk) threads/threadgroup (pipeline max \(pipe.maxTotalThreadsPerThreadgroup)) and \(slotBytes) tg bytes (device max \(dev.maxThreadgroupMemoryLength))."
        )
      }
    }
    let res = engine.res
    let ecfg = bufs.ecfg
    func bindFleet(_ enc: MTLComputeCommandEncoder, _ pipe: MTLComputePipelineState) {
      enc.setComputePipelineState(pipe)
      for j in res.indices { enc.setBuffer(res[j], offset: 0, index: j) }
      enc.setBuffer(bufs.laneBuf, offset: 0, index: OakBuf.lanes.rawValue)
      enc.setBuffer(bufs.seedsBuf, offset: 0, index: OakBuf.seeds.rawValue)
      enc.setBuffer(bufs.envStateBuf, offset: 0, index: 14)
      enc.setBuffer(bufs.envOffBuf, offset: 0, index: 15)
      enc.setBuffer(bufs.mjModelsBuf, offset: 0, index: 16)
      enc.setBuffer(bufs.mjScratchBuf, offset: 0, index: 17)
      enc.setBuffer(bufs.mjScratchOffBuf, offset: 0, index: 18)
      enc.setBuffer(bufs.envConstantsBuf, offset: 0, index: 19)
      var c = ecfg
      enc.setBytes(&c, length: MemoryLayout<FleetEnvCfg>.stride, index: OakBuf.envCfg.rawValue)
    }
    let laneTG = mtlSize(
      min(64, initPipe.maxTotalThreadsPerThreadgroup, snapPipe.maxTotalThreadsPerThreadgroup))
    let laneRanks = engine.laneKindIdx.map { Int(engine.kinds[$0].off.rank) }
    let diagHost = OakDiagHost(laneRanks: laneRanks)
    var diagBase: [Int] = []
    var diagHalf = 0
    for kd in engine.kinds {
      diagBase.append(diagHalf)
      diagHalf += kd.count * Diag.oakFloats(Int(kd.off.rank))
    }
    precondition(2 * diagHalf * 4 <= bufs.diagOutBuf.length, "diagOut arena undersized")
    let diagPtr = bufs.diagOutBuf.contents().bindMemory(to: Float.self, capacity: 2 * diagHalf)
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
          let gk = kindMegaG[run.kindIdx]
          enc.setThreadgroupMemoryLength(laneSlotBytes(gk), index: 0)
          enc.dispatchThreadgroups(
            MTLSize(width: run.tgCount, height: 1, depth: 1),
            threadsPerThreadgroup: mtlSize(core * gk))
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
          let ki = engine.kinds.firstIndex { $0.kind == kd.kind }!
          var ctl = SIMD4<UInt32>(
            UInt32(half * B), UInt32(kd.firstLane),
            UInt32(half * diagHalf + diagBase[ki]),
            UInt32(Diag.oakFloats(Int(kd.off.rank))))
          enc.setBytes(&ctl, length: MemoryLayout<SIMD4<UInt32>>.stride, index: OakBuf.aux1.rawValue)
          enc.setBytes(&fullScan, length: 4, index: OakBuf.aux2.rawValue)
          enc.setBuffer(bufs.diagAccumBuf, offset: 0, index: OakBuf.diagAccum.rawValue)
          enc.setBuffer(bufs.diagOutBuf, offset: 0, index: OakBuf.diagOut.rawValue)
          enc.dispatchThreadgroups(mtlSize(kd.count), threadsPerThreadgroup: mtlSize(core))
        }
      },
      laneHashes: { lane, laneBytes in
        let spec = engine.envTable[Int(engine.kind(of: lane).kind)]
        let envState = bufs.envStateBuf.contents().bindMemory(
          to: UInt32.self, capacity: bufs.envStateBuf.length / 4) + bufs.envOffsets[lane]
        let laneWords = laneBytes.bindMemory(to: UInt32.self, capacity: fleetLaneStride / 4)
        let isCheetah = engine.kind(of: lane).kind == 0
        var canonical = [UInt32](repeating: 0, count: 60)
        if isCheetah {
          for i in 0..<33 { canonical[i] = envState[spec.actionCount + i] }
        }
        canonical[33] = laneWords[0]
        canonical[34] = laneWords[1]
        canonical[35] = laneWords[2]
        for d in 0..<spec.actionCount { canonical[36 + d] = envState[d] }
        for i in 0..<5 { canonical[42 + i] = laneWords[3 + i] }
        canonical[47] = laneWords[8]
        if !isCheetah {
          for i in 0..<10 { canonical[48 + i] = envState[spec.actionCount + i] }
        }
        canonical[58] = laneWords[9]
        canonical[59] = laneWords[10]
        return canonical.withUnsafeBytes { raw in
          engine.stateHashes(lane, laneBytes: raw.baseAddress!, laneByteCount: raw.count)
        }
      },
      diagFields: [
        "dormantFraction", "utilityDormantFraction", "featureBirths",
        "weightAbsMean", "weightAbsMeanValue", "erank",
        "logAlphaValueP10", "logAlphaValueP50", "logAlphaValueP90",
        "logAlphaOptionValueP50", "logAlphaModelP50",
      ],
      diagValues: { half, lane, stepCount in
        let ki = engine.laneKindIdx[lane]
        let kd = engine.kinds[ki]
        let rank = Int(kd.off.rank)
        let floats = Diag.oakFloats(rank)
        let rec = diagPtr + half * diagHalf + diagBase[ki] + (lane - kd.firstLane) * floats
        let prevAge = diagHost.prevAge + diagHost.ageOffsets[lane]
        var ages = [UInt32](repeating: 0, count: rank)
        for i in 0..<rank { ages[i] = rec[i].bitPattern }
        var dormantFraction = Float.nan
        if diagHost.hasPrev[lane] {
          var delta = [Double](repeating: .nan, count: rank)
          var sum = 0.0
          var validCount = 0
          for i in 0..<rank {
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
            for i in 0..<rank where delta[i].isFinite && delta[i] / mean < DiagMath.dormantTau {
              dormant += 1
            }
            dormantFraction = Float(dormant) / Float(rank)
          }
        }
        let utility = (0..<rank).map { Double(abs(rec[rank + $0])) }
        let births = rec[5 * rank]
        let birthsPerStep: Float = diagHost.hasPrev[lane] && stepCount > 0
          ? (births - diagHost.prevBirths[lane]) / Float(stepCount) : .nan
        func alphas(_ off: Int) -> [Double] { (0..<rank).map { Double(rec[off + $0]) } }
        let valueQ = DiagMath.logQuantiles(alphas(2 * rank), [0.1, 0.5, 0.9])
        let optionQ = DiagMath.logQuantiles(alphas(3 * rank), [0.5])
        let modelQ = DiagMath.logQuantiles(alphas(4 * rank), [0.5])
        for i in 0..<rank { prevAge[i] = ages[i] }
        diagHost.prevBirths[lane] = births
        diagHost.hasPrev[lane] = true
        return [
          dormantFraction, DiagMath.dormantFraction(utility), birthsPerStep,
          rec[5 * rank + 1], rec[5 * rank + 2],
          DiagMath.erank(rec + 5 * rank + 3),
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
    let envProbe = try Engine.runEnvTableProbe(dev, lib)
    let envTable = envProbe.specs
    let engine: Engine?
    let q: MTLCommandQueue
    if policy == "oak" {
      let eng = try Engine(
        laneEnvs: laneEnvs, batch: seeds.count, dev: dev, lib: lib, envTable: envTable)
      engine = eng
      q = eng.q
    } else {
      engine = nil
      guard let queue = dev.makeCommandQueue() else { throw err("Metal unavailable.") }
      q = queue
    }
    do {
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
    Self.pendingMeta["hardware"] = Self.hardwareInfo(dev)
    Self.pendingMeta["hardwareKey"] = Self.hardwareKey(dev)
    try Self.writeRunMeta(root, envTable)
    for flag in ["OAK_PROBE_FIXEDIDX", "OAK_MEGA_STAGE_MASK"] where e[flag] != nil {
      fputs("\(flag)=\(e[flag]!): timing instrument, trajectories are garbage\n", stderr)
    }
    let bufs = try FleetBuffers(
      dev, lib: lib, seeds: seeds, laneEnvKinds: laneEnvKinds, envTable: envTable,
      envConstantWords: envProbe.constantWords, steps: steps)
    let pipes: FleetPipes
    if let engine {
      pipes = try oakPipes(engine, lib, dev, bufs, ablations, e)
    } else {
      pipes = try baselinePipes(policy, lib, dev, bufs, envTable, laneEnvKinds, e)
    }
    let gpuSeconds = try runFleetLoop(
      q, bufs, pipes, seeds: seeds, steps: steps, root: root,
      laneEnvKinds: laneEnvKinds)
    Self.pendingMeta["gpuSeconds"] = gpuSeconds
    Self.pendingMeta["stepsPerSecond"] = gpuSeconds > 0
      ? Double(steps) * Double(seeds.count) / gpuSeconds : 0
    Self.pendingMeta["finishedAt"] = ISO8601DateFormatter().string(from: Date())
    try Self.writeRunMeta(root, envTable)
  }
  static let ppoLR = 3e-4
  static let ppoAdamB2 = 0.999
  static let ppoAdamEps = 1e-5
  static func stageMaskEnv(_ e: [String: String]) throws -> UInt32? {
    guard let raw = e["OAK_MEGA_STAGE_MASK"] else { return nil }
    guard let v = UInt32(raw.trimmingCharacters(in: .whitespaces)) else {
      throw err("OAK_MEGA_STAGE_MASK='\(raw)' is not an unsigned decimal integer.")
    }
    return v
  }
  static func baselinePipes(
    _ policy: String, _ lib: MTLLibrary, _ dev: MTLDevice, _ bufs: FleetBuffers,
    _ envTable: [EnvSpec], _ laneEnvKinds: [UInt32], _ e: [String: String]
  ) throws -> FleetPipes {
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let constants = MTLFunctionConstantValues()
      if let mask = try stageMaskEnv(e) {
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
    let obsMax = envTable.map(\.naturalO).max()!
    let actMax = envTable.map(\.actionCount).max()!
    let ppoH = Diag.k
    let ppoT = 128
    let ppoMB = 32
    let ppoParamStride = Diag.ppoParams(obsMax: obsMax, actMax: actMax)
    let ppoRollStride = ((ppoT + 1) * obsMax + ppoT * actMax + 6 * ppoT + 15) & ~15
    let ppoScratchStride =
      ((ppoMB * obsMax + 4 * ppoMB * ppoH + 2 * ppoMB * actMax + ppoMB + ppoT
        + ppoParamStride + 15) & ~15) + 16
    let ppoStateStride = (2 * obsMax + 6 + 15) & ~15
    let ppoFloats = Diag.ppoFloats(ppoParamStride)
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
      enc.setBuffer(bufs.envStateBuf, offset: 0, index: 14)
      enc.setBuffer(bufs.envOffBuf, offset: 0, index: 15)
      enc.setBuffer(bufs.mjModelsBuf, offset: 0, index: 16)
      enc.setBuffer(bufs.mjScratchBuf, offset: 0, index: 17)
      enc.setBuffer(bufs.mjScratchOffBuf, offset: 0, index: 18)
      enc.setBuffer(bufs.envConstantsBuf, offset: 0, index: 19)
      var c = ecfg
      enc.setBytes(&c, length: MemoryLayout<FleetEnvCfg>.stride, index: OakBuf.envCfg.rawValue)
    }
    let laneTG = mtlSize(min(
      64, initPipe.maxTotalThreadsPerThreadgroup, stepPipe.maxTotalThreadsPerThreadgroup,
      snapPipe.maxTotalThreadsPerThreadgroup))
    let stepThreads = B * 32
    let stepTG = mtlSize(32)
    let diagPtr = bufs.diagOutBuf.contents().bindMemory(
      to: Float.self, capacity: 2 * B * ppoFloats)
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
        let rec = diagPtr + (half * B + lane) * ppoFloats
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
          rec[Diag.ppoWeightAbsMean], DiagMath.erank(rec + Diag.ppoGram(ppoParamStride)),
          stepQ[0], stepQ[1], stepQ[2],
        ]
      }
    )
  }
  static func runFleetLoop(
    _ q: MTLCommandQueue, _ bufs: FleetBuffers, _ pipes: FleetPipes,
    seeds: [Int], steps: Int, root: URL, laneEnvKinds: [UInt32]
  ) throws -> Double {
    let e = ProcessInfo.processInfo.environment
    let fm = FileManager.default
    let B = bufs.B
    let dev = q.device
    let poseTape = try PoseTapeWriter(root: root, laneEnvKinds: laneEnvKinds, seeds: seeds)
    let maxChunkSteps = bufs.maxChunkSteps
    let windowSteps = bufs.windowSteps
    let recordCapacity = bufs.recordCapacity
    let staging = UnsafeMutablePointer<Float>.allocate(capacity: windowSteps * recordFloats * B)
    let snapPtr = bufs.snapBuf.contents().bindMemory(to: LaneSnapshot.self, capacity: 2 * B)
    let dumpEvery = max(e["OAK_DUMP_STATE_EVERY"].flatMap(Int.init) ?? 0, 0)
    var hashHandle: FileHandle? = nil
    if dumpEvery > 0 {
      let url = root.appendingPathComponent("state-hashes.jsonl")
      fm.createFile(atPath: url.path, contents: nil)
      hashHandle = try FileHandle(forWritingTo: url)
    }
    let tape = try TapeWriter(
      url: root.appendingPathComponent("tape.bin"), laneCount: B, diagFields: pipes.diagFields)
    defer {
      try? hashHandle?.close()
      try? poseTape.close()
      staging.deallocate()
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
      var error: Error? {
        get { errLock.lock(); defer { errLock.unlock() }; return failure }
        set {
          errLock.lock()
          defer { errLock.unlock() }
          if failure == nil { failure = newValue }
        }
      }
      var didFail: Bool { errLock.lock(); defer { errLock.unlock() }; return failure != nil }
      init(_ b: Int) {
        deadStep = Array(repeating: nil, count: b)
        deadDetector = Array(repeating: nil, count: b)
      }
    }
    let st = WriterState(B)
    let writer = DispatchQueue(label: "oak.fleet.gpu.writer")
    let wallStart = DispatchTime.now().uptimeNanoseconds
    func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1e9 }
    func stageRecords(stepStart: Int, stepCount: Int) {
      let rp = bufs.recordBuf.contents().bindMemory(
        to: Float.self, capacity: B * recordCapacity * recordFloats)
      let rowStride = recordFloats * B
      for r in stepStart..<(stepStart + stepCount) {
        memcpy(
          staging + (r % windowSteps) * rowStride,
          rp + (r % recordCapacity) * rowStride,
          rowStride * 4)
      }
    }
    func drainRecords(step: Int) throws {
      guard step > st.lastDrainStep else { return }
      let t0 = DispatchTime.now().uptimeNanoseconds
      let start = st.lastDrainStep
      let count = step - start
      let before = timing ? poseTape.bytesWritten : 0
      try poseTape.encodeWindow(
        UnsafePointer(staging), capacity: windowSteps, stepStart: start, stepCount: count
      ) { st.error = $0 }
      if st.didFail, let e = st.error { throw e }
      st.lastDrainStep = step
      if timing {
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let wrote = poseTape.bytesWritten - before
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
      guard let h = hashHandle else { return }
      for lane in 0..<B {
        let laneBytes = bufs.laneBuf.contents() + lane * fleetLaneStride
        let hs = pipes.laneHashes(lane, laneBytes)
        var obj: [String: Any] = ["seed": seeds[lane], "step": step]
        for (k, v) in hs { obj[k] = v }
        try h.write(contentsOf: jsonLine(obj))
      }
    }
    let explicitN = e["OAK_STEPS_PER_CB"].flatMap(Int.init)
    if let n = explicitN, n > maxChunkSteps {
      throw err("OAK_STEPS_PER_CB \(n) exceeds the record ring span \(maxChunkSteps).")
    }
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
          let fit = max(1, min(Int(0.016 / (gpuT / Double(nSteps))), maxChunkSteps))
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
          if st.error == nil {
            stageRecords(stepStart: tStep - nSteps, stepCount: nSteps)
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
      throw err("Usage: OaK compare-hashes <left.jsonl> <right.jsonl>")
    }
    func asInt(_ v: Any?) -> Int? {
      if let i = v as? Int { return i }
      if let n = v as? NSNumber { return n.intValue }
      return nil
    }
    func load(_ path: String) throws -> [String: [String: String]] {
      let url = URL(fileURLWithPath: path)
      var out: [String: [String: String]] = [:]
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
      !(try fm.contentsOfDirectory(atPath: dir.path).filter {
        $0 != ".DS_Store" && $0 != "identity.json"
      }).isEmpty
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
