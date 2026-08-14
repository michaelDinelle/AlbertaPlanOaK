import CryptoKit
import Darwin
import Foundation
import Metal
import zlib

struct OakError: Error, CustomStringConvertible { let description: String }
func err(_ s: String) -> OakError { OakError(description: s) }
let defaultEnv = "HalfCheetah-v4"
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
  var rho: Float = 0
}

struct FleetEnvCfg {
  var laneCount, episodeLen, recordCapacity: UInt32
  var invObsFloor: Float
}
struct LaneSnapshot {
  var scalars: KernelScalars
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
    "option-models", "option-keyboard",
    "keyboard-to-policy", "search-control", "search-to-generate-test", "generate-and-test",
    "joint-geometry",
  ]
  static let functionConstantIndex = [
    "option-keyboard": 6,
    "keyboard-to-policy": 8,
    "search-control": 9,
    "search-to-generate-test": 10,
    "generate-and-test": 11,
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

struct ShapeProbe {
  var rank, optionCount, nursery, traceLevels, anchorCount: UInt32
  var paramStride, stateStride, uintStride: UInt32
  var pPsi, pOptPolicy, pVStop2: UInt32
  var stObs, stPolicyMean, stOptionPolicyMean, stUtilOpt: UInt32
  var stAlpha, stBeta, stAlphaPsi, stBetaPsi, stAlphaModel, stBetaModel: UInt32
  var stAlphaAct, stBetaAct: UInt32
  var uOptFeature, uFeatureAge, uParentA, uParentB, uKind, uNurParentA: UInt32
  var invRank, invOptionCount, metaEta, sampleSigma, sampleInv2S2, actorAlphaDenom: Float
}
struct RewardSlices {
  let state: [(Int, Int)]
  let uints: [(Int, Int)]
  init(_ sp: ShapeProbe, o: Int, d: Int) {
    let R = Int(sp.rank), K = Int(sp.optionCount), Rn = Int(sp.nursery)
    let T = Int(sp.traceLevels), O = o, D = d
    var s = 0
    var keepS: [(Int, Int)] = []
    func keep(_ n: Int) -> Int {
      let a = s
      s += n
      keepS.append((a, n))
      return a
    }
    func drop(_ n: Int) -> Int {
      let a = s
      s += n
      return a
    }
    let stObs = keep(O)
    _ = keep(O)
    _ = keep(O)
    _ = keep(O)
    _ = keep(O)
    _ = keep(O)
    _ = keep(O)
    _ = keep(2 * R)
    let stPolicyMean = keep(D)
    _ = keep(D)
    _ = keep(D)
    _ = keep(K * D)
    let stOptionPolicyMean = keep(K * D)
    _ = keep(K)
    _ = keep(1)
    _ = keep(1)
    _ = keep(K)
    _ = keep(R)
    let stUtilOpt = keep(K)
    _ = keep(R)
    _ = keep(R)
    _ = keep(R)
    let stAlpha = keep(R)
    let stBeta = keep(R)
    _ = keep(R)
    _ = keep(R)
    let stAlphaPsi = keep(R)
    let stBetaPsi = keep(R)
    _ = keep(R)
    _ = keep(R)
    let stAlphaModel = keep(R)
    let stBetaModel = keep(R)
    _ = keep(R)
    _ = keep(R)
    _ = keep(K)
    _ = keep(K)
    _ = keep(K)
    _ = keep(T * O)
    _ = keep(K)
    _ = keep(K)
    _ = keep(K)
    _ = keep(1)
    _ = keep(K)
    _ = keep(1)
    _ = keep(1)
    _ = keep(K)
    _ = keep(K)
    _ = keep(K)
    _ = keep(K)
    _ = keep(K)
    let stAlphaAct = keep(K)
    let stBetaAct = keep(K)
    _ = keep(K)
    _ = keep(K)
    _ = keep(K)
    _ = keep(K * D)
    _ = drop(8)
    _ = drop(K)
    _ = drop(K)
    _ = drop(K)
    _ = drop(K)
    _ = drop(K)
    _ = drop(K)
    _ = drop(K)
    _ = drop(K)
    _ = drop(1)
    s = (s + 3) & ~3
    _ = keep(4 * Rn)
    _ = drop(K * D)
    _ = drop(K * max(D / 2, 1))
    _ = drop(R)
    s = (s + 3) & ~3
    _ = drop(4 * R)
    _ = drop(16)
    _ = drop(4)
    var u = 0
    var keepU: [(Int, Int)] = []
    func ukeep(_ n: Int) -> Int {
      let a = u
      u += n
      keepU.append((a, n))
      return a
    }
    let uOptFeature = ukeep(K)
    let uFeatureAge = ukeep(R)
    _ = ukeep(K)
    let uParentA = ukeep(R)
    let uParentB = ukeep(R)
    let uKind = ukeep(R)
    _ = ukeep(R)
    _ = ukeep(1)
    _ = ukeep(K)
    let uNurParentA = ukeep(Rn)
    _ = ukeep(Rn)
    _ = ukeep(Rn)
    _ = ukeep(Rn)
    _ = ukeep(2 * 96)
    _ = ukeep(4)
    _ = ukeep(K)
    precondition(stObs == Int(sp.stObs), "stObs drift")
    precondition(stPolicyMean == Int(sp.stPolicyMean), "stPolicyMean drift")
    precondition(stOptionPolicyMean == Int(sp.stOptionPolicyMean), "stOptionPolicyMean drift")
    precondition(stUtilOpt == Int(sp.stUtilOpt), "stUtilOpt drift")
    precondition(stAlpha == Int(sp.stAlpha), "stAlpha drift")
    precondition(stBeta == Int(sp.stBeta), "stBeta drift")
    precondition(stAlphaPsi == Int(sp.stAlphaPsi), "stAlphaPsi drift")
    precondition(stBetaPsi == Int(sp.stBetaPsi), "stBetaPsi drift")
    precondition(stAlphaModel == Int(sp.stAlphaModel), "stAlphaModel drift")
    precondition(stBetaModel == Int(sp.stBetaModel), "stBetaModel drift")
    precondition(stAlphaAct == Int(sp.stAlphaAct), "stAlphaAct drift")
    precondition(stBetaAct == Int(sp.stBetaAct), "stBetaAct drift")
    precondition(uOptFeature == Int(sp.uOptFeature), "uOptFeature drift")
    precondition(uFeatureAge == Int(sp.uFeatureAge), "uFeatureAge drift")
    precondition(uParentA == Int(sp.uParentA), "uParentA drift")
    precondition(uParentB == Int(sp.uParentB), "uParentB drift")
    precondition(uKind == Int(sp.uKind), "uKind drift")
    precondition(uNurParentA == Int(sp.uNurParentA), "uNurParentA drift")
    func merge(_ xs: [(Int, Int)]) -> [(Int, Int)] {
      let sorted = xs.sorted { $0.0 < $1.0 }
      var out: [(Int, Int)] = []
      for (a, n) in sorted {
        if let last = out.last, last.0 + last.1 == a {
          out[out.count - 1] = (last.0, last.1 + n)
        } else {
          out.append((a, n))
        }
      }
      return out
    }
    state = merge(keepS)
    uints = merge(keepU)
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
      MemoryLayout<ShapeProbe>.stride == 140, "ShapeProbe layout drifted from OakShapeProbe")
    let constants = MTLFunctionConstantValues()
    var o32 = UInt32(obsDim)
    constants.setConstantValue(&o32, type: .uint, index: 13)
    var d32 = UInt32(dPolicy)
    constants.setConstantValue(&d32, type: .uint, index: 16)
    let fn = try lib.makeFunction(name: "oak_shape_probe", constantValues: constants)
    let pipe = try dev.makeComputePipelineState(function: fn)
    guard let outB = dev.makeBuffer(length: 140, options: [.storageModeShared]),
      let dummy = dev.makeBuffer(length: 16, options: [.storageModeShared]),
      let pq = dev.makeCommandQueue(), let cb = pq.makeCommandBuffer(),
      let enc = cb.makeComputeCommandEncoder()
    else { throw err("Shape probe dispatch failed.") }
    enc.setComputePipelineState(pipe)
    enc.setBuffer(outB, offset: 0, index: 0)
    for j in 1...4 { enc.setBuffer(dummy, offset: 0, index: j) }
    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return outB.contents().bindMemory(to: ShapeProbe.self, capacity: 1).pointee
  }
  static func runEnvTableProbe(_ dev: MTLDevice, _ lib: MTLLibrary)
    throws -> [(naturalO: Int, actionKind: Int, actionCount: Int)]
  {
    let fn = lib.makeFunction(name: "oak_env_table_probe")!
    let pipe = try dev.makeComputePipelineState(function: fn)
    let words = 1 + Main.envIdsByKind.count * 4
    guard let outB = dev.makeBuffer(length: words * 4, options: [.storageModeShared]),
      let pq = dev.makeCommandQueue(), let cb = pq.makeCommandBuffer(),
      let enc = cb.makeComputeCommandEncoder()
    else { throw err("Env table probe dispatch failed.") }
    enc.setComputePipelineState(pipe)
    enc.setBuffer(outB, offset: 0, index: 0)
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
      (Int(tbl[1 + i * 4 + 1]), Int(tbl[1 + i * 4 + 2]), Int(tbl[1 + i * 4 + 3]))
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
      precondition(Int(sp.optionCount) == 32, "K must equal the simd width")
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
      _ = RewardSlices(sp, o: naturalO, d: dk)
      built.append(kd)
      for l in r.firstLane..<(r.firstLane + r.count) { laneIdx[l] = built.count - 1 }
      pBase += r.count * kd.paramTotal
      sBase += r.count * kd.stateTotal
      uBase += r.count * kd.uintTotal
    }
    kinds = built
    laneKindIdx = laneIdx
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
    let slices = RewardSlices(kd.off, o: kd.naturalO, d: kd.d)
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


enum RecordEnc: String { case u8, i16, dx32 }

struct RecordField {
  let idx: Int
  let enc: RecordEnc
  let scale: Double
  let wrap: Bool
  init(_ idx: Int, _ enc: RecordEnc, _ scale: Double = 0, _ wrap: Bool = false) {
    self.idx = idx
    self.enc = enc
    self.scale = scale
    self.wrap = wrap
  }
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

struct RecordCodec {
  let category: String
  let ctrlBox: Int
  let fields: [RecordField]
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
  static func table(_ envKind: UInt32) throws -> RecordCodec {
    func grid(_ idxs: [Int], _ consts: [Int: Double] = [:]) -> RecordCodec {
      RecordCodec(
        category: "grid", ctrlBox: 0, fields: idxs.map { RecordField($0, .u8) },
        poseConst: consts)
    }
    switch envKind {
    case 0:
      var f = [RecordField(0, .dx32, 0.01), RecordField(1, .i16, 1e-4)]
      for i in 2...8 { f.append(RecordField(i, .i16, 1e-4, true)) }
      return RecordCodec(category: "cheetah", ctrlBox: 6, fields: f, poseConst: [:])
    case 1: return grid([0, 1, 2, 3])
    case 2: return grid([0, 1], [2: 7, 3: 7])
    case 3: return grid([0, 1, 2, 3, 4, 5, 6, 7])
    case 4:
      return RecordCodec(
        category: "classic", ctrlBox: 0,
        fields: [
          RecordField(0, .i16, 1e-4, true), RecordField(1, .i16, 1e-4, true),
          RecordField(2, .i16, 1e-3), RecordField(3, .i16, 1e-3),
        ], poseConst: [:])
    case 5:
      return RecordCodec(
        category: "classic", ctrlBox: 0,
        fields: [RecordField(0, .i16, 1e-4), RecordField(1, .i16, 1e-5)], poseConst: [2: 0.5])
    case 6:
      return RecordCodec(
        category: "classic", ctrlBox: 0,
        fields: [
          RecordField(0, .i16, 1e-4), RecordField(1, .i16, 1e-3),
          RecordField(2, .i16, 1e-4, true), RecordField(3, .i16, 1e-3),
        ], poseConst: [:])
    case 7:
      return RecordCodec(
        category: "classic", ctrlBox: 1,
        fields: [
          RecordField(0, .i16, 1e-4, true), RecordField(1, .i16, 1e-3), RecordField(2, .i16, 1e-3),
        ], poseConst: [:])
    case 8: return grid([0], [1: 15])
    case 9: return grid([0, 1], [2: 15, 3: 15])
    case 10: return grid([0, 1, 2, 3, 4, 5, 6, 7])
    case 11: return grid([0, 2], [1: 31])
    case 12: return grid([0, 1, 2, 3])
    case 13: return grid([0, 1, 2, 3, 4])
    case 14: return grid([0, 1, 2, 3])
    case 15: return grid([0, 2], [1: 31])
    default: throw err("No record codec for envKind \(envKind).")
    }
  }
}

final class RecordWriter: @unchecked Sendable {
  static let fileMagic: UInt32 = 0x344B_414F
  static let chunkMagic: UInt32 = 0x344B_4843
  static let codecVersion = 1

  let handle: FileHandle
  let codec: RecordCodec
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
    self.codec = try RecordCodec.table(envKind)
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

  func encodeChunk(
    _ rp: UnsafePointer<Float>, slot: Int, capacity: Int, stepStart: Int, stepCount: Int,
    rewardDelta: Float
  ) throws {
    guard stepCount > 0 else { return }
    let nBox = codec.ctrlBox
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
              esc = true
            }
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

  var bytesWritten: Int { fileOffset }
}

final class MetricsTapeWriter {
  static let fileMagic: UInt32 = 0x544B_414F
  static let segMagic: UInt32 = 0x54_4745_53
  static let fields: [String] = [
    "rewardSum", "doneCount", "rewardRate", "kernelRho", "deltaSum",
    "deltaPeak", "paramAbsMax",
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
    push { $0.paramNonFinite != 0 ? Float.infinity : $0.paramAbsMax }
    pendingWindows += 1
  }

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
  static func main() {
    _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    do {
      switch CommandLine.arguments.dropFirst().first {
      case "benchmark": try benchmark()
      case "compare-hashes": try compareHashes()
      case "record-probe": try recordProbe()
      default: throw err("Use `OaK benchmark`.")
      }
    } catch {
      fputs("\(error)\n", stderr)
      Foundation.exit(1)
    }
  }
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
    _ seeds: [Int], _ steps: Int, _ root: URL, _ laneEnvs: [String],
    _ ablations: AblationSpec
  ) throws {
    guard laneEnvs.count == seeds.count else {
      throw err("laneEnvs (\(laneEnvs.count)) must match seed count (\(seeds.count)).")
    }
    let laneEnvKinds = try laneEnvs.map(envKind)
    let e = ProcessInfo.processInfo.environment
    let fm = FileManager.default
    let B = seeds.count
    let actDim = 6
    guard let dev = MTLCreateSystemDefaultDevice() else { throw err("Metal unavailable.") }
    let engine = try Engine(laneEnvs: laneEnvs, batch: B, dev: dev)
    let q = engine.q
    let lib = try dev.makeLibrary(URL: try requireFreshMetallib())
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let fn = try lib.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
      return try dev.makeComputePipelineState(function: fn)
    }
    let initPipe = try pipeline("oak_fleet_init")
    let snapPipe = try pipeline("oak_fleet_snapshot")
    do {
      let tablePipe = try pipeline("oak_env_table_probe")
      let words = 1 + Self.envIdsByKind.count * 4
      guard let outB = dev.makeBuffer(length: words * 4, options: [.storageModeShared]),
        let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
      else { throw err("Env table probe dispatch failed.") }
      enc.setComputePipelineState(tablePipe)
      enc.setBuffer(outB, offset: 0, index: 0)
      enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
      enc.endEncoding()
      cb.commit()
      cb.waitUntilCompleted()
      let tbl = outB.contents().bindMemory(to: UInt32.self, capacity: words)
      guard tbl[0] == UInt32(Self.envIdsByKind.count) else {
        throw err("Env table count drift: kernel \(tbl[0]) vs host \(Self.envIdsByKind.count).")
      }
      func fnv1a(_ s: String) -> UInt32 {
        var h: UInt32 = 0x811c_9dc5
        for b in Array(s.utf8) { h = (h ^ UInt32(b)) &* 16_777_619 }
        return h
      }
      var envsJson: [String: Any] = [:]
      for (i, name) in Self.envIdsByKind.enumerated() {
        guard tbl[1 + i * 4] == fnv1a(name) else {
          throw err("Env table drift at kind \(i): kernel hash != fnv1a(\(name)).")
        }
        envsJson[name] = [
          "kind": i, "naturalO": Int(tbl[1 + i * 4 + 1]),
          "actionKind": Int(tbl[1 + i * 4 + 2]), "actionCount": Int(tbl[1 + i * 4 + 3]),
        ]
      }
      let ej = try JSONSerialization.data(
        withJSONObject: envsJson, options: [.prettyPrinted, .sortedKeys])
      try ej.write(to: root.appendingPathComponent("envs.json"))
    }
    let autoG = max(2, min(32, B / 32))
    let megaG = e["OAK_MEGA_LANES_PER_TG"].flatMap(Int.init) ?? autoG
    let megaW = 32 * megaG
    func baseConstants() -> MTLFunctionConstantValues {
      let c = MTLFunctionConstantValues()
      ablations.apply(to: c)
      var megaG32 = UInt32(megaG)
      c.setConstantValue(&megaG32, type: .uint, index: 12)
      if e["OAK_PROBE_FTZ"] == "1" {
        var b = true
        c.setConstantValue(&b, type: .bool, index: 31)
      }
      if e["OAK_PROBE_FIXEDIDX"] == "1" {
        var b = true
        c.setConstantValue(&b, type: .bool, index: 30)
      }
      if let mask = e["OAK_MEGA_STAGE_MASK"].flatMap(UInt32.init) {
        var m = mask
        c.setConstantValue(&m, type: .uint, index: 25)
      }
      return c
    }
    for flag in ["OAK_PROBE_FTZ", "OAK_PROBE_FIXEDIDX", "OAK_MEGA_STAGE_MASK"] where e[flag] != nil {
      fputs("\(flag)=\(e[flag]!): timing instrument, trajectories are garbage\n", stderr)
    }
    let megaPipe = try dev.makeComputePipelineState(
      function: lib.makeFunction(name: "oak_megastep", constantValues: baseConstants()))
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
      c.setConstantValue(&k32, type: .uint, index: 29)
      variantPipes.append(try dev.makeComputePipelineState(
        function: lib.makeFunction(name: "oak_megastep", constantValues: c)))
      envRuns.append((ki, kd.kind, kd.firstLane / megaG, (kd.count + megaG - 1) / megaG))
    }
    let pipeMaxThreads = ([megaPipe] + variantPipes)
      .map(\.maxTotalThreadsPerThreadgroup).min() ?? 0
    let tgOptionCount = Int(engine.kinds[0].off.optionCount)
    let megaSlotBytes = (5 * tgOptionCount + 1) * megaG * MemoryLayout<Float>.stride
    guard megaG >= 2, megaW <= pipeMaxThreads,
      megaSlotBytes <= dev.maxThreadgroupMemoryLength
    else {
      throw err(
        "OAK_MEGA_LANES_PER_TG=\(megaG) needs \(megaW) threads/threadgroup (pipeline max \(pipeMaxThreads)) and \(megaSlotBytes) tg bytes (device max \(dev.maxThreadgroupMemoryLength)).")
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
    let res = engine.res
    let episodeLen = max(e["OAK_EPISODE_STEPS"].flatMap(Int.init) ?? 0, 0)
    let hb = max(e["OAK_BENCH_HEARTBEAT_STEPS"].flatMap(Int.init) ?? steps / 10, 1)
    func seedDir(_ lane: Int) throws -> URL {
      let dir = root.appendingPathComponent(laneEnvs[lane], isDirectory: true)
        .appendingPathComponent("seed\(seeds[lane])", isDirectory: true)
      try fm.createDirectory(at: dir, withIntermediateDirectories: true)
      return dir
    }
    let recordAll = e["OAK_RECORD_ACTIONS"] != "0"
    let recordSeeds = Set(
      (e["OAK_RECORD_SEEDS"] ?? "").split(separator: ",").compactMap {
        Int(String($0).trimmingCharacters(in: .whitespaces))
      })
    var recordLanes: [Int] = []
    var recordHandles: [FileHandle] = []
    var records: [RecordWriter] = []
    for (lane, s) in seeds.enumerated() where recordAll || recordSeeds.contains(s) {
      let dir = try seedDir(lane)
      let url = dir.appendingPathComponent("actions.bin")
      let metaURL = dir.appendingPathComponent("meta.json")
      fm.createFile(atPath: url.path, contents: nil)
      let h = try FileHandle(forWritingTo: url)
      records.append(
        try RecordWriter(
          handle: h, envKind: laneEnvKinds[lane], metaURL: metaURL, envId: laneEnvs[lane],
          seed: s, actDim: actDim))
      recordHandles.append(h)
      recordLanes.append(lane)
    }
    let maxChunkSteps = 1024
    let drainEvery = maxChunkSteps
    let recordCapacity = recordLanes.isEmpty ? 0 : 2 * maxChunkSteps
    let recordSlotBuf = try makeBuf(B * 4)
    do {
      let rp = recordSlotBuf.contents().bindMemory(to: Int32.self, capacity: B)
      for i in 0..<B { rp[i] = -1 }
      for (slot, lane) in recordLanes.enumerated() { rp[lane] = Int32(slot) }
    }
    let recordBuf = try makeBuf(recordLanes.count * recordCapacity * 16 * 4)
    let snapStride = MemoryLayout<LaneSnapshot>.stride
    precondition(snapStride == 60, "LaneSnapshot layout drifted from OakLaneSnapshot")
    let snapBuf = try makeBuf(2 * B * snapStride)
    let snapPtr = snapBuf.contents().bindMemory(to: LaneSnapshot.self, capacity: 2 * B)
    let dumpEvery = max(e["OAK_DUMP_STATE_EVERY"].flatMap(Int.init) ?? 0, 0)
    var hashHandles: [FileHandle?] = Array(repeating: nil, count: B)
    if dumpEvery > 0 {
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
      url: root.appendingPathComponent("metrics.oak.oakt"), laneCount: B)
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
      recordCapacity: UInt32(recordCapacity), invObsFloor: 1 / Float(64))
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
    func encodeMegastep(_ enc: MTLComputeCommandEncoder, _ n: Int, _ base: Int) {
      for run in envRuns {
        let kd = engine.kinds[run.kindIdx]
        bindFleet(enc, variantPipes[run.kindIdx])
        enc.setBuffer(engine.params.b, offset: kd.paramBase * 4, index: 0)
        enc.setBuffer(engine.state.b, offset: kd.stateBase * 4, index: 1)
        enc.setBuffer(engine.uints.b, offset: kd.uintBase * 4, index: 2)
        enc.setBuffer(kd.configB, offset: 0, index: 4)
        var ce = ecfg
        ce.invObsFloor = 1 / Float(2 * Int(kd.off.anchorCount))
        enc.setBytes(&ce, length: MemoryLayout<FleetEnvCfg>.stride, index: 8)
        enc.setBuffer(recordSlotBuf, offset: 0, index: 9)
        enc.setBuffer(recordBuf, offset: 0, index: 10)
        var span = SIMD3<UInt32>(UInt32(n), UInt32(base), UInt32(run.tgStart))
        enc.setBytes(&span, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 11)
        enc.setThreadgroupMemoryLength(megaTGBytes, index: 0)
        enc.dispatchThreadgroups(
          MTLSize(width: run.tgCount, height: 1, depth: 1),
          threadsPerThreadgroup: mtlSize(megaW))
      }
      enc.memoryBarrier(scope: [.buffers])
    }
    do {
      guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
        throw err("Failed to encode Metal command.")
      }
      for kd in engine.kinds {
        bindFleet(enc, initPipe)
        enc.setBuffer(engine.params.b, offset: kd.paramBase * 4, index: 0)
        enc.setBuffer(engine.state.b, offset: kd.stateBase * 4, index: 1)
        enc.setBuffer(engine.uints.b, offset: kd.uintBase * 4, index: 2)
        enc.setBuffer(kd.configB, offset: 0, index: 4)
        var ce = ecfg
        ce.invObsFloor = 1 / Float(2 * Int(kd.off.anchorCount))
        enc.setBytes(&ce, length: MemoryLayout<FleetEnvCfg>.stride, index: 8)
        enc.setBuffer(envKindsBuf, offset: 0, index: 9)
        var off = UInt32(kd.firstLane)
        enc.setBytes(&off, length: 4, index: 10)
        enc.dispatchThreads(mtlSize(kd.count), threadsPerThreadgroup: laneTG)
      }
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
      init(_ b: Int, _ rec: Int) {
        deadStep = Array(repeating: nil, count: b)
        deadDetector = Array(repeating: nil, count: b)
        prevRewardSum = Array(repeating: 0, count: b)
      }
    }
    let st = WriterState(B, recordLanes.count)
    let writer = DispatchQueue(label: "oak.fleet.gpu.writer")
    let detectorNames: [UInt32: String] = [1: "envReward", 2: "policyMean", 3: "envObs"]
    let wallStart = DispatchTime.now().uptimeNanoseconds
    func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1e9 }
    func health(_ snap: LaneSnapshot, lane: Int) -> [String: Any] {
      let adm = Double(snap.deltaSum) / Double(max(Int(snap.steps), 1))
      return [
        "rewardRate": snap.rho.isFinite ? Double(snap.rho) : NSNull(),
        "actionDeltaL2Mean": adm.isFinite ? adm : NSNull(),
        "actionDeltaPeak": snap.deltaPeak.isFinite ? Double(snap.deltaPeak) : NSNull(),
      ]
    }
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
    func drainRecords(step: Int, half: Int?) throws {
      guard !recordLanes.isEmpty, step > st.lastDrainStep else { return }
      let t0 = DispatchTime.now().uptimeNanoseconds
      let start = st.lastDrainStep
      let count = step - start
      let rp = recordBuf.contents().bindMemory(
        to: Float.self, capacity: recordLanes.count * recordCapacity * 16)
      let before = records.reduce(0) { $0 + $1.bytesWritten }
      let box = DrainErr()
      let writers = records
      let cap = recordCapacity
      DispatchQueue.concurrentPerform(iterations: writers.count) { slot in
        do {
          try writers[slot].encodeChunk(
            rp, slot: slot, capacity: cap, stepStart: start, stepCount: count,
            rewardDelta: .nan)
        } catch {
          box.record(error)
        }
      }
      if let e = box.take() { throw e }
      st.lastDrainStep = step
      if timing {
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let wrote = records.reduce(0) { $0 + $1.bytesWritten } - before
        let raw = recordLanes.count * count * 64
        fputs(
          String(
            format: "[record] step %d | %d lanes x %d rows | %.1f ms | %.2f MiB -> %.2f MiB (%.1fx)\n",
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
          } else {
            if snap.paramNonFinite != 0 {
              detector = "paramNonFinite"
            } else if Double(snap.paramAbsMax) > 1e16 {
              detector = "paramExplosion"
            } else if !snap.scalars.rho.isFinite {
              detector = "kernelRho"
            }
          }
          if let det = detector {
            st.deadStep[lane] = dStep
            st.deadDetector[lane] = det
            let pm = snap.paramNonFinite != 0 ? Double.infinity : Double(snap.paramAbsMax)
            deathLines.append(try jsonLine([
              "seed": seeds[lane], "lane": lane, "step": dStep, "detector": det,
              "policy": "oak", "rewardSumAtDeath": dRewardSum,
              "lastGoodRewardRate": dRho.isFinite ? dRho : NSNull(),
              "rhoAtDeath": snap.scalars.rho.isFinite ? Double(snap.scalars.rho) : NSNull(),
              "paramAbsMax": pm.isFinite ? pm : NSNull(),
            ] as [String: Any]))
          }
        }
        if kind == "final" {
          let pm = snap.paramNonFinite != 0 ? Double.infinity : Double(snap.paramAbsMax)
          var m: [String: Any] = [
            "v": 1, "kind": kind, "envId": laneEnvs[lane], "policy": "oak",
            "ablationLabel": ablations.label, "lane": lane, "seed": seeds[lane],
            "step": Int(snap.steps), "wallSeconds": wall, "rewardSum": Double(snap.rewardSum),
            "doneCount": Int(snap.doneCount), "health": health(snap, lane: lane),
            "firstNonFiniteStep": st.deadStep[lane] ?? NSNull(),
            "firstNonFiniteDetector": st.deadDetector[lane] ?? NSNull(),
            "kernelRho": snap.scalars.rho.isFinite ? Double(snap.scalars.rho) : NSNull(),
            "paramAbsMax": pm.isFinite ? pm : NSNull(),
          ]
          if wall > 0 { m["stepsPerSecond"] = Double(snap.steps) / wall }
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
          "step": step, "kind": kind, "policy": "oak", "alive": alive, "dead": B - alive,
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
            format: "[fleet oak gpu] step %d alive %d/%d | gpu %.1fs wall %.1fs | %.0f agg steps/s\n",
            step, alive, B, st.gpuSeconds, wall,
            wall > 0 ? Double(step) * Double(B) / wall : 0), stderr)
      }
    }
    func writeHashes(step: Int) throws {
      if let rawDir = e["OAK_DEBUG_RAW"] {
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
      }
      for lane in 0..<B {
        guard let h = hashHandles[lane] else { continue }
        let laneBytes = laneBuf.contents() + lane * fleetLaneStride
        let hs = engine.stateHashes(lane, laneBytes: laneBytes)
        let keys = ["params", "state", "uints", "scalars", "stateReward", "uintsReward", "lane"]
        var obj: [String: Any] = ["seed": seeds[lane], "step": step]
        for k in keys { obj[k] = hs[k]! }
        try h.write(contentsOf: jsonLine(obj))
      }
    }
    let explicitN = e["OAK_STEPS_PER_CB"].flatMap(Int.init)
    let tuneLock = NSLock()
    var tunedN = max(explicitN ?? 32, 1)
    guard let chunkOrder = dev.makeEvent() else { throw err("Failed to create Metal event.") }
    var chunkSeq: UInt64 = 0
    let inflight = DispatchSemaphore(value: 2)
    let snapSlots = DispatchSemaphore(value: (recordLanes.isEmpty && dumpEvery == 0) ? 2 : 1)
    var snapHalf = 0
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
      let isDrain = !recordLanes.isEmpty && !isSnap && target % drainEvery == 0
      let isDump = dumpEvery > 0 && target % dumpEvery == 0
      let isTape = isSnap || target - lastTapeStep >= tapeWindow
      inflight.wait()
      if let err = writer.sync(execute: { st.error }) { throw err }
      if isTape || isDrain { snapSlots.wait() }
      guard let cb = q.makeCommandBuffer() else {
        throw err("Failed to encode Metal command.")
      }
      cb.encodeWaitForEvent(chunkOrder, value: chunkSeq)
      guard let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else {
        throw err("Failed to encode Metal command.")
      }
      encodeMegastep(enc, target - step, step)
      if isTape {
        var fullScan: UInt32 = isSnap ? 1 : 0
        for kd in engine.kinds {
          bindFleet(enc, snapPipe)
          enc.setBuffer(engine.params.b, offset: kd.paramBase * 4, index: 0)
          enc.setBuffer(engine.state.b, offset: kd.stateBase * 4, index: 1)
          enc.setBuffer(engine.uints.b, offset: kd.uintBase * 4, index: 2)
          enc.setBuffer(kd.configB, offset: 0, index: 4)
          enc.setBuffer(snapBuf, offset: 0, index: 9)
          var ctl = SIMD2<UInt32>(UInt32(snapHalf * B), UInt32(kd.firstLane))
          enc.setBytes(&ctl, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 10)
          enc.setBytes(&fullScan, length: 4, index: 11)
          enc.dispatchThreadgroups(mtlSize(kd.count), threadsPerThreadgroup: mtlSize(32))
        }
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
    for w in records { try w.writeMeta(final: true) }
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
        guard let lv = l[field], let rv = r[field] else {
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
  static func recordProbe() throws {
    let args = CommandLine.arguments
    guard args.count >= 4 else {
      throw err("Usage: OaK record-probe <cases.bin> <out-dir>")
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: args[2]))
    guard data.count >= 12 else { throw err("record-probe: case file too small.") }
    let (kind, rowCount, chunkSteps): (UInt32, Int, Int) = data.withUnsafeBytes { raw in
      (
        raw.load(fromByteOffset: 0, as: UInt32.self),
        Int(raw.load(fromByteOffset: 4, as: UInt32.self)),
        Int(raw.load(fromByteOffset: 8, as: UInt32.self))
      )
    }
    guard rowCount > 0, chunkSteps > 0, data.count == 12 + rowCount * 16 * 4 else {
      throw err(
        "record-probe: case size mismatch — \(data.count) bytes for \(rowCount) rows (want \(12 + rowCount * 16 * 4)), chunkSteps \(chunkSteps).")
    }
    let envId = try envId(kind: kind)
    let dir = URL(fileURLWithPath: args[3], isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appendingPathComponent("actions.bin")
    FileManager.default.createFile(atPath: url.path, contents: nil)
    let h = try FileHandle(forWritingTo: url)
    defer { try? h.close() }
    let w = try RecordWriter(
      handle: h, envKind: kind, metaURL: dir.appendingPathComponent("meta.json"),
      envId: envId, seed: 0, actDim: 6)
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
  static func benchmark() throws {
    let env = ProcessInfo.processInfo.environment
    let ablations = try AblationSpec.fromEnv(env)
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
    let defaultSchedule =
      "HalfCheetah-v4:1024,KeyCorridor-cont-v0:128,FourRooms-cont-v0:64,DeepSea-cont-v0:64,Acrobot-cont-v0:64,MountainCar-cont-v0:64"
    var raw = env["OAK_BENCH_ENVS"] ?? defaultSchedule
    if raw.trimmingCharacters(in: .whitespaces).isEmpty { raw = defaultSchedule }
    var laneEnvs = [String](repeating: defaultEnv, count: seedValues.count)
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
      dir = URL(
        fileURLWithPath: "\(dir.path)-\(ProcessInfo.processInfo.processIdentifier)",
        isDirectory: true)
    }
    try fm.createDirectory(at: dir, withIntermediateDirectories: true)
    let runMeta: [String: Any] = [
      "fleetSchemaVersion": 1, "command": "benchmark", "envId": entries[0].envId,
      "batch": seedValues.count, "seeds": seedValues, "steps": steps,
      "policies": ["oak"], "ablation": ablations.names,
      "ablationLabel": ablations.label,
      "startedAt": ISO8601DateFormatter().string(from: Date()),
      "envSchedule": entries.map { e in
        ["envId": e.envId, "seedStart": e.seedStart, "seedEnd": e.seedEnd] as [String: Any]
      },
    ]
    try JSONSerialization.data(
      withJSONObject: runMeta, options: [.prettyPrinted, .sortedKeys]
    ).write(to: dir.appendingPathComponent("meta.json"), options: .atomic)
    let envDesc = entries.map { "\($0.envId):\($0.seedEnd - $0.seedStart + 1)" }.joined(
      separator: ",")
    fputs(
      "OaK fleet: B=\(seedValues.count) lanes × \(steps) steps on \(envDesc) (ablation=\(ablations.label), dir=\(dir.path))\n",
      stderr)
    try fleetGPU(seedValues, steps, dir, laneEnvs, ablations)
  }
}
