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
  var obsDim, featureCount, optionCount, actionParamCount, obsTraceLevels, perceptCount: UInt32
  var invFeatureTimeScale, invOptionCount, metaEta: Float
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
  case specFeatureCount = 14
  case specOptionCount = 15
  case specActionParams = 16
  case specTraceLevels = 17
  case specPerceptCount = 18
  case specParamStride = 19
  case specStateStride = 20
  case specUintStride = 21
  case specInvFeatureTimeScale = 22
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
  static func oakFloats(_ featureCount: Int) -> Int { 5 * featureCount + 3 + gram }
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
  static func parse(_ raw: String) throws -> AblationSpec {
    var seen = Set<String>()
    for item in raw.split(separator: ",") {
      let name = String(item).trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      if name.isEmpty || name == "none" { continue }
      guard ordered.contains(name) else {
        throw err("Unknown ablation '\(name)'. Known: none,\(ordered.joined(separator: ","))")
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

struct BenchmarkRequest {
  let policy: String
  let seeds: [Int]
  let steps: Int
  let envs: String
  let ablation: String
  let root: URL
  let timing: Bool

  let resume: URL?

  init(json: Data) throws {
    guard let obj = try JSONSerialization.jsonObject(with: json) as? [String: Any],
      let root = obj["root"] as? String
    else { throw err("benchmark request needs root") }
    self.root = URL(fileURLWithPath: root, isDirectory: true)
    timing = obj["timing"] as? Bool ?? false
    if let resumePath = obj["resume"] as? String {
      let checkpoint = URL(fileURLWithPath: resumePath)
      let parent = try RunMeta(root: checkpoint.deletingLastPathComponent())
      resume = checkpoint
      policy = parent.policy
      seeds = parent.seeds
      steps = parent.steps
      envs = parent.envLanes.map { "\($0.envId):\($0.lanes.count)" }.joined(separator: ",")
      ablation = parent.raw["ablationLabel"] as? String ?? "none"
      return
    }
    guard let policy = obj["policy"] as? String, let seeds = obj["seeds"] as? [Int],
      let steps = obj["steps"] as? Int, let envs = obj["envs"] as? String
    else { throw err("benchmark request needs policy, seeds, steps, envs, root") }
    resume = nil
    self.policy = policy
    self.seeds = seeds
    self.steps = steps
    self.envs = envs
    ablation = obj["ablation"] as? String ?? "none"
  }

  static func fromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment) throws -> BenchmarkRequest {
    let seed0 = env["OAK_BENCH_SEED_START"].flatMap(Int.init) ?? 0
    let count = env["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 1024
    let seeds = env["OAK_BENCH_SEEDS"]?.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    guard let envs = env["OAK_BENCH_ENVS"], let storage = env["OAK_METRICS_STORAGE"] else {
      throw err("OAK_BENCH_ENVS and OAK_METRICS_STORAGE are required to start a run from the environment")
    }
    let obj: [String: Any] = [
      "policy": env["OAK_BENCH_POLICY"] ?? "oak",
      "seeds": (seeds?.isEmpty == false ? seeds! : Array(seed0..<(seed0 + count))),
      "steps": env["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 1_000_000,
      "envs": envs, "ablation": env["OAK_ABLATE"] ?? "none",
      "root": (storage as NSString).expandingTildeInPath, "timing": env["OAK_FLEET_TIMING"] == "1",
    ]
    return try BenchmarkRequest(json: try JSONSerialization.data(withJSONObject: obj))
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
func presetName(_ lib: MTLLibrary) throws -> String {
  let markers = lib.functionNames.filter { $0.hasPrefix("oak_preset_") }
  guard markers.count == 1, let marker = markers.first else {
    throw err("The Metal library must carry exactly one oak_preset_* marker, found \(markers).")
  }
  return String(marker.dropFirst("oak_preset_".count))
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
  var featureCount, optionCount, nursery, traceLevels, perceptCount: UInt32
  var paramStride, stateStride, uintStride: UInt32
  var pPsi, pOptPolicy, stUtilOpt: UInt32
  var stAlpha, stBeta, stAlphaPsi, stBetaPsi, stAlphaModel, stBetaModel: UInt32
  var stAlphaAct, stBetaAct, uOptFeature, uParentA, uParentB, uKind, uNurParentA: UInt32
  var invFeatureTimeScale, invOptionCount, metaEta, sampleSigma, sampleInv2S2, actorAlphaDenom: Float
  var stKeep0, stKeep0n, stKeep1, stKeep1n, uKeep0, uKeep0n: UInt32
  var simdWidth, threadgroupLanes, pModel, pAux, modelRows: UInt32
  var activeFeatures, featureTimeScale, optionsPerThread: UInt32
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
  convenience init(_ d: MTLDevice, _ values: [T]) {
    precondition(!values.isEmpty)
    self.init(d, values.count, values[0])
    p.update(from: values, count: values.count)
  }
}
func sharedBuffer(_ device: MTLDevice, _ bytes: Int) -> MTLBuffer {
  device.makeBuffer(length: max(bytes, 16), options: [.storageModeShared])!
}
func runProbe(_ device: MTLDevice, _ function: MTLFunction, _ buffers: [MTLBuffer]) throws {
  let pipe = try device.makeComputePipelineState(function: function)
  guard let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
    let encoder = command.makeComputeCommandEncoder()
  else { throw err("Probe command allocation failed.") }
  encoder.setComputePipelineState(pipe)
  for (j, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: j) }
  encoder.dispatchThreads(mtlSize(1), threadsPerThreadgroup: mtlSize(1))
  encoder.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  guard command.status == .completed else { throw err("Probe failed: \(String(describing: command.error))") }
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

final class EnvironmentStorage {
  let fixed: [Int], entries: [Int]
  let offsets: Shared<UInt32>
  private(set) var arena: Shared<UInt32>
  private var slots: Int, safeThroughStep: Int
  var mask: UInt32 { UInt32(slots - 1) }

  private static func layout(_ fixed: [Int], _ entries: [Int], _ slots: Int) throws -> ([UInt32], Int) {
    var cursor = 0
    let starts = fixed.indices.map { lane -> Int in
      let start = cursor
      cursor += fixed[lane] + entries[lane] * slots
      return start
    }
    guard cursor <= Int(UInt32.max) else { throw err("Environment arena exceeds its addressable word count.") }
    return (starts.map(UInt32.init), cursor)
  }

  init(_ device: MTLDevice, specs: [EnvSpec], kinds: [UInt32], steps: Int, window: Int) throws {
    fixed = kinds.map {
      let spec = specs[Int($0)]
      return (spec.actionCount + spec.envStateWords + spec.naturalO + 3) & ~3
    }
    entries = kinds.map { specs[Int($0)].persistentEntryWords }
    guard entries.allSatisfy({ $0 == 0 || $0 == 3 }) else {
      throw err("Persistent entries require two coordinate words and a timestamp.")
    }
    slots = 2
    while slots < 2 * min(steps, window) { slots *= 2 }
    safeThroughStep = entries.contains(3) ? slots / 2 : steps
    let (starts, words) = try Self.layout(fixed, entries, slots)
    offsets = Shared(device, starts)
    arena = Shared(device, words, 0)
  }

  func needsCapacity(through step: Int) -> Bool { step > safeThroughStep }

  private let shape = UnsafeMutablePointer<UInt64>.allocate(capacity: 2)
  var checkpointRegions: [CheckpointRegion] {
    shape[0] = UInt64(slots)
    shape[1] = UInt64(safeThroughStep)
    return [
      CheckpointRegion(name: "envShape", base: UnsafeMutableRawPointer(shape), bytes: 16),
      CheckpointRegion(name: "envOffsets", base: offsets.b.contents(), bytes: offsets.b.length),
      CheckpointRegion(name: "envArena", base: arena.b.contents(), bytes: arena.b.length),
    ]
  }

  func restore(_ saved: [String: Data]) throws {
    guard let shapeData = saved["envShape"], shapeData.count == 16 else { throw err("checkpoint lacks envShape") }
    let words = shapeData.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
    let savedSlots = Int(words[0])
    if savedSlots != slots {
      let (_, count) = try Self.layout(fixed, entries, savedSlots)
      arena = Shared<UInt32>(arena.b.device, count, 0)
      slots = savedSlots
    }
    safeThroughStep = Int(words[1])
    try Checkpoint.restore(saved, into: Array(checkpointRegions.dropFirst()))
  }

  private static func slot(_ high: UInt32, _ low: UInt32, _ mask: Int) -> Int {
    var key = UInt64(high) << 32 | UInt64(low)
    key ^= key >> 30
    key = key &* 0xbf58476d1ce4e5b9
    key ^= key >> 27
    key = key &* 0x94d049bb133111eb
    key ^= key >> 31
    return Int(key & UInt64(mask))
  }

  func prepare(completedStep: Int, throughStep: Int) throws {
    precondition(throughStep >= completedStep)
    var largest = 0
    for lane in entries.indices where entries[lane] != 0 {
      let history = arena.p + Int(offsets.p[lane]) + fixed[lane]
      let count = (0..<slots).reduce(0) { $0 + (history[3 * $1 + 2] == 0 ? 0 : 1) }
      largest = max(largest, count)
    }
    var capacity = slots
    while capacity / 2 < largest + throughStep - completedStep {
      guard capacity <= Int(UInt32.max) / 2 else { throw err("Persistent storage capacity is not addressable.") }
      capacity *= 2
    }
    if capacity != slots {
      let (starts, words) = try Self.layout(fixed, entries, capacity)
      let grown = Shared<UInt32>(arena.b.device, words, 0)
      for lane in entries.indices {
        let source = arena.p + Int(offsets.p[lane])
        let destination = grown.p + Int(starts[lane])
        destination.update(from: source, count: fixed[lane])
        if entries[lane] == 0 { continue }
        let history = destination + fixed[lane]
        for index in 0..<slots {
          let entry = source + fixed[lane] + 3 * index
          if entry[2] == 0 { continue }
          var target = Self.slot(entry[0], entry[1], capacity - 1)
          while history[3 * target + 2] != 0 { target = (target + 1) & (capacity - 1) }
          (history + 3 * target).update(from: entry, count: 3)
        }
      }
      arena = grown
      offsets.p.update(from: starts, count: starts.count)
      slots = capacity
    }
    safeThroughStep = completedStep + capacity / 2 - largest
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
  static func runShapeProbe(
    _ dev: MTLDevice, _ lib: MTLLibrary, obsDim: Int, dPolicy: Int
  ) throws -> ShapeProbe {
    precondition(
      MemoryLayout<ShapeProbe>.stride == 176, "ShapeProbe layout drifted from OakShapeProbe")
    let constants = MTLFunctionConstantValues()
    var o32 = UInt32(obsDim)
    constants.setConstantValue(&o32, type: .uint, index: OakConst.specObsDim.rawValue)
    var d32 = UInt32(dPolicy)
    constants.setConstantValue(&d32, type: .uint, index: OakConst.specActionParams.rawValue)
    let fn = try lib.makeFunction(name: "oak_shape_probe", constantValues: constants)
    let outB = sharedBuffer(dev, 176), dummy = sharedBuffer(dev, 16)
    try runProbe(dev, fn, [outB] + Array(repeating: dummy, count: OakBuf.config.rawValue))
    return outB.contents().bindMemory(to: ShapeProbe.self, capacity: 1).pointee
  }
  static func runEnvTableProbe(_ dev: MTLDevice, _ lib: MTLLibrary) throws -> EnvTable {
    let fn = lib.makeFunction(name: "oak_env_table_probe")!
    let words = 5 + Main.envIdsByKind.count * EnvSpec.stride
    let outB = sharedBuffer(dev, words * 4)
    try runProbe(dev, fn, [outB])
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
    var pBase = 0, sBase = 0, uBase = 0
    var laneIdx = [Int](repeating: 0, count: b)
    for r in runs {
      let spec = table[Int(r.kind)]
      let naturalO = spec.naturalO
      let dk = spec.actionParams
      let sp = try Engine.runShapeProbe(
        dev, lib, obsDim: naturalO, dPolicy: dk)
      let percepts = Int(sp.perceptCount)
      precondition(
        percepts % Int(sp.simdWidth) == 0, "perception budget must tile the simd width")
      precondition(Int(sp.stateStride) % 4 == 0, "stateStride must stay 16B-aligned")
      let cfgB = Shared(dev, 1, KernelConfig(
          obsDim: UInt32(naturalO), featureCount: sp.featureCount, optionCount: sp.optionCount,
          actionParamCount: UInt32(dk), obsTraceLevels: sp.traceLevels,
          perceptCount: sp.perceptCount, invFeatureTimeScale: sp.invFeatureTimeScale,
          invOptionCount: sp.invOptionCount, metaEta: sp.metaEta,
          agentCount: UInt32(b), paramStride: sp.paramStride,
          stateStride: sp.stateStride, uintStride: sp.uintStride,
          nurseryCount: sp.nursery,
          sampleSigma: sp.sampleSigma,
          sampleInv2S2: sp.sampleInv2S2,
          actorAlphaDenom: sp.actorAlphaDenom)).b
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
    let scalarStorage = Shared(dev, batch, KernelScalars())
    scalarsB = scalarStorage.b
    scalars = scalarStorage.p
    res = [params.b, state.b, uints.b, scalarsB, kinds[0].configB]
    for kd in kinds {
      let sp = kd.off
      let featureCount = Int(sp.featureCount)
      let k = Int(sp.optionCount)
      let rn = Int(sp.nursery)
      let percepts = Int(sp.perceptCount)
      let psi = Int(sp.pPsi)
      let inv = 1 / sqrt(Float(featureCount))
      let alpha0 = 1 / Float(2 * Int(sp.perceptCount))
      let traceChannels = max(Int(sp.traceLevels) * kd.naturalO, 1)
      let basis = 1 / Float(max(kd.d, 1))
      for local in 0..<kd.count {
        let lane = kd.firstLane + local
        let lp = params.p + kd.paramBase + local * kd.paramTotal
        let ls = state.p + kd.stateBase + local * kd.stateTotal
        let lu = uints.p + kd.uintBase + local * kd.uintTotal
        for n in 0..<rn {
          lu[Int(sp.uNurParentA) + n] = UInt32(n % ((n % 4 == 3) ? traceChannels : percepts))
          lu[Int(sp.uNurParentA) + rn + n] = UInt32((n * 7 + 3) % percepts)
          lu[Int(sp.uNurParentA) + 2 * rn + n] = UInt32(n % 4)
        }
        for i in 0..<featureCount {
          if i >= percepts {
            let idx = i - percepts
            if idx < traceChannels {
              lu[Int(sp.uParentA) + i] = UInt32(idx)
              lu[Int(sp.uParentB) + i] = 0
              lu[Int(sp.uKind) + i] = 3
            } else {
              lu[Int(sp.uParentA) + i] = UInt32(idx % percepts)
              lu[Int(sp.uParentB) + i] = UInt32((idx * 7 + 3) % percepts)
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
          ls[Int(sp.stUtilOpt) + kk] = Float(featureCount)
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
      (.specObsDim, cfg.obsDim), (.specFeatureCount, cfg.featureCount), (.specOptionCount, cfg.optionCount),
      (.specActionParams, cfg.actionParamCount), (.specTraceLevels, cfg.obsTraceLevels),
      (.specPerceptCount, cfg.perceptCount), (.specParamStride, cfg.paramStride),
      (.specStateStride, cfg.stateStride), (.specUintStride, cfg.uintStride),
      (.specNursery, cfg.nurseryCount),
    ]
    for (fc, val) in us {
      var v = val
      constants.setConstantValue(&v, type: .uint, index: fc.rawValue)
    }
    let fs: [(OakConst, Float)] = [
      (.specInvFeatureTimeScale, cfg.invFeatureTimeScale), (.specInvOptionCount, cfg.invOptionCount),
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
    let words = UnsafeRawBufferPointer(start: paramPtr(lane), count: kd.paramTotal * 4).bindMemory(to: UInt32.self)
    var ordered = Array(words)
    let model = Int(kd.off.pModel), auxiliary = Int(kd.off.pAux), rows = Int(kd.off.modelRows)
    for i in 0..<rows {
      ordered[auxiliary + i] = words[model + 4 * i]
      ordered[auxiliary + rows + i] = words[model + 4 * i + 1]
      ordered[auxiliary + 2 * rows + i] = words[auxiliary + i]
      ordered[auxiliary + 3 * rows + i] = words[model + 4 * i + 2]
      ordered[auxiliary + 4 * rows + i] = words[model + 4 * i + 3]
    }
    return [
      "params": ordered.withUnsafeBytes { sha256Hex($0) },
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

let recordFloats = 2 + EnvSpec.poseLen
let recordRewardField = 1 + EnvSpec.poseLen
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
  private(set) var closed: Bool

  init(envKind: UInt32, closed: Bool) throws {
    self.codec = try RecordCodec.table(envKind)
    self.closed = closed
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
    for featureCount in laneRanks {
      offsets.append(cursor)
      cursor += featureCount
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

@main struct Main {
  static func main() {
    _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    do {
      switch CommandLine.arguments.dropFirst().first {
      case nil:
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        let port = UInt16(ProcessInfo.processInfo.environment["OAK_PORT"] ?? "") ?? 5174
        try Daemon(
          bench: cwd.appendingPathComponent(".oak-benchmarks", isDirectory: true),
          dist: cwd.appendingPathComponent("frontend/dist", isDirectory: true), port: port
        ).run()
      case "compare-hashes": try compareHashes()
      default: throw err("Run `OaK` with no arguments.")
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
    "Acrobot-cont-v0", "MountainCar-cont-v0", "Pendulum-cont-v0",
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
    let checkpointRegions: [CheckpointRegion]
    let encodeInit: (MTLComputeCommandEncoder) -> Void
    let encodeMegastep: (MTLCommandBuffer, Int, Int) -> Void
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
    let environment: EnvironmentStorage
    var envStateBuf: MTLBuffer { environment.arena.b }
    var envOffBuf: MTLBuffer { environment.offsets.b }
    func envOffset(_ lane: Int) -> Int { Int(environment.offsets.p[lane]) }
    let mjModelsBuf: MTLBuffer
    let mjScratchBuf: MTLBuffer
    let mjScratchOffBuf: MTLBuffer
    let envConstantsBuf: MTLBuffer
    var ecfg: FleetEnvCfg {
      FleetEnvCfg(
        laneCount: UInt32(B), recordCapacity: UInt32(recordCapacity),
        persistentMask: environment.mask, invObsFloor: 1 / Float(64))
    }
    var checkpointRegions: [CheckpointRegion] {
      [
        CheckpointRegion(name: "lanes", base: laneBuf.contents(), bytes: laneBuf.length),
        CheckpointRegion(name: "diagAccum", base: diagAccumBuf.contents(), bytes: diagAccumBuf.length),
      ] + environment.checkpointRegions
    }
    func bind(_ enc: MTLComputeCommandEncoder, _ pipe: MTLComputePipelineState, _ resources: [MTLBuffer], kind: Engine.Kind? = nil) {
      enc.setComputePipelineState(pipe)
      for (j, buffer) in resources.enumerated() { enc.setBuffer(buffer, offset: 0, index: j) }
      for (j, buffer) in [(6, laneBuf), (7, seedsBuf), (14, envStateBuf), (15, envOffBuf),
                          (16, mjModelsBuf), (17, mjScratchBuf), (18, mjScratchOffBuf), (19, envConstantsBuf)] {
        enc.setBuffer(buffer, offset: 0, index: j)
      }
      var config = ecfg
      if let kind {
        for (j, base) in [kind.paramBase, kind.stateBase, kind.uintBase].enumerated() {
          enc.setBuffer(resources[j], offset: base * 4, index: j)
        }
        enc.setBuffer(kind.configB, offset: 0, index: OakBuf.config.rawValue)
        config.invObsFloor = 1 / Float(2 * Int(kind.off.perceptCount))
      }
      enc.setBytes(&config, length: MemoryLayout<FleetEnvCfg>.stride, index: OakBuf.envCfg.rawValue)
    }
    init(
      _ dev: MTLDevice, lib: MTLLibrary, seeds: [Int], laneEnvKinds: [UInt32],
      envTable: [EnvSpec], envConstantWords: Int, steps: Int, policy: String
    ) throws {
      B = seeds.count
      guard steps > 0, steps <= Int(UInt32.max), steps <= Int.max / 2 else {
        throw err("Declared run horizon \(steps) cannot be represented exactly.")
      }
      windowSteps = 1024
      maxChunkSteps = windowSteps / 4
      recordCapacity = 2 * maxChunkSteps
      laneBuf = sharedBuffer(dev, B * fleetLaneStride)
      seedsBuf = Shared(dev, seeds.map { UInt32(truncatingIfNeeded: $0) }).b
      envKindsBuf = Shared(dev, laneEnvKinds).b
      recordSlotBuf = Shared(dev, (0..<B).map(Int32.init)).b
      recordBuf = sharedBuffer(dev, B * recordCapacity * recordFloats * 4)
      let snapStride = MemoryLayout<LaneSnapshot>.stride
      precondition(snapStride == 36, "LaneSnapshot layout drifted from OakLaneSnapshot")
      snapBuf = sharedBuffer(dev, 2 * B * snapStride)
      diagAccumBuf = sharedBuffer(dev, B * Diag.accumFloats * 4)
      var diagFloats = 0
      let ppoP = Diag.ppoParams(
        obsMax: envTable.map(\.naturalO).max()!, actMax: envTable.map(\.actionCount).max()!)
      var oakRankByKind: [UInt32: Int] = [:]
      for kind in Set(laneEnvKinds) {
        let spec = envTable[Int(kind)]
        oakRankByKind[kind] = Int(
          try Engine.runShapeProbe(
            dev, lib, obsDim: spec.naturalO, dPolicy: spec.actionParams
          ).featureCount)
      }
      for kind in laneEnvKinds {
        diagFloats += policy == "oak"
          ? Diag.oakFloats(oakRankByKind[kind]!) : Diag.ppoFloats(ppoP)
      }
      diagOutBuf = sharedBuffer(dev, 2 * diagFloats * 4)
      memset(diagAccumBuf.contents(), 0, diagAccumBuf.length)
      memset(diagOutBuf.contents(), 0, diagOutBuf.length)
      environment = try EnvironmentStorage(
        dev, specs: envTable, kinds: laneEnvKinds, steps: steps, window: windowSteps)
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
      mjModelsBuf = Shared(dev, Array(blob.data)).b
      var scratchOffsets = [UInt32](repeating: 0, count: B)
      var scratchCursor = 0
      for i in 0..<B {
        if let entry = mjEntryByKind[Int(laneEnvKinds[i])] {
          scratchOffsets[i] = UInt32(scratchCursor)
          scratchCursor += entry.scratchWords
        }
      }
      mjScratchBuf = sharedBuffer(dev, max(scratchCursor, 1) * 4)
      mjScratchOffBuf = Shared(dev, scratchOffsets).b
      let constants = sharedBuffer(dev, envConstantWords * 4)
      guard let fn = lib.makeFunction(name: "oak_env_constants_init") else {
        throw err("Environment constants kernel is unavailable.")
      }
      try runProbe(dev, fn, [constants])
      envConstantsBuf = constants
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
    let declared = Int(engine.kinds[0].off.simdWidth)
    guard core == declared else {
      throw err("simd width \(core) is not the \(declared) the kernel is compiled for.")
    }
    let stageMask = try stageMaskEnv(e)
    func baseConstants() -> MTLFunctionConstantValues {
      let c = MTLFunctionConstantValues()
      ablations.apply(to: c)
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
    var variantPipes: [MTLComputePipelineState] = []
    var kindMegaG: [Int] = []
    let tgOptionCount = Int(engine.kinds[0].off.optionCount)
    var geometries: [[String: Any]] = []
    for kd in engine.kinds {
      let sp = kd.off
      let shape: [String: Any] = [
        "percepts": Int(sp.perceptCount), "features": Int(sp.featureCount),
        "activeFeaturesPerStep": Int(sp.activeFeatures), "nursery": Int(sp.nursery),
        "options": Int(sp.optionCount), "optionsPerThread": Int(sp.optionsPerThread),
        "traceLevels": Int(sp.traceLevels), "featureTimeScale": Int(sp.featureTimeScale),
        "actorStepDenominator": Int(sp.actorAlphaDenom),
      ]
      let envId = Main.envIdsByKind[Int(kd.kind)]
      if let i = geometries.firstIndex(where: { NSDictionary(dictionary: $0["shape"] as! [String: Any]).isEqual(to: shape) }) {
        geometries[i]["envIds"] = (geometries[i]["envIds"] as! [String]) + [envId]
      } else {
        geometries.append(["envIds": [envId], "shape": shape])
        let line = "OaK shape: percepts \(sp.perceptCount) | features \(sp.featureCount) | active per step \(sp.activeFeatures)"
          + " | nursery \(sp.nursery) | options \(sp.optionCount) (\(sp.optionsPerThread) per wave thread)"
          + " | trace levels \(sp.traceLevels) | tracking floor \(sp.featureTimeScale) steps | actor step ÷\(Int(sp.actorAlphaDenom))"
        FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
      }
    }
    Self.pendingMeta["geometry"] = geometries
    func laneSlotBytes(_ gk: Int) -> Int {
      (5 * tgOptionCount + 1) * gk * MemoryLayout<Float>.stride
    }
    for kd in engine.kinds {
      var gk = Int(kd.off.threadgroupLanes)
      while gk > 1,
        kd.firstLane % gk != 0 || kd.count % gk != 0
          || gk * Int(kd.off.perceptCount) > core * core
          || laneSlotBytes(gk) > dev.maxThreadgroupMemoryLength
      { gk /= 2 }
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
      desc.maxTotalThreadsPerThreadgroup = core * gk
      desc.label = "oak_megastep \(Main.envIdsByKind[Int(kd.kind)])"
      variantPipes.append(try dev.makeComputePipelineState(descriptor: desc, options: []).0)
      kindMegaG.append(gk)
    }
    guard variantPipes.allSatisfy({ $0.threadExecutionWidth == core }) else {
      throw err("simd width \(core) differs across specialized pipelines.")
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
    let laneTG = mtlSize(
      min(64, initPipe.maxTotalThreadsPerThreadgroup, snapPipe.maxTotalThreadsPerThreadgroup))
    let laneRanks = engine.laneKindIdx.map { Int(engine.kinds[$0].off.featureCount) }
    let diagHost = OakDiagHost(laneRanks: laneRanks)
    var diagBase: [Int] = []
    var diagHalf = 0
    for kd in engine.kinds {
      diagBase.append(diagHalf)
      diagHalf += kd.count * Diag.oakFloats(Int(kd.off.featureCount))
    }
    precondition(2 * diagHalf * 4 <= bufs.diagOutBuf.length, "diagOut arena undersized")
    let diagPtr = bufs.diagOutBuf.contents().bindMemory(to: Float.self, capacity: 2 * diagHalf)
    let oakRegions = [
      CheckpointRegion(name: "params", base: engine.params.b.contents(), bytes: engine.params.b.length),
      CheckpointRegion(name: "state", base: engine.state.b.contents(), bytes: engine.state.b.length),
      CheckpointRegion(name: "uints", base: engine.uints.b.contents(), bytes: engine.uints.b.length),
      CheckpointRegion(name: "scalars", base: engine.scalarsB.contents(), bytes: engine.scalarsB.length),
      CheckpointRegion(name: "diagPrevAge", base: UnsafeMutableRawPointer(diagHost.prevAge), bytes: max(laneRanks.reduce(0, +), 1) * 4),
      CheckpointRegion(name: "diagPrevBirths", base: UnsafeMutableRawPointer(diagHost.prevBirths), bytes: laneRanks.count * 4),
      CheckpointRegion(name: "diagHasPrev", base: UnsafeMutableRawPointer(diagHost.hasPrev), bytes: laneRanks.count),
    ]
    return FleetPipes(
      label: "oak",
      checkpointRegions: oakRegions,
      encodeInit: { enc in
        for kd in engine.kinds {
          bufs.bind(enc, initPipe, res, kind: kd)
          enc.setBuffer(bufs.envKindsBuf, offset: 0, index: OakBuf.aux0.rawValue)
          var off = UInt32(kd.firstLane)
          enc.setBytes(&off, length: 4, index: OakBuf.aux1.rawValue)
          enc.dispatchThreads(mtlSize(kd.count), threadsPerThreadgroup: laneTG)
        }
      },
      encodeMegastep: { cb, n, base in
        guard let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else { return }
        for (ki, kd) in engine.kinds.enumerated() {
          let gk = kindMegaG[ki]
          bufs.bind(enc, variantPipes[ki], res, kind: kd)
          enc.setBuffer(bufs.recordSlotBuf, offset: 0, index: OakBuf.aux0.rawValue)
          enc.setBuffer(bufs.recordBuf, offset: 0, index: OakBuf.aux1.rawValue)
          var span = SIMD3<UInt32>(UInt32(n), UInt32(base), UInt32(kd.firstLane / gk))
          enc.setBytes(&span, length: MemoryLayout<SIMD3<UInt32>>.stride, index: OakBuf.aux2.rawValue)
          enc.setBuffer(bufs.diagAccumBuf, offset: 0, index: OakBuf.diagAccum.rawValue)
          enc.setThreadgroupMemoryLength(laneSlotBytes(gk), index: 0)
          enc.dispatchThreadgroups(
            MTLSize(width: kd.count / gk, height: 1, depth: 1),
            threadsPerThreadgroup: mtlSize(core * gk))
        }
        enc.endEncoding()
      },
      encodeSnapshot: { enc, half in
        for kd in engine.kinds {
          bufs.bind(enc, snapPipe, res, kind: kd)
          enc.setBuffer(bufs.snapBuf, offset: 0, index: OakBuf.aux0.rawValue)
          let ki = engine.kinds.firstIndex { $0.kind == kd.kind }!
          var ctl = SIMD4<UInt32>(
            UInt32(half * B), UInt32(kd.firstLane),
            UInt32(half * diagHalf + diagBase[ki]),
            UInt32(Diag.oakFloats(Int(kd.off.featureCount))))
          enc.setBytes(&ctl, length: MemoryLayout<SIMD4<UInt32>>.stride, index: OakBuf.aux1.rawValue)
          enc.setBuffer(bufs.diagAccumBuf, offset: 0, index: OakBuf.diagAccum.rawValue)
          enc.setBuffer(bufs.diagOutBuf, offset: 0, index: OakBuf.diagOut.rawValue)
          enc.dispatchThreadgroups(mtlSize(kd.count), threadsPerThreadgroup: mtlSize(core))
        }
      },
      laneHashes: { lane, laneBytes in
        let spec = engine.envTable[Int(engine.kind(of: lane).kind)]
        let envState = bufs.envStateBuf.contents().bindMemory(
          to: UInt32.self, capacity: bufs.envStateBuf.length / 4) + bufs.envOffset(lane)
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
        let featureCount = Int(kd.off.featureCount)
        let floats = Diag.oakFloats(featureCount)
        let rec = diagPtr + half * diagHalf + diagBase[ki] + (lane - kd.firstLane) * floats
        let prevAge = diagHost.prevAge + diagHost.ageOffsets[lane]
        var ages = [UInt32](repeating: 0, count: featureCount)
        for i in 0..<featureCount { ages[i] = rec[i].bitPattern }
        var dormantFraction = Float.nan
        if diagHost.hasPrev[lane] {
          var delta = [Double](repeating: .nan, count: featureCount)
          var sum = 0.0
          var validCount = 0
          for i in 0..<featureCount {
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
            for i in 0..<featureCount where delta[i].isFinite && delta[i] / mean < DiagMath.dormantTau {
              dormant += 1
            }
            dormantFraction = Float(dormant) / Float(featureCount)
          }
        }
        let utility = (0..<featureCount).map { Double(abs(rec[featureCount + $0])) }
        let births = rec[5 * featureCount]
        let birthsPerStep: Float = diagHost.hasPrev[lane] && stepCount > 0
          ? (births - diagHost.prevBirths[lane]) / Float(stepCount) : .nan
        func alphas(_ off: Int) -> [Double] { (0..<featureCount).map { Double(rec[off + $0]) } }
        let valueQ = DiagMath.logQuantiles(alphas(2 * featureCount), [0.1, 0.5, 0.9])
        let optionQ = DiagMath.logQuantiles(alphas(3 * featureCount), [0.5])
        let modelQ = DiagMath.logQuantiles(alphas(4 * featureCount), [0.5])
        for i in 0..<featureCount { prevAge[i] = ages[i] }
        diagHost.prevBirths[lane] = births
        diagHost.hasPrev[lane] = true
        return [
          dormantFraction, DiagMath.dormantFraction(utility), birthsPerStep,
          rec[5 * featureCount + 1], rec[5 * featureCount + 2],
          DiagMath.erank(rec + 5 * featureCount + 3),
          valueQ[0], valueQ[1], valueQ[2], optionQ[0], modelQ[0],
        ]
      }
    )
  }
  static func fleetGPU(
    _ policy: String, _ seeds: [Int], _ steps: Int, _ root: URL, _ laneEnvs: [String],
    _ ablations: AblationSpec, timing: Bool, control: LiveControl, resume: URL?
  ) throws {
    guard laneEnvs.count == seeds.count else {
      throw err("laneEnvs (\(laneEnvs.count)) must match seed count (\(seeds.count)).")
    }
    let laneEnvKinds = try laneEnvs.map(envKind)
    let e = ProcessInfo.processInfo.environment
    guard let dev = MTLCreateSystemDefaultDevice() else { throw err("Metal unavailable.") }
    let lib = try dev.makeLibrary(URL: try requireFreshMetallib())
    let preset = try presetName(lib)
    Self.pendingMeta["preset"] = preset
    fputs("OaK preset: \(preset)\n", stderr)
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
      envConstantWords: envProbe.constantWords, steps: steps, policy: policy)
    let pipes: FleetPipes
    if let engine {
      pipes = try oakPipes(engine, lib, dev, bufs, ablations, e)
    } else {
      pipes = try baselinePipes(policy, lib, dev, bufs, envTable, laneEnvKinds, e)
    }
    var restore: LoopRestore? = nil
    if let resume {
      let saved = try Checkpoint.load(resume)
      guard saved.laneCount == seeds.count else {
        throw err("checkpoint holds \(saved.laneCount) lanes, the run has \(seeds.count)")
      }
      try bufs.environment.restore(saved.regions)
      try Checkpoint.restore(saved.regions, into: bufs.checkpointRegions.filter { !$0.name.hasPrefix("env") } + pipes.checkpointRegions)
      restore = try LoopRestore(step: saved.step, laneCount: seeds.count, regions: saved.regions)
      Self.pendingMeta["parent"] = resume.deletingLastPathComponent().lastPathComponent
      Self.pendingMeta["startStep"] = saved.step
      try Self.writeRunMeta(root, envTable)
    }
    let gpuSeconds = try runFleetLoop(
      q, bufs, pipes, seeds: seeds, steps: steps, root: root,
      laneEnvKinds: laneEnvKinds, timing: timing, control: control, restore: restore)
    Self.pendingMeta["environmentMemoryBytes"] = bufs.envStateBuf.length
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
    let ppoBufs: [MTLBuffer]
    if isPpo {
      guard stepPipe.maxTotalThreadsPerThreadgroup >= 32 else {
        throw err("ppo_megastep supports only \(stepPipe.maxTotalThreadsPerThreadgroup) threads/threadgroup.")
      }
      ppoBufs = [
        sharedBuffer(dev, B * ppoParamStride * 4),
        sharedBuffer(dev, B * 2 * ppoParamStride * 4),
        sharedBuffer(dev, B * ppoRollStride * 4),
        sharedBuffer(dev, B * ppoScratchStride * 4),
        sharedBuffer(dev, B * ppoStateStride * 4),
      ]
    } else {
      ppoBufs = []
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
    let ppoRegions = zip(["ppoParams", "ppoAdam", "ppoRollout", "ppoScratch", "ppoState"], ppoBufs).map {
      CheckpointRegion(name: $0, base: $1.contents(), bytes: $1.length)
    }
    return FleetPipes(
      label: policy,
      checkpointRegions: ppoRegions,
      encodeInit: { enc in
        bufs.bind(enc, initPipe, ppoBufs)
        enc.setBuffer(bufs.envKindsBuf, offset: 0, index: OakBuf.aux0.rawValue)
        enc.dispatchThreads(mtlSize(stepThreads), threadsPerThreadgroup: stepTG)
      },
      encodeMegastep: { cb, n, base in
        guard let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else { return }
        bufs.bind(enc, stepPipe, ppoBufs)
        enc.setBuffer(bufs.recordSlotBuf, offset: 0, index: OakBuf.aux0.rawValue)
        enc.setBuffer(bufs.recordBuf, offset: 0, index: OakBuf.aux1.rawValue)
        var span = SIMD3<UInt32>(UInt32(n), UInt32(base), 0)
        enc.setBytes(&span, length: MemoryLayout<SIMD3<UInt32>>.stride, index: OakBuf.aux2.rawValue)
        if isPpo {
          enc.setBuffer(bufs.diagAccumBuf, offset: 0, index: OakBuf.diagAccum.rawValue)
        }
        enc.dispatchThreads(mtlSize(stepThreads), threadsPerThreadgroup: stepTG)
        enc.memoryBarrier(scope: [.buffers])
        enc.endEncoding()
      },
      encodeSnapshot: { enc, half in
        bufs.bind(enc, snapPipe, ppoBufs)
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
    seeds: [Int], steps: Int, root: URL, laneEnvKinds: [UInt32], timing: Bool, control: LiveControl,
    restore: LoopRestore?
  ) throws -> Double {
    let e = ProcessInfo.processInfo.environment
    let fm = FileManager.default
    let B = bufs.B
    let dev = q.device
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0)
    let runWriter = try RunWriter(
      root: root, laneCount: B, steps: steps, laneEnvKinds: laneEnvKinds,
      diagFields: pipes.diagFields, closed: restore?.closed)
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
    defer {
      try? hashHandle?.close()
      try? runWriter.close()
      staging.deallocate()
    }
    if restore == nil {
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
    if let restore {
      st.deadStep = restore.deadStep
      st.deadDetector = restore.deadDetector
      st.deadCount = restore.deadStep.filter { $0 != nil }.count
    }
    func saveCheckpoint(_ name: String, step: Int) throws {
      var dead = st.deadStep.map { $0.map(UInt32.init) ?? UInt32.max }
      var detector = st.deadDetector.map { $0 ?? UInt32.max }
      var closed = runWriter.closed.map { UInt8($0 ? 1 : 0) }
      try dead.withUnsafeMutableBytes { d in
        try detector.withUnsafeMutableBytes { t in
          try closed.withUnsafeMutableBytes { c in
            try Checkpoint.save(
              root.appendingPathComponent(name), step: step, laneCount: B,
              regions: bufs.checkpointRegions + pipes.checkpointRegions + [
                CheckpointRegion(name: "deadStep", base: d.baseAddress!, bytes: d.count),
                CheckpointRegion(name: "deadDetector", base: t.baseAddress!, bytes: t.count),
                CheckpointRegion(name: "closed", base: c.baseAddress!, bytes: c.count),
              ])
          }
        }
      }
    }
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
      let t0 = DispatchTime.now().uptimeNanoseconds
      let before = runWriter.payload.bytesWritten
      try runWriter.writeWindow(
        stepStart: stepStart, stepCount: step - stepStart, wallSeconds: now(),
        snap: UnsafePointer(snapPtr + half * B), deadStep: st.deadStep,
        deadDetector: st.deadDetector, diag: diag,
        records: UnsafePointer(staging), capacity: windowSteps
      ) { st.error = $0 }
      if st.didFail, let e = st.error { throw e }
      if timing {
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let wrote = runWriter.payload.bytesWritten - before
        let raw = B * (step - stepStart) * recordFloats * 4
        fputs(
          String(
            format: "[record] step %d | %d lanes x %d rows | %.1f ms | %.2f MiB -> %.2f MiB (%.1fx)\n",
            step, B, step - stepStart, ms, Double(raw) / 1_048_576,
            Double(wrote) / 1_048_576, wrote > 0 ? Double(raw) / Double(wrote) : 0), stderr)
      }
      control.onWindow?(step)
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
    var lastWindowStep = restore?.step ?? 0
    var step = restore?.step ?? 0
    while step < steps {
      tuneLock.lock()
      let n = tunedN
      tuneLock.unlock()
      var boundary = min(((step / windowSteps) + 1) * windowSteps, steps)
      if dumpEvery > 0 { boundary = min(boundary, ((step / dumpEvery) + 1) * dumpEvery) }
      let target = min(boundary, step + n)
      let isWindow = target % windowSteps == 0 || target == steps
      let isDump = dumpEvery > 0 && target % dumpEvery == 0
      if bufs.environment.needsCapacity(through: target) || control.pending {
        inflight.wait()
        inflight.wait()
        writer.sync {}
        if let error = st.error { throw error }
        if bufs.environment.needsCapacity(through: target) {
          try bufs.environment.prepare(completedStep: step, throughStep: target)
        }
        let taken = control.take(allowSave: step % windowSteps == 0)
        for s in taken.signals where s.lane >= 0 && s.lane < B {
          bufs.environment.arena.p[Int(bufs.environment.offsets.p[s.lane]) + s.word] = s.value
        }
        if taken.save { try saveCheckpoint("checkpoint.\(step)", step: step) }
        inflight.signal()
        inflight.signal()
      }
      inflight.wait()
      if st.didFail, let err = st.error { throw err }
      if isWindow { windowSlots.wait() }
      guard let cb = q.makeCommandBuffer() else {
        throw err("Failed to encode Metal command.")
      }
      cb.encodeWaitForEvent(chunkOrder, value: chunkSeq)
      pipes.encodeMegastep(cb, target - step, step)
      if isWindow {
        guard let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else {
          throw err("Failed to encode Metal command.")
        }
        pipes.encodeSnapshot(enc, snapHalf)
        enc.endEncoding()
      }
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
        if gpuT > 0, nSteps > 0 {
          let perStep = gpuT / Double(nSteps)
          tuneLock.lock()
          if explicitN == nil { tunedN = max(1, min(max(1, min(Int(0.016 / perStep), maxChunkSteps)), tunedN * 4)) }
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
    try saveCheckpoint("checkpoint", step: step)
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
  static func benchmark(_ request: BenchmarkRequest, control: LiveControl) throws {
    let ablations = try AblationSpec.parse(request.ablation)
    let policy = request.policy
    guard ["oak", "random", "ppo"].contains(policy) else {
      throw err("policy must be one of oak,random,ppo (got \(policy)).")
    }
    guard policy == "oak" || ablations.label == "none" else {
      throw err("Ablations apply only to the oak policy.")
    }
    let fm = FileManager.default
    let seedValues = request.seeds
    guard !seedValues.isEmpty else { throw err("seeds are required.") }
    let steps = max(request.steps, 1)
    let raw = request.envs.trimmingCharacters(in: .whitespaces)
    guard !raw.isEmpty else {
      throw err("envs is required (EnvId:count,... covering every lane).")
    }
    var laneEnvs = [String](repeating: "", count: seedValues.count)
    var entries: [(envId: String, seedStart: Int, seedEnd: Int)] = []
    var offset = 0
    for item in raw.split(separator: ",") {
      let parts = item.split(separator: ":")
      guard parts.count == 2, let count = Int(parts[1].trimmingCharacters(in: .whitespaces)),
        count > 0
      else {
        throw err("Bad envs entry '\(item)' (expected EnvId:count).")
      }
      let name = String(parts[0]).trimmingCharacters(in: .whitespaces)
      _ = try Main.envKind(name)
      guard offset + count <= seedValues.count else {
        throw err(
          "envs schedule overflows the fleet: \(offset + count) lanes > B=\(seedValues.count).")
      }
      for i in 0..<count { laneEnvs[offset + i] = name }
      entries.append((name, seedValues[offset], seedValues[offset + count - 1]))
      offset += count
    }
    guard offset == seedValues.count else {
      throw err("envs schedule covers \(offset) of B=\(seedValues.count) lanes.")
    }
    let dir = request.root
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
        $0 != ".DS_Store" && $0 != "identity.json" && $0 != "baselines" && $0 != "run.log"
      }).isEmpty
    {
      throw err("\(dir.path) is not empty; refusing to write beside existing data.")
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
    try fleetGPU(
      policy, seedValues, steps, dir, laneEnvs, ablations, timing: request.timing, control: control,
      resume: request.resume)
  }
}
