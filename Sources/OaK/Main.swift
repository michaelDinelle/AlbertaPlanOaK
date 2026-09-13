import CryptoKit
import Darwin
import Foundation
import Metal

struct OakError: Error, CustomStringConvertible { let description: String }
func err(_ s: String) -> OakError { OakError(description: s) }

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
  case specObsDim = 0
  case specActionParams = 1
  case specEnvKind = 2
  case megaG = 3
  case megaStageMask = 4
  case probeFixedIdx = 5
}

enum OakBuf: Int {
  case envCfg = 8
  case aux0 = 9
  case aux1 = 10
  case aux2 = 11
  case ppoAccum = 12
  case ppoRecord = 13
}

enum Ppo {
  static let hidden = 64
  static let gram = hidden * hidden
  static let accumFloats = gram + 2 * hidden
  static func params(obsMax: Int, actMax: Int) -> Int {
    (2 * obsMax * hidden + 2 * hidden + hidden * actMax + 2 * actMax + hidden + 1 + 3) & ~3
  }
  static func recordFloats(_ params: Int) -> Int { 2 * hidden + 2 + params + gram }
}

struct BenchmarkRequest {
  let policy: String
  let seeds: [Int]
  let steps: Int
  let envs: String
  let root: URL

  init(json: Data) throws {
    guard let obj = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
      throw err("benchmark request is not an object")
    }
    guard let policy = obj["policy"] as? String, let seeds = obj["seeds"] as? [Int],
      let steps = obj["steps"] as? Int, let envs = obj["envs"] as? String, let root = obj["root"] as? String
    else { throw err("benchmark request needs policy, seeds, steps, envs, root") }
    self.root = URL(fileURLWithPath: root, isDirectory: true)
    self.policy = policy
    self.seeds = seeds
    self.steps = steps
    self.envs = envs
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
  var pOption, recordFloats, recordValue, recordPolicy: UInt32
  var stUtilOpt, stValueMeta, stOptionValueMeta, stModelMeta, stActorMeta: UInt32
  var uOptFeature, uFeatureMeta, uNurseryMeta: UInt32
  var simdWidth, threadgroupLanes, activeFeatures, featureTimeScale, optionsPerThread, actorAlphaDenom: UInt32
  var laneThreadgroupFloats: UInt32
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
func runProbe(_ queue: MTLCommandQueue, _ function: MTLFunction, _ buffers: [MTLBuffer]) throws {
  let pipe = try queue.device.makeComputePipelineState(function: function)
  guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
    throw err("Probe command allocation failed.")
  }
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
  let nameHash: UInt32
  let naturalO: Int
  let actionCount: Int
  let actionParams: Int
  let controlHz: Int
  let envStateWords: Int
  let mjIndex: Int
  let slots: [Int]
  let persistentEntryWords: Int
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
    let paramBase, stateBase, uintBase: Int
  }
  let batch: Int
  let kinds: [Kind]
  let envTable: [EnvSpec]
  let q: MTLCommandQueue, params, state: Shared<Float>,
    uints: Shared<UInt32>, res: [MTLBuffer]
  let scalarsB: MTLBuffer
  static func specConstants(_ spec: EnvSpec, kind: UInt32) -> MTLFunctionConstantValues {
    let constants = MTLFunctionConstantValues()
    var o32 = UInt32(spec.naturalO)
    constants.setConstantValue(&o32, type: .uint, index: OakConst.specObsDim.rawValue)
    var d32 = UInt32(spec.actionParams)
    constants.setConstantValue(&d32, type: .uint, index: OakConst.specActionParams.rawValue)
    var k32 = kind
    constants.setConstantValue(&k32, type: .uint, index: OakConst.specEnvKind.rawValue)
    return constants
  }
  static func runShapeProbe(_ queue: MTLCommandQueue, _ lib: MTLLibrary, _ spec: EnvSpec, kind: UInt32) throws -> ShapeProbe {
    precondition(MemoryLayout<ShapeProbe>.stride == 108, "ShapeProbe layout drifted from OakShapeProbe")
    let fn = try lib.makeFunction(name: "oak_shape_probe", constantValues: specConstants(spec, kind: kind))
    let outB = sharedBuffer(queue.device, 108)
    try runProbe(queue, fn, [outB])
    return outB.contents().bindMemory(to: ShapeProbe.self, capacity: 1).pointee
  }
  static func runEnvTableProbe(_ queue: MTLCommandQueue, _ lib: MTLLibrary) throws -> EnvTable {
    let fn = lib.makeFunction(name: "oak_env_table_probe")!
    let words = 5 + Main.envIdsByKind.count * EnvSpec.stride
    let outB = sharedBuffer(queue.device, words * 4)
    try runProbe(queue, fn, [outB])
    let tbl = outB.contents().bindMemory(to: UInt32.self, capacity: words)
    guard tbl[0] == UInt32(Main.envIdsByKind.count) else {
      throw err("Env table count drift: kernel \(tbl[0]) vs host \(Main.envIdsByKind.count).")
    }
    let specs = (0..<Main.envIdsByKind.count).map { i in
      let row = 5 + i * EnvSpec.stride
      return EnvSpec(
        nameHash: tbl[row], naturalO: Int(tbl[row + 1]),
        actionCount: Int(tbl[row + 3]), actionParams: Int(tbl[row + 4]),
        controlHz: Int(tbl[row + 7]), envStateWords: Int(tbl[row + 8]),
        mjIndex: tbl[row + 9] == 0xffff_ffff ? -1 : Int(tbl[row + 9]),
        slots: (0..<EnvSpec.poseLen).map { Int(tbl[row + 11 + $0]) },
        persistentEntryWords: Int(tbl[row + 11 + EnvSpec.poseLen]))
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
    laneEnvs: [String], batch b: Int, queue: MTLCommandQueue, lib: MTLLibrary,
    envTable table: [EnvSpec]
  ) throws {
    precondition(b >= 1 && laneEnvs.count == b, "Engine batch/laneEnvs mismatch")
    batch = b
    q = queue
    let dev = queue.device
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
    for r in runs {
      let spec = table[Int(r.kind)]
      let sp = try Engine.runShapeProbe(queue, lib, spec, kind: r.kind)
      precondition(
        Int(sp.perceptCount) % Int(sp.simdWidth) == 0, "perception budget must tile the simd width")
      precondition(Int(sp.stateStride) % 4 == 0, "stateStride must stay 16B-aligned")
      let kd = Kind(
        kind: r.kind, naturalO: spec.naturalO, d: spec.actionParams, count: r.count, firstLane: r.firstLane,
        off: sp, paramBase: pBase, stateBase: sBase, uintBase: uBase)
      built.append(kd)
      pBase += r.count * Int(sp.paramStride)
      sBase += r.count * Int(sp.stateStride)
      uBase += r.count * Int(sp.uintStride)
    }
    kinds = built
    envTable = table
    params = Shared(dev, pBase, 0)
    state = Shared(dev, sBase, 0)
    uints = Shared(dev, uBase, 0)
    scalarsB = Shared(dev, batch, KernelScalars()).b
    res = [params.b, state.b, uints.b, scalarsB]
    for kd in kinds {
      let sp = kd.off
      let featureCount = Int(sp.featureCount)
      let k = Int(sp.optionCount)
      let rn = Int(sp.nursery)
      let percepts = Int(sp.perceptCount)
      let record = Int(sp.recordFloats)
      let inv = 1 / sqrt(Float(featureCount))
      let alpha0 = 1 / Float(2 * Int(sp.perceptCount))
      let traceChannels = max(Int(sp.traceLevels) * kd.naturalO, 1)
      let basis = 1 / Float(max(kd.d, 1))
      for local in 0..<kd.count {
        let lane = kd.firstLane + local
        let lp = params.p + kd.paramBase + local * Int(sp.paramStride)
        let ls = state.p + kd.stateBase + local * Int(sp.stateStride)
        let lu = uints.p + kd.uintBase + local * Int(sp.uintStride)
        for n in 0..<rn {
          let meta = lu + Int(sp.uNurseryMeta) + 4 * n
          meta[0] = UInt32(n % ((n % 4 == 3) ? traceChannels : percepts))
          meta[1] = UInt32((n * 7 + 3) % percepts)
          meta[2] = UInt32(n % 4)
        }
        for i in 0..<featureCount {
          let meta = lu + Int(sp.uFeatureMeta) + 4 * i
          if i >= percepts {
            let idx = i - percepts
            if idx < traceChannels {
              meta[0] = UInt32(idx)
              meta[1] = 0
              meta[2] = 3
            } else {
              meta[0] = UInt32(idx % percepts)
              meta[1] = UInt32((idx * 7 + 3) % percepts)
              meta[2] = UInt32(idx % 3)
            }
          }
          for base in [sp.stValueMeta, sp.stOptionValueMeta, sp.stModelMeta] {
            ls[Int(base) + 4 * i] = alpha0
            ls[Int(base) + 4 * i + 1] = log(alpha0)
          }
        }
        for kk in 0..<k {
          lu[Int(sp.uOptFeature) + kk] = UInt32(kk)
          lp[Int(sp.pOption) + (kk * k + kk) * record + Int(sp.recordValue)] = inv
          ls[Int(sp.stUtilOpt) + kk] = Float(featureCount)
          ls[Int(sp.stActorMeta) + 4 * kk] = 1
          let h = UInt32(truncatingIfNeeded: lane &* 2_654_435_761 &+ kk &* 40_503 &+ 977)
          let slot = Int(h % UInt32(max(kd.d / 2, 1)))
          let sign: Float = ((h >> 16) & 1) == 0 ? 1 : -1
          lp[Int(sp.pOption) + kk * record + Int(sp.recordPolicy) + slot] = sign * basis
        }
      }
    }
  }
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

  init(envKind: UInt32) throws {
    codec = try RecordCodec.table(envKind)
    closed = false
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

@main struct Main {
  static func main() {
    _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    do {
      guard CommandLine.arguments.count == 1 else { throw err("Run `OaK` with no arguments.") }
      let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
      let port = UInt16(ProcessInfo.processInfo.environment["OAK_PORT"] ?? "") ?? 5174
      try Daemon(
        bench: cwd.appendingPathComponent(".oak-benchmarks", isDirectory: true),
        dist: cwd.appendingPathComponent("frontend/dist", isDirectory: true), port: port
      ).run()
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
        "envId": envId,
        "controlHz": spec.controlHz,
        "category": codec.category,
        "fields": codec.fields.map { f in ["slot": f.idx, "wrap": f.wrap] as [String: Any] },
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
  struct FleetPipes {
    let label: String
    let encodeInit: (MTLComputeCommandEncoder) -> Void
    let encodeMegastep: (MTLCommandBuffer, Int, Int) -> Void
    let encodeSnapshot: (MTLComputeCommandEncoder, Int) -> Void
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
    let environment: EnvironmentStorage
    var envStateBuf: MTLBuffer { environment.arena.b }
    var envOffBuf: MTLBuffer { environment.offsets.b }
    let mjModelsBuf: MTLBuffer
    let mjScratchBuf: MTLBuffer
    let mjScratchOffBuf: MTLBuffer
    let envConstantsBuf: MTLBuffer
    var ecfg: FleetEnvCfg {
      FleetEnvCfg(
        laneCount: UInt32(B), recordCapacity: UInt32(recordCapacity),
        persistentMask: environment.mask, invObsFloor: 1 / Float(64))
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
        config.invObsFloor = 1 / Float(2 * Int(kind.off.perceptCount))
      }
      enc.setBytes(&config, length: MemoryLayout<FleetEnvCfg>.stride, index: OakBuf.envCfg.rawValue)
    }
    init(
      _ queue: MTLCommandQueue, lib: MTLLibrary, seeds: [Int], laneEnvKinds: [UInt32],
      envTable: [EnvSpec], envConstantWords: Int, steps: Int
    ) throws {
      let dev = queue.device
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
      try runProbe(queue, fn, [constants])
      envConstantsBuf = constants
    }
  }
  static func oakPipes(
    _ engine: Engine, _ lib: MTLLibrary, _ bufs: FleetBuffers, _ e: [String: String]
  ) throws -> FleetPipes {
    let dev = engine.q.device
    let stageMask = try stageMaskEnv(e)
    func constants(_ kd: Engine.Kind) -> MTLFunctionConstantValues {
      let c = Engine.specConstants(engine.envTable[Int(kd.kind)], kind: kd.kind)
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
    func pipeline(_ name: String, _ kd: Engine.Kind) throws -> MTLComputePipelineState {
      try dev.makeComputePipelineState(function: try lib.makeFunction(name: name, constantValues: constants(kd)))
    }
    let core = Int(engine.kinds[0].off.simdWidth)
    var initPipes: [MTLComputePipelineState] = []
    var snapPipes: [MTLComputePipelineState] = []
    var stepPipes: [MTLComputePipelineState] = []
    var kindMegaG: [Int] = []
    var shapes = Set<UInt32>()
    for kd in engine.kinds {
      let sp = kd.off
      let envId = Main.envIdsByKind[Int(kd.kind)]
      if shapes.insert(sp.perceptCount).inserted {
        let line = "OaK shape: percepts \(sp.perceptCount) | features \(sp.featureCount) | active per step \(sp.activeFeatures)"
          + " | nursery \(sp.nursery) | options \(sp.optionCount) (\(sp.optionsPerThread) per wave thread)"
          + " | trace levels \(sp.traceLevels) | tracking floor \(sp.featureTimeScale) steps | actor step ÷\(Int(sp.actorAlphaDenom))"
        FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
      }
      func laneSlotBytes(_ gk: Int) -> Int { Int(sp.laneThreadgroupFloats) * gk * MemoryLayout<Float>.stride }
      var gk = Int(sp.threadgroupLanes)
      while gk > 1,
        kd.firstLane % gk != 0 || kd.count % gk != 0
          || gk * Int(sp.perceptCount) > core * core
          || laneSlotBytes(gk) > dev.maxThreadgroupMemoryLength
      { gk /= 2 }
      let c = constants(kd)
      var g32 = UInt32(gk)
      c.setConstantValue(&g32, type: .uint, index: OakConst.megaG.rawValue)
      let desc = MTLComputePipelineDescriptor()
      desc.computeFunction = try lib.makeFunction(name: "oak_megastep", constantValues: c)
      desc.maxTotalThreadsPerThreadgroup = core * gk
      desc.label = "oak_megastep \(envId)"
      let stepPipe = try dev.makeComputePipelineState(descriptor: desc, options: []).0
      guard stepPipe.threadExecutionWidth == core, core * gk <= stepPipe.maxTotalThreadsPerThreadgroup else {
        throw err("megaG=\(gk) needs \(core * gk) threads/threadgroup at simd width \(core); pipeline offers \(stepPipe.maxTotalThreadsPerThreadgroup) at width \(stepPipe.threadExecutionWidth).")
      }
      stepPipes.append(stepPipe)
      kindMegaG.append(gk)
      initPipes.append(try pipeline("oak_fleet_init", kd))
      snapPipes.append(try pipeline("oak_fleet_snapshot", kd))
    }
    let res = engine.res
    return FleetPipes(
      label: "oak",
      encodeInit: { enc in
        for (ki, kd) in engine.kinds.enumerated() {
          bufs.bind(enc, initPipes[ki], res, kind: kd)
          var off = UInt32(kd.firstLane)
          enc.setBytes(&off, length: 4, index: OakBuf.aux1.rawValue)
          enc.dispatchThreads(mtlSize(kd.count), threadsPerThreadgroup: mtlSize(min(64, initPipes[ki].maxTotalThreadsPerThreadgroup)))
        }
      },
      encodeMegastep: { cb, n, base in
        guard let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else { return }
        for (ki, kd) in engine.kinds.enumerated() {
          let gk = kindMegaG[ki]
          bufs.bind(enc, stepPipes[ki], res, kind: kd)
          enc.setBuffer(bufs.recordSlotBuf, offset: 0, index: OakBuf.aux0.rawValue)
          enc.setBuffer(bufs.recordBuf, offset: 0, index: OakBuf.aux1.rawValue)
          var span = SIMD3<UInt32>(UInt32(n), UInt32(base), UInt32(kd.firstLane / gk))
          enc.setBytes(&span, length: MemoryLayout<SIMD3<UInt32>>.stride, index: OakBuf.aux2.rawValue)
          enc.setThreadgroupMemoryLength(Int(kd.off.laneThreadgroupFloats) * gk * MemoryLayout<Float>.stride, index: 0)
          enc.dispatchThreadgroups(
            MTLSize(width: kd.count / gk, height: 1, depth: 1),
            threadsPerThreadgroup: mtlSize(core * gk))
        }
        enc.endEncoding()
      },
      encodeSnapshot: { enc, half in
        for (ki, kd) in engine.kinds.enumerated() {
          bufs.bind(enc, snapPipes[ki], res, kind: kd)
          enc.setBuffer(bufs.snapBuf, offset: 0, index: OakBuf.aux0.rawValue)
          var ctl = SIMD2<UInt32>(UInt32(half * bufs.B), UInt32(kd.firstLane))
          enc.setBytes(&ctl, length: MemoryLayout<SIMD2<UInt32>>.stride, index: OakBuf.aux1.rawValue)
          enc.dispatchThreadgroups(mtlSize(kd.count), threadsPerThreadgroup: mtlSize(core))
        }
      }
    )
  }
  static func fleetGPU(
    _ policy: String, _ seeds: [Int], _ steps: Int, _ root: URL, _ laneEnvs: [String],
    onWindow: @escaping (Int) -> Void
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
    guard let q = dev.makeCommandQueue() else { throw err("Metal command queue unavailable.") }
    let envProbe = try Engine.runEnvTableProbe(q, lib)
    let envTable = envProbe.specs
    let engine = policy == "oak"
      ? try Engine(laneEnvs: laneEnvs, batch: seeds.count, queue: q, lib: lib, envTable: envTable) : nil
    for (i, name) in Self.envIdsByKind.enumerated() {
      guard envTable[i].nameHash == fnv1a(name) else {
        throw err("Env table drift at kind \(i): kernel hash != fnv1a(\(name)).")
      }
      let declared = Set(envTable[i].slots.indices.filter { envTable[i].slots[$0] != 0 })
      let recorded = Set(RecordCodec.byKind[i].fields.map(\.idx))
      guard declared == recorded else {
        throw err("\(name): kernel declares slots \(declared.sorted()) but the codec records \(recorded.sorted()).")
      }
    }
    Self.pendingMeta["hardwareKey"] = Self.hardwareKey(dev)
    try Self.writeRunMeta(root, envTable)
    for flag in ["OAK_PROBE_FIXEDIDX", "OAK_MEGA_STAGE_MASK"] where e[flag] != nil {
      fputs("\(flag)=\(e[flag]!): timing instrument, trajectories are garbage\n", stderr)
    }
    let bufs = try FleetBuffers(
      q, lib: lib, seeds: seeds, laneEnvKinds: laneEnvKinds, envTable: envTable,
      envConstantWords: envProbe.constantWords, steps: steps)
    let pipes: FleetPipes
    if let engine {
      pipes = try oakPipes(engine, lib, bufs, e)
    } else {
      pipes = try baselinePipes(policy, lib, dev, bufs, envTable, laneEnvKinds, e)
    }
    let gpuSeconds = try runFleetLoop(
      q, bufs, pipes, steps: steps, root: root, laneEnvKinds: laneEnvKinds, onWindow: onWindow)
    Self.pendingMeta["stepsPerSecond"] = gpuSeconds > 0
      ? Double(steps) * Double(seeds.count) / gpuSeconds : 0
    Self.pendingMeta["finishedAt"] = ISO8601DateFormatter().string(from: Date())
    try Self.writeRunMeta(root, envTable)
  }
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
    let stageMask = try stageMaskEnv(e)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let constants = MTLFunctionConstantValues()
      if let mask = stageMask {
        var m = mask
        constants.setConstantValue(&m, type: .uint, index: OakConst.megaStageMask.rawValue)
      }
      let fn = try lib.makeFunction(name: name, constantValues: constants)
      return try dev.makeComputePipelineState(function: fn)
    }
    let isPpo = policy == "ppo"
    let obsMax = envTable.map(\.naturalO).max()!
    let actMax = envTable.map(\.actionCount).max()!
    let ppoH = Ppo.hidden
    let ppoT = 128
    let ppoMB = 32
    let ppoParamStride = Ppo.params(obsMax: obsMax, actMax: actMax)
    let ppoRollStride = ((ppoT + 1) * obsMax + ppoT * actMax + 6 * ppoT + 15) & ~15
    let ppoScratchStride =
      ((ppoMB * obsMax + 4 * ppoMB * ppoH + 2 * ppoMB * actMax + ppoMB + ppoT
        + ppoParamStride + 15) & ~15) + 16
    let ppoStateStride = (2 * obsMax + 6 + 15) & ~15
    let initPipe = try pipeline(isPpo ? "ppo_fleet_init" : "baseline_fleet_init")
    let stepPipe = try pipeline("\(policy)_megastep")
    let snapPipe = try pipeline(isPpo ? "ppo_fleet_snapshot" : "baseline_fleet_snapshot")
    let B = bufs.B
    let ppoBufs: [MTLBuffer]
    let ppoAccum: MTLBuffer
    let ppoRecord: MTLBuffer
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
      ppoAccum = sharedBuffer(dev, B * Ppo.accumFloats * 4)
      ppoRecord = sharedBuffer(dev, 2 * B * Ppo.recordFloats(ppoParamStride) * 4)
      memset(ppoAccum.contents(), 0, ppoAccum.length)
      memset(ppoRecord.contents(), 0, ppoRecord.length)
    } else {
      ppoBufs = []
      ppoAccum = sharedBuffer(dev, 16)
      ppoRecord = sharedBuffer(dev, 16)
    }
    let laneTG = mtlSize(min(
      64, initPipe.maxTotalThreadsPerThreadgroup, stepPipe.maxTotalThreadsPerThreadgroup,
      snapPipe.maxTotalThreadsPerThreadgroup))
    let stepThreads = B * 32
    let stepTG = mtlSize(32)
    return FleetPipes(
      label: policy,
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
        if isPpo { enc.setBuffer(ppoAccum, offset: 0, index: OakBuf.ppoAccum.rawValue) }
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
          enc.setBuffer(ppoAccum, offset: 0, index: OakBuf.ppoAccum.rawValue)
          enc.setBuffer(ppoRecord, offset: 0, index: OakBuf.ppoRecord.rawValue)
        }
        enc.dispatchThreads(mtlSize(B), threadsPerThreadgroup: laneTG)
      }
    )
  }
  static func runFleetLoop(
    _ q: MTLCommandQueue, _ bufs: FleetBuffers, _ pipes: FleetPipes,
    steps: Int, root: URL, laneEnvKinds: [UInt32], onWindow: @escaping (Int) -> Void
  ) throws -> Double {
    let e = ProcessInfo.processInfo.environment
    let B = bufs.B
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0)
    let runWriter = try RunWriter(root: root, laneCount: B, steps: steps, laneEnvKinds: laneEnvKinds)
    let maxChunkSteps = bufs.maxChunkSteps
    let windowSteps = bufs.windowSteps
    let recordCapacity = bufs.recordCapacity
    let staging = UnsafeMutablePointer<Float>.allocate(capacity: windowSteps * recordFloats * B)
    let snapPtr = bufs.snapBuf.contents().bindMemory(to: LaneSnapshot.self, capacity: 2 * B)
    defer {
      try? runWriter.close()
      staging.deallocate()
    }
    guard let initCb = q.makeCommandBuffer(), let initEnc = initCb.makeComputeCommandEncoder() else {
      throw err("Failed to encode Metal command.")
    }
    pipes.encodeInit(initEnc)
    initEnc.endEncoding()
    initCb.commit()
    initCb.waitUntilCompleted()
    if let e = initCb.error { throw err("Metal command failed: \(e)") }
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
      let t0 = DispatchTime.now().uptimeNanoseconds
      let before = runWriter.payload.bytesWritten
      try runWriter.writeWindow(
        stepStart: stepStart, stepCount: step - stepStart, wallSeconds: now(),
        snap: UnsafePointer(snapPtr + half * B), deadStep: st.deadStep,
        deadDetector: st.deadDetector,
        records: UnsafePointer(staging), capacity: windowSteps
      ) { st.error = $0 }
      if st.didFail, let e = st.error { throw e }
      let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
      let wrote = runWriter.payload.bytesWritten - before
      let raw = B * (step - stepStart) * recordFloats * 4
      fputs(
        String(
          format: "[record] step %d | %d lanes x %d rows | %.1f ms | %.2f MiB -> %.2f MiB (%.1fx)\n",
          step, B, step - stepStart, ms, Double(raw) / 1_048_576,
          Double(wrote) / 1_048_576, wrote > 0 ? Double(raw) / Double(wrote) : 0), stderr)
      onWindow(step)
      let alive = B - st.deadCount
      let wall = now()
      fputs(
        "[fleet \(pipes.label) gpu]" + String(
          format: " step %d alive %d/%d | gpu %.1fs wall %.1fs | %.0f agg steps/s\n",
          step, alive, B, st.gpuSeconds, wall,
          wall > 0 ? Double(step) * Double(B) / wall : 0), stderr)
    }
    let explicitN = e["OAK_STEPS_PER_CB"].flatMap(Int.init)
    if let n = explicitN, n > maxChunkSteps {
      throw err("OAK_STEPS_PER_CB \(n) exceeds the record ring span \(maxChunkSteps).")
    }
    let tuneLock = NSLock()
    var tunedN = max(explicitN ?? 32, 1)
    guard let chunkOrder = q.device.makeEvent() else { throw err("Failed to create Metal event.") }
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
      let boundary = min(((step / windowSteps) + 1) * windowSteps, steps)
      let target = min(boundary, step + n)
      let isWindow = target % windowSteps == 0 || target == steps
      if bufs.environment.needsCapacity(through: target) {
        inflight.wait()
        inflight.wait()
        writer.sync {}
        if let error = st.error { throw error }
        try bufs.environment.prepare(completedStep: step, throughStep: target)
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
  static func benchmark(_ request: BenchmarkRequest, onWindow: @escaping (Int) -> Void) throws {
    let policy = request.policy
    guard ["oak", "random", "ppo"].contains(policy) else {
      throw err("policy must be one of oak,random,ppo (got \(policy)).")
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
    if fm.fileExists(atPath: dir.path),
      !(try fm.contentsOfDirectory(atPath: dir.path).filter {
        $0 != ".DS_Store" && $0 != "identity.json" && $0 != "baselines"
      }).isEmpty
    {
      throw err("\(dir.path) is not empty; refusing to write beside existing data.")
    }
    try fm.createDirectory(at: dir, withIntermediateDirectories: true)
    Self.pendingMeta = [
      "batch": seedValues.count, "seeds": seedValues, "steps": steps,
      "policy": policy,
      "startedAt": ISO8601DateFormatter().string(from: Date()),
      "envSchedule": entries.map { e in
        ["envId": e.envId, "seedStart": e.seedStart, "seedEnd": e.seedEnd] as [String: Any]
      },
    ]
    let envDesc = entries.map { "\($0.envId):\($0.seedEnd - $0.seedStart + 1)" }.joined(
      separator: ",")
    fputs(
      "OaK fleet: policy=\(policy) B=\(seedValues.count) lanes × \(steps) steps on \(envDesc) (dir=\(dir.path))\n",
      stderr)
    try fleetGPU(policy, seedValues, steps, dir, laneEnvs, onWindow: onWindow)
  }
}
