import Darwin
import Foundation
import Metal
import Synchronization

struct RunError: Error, CustomStringConvertible {
  let description: String
}

func err(_ message: String) -> RunError {
  RunError(description: message)
}

func size(_ width: Int) -> MTLSize {
  MTLSize(width: width, height: 1, depth: 1)
}

let schedule = [
  ("HalfCheetah-v4", 0),
  ("FourRooms-cont-v0", 64),
  ("DeepSea-cont-v0", 128),
  ("KeyCorridor-cont-v0", 192),
  ("Acrobot-cont-v0", 256),
  ("MountainCar-cont-v0", 320),
  ("Pendulum-cont-v0", 448),
  ("Chain-cont-v0", 512),
  ("LargeDeepSea-cont-v0", 576),
  ("MultiKeyCorridor-cont-v0", 640),
  ("MemoryChain-cont-v0", 704),
  ("ProceduralMaze-cont-v0", 768),
  ("HighDimMaze-cont-v0", 832),
  ("POMaze-cont-v0", 896),
  ("DelayedChain-cont-v0", 960),
  ("JellyBeanWorld-cont-v0", 1024),
]
let seedsPerEnvironment = 64
let windowSteps = 1024
let chunkSteps = windowSteps / 4
let recordCapacity = 2 * windowSteps

struct LaneSnapshot {
  var paramNonFinite: UInt32
  var rewardSum: Float
  var rho: Float
  var deathStep: UInt32
  var deathDetector: UInt32
  var hashLo: UInt32
  var hashHi: UInt32
}

struct FleetConfig {
  var laneCount: UInt32
  var recordCapacity: UInt32
  var persistentMask: UInt32
  var invObsFloor: Float
}

struct ShapeProbe {
  var featureCount, perceptCount, actionCount: UInt32
  var parameterStride, memoryStride, indexStride: UInt32
  var simdWidth, threadgroupLanes: UInt32
}

struct EnvSpec {
  static let poseLen = 9
  let observations: Int
  let actionCount: Int
  let controlHz: Int
  let envStateWords: Int
  let slots: [Int]
  let persistentEntryWords: Int
  let scratchWords: Int
}

final class GPU {
  let device: MTLDevice
  let queue: MTLCommandQueue
  let library: MTLLibrary

  init() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw err("Metal is unavailable")
    }
    self.device = device
    self.queue = queue
    library = try device.makeLibrary(URL: URL(fileURLWithPath: ".build/OakKernel.metallib"))
  }

  func pipeline(
    _ name: String, constants: MTLFunctionConstantValues? = nil,
    threads: Int? = nil
  ) throws -> MTLComputePipelineState {
    let descriptor = MTLComputePipelineDescriptor()
    if let constants {
      descriptor.computeFunction = try library.makeFunction(name: name, constantValues: constants)
    } else {
      descriptor.computeFunction = library.makeFunction(name: name)
    }
    if let threads { descriptor.maxTotalThreadsPerThreadgroup = threads }
    return try device.makeComputePipelineState(descriptor: descriptor, options: []).0
  }

  func probe(_ name: String, buffer: MTLBuffer, kind: Int? = nil) throws {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(try pipeline(name, constants: kind.map(constants)))
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.dispatchThreads(size(1), threadsPerThreadgroup: size(1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { throw error }
  }

  func constants(_ kind: Int) -> MTLFunctionConstantValues {
    let constants = MTLFunctionConstantValues()
    var value = UInt32(kind)
    constants.setConstantValue(&value, type: .uint, index: 2)
    return constants
  }
}

struct EnvironmentTable {
  let specs: [EnvSpec]
  let constantWords: Int
  let ppoStrides: [Int]

  init(_ gpu: GPU) throws {
    let header = 9
    let stride = 10 + EnvSpec.poseLen
    let table = SharedBuffer<UInt32>(gpu.device, header + schedule.count * stride, 0)
    try gpu.probe("oak_env_table_probe", buffer: table.buffer)
    guard table.values[0] == schedule.count else {
      throw err("Environment table count differs from the schedule")
    }
    specs = try schedule.enumerated().map { kind, entry in
      let row = table.values + header + kind * stride
      let hash = entry.0.utf8.reduce(UInt32(0x811c_9dc5)) { ($0 ^ UInt32($1)) &* 16_777_619 }
      guard row[0] == hash else { throw err("Environment table differs at \(entry.0)") }
      return EnvSpec(
        observations: Int(row[1]), actionCount: Int(row[3]), controlHz: Int(row[4]),
        envStateWords: Int(row[5]), slots: (0..<EnvSpec.poseLen).map { Int(row[8 + $0]) },
        persistentEntryWords: Int(row[8 + EnvSpec.poseLen]),
        scratchWords: Int(row[9 + EnvSpec.poseLen]))
    }
    constantWords = Int(table.values[4])
    ppoStrides = [5, 5, 6, 7, 8].enumerated().map { index, word in
      Int(table.values[word]) * (index == 1 ? 2 : 1)
    }
  }
}

final class SharedBuffer<T> {
  let buffer: MTLBuffer
  let values: UnsafeMutablePointer<T>

  init(_ device: MTLDevice, _ count: Int, _ value: T) {
    buffer = device.makeBuffer(
      length: max(count, 1) * MemoryLayout<T>.stride, options: .storageModeShared)!
    values = buffer.contents().bindMemory(to: T.self, capacity: max(count, 1))
    values.initialize(repeating: value, count: max(count, 1))
  }

  convenience init(_ device: MTLDevice, _ values: [T]) {
    self.init(device, values.count, values[0])
    self.values.update(from: values, count: values.count)
  }
}

final class EnvironmentStorage {
  let fixed: [Int], entries: [Int]
  let offsets: SharedBuffer<UInt32>
  private(set) var arena: SharedBuffer<UInt32>
  private var slots: Int, safeThroughStep: Int
  var mask: UInt32 { UInt32(slots - 1) }

  private static func layout(_ fixed: [Int], _ entries: [Int], _ slots: Int) throws -> (
    [UInt32], Int
  ) {
    var cursor = 0
    let starts = fixed.indices.map { lane -> Int in
      let start = cursor
      cursor += fixed[lane] + entries[lane] * slots
      return start
    }
    guard cursor <= Int(UInt32.max) else {
      throw err("Environment arena exceeds its addressable word count.")
    }
    return (starts.map(UInt32.init), cursor)
  }

  init(_ device: MTLDevice, specs: [EnvSpec], kinds: [UInt32], steps: Int, window: Int) throws {
    fixed = kinds.map {
      let spec = specs[Int($0)]
      return (spec.actionCount + spec.envStateWords + spec.observations + 3) & ~3
    }
    entries = kinds.map { specs[Int($0)].persistentEntryWords }
    guard entries.allSatisfy({ $0 == 0 || $0 == 3 }) else {
      throw err("Persistent entries require two coordinate words and a timestamp.")
    }
    slots = 2
    while slots < 2 * min(steps, window) { slots *= 2 }
    safeThroughStep = entries.contains(3) ? slots / 2 : steps
    let (starts, words) = try Self.layout(fixed, entries, slots)
    offsets = SharedBuffer(device, starts)
    arena = SharedBuffer(device, words, 0)
  }

  func needsCapacity(through step: Int) -> Bool { step > safeThroughStep }
  private static func slot(_ high: UInt32, _ low: UInt32, _ mask: Int) -> Int {
    var key = UInt64(high) << 32 | UInt64(low)
    key ^= key >> 30
    key = key &* 0xbf58_476d_1ce4_e5b9
    key ^= key >> 27
    key = key &* 0x94d0_49bb_1331_11eb
    key ^= key >> 31
    return Int(key & UInt64(mask))
  }

  func prepare(completedStep: Int, throughStep: Int) throws {
    var largest = 0
    for lane in entries.indices where entries[lane] != 0 {
      let history = arena.values + Int(offsets.values[lane]) + fixed[lane]
      let count = (0..<slots).reduce(0) { $0 + (history[3 * $1 + 2] == 0 ? 0 : 1) }
      largest = max(largest, count)
    }
    var capacity = slots
    while capacity / 2 < largest + throughStep - completedStep {
      guard capacity <= Int(UInt32.max) / 2 else {
        throw err("Persistent storage capacity is not addressable.")
      }
      capacity *= 2
    }
    if capacity != slots {
      let (starts, words) = try Self.layout(fixed, entries, capacity)
      let grown = SharedBuffer<UInt32>(arena.buffer.device, words, 0)
      for lane in entries.indices {
        let source = arena.values + Int(offsets.values[lane])
        let destination = grown.values + Int(starts[lane])
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
      offsets.values.update(from: starts, count: starts.count)
      slots = capacity
    }
    safeThroughStep = completedStep + capacity / 2 - largest
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
  static func byKind(_ table: [EnvSpec]) -> [RecordCodec] {
    var cheetah = [RecordField(0, 0.01, false, anchor: true), RecordField(1, 1e-4)]
    for i in 2...8 { cheetah.append(RecordField(i, 1e-4, true)) }
    let classic: [Int: [RecordField]] = [
      4: [
        RecordField(0, 1e-4, true), RecordField(1, 1e-4, true), RecordField(2, 1e-3),
        RecordField(3, 1e-3), RecordField(4),
      ],
      5: [RecordField(0, 1e-4), RecordField(1, 1e-5), RecordField(3)],
      6: [RecordField(0, 1e-4, true), RecordField(1, 1e-3), RecordField(2, 1e-3)],
    ]
    return table.indices.map { kind in
      switch kind {
      case 0: RecordCodec(category: "cheetah", fields: cheetah)
      case 4, 5, 6: RecordCodec(category: "classic", fields: classic[kind]!)
      case 15: RecordCodec(category: "jellyBean", fields: (0..<7).map { RecordField($0) })
      default: grid(table[kind].slots.indices.filter { table[kind].slots[$0] != 0 })
      }
    }
  }
}

let recordFloats = 2 + EnvSpec.poseLen
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
  let codec: RecordCodec
  var closed = false
  var values: [Double] = []

  init(codec: RecordCodec) {
    self.codec = codec
  }

  func encodeChunk(
    _ records: UnsafePointer<Float>, slot: Int, capacity: Int, laneCount: Int,
    stepStart: Int, stepCount: Int
  ) throws -> Data? {
    if closed || stepCount == 0 { return nil }
    let fields = codec.fields
    values = Array(repeating: 0, count: fields.count * stepCount)
    func cell(_ row: Int, _ field: Int) -> Double {
      let offset = ((stepStart + row) % capacity * recordFloats + field) * laneCount
      return Double(records[offset + slot])
    }
    var live = stepCount
    scan: for row in 0..<stepCount {
      if !cell(row, 0).isFinite {
        live = row
        break
      }
      for (column, field) in fields.enumerated() {
        var value = cell(row, 1 + field.idx)
        if field.wrap {
          value -= 2 * Double.pi * ((value + Double.pi) / (2 * Double.pi)).rounded(.down)
        }
        if !value.isFinite || (value / field.scale).rounded().magnitude >= quantizeLimit {
          live = row
          break scan
        }
        values[column * stepCount + row] = value
      }
    }
    closed = live < stepCount
    if live == 0 { return nil }
    var bytes = [UInt8](repeating: 0, count: (live + 7) / 8)
    for row in 0..<live where cell(row, 0) != 0 {
      bytes[row >> 3] |= UInt8(1 << (row & 7))
    }
    bytes.append(contentsOf: repeatElement(0, count: fields.count))
    for (column, field) in fields.enumerated() {
      let offset = column * stepCount
      var previous: Int64 = 0
      var reconstructed = values[offset]
      if field.anchor {
        withUnsafeBytes(of: reconstructed.bitPattern.littleEndian) { bytes.append(contentsOf: $0) }
      }
      for row in 0..<live {
        let value = values[offset + row]
        if field.anchor {
          let delta = Int64(((value - reconstructed) / field.scale).rounded())
          reconstructed += Double(delta) * field.scale
          putVarint(&bytes, delta)
        } else {
          let quantized = Int64((value / field.scale).rounded())
          putVarint(&bytes, quantized &- previous)
          previous = quantized
        }
      }
    }
    let payload = Data(bytes)
    let compressed = try (payload as NSData).compressed(using: .zlib) as Data
    let useCompression = compressed.count < payload.count
    let stored = useCompression ? compressed : payload
    var chunk = Data()
    for word: UInt32 in [
      0x524B_4843, UInt32(live), UInt32(payload.count), UInt32(stored.count),
      useCompression ? 1 << 31 : 0,
    ] { chunk.appendLE(word) }
    chunk.append(stored)
    return chunk
  }
}

struct Group {
  let kind: Int?
  let shape: ShapeProbe?
  let first: Int
  let count: Int
  let packed: Int
  let threads: Int
  let initialize: MTLComputePipelineState
  let step: MTLComputePipelineState

  init(_ gpu: GPU, policy: String, kind: Int?, count: Int) throws {
    self.kind = kind
    self.count = count
    first = (kind ?? 0) * seedsPerEnvironment
    var constants: MTLFunctionConstantValues?
    if let kind {
      let result = SharedBuffer<UInt32>(gpu.device, MemoryLayout<ShapeProbe>.stride / 4, 0)
      try gpu.probe("oak_shape_probe", buffer: result.buffer, kind: kind)
      let shape = result.buffer.contents().load(as: ShapeProbe.self)
      self.shape = shape
      packed = Int(shape.threadgroupLanes)
      threads = packed * Int(shape.simdWidth)
      constants = gpu.constants(kind)
      var laneCount = shape.threadgroupLanes
      constants!.setConstantValue(&laneCount, type: .uint, index: 3)
      if let raw = ProcessInfo.processInfo.environment["OAK_MEGA_STAGE_MASK"] {
        guard var mask = UInt32(raw) else {
          throw err("OAK_MEGA_STAGE_MASK must be an unsigned integer")
        }
        constants!.setConstantValue(&mask, type: .uint, index: 4)
      }
    } else {
      shape = nil
      packed = 1
      threads = 32
    }
    initialize = try gpu.pipeline("\(policy)_fleet_init", constants: constants)
    step = try gpu.pipeline("\(policy)_megastep", constants: constants, threads: threads)
  }
}

final class Fleet {
  let snapshot: MTLComputePipelineState
  let gpu: GPU
  let groups: [Group]
  let resources: [MTLBuffer]
  let environment: EnvironmentStorage
  let common: [(Int, MTLBuffer)]
  let kinds: MTLBuffer
  let records: SharedBuffer<Float>
  let snapshots: SharedBuffer<LaneSnapshot>
  let count = schedule.count * seedsPerEnvironment

  init(_ gpu: GPU, table: EnvironmentTable, policy: String, steps: Int) throws {
    self.gpu = gpu
    snapshot = try gpu.pipeline(policy == "ppo" ? "ppo_fleet_snapshot" : "fleet_snapshot")
    let device = gpu.device
    let laneKinds = schedule.indices.flatMap {
      Array(repeating: UInt32($0), count: seedsPerEnvironment)
    }
    let seeds = schedule.flatMap { entry in
      (entry.1..<(entry.1 + seedsPerEnvironment)).map(UInt32.init)
    }
    kinds = SharedBuffer(device, laneKinds).buffer
    environment = try EnvironmentStorage(
      device, specs: table.specs, kinds: laneKinds, steps: steps, window: windowSteps)
    records = SharedBuffer(device, count * recordCapacity * recordFloats, 0)
    snapshots = SharedBuffer(
      device, 2 * count,
      LaneSnapshot(
        paramNonFinite: 0,
        rewardSum: 0, rho: 0, deathStep: 0, deathDetector: 0, hashLo: 0, hashHi: 0))
    guard let modelURL = Bundle.module.url(forResource: "mjmodels", withExtension: "bin") else {
      throw err("mjmodels.bin is missing")
    }
    let model = try Data(contentsOf: modelURL)
    let modelBuffer = SharedBuffer(device, Array(model)).buffer
    var scratchSize = 0
    let scratchOffsets = laneKinds.map { kind -> UInt32 in
      defer { scratchSize += table.specs[Int(kind)].scratchWords }
      return UInt32(scratchSize)
    }
    let constants = SharedBuffer<Float>(device, table.constantWords, 0)
    try gpu.probe("oak_env_constants_init", buffer: constants.buffer)
    common = [
      (6, SharedBuffer<UInt8>(device, count * 44, 0).buffer),
      (7, SharedBuffer(device, seeds).buffer),
      (16, modelBuffer),
      (17, SharedBuffer<Float>(device, scratchSize, 0).buffer),
      (18, SharedBuffer(device, scratchOffsets).buffer),
      (19, constants.buffer),
    ]
    var strides: [Int] = []
    if policy == "oak" {
      groups = try schedule.indices.map {
        try Group(gpu, policy: policy, kind: $0, count: seedsPerEnvironment)
      }
      let shape = groups[0].shape!
      strides = [shape.parameterStride, shape.memoryStride, shape.indexStride].map(Int.init)
    } else {
      groups = [try Group(gpu, policy: policy, kind: nil, count: seeds.count)]
      if policy == "ppo" { strides = table.ppoStrides }
    }
    resources = strides.map { SharedBuffer<Float>(device, seeds.count * $0, 0).buffer }
  }

  func encode(
    _ encoder: MTLComputeCommandEncoder, group: Group,
    pipeline: MTLComputePipelineState
  ) {
    encoder.setComputePipelineState(pipeline)
    for (index, buffer) in resources.enumerated() {
      let offset = group.first * (buffer.length / count)
      encoder.setBuffer(buffer, offset: offset, index: index)
    }
    for (index, buffer) in common {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBuffer(environment.arena.buffer, offset: 0, index: 14)
    encoder.setBuffer(environment.offsets.buffer, offset: 0, index: 15)
    var config = FleetConfig(
      laneCount: UInt32(count), recordCapacity: UInt32(recordCapacity),
      persistentMask: environment.mask,
      invObsFloor: 1 / Float(2 * (group.shape?.perceptCount ?? 32)))
    encoder.setBytes(&config, length: MemoryLayout<FleetConfig>.stride, index: 8)
  }

  func initialize() throws {
    let command = gpu.queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    for group in groups {
      encode(encoder, group: group, pipeline: group.initialize)
      if let shape = group.shape {
        let alpha = 1 / Float(2 * shape.perceptCount)
        var initial = SIMD4<Float>(
          1 / sqrt(Float(shape.featureCount)), alpha, log(alpha), 1 / Float(shape.actionCount))
        encoder.setBytes(&initial, length: MemoryLayout<SIMD4<Float>>.stride, index: 12)
        var first = UInt32(group.first)
        encoder.setBytes(&first, length: 4, index: 10)
        encoder.dispatchThreads(
          size(group.count),
          threadsPerThreadgroup: size(min(64, group.initialize.maxTotalThreadsPerThreadgroup)))
      } else {
        encoder.setBuffer(kinds, offset: 0, index: 9)
        encoder.dispatchThreads(size(count * 32), threadsPerThreadgroup: size(32))
      }
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { throw error }
  }

  func advance(from start: Int, through end: Int, snapshotHalf: Int?) -> MTLCommandBuffer {
    let command = gpu.queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent)!
    for group in groups {
      encode(encoder, group: group, pipeline: group.step)
      encoder.setBuffer(records.buffer, offset: 0, index: 10)
      var span = SIMD3<UInt32>(
        UInt32(end - start), UInt32(start), UInt32(group.first / group.packed))
      encoder.setBytes(&span, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 11)
      encoder.dispatchThreadgroups(
        size(group.count / group.packed), threadsPerThreadgroup: size(group.threads))
    }
    encoder.endEncoding()
    if let half = snapshotHalf {
      let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent)!
      encode(encoder, group: groups[0], pipeline: snapshot)
      encoder.setBuffer(snapshots.buffer, offset: 0, index: 9)
      var first = UInt32(half * count)
      encoder.setBytes(&first, length: 4, index: 10)
      encoder.dispatchThreads(size(count), threadsPerThreadgroup: size(min(64, snapshot.maxTotalThreadsPerThreadgroup)))
      encoder.endEncoding()
    }
    return command
  }

}

func appendFile(_ url: URL, _ data: Data) throws {
  let fd = open(url.path, O_WRONLY | O_APPEND | O_CREAT, 0o644)
  guard fd >= 0 else { throw err("\(url.path): \(String(cString: strerror(errno)))") }
  defer { close(fd) }
  try data.withUnsafeBytes { raw in
    var done = 0
    while done < raw.count {
      let n = write(fd, raw.baseAddress! + done, raw.count - done)
      guard n > 0 else { throw err("\(url.path): \(String(cString: strerror(errno)))") }
      done += n
    }
  }
}

final class Recording: @unchecked Sendable {
  let root: URL
  let fleet: Fleet
  let encoders: [RecordWriter]
  let failure = Mutex<Error?>(nil)
  var gpuSeconds = 0.0
  var health: [UInt32]

  init(root: URL, fleet: Fleet, codecs: [RecordCodec]) throws {
    self.root = root
    self.fleet = fleet
    encoders = codecs.flatMap { codec in
      (0..<seedsPerEnvironment).map { _ in RecordWriter(codec: codec) }
    }
    health = Array(repeating: UInt32.max, count: fleet.count)
    for directory in ["series", "poses"] {
      try FileManager.default.createDirectory(
        at: root.appendingPathComponent(directory), withIntermediateDirectories: true)
    }
  }

  func check() throws {
    if let error = failure.withLock({ $0 }) { throw error }
  }

  func append<T>(_ name: String, _ values: [T]) throws {
    let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
    try appendFile(root.appendingPathComponent(name), data)
  }

  func replace<T>(_ name: String, _ values: [T]) throws {
    let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
    try data.write(to: root.appendingPathComponent(name), options: .atomic)
  }

  func write(from start: Int, through end: Int, half: Int) throws {
    let count = fleet.count
    let snapshots = fleet.snapshots.values + half * count
    for lane in 0..<count where health[lane] == UInt32.max {
      let snapshot = snapshots[lane]
      if snapshot.deathStep != UInt32.max {
        health[lane] = snapshot.deathStep
      } else if snapshot.paramNonFinite != 0 {
        health[lane] = UInt32(end)
      }
    }
    try append("ends.u32", [UInt32(end)])
    try append("series/rewardSum.f32", (0..<count).map { snapshots[$0].rewardSum })
    try replace("hash.u64", (0..<count).map { UInt64(snapshots[$0].hashHi) << 32 | UInt64(snapshots[$0].hashLo) })
    try replace("health.u32", health)
    DispatchQueue.concurrentPerform(iterations: count) { lane in
      do {
        if let chunk = try encoders[lane].encodeChunk(
          fleet.records.values, slot: lane, capacity: recordCapacity, laneCount: count,
          stepStart: start, stepCount: end - start)
        {
          try appendFile(root.appendingPathComponent("poses/\(lane)"), chunk)
        }
      } catch {
        failure.withLock { if $0 == nil { $0 = error } }
      }
    }
    try check()
  }

  func run(steps: Int) throws {
    try fleet.initialize()
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0)
    let writer = DispatchQueue(label: "recording")
    let inflight = DispatchSemaphore(value: 2)
    var step = 0
    var windowStart = 0
    var half = 0
    while step < steps {
      let boundary = min((step / windowSteps + 1) * windowSteps, steps)
      let end = min(boundary, step + chunkSteps)
      let completesWindow = end == boundary
      if fleet.environment.needsCapacity(through: end) {
        inflight.wait()
        inflight.wait()
        inflight.signal()
        inflight.signal()
        try check()
        try fleet.environment.prepare(completedStep: step, throughStep: end)
      }
      inflight.wait()
      do {
        try check()
      } catch {
        inflight.signal()
        throw error
      }
      let windowHalf = half
      let start = windowStart
      let command = fleet.advance(
        from: step, through: end, snapshotHalf: completesWindow ? half : nil)
      command.addCompletedHandler { command in
        let duration = command.gpuEndTime - command.gpuStartTime
        let gpuError = command.error
        writer.async { [self] in
          defer { inflight.signal() }
          gpuSeconds += duration
          do {
            if let error = gpuError { throw error }
            try check()
            if completesWindow { try write(from: start, through: end, half: windowHalf) }
          } catch {
            failure.withLock { if $0 == nil { $0 = error } }
          }
        }
      }
      command.commit()
      step = end
      if completesWindow {
        half ^= 1
        windowStart = end
      }
    }
    inflight.wait()
    inflight.wait()
    inflight.signal()
    inflight.signal()
    try check()
  }
}

func main() throws {
  let arguments = CommandLine.arguments
  guard arguments.count == 4, ["oak", "ppo", "random"].contains(arguments[1]),
    let steps = Int(arguments[2]), steps > 0, steps <= UInt32.max
  else {
    throw err("Usage: OaK oak|ppo|random <steps> <root>")
  }
  let policy = arguments[1]
  let root = URL(fileURLWithPath: arguments[3], isDirectory: true)
  guard !FileManager.default.fileExists(atPath: root.path) else {
    throw err("Recording already exists: \(root.path)")
  }
  let gpu = try GPU()
  let table = try EnvironmentTable(gpu)
  let codecs = RecordCodec.byKind(table.specs)
  let presets = gpu.library.functionNames.filter { $0.hasPrefix("oak_preset_") }
  guard presets.count == 1 else { throw err("Metal library must declare one geometry preset") }
  let fleet = try Fleet(gpu, table: table, policy: policy, steps: steps)
  let recording = try Recording(root: root, fleet: fleet, codecs: codecs)
  var metadata: [String: Any] = [
    "batch": fleet.count, "steps": steps, "window": windowSteps, "policy": policy,
    "preset": String(presets[0].dropFirst("oak_preset_".count)),
    "startedAt": ISO8601DateFormatter().string(from: Date()),
    "seeds": schedule.flatMap { $0.1..<($0.1 + seedsPerEnvironment) },
    "envSchedule": schedule.map {
      ["envId": $0.0, "seedStart": $0.1, "seedEnd": $0.1 + seedsPerEnvironment - 1] as [String: Any]
    },
    "envCatalog": schedule.enumerated().map { kind, entry in
      [
        "envId": entry.0, "controlHz": table.specs[kind].controlHz,
        "category": codecs[kind].category,
        "fields": codecs[kind].fields.map {
          ["slot": $0.idx, "wrap": $0.wrap, "scale": $0.scale, "anchor": $0.anchor] as [String: Any]
        },
      ] as [String: Any]
    },
  ]
  func save() throws {
    try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys])
      .write(to: root.appendingPathComponent("meta.json"), options: .atomic)
  }
  try save()
  try recording.run(steps: steps)
  metadata["stepsPerSecond"] = Double(steps) * Double(fleet.count) / recording.gpuSeconds
  metadata["finishedAt"] = ISO8601DateFormatter().string(from: Date())
  try save()
}

pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
do {
  try main()
} catch {
  fputs("\(error)\n", stderr)
  exit(1)
}
