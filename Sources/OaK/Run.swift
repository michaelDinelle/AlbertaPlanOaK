import Foundation

enum RunLayout {
  static let headersMagic: UInt32 = 0x544B_414F
  static let payloadMagic: UInt32 = 0x504B_414F
  static let windowSteps = 1024
  static let payloadParts = 10
  static func part(stepStart: Int, steps: Int) -> Int {
    min(stepStart * payloadParts / steps, payloadParts - 1)
  }
  static func payloadName(_ part: Int) -> String { "payload.\(part)" }
}

struct ByteReader {
  let bytes: UnsafeRawBufferPointer
  var off = 0
  init(_ bytes: UnsafeRawBufferPointer) { self.bytes = bytes }
  var remaining: Int { bytes.count - off }
  mutating func take(_ n: Int, _ what: String) throws -> Int {
    guard off + n <= bytes.count else {
      throw err("chunk payload is \(bytes.count) bytes, too short for \(what) at \(off) (+\(n))")
    }
    let at = off
    off += n
    return at
  }
  mutating func u8(_ what: String) throws -> UInt8 { bytes[try take(1, what)] }
  mutating func u32(_ what: String) throws -> UInt32 {
    UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: try take(4, what), as: UInt32.self))
  }
  mutating func u64(_ what: String) throws -> UInt64 {
    UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: try take(8, what), as: UInt64.self))
  }
  mutating func i32(_ what: String) throws -> Int32 {
    Int32(littleEndian: bytes.loadUnaligned(fromByteOffset: try take(4, what), as: Int32.self))
  }
}

struct ColumnSpec {
  let scale: Double
  let anchor: Bool
}

struct DecodedChunk {
  let live: Int
  var rows: [Float]
}

extension RecordWriter {
  static func columnSpecs(_ fields: [RecordField]) -> [ColumnSpec] {
    fields.map { ColumnSpec(scale: $0.scale, anchor: $0.anchor) }
  }

  static func unzig(_ z: UInt64) -> Int64 { Int64(bitPattern: z >> 1) ^ -Int64(bitPattern: z & 1) }

  static func decodeColumns(
    _ payload: UnsafeRawBufferPointer, live: Int, specs: [ColumnSpec]
  ) throws -> (done: [Bool], columns: [[Double]]) {
    var r = ByteReader(payload)
    let doneAt = try r.take((live + 7) >> 3, "the done bitstream")
    var done = [Bool](repeating: false, count: live)
    for i in 0..<live where (payload[doneAt + (i >> 3)] >> UInt8(i & 7)) & 1 != 0 { done[i] = true }
    let metaAt = try r.take(specs.count, "the column meta bytes")
    var columns: [[Double]] = []
    columns.reserveCapacity(specs.count)
    for (k, spec) in specs.enumerated() {
      let meta = payload[metaAt + k]
      let pred = meta & 3
      let rice = meta & 4 != 0
      var anchored = 0.0
      if spec.anchor {
        anchored = Double(bitPattern: try r.u64("anchor seed of column \(k)"))
      }
      var coefs = [Int64](repeating: 0, count: lpcOrder)
      if pred == 2 {
        for j in 0..<lpcOrder { coefs[j] = Int64(try r.i32("LPC coefficients of column \(k)")) }
      }
      var residuals = [Int64](repeating: 0, count: live)
      if rice {
        let riceK = Int(try r.u8("Rice parameter of column \(k)"))
        var cursor = r.off
        var bitPos = 0
        func readBit() throws -> UInt64 {
          guard cursor < payload.count else { throw err("column \(k) ran past the payload") }
          let b = (payload[cursor] >> UInt8(bitPos)) & 1
          bitPos += 1
          if bitPos == 8 { bitPos = 0; cursor += 1 }
          return UInt64(b)
        }
        for i in 0..<live {
          var unary: UInt64 = 0
          while try readBit() == 0 { unary += 1 }
          var low: UInt64 = 0
          for j in 0..<riceK { low |= try readBit() << UInt64(j) }
          residuals[i] = unzig(unary << UInt64(riceK) | low)
        }
        _ = try r.take((bitPos > 0 ? cursor + 1 : cursor) - r.off, "column \(k)")
      } else {
        var cursor = r.off
        for i in 0..<live {
          var shift: UInt64 = 0
          var raw: UInt64 = 0
          while true {
            guard cursor < payload.count else { throw err("column \(k) ran past the payload") }
            let byte = payload[cursor]
            cursor += 1
            raw |= UInt64(byte & 0x7F) << shift
            if byte & 0x80 == 0 { break }
            shift += 7
          }
          residuals[i] = unzig(raw)
        }
        _ = try r.take(cursor - r.off, "column \(k)")
      }
      var q = [Int64](repeating: 0, count: live)
      if pred == 2 {
        var value: Int64 = 0
        for i in 0..<live {
          if i < lpcOrder {
            value &+= residuals[i]
          } else {
            var acc: Int64 = 0
            for j in 0..<lpcOrder { acc &+= coefs[j] &* q[i - 1 - j] }
            value = (acc >> Int64(lpcShift)) &+ residuals[i]
          }
          q[i] = value
        }
      } else {
        var value: Int64 = 0
        var delta: Int64 = 0
        for i in 0..<live {
          let res = residuals[i]
          delta = pred == 1 && i >= 2 ? delta &+ res : res
          value = i == 0 ? delta : value &+ delta
          q[i] = value
        }
      }
      var out = [Double](repeating: 0, count: live)
      if spec.anchor {
        var prev: Int64 = 0
        for i in 0..<live {
          anchored += Double(q[i] &- prev) * spec.scale
          prev = q[i]
          out[i] = anchored
        }
      } else {
        for i in 0..<live { out[i] = Double(q[i]) * spec.scale }
      }
      columns.append(out)
    }
    guard r.remaining == 0 else { throw err("chunk payload has \(r.remaining) trailing bytes") }
    return (done, columns)
  }

  static func chunkPayload(_ chunk: Data) throws -> (live: Int, payload: Data) {
    guard chunk.count >= 20 else { throw err("chunk shorter than its header") }
    let head: [UInt32] = chunk.withUnsafeBytes { raw in
      (0..<5).map { UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 4 * $0, as: UInt32.self)) }
    }
    guard head[0] == chunkMagic else { throw err("chunk magic \(head[0]) is not a record chunk") }
    let live = Int(head[1])
    let payloadBytes = Int(head[2])
    let storedBytes = Int(head[3])
    guard chunk.count == 20 + storedBytes else {
      throw err("chunk stores \(chunk.count - 20) bytes, header claims \(storedBytes)")
    }
    let stored = chunk.subdata(in: 20..<chunk.count)
    let payload = head[4] >> 31 == 0 ? stored : try (stored as NSData).decompressed(using: .zlib) as Data
    guard payload.count == payloadBytes else {
      throw err("chunk payload is \(payload.count) bytes, header claims \(payloadBytes)")
    }
    return (live, payload)
  }

  static func decodeChunk(_ chunk: Data, fields: [RecordField]) throws -> DecodedChunk {
    let (live, payload) = try chunkPayload(chunk)
    let (done, columns) = try payload.withUnsafeBytes {
      try decodeColumns($0, live: live, specs: columnSpecs(fields))
    }
    var rows = [Float](repeating: 0, count: live * recordFloats)
    for i in 0..<live where done[i] { rows[i * recordFloats] = 1 }
    for (k, f) in fields.enumerated() {
      let column = columns[k]
      for i in 0..<live { rows[i * recordFloats + 1 + f.idx] = Float(column[i]) }
    }
    return DecodedChunk(live: live, rows: rows)
  }
}

final class FrameEncoder: @unchecked Sendable {
  let encoders: [RecordWriter]
  var closed: [Bool] { encoders.map(\.closed) }

  init(laneEnvKinds: [UInt32], closed: [Bool]?) throws {
    encoders = try laneEnvKinds.enumerated().map { lane, kind in
      try RecordWriter(envKind: kind, closed: closed?[lane] ?? false)
    }
  }

  func encode(
    _ records: UnsafePointer<Float>, capacity: Int, stepStart: Int, stepCount: Int,
    onError: (Error) -> Void
  ) -> [Data?] {
    let lanes = encoders.count
    var chunks: [Data?] = Array(repeating: nil, count: lanes)
    chunks.withUnsafeMutableBufferPointer { out in
      let base = out.baseAddress!
      DispatchQueue.concurrentPerform(iterations: lanes) { slot in
        do {
          base[slot] = try encoders[slot].encodeChunk(
            records, slot: slot, capacity: capacity, laneCount: lanes,
            stepStart: stepStart, stepCount: stepCount)
        } catch {
          onError(error)
        }
      }
    }
    return chunks
  }
}

func rewardBlocks(
  _ records: UnsafePointer<Float>, capacity: Int, stepStart: Int, stepCount: Int,
  laneCount: Int, envLanes: [Range<Int>]
) throws -> [Data] {
  try envLanes.map { lanes in
    var rows = [Float](repeating: 0, count: stepCount * lanes.count)
    for i in 0..<stepCount {
      let plane = records + ((stepStart + i) % capacity) * recordFloats * laneCount + recordRewardField * laneCount
      for (j, lane) in lanes.enumerated() { rows[i * lanes.count + j] = plane[lane] }
    }
    return try rows.withUnsafeBufferPointer { try (Data(buffer: $0) as NSData).compressed(using: .zlib) as Data }
  }
}

final class PayloadWriter {
  private let parts: [FileHandle]
  private let steps: Int
  private(set) var bytesWritten = 0

  init(root: URL, laneCount: Int, steps: Int) throws {
    self.steps = steps
    var head = Data()
    for v: UInt32 in [RunLayout.payloadMagic, UInt32(laneCount)] { head.appendLE(v) }
    parts = try (0..<RunLayout.payloadParts).map { part in
      let url = root.appendingPathComponent(RunLayout.payloadName(part))
      FileManager.default.createFile(atPath: url.path, contents: nil)
      let handle = try FileHandle(forWritingTo: url)
      try handle.write(contentsOf: head)
      return handle
    }
  }

  func appendFrame(stepStart: Int, stepCount: Int, chunks: [Data?], rewards: [Data]) throws {
    var table = Data()
    var body = Data()
    var chunkCount: UInt32 = 0
    for lane in chunks.indices {
      guard let c = chunks[lane] else { continue }
      chunkCount += 1
      for v: UInt32 in [UInt32(lane), UInt32(c.count)] { table.appendLE(v) }
      body.append(c)
    }
    for block in rewards {
      body.appendLE(UInt32(block.count))
      body.append(block)
    }
    var frame = Data(capacity: 16 + table.count + body.count)
    for v: UInt32 in [UInt32(stepStart), UInt32(stepCount), chunkCount, UInt32(table.count + body.count)] {
      frame.appendLE(v)
    }
    frame.append(table)
    frame.append(body)
    try parts[RunLayout.part(stepStart: stepStart, steps: steps)].write(contentsOf: frame)
    bytesWritten += frame.count
  }

  func close() throws {
    for part in parts { try part.close() }
  }
}

final class HeaderWriter {
  static let baseFields = ["rewardSum", "rho", "deathStep", "deathDetector"]
  let fields: [String]
  private let handle: FileHandle
  private let laneCount: Int
  private let diagCount: Int
  private var block: [Float]

  init(root: URL, laneCount: Int, diagFields: [String]) throws {
    let url = root.appendingPathComponent("headers")
    FileManager.default.createFile(atPath: url.path, contents: nil)
    handle = try FileHandle(forWritingTo: url)
    self.laneCount = laneCount
    diagCount = diagFields.count
    fields = Self.baseFields + diagFields
    block = [Float](repeating: 0, count: fields.count * laneCount)
    let json = try JSONSerialization.data(withJSONObject: ["fields": fields])
    var head = Data()
    for v: UInt32 in [RunLayout.headersMagic, UInt32(laneCount), UInt32(json.count)] { head.appendLE(v) }
    head.append(json)
    try handle.write(contentsOf: head)
  }

  func append(
    snap: UnsafePointer<LaneSnapshot>, deadStep: [Int?], deadDetector: [UInt32?], diag: [Float],
    stepStart: Int, stepCount: Int, wallSeconds: Double
  ) throws {
    precondition(diag.count == diagCount * laneCount, "diag matrix shape drifted from the field list")
    var out = Data(capacity: 16 + fields.count * laneCount * 4)
    for v: UInt32 in [UInt32(stepStart), UInt32(stepCount)] { out.appendLE(v) }
    out.appendLE(Float(wallSeconds).bitPattern)
    var field = 0
    func push(_ get: (Int, LaneSnapshot) -> Float) {
      for l in 0..<laneCount { block[field * laneCount + l] = get(l, snap[l]) }
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

  func close() throws { try handle.close() }
}

final class RunWriter {
  let headers: HeaderWriter
  let frames: FrameEncoder
  let payload: PayloadWriter
  let laneCount: Int
  let envLanes: [Range<Int>]
  var closed: [Bool] { frames.closed }

  init(root: URL, laneCount: Int, steps: Int, laneEnvKinds: [UInt32], diagFields: [String], closed: [Bool]?) throws {
    headers = try HeaderWriter(root: root, laneCount: laneCount, diagFields: diagFields)
    frames = try FrameEncoder(laneEnvKinds: laneEnvKinds, closed: closed)
    payload = try PayloadWriter(root: root, laneCount: laneCount, steps: steps)
    self.laneCount = laneCount
    envLanes = Self.envLanes(laneEnvKinds)
  }

  static func envLanes(_ laneEnvKinds: [UInt32]) -> [Range<Int>] {
    var out: [Range<Int>] = []
    var lane = 0
    while lane < laneEnvKinds.count {
      var end = lane
      while end < laneEnvKinds.count && laneEnvKinds[end] == laneEnvKinds[lane] { end += 1 }
      out.append(lane..<end)
      lane = end
    }
    return out
  }

  func writeWindow(
    stepStart: Int, stepCount: Int, wallSeconds: Double,
    snap: UnsafePointer<LaneSnapshot>, deadStep: [Int?], deadDetector: [UInt32?], diag: [Float],
    records: UnsafePointer<Float>, capacity: Int, onError: (Error) -> Void
  ) throws {
    try headers.append(
      snap: snap, deadStep: deadStep, deadDetector: deadDetector, diag: diag,
      stepStart: stepStart, stepCount: stepCount, wallSeconds: wallSeconds)
    let chunks = frames.encode(
      records, capacity: capacity, stepStart: stepStart, stepCount: stepCount, onError: onError)
    let rewards = try rewardBlocks(
      records, capacity: capacity, stepStart: stepStart, stepCount: stepCount,
      laneCount: laneCount, envLanes: envLanes)
    try payload.appendFrame(stepStart: stepStart, stepCount: stepCount, chunks: chunks, rewards: rewards)
  }

  func close() throws {
    try headers.close()
    try payload.close()
  }
}

struct RunMeta {
  let policy: String
  let steps: Int
  let seeds: [Int]
  let laneEnvKinds: [UInt32]
  let envLanes: [(envId: String, lanes: Range<Int>)]
  let raw: [String: Any]

  init(root: URL) throws {
    let data = try Data(contentsOf: root.appendingPathComponent("meta.json"))
    guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw err("\(root.path): meta.json is not an object")
    }
    guard let policy = obj["policy"] as? String, let steps = obj["steps"] as? Int,
      let seeds = obj["seeds"] as? [Int], let schedule = obj["envSchedule"] as? [[String: Any]]
    else { throw err("\(root.path): meta.json lacks policy, steps, seeds, or envSchedule") }
    var kinds = [UInt32](repeating: 0, count: seeds.count)
    var envLanes: [(String, Range<Int>)] = []
    var lane = 0
    for entry in schedule {
      guard let envId = entry["envId"] as? String, let start = entry["seedStart"] as? Int,
        let end = entry["seedEnd"] as? Int, end >= start, lane + (end - start + 1) <= seeds.count
      else { throw err("\(root.path): envSchedule entry does not fit the fleet") }
      let count = end - start + 1
      let kind = try Main.envKind(envId)
      for l in lane..<(lane + count) { kinds[l] = kind }
      envLanes.append((envId, lane..<(lane + count)))
      lane += count
    }
    guard lane == seeds.count else { throw err("\(root.path): envSchedule covers \(lane) of \(seeds.count) lanes") }
    self.policy = policy
    self.steps = steps
    self.seeds = seeds
    laneEnvKinds = kinds
    self.envLanes = envLanes
    raw = obj
  }
}

struct WindowRow {
  let stepStart: Int
  let stepCount: Int
  let wallSeconds: Float
  let offset: Int
  let compressedBytes: Int
}

struct FrameEntry {
  let stepStart: Int
  let stepCount: Int
  let part: Int
  let chunks: [Int: (offset: Int, length: Int)]
  let rewards: [(offset: Int, length: Int)]
}

final class RunReader {
  let root: URL
  let meta: RunMeta
  let fields: [String]
  let laneCount: Int
  private(set) var windows: [WindowRow] = []
  private(set) var frames: [FrameEntry] = []
  private var headersScanned = 0
  private var partsScanned: [Int]
  private var rowCache: [Int: [Float]] = [:]
  private var chunkCache: [Int: DecodedChunk] = [:]
  private var chunkOrder: [Int] = []
  private let chunkCacheLimit = 4096
  private var rewardCache: [Int: [Float]] = [:]
  private var rewardOrder: [Int] = []
  private let rewardCacheLimit = 1024
  private let codecs: [RecordCodec]

  init(root: URL) throws {
    self.root = root
    meta = try RunMeta(root: root)
    codecs = try meta.laneEnvKinds.map { try RecordCodec.table($0) }
    let url = root.appendingPathComponent("headers")
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    guard let head = try handle.read(upToCount: 12), head.count == 12 else { throw err("\(url.path): no header") }
    let magic = head.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) }
    let lanes = head.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self)) }
    let jsonLen = head.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self)) }
    guard magic == RunLayout.headersMagic else { throw err("\(url.path): not a run headers file") }
    guard Int(lanes) == meta.seeds.count else { throw err("\(url.path): lane count contradicts meta.json") }
    guard let json = try handle.read(upToCount: Int(jsonLen)), json.count == Int(jsonLen),
      let obj = try JSONSerialization.jsonObject(with: json) as? [String: Any],
      let fields = obj["fields"] as? [String]
    else { throw err("\(url.path): field list unreadable") }
    self.fields = fields
    laneCount = Int(lanes)
    headersScanned = 12 + Int(jsonLen)
    partsScanned = Array(repeating: 8, count: RunLayout.payloadParts)
    try refresh()
  }

  private func fileSize(_ url: URL) -> Int {
    (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
  }

  func refresh() throws {
    let headersURL = root.appendingPathComponent("headers")
    let size = fileSize(headersURL)
    if size > headersScanned {
      let handle = try FileHandle(forReadingFrom: headersURL)
      defer { try? handle.close() }
      while headersScanned + 16 <= size {
        try handle.seek(toOffset: UInt64(headersScanned))
        guard let head = try handle.read(upToCount: 16), head.count == 16 else { break }
        let v: [UInt32] = head.withUnsafeBytes { raw in
          (0..<4).map { UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 4 * $0, as: UInt32.self)) }
        }
        let comp = Int(v[3])
        guard headersScanned + 16 + comp <= size else { break }
        windows.append(WindowRow(
          stepStart: Int(v[0]), stepCount: Int(v[1]), wallSeconds: Float(bitPattern: v[2]),
          offset: headersScanned + 16, compressedBytes: comp))
        headersScanned += 16 + comp
      }
    }
    for part in 0..<RunLayout.payloadParts {
      let url = root.appendingPathComponent(RunLayout.payloadName(part))
      let partSize = fileSize(url)
      guard partSize > partsScanned[part] else { continue }
      let handle = try FileHandle(forReadingFrom: url)
      defer { try? handle.close() }
      if partsScanned[part] == 8 {
        try handle.seek(toOffset: 0)
        guard let head = try handle.read(upToCount: 8), head.count == 8 else { continue }
        let magic = head.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) }
        guard magic == RunLayout.payloadMagic else { throw err("\(url.path): not a run payload file") }
      }
      while partsScanned[part] + 16 <= partSize {
        try handle.seek(toOffset: UInt64(partsScanned[part]))
        guard let head = try handle.read(upToCount: 16), head.count == 16 else { break }
        let v: [UInt32] = head.withUnsafeBytes { raw in
          (0..<4).map { UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 4 * $0, as: UInt32.self)) }
        }
        let chunkCount = Int(v[2])
        let frameBytes = Int(v[3])
        guard partsScanned[part] + 16 + frameBytes <= partSize else { break }
        guard let table = try handle.read(upToCount: chunkCount * 8), table.count == chunkCount * 8 else { break }
        var chunks: [Int: (offset: Int, length: Int)] = [:]
        var cursor = partsScanned[part] + 16 + chunkCount * 8
        table.withUnsafeBytes { raw in
          for c in 0..<chunkCount {
            let lane = Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 8 * c, as: UInt32.self)))
            let len = Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 8 * c + 4, as: UInt32.self)))
            chunks[lane] = (cursor, len)
            cursor += len
          }
        }
        var rewards: [(offset: Int, length: Int)] = []
        for _ in meta.envLanes {
          try handle.seek(toOffset: UInt64(cursor))
          guard let lenData = try handle.read(upToCount: 4), lenData.count == 4 else { throw err("\(url.path): reward block table truncated") }
          let len = Int(lenData.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) })
          rewards.append((cursor + 4, len))
          cursor += 4 + len
        }
        guard cursor == partsScanned[part] + 16 + frameBytes else { throw err("\(url.path): frame at \(partsScanned[part]) is misassembled") }
        frames.append(FrameEntry(stepStart: Int(v[0]), stepCount: Int(v[1]), part: part, chunks: chunks, rewards: rewards))
        partsScanned[part] += 16 + frameBytes
      }
    }
    frames.sort { $0.stepStart < $1.stepStart }
  }

  var maxStep: Int { windows.last.map { $0.stepStart + $0.stepCount } ?? 0 }
  var complete: Bool { meta.raw["finishedAt"] != nil && maxStep >= meta.steps }

  func row(_ index: Int) throws -> [Float] {
    if let cached = rowCache[index] { return cached }
    let w = windows[index]
    let handle = try FileHandle(forReadingFrom: root.appendingPathComponent("headers"))
    defer { try? handle.close() }
    try handle.seek(toOffset: UInt64(w.offset))
    guard let comp = try handle.read(upToCount: w.compressedBytes), comp.count == w.compressedBytes else {
      throw err("headers row \(index) truncated")
    }
    let block = try (comp as NSData).decompressed(using: .zlib) as Data
    guard block.count == fields.count * laneCount * 4 else {
      throw err("headers row \(index) is \(block.count) bytes, expected \(fields.count * laneCount * 4)")
    }
    let floats = block.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    rowCache[index] = floats
    return floats
  }

  func presented(_ index: Int) throws -> [Float] {
    var floats = try row(index)
    guard let death = fields.firstIndex(of: "deathStep") else { return floats }
    for l in 0..<laneCount where !floats[death * laneCount + l].isNaN {
      floats[death * laneCount + l] = Float(floats[death * laneCount + l].bitPattern)
    }
    return floats
  }

  func series(_ column: String) throws -> Data {
    guard let f = fields.firstIndex(of: column) else { throw err("no column \(column)") }
    var out = Data(capacity: 8 + windows.count * (8 + laneCount * 4))
    for v: UInt32 in [UInt32(windows.count), UInt32(laneCount)] { out.appendLE(v) }
    for w in windows { out.appendLE(UInt32(w.stepStart + w.stepCount)) }
    for w in windows { out.appendLE(w.wallSeconds.bitPattern) }
    for index in windows.indices {
      let floats = try presented(index)
      floats.withUnsafeBufferPointer { buf in
        out.append(Data(buffer: UnsafeBufferPointer(rebasing: buf[(f * laneCount)..<((f + 1) * laneCount)])))
      }
    }
    return out
  }

  private func chunk(_ frameIndex: Int, lane: Int) throws -> DecodedChunk? {
    let frame = frames[frameIndex]
    guard let entry = frame.chunks[lane] else { return nil }
    let key = frameIndex * laneCount + lane
    if let cached = chunkCache[key] { return cached }
    let handle = try FileHandle(forReadingFrom: root.appendingPathComponent(RunLayout.payloadName(frame.part)))
    defer { try? handle.close() }
    try handle.seek(toOffset: UInt64(entry.offset))
    guard let data = try handle.read(upToCount: entry.length), data.count == entry.length else {
      throw err("chunk for lane \(lane) at step \(frame.stepStart) truncated")
    }
    var decoded = try RecordWriter.decodeChunk(data, fields: codecs[lane].fields)
    let env = meta.envLanes.firstIndex { $0.lanes.contains(lane) }!
    let block = try rewardBlock(frameIndex, env: env)
    let width = meta.envLanes[env].lanes.count
    let column = lane - meta.envLanes[env].lanes.lowerBound
    for i in 0..<decoded.live { decoded.rows[i * recordFloats + recordRewardField] = block[i * width + column] }
    chunkCache[key] = decoded
    chunkOrder.append(key)
    if chunkOrder.count > chunkCacheLimit {
      chunkCache.removeValue(forKey: chunkOrder.removeFirst())
    }
    return decoded
  }

  func rewardBlock(_ frameIndex: Int, env: Int) throws -> [Float] {
    let key = frameIndex * meta.envLanes.count + env
    if let cached = rewardCache[key] { return cached }
    let frame = frames[frameIndex]
    let entry = frame.rewards[env]
    let handle = try FileHandle(forReadingFrom: root.appendingPathComponent(RunLayout.payloadName(frame.part)))
    defer { try? handle.close() }
    try handle.seek(toOffset: UInt64(entry.offset))
    guard let comp = try handle.read(upToCount: entry.length), comp.count == entry.length else {
      throw err("reward block at step \(frame.stepStart) truncated")
    }
    let raw = try (comp as NSData).decompressed(using: .zlib) as Data
    let width = meta.envLanes[env].lanes.count
    guard raw.count == frame.stepCount * width * 4 else { throw err("reward block at step \(frame.stepStart) has the wrong shape") }
    let rows = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    rewardCache[key] = rows
    rewardOrder.append(key)
    if rewardOrder.count > rewardCacheLimit { rewardCache.removeValue(forKey: rewardOrder.removeFirst()) }
    return rows
  }

  func available(lane: Int) throws -> Int {
    for (index, frame) in frames.enumerated() {
      guard frame.chunks[lane] != nil else { return frame.stepStart }
      if index == frames.count - 1 {
        guard let decoded = try chunk(index, lane: lane) else { return frame.stepStart }
        return frame.stepStart + decoded.live
      }
    }
    return 0
  }

  private func frameIndex(containing step: Int) -> Int? {
    var lo = 0, hi = frames.count
    while lo < hi {
      let mid = (lo + hi) / 2
      if frames[mid].stepStart + frames[mid].stepCount <= step { lo = mid + 1 } else { hi = mid }
    }
    guard lo < frames.count, frames[lo].stepStart <= step else { return nil }
    return lo
  }

  func frames(lane: Int, from: Int, count: Int) throws -> Data {
    let end = min(from + count, try available(lane: lane))
    var out = Data()
    guard end > from, let first = frameIndex(containing: from) else { return out }
    var step = from
    var index = first
    while step < end, index < frames.count {
      let frame = frames[index]
      guard let decoded = try chunk(index, lane: lane) else { break }
      let lo = step - frame.stepStart
      let hi = min(decoded.live, end - frame.stepStart)
      guard hi > lo else { break }
      decoded.rows.withUnsafeBufferPointer { rows in
        out.append(Data(buffer: UnsafeBufferPointer(rebasing: rows[(lo * recordFloats)..<(hi * recordFloats)])))
      }
      step = frame.stepStart + hi
      index += 1
    }
    return out
  }

  func rewards(envId: String, from: Int, count: Int) throws -> Data {
    guard let env = meta.envLanes.firstIndex(where: { $0.envId == envId }) else { throw err("no environment \(envId)") }
    let width = meta.envLanes[env].lanes.count
    let end = min(from + count, frames.last.map { $0.stepStart + $0.stepCount } ?? 0)
    var out = Data()
    guard end > from, let first = frameIndex(containing: from) else { return out }
    var index = first
    var step = from
    while index < frames.count, step < end {
      let frame = frames[index]
      let rows = try rewardBlock(index, env: env)
      let lo = step - frame.stepStart
      let hi = min(frame.stepCount, end - frame.stepStart)
      rows.withUnsafeBufferPointer { buf in
        out.append(Data(buffer: UnsafeBufferPointer(rebasing: buf[(lo * width)..<(hi * width)])))
      }
      step = frame.stepStart + hi
      index += 1
    }
    return out
  }
}

struct CheckpointRegion {
  let name: String
  let base: UnsafeMutableRawPointer
  let bytes: Int
}

enum Checkpoint {
  static let magic: UInt32 = 0x504B_4B4F

  static func save(_ url: URL, step: Int, laneCount: Int, regions: [CheckpointRegion]) throws {
    var out = Data()
    out.appendLE(magic)
    out.appendLE(UInt64(step))
    out.appendLE(UInt32(laneCount))
    for region in regions {
      let name = Data(region.name.utf8)
      out.appendLE(UInt32(name.count))
      out.append(name)
      out.appendLE(UInt64(region.bytes))
      out.append(Data(bytes: region.base, count: region.bytes))
    }
    try out.write(to: url, options: .atomic)
  }

  static func load(_ url: URL) throws -> (step: Int, laneCount: Int, regions: [String: Data]) {
    let data = try Data(contentsOf: url)
    var regions: [String: Data] = [:]
    let (step, laneCount) = try data.withUnsafeBytes { raw -> (Int, Int) in
      var r = ByteReader(raw)
      guard try r.u32("magic") == magic else { throw err("\(url.path): not a checkpoint") }
      let step = Int(try r.u64("step"))
      let lanes = Int(try r.u32("lane count"))
      while r.remaining > 0 {
        let nameLen = Int(try r.u32("region name length"))
        let nameAt = try r.take(nameLen, "region name")
        let name = String(decoding: UnsafeRawBufferPointer(rebasing: raw[nameAt..<(nameAt + nameLen)]), as: UTF8.self)
        let bytes = Int(try r.u64("region size"))
        let at = try r.take(bytes, "region \(name)")
        regions[name] = Data(raw[at..<(at + bytes)])
      }
      return (step, lanes)
    }
    return (step, laneCount, regions)
  }

  static func restore(_ saved: [String: Data], into regions: [CheckpointRegion]) throws {
    for region in regions {
      guard let data = saved[region.name] else { throw err("checkpoint lacks region \(region.name)") }
      guard data.count == region.bytes else {
        throw err("checkpoint region \(region.name) is \(data.count) bytes, engine expects \(region.bytes)")
      }
      data.withUnsafeBytes { region.base.copyMemory(from: $0.baseAddress!, byteCount: region.bytes) }
    }
  }
}

struct LoopRestore {
  let step: Int
  let deadStep: [Int?]
  let deadDetector: [UInt32?]
  let closed: [Bool]

  init(step: Int, laneCount: Int, regions: [String: Data]) throws {
    guard let dead = regions["deadStep"], let detector = regions["deadDetector"], let closed = regions["closed"],
      dead.count == laneCount * 4, detector.count == laneCount * 4, closed.count == laneCount
    else { throw err("checkpoint lacks loop state for \(laneCount) lanes") }
    self.step = step
    let deadWords = dead.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
    let detectorWords = detector.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
    deadStep = deadWords.map { $0 == UInt32.max ? nil : Int($0) }
    deadDetector = detectorWords.map { $0 == UInt32.max ? nil : $0 }
    self.closed = closed.map { $0 != 0 }
  }
}
