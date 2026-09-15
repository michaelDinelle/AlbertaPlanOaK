import Foundation

final class FrameEncoder: @unchecked Sendable {
  let encoders: [RecordWriter]

  init(laneEnvKinds: [UInt32]) throws {
    encoders = try laneEnvKinds.map { try RecordWriter(envKind: $0) }
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

final class RunWriter {
  static let series = ["rewardSum", "rho", "deathStep", "deathDetector"]
  let root: URL
  let frames: FrameEncoder
  let laneCount: Int
  let envLanes: [(envId: String, lanes: Range<Int>)]
  private(set) var bytesWritten = 0

  init(root: URL, laneEnvKinds: [UInt32]) throws {
    self.root = root
    frames = try FrameEncoder(laneEnvKinds: laneEnvKinds)
    laneCount = laneEnvKinds.count
    envLanes = Self.envLanes(laneEnvKinds).map { (Main.envIdsByKind[Int(laneEnvKinds[$0.lowerBound])], $0) }
    for d in ["series", "rewards", "poses"] {
      try FileManager.default.createDirectory(at: root.appendingPathComponent(d), withIntermediateDirectories: true)
    }
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
    snap: UnsafePointer<LaneSnapshot>, deadStep: [Int?], deadDetector: [UInt32?],
    records: UnsafePointer<Float>, capacity: Int, onError: (Error) -> Void
  ) throws {
    var end = Data(), wall = Data()
    end.appendLE(UInt32(stepStart + stepCount))
    wall.appendLE(Float(wallSeconds).bitPattern)
    try appendFile(root.appendingPathComponent("ends.u32"), end)
    try appendFile(root.appendingPathComponent("wall.f32"), wall)
    let columns: [[Float]] = [
      (0..<laneCount).map { snap[$0].rewardSum },
      (0..<laneCount).map { snap[$0].rho },
      deadStep.map { $0.map(Float.init) ?? .nan },
      deadDetector.map { $0.map(Float.init) ?? .nan },
    ]
    for (name, column) in zip(Self.series, columns) {
      try appendFile(root.appendingPathComponent("series/\(name).f32"), column.withUnsafeBufferPointer { Data(buffer: $0) })
    }
    for (envId, lanes) in envLanes {
      var rows = [Float](repeating: 0, count: stepCount * lanes.count)
      for i in 0..<stepCount {
        let plane = records + ((stepStart + i) % capacity) * recordFloats * laneCount + recordRewardField * laneCount
        for (j, lane) in lanes.enumerated() { rows[i * lanes.count + j] = plane[lane] }
      }
      try appendFile(root.appendingPathComponent("rewards/\(envId).f32"), rows.withUnsafeBufferPointer { Data(buffer: $0) })
    }
    let chunks = frames.encode(records, capacity: capacity, stepStart: stepStart, stepCount: stepCount, onError: onError)
    for (lane, chunk) in chunks.enumerated() {
      guard let chunk else { continue }
      try appendFile(root.appendingPathComponent("poses/\(lane)"), chunk)
      bytesWritten += chunk.count
    }
  }
}
