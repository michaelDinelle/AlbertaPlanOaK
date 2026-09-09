import CryptoKit
import Foundation
import IOKit.pwr_mgt
import Network

struct HTTPRequest {
  let method: String
  let path: String
  let query: [String: String]
  let headers: [String: String]
  let body: Data
}

struct HTTPResponse {
  var status = 200
  var contentType = "application/octet-stream"
  var body = Data()
  static func json(_ obj: Any, status: Int = 200) throws -> HTTPResponse {
    HTTPResponse(
      status: status, contentType: "application/json; charset=utf-8",
      body: try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]))
  }
  static func fail(_ status: Int, _ message: String) -> HTTPResponse {
    HTTPResponse(status: status, contentType: "text/plain; charset=utf-8", body: Data(message.utf8))
  }
  static func bytes(_ data: Data) -> HTTPResponse {
    HTTPResponse(status: data.isEmpty ? 204 : 200, body: data)
  }
}

final class Connection: @unchecked Sendable {
  let nw: NWConnection
  let daemon: Daemon
  private var buffer = Data()
  private var websocket = false
  var follows: Set<String> = []

  init(_ nw: NWConnection, daemon: Daemon) {
    self.nw = nw
    self.daemon = daemon
  }

  func start() {
    nw.stateUpdateHandler = { [weak self] state in
      if case .failed = state { self?.close() }
      if case .cancelled = state { self?.daemon.forget(self!) }
    }
    nw.start(queue: daemon.queue)
    receive()
  }

  func close() {
    nw.cancel()
  }

  private func receive() {
    nw.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, complete, error in
      guard let self else { return }
      if let data { self.buffer.append(data) }
      if error != nil || (complete && data == nil) {
        self.close()
        return
      }
      do {
        try self.drain()
      } catch {
        fputs("[serve] \(error)\n", stderr)
        self.close()
        return
      }
      self.receive()
    }
  }

  private func drain() throws {
    if websocket {
      while try websocketFrame() {}
      return
    }
    while let request = try httpRequest() {
      if request.headers["upgrade"]?.lowercased() == "websocket", let key = request.headers["sec-websocket-key"] {
        let accept = Data(Insecure.SHA1.hash(data: Data((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").utf8))).base64EncodedString()
        let head = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: \(accept)\r\n\r\n"
        nw.send(content: Data(head.utf8), completion: .contentProcessed { _ in })
        websocket = true
        daemon.attach(self)
        while try websocketFrame() {}
        return
      }
      let response = daemon.route(request)
      var head = "HTTP/1.1 \(response.status) \(Self.reason(response.status))\r\n"
      head += "Content-Type: \(response.contentType)\r\nContent-Length: \(response.body.count)\r\n"
      head += "Cache-Control: no-store\r\nConnection: keep-alive\r\n\r\n"
      var out = Data(head.utf8)
      if request.method != "HEAD" { out.append(response.body) }
      nw.send(content: out, completion: .contentProcessed { _ in })
    }
  }

  static func reason(_ status: Int) -> String {
    switch status {
    case 200: return "OK"
    case 202: return "Accepted"
    case 204: return "No Content"
    case 400: return "Bad Request"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 409: return "Conflict"
    default: return "Internal Server Error"
    }
  }

  private func httpRequest() throws -> HTTPRequest? {
    guard let headEnd = buffer.range(of: Data("\r\n\r\n".utf8)) else { return nil }
    let headText = String(decoding: buffer[buffer.startIndex..<headEnd.lowerBound], as: UTF8.self)
    var lines = headText.components(separatedBy: "\r\n")
    let requestLine = lines.removeFirst().split(separator: " ")
    guard requestLine.count >= 2 else { throw err("bad request line") }
    var headers: [String: String] = [:]
    for line in lines {
      guard let colon = line.firstIndex(of: ":") else { continue }
      headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
    }
    let length = Int(headers["content-length"] ?? "0") ?? 0
    let bodyStart = headEnd.upperBound
    guard buffer.count - (bodyStart - buffer.startIndex) >= length else { return nil }
    let body = buffer[bodyStart..<(bodyStart + length)]
    buffer.removeSubrange(buffer.startIndex..<(bodyStart + length))
    let target = String(requestLine[1])
    let parts = target.split(separator: "?", maxSplits: 1)
    var query: [String: String] = [:]
    if parts.count == 2 {
      for pair in parts[1].split(separator: "&") {
        let kv = pair.split(separator: "=", maxSplits: 1)
        query[String(kv[0])] = kv.count == 2 ? String(kv[1]).removingPercentEncoding ?? "" : ""
      }
    }
    return HTTPRequest(
      method: String(requestLine[0]), path: String(parts[0]),
      query: query, headers: headers, body: Data(body))
  }

  private func websocketFrame() throws -> Bool {
    guard buffer.count >= 2 else { return false }
    let b0 = buffer[buffer.startIndex]
    let b1 = buffer[buffer.startIndex + 1]
    let opcode = b0 & 0x0F
    var length = Int(b1 & 0x7F)
    var at = buffer.startIndex + 2
    if length == 126 {
      guard buffer.count >= 4 else { return false }
      length = Int(buffer[at]) << 8 | Int(buffer[at + 1])
      at += 2
    } else if length == 127 {
      guard buffer.count >= 10 else { return false }
      length = 0
      for i in 0..<8 { length = length << 8 | Int(buffer[at + i]) }
      at += 8
    }
    let masked = b1 & 0x80 != 0
    guard buffer.endIndex - at >= (masked ? 4 : 0) + length else { return false }
    var payload = Data(count: length)
    if masked {
      let key = [buffer[at], buffer[at + 1], buffer[at + 2], buffer[at + 3]]
      at += 4
      for i in 0..<length { payload[i] = buffer[at + i] ^ key[i & 3] }
    } else {
      payload = Data(buffer[at..<(at + length)])
    }
    buffer.removeSubrange(buffer.startIndex..<(at + length))
    switch opcode {
    case 1: daemon.command(String(decoding: payload, as: UTF8.self), from: self)
    case 8: close()
    case 9: send(opcode: 10, payload)
    default: break
    }
    return true
  }

  func send(opcode: UInt8, _ payload: Data) {
    var frame = Data([0x80 | opcode])
    if payload.count < 126 {
      frame.append(UInt8(payload.count))
    } else if payload.count < 1 << 16 {
      frame.append(126)
      frame.append(UInt8(payload.count >> 8))
      frame.append(UInt8(payload.count & 0xFF))
    } else {
      frame.append(127)
      for i in stride(from: 56, through: 0, by: -8) { frame.append(UInt8((payload.count >> i) & 0xFF)) }
    }
    frame.append(payload)
    nw.send(content: frame, completion: .contentProcessed { _ in })
  }

  func sendText(_ text: String) { send(opcode: 1, Data(text.utf8)) }
  func sendBinary(_ data: Data) { send(opcode: 2, data) }
}

final class LiveControl: @unchecked Sendable {
  private let lock = NSLock()
  private var signals: [(lane: Int, word: Int, value: UInt32)] = []
  private var saveRequested = false
  var onWindow: ((Int) -> Void)?

  func signal(lane: Int, word: Int, value: UInt32) {
    lock.lock()
    signals.append((lane, word, value))
    lock.unlock()
  }
  func requestSave() {
    lock.lock()
    saveRequested = true
    lock.unlock()
  }
  var pending: Bool {
    lock.lock()
    defer { lock.unlock() }
    return !signals.isEmpty || saveRequested
  }
  func take(allowSave: Bool) -> (signals: [(lane: Int, word: Int, value: UInt32)], save: Bool) {
    lock.lock()
    defer { lock.unlock() }
    let out = (signals, saveRequested && allowSave)
    signals = []
    if allowSave { saveRequested = false }
    return out
  }
}

final class RunningJob {
  let id: String
  let root: URL
  let control = LiveControl()
  init(id: String, root: URL) {
    self.id = id
    self.root = root
  }
}

final class Daemon: @unchecked Sendable {
  let bench: URL
  let dist: URL
  let port: UInt16
  let queue = DispatchQueue(label: "oak.daemon")
  private var readers: [String: RunReader] = [:]
  private var connections: Set<ObjectIdentifier> = []
  private var sockets: [ObjectIdentifier: Connection] = [:]
  private(set) var running: RunningJob?
  private var listener: NWListener?
  let binarySha: String

  init(bench: URL, dist: URL, port: UInt16) throws {
    self.bench = bench
    self.dist = dist
    self.port = port
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    let bytes = try Data(contentsOf: executable)
    binarySha = bytes.withUnsafeBytes { sha256Hex($0) }
  }

  func run() throws -> Never {
    let params = NWParameters.tcp
    params.allowLocalEndpointReuse = true
    params.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
    let listener = try NWListener(using: params)
    listener.newConnectionHandler = { [self] nw in
      let c = Connection(nw, daemon: self)
      connections.insert(ObjectIdentifier(c))
      sockets[ObjectIdentifier(c)] = c
      c.start()
    }
    let ready = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var failure: Error?
    listener.stateUpdateHandler = { state in
      switch state {
      case .ready: ready.signal()
      case .failed(let e): failure = e; ready.signal()
      default: break
      }
    }
    listener.start(queue: queue)
    ready.wait()
    if let failure { throw err("port \(port) is taken: \(failure)") }
    self.listener = listener
    fputs("OaK on http://127.0.0.1:\(port)\n", stderr)
    if ProcessInfo.processInfo.environment["OAK_BENCH_POLICY"] != nil {
      let request = try BenchmarkRequest.fromEnvironment()
      try queue.sync { try start(request) }
    }
    dispatchMain()
  }

  func forget(_ c: Connection) {
    connections.remove(ObjectIdentifier(c))
    sockets.removeValue(forKey: ObjectIdentifier(c))
  }
  func attach(_ c: Connection) {}

  private func reader(_ id: String) throws -> RunReader {
    if let r = readers[id] {
      try r.refresh()
      return r
    }
    let r = try RunReader(root: try runRoot(id))
    readers[id] = r
    return r
  }

  private func runRoot(_ id: String) throws -> URL {
    let stored = id.split(separator: "/").map(String.init)
    if stored.count == 3, stored[0] == "baselines", ["ppo", "random"].contains(stored[2]), !stored[1].hasPrefix(".") {
      let root = bench.appendingPathComponent(id, isDirectory: true)
      guard FileManager.default.fileExists(atPath: root.appendingPathComponent("meta.json").path) else {
        throw err("unknown run \(id)")
      }
      return root
    }
    let parts = id.split(separator: ":", maxSplits: 1).map(String.init)
    guard let name = parts.first, !name.isEmpty, !name.contains("/"), !name.hasPrefix(".") else {
      throw err("bad run id")
    }
    var root = bench.appendingPathComponent(name, isDirectory: true)
    if parts.count == 2 {
      guard ["ppo", "random"].contains(parts[1]) else { throw err("bad baseline id") }
      root = root.appendingPathComponent("baselines/\(parts[1])", isDirectory: true)
    }
    guard FileManager.default.fileExists(atPath: root.appendingPathComponent("meta.json").path) else {
      throw err("unknown run \(id)")
    }
    return root
  }

  private func bestId() -> String? {
    let text = try? String(contentsOf: bench.deletingLastPathComponent().appendingPathComponent("best"), encoding: .utf8)
    let trimmed = text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    return trimmed.isEmpty ? nil : trimmed
  }

  private func entry(_ id: String, _ r: RunReader, best: String?) -> [String: Any] {
    let meta = r.meta.raw
    return [
      "id": id,
      "envIds": r.meta.envLanes.map(\.envId),
      "steps": r.meta.steps,
      "ablationLabel": meta["ablationLabel"] ?? "none",
      "startedAt": meta["startedAt"] ?? NSNull(),
      "complete": r.complete,
      "maxStep": r.maxStep,
      "inFlight": running?.root.standardizedFileURL == r.root.standardizedFileURL,
      "hasSummary": FileManager.default.fileExists(atPath: r.root.appendingPathComponent("summary.json").path),
      "hasRecordings": !r.frames.isEmpty,
      "policy": r.meta.policy,
      "preset": meta["preset"] ?? "",
      "role": id == best ? "best" : NSNull(),
    ]
  }

  private func runList() throws -> [[String: Any]] {
    let fm = FileManager.default
    let best = bestId()
    var entries: [(String, [String: Any])] = []
    let names = try fm.contentsOfDirectory(atPath: bench.path).filter { $0 != "baselines" && !$0.hasPrefix(".") }
    for name in names {
      guard let r = try? reader(name), !r.windows.isEmpty else { continue }
      entries.append((name, entry(name, r, best: best)))
      guard r.meta.policy == "oak" else { continue }
      for policy in ["ppo", "random"] {
        let id = "\(name):\(policy)"
        guard let b = try? reader(id), b.complete,
          fm.fileExists(atPath: b.root.appendingPathComponent("summary.json").path)
        else { continue }
        entries.append((id, entry(id, b, best: best)))
      }
    }
    entries.sort { a, b in
      let sa = a.1["startedAt"] as? String ?? ""
      let sb = b.1["startedAt"] as? String ?? ""
      if sa != sb { return sa > sb }
      return a.0 > b.0
    }
    return entries.map(\.1)
  }

  private func detail(_ id: String) throws -> [String: Any] {
    let r = try reader(id)
    let fm = FileManager.default
    var summary: Any = NSNull()
    let summaryURL = r.root.appendingPathComponent("summary.json")
    if fm.fileExists(atPath: summaryURL.path) {
      summary = try JSONSerialization.jsonObject(with: Data(contentsOf: summaryURL))
    }
    var recordings: [[String: Any]] = []
    if !r.frames.isEmpty {
      for env in r.meta.envLanes.sorted(by: { ((try? Main.envKind($0.envId)) ?? 0) < ((try? Main.envKind($1.envId)) ?? 0) }) {
        for lane in env.lanes { recordings.append(["envId": env.envId, "seed": r.meta.seeds[lane]]) }
      }
    }
    var baselines: [String: Any] = [:]
    if r.meta.policy == "oak" {
      for policy in ["ppo", "random"] {
        let bid = "\(id):\(policy)"
        if let b = try? reader(bid), b.complete,
          fm.fileExists(atPath: b.root.appendingPathComponent("summary.json").path)
        { baselines[policy] = bid }
      }
    }
    return ["id": id, "meta": r.meta.raw, "summary": summary, "recordings": recordings, "baselines": baselines]
  }

  func route(_ request: HTTPRequest) -> HTTPResponse {
    do {
      let segs = request.path.split(separator: "/").map { String($0).removingPercentEncoding ?? String($0) }
      guard segs.first == "api" else {
        guard request.method == "GET" || request.method == "HEAD" else { return .fail(405, "method not allowed") }
        return try staticFile(request.path)
      }
      switch (request.method, segs.dropFirst().map { $0 }) {
      case ("GET", ["identity"]):
        return try .json(["binary": binarySha, "port": Int(port), "running": running?.id ?? NSNull()])
      case ("GET", ["runs"]):
        return try .json(runList())
      case ("GET", let s) where s.count == 2 && s[0] == "runs":
        return try .json(detail(s[1]))
      case ("GET", let s) where s.count == 4 && s[0] == "runs" && s[2] == "series":
        return .bytes(try reader(s[1]).series(s[3]))
      case ("GET", let s) where s.count == 4 && s[0] == "runs" && s[2] == "frames":
        guard let lane = Int(s[3]), let from = Int(request.query["from"] ?? ""), let n = Int(request.query["n"] ?? "") else {
          return .fail(400, "frames need lane, from, n")
        }
        return .bytes(try reader(s[1]).frames(lane: lane, from: from, count: n))
      case ("GET", let s) where s.count == 4 && s[0] == "runs" && s[2] == "rewards":
        guard let from = Int(request.query["from"] ?? ""), let n = Int(request.query["n"] ?? "") else {
          return .fail(400, "rewards need from, n")
        }
        return .bytes(try reader(s[1]).rewards(envId: s[3], from: from, count: n))
      case ("POST", ["benchmark"]):
        guard running == nil else { return .fail(409, "a run is executing: \(running!.id)") }
        let request = try BenchmarkRequest(json: request.body)
        try start(request)
        return try .json(["id": running!.id], status: 202)
      case ("POST", ["stop"]):
        guard running == nil else { return .fail(409, "a run is executing: \(running!.id)") }
        queue.asyncAfter(deadline: .now() + 0.1) { Foundation.exit(0) }
        return try .json(["stopping": true])
      default:
        return .fail(404, "unknown endpoint")
      }
    } catch {
      return .fail(error.localizedDescription.contains("unknown run") ? 404 : 500, "\(error)")
    }
  }

  private func staticFile(_ path: String) throws -> HTTPResponse {
    let rel = path.split(separator: "/").map { String($0).removingPercentEncoding ?? String($0) }
      .filter { $0 != ".." && !$0.isEmpty && !$0.contains("/") }
    var url = dist
    for part in rel { url.appendPathComponent(part) }
    var isDir: ObjCBool = false
    if !FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) || isDir.boolValue {
      if rel.first == "assets" || (rel.last?.contains(".") ?? false) { return .fail(404, "not found") }
      url = dist.appendingPathComponent("index.html")
    }
    guard FileManager.default.fileExists(atPath: url.path) else {
      return .fail(404, "no frontend build; run `bun run build` in frontend")
    }
    let types = [
      "html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8", "css": "text/css; charset=utf-8",
      "json": "application/json; charset=utf-8", "svg": "image/svg+xml", "png": "image/png", "webp": "image/webp",
      "ico": "image/x-icon", "woff2": "font/woff2", "map": "application/json; charset=utf-8", "bin": "application/octet-stream",
    ]
    return HTTPResponse(
      status: 200, contentType: types[url.pathExtension.lowercased()] ?? "application/octet-stream",
      body: try Data(contentsOf: url))
  }

  func command(_ text: String, from c: Connection) {
    let parts = text.split(separator: " ").map(String.init)
    guard let verb = parts.first else { return }
    switch verb {
    case "follow" where parts.count == 3:
      c.follows.insert("\(parts[1]) \(parts[2])")
      if let r = try? reader(parts[1]) { c.sendText("columns \(parts[1]) \(r.fields.joined(separator: ","))") }
    case "unfollow" where parts.count == 3:
      c.follows.remove("\(parts[1]) \(parts[2])")
    case "signal" where parts.count == 5:
      guard let job = running, job.id == parts[1], let lane = Int(parts[2]), let word = Int(parts[3]),
        let value = UInt32(parts[4])
      else { c.sendText("error signal needs a running run, lane, word, value"); return }
      job.control.signal(lane: lane, word: word, value: value)
    case "save" where parts.count == 2:
      guard let job = running, job.id == parts[1] else { c.sendText("error save needs the running run"); return }
      job.control.requestSave()
    default:
      c.sendText("error unknown command")
    }
  }

  private func windowLanded(_ job: RunningJob, step: Int) {
    guard let r = try? reader(job.id), let index = r.windows.indices.last else { return }
    let w = r.windows[index]
    let id = Data(job.id.utf8)
    func envelope(_ kind: UInt8) -> Data {
      var out = Data([kind])
      out.appendLE(UInt16(id.count))
      out.append(id)
      return out
    }
    var header: Data? = nil
    for c in sockets.values {
      let lanes = c.follows.compactMap { key -> Int? in
        let parts = key.split(separator: " ")
        guard parts.count == 2, parts[0] == job.id else { return nil }
        return Int(parts[1])
      }
      guard !lanes.isEmpty else { continue }
      if header == nil {
        var h = envelope(1)
        for v: UInt32 in [UInt32(index), UInt32(w.stepStart + w.stepCount)] { h.appendLE(v) }
        h.appendLE(w.wallSeconds.bitPattern)
        if let row = try? r.presented(index) { row.withUnsafeBufferPointer { h.append(Data(buffer: $0)) } }
        header = h
      }
      c.sendBinary(header!)
      for lane in Set(lanes) where lane >= 0 {
        guard let frames = try? r.frames(lane: lane, from: w.stepStart, count: w.stepCount), !frames.isEmpty else { continue }
        var push = envelope(2)
        for v: UInt32 in [UInt32(lane), UInt32(w.stepStart)] { push.appendLE(v) }
        push.append(frames)
        c.sendBinary(push)
      }
    }
  }

  private func start(_ request: BenchmarkRequest) throws {
    let rel = request.root.standardizedFileURL.path.replacingOccurrences(of: bench.standardizedFileURL.path + "/", with: "")
    let job = RunningJob(id: rel, root: request.root)
    job.control.onWindow = { [weak self, weak job] step in
      guard let self, let job else { return }
      self.queue.async { self.windowLanded(job, step: step) }
    }
    running = job
    readers.removeValue(forKey: job.id)
    Thread.detachNewThread { [self] in
      var assertion: IOPMAssertionID = 0
      IOPMAssertionCreateWithName(
        kIOPMAssertionTypePreventUserIdleSystemSleep as CFString, IOPMAssertionLevel(kIOPMAssertionLevelOn),
        "OaK run \(job.id)" as CFString, &assertion)
      defer { IOPMAssertionRelease(assertion) }
      do {
        try Main.benchmark(request, control: job.control)
        fputs("[serve] run \(job.id) complete\n", stderr)
      } catch {
        fputs("[serve] run \(job.id) failed: \(error)\n", stderr)
      }
      queue.async {
        if self.running === job { self.running = nil }
      }
    }
  }
}
