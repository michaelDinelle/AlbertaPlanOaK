import Darwin
import Foundation
import Metal

struct OakError: Error, CustomStringConvertible { let description: String }
func err(_ s: String) -> OakError { OakError(description: s) }
enum Policy: String { case oak, zero, random }
let defaultEnv = "HalfCheetah-v4"
let lockName = ".oak-benchmark.lock"
typealias Raw = UnsafeMutableRawPointer

@_silgen_name("oak_env_open") func oakOpen(
  _: Int32, _: UnsafePointer<CChar>?, _: UnsafePointer<CChar>?
) -> Raw?
@_silgen_name("oak_env_close") func oakClose(_: Raw?)
@_silgen_name("oak_env_error") func oakError(_: Raw?) -> UnsafePointer<CChar>?
@_silgen_name("oak_env_prepare") func oakPrepare(_: Raw?, _: Int32) -> Int32
@_silgen_name("oak_env_reset") func oakReset(_: Raw?) -> Int32
@_silgen_name("oak_env_step_random") func oakStepRandom(_: Raw?) -> Int32
@_silgen_name("oak_env_step_zero") func oakStepZero(_: Raw?) -> Int32
@_silgen_name("oak_env_step_oak") func oakStepOak(
  _: Raw?, _: UnsafePointer<Float>?, _: UnsafePointer<Float>?, _: Int32
) -> Int32
@_silgen_name("oak_env_view") func oakView(_: Raw?, _: UnsafeMutablePointer<EnvView>?) -> Int32

struct KernelConfig {
  var obsDim, rank, optionCount, actionParamCount, obsTraceLevels, linearAnchorCount: UInt32
  var invRank, invOptionCount, metaEta: Float
}
struct KernelScalars {
  var stepCount: UInt32 = 0, stateSlot: UInt32 = 0
  var rho: Float = 0
}
struct KernelStats {
  var encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount: Float
  var oakOptionReinitCount, oakFeatureReinitCount, terminationRate, optionDurationMean: Float
  var importanceEss, actorUpdateNormMean, actorSupportEntropy, modelPriorityMax: Float
  var selectedOptionAge, selectedOptionDuration, selectedOptionPriority, keyboardL1, actorAdvantageL1: Float
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
func envError(_ raw: Raw?) -> String {
  oakError(raw).map { String(cString: $0) } ?? "native env error"
}
func mtlSize(_ width: Int) -> MTLSize {
  MTLSize(width: width, height: 1, depth: 1)
}
struct EnvView {
  var obsDim, actDim, dPolicy, done: Int32
  var reward: Float
  var actionDeltaMean: Double
  var infoCount: Int32
  var infoNames: UnsafePointer<UnsafePointer<CChar>?>?
  var infoValues: UnsafePointer<Double>?
  var obs, action, credit, imp: UnsafePointer<Float>?
}
final class Env {
  let raw: Raw, envId: String
  var v = EnvView(
    obsDim: 0, actDim: 0, dPolicy: 0, done: 0, reward: 0, actionDeltaMean: 0,
    infoCount: 0, infoNames: nil, infoValues: nil, obs: nil, action: nil, credit: nil, imp: nil)
  init(_ id: String, seed: Int) throws {
    let assets = try require(".build/oak-native/env-assets", "native env assets").path
    raw = try id.withCString { eid in
      try assets.withCString { base in
        guard let r = oakOpen(Int32(seed), eid, base) else {
          throw err(envError(nil))
        }
        return r
      }
    }
    envId = id
    try refresh()
  }
  deinit { oakClose(raw) }
  var obsDim: Int { Int(v.obsDim) }
  var actDim: Int { Int(v.actDim) }
  var dPolicy: Int { Int(v.dPolicy) }
  var reward: Float { v.reward }
  var done: Int { Int(v.done) }
  var actionDeltaMean: Double { Double(v.actionDeltaMean) }
  var envInfo: [(String, Double)] {
    guard let names = v.infoNames, let values = v.infoValues else { return [] }
    return (0..<Int(max(v.infoCount, 0))).compactMap { i in
      names[i].map { (String(cString: $0), values[i]) }
    }
  }
  func ok(_ x: Int32) throws { if x != 1 { throw err(envError(raw)) } }
  func refresh() throws { try ok(oakView(raw, &v)) }
  func update(_ x: Int32) throws {
    try ok(x)
    try refresh()
  }
  func prepare(_ k: Int) throws {
    try update(oakPrepare(raw, Int32(k)))
  }
  func reset() throws {
    try update(oakReset(raw))
  }
  func stepOak(_ p: UnsafePointer<Float>, _ o: UnsafePointer<Float>, _ k: Int) throws {
    try update(oakStepOak(raw, p, o, Int32(k)))
  }
  func stepRandom() throws {
    try update(oakStepRandom(raw))
  }
  func stepZero() throws {
    try update(oakStepZero(raw))
  }
}

struct Arena {
  let obs, policy, options, action, credit, imp, reward, utilOpt, bonus, alpha, beta, alphaPsi,
    betaPsi, alphaModel, betaModel, total: Int
  static func traces(_ o: Int) -> Int {
    Int(log2(Double(max(o, 1)))) + 1
  }
  static func input(_ o: Int, _ d: Int) -> Int { traces(o) * max(o, 1) + max(d, 0) + 2 }
  init(_ o: Int, _ r: Int, _ d: Int) {
    let traceExtra = max(Self.traces(o) - 1, 0)
    obs = 0
    policy = 3 * o + 2 * r
    action = policy + d
    credit = action + d
    options = credit + r * d
    imp = options + r * d
    reward = imp + r
    utilOpt = reward + 3 + 2 * r
    bonus = utilOpt + r
    alpha = bonus + 3 * r
    beta = alpha + r
    alphaPsi = alpha + 4 * r
    betaPsi = alpha + 5 * r
    alphaModel = alpha + 8 * r
    betaModel = alpha + 9 * r
    total = alpha + r * r + 21 * r + traceExtra * max(o, 1) + 1
  }
}
final class Shared<T> {
  let b: MTLBuffer
  let p: UnsafeMutablePointer<T>
  init(_ d: MTLDevice, _ n: Int, _ z: T) {
    b = d.makeBuffer(length: max(n, 1) * MemoryLayout<T>.stride, options: [.storageModeShared])!
    p = b.contents().bindMemory(to: T.self, capacity: max(n, 1))
    p.initialize(repeating: z, count: max(n, 1))
  }
}
final class Engine {
  let rank, d, tg, learnTotal, actionGroups: Int
  let a: Arena
  let q: MTLCommandQueue, pipes: [MTLComputePipelineState], params, state: Shared<Float>,
    uints: Shared<UInt32>, res: [MTLBuffer]
  let scalarsB, statsB, configB: MTLBuffer
  let scalars: UnsafeMutablePointer<KernelScalars>, stats: UnsafeMutablePointer<KernelStats>,
    tgBytes: Int
  var policy: UnsafePointer<Float> { UnsafePointer(state.p + a.policy) }
  var options: UnsafePointer<Float> { UnsafePointer(state.p + a.options) }
  init(_ env: Env, _ dev: MTLDevice) throws {
    rank = max(env.obsDim, env.dPolicy * 2, 2)
    d = env.dPolicy
    a = Arena(env.obsDim, rank, d)
    guard let cq = dev.makeCommandQueue() else {
      throw err("Metal command queue unavailable.")
    }
    q = cq
    let lib = try dev.makeLibrary(
      URL: try require(".build/OakKernel.metallib", "non-empty Metal library"))
    pipes = try ["oak_encode", "oak_plan", "oak_learn_heads", "oak_replace", "oak_action_emit"].map
    { try dev.makeComputePipelineState(function: lib.makeFunction(name: $0)!) }
    let maxTG = pipes.map(\.maxTotalThreadsPerThreadgroup).min() ?? rank
    guard rank <= maxTG else {
      throw err("Rank \(rank) exceeds device threadgroup capacity \(maxTG).")
    }
    tg = min(max(rank, pipes.map { Int($0.threadExecutionWidth) }.max() ?? 1), maxTG)
    tgBytes = tg * MemoryLayout<Float>.stride
    let k = rank
    let paramTotal = rank + k * rank + k * d * rank + k * rank + k * rank * rank + k * rank
    learnTotal =
      rank + rank * rank + rank * d * rank + rank * rank + rank * rank * rank + rank * rank
    actionGroups = max(1, (rank * d + tg - 1) / tg)
    params = Shared(dev, paramTotal, 0)
    state = Shared(dev, a.total, 0)
    uints = Shared(dev, 8 + 8 * rank, 0)
    scalarsB = dev.makeBuffer(
      length: MemoryLayout<KernelScalars>.stride, options: [.storageModeShared])!
    statsB = dev.makeBuffer(
      length: MemoryLayout<KernelStats>.stride, options: [.storageModeShared])!
    configB = dev.makeBuffer(
      length: MemoryLayout<KernelConfig>.stride, options: [.storageModeShared])!
    res = [params.b, state.b, uints.b, scalarsB, statsB, configB]
    scalars = scalarsB.contents().bindMemory(to: KernelScalars.self, capacity: 1)
    stats = statsB.contents().bindMemory(to: KernelStats.self, capacity: 1)
    scalars.initialize(to: KernelScalars())
    memset(stats, 0, MemoryLayout<KernelStats>.stride)
    let psi = rank
    let optPolicy = psi + rank * rank
    let inv = 1 / sqrt(Float(rank))
    let anchors = min(rank, max(env.obsDim, d, 1))
    let alpha = 1 / Float(max(env.obsDim, 1))
    for i in 0..<rank {
      params.p[psi + i * rank + i] = inv
      uints.p[7 + i] = UInt32(i)
      uints.p[7 + 3 * rank + i] = UInt32(i % max(anchors, 1))
      uints.p[7 + 4 * rank + i] = UInt32((i * 7 + 3) % max(anchors, 1))
      uints.p[7 + 5 * rank + i] = UInt32(i % 3)
      state.p[a.utilOpt + i] = Float(rank)
      [a.alpha, a.alphaPsi, a.alphaModel].forEach { state.p[$0 + i] = alpha }
      [a.beta, a.betaPsi, a.betaModel].forEach { state.p[$0 + i] = log(alpha) }
    }
    let basis = 1 / Float(max(d, 1))
    for k in 0..<rank {
      let slot = k % max(d, 1)
      let sign: Float = ((k / max(d, 1)) % 2 == 0) ? 1 : -1
      params.p[optPolicy + (k * d + slot) * rank] = sign * basis
    }
    configB.contents().bindMemory(to: KernelConfig.self, capacity: 1).initialize(
      to: KernelConfig(
        obsDim: UInt32(env.obsDim), rank: UInt32(rank), optionCount: UInt32(rank),
        actionParamCount: UInt32(d), obsTraceLevels: UInt32(max(Arena.traces(env.obsDim) - 1, 0)),
        linearAnchorCount: UInt32(anchors), invRank: 1 / Float(rank),
        invOptionCount: 1 / Float(rank), metaEta: 1 / Float(rank) / Float(max(rank + d, 1))))
  }
  func step(_ env: Env, capture: Bool) throws -> KernelStats? {
    memcpy(state.p + a.obs, env.v.obs!, env.obsDim * MemoryLayout<Float>.stride)
    state.p[a.reward] = env.reward
    memcpy(state.p + a.action, env.v.action!, d * MemoryLayout<Float>.stride)
    memcpy(state.p + a.credit, env.v.credit!, rank * d * MemoryLayout<Float>.stride)
    memcpy(state.p + a.imp, env.v.imp!, rank * MemoryLayout<Float>.stride)
    let c = uints.p
    c[0] = 0
    c[1] = 0
    c[2] = 0
    c[5] = 0
    c[6] = 0
    guard let cb = q.makeCommandBufferWithUnretainedReferences(),
      let e = cb.makeComputeCommandEncoder()
    else { throw err("Failed to encode Metal command.") }
    func bind(_ i: Int) {
      e.setComputePipelineState(pipes[i])
      for j in res.indices { e.setBuffer(res[j], offset: 0, index: j) }
    }
    func groups(_ i: Int, _ n: Int, _ mem: Int = 1, _ barrier: Bool = true) {
      bind(i)
      for m in 0..<mem { e.setThreadgroupMemoryLength(tgBytes, index: m) }
      e.dispatchThreadgroups(mtlSize(n), threadsPerThreadgroup: mtlSize(tg))
      if barrier { e.memoryBarrier(scope: [.buffers]) }
    }
    groups(0, 1, 7)
    groups(1, rank)
    bind(2)
    let ltg = min(
      max(Int(pipes[2].threadExecutionWidth), 1), pipes[2].maxTotalThreadsPerThreadgroup)
    e.dispatchThreads(mtlSize(learnTotal), threadsPerThreadgroup: mtlSize(ltg))
    e.memoryBarrier(scope: [.buffers])
    groups(3, 1)
    groups(4, actionGroups, 1, false)
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw err("Metal command failed: \(e)") }
    return capture ? stats.pointee : nil
  }
}

final class Run {
  var steps = 0, done = 0, pulse = 0, nfE = 0, nfL = 0, nfP = 0, nfT = 0, optR = 0,
    featR = 0
  var seconds = 0.0, rewardSum = 0.0, rho = 0.0, floor: Double,
    start = DispatchTime.now().uptimeNanoseconds
  var envInfo: [String: InfoStat] = [:]
  var last: KernelStats?
  init(obsDim: Int) { floor = 1 / Double(max(obsDim, 1)) }
  func note(_ env: Env, _ stats: KernelStats?) throws {
    steps += 1
    seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    done += env.done
    let r = Double(env.reward)
    guard r.isFinite else {
      throw err("Environment produced a non-finite reward.")
    }
    rewardSum += r
    rho += max(1 / Double(steps), floor) * (r - rho)
    for (name, value) in env.envInfo { envInfo[name, default: InfoStat()].note(value) }
    if let s = stats {
      last = s
      nfE = Int(s.encodeNonFiniteCount)
      nfL = Int(s.learnNonFiniteCount)
      nfP = Int(s.planNonFiniteCount)
      nfT = Int(s.tdPsiNonFiniteCount)
      optR = Int(s.oakOptionReinitCount)
      featR = Int(s.oakFeatureReinitCount)
    }
  }
  func health(_ env: Env) -> [String: Any] {
    var h: [String: Any] = [
      "rewardRate": rho.isFinite ? rho : NSNull(), "actionDeltaL2Mean": env.actionDeltaMean.isFinite ? env.actionDeltaMean : NSNull(),
      "continuingRestartCount": Double(done),
    ]
    if let s = last {
      h["oakOptionReinitCountFinal"] = optR
      h["oakFeatureReinitCountFinal"] = featR
      h["nonFiniteByPhase"] = ["encode": nfE, "learn": nfL, "plan": nfP, "tdPsi": nfT]
      h["terminationRate"] = s.terminationRate.isFinite ? Double(s.terminationRate) : NSNull()
      h["optionDurationMean"] = s.optionDurationMean.isFinite ? Double(s.optionDurationMean) : NSNull()
      h["importanceEss"] = s.importanceEss.isFinite ? Double(s.importanceEss) : NSNull()
      h["actorUpdateNormMean"] = s.actorUpdateNormMean.isFinite ? Double(s.actorUpdateNormMean) : NSNull()
      h["actorSupportEntropy"] = s.actorSupportEntropy.isFinite ? Double(s.actorSupportEntropy) : NSNull()
      h["modelPriorityMax"] = s.modelPriorityMax.isFinite ? Double(s.modelPriorityMax) : NSNull()
      h["selectedOptionAge"] = s.selectedOptionAge.isFinite ? Double(s.selectedOptionAge) : NSNull()
      h["selectedOptionDuration"] = s.selectedOptionDuration.isFinite ? Double(s.selectedOptionDuration) : NSNull()
      h["selectedOptionPriority"] = s.selectedOptionPriority.isFinite ? Double(s.selectedOptionPriority) : NSNull()
      h["keyboardL1"] = s.keyboardL1.isFinite ? Double(s.keyboardL1) : NSNull()
      h["actorAdvantageL1"] = s.actorAdvantageL1.isFinite ? Double(s.actorAdvantageL1) : NSNull()
    }
    if !envInfo.isEmpty {
      h["envInfo"] = Dictionary(uniqueKeysWithValues: envInfo.sorted { $0.key < $1.key }.map {
        ($0.key, $0.value.json)
      })
    }
    return h
  }
}
struct InfoStat {
  var count = 0, finiteCount = 0, nonFiniteCount = 0
  var sum = 0.0
  var first: Double?
  var last: Double?
  mutating func note(_ x: Double) {
    count += 1
    guard x.isFinite else { nonFiniteCount += 1; return }
    if first == nil { first = x }
    last = x
    sum += x
    finiteCount += 1
  }
  var json: [String: Any] {
    [
      "count": count, "finiteCount": finiteCount, "nonFiniteCount": nonFiniteCount,
      "sum": finiteCount > 0 ? sum : NSNull() as Any,
      "mean": finiteCount > 0 ? sum / Double(finiteCount) : NSNull() as Any,
      "last": last.map { $0 as Any } ?? NSNull(),
      "delta": first.flatMap { a in last.map { ($0 - a) as Any } } ?? NSNull(),
    ]
  }
}
func stem(_ env: String, _ p: Policy, _ seed: Int) -> String {
  "\(env)-\(p.rawValue)-seed\(seed)"
}
func writeMetric(
  _ env: Env, _ p: Policy, _ seed: Int, _ stop: Int, _ r: Run, _ seq: Int?, _ root: URL
) throws {
  var m: [String: Any] = [
    "schemaVersion": 6, "command": "benchmark", "envId": env.envId, "policy": p.rawValue,
    "seed": seed, "numEnvs": 1, "stopAfterMeasuredSteps": stop, "observationDim": env.obsDim,
    "actionDim": env.actDim, "measuredStepCount": r.steps, "measuredSeconds": r.seconds,
    "rewardSum": r.rewardSum, "doneCount": r.done,
    "stopReason": seq == nil ? "measured_steps" : "heartbeat", "health": r.health(env),
  ]
  if r.seconds > 0 { m["stepsPerSecond"] = Double(r.steps) / r.seconds }
  if let seq { m["heartbeatSequence"] = seq }
  let dir = root.appendingPathComponent(env.envId, isDirectory: true).appendingPathComponent(
    "seed\(seed)", isDirectory: true)
  try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  let name =
    seq.map { "\(stem(env.envId, p, seed)).heartbeat-\(String(format: "%06d", $0)).json" }
    ?? "\(stem(env.envId, p, seed)).json"
  let data = try JSONSerialization.data(withJSONObject: m, options: [.prettyPrinted, .sortedKeys])
  try data.write(to: dir.appendingPathComponent(name), options: .atomic)
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

@main struct Main {
  static func main() {
    _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    do {
      switch CommandLine.arguments.dropFirst().first {
      case "benchmark": try benchmark()
      case "benchmark-worker": try worker()
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
  static func run(_ p: Policy, _ seed: Int, _ steps: Int, _ root: URL, _ envId: String) throws {
    let env = try Env(envId, seed: seed)
    try env.prepare(p == .oak ? max(env.obsDim, env.dPolicy * 2, 2) : 0)
    try env.reset()
    let engine = p == .oak ? try Engine(env, MTLCreateSystemDefaultDevice()!) : nil
    let run = Run(obsDim: env.obsDim)
    let hb = max(
      ProcessInfo.processInfo.environment["OAK_BENCH_HEARTBEAT_STEPS"].flatMap(Int.init) ?? steps / 10,
      1)
    while run.steps < steps {
      try autoreleasepool {
        let next = run.steps + 1
        let heartbeat = run.steps / hb < next / hb
        let end = next == steps
        let s = try engine?.step(env, capture: heartbeat || end)
        if let e = engine {
          try env.stepOak(e.policy, e.options, e.rank)
        } else if p == .zero {
          try env.stepZero()
        } else {
          try env.stepRandom()
        }
        try run.note(env, s)
        if heartbeat {
          run.pulse += 1
          try writeMetric(env, p, seed, steps, run, run.pulse, root)
        }
      }
    }
    try writeMetric(env, p, seed, steps, run, nil, root)
  }
  static func worker() throws {
    let e = ProcessInfo.processInfo.environment
    guard let pr = e["OAK_POLICY"], let seed = e["OAK_SEED"].flatMap(Int.init),
      let stop = e["OAK_STOP_AFTER_STEPS"].flatMap(Int.init)
    else { throw err("Incomplete benchmark-worker environment.") }
    try run(try pol(pr), seed, max(stop, 1), metricsRoot(), e["OAK_BENCH_ENV"] ?? defaultEnv)
  }
  static func start(_ item: (Int, Policy), _ steps: Int, _ dir: URL, _ envId: String) throws
    -> pid_t
  {
    var e = ProcessInfo.processInfo.environment
    e["OAK_POLICY"] = item.1.rawValue
    e["OAK_SEED"] = "\(item.0)"
    e["OAK_STOP_AFTER_STEPS"] = "\(steps)"
    e["OAK_METRICS_STORAGE"] = dir.path
    e["OAK_BENCH_ENV"] = envId
    let path = CommandLine.arguments[0]
    var pid = pid_t()
    var argv = [strdup(path), strdup("benchmark-worker"), nil]
    var envp = e.map { strdup("\($0.key)=\($0.value)") } + [nil]
    defer {
      for x in argv { if let x { free(x) } }
      for x in envp { if let x { free(x) } }
    }
    let rc = path.withCString { p in
      argv.withUnsafeMutableBufferPointer { a in
        envp.withUnsafeMutableBufferPointer { v in
          posix_spawnp(&pid, p, nil, nil, a.baseAddress, v.baseAddress)
        }
      }
    }
    guard rc == 0 else {
      throw err("posix_spawn failed: \(String(cString: strerror(rc)))")
    }
    return pid
  }
  static func benchmark() throws {
    let env = ProcessInfo.processInfo.environment
    let fm = FileManager.default
    let envId = env["OAK_BENCH_ENV"] ?? defaultEnv
    let seed0 = env["OAK_BENCH_SEED_START"].flatMap(Int.init) ?? 0
    let seedValues: [Int]
    if let explicitSeeds = env["OAK_BENCH_SEEDS"] {
      let parsed = explicitSeeds.split(separator: ",").compactMap {
        Int(String($0).trimmingCharacters(in: .whitespaces))
      }
      seedValues = parsed.isEmpty
        ? Array(seed0..<(seed0 + (env["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 10)))
        : parsed
    } else {
      seedValues = Array(seed0..<(seed0 + (env["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 10)))
    }
    let steps = max(env["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 500_000, 1)
    let policies = try (env["OAK_BENCH_POLICIES"] ?? "oak,zero,random").split(separator: ",").map {
      try pol(String($0))
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
    let work = seedValues.flatMap { s in policies.map { (s, $0) } }
    let total = work.count
    let conc = max(
      env["OAK_BENCH_CONCURRENCY"].flatMap(Int.init) ?? ProcessInfo.processInfo.processorCount, 1)
    fputs(
      "OaK benchmark: \(total) \(envId) work items, single-stream B=1, N=\(conc) worker processes (steps=\(steps), seeds=\(seedValues.count), dir=\(dir.path))\n",
      stderr)
    var next = 0
    var done = 0
    var running: [pid_t: (Int, Policy)] = [:]
    func fill() throws {
      while next < total && running.count < conc {
        let item = work[next]
        running[try start(item, steps, dir, envId)] = item
        next += 1
      }
    }
    try fill()
    while done < total {
      var st: Int32 = 0
      let pid = waitpid(-1, &st, 0)
      guard pid > 0, let item = running.removeValue(forKey: pid) else { continue }
      guard st == 0 else {
        throw err(
          "Benchmark worker failed for \(envId) seed \(item.0) policy \(item.1.rawValue) (status \(st))."
        )
      }
      done += 1
      try fill()
    }
  }
}
