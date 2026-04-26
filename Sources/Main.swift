import Foundation
import Metal
import Darwin

enum OakError: Error, CustomStringConvertible { case message(String); var description: String { if case let .message(m) = self { return m }; return "" } }
enum Policy: String, Codable { case oak, random }
let oakEnvId = "HalfCheetah-v4", oakLock = ".oak-benchmark.lock"

struct RunHealth: Codable {
    let rewardRate: Double?, actionDeltaL2Mean: Double?, continuingRestartCount: Double?
    let oakOptionReinitCountFinal: Int?, oakFeatureReinitCountFinal: Int?, nonFiniteByPhase: [String: Int]?
    let terminationRate: Double?, optionDurationMean: Double?, importanceEss: Double?
    let actorUpdateNormMean: Double?, actorSupportEntropy: Double?, modelPriorityMax: Double?
}
private struct RunMetricsFile: Encodable {
    let schemaVersion: Int, command: String, envId: String, policy: Policy, seed: Int?
    let numEnvs: Int, stopAfterMeasuredSteps: Int, observationDim: Int, actionDim: Int, actionDiscreteCount: Int?
    let measuredStepCount: Int, measuredSeconds: Double, stepsPerSecond: Double?, heartbeatSequence: Int?
    let rewardSum: Double, doneCount: Int, stopReason: String, health: RunHealth?
}
struct KernelConfig { var obsDim, rank, optionCount, actionParamCount, obsTraceLevels, linearAnchorCount: UInt32; var invRank, invOptionCount, metaEta: Float }
struct KernelScalars { var stepCount: UInt32 = 0, stateSlot: UInt32 = 0; var rho: Float = 0 }
struct KernelStats {
    var encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount: Float
    var oakOptionReinitCount, oakFeatureReinitCount, terminationRate, optionDurationMean: Float
    var importanceEss, actorUpdateNormMean, actorSupportEntropy, modelPriorityMax: Float
}

func decodeCString(_ b: [CChar]) -> String { String(decoding: b.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self) }
func roots() -> [URL] {
    let fm = FileManager.default
    let exe = URL(fileURLWithPath: CommandLine.arguments[0],
                  relativeTo: URL(fileURLWithPath: fm.currentDirectoryPath)).standardizedFileURL
    return [fm.currentDirectoryPath, exe.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().path].map { URL(fileURLWithPath: $0).standardizedFileURL }
}
func buildPath(_ p: String, _ what: String, nonEmpty: Bool = false) throws -> URL {
    guard let u = roots().map({ $0.appendingPathComponent(p) }).first(where: {
        FileManager.default.fileExists(atPath: $0.path) && (!nonEmpty || (((try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 0))
    }) else { throw OakError.message("Missing \(what). Run ./setup.sh first.") }
    return u
}

func copyFloats(from src: UnsafePointer<Float>, to dst: inout [Float], count: Int) {
    _ = dst.withUnsafeMutableBufferPointer { memcpy($0.baseAddress!, src, count * MemoryLayout<Float>.stride) }
}

private final class NativeBridge: @unchecked Sendable {
    typealias OpenFn = @convention(c) (Int32, UnsafePointer<CChar>, UnsafeMutablePointer<CChar>?, Int32) -> UnsafeMutableRawPointer?
    typealias CloseFn = @convention(c) (UnsafeMutableRawPointer?) -> Void
    typealias IntFn = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    typealias BoundsFn = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<Float>?) -> Int32
    typealias ResetFn = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<CChar>?, Int32) -> Int32
    typealias StepFn = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Float>?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<UInt8>?, UnsafeMutablePointer<CChar>?, Int32) -> Int32
    typealias WarnFn = @convention(c) (UnsafeMutablePointer<CChar>?, Int32) -> Int32
    static let shared = { do { return try NativeBridge() } catch { fatalError("\(error)") } }()
    private let dylib: UnsafeMutableRawPointer, assets: String
    let openFn: OpenFn; let closeFn: CloseFn; let obsFn, actFn: IntFn; let boundsFn: BoundsFn; let resetFn: ResetFn; let stepFn: StepFn; let warnFn: WarnFn
    private init() throws {
        assets = try buildPath(".build/oak-native/env-assets", "native env assets").path
        let url = try buildPath(".build/oak-native/liboakenv.dylib", "native env bridge")
        guard let h = dlopen(url.path, RTLD_NOW | RTLD_LOCAL) else { throw OakError.message(String(cString: dlerror())) }
        dylib = h
        func load<T>(_ n: String) throws -> T { guard let r = dlsym(h, n) else { throw OakError.message("Missing symbol \(n).") }; return unsafeBitCast(r, to: T.self) }
        openFn = try load("oak_env_open"); closeFn = try load("oak_env_close"); obsFn = try load("oak_env_obs_dim"); actFn = try load("oak_env_act_dim")
        boundsFn = try load("oak_env_action_bounds"); resetFn = try load("oak_env_reset"); stepFn = try load("oak_env_step"); warnFn = try load("oak_env_take_warning")
    }
    deinit { dlclose(dylib) }
    func open(seed: Int, error: inout [CChar]) -> UnsafeMutableRawPointer? {
        let cap = Int32(error.count)
        return assets.withCString { a in error.withUnsafeMutableBufferPointer { openFn(Int32(seed), a, $0.baseAddress, cap) } }
    }
}

final class HalfCheetahEnv {
    private let bridge = NativeBridge.shared
    let handle: UnsafeMutableRawPointer, obsDim: Int, actDim: Int, dPolicy: Int
    var obs, reward, low, high, envAction, actionSignal, optionActorCredit, importance, optionLog: [Float]
    var done: [UInt8] = [0], msg = [CChar](repeating: 0, count: 4096), rng: UInt64
    var actionDiscreteCount: Int? { nil }
    init(seed: Int) throws {
        rng = UInt64(bitPattern: Int64(seed)) &* 0x9e3779b97f4a7c15 &+ 1
        guard let h = bridge.open(seed: seed, error: &msg) else { throw OakError.message(decodeCString(msg)) }
        handle = h; obsDim = Int(bridge.obsFn(h)); actDim = Int(bridge.actFn(h)); dPolicy = 2 * actDim
        obs = Array(repeating: 0, count: obsDim); reward = [0]; low = Array(repeating: 0, count: actDim); high = low
        envAction = low; actionSignal = Array(repeating: 0, count: dPolicy); optionActorCredit = []; importance = []; optionLog = []
        low.withUnsafeMutableBufferPointer { l in high.withUnsafeMutableBufferPointer { _ = bridge.boundsFn(h, l.baseAddress, $0.baseAddress) } }
        envAction = low
    }
    deinit { bridge.closeFn(handle) }
    func prepare(optionCount: Int) {
        optionActorCredit = Array(repeating: 0, count: optionCount * dPolicy); importance = Array(repeating: 1, count: optionCount); optionLog = Array(repeating: 0, count: optionCount)
    }
    func checked(_ body: (UnsafeMutablePointer<CChar>?, Int32) -> Int32) throws {
        let ok = msg.withUnsafeMutableBufferPointer { body($0.baseAddress, Int32(msg.count)) }
        guard ok == 1 else { throw OakError.message(decodeCString(msg)) }
        let w = msg.withUnsafeMutableBufferPointer { bridge.warnFn($0.baseAddress, Int32(msg.count)) }
        if w == 1 { throw OakError.message("MuJoCo warning: \(decodeCString(msg))") }
    }
    func reset() throws { try checked { e, c in obs.withUnsafeMutableBufferPointer { bridge.resetFn(handle, $0.baseAddress, e, c) } } }
    private func u01() -> Float {
        rng &+= 0x9e3779b97f4a7c15; var z = rng
        z = (z ^ (z &>> 30)) &* 0xbf58476d1ce4e5b9; z = (z ^ (z &>> 27)) &* 0x94d049bb133111eb; z ^= z &>> 31
        let bits = Float.significandBitCount
        return Float(z >> (UInt64.bitWidth - bits)) / Float(UInt64(1) << bits)
    }
    private func normal() -> Float { sqrt(-2 * log(max(u01(), 1 / Float(UInt64(1) << Float.significandBitCount)))) * cos(2 * Float.pi * u01()) }
    func step(policy: UnsafePointer<Float>, options: UnsafePointer<Float>?, optionCount: Int) throws {
        let sd = 1 / Float(max(dPolicy, 1)), logSigma = -log(sd)
        for k in 0..<optionCount { optionLog[k] = 0; importance[k] = 0 }
        var behaviorLog: Float = 0
        for d in 0..<actDim {
            let mu = policy[d].isFinite ? policy[d] : 0
            let z = normal(), raw = mu + sd * z, sample = raw.isFinite ? raw : 0
            envAction[d] = max(low[d], min(high[d], sample)); actionSignal[d] = sample - mu; actionSignal[actDim+d] = 0; behaviorLog += logSigma - 0.5 * z * z
            if let opt = options {
                for k in 0..<optionCount {
                    let off = k * dPolicy, om = opt[off+d].isFinite ? opt[off+d] : 0, oz = (sample - om) / sd
                    optionLog[k] += logSigma - 0.5 * oz * oz
                    optionActorCredit[off+d] = sample - om
                    optionActorCredit[off+actDim+d] = 0
                }
            }
        }
        for k in 0..<optionCount { importance[k] = exp(optionLog[k] - behaviorLog) }
        try checked { e, c in
            envAction.withUnsafeBufferPointer { ab in obs.withUnsafeMutableBufferPointer { ob in reward.withUnsafeMutableBufferPointer { rb in done.withUnsafeMutableBufferPointer { db in
                bridge.stepFn(handle, ab.baseAddress, ob.baseAddress, rb.baseAddress, db.baseAddress, e, c)
            }}}}
        }
    }
}

private struct Health {
    var envSteps = 0, kernelSteps = 0, rho = 0.0, actDelta = 0.0, nfE = 0, nfL = 0, nfP = 0, nfT = 0, optR = 0, featR = 0
    var last: KernelStats?, floor: Double
    init(obsDim: Int) { floor = 1 / Double(max(obsDim, 1)) }
    mutating func env(reward r0: Float, prev: UnsafeBufferPointer<Float>, curr: UnsafeBufferPointer<Float>) {
        envSteps += 1; let r = Double(r0); if r.isFinite { let a = max(1 / Double(envSteps), floor); rho += a * (r - rho) }
        var d = 0.0; for i in 0..<curr.count { let x = Double(curr[i] - prev[i]); d += x * x }
        let m = Double.greatestFiniteMagnitude / 4; actDelta = min(m, actDelta + (d.isFinite ? min(d / Double(max(curr.count, 1)), m) : m))
    }
    mutating func kernel(_ s: KernelStats) {
        kernelSteps += 1; last = s; nfE += Int(s.encodeNonFiniteCount); nfL += Int(s.learnNonFiniteCount); nfP += Int(s.planNonFiniteCount); nfT += Int(s.tdPsiNonFiniteCount)
        optR = Int(s.oakOptionReinitCount); featR = Int(s.oakFeatureReinitCount)
    }
    func snapshot(done: Int) -> RunHealth? {
        guard envSteps > 0 else { return nil }
        func stat(_ f: (KernelStats) -> Float) -> Double? { guard let s = last, kernelSteps > 0 else { return nil }; let v = Double(f(s)); return v.isFinite ? v : 0 }
        return RunHealth(rewardRate: rho, actionDeltaL2Mean: actDelta / Double(envSteps), continuingRestartCount: Double(done),
                         oakOptionReinitCountFinal: kernelSteps > 0 ? optR : nil, oakFeatureReinitCountFinal: kernelSteps > 0 ? featR : nil,
                         nonFiniteByPhase: kernelSteps > 0 ? ["encode": nfE, "learn": nfL, "plan": nfP, "tdPsi": nfT] : nil,
                         terminationRate: stat { $0.terminationRate }, optionDurationMean: stat { $0.optionDurationMean }, importanceEss: stat { $0.importanceEss },
                         actorUpdateNormMean: stat { $0.actorUpdateNormMean }, actorSupportEntropy: stat { $0.actorSupportEntropy }, modelPriorityMax: stat { $0.modelPriorityMax })
    }
}

private enum Layout {
    struct P { let value, psi, optPolicy, optReward, optTransition, optDuration, total: Int }
    struct S { let observation, obsMean, obsSM, state, policyMean, actionSignal, optionActorCredit, optionPolicyMean, importance, reward, td, tdPsi, utilFeat, utilOpt, bonusW, featureMean, featureSM, alpha, beta, h, gradEnergy, alphaPsi, betaPsi, hPsi, gradEnergyPsi, alphaModel, betaModel, hModel, gradEnergyModel, actorAdv, nTdMat, optRewardTd, headDenom, durationTd, durationHat, obsTraceMulti, prioOpt, continuation, modelDenom, valueGauge, total: Int }
    struct U { let counters, optFeature, featureAge, optAge, featureParentA, featureParentB, featureKind, featureBirthCount, selectedOption, badFeatureAge, total: Int }
    static func traces(_ o: Int) -> Int { var v = max(o, 1), m = 0; while v > 1 { v >>= 1; m += 1 }; return m + 1 }
    static func input(_ obs: Int, _ d: Int) -> Int { traces(obs) * max(obs, 1) + max(d, 0) + 2 }
    static func anchors(_ r: Int, _ obs: Int, _ d: Int) -> Int { min(r, input(obs, d)) }
    static func p(_ r: Int, _ d: Int) -> P { let k = r; var o = 0; func take(_ n: Int) -> Int { let x = o; o += n; return x }; return P(value: take(r), psi: take(k*r), optPolicy: take(k*d*r), optReward: take(k*r), optTransition: take(k*r*r), optDuration: take(k*r), total: o) }
    static func s(_ obs: Int, _ r: Int, _ d: Int) -> S {
        let k = r, m = traces(obs); var o = 0; func take(_ n: Int) -> Int { let x = o; o += n; return x }
        return S(observation: take(obs), obsMean: take(obs), obsSM: take(obs), state: take(2*r), policyMean: take(d), actionSignal: take(d),
                 optionActorCredit: take(k*d), optionPolicyMean: take(k*d), importance: take(k), reward: take(1), td: take(1), tdPsi: take(k),
                 utilFeat: take(r), utilOpt: take(k), bonusW: take(r), featureMean: take(r), featureSM: take(r), alpha: take(r), beta: take(r),
                 h: take(r), gradEnergy: take(r), alphaPsi: take(r), betaPsi: take(r), hPsi: take(r), gradEnergyPsi: take(r), alphaModel: take(r),
                 betaModel: take(r), hModel: take(r), gradEnergyModel: take(r), actorAdv: take(k), nTdMat: take(k*r), optRewardTd: take(k),
                 headDenom: take(k), durationTd: take(k), durationHat: take(k), obsTraceMulti: take(max(m-1,0)*max(obs,1)), prioOpt: take(k),
                 continuation: take(k), modelDenom: take(k), valueGauge: take(1), total: o)
    }
    static func u(_ r: Int) -> U { var o = 0; func t(_ n: Int) -> Int { let x=o; o += n; return x }; return U(counters: t(7), optFeature: t(r), featureAge: t(r), optAge: t(r), featureParentA: t(r), featureParentB: t(r), featureKind: t(r), featureBirthCount: t(r), selectedOption: t(1), badFeatureAge: t(r), total: o) }
}

private final class Shared<T> {
    let buffer: MTLBuffer; let ptr: UnsafeMutablePointer<T>
    init(_ dev: MTLDevice, _ n: Int, _ z: T) { buffer = dev.makeBuffer(length: max(n,1) * MemoryLayout<T>.stride, options: [.storageModeShared])!; ptr = buffer.contents().bindMemory(to: T.self, capacity: max(n,1)); ptr.initialize(repeating: z, count: max(n,1)) }
}

final class OakMetalEngine {
    let obsDim, rank, dPolicy, K, tg, learnTotal, actionGroups: Int
    private let q: MTLCommandQueue, pipes: [MTLComputePipelineState], scalarsB, statsB, configB: MTLBuffer
    private let scalars: UnsafeMutablePointer<KernelScalars>, stats: UnsafeMutablePointer<KernelStats>, config: UnsafeMutablePointer<KernelConfig>
    private let params: Shared<Float>, state: Shared<Float>, uints: Shared<UInt32>, sl: Layout.S, ul: Layout.U, resources: [MTLBuffer], tgBytes: Int
    var policyPointer: UnsafePointer<Float> { UnsafePointer(state.ptr.advanced(by: sl.policyMean)) }
    var optionPointer: UnsafePointer<Float> { UnsafePointer(state.ptr.advanced(by: sl.optionPolicyMean)) }
    init(obsDim: Int, rank: Int, dPolicy: Int, device: MTLDevice, seed: Int) throws {
        self.obsDim = obsDim; self.rank = rank; self.dPolicy = dPolicy; K = rank
        guard let q = device.makeCommandQueue() else { throw OakError.message("Failed to create Metal command queue.") }; self.q = q
        let lib = try device.makeLibrary(URL: try buildPath(".build/OakKernel.metallib", "non-empty Metal library", nonEmpty: true))
        pipes = try ["oak_encode","oak_plan","oak_learn_heads","oak_replace","oak_action_emit"].map { n in guard let f = lib.makeFunction(name: n) else { throw OakError.message("Missing \(n).") }; return try device.makeComputePipelineState(function: f) }
        let maxTG = pipes.map { $0.maxTotalThreadsPerThreadgroup }.min() ?? rank; guard rank <= maxTG else { throw OakError.message("Rank \(rank) exceeds device threadgroup capacity \(maxTG).") }
        tg = min(max(rank, pipes.map { Int($0.threadExecutionWidth) }.max() ?? 1), maxTG); tgBytes = tg * MemoryLayout<Float>.stride
        let pl = Layout.p(rank, dPolicy); sl = Layout.s(obsDim, rank, dPolicy); ul = Layout.u(rank)
        learnTotal = rank + rank*rank + rank*dPolicy*rank + rank*rank + rank*rank*rank + rank*rank; actionGroups = max(1, (rank*dPolicy + tg - 1) / tg)
        scalarsB = device.makeBuffer(length: MemoryLayout<KernelScalars>.stride, options: [.storageModeShared])!; statsB = device.makeBuffer(length: MemoryLayout<KernelStats>.stride, options: [.storageModeShared])!; configB = device.makeBuffer(length: MemoryLayout<KernelConfig>.stride, options: [.storageModeShared])!
        scalars = scalarsB.contents().bindMemory(to: KernelScalars.self, capacity: 1); stats = statsB.contents().bindMemory(to: KernelStats.self, capacity: 1); config = configB.contents().bindMemory(to: KernelConfig.self, capacity: 1)
        scalars.initialize(to: KernelScalars()); stats.initialize(to: KernelStats(encodeNonFiniteCount: 0, learnNonFiniteCount: 0, planNonFiniteCount: 0, tdPsiNonFiniteCount: 0, oakOptionReinitCount: 0, oakFeatureReinitCount: 0, terminationRate: 0, optionDurationMean: 0, importanceEss: 0, actorUpdateNormMean: 0, actorSupportEntropy: 0, modelPriorityMax: 0))
        params = Shared(device, pl.total, Float(0)); state = Shared(device, sl.total, Float(0)); uints = Shared(device, ul.total, UInt32(0)); resources = [params.buffer, state.buffer, uints.buffer, scalarsB, statsB, configB]
        let inv = 1 / sqrt(Float(max(rank,1))), seedBase = UInt32(bitPattern: Int32(seed)) &* 0x9e3779b9
        func sample(_ off: Int, _ n: Int, _ scale: Float) { for i in 0..<n { var x = seedBase &+ UInt32(off+i) &* 0x7feb352d; x = (x ^ (x &>> 16)) &* 0x846ca68b; x ^= x &>> 16; params.ptr[off+i] = (Float(x >> 8) / Float(UInt32(1) << 24) * 2 - 1) * scale } }
        sample(pl.value, rank, inv); sample(pl.psi, rank*rank, inv); sample(pl.optReward, rank*rank, inv); sample(pl.optTransition, rank*rank*rank, inv)
        for k in 0..<rank { params.ptr[pl.psi + k*rank + k] += inv; uints.ptr[ul.optFeature+k] = UInt32(k); state.ptr[sl.importance+k] = 1; state.ptr[sl.utilOpt+k] = Float(rank) }
        let anchors = Layout.anchors(rank, obsDim, dPolicy)
        for i in 0..<rank { uints.ptr[ul.featureParentA+i] = UInt32(i % max(anchors,1)); uints.ptr[ul.featureParentB+i] = UInt32((i &* 7 &+ 3) % max(anchors,1)); uints.ptr[ul.featureKind+i] = UInt32(i % 3); let alpha = 1 / Float(max(obsDim,1)); [sl.alpha,sl.alphaPsi,sl.alphaModel].forEach { state.ptr[$0+i] = alpha }; [sl.beta,sl.betaPsi,sl.betaModel].forEach { state.ptr[$0+i] = log(alpha) }; state.ptr[sl.bonusW+i] = params.ptr[pl.value+i] }
        config.initialize(to: KernelConfig(obsDim: UInt32(obsDim), rank: UInt32(rank), optionCount: UInt32(rank), actionParamCount: UInt32(dPolicy), obsTraceLevels: UInt32(max(Layout.traces(obsDim)-1,0)), linearAnchorCount: UInt32(anchors), invRank: 1/Float(rank), invOptionCount: 1/Float(rank), metaEta: 1/Float(rank) / Float(max(rank+dPolicy,1))))
    }
    func step(env: HalfCheetahEnv, prevAction: [Float], prevCredit: [Float], prevImp: [Float], prevReward: Float, capture: Bool) throws -> KernelStats? {
        _ = env.obs.withUnsafeBufferPointer { memcpy(state.ptr.advanced(by: sl.observation), $0.baseAddress!, obsDim * MemoryLayout<Float>.stride) }
        state.ptr[sl.reward] = prevReward
        _ = prevAction.withUnsafeBufferPointer { memcpy(state.ptr.advanced(by: sl.actionSignal), $0.baseAddress!, dPolicy * MemoryLayout<Float>.stride) }
        _ = prevCredit.withUnsafeBufferPointer { memcpy(state.ptr.advanced(by: sl.optionActorCredit), $0.baseAddress!, K*dPolicy * MemoryLayout<Float>.stride) }
        _ = prevImp.withUnsafeBufferPointer { memcpy(state.ptr.advanced(by: sl.importance), $0.baseAddress!, K * MemoryLayout<Float>.stride) }
        let c = uints.ptr.advanced(by: ul.counters); c[0]=0; c[1]=0; c[2]=0; c[5]=0; c[6]=0
        guard let cb = q.makeCommandBufferWithUnretainedReferences(), let enc = cb.makeComputeCommandEncoder() else { throw OakError.message("Failed to encode Metal command.") }
        func bind(_ i: Int) { enc.setComputePipelineState(pipes[i]); for j in 0..<resources.count { enc.setBuffer(resources[j], offset: 0, index: j) } }
        bind(0); for i in 0..<7 { enc.setThreadgroupMemoryLength(tgBytes, index: i) }; enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)); enc.memoryBarrier(scope: [.buffers])
        bind(1); enc.setThreadgroupMemoryLength(tgBytes, index: 0); enc.dispatchThreadgroups(MTLSize(width: K, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)); enc.memoryBarrier(scope: [.buffers])
        bind(2); let ltg = min(max(Int(pipes[2].threadExecutionWidth),1), pipes[2].maxTotalThreadsPerThreadgroup); enc.dispatchThreads(MTLSize(width: learnTotal, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: ltg, height: 1, depth: 1)); enc.memoryBarrier(scope: [.buffers])
        bind(3); enc.setThreadgroupMemoryLength(tgBytes, index: 0); enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)); enc.memoryBarrier(scope: [.buffers])
        bind(4); enc.setThreadgroupMemoryLength(tgBytes, index: 0); enc.dispatchThreadgroups(MTLSize(width: actionGroups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted(); if let e = cb.error { throw OakError.message("Metal command failed: \(e)") }
        return capture ? stats.pointee : nil
    }
}

private final class RunState {
    var prevReward: Float = 0, prevPolicy, prevAction, prevCredit, prevImp, randomMean: [Float], rng: UInt64, steps = 0, seconds = 0.0, start: UInt64?, rewardSum = 0.0, doneCount = 0, pulse = 0, health: Health
    init(env: HalfCheetahEnv, seed: Int, optionCount: Int) { prevPolicy = Array(repeating: 0, count: env.dPolicy); prevAction = prevPolicy; prevCredit = Array(repeating: 0, count: optionCount * env.dPolicy); prevImp = Array(repeating: 1, count: optionCount); randomMean = prevPolicy; rng = UInt64(bitPattern: Int64(seed)); health = Health(obsDim: env.obsDim) }
}

private enum Snapshot {
    static func dir(_ explicit: URL?) -> URL { if let e = explicit { return e }; if let r = ProcessInfo.processInfo.environment["OAK_METRICS_STORAGE"], !r.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return URL(fileURLWithPath: (r as NSString).expandingTildeInPath, isDirectory: true) }; return URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".oak-benchmarks/\(Main.runId())", isDirectory: true) }
    static func seed(_ s: Int?) -> String { s.map { "seed\($0)" } ?? "seedn" }
    static func stem(_ env: String, _ p: Policy, _ s: Int?) -> String { "\(env)-\(p.rawValue)-\(seed(s))".replacingOccurrences(of: #"[^A-Za-z0-9]+"#, with: "-", options: .regularExpression) }
    static func write(_ f: RunMetricsFile, to root: URL) throws { let leaf = f.heartbeatSequence.map { "\(stem(f.envId, f.policy, f.seed)).heartbeat-\(String(format: "%06d", $0)).json" } ?? "\(stem(f.envId, f.policy, f.seed)).json"; let d = root.appendingPathComponent(f.envId, isDirectory: true).appendingPathComponent(seed(f.seed), isDirectory: true); let e = JSONEncoder(); e.outputFormatting = [.prettyPrinted, .sortedKeys]; try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true); try e.encode(f).write(to: d.appendingPathComponent(leaf), options: .atomic) }
}

@main struct Main {
    static let fmt: DateFormatter = { let f = DateFormatter(); f.calendar = Calendar(identifier: .iso8601); f.locale = Locale(identifier: "en_US_POSIX"); f.timeZone = TimeZone(secondsFromGMT: 0); f.dateFormat = "yyyyMMdd-HHmmss"; return f }()
    static func runId() -> String { fmt.string(from: Date()) + "-\(ProcessInfo.processInfo.processIdentifier)" }
    static func main() { _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); do { switch CommandLine.arguments.dropFirst().first { case "benchmark": try benchmark(); case "benchmark-worker": try worker(); default: throw OakError.message("Use `OaK benchmark`.") } } catch { fputs("\(error)\n", stderr); Foundation.exit(1) } }
    private static func policy(_ raw: String) throws -> Policy { guard let p = Policy(rawValue: raw.lowercased()) else { throw OakError.message("Unknown policy '\(raw)'.") }; return p }
    private static func random(_ s: inout UInt64, _ xs: inout [Float]) { for i in xs.indices { s &+= 1; var z = s &+ 0x9e3779b97f4a7c15; z = (z ^ (z &>> 30)) &* 0xbf58476d1ce4e5b9; z = (z ^ (z &>> 27)) &* 0x94d049bb133111eb; z ^= z &>> 31; xs[i] = Float(z >> (UInt64.bitWidth - Float.significandBitCount)) / Float(UInt64(1) << Float.significandBitCount) * 2 - 1 } }
    private static func metric(_ env: HalfCheetahEnv, _ p: Policy, _ seed: Int, _ stop: Int, _ s: RunState, _ seq: Int?) -> RunMetricsFile {
        RunMetricsFile(schemaVersion: 5, command: "benchmark", envId: oakEnvId, policy: p, seed: seed, numEnvs: 1, stopAfterMeasuredSteps: stop, observationDim: env.obsDim, actionDim: env.actDim, actionDiscreteCount: nil, measuredStepCount: s.steps, measuredSeconds: s.seconds, stepsPerSecond: s.seconds > 0 ? Double(s.steps)/s.seconds : nil, heartbeatSequence: seq, rewardSum: s.rewardSum, doneCount: s.doneCount, stopReason: seq == nil ? "measured_steps" : "heartbeat", health: s.health.snapshot(done: s.doneCount))
    }
    private static func run(policy p: Policy, seed: Int, steps: Int, dir: URL) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw OakError.message("Metal device unavailable.") }
        let env = try HalfCheetahEnv(seed: seed), rank = max(env.obsDim, env.dPolicy * 2, 2)
        let engine = p == .oak ? try OakMetalEngine(obsDim: env.obsDim, rank: rank, dPolicy: env.dPolicy, device: dev, seed: seed) : nil
        env.prepare(optionCount: engine?.K ?? 0); try env.reset(); let st = RunState(env: env, seed: seed, optionCount: engine?.K ?? 0)
        let hb = max(steps / 10, 1); st.start = DispatchTime.now().uptimeNanoseconds
        while st.steps < steps {
            let next = st.steps + 1, willHB = st.steps / hb < next / hb, willEnd = next == steps
            func finish(_ pol: UnsafePointer<Float>, _ stats: KernelStats?) throws {
                st.steps = next; st.seconds = Double(DispatchTime.now().uptimeNanoseconds - st.start!) / 1_000_000_000; st.doneCount += Int(env.done[0])
                let r = env.reward[0]; guard r.isFinite else { throw OakError.message("Environment produced a non-finite reward.") }; st.rewardSum += Double(r)
                st.prevPolicy.withUnsafeBufferPointer { st.health.env(reward: r, prev: $0, curr: UnsafeBufferPointer(start: pol, count: env.dPolicy)) }
                if let x = stats { st.health.kernel(x) }; if willHB { st.pulse += 1; try Snapshot.write(metric(env, p, seed, steps, st, st.pulse), to: dir) }
                copyFloats(from: pol, to: &st.prevPolicy, count: env.dPolicy); st.prevReward = r
                env.actionSignal.withUnsafeBufferPointer { src in if let p = src.baseAddress { copyFloats(from: p, to: &st.prevAction, count: env.dPolicy) } }
                env.optionActorCredit.withUnsafeBufferPointer { src in if let p = src.baseAddress, !src.isEmpty { copyFloats(from: p, to: &st.prevCredit, count: src.count) } }
                env.importance.withUnsafeBufferPointer { src in if let p = src.baseAddress, !src.isEmpty { copyFloats(from: p, to: &st.prevImp, count: src.count) } }
            }
            if let e = engine { let stats = try e.step(env: env, prevAction: st.prevAction, prevCredit: st.prevCredit, prevImp: st.prevImp, prevReward: st.prevReward, capture: willHB || willEnd); try env.step(policy: e.policyPointer, options: e.optionPointer, optionCount: e.K); try finish(e.policyPointer, stats) }
            else { random(&st.rng, &st.randomMean); try st.randomMean.withUnsafeBufferPointer { try env.step(policy: $0.baseAddress!, options: nil, optionCount: 0); try finish($0.baseAddress!, nil) } }
        }
        try Snapshot.write(metric(env, p, seed, steps, st, nil), to: dir)
    }
    private static func worker() throws {
        let e = ProcessInfo.processInfo.environment
        guard let pr = e["OAK_POLICY"], let seed = e["OAK_SEED"].flatMap(Int.init), let stop = e["OAK_STOP_AFTER_STEPS"].flatMap(Int.init) else { throw OakError.message("Incomplete benchmark-worker environment.") }
        try run(policy: try policy(pr), seed: seed, steps: max(stop, 1), dir: Snapshot.dir(nil))
    }
    private static func exePath() throws -> String {
        let r = CommandLine.arguments[0]
        let u = r.hasPrefix("/") ? URL(fileURLWithPath: r) : URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(r)
        guard FileManager.default.isExecutableFile(atPath: u.path) else { throw OakError.message("Cannot locate worker executable.") }
        return u.path
    }
    private static func start(_ item: (Int, Policy), _ steps: Int, _ dir: URL) throws -> pid_t {
        var e = ProcessInfo.processInfo.environment; e["OAK_POLICY"] = item.1.rawValue; e["OAK_SEED"] = "\(item.0)"; e["OAK_STOP_AFTER_STEPS"] = "\(steps)"; e["OAK_METRICS_STORAGE"] = dir.path
        let path = try exePath()
        var pid = pid_t()
        var argv = [strdup(path), strdup("benchmark-worker"), nil]
        var envp = e.map { strdup("\($0.key)=\($0.value)") }
        envp.append(nil)
        defer {
            for ptr in argv { if let ptr { free(ptr) } }
            for ptr in envp { if let ptr { free(ptr) } }
        }
        let rc = path.withCString { cpath in argv.withUnsafeMutableBufferPointer { av in envp.withUnsafeMutableBufferPointer { ep in posix_spawn(&pid, cpath, nil, nil, av.baseAddress, ep.baseAddress) } } }
        guard rc == 0 else { throw OakError.message("posix_spawn failed for worker: \(String(cString: strerror(rc)))") }
        return pid
    }
    private static func benchmark() throws {
        let env = ProcessInfo.processInfo.environment, root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let seed0 = env["OAK_BENCH_SEED_START"].flatMap(Int.init) ?? 0, seeds = env["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 5, steps = max(env["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 500000, 1)
        let policies = try (env["OAK_BENCH_POLICIES"] ?? "oak,random").split(separator: ",").map { try policy(String($0)) }
        var dir = Snapshot.dir(nil); let lockDir = root.appendingPathComponent(".oak-benchmarks", isDirectory: true); try FileManager.default.createDirectory(at: lockDir, withIntermediateDirectories: true)
        let fd = open(lockDir.appendingPathComponent(oakLock).path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR); guard fd >= 0, flock(fd, LOCK_EX) == 0 else { throw OakError.message("Cannot lock benchmark workspace.") }; defer { _ = flock(fd, LOCK_UN); _ = close(fd) }
        if FileManager.default.fileExists(atPath: dir.path), !(try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0 != ".DS_Store" }).isEmpty { dir = URL(fileURLWithPath: "\(dir.path)-\(runId())", isDirectory: true) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let work = (seed0..<(seed0+seeds)).flatMap { s in policies.map { (s, $0) } }, total = work.count, concurrency = max(ProcessInfo.processInfo.processorCount, 1)
        fputs("OaK benchmark: \(total) \(oakEnvId) work items, single-stream B=1, N=\(concurrency) worker processes (steps=\(steps), seeds=\(seeds), dir=\(dir.path))\n", stderr)
        var next = 0, done = 0, running: [pid_t:(Int,Policy)] = [:]
        func fill() throws { while next < total && running.count < concurrency { let item = work[next], pid = try start(item, steps, dir); running[pid] = item; next += 1 } }
        try fill(); let progress = max(1, total / 20)
        while done < total { var st: Int32 = 0; let pid = waitpid(-1, &st, 0); guard pid > 0, let item = running.removeValue(forKey: pid) else { continue }; guard st == 0 else { throw OakError.message("Benchmark worker failed for \(oakEnvId) seed \(item.0) policy \(item.1.rawValue) (status \(st)).") }; done += 1; if done % progress == 0 || done == total { fputs("OaK benchmark progress: \(done)/\(total)\n", stderr) }; try fill() }
    }
}
