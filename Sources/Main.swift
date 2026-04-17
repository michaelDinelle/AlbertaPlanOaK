import Foundation
import GameplayKit
import Metal
import Darwin

enum OakError: Error, CustomStringConvertible {
    case message(String)
    var description: String { switch self { case let .message(message): return message } }
}

enum Policy: String, Codable { case oak, random }

struct RunHealth: Codable {
    let rewardRate: Double?
    let latentPredictionMSE: Double?
    let rewardPredictionMSE: Double?
    let latentStateMeanSquare: Double?
    let activeRank: Double?
    let actionDeltaL2Mean: Double?
    let planningActionChangeL2Mean: Double?
    let optionModelPredictionMSE: Double?
    let planningUsedOptionDynamicsMean: Double?
    let oakOptionReinitCount: Double?
    let nonFiniteRawTotal: Int
    let nonFiniteSanitizedTotal: Int

    enum CodingKeys: String, CodingKey {
        case rewardRate, latentPredictionMSE, rewardPredictionMSE
        case latentStateMeanSquare = "latentVarianceTrace"
        case activeRank, actionDeltaL2Mean, planningActionChangeL2Mean
        case optionModelPredictionMSE, planningUsedOptionDynamicsMean, oakOptionReinitCount
        case nonFiniteRawTotal, nonFiniteSanitizedTotal
    }
}

struct RunDiagnostic: Codable {
    let nonFiniteByPhase: [String: Int]
    let maxAbsLatentPrediction: Double?
    let maxAbsRewardPrediction: Double?
    let maxAbsPolicyOutput: Double?
    let agentShare: Double?
    let envShare: Double?
    let hostShare: Double?
    let metalPhaseGpuMsMean: [String: Double]?
    let metalPhaseGpuTimingInvalid: [String]?
}

private struct DiscreteActionMetrics: Encodable { let histogram: [Int]; let uniqueCount: Int }
private struct ContinuousActionMetrics: Encodable { let meanPerDim: [Double]; let stdPerDim: [Double]; let meanL2: Double; let stdL2: Double }

private final class MeasuredActionCollector {
    private var discrete: [Int]?
    private var sum: [Double]?, sumsq: [Double]?
    private var l2sum = 0.0, l2sumsq = 0.0, n = 0

    init(discreteCount: Int?, actDim: Int) {
        if let c = discreteCount { discrete = [Int](repeating: 0, count: c) }
        else { sum = [Double](repeating: 0, count: actDim); sumsq = [Double](repeating: 0, count: actDim) }
    }

    func addMeasured(_ batch: ActionBatch, numEnvs: Int, actDim: Int) {
        if var h = discrete, let idx = batch.indices {
            for row in 0..<numEnvs { let a = Int(idx[row]); if a >= 0 && a < h.count { h[a] &+= 1 } }
            discrete = h
        }
        guard var s = sum, var q = sumsq else { return }
        let f = batch.floats
        for row in 0..<numEnvs {
            var l2s = 0.0
            let b = row * actDim
            for d in 0..<actDim {
                let v = Double(f[b + d])
                s[d] += v; q[d] += v * v; l2s += v * v
            }
            let l2 = sqrt(l2s); l2sum += l2; l2sumsq += l2 * l2; n += 1
        }
        sum = s; sumsq = q
    }

    func partialMetrics(actDim: Int) -> (DiscreteActionMetrics?, ContinuousActionMetrics?) {
        let d = discrete.map { DiscreteActionMetrics(histogram: $0, uniqueCount: $0.filter { $0 > 0 }.count) }
        guard let s = sum, let q = sumsq, n > 0 else { return (d, nil) }
        let nn = Double(n)
        let mean = (0..<actDim).map { s[$0] / nn }
        let std = (0..<actDim).map { let m = mean[$0]; return sqrt(max(0, q[$0] / nn - m * m)) }
        let ml = l2sum / nn
        return (d, ContinuousActionMetrics(meanPerDim: mean, stdPerDim: std, meanL2: ml, stdL2: sqrt(max(0, l2sumsq / nn - ml * ml))))
    }
}

private struct RunMetricsFile: Encodable {
    let schemaVersion: Int
    let command: String
    let envId: String
    let policy: Policy
    let seed: Int?
    let numEnvs: Int
    let warmupSteps: Int
    let stopAfterMeasuredSteps: Int
    let observationDim: Int
    let actionDim: Int
    let actionDiscreteCount: Int?
    let measuredStepCount: Int
    let measuredBatchCount: Int
    let measuredSeconds: Double
    let stepsPerSecond: Double?
    let heartbeatSequence: Int?
    let rewardSum: Double
    let doneCount: Int
    let stopReason: String
    let agentStepMsMean: Double
    let envStepMsMean: Double
    let discreteAction: DiscreteActionMetrics?
    let continuousAction: ContinuousActionMetrics?
    let health: RunHealth?
    let diagnostic: RunDiagnostic?
}

struct ActionBatch { let floats: [Float]; let indices: [Int32]? }

private struct Schema: Hashable, Comparable {
    let observationDim: Int, actionDim: Int, actionDiscreteCount: Int?
    static func < (lhs: Schema, rhs: Schema) -> Bool {
        (lhs.observationDim, lhs.actionDim, lhs.actionDiscreteCount ?? -1) < (rhs.observationDim, rhs.actionDim, rhs.actionDiscreteCount ?? -1)
    }
}

private struct BenchmarkEnv { let envId: String; let schema: Schema }

private func oakRoots() -> [URL] {
    let fm = FileManager.default
    let exe = URL(fileURLWithPath: CommandLine.arguments[0], relativeTo: URL(fileURLWithPath: fm.currentDirectoryPath)).standardizedFileURL
    return [
        fm.currentDirectoryPath,
        exe.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().path
    ].map { URL(fileURLWithPath: $0).standardizedFileURL }
}

private func requiredBuildPath(_ path: String, description: String, nonEmpty: Bool = false) throws -> URL {
    guard let url = oakRoots().map({ $0.appendingPathComponent(path) }).first(where: {
        guard FileManager.default.fileExists(atPath: $0.path) else { return false }
        return !nonEmpty || ((try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 0
    }) else { throw OakError.message("Missing \(description). Run ./setup.sh first.") }
    return url
}

private func decodeCString(_ buffer: [CChar]) -> String {
    String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

private func roundDownPowerOfTwo(_ value: Int) -> Int {
    guard value > 1 else { return max(value, 1) }
    var p = 1; while p <= value / 2 { p *= 2 }
    return p
}

private func directNumEnvs(obsDim: Int, actDim: Int, actionDiscreteCount: Int?, device: MTLDevice) throws -> Int {
    let width = try OakMetalEngine.executionWidth(device: device)
    let rank = obsDim
    let oc = OakMetalLayout.derivedOptionCount(rank: rank, actDim: actDim)
    // Structural span — sized from interface widths only (no tuned constants); see invariants 1–3.
    let computeSpan = max(rank * (rank + actDim) + oc * (rank + rank * actDim), width)
    let bytesPerEnv = actionDiscreteCount == nil ? (1 << 20) : (1 << 14)
    let memBudget = max(Int(ProcessInfo.processInfo.physicalMemory / 16), width * bytesPerEnv)
    let memSpan = max(memBudget / bytesPerEnv, width)
    let raw = max(roundDownPowerOfTwo(min(computeSpan, memSpan)), width)
    let ceiling = max(ProcessInfo.processInfo.environment["OAK_MAX_PARALLEL_ENVS"].flatMap(Int.init) ?? 4096, width)
    return min(raw, ceiling)
}

private struct HealthAccumulator {
    private struct Mean {
        var sum = 0.0, count = 0
        mutating func add(_ value: Float, raw: inout Int, sanitized: inout Int) {
            let d = Double(value)
            guard d.isFinite else { raw += 1; sanitized += 1; return }
            sum += d; count += 1
        }
        var snapshot: Double? { count > 0 ? sum / Double(count) : nil }
    }

    var steps = 0
    private var lastOnlineRewardRate: Double?
    private var means: [(KeyPath<KernelStats, Float>, Mean, WritableKeyPath<HealthSnapshot, Double?>)]
    var rawNonFinite = 0, sanitizedNonFinite = 0
    private var encodeNonFinite = 0, learnNonFinite = 0, planNonFinite = 0
    private var maxAbsLatentPrediction = 0.0, maxAbsRewardPrediction = 0.0, maxAbsPolicyOutput = 0.0

    private struct HealthSnapshot {
        var latentPredictionMSE: Double?, rewardPredictionMSE: Double?, latentStateMeanSquare: Double?
        var activeRank: Double?, actionDeltaL2Mean: Double?, planningActionChangeL2Mean: Double?
        var optionModelPredictionMSE: Double?, planningUsedOptionDynamicsMean: Double?, oakOptionReinitCount: Double?
    }

    init() {
        means = [
            (\KernelStats.latentPredictionMSE, Mean(), \HealthSnapshot.latentPredictionMSE),
            (\KernelStats.rewardPredictionMSE, Mean(), \HealthSnapshot.rewardPredictionMSE),
            (\KernelStats.latentStateMeanSquare, Mean(), \HealthSnapshot.latentStateMeanSquare),
            (\KernelStats.activeRank, Mean(), \HealthSnapshot.activeRank),
            (\KernelStats.actionDeltaL2Mean, Mean(), \HealthSnapshot.actionDeltaL2Mean),
            (\KernelStats.planningActionChangeL2Mean, Mean(), \HealthSnapshot.planningActionChangeL2Mean),
            (\KernelStats.optionModelPredictionMSE, Mean(), \HealthSnapshot.optionModelPredictionMSE),
            (\KernelStats.planningUsedOptionDynamicsMean, Mean(), \HealthSnapshot.planningUsedOptionDynamicsMean),
            (\KernelStats.oakOptionReinitCount, Mean(), \HealthSnapshot.oakOptionReinitCount),
        ]
    }

    mutating func record(_ stats: KernelStats) {
        steps += 1
        let rr = Double(stats.rewardRate); if rr.isFinite { lastOnlineRewardRate = rr }
        for index in means.indices {
            means[index].1.add(stats[keyPath: means[index].0], raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
        }
        rawNonFinite += Int(stats.rawNonFiniteCount)
        sanitizedNonFinite += Int(stats.sanitizedNonFiniteCount)
        encodeNonFinite += Int(stats.encodeNonFiniteCount)
        learnNonFinite += Int(stats.learnNonFiniteCount)
        planNonFinite += Int(stats.planNonFiniteCount)
        maxAbsLatentPrediction = max(maxAbsLatentPrediction, Double(stats.maxAbsLatentPrediction))
        maxAbsRewardPrediction = max(maxAbsRewardPrediction, Double(stats.maxAbsRewardPrediction))
        maxAbsPolicyOutput = max(maxAbsPolicyOutput, Double(stats.maxAbsPolicyOutput))
    }

    func snapshot() -> RunHealth? {
        guard steps > 0 else { return nil }
        var s = HealthSnapshot()
        for entry in means { s[keyPath: entry.2] = entry.1.snapshot }
        return RunHealth(
            rewardRate: lastOnlineRewardRate,
            latentPredictionMSE: s.latentPredictionMSE,
            rewardPredictionMSE: s.rewardPredictionMSE,
            latentStateMeanSquare: s.latentStateMeanSquare,
            activeRank: s.activeRank,
            actionDeltaL2Mean: s.actionDeltaL2Mean,
            planningActionChangeL2Mean: s.planningActionChangeL2Mean,
            optionModelPredictionMSE: s.optionModelPredictionMSE,
            planningUsedOptionDynamicsMean: s.planningUsedOptionDynamicsMean,
            oakOptionReinitCount: s.oakOptionReinitCount,
            nonFiniteRawTotal: rawNonFinite,
            nonFiniteSanitizedTotal: sanitizedNonFinite
        )
    }

    func diagnosticSnapshot() -> RunDiagnostic? {
        guard steps > 0 else { return nil }
        return RunDiagnostic(
            nonFiniteByPhase: ["encode": encodeNonFinite, "learn": learnNonFinite, "plan": planNonFinite],
            maxAbsLatentPrediction: maxAbsLatentPrediction,
            maxAbsRewardPrediction: maxAbsRewardPrediction,
            maxAbsPolicyOutput: maxAbsPolicyOutput,
            agentShare: nil, envShare: nil, hostShare: nil,
            metalPhaseGpuMsMean: nil, metalPhaseGpuTimingInvalid: nil
        )
    }
}

final class NativeBridge: @unchecked Sendable {
    typealias OpenFn = @convention(c) (UnsafePointer<CChar>, Int32, Int32, UnsafePointer<CChar>, UnsafeMutablePointer<CChar>?, Int32) -> UnsafeMutableRawPointer?
    typealias CloseFn = @convention(c) (UnsafeMutableRawPointer?) -> Void
    typealias IntFn = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    typealias BoundsFn = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<Float>?) -> Int32
    typealias ResetFn = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<CChar>?, Int32) -> Int32
    typealias StepContFn = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Float>?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<UInt8>?, UnsafeMutablePointer<CChar>?, Int32) -> Int32
    typealias StepDiscFn = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Int32>?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<UInt8>?, UnsafeMutablePointer<CChar>?, Int32) -> Int32
    typealias TakeWarningFn = @convention(c) (UnsafeMutablePointer<CChar>?, Int32) -> Int32
    typealias BenchmarkCountFn = @convention(c) () -> Int32
    typealias BenchmarkIdFn = @convention(c) (Int32) -> UnsafePointer<CChar>?

    static let shared: NativeBridge = { do { return try NativeBridge() } catch { fatalError("\(error)") } }()

    private let dylib: UnsafeMutableRawPointer
    private let assetBasePath: String
    private let openFn: OpenFn; private let closeFn: CloseFn
    private let obsDimFn: IntFn; private let actDimFn: IntFn; private let discreteCountFn: IntFn
    private let boundsFn: BoundsFn; private let resetFn: ResetFn
    private let stepContFn: StepContFn; private let stepDiscFn: StepDiscFn
    private let takeWarningFn: TakeWarningFn
    private let benchmarkCountFn: BenchmarkCountFn; private let benchmarkIdFn: BenchmarkIdFn

    private init() throws {
        assetBasePath = try requiredBuildPath(".build/oak-native/env-assets", description: "native env assets").path
        let url = try requiredBuildPath(".build/oak-native/liboakenv.dylib", description: ".build/oak-native/liboakenv.dylib")
        guard let handle = dlopen(url.path, RTLD_NOW | RTLD_LOCAL) else { throw OakError.message(String(cString: dlerror())) }
        dylib = handle
        func load<T>(_ name: String) throws -> T {
            guard let raw = dlsym(handle, name) else { throw OakError.message("Missing symbol \(name) in native env bridge.") }
            return unsafeBitCast(raw, to: T.self)
        }
        openFn = try load("oak_env_open")
        closeFn = try load("oak_env_close")
        obsDimFn = try load("oak_env_obs_dim")
        actDimFn = try load("oak_env_act_dim")
        discreteCountFn = try load("oak_env_discrete_count")
        boundsFn = try load("oak_env_action_bounds")
        resetFn = try load("oak_env_reset")
        stepContFn = try load("oak_env_step_continuous")
        stepDiscFn = try load("oak_env_step_discrete")
        takeWarningFn = try load("oak_env_take_warning")
        benchmarkCountFn = try load("oak_benchmark_env_count")
        benchmarkIdFn = try load("oak_benchmark_env_id")
    }

    deinit { dlclose(dylib) }

    private func errorBuffer() -> [CChar] { Array(repeating: 0, count: 4096) }
    private func checked(_ body: (UnsafeMutablePointer<CChar>?, Int32) -> Int32) throws {
        var error = errorBuffer()
        guard body(&error, Int32(error.count)) == 1 else { throw OakError.message(decodeCString(error)) }
    }
    private func step<Action>(_ handle: UnsafeMutableRawPointer, action: [Action], obs: inout [Float], reward: inout [Float], done: inout [UInt8], _ call: (UnsafeMutableRawPointer?, UnsafePointer<Action>?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<UInt8>?, UnsafeMutablePointer<CChar>?, Int32) -> Int32) throws {
        try checked { error, capacity in
            action.withUnsafeBufferPointer { actionBuffer in
                obs.withUnsafeMutableBufferPointer { obsBuffer in
                    reward.withUnsafeMutableBufferPointer { rewardBuffer in
                        done.withUnsafeMutableBufferPointer { doneBuffer in
                            call(handle, actionBuffer.baseAddress, obsBuffer.baseAddress, rewardBuffer.baseAddress, doneBuffer.baseAddress, error, capacity)
                        }
                    }
                }
            }
        }
    }

    func takeWarning() -> String? {
        var error = errorBuffer()
        return takeWarningFn(&error, Int32(error.count)) == 1 ? decodeCString(error) : nil
    }

    func open(envId: String, numEnvs: Int, seed: Int?) throws -> UnsafeMutableRawPointer {
        var error = errorBuffer()
        let actualSeed = Int32(seed ?? 42)
        let handle = envId.withCString { envCString in
            assetBasePath.withCString { baseCString in
                openFn(envCString, Int32(numEnvs), actualSeed, baseCString, &error, Int32(error.count))
            }
        }
        guard let handle else { throw OakError.message(decodeCString(error)) }
        return handle
    }

    func close(_ handle: UnsafeMutableRawPointer) { closeFn(handle) }
    func obsDim(_ handle: UnsafeMutableRawPointer) -> Int { Int(obsDimFn(handle)) }
    func actDim(_ handle: UnsafeMutableRawPointer) -> Int { Int(actDimFn(handle)) }
    func discreteCount(_ handle: UnsafeMutableRawPointer) -> Int? {
        let count = Int(discreteCountFn(handle)); return count > 0 ? count : nil
    }

    func actionBounds(_ handle: UnsafeMutableRawPointer, actDim: Int) -> ([Float], [Float]) {
        var low = Array(repeating: Float(0), count: actDim)
        var high = Array(repeating: Float(0), count: actDim)
        let count = Int(boundsFn(handle, &low, &high))
        return count > 0 ? (low, high) : ([], [])
    }

    func reset(_ handle: UnsafeMutableRawPointer, obs: inout [Float]) throws { try checked { error, capacity in obs.withUnsafeMutableBufferPointer { resetFn(handle, $0.baseAddress, error, capacity) } } }
    func stepContinuous(_ handle: UnsafeMutableRawPointer, action: [Float], obs: inout [Float], reward: inout [Float], done: inout [UInt8]) throws { try step(handle, action: action, obs: &obs, reward: &reward, done: &done, stepContFn) }
    func stepDiscrete(_ handle: UnsafeMutableRawPointer, action: [Int32], obs: inout [Float], reward: inout [Float], done: inout [UInt8]) throws { try step(handle, action: action, obs: &obs, reward: &reward, done: &done, stepDiscFn) }

    func benchmarkEnvIds() -> [String] {
        (0..<Int(benchmarkCountFn())).compactMap { index in benchmarkIdFn(Int32(index)).map { String(cString: $0) } }
    }
}

final class EnvPoolEnv {
    private let bridge = NativeBridge.shared
    private let handle: UnsafeMutableRawPointer
    private(set) var observationScratch: [Float]
    private(set) var rewardScratch: [Float]
    private(set) var doneScratch: [UInt8]
    let numEnvs: Int, obsDim: Int, actDim: Int
    let actionDiscreteCount: Int?
    let actionLow: [Float], actionHigh: [Float]

    init(id: String, numEnvs: Int, seed: Int?) throws {
        self.numEnvs = numEnvs
        handle = try bridge.open(envId: id, numEnvs: numEnvs, seed: seed)
        obsDim = bridge.obsDim(handle)
        actDim = bridge.actDim(handle)
        actionDiscreteCount = bridge.discreteCount(handle)
        (actionLow, actionHigh) = bridge.actionBounds(handle, actDim: actDim)
        observationScratch = Array(repeating: 0, count: numEnvs * obsDim)
        rewardScratch = Array(repeating: 0, count: numEnvs)
        doneScratch = Array(repeating: 0, count: numEnvs)
    }

    deinit { bridge.close(handle) }
    private func checked<T>(_ body: () throws -> T) throws -> T {
        let value = try body()
        if let warning = bridge.takeWarning() { throw OakError.message("MuJoCo warning: \(warning)") }
        return value
    }

    func reset() throws { try checked { try bridge.reset(handle, obs: &observationScratch) } }

    func step(_ action: ActionBatch) throws {
        try checked {
            if let indices = action.indices { try bridge.stepDiscrete(handle, action: indices, obs: &observationScratch, reward: &rewardScratch, done: &doneScratch) }
            else { try bridge.stepContinuous(handle, action: action.floats, obs: &observationScratch, reward: &rewardScratch, done: &doneScratch) }
        }
    }
}

/// Mirror of `KernelConfig` in `OakKernel.metal`. Same packed field order, same types.
private struct KernelConfig {
    var obsDim: UInt32 = 0, rank: UInt32 = 0, actDim: UInt32 = 0, batchSize: UInt32 = 0
    var actionIsDiscrete: UInt32 = 0, optionCount: UInt32 = 0
    var optionForwardStride: UInt32 = 0, planK: UInt32 = 0, baseObsDim: UInt32 = 0
    var pEncoder: UInt32 = 0, pPolicy: UInt32 = 0, pValue: UInt32 = 0, pReward: UInt32 = 0, pStateWeights: UInt32 = 0
    var pOptPolicy: UInt32 = 0, pOptValues: UInt32 = 0, pOptCritic: UInt32 = 0
    var sObservation: UInt32 = 0, sObsMean: UInt32 = 0, sPrevState: UInt32 = 0, sPrevAction: UInt32 = 0
    var sReward: UInt32 = 0, sState: UInt32 = 0, sAction: UInt32 = 0, sLatentError: UInt32 = 0
    var sRewardPred: UInt32 = 0, sTd: UInt32 = 0, sActionGrad: UInt32 = 0, sRollout: UInt32 = 0
    var sLatentPower: UInt32 = 0
    var uDone: UInt32 = 0, uActive: UInt32 = 0, uStomp: UInt32 = 0, uCounters: UInt32 = 0
    var aOffset: UInt32 = 0
    var aFeatSM: UInt32 = 0, aFeatRes: UInt32 = 0
    var aTdOption: UInt32 = 0, aPlanDelta: UInt32 = 0
    var aGvfW: UInt32 = 0, aGvfH: UInt32 = 0, aGvfU: UInt32 = 0, aGvfTd: UInt32 = 0, aGvfRho: UInt32 = 0
    var aSubRho: UInt32 = 0, aOptBeta: UInt32 = 0, aOptCounts: UInt32 = 0
    var aKeyboard: UInt32 = 0, aKeyboardPred: UInt32 = 0, aOptScore: UInt32 = 0, aOptScoreSum: UInt32 = 0
    var aPlanState: UInt32 = 0, aPlanScratch: UInt32 = 0
    var aLearnGrad: UInt32 = 0, aGvfGrad: UInt32 = 0
    var aUtilFeat: UInt32 = 0, aUtilOpt: UInt32 = 0
    var nGvfHeads: UInt32 = 0
    var capacityRidge: Float = 0, structuralCap: Float = 0, nlmsStepScale: Float = 0
    var rsqrtPlanLambda: Float = 0
    var boundLat: Float = 0, boundRew: Float = 0, boundAct: Float = 0
    var invBoundLat: Float = 0, invBoundRew: Float = 0, invBoundAct: Float = 0
}

private struct KernelScalars {
    var stepCount: UInt32 = 0
    var doneCount: UInt32 = 0
    var activeCount: UInt32 = 0
    var pad: UInt32 = 0
    var rewardRate: Float = 0
    var pEff: Float = 0
}

private enum OakMetalLayout {
    static let kernelConfigBufferIndex = 6

    static func derivedOptionCount(rank: Int, actDim: Int) -> Int { max(1, min(rank, actDim)) }
    static func gvfBankFloatCount(rank: Int) -> Int { rank * rank }
    static func optionForwardStride(rank: Int, actDim: Int) -> Int { rank * (rank + actDim) }
    static func packedStateWeightsSize(rank: Int, actDim: Int) -> Int {
        let oc = derivedOptionCount(rank: rank, actDim: actDim)
        return gvfBankFloatCount(rank: rank) + oc * optionForwardStride(rank: rank, actDim: actDim)
    }
    static func structuralPlanK(rank: Int) -> Int { min(4, max(rank, 1)) }
    /// Single structural cap on GVF horde width: tied to rank (no free constant).
    static func structuralMaxGvfHeads(rank: Int) -> Int { max(rank, 1) }
    static func optionPolicyParameterCount(rank: Int, actDim: Int) -> Int {
        derivedOptionCount(rank: rank, actDim: actDim) * rank * actDim
    }
    static func optionValueParameterCount(rank: Int, actDim: Int) -> Int {
        derivedOptionCount(rank: rank, actDim: actDim) * rank
    }
    static func optionCriticBufferSize(rank: Int, actDim: Int) -> Int {
        let oc = derivedOptionCount(rank: rank, actDim: actDim); return oc * rank + oc
    }
    static func discreteLogitHalfWidth(rank: Int, actDim: Int) -> Float {
        min(sqrtf(Float(max(rank * actDim, 1))), logf(Float.greatestFiniteMagnitude))
    }
    static func statsAbsBoundForSumSquares(termCount: Int) -> Float {
        sqrtf(Float.greatestFiniteMagnitude / Float(max(termCount, 1)))
    }

    struct ParamArenaLayout {
        let encoder: Int, policy: Int, value: Int, reward: Int, stateWeights: Int
        let optionPolicy: Int, optionValues: Int, optionCritic: Int, totalFloats: Int
    }

    static func paramArenaLayout(obsDim: Int, rank: Int, actDim: Int) -> ParamArenaLayout {
        var o = 0
        let enc = o; o += obsDim
        let pol = o; o += actDim * rank
        let val = o; o += rank
        let rew = o; o += rank
        let st = o; o += packedStateWeightsSize(rank: rank, actDim: actDim)
        let op = o; o += optionPolicyParameterCount(rank: rank, actDim: actDim)
        let ov = o; o += optionValueParameterCount(rank: rank, actDim: actDim)
        let ocr = o; o += optionCriticBufferSize(rank: rank, actDim: actDim)
        return ParamArenaLayout(encoder: enc, policy: pol, value: val, reward: rew, stateWeights: st, optionPolicy: op, optionValues: ov, optionCritic: ocr, totalFloats: o)
    }

    struct ArenaLayout {
        let featSM: Int, featRes: Int, tdOption: Int, planDelta: Int
        let gvfW: Int, gvfH: Int, gvfU: Int, gvfTd: Int, gvfRho: Int
        let subRho: Int, optBeta: Int, optCounts: Int
        let keyboard: Int, keyboardPred: Int, optScore: Int, optScoreSum: Int
        let planState: Int, planScratch: Int, learnGrad: Int, gvfGrad: Int
        let utilFeat: Int, utilOpt: Int
        let totalFloats: Int
    }

    /// Float arena tail (read by the kernel after the latentPower core slots `[2*batch]`).
    static func arenaLayout(rank: Int, actDim: Int, batchSize: Int) -> ArenaLayout {
        let oc = derivedOptionCount(rank: rank, actDim: actDim)
        let nGvf = structuralMaxGvfHeads(rank: rank)
        var o = 0
        let featSM = o; o += rank
        let featRes = o; o += rank
        let tdOption = o; o += batchSize
        let planDelta = o; o += batchSize * actDim
        let gvfW = o; o += rank * nGvf
        let gvfH = o; o += rank * nGvf
        let gvfU = o; o += rank * nGvf
        let gvfTd = o; o += batchSize * nGvf
        let gvfRho = o; o += 1
        let subRho = o; o += oc
        let optBeta = o; o += oc
        let optCounts = o; o += oc
        let keyboard = o; o += batchSize * oc
        let keyboardPred = o; o += oc
        let optScore = o; o += batchSize * oc
        let optScoreSum = o; o += batchSize
        let planState = o; o += batchSize * rank
        let planScratch = o; o += batchSize * rank
        let learnGrad = o; o += rank * (rank + actDim)
        let gvfGrad = o; o += rank * rank
        let utilFeat = o; o += rank
        let utilOpt = o; o += oc
        return ArenaLayout(featSM: featSM, featRes: featRes, tdOption: tdOption, planDelta: planDelta, gvfW: gvfW, gvfH: gvfH, gvfU: gvfU, gvfTd: gvfTd, gvfRho: gvfRho, subRho: subRho, optBeta: optBeta, optCounts: optCounts, keyboard: keyboard, keyboardPred: keyboardPred, optScore: optScore, optScoreSum: optScoreSum, planState: planState, planScratch: planScratch, learnGrad: learnGrad, gvfGrad: gvfGrad, utilFeat: utilFeat, utilOpt: utilOpt, totalFloats: o)
    }

    /// Core latentPower slots: [active option per row | next active option per row] then arena tail.
    static func latentPowerCore(batchSize: Int) -> Int { 2 * batchSize }

    struct StateArenaLayout {
        let observation: Int, observationMean: Int
        let previousState: Int, previousAction: Int
        let reward: Int, state: Int, action: Int
        let latentError: Int, rewardPrediction: Int, td: Int, actionGrad: Int, rolloutState: Int
        let latentPower: Int, totalFloats: Int
    }

    static func stateArenaLayout(obsDim: Int, rank: Int, actDim: Int, batchSize: Int) -> StateArenaLayout {
        var o = 0
        let obs = o; o += batchSize * obsDim
        let mean = o; o += obsDim
        let prevS = o; o += batchSize * rank
        let prevA = o; o += batchSize * actDim
        let rew = o; o += batchSize
        let st = o; o += batchSize * rank
        let act = o; o += batchSize * actDim
        let latE = o; o += batchSize * rank
        let rPred = o; o += batchSize
        let td = o; o += batchSize
        let aG = o; o += batchSize * actDim
        let roll = o; o += batchSize * rank
        let lp = o
        let lpSz = latentPowerCore(batchSize: batchSize) + arenaLayout(rank: rank, actDim: actDim, batchSize: batchSize).totalFloats
        o += lpSz
        return StateArenaLayout(observation: obs, observationMean: mean, previousState: prevS, previousAction: prevA, reward: rew, state: st, action: act, latentError: latE, rewardPrediction: rPred, td: td, actionGrad: aG, rolloutState: roll, latentPower: lp, totalFloats: o)
    }

    struct UIntArenaLayout { let done: Int, active: Int, stomp: Int, counters: Int, totalUInts: Int }

    static func uintArenaLayout(rank: Int, batchSize: Int) -> UIntArenaLayout {
        var o = 0
        let d = o; o += batchSize
        let a = o; o += 2 * rank
        let st = o; o += max(5 * batchSize, 5)
        let c = o; o += 5
        return UIntArenaLayout(done: d, active: a, stomp: st, counters: c, totalUInts: o)
    }
}

fileprivate struct KernelStats {
    var rewardRate: Float = 0, latentPredictionMSE: Float = 0, rewardPredictionMSE: Float = 0
    var latentStateMeanSquare: Float = 0, activeRank: Float = 0
    var actionDeltaL2Mean: Float = 0, planningActionChangeL2Mean: Float = 0
    var rawNonFiniteCount: Float = 0, sanitizedNonFiniteCount: Float = 0
    var encodeNonFiniteCount: Float = 0, learnNonFiniteCount: Float = 0, planNonFiniteCount: Float = 0
    var maxAbsLatentPrediction: Float = 0, maxAbsRewardPrediction: Float = 0, maxAbsPolicyOutput: Float = 0
    var optionModelPredictionMSE: Float = 0
    var planningUsedOptionDynamicsMean: Float = 0
    var oakOptionReinitCount: Float = 0
    var utilityFeatureMean: Float = 0, utilitySubtaskMean: Float = 0, utilityOptionMean: Float = 0, utilityPlanningMean: Float = 0
}

private final class SharedArray<Element> {
    let count: Int
    let buffer: MTLBuffer
    let pointer: UnsafeMutablePointer<Element>

    init(device: MTLDevice, count: Int, repeating value: Element) {
        self.count = count
        buffer = device.makeBuffer(length: max(count, 1) * MemoryLayout<Element>.stride, options: [.storageModeShared])!
        pointer = buffer.contents().bindMemory(to: Element.self, capacity: max(count, 1))
        pointer.initialize(repeating: value, count: max(count, 1))
    }
}

private typealias SharedFloats = SharedArray<Float>
private typealias SharedHalfs = SharedArray<Float16>
private typealias SharedUInts = SharedArray<UInt32>

private final class DiscreteSoftmaxBehavior {
    private let rng: GKMersenneTwisterRandomSource
    private var expScratch: [Float]
    private let actDim: Int

    init(actDim: Int, rng: GKMersenneTwisterRandomSource) {
        self.actDim = actDim
        expScratch = [Float](repeating: 0, count: actDim)
        self.rng = rng
    }

    func apply(hostAction: inout [Float], hostDiscrete: inout [Int32], batchSize: Int) {
        for row in 0..<batchSize {
            let o = row * actDim
            var mx = hostAction[o]
            for i in 1..<actDim { mx = max(mx, hostAction[o + i]) }
            var s: Float = 0
            for i in 0..<actDim { let e = exp(hostAction[o + i] - mx); expScratch[i] = e; s += e }
            let invS = 1 / s
            let u = rng.nextUniform()
            var c: Float = 0
            var pick = actDim - 1
            for i in 0..<actDim { c += expScratch[i] * invS; if u < c { pick = i; break } }
            hostDiscrete[row] = Int32(pick)
            for i in 0..<actDim { hostAction[o + i] = i == pick ? 1 : 0 }
        }
    }
}

final class OakMetalEngine {
    private let obsDim: Int, baseObsDim: Int, actDim: Int, batchSize: Int, rank: Int
    private let actionIsDiscrete: Bool
    private let queue: MTLCommandQueue
    private let commandBufferDescriptor: MTLCommandBufferDescriptor
    private let pipelines: [MTLComputePipelineState]
    private let actionBoundsBuffer: MTLBuffer
    private let scalarsBuffer: MTLBuffer
    private let statsBuffer: MTLBuffer
    private let configBuffer: MTLBuffer
    private let scalarsPointer: UnsafeMutablePointer<KernelScalars>
    private let statsPointer: UnsafeMutablePointer<KernelStats>
    private let configPointer: UnsafeMutablePointer<KernelConfig>
    private let bufferZeroOffsets: [Int]

    private let paramsArena: SharedHalfs
    private let stateArena: SharedFloats
    private let uintArena: SharedUInts
    private let paramLayout: OakMetalLayout.ParamArenaLayout
    private let stateLayout: OakMetalLayout.StateArenaLayout
    private let uintLayout: OakMetalLayout.UIntArenaLayout
    private let arenaLayout: OakMetalLayout.ArenaLayout
    private let arenaBaseInLatent: Int
    private let metalDebugSignposts: Bool
    private let diagnosticGpuPhaseTiming: Bool
    private var phaseGpuTimeSumSeconds: [Double]
    private var phaseGpuInvalidStepCount: [Int]
    private var hostAction: [Float]
    private var hostDiscrete: [Int32]
    private let discreteBehavior: DiscreteSoftmaxBehavior?

    private static let phaseKernelNames = [
        "oak_encode", "oak_forward", "oak_predict", "oak_learn",
        "oak_meta_row", "oak_meta_action", "oak_plan",
        "oak_bk_reduce", "oak_bk_option", "oak_bk_feature", "oak_bk_admit", "oak_bk_commit",
    ]

    private static func makePipelines(device: MTLDevice, planningIsDiscrete: Bool) throws -> [MTLComputePipelineState] {
        let library = try device.makeLibrary(URL: try kernelURL())
        return try phaseKernelNames.map { name in
            let fc = MTLFunctionConstantValues()
            var discrete = planningIsDiscrete
            fc.setConstantValue(&discrete, type: .bool, index: 0)
            guard let function = try? library.makeFunction(name: name, constantValues: fc) else {
                throw OakError.message("Missing \(name) in OakKernel.metallib.")
            }
            return try device.makeComputePipelineState(function: function)
        }
    }

    static func executionWidth(device: MTLDevice) throws -> Int {
        let states = try makePipelines(device: device, planningIsDiscrete: false)
        var w = 1; for p in states { w = max(w, p.threadExecutionWidth) }
        return max(w, 1)
    }

    private static func kernelURL() throws -> URL {
        try requiredBuildPath(".build/OakKernel.metallib", description: "non-empty .build/OakKernel.metallib", nonEmpty: true)
    }

    init(obsDim: Int, baseObsDim: Int, actDim: Int, batchSize: Int, rank: Int, actionIsDiscrete: Bool, actionLow: [Float], actionHigh: [Float], device: MTLDevice, seed: Int?, diagnosticGpuPhaseTiming: Bool = false) throws {
        precondition(obsDim == rank, "obsDim must equal rank (interface-derived latent width).")
        precondition(baseObsDim <= obsDim && baseObsDim >= 1, "baseObsDim must be positive and ≤ obsDim.")
        self.obsDim = obsDim; self.baseObsDim = baseObsDim; self.actDim = actDim
        self.batchSize = batchSize; self.rank = rank; self.actionIsDiscrete = actionIsDiscrete
        guard let queue = device.makeCommandQueue() else { throw OakError.message("Failed to create Metal command queue.") }
        self.queue = queue
        let cbDesc = MTLCommandBufferDescriptor(); cbDesc.retainedReferences = false
        commandBufferDescriptor = cbDesc
        pipelines = try Self.makePipelines(device: device, planningIsDiscrete: actionIsDiscrete)

        let half = OakMetalLayout.discreteLogitHalfWidth(rank: rank, actDim: actDim)
        let lows = actionIsDiscrete ? Array(repeating: -half, count: actDim) : actionLow
        let highs = actionIsDiscrete ? Array(repeating: half, count: actDim) : actionHigh
        actionBoundsBuffer = device.makeBuffer(bytes: lows + highs, length: 2 * actDim * MemoryLayout<Float>.stride, options: [.storageModeShared])!
        scalarsBuffer = device.makeBuffer(length: MemoryLayout<KernelScalars>.stride, options: [.storageModeShared])!
        statsBuffer = device.makeBuffer(length: MemoryLayout<KernelStats>.stride, options: [.storageModeShared])!
        configBuffer = device.makeBuffer(length: MemoryLayout<KernelConfig>.stride, options: [.storageModeShared])!
        scalarsPointer = scalarsBuffer.contents().bindMemory(to: KernelScalars.self, capacity: 1)
        statsPointer = statsBuffer.contents().bindMemory(to: KernelStats.self, capacity: 1)
        configPointer = configBuffer.contents().bindMemory(to: KernelConfig.self, capacity: 1)
        scalarsPointer.initialize(to: KernelScalars())
        scalarsPointer.pointee.pEff = Float(rank)
        statsPointer.initialize(to: KernelStats())
        configPointer.initialize(to: KernelConfig())
        bufferZeroOffsets = [Int](repeating: 0, count: 6)

        paramLayout = OakMetalLayout.paramArenaLayout(obsDim: obsDim, rank: rank, actDim: actDim)
        stateLayout = OakMetalLayout.stateArenaLayout(obsDim: obsDim, rank: rank, actDim: actDim, batchSize: batchSize)
        uintLayout = OakMetalLayout.uintArenaLayout(rank: rank, batchSize: batchSize)
        arenaLayout = OakMetalLayout.arenaLayout(rank: rank, actDim: actDim, batchSize: batchSize)
        arenaBaseInLatent = OakMetalLayout.latentPowerCore(batchSize: batchSize)
        paramsArena = SharedHalfs(device: device, count: paramLayout.totalFloats, repeating: Float16(0))
        stateArena = SharedFloats(device: device, count: stateLayout.totalFloats, repeating: 0)
        uintArena = SharedUInts(device: device, count: uintLayout.totalUInts, repeating: 0)

        metalDebugSignposts = ProcessInfo.processInfo.environment["OAK_METAL_SIGNPOSTS"] == "1"
        self.diagnosticGpuPhaseTiming = diagnosticGpuPhaseTiming
        phaseGpuTimeSumSeconds = Array(repeating: 0, count: Self.phaseKernelNames.count)
        phaseGpuInvalidStepCount = Array(repeating: 0, count: Self.phaseKernelNames.count)
        hostAction = Array(repeating: 0, count: batchSize * actDim)
        hostDiscrete = Array(repeating: 0, count: batchSize)

        let rng = seed.map { GKMersenneTwisterRandomSource(seed: UInt64(bitPattern: Int64($0))) }
        func uniform() -> Float { rng?.nextUniform() ?? Float.random(in: 0..<1) }
        discreteBehavior = actionIsDiscrete ? DiscreteSoftmaxBehavior(actDim: actDim, rng: rng ?? GKMersenneTwisterRandomSource()) : nil

        // ---- Param arena init: encoder = 1/√rank (structural), all other heads = small Gaussian-ish
        // probe. Identity init on the option-value diagonal so option k starts with a non-zero
        // signal toward feature k (RRS bonus needs this; A.6 — single 1/√rank scale).
        let invRoot = Float(1.0 / sqrt(Double(max(rank, 1))))
        let initScale = Float(1.0 / sqrt(Double(max(1 + actDim, 1))))
        func fill(_ off: Int, _ count: Int, _ scale: Float) {
            let base = paramsArena.pointer.advanced(by: off)
            for i in 0..<count { base[i] = Float16((uniform() * 2 - 1) * scale) }
        }
        fill(paramLayout.encoder, obsDim, invRoot)
        fill(paramLayout.policy, actDim * rank, initScale)
        fill(paramLayout.value, rank, initScale)
        fill(paramLayout.reward, rank, initScale)
        fill(paramLayout.stateWeights, OakMetalLayout.packedStateWeightsSize(rank: rank, actDim: actDim), initScale)
        fill(paramLayout.optionPolicy, OakMetalLayout.optionPolicyParameterCount(rank: rank, actDim: actDim), initScale)
        fill(paramLayout.optionValues, OakMetalLayout.optionValueParameterCount(rank: rank, actDim: actDim), initScale)
        fill(paramLayout.optionCritic, OakMetalLayout.optionCriticBufferSize(rank: rank, actDim: actDim), initScale)
        let oc = OakMetalLayout.derivedOptionCount(rank: rank, actDim: actDim)
        let ov = paramsArena.pointer.advanced(by: paramLayout.optionValues)
        for k in 0..<oc where k < rank {
            ov[k * rank + k] = Float16(Float(ov[k * rank + k]) + invRoot)
        }

        // ---- Active mask: all features admitted on step 0; meta-policy fills latentPower[batch..2*batch].
        scalarsPointer.pointee.activeCount = UInt32(rank)
        let activePtr = uintArena.pointer.advanced(by: uintLayout.active)
        for f in 0..<rank { activePtr[f] = 1; activePtr[rank + f] = UInt32(f) }

        // ---- Arena init: keyboard chord = uniform 1/oc; subRho/optBeta/utilFeat/utilOpt = 0.
        let arenaPtr = stateArena.pointer.advanced(by: stateLayout.latentPower + arenaBaseInLatent)
        let invKey = 1.0 / Float(max(oc, 1))
        for row in 0..<batchSize { for k in 0..<oc { arenaPtr[arenaLayout.keyboard + row * oc + k] = invKey } }
        for k in 0..<oc {
            arenaPtr[arenaLayout.optBeta + k] = 1.0 / Float(max(oc + rank, 1))
        }

        // ---- GVF horde init: U column 0 = unit vector; W cols = small structural noise.
        let nGvf = OakMetalLayout.structuralMaxGvfHeads(rank: rank)
        for j in 0..<nGvf {
            let scale = invRoot
            for f in 0..<rank {
                arenaPtr[arenaLayout.gvfW + j * rank + f] = (uniform() * 2 - 1) * scale
                arenaPtr[arenaLayout.gvfH + j * rank + f] = 0
                arenaPtr[arenaLayout.gvfU + j * rank + f] = (j == 0 && f == 0) ? 1 : 0
            }
        }

        writeKernelConfig()
    }

    fileprivate func step(obs: UnsafeBufferPointer<Float>, prevAction: [Float], prevReward: [Float], prevDone: [UInt8], accumulatePhaseGpu: Bool) throws -> (ActionBatch, Double, KernelStats) {
        let start = DispatchTime.now().uptimeNanoseconds
        // (a) Promote next-step option (written by meta_action at index batch+row) to active option (index row).
        let lp = stateArena.pointer.advanced(by: stateLayout.latentPower)
        memcpy(lp, lp.advanced(by: batchSize), batchSize * MemoryLayout<Float>.stride)
        // (b) Stage per-step inputs: observation, prev reward, prev action, prev state.
        let obsCount = batchSize * obsDim
        precondition(obs.count == obsCount)
        memcpy(stateArena.pointer.advanced(by: stateLayout.observation), obs.baseAddress!, obsCount * MemoryLayout<Float>.stride)
        precondition(prevReward.count == batchSize)
        memcpy(stateArena.pointer.advanced(by: stateLayout.reward), prevReward, batchSize * MemoryLayout<Float>.stride)
        precondition(prevAction.count == batchSize * actDim)
        memcpy(stateArena.pointer.advanced(by: stateLayout.previousAction), prevAction, batchSize * actDim * MemoryLayout<Float>.stride)
        memcpy(stateArena.pointer.advanced(by: stateLayout.previousState), stateArena.pointer.advanced(by: stateLayout.state), batchSize * rank * MemoryLayout<Float>.stride)
        let donePtr = uintArena.pointer.advanced(by: uintLayout.done)
        for index in 0..<batchSize { donePtr[index] = UInt32(prevDone[index]) }
        // (c) Per-step zero of plan bookkeeping arena tail entries (arena lives past latentPower core).
        let arenaPtr = lp.advanced(by: arenaBaseInLatent)
        memset(arenaPtr.advanced(by: arenaLayout.planDelta), 0, batchSize * actDim * MemoryLayout<Float>.stride)
        let ctr = uintArena.pointer.advanced(by: uintLayout.counters)
        for i in 0..<5 { ctr[i] = 0 }
        statsPointer.pointee = KernelStats()
        // (d) Dispatch all phases (one command buffer or one-per-phase for diagnostic GPU timing).
        try runKernel(accumulatePhaseGpu: accumulatePhaseGpu)
        memcpy(&hostAction, stateArena.pointer.advanced(by: stateLayout.action), batchSize * actDim * MemoryLayout<Float>.stride)
        discreteBehavior?.apply(hostAction: &hostAction, hostDiscrete: &hostDiscrete, batchSize: batchSize)
        return (ActionBatch(floats: hostAction, indices: actionIsDiscrete ? hostDiscrete : nil),
                Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000,
                statsPointer.pointee)
    }

    private func kernelBuffers() -> [MTLBuffer?] {
        [paramsArena.buffer, stateArena.buffer, uintArena.buffer, scalarsBuffer, statsBuffer, actionBoundsBuffer]
    }

    /// Phase grid widths — split bookkeeping into 4 parallel kernels (B.1) sharing the arena/context.
    private func phaseGrid() -> [Int] {
        let oc = OakMetalLayout.derivedOptionCount(rank: rank, actDim: actDim)
        return [
            batchSize * rank,           // encode
            batchSize * rank,           // forward
            batchSize,                  // predict
            rank,                       // learn
            batchSize,                  // meta_row
            batchSize * actDim,         // meta_action
            batchSize,                  // plan
            1,                          // bk_reduce
            oc,                         // bk_option
            rank,                       // bk_feature
            1,                          // bk_admit
            1,                          // bk_commit
        ]
    }

    private func writeKernelConfig() {
        let oc = OakMetalLayout.derivedOptionCount(rank: rank, actDim: actDim)
        let ofs = OakMetalLayout.optionForwardStride(rank: rank, actDim: actDim)
        let nGvf = OakMetalLayout.structuralMaxGvfHeads(rank: rank)
        let cap = Float(max(rank * max(batchSize, 1), 1))
        let p = paramLayout, s = stateLayout, u = uintLayout, a = arenaLayout
        var c = KernelConfig()
        c.obsDim = UInt32(obsDim); c.rank = UInt32(rank); c.actDim = UInt32(actDim); c.batchSize = UInt32(batchSize)
        c.actionIsDiscrete = actionIsDiscrete ? 1 : 0
        c.optionCount = UInt32(oc); c.optionForwardStride = UInt32(ofs)
        c.planK = UInt32(OakMetalLayout.structuralPlanK(rank: rank))
        c.baseObsDim = UInt32(baseObsDim)
        c.pEncoder = UInt32(p.encoder); c.pPolicy = UInt32(p.policy); c.pValue = UInt32(p.value)
        c.pReward = UInt32(p.reward); c.pStateWeights = UInt32(p.stateWeights)
        c.pOptPolicy = UInt32(p.optionPolicy); c.pOptValues = UInt32(p.optionValues); c.pOptCritic = UInt32(p.optionCritic)
        c.sObservation = UInt32(s.observation); c.sObsMean = UInt32(s.observationMean)
        c.sPrevState = UInt32(s.previousState); c.sPrevAction = UInt32(s.previousAction)
        c.sReward = UInt32(s.reward); c.sState = UInt32(s.state); c.sAction = UInt32(s.action)
        c.sLatentError = UInt32(s.latentError); c.sRewardPred = UInt32(s.rewardPrediction)
        c.sTd = UInt32(s.td); c.sActionGrad = UInt32(s.actionGrad); c.sRollout = UInt32(s.rolloutState)
        c.sLatentPower = UInt32(s.latentPower)
        c.uDone = UInt32(u.done); c.uActive = UInt32(u.active); c.uStomp = UInt32(u.stomp); c.uCounters = UInt32(u.counters)
        c.aOffset = UInt32(arenaBaseInLatent)
        c.aFeatSM = UInt32(a.featSM); c.aFeatRes = UInt32(a.featRes)
        c.aTdOption = UInt32(a.tdOption); c.aPlanDelta = UInt32(a.planDelta)
        c.aGvfW = UInt32(a.gvfW); c.aGvfH = UInt32(a.gvfH); c.aGvfU = UInt32(a.gvfU)
        c.aGvfTd = UInt32(a.gvfTd); c.aGvfRho = UInt32(a.gvfRho)
        c.aSubRho = UInt32(a.subRho); c.aOptBeta = UInt32(a.optBeta); c.aOptCounts = UInt32(a.optCounts)
        c.aKeyboard = UInt32(a.keyboard); c.aKeyboardPred = UInt32(a.keyboardPred)
        c.aOptScore = UInt32(a.optScore); c.aOptScoreSum = UInt32(a.optScoreSum)
        c.aPlanState = UInt32(a.planState); c.aPlanScratch = UInt32(a.planScratch)
        c.aLearnGrad = UInt32(a.learnGrad); c.aGvfGrad = UInt32(a.gvfGrad)
        c.aUtilFeat = UInt32(a.utilFeat); c.aUtilOpt = UInt32(a.utilOpt)
        c.nGvfHeads = UInt32(nGvf)
        c.capacityRidge = 1.0 / (cap + 1.0)
        c.structuralCap = cap
        c.nlmsStepScale = 1.0 / cap
        c.rsqrtPlanLambda = 1.0 / sqrtf(Float(max(rank + actDim, 1)))
        let nLat = max(batchSize * rank, 1), nRew = max(batchSize, 1), nAct = max(batchSize * actDim, 1)
        c.boundLat = OakMetalLayout.statsAbsBoundForSumSquares(termCount: nLat)
        c.boundRew = OakMetalLayout.statsAbsBoundForSumSquares(termCount: nRew)
        c.boundAct = OakMetalLayout.statsAbsBoundForSumSquares(termCount: nAct)
        c.invBoundLat = 1.0 / c.boundLat; c.invBoundRew = 1.0 / c.boundRew; c.invBoundAct = 1.0 / c.boundAct
        configPointer.pointee = c
    }

    fileprivate func metalPhaseGpuDiagnostics(measuredBatchSteps: Int) -> (meanMs: [String: Double]?, invalidPhases: [String]?) {
        guard diagnosticGpuPhaseTiming, measuredBatchSteps > 0 else { return (nil, nil) }
        var mean: [String: Double] = [:]
        var invalid: [String] = []
        let d = Double(measuredBatchSteps)
        for index in 0..<Self.phaseKernelNames.count {
            let name = Self.phaseKernelNames[index]
            mean[name] = phaseGpuTimeSumSeconds[index] / d * 1000
            if phaseGpuInvalidStepCount[index] > 0 { invalid.append(name) }
        }
        return (mean, invalid.isEmpty ? nil : invalid)
    }

    private func alignedGridWidth(threadCount: Int, pipeline: MTLComputePipelineState) -> Int {
        let align = max(Int(pipeline.threadExecutionWidth), 32)
        let n = max(threadCount, 1)
        return ((n + align - 1) / align) * align
    }

    /// Single dispatcher (C dedup). When `splitPerPhase` is true (diagnose), one command buffer per
    /// phase records `gpuStartTime`/`gpuEndTime`; otherwise one buffer encodes all phases (fast path).
    private func runKernel(accumulatePhaseGpu: Bool) throws {
        let buffers = kernelBuffers()
        let grid = phaseGrid()
        let split = diagnosticGpuPhaseTiming && accumulatePhaseGpu
        if !split {
            guard let cb = queue.makeCommandBuffer(descriptor: commandBufferDescriptor) else { throw OakError.message("Failed to create Metal command buffer.") }
            guard let enc = cb.makeComputeCommandEncoder() else { throw OakError.message("Failed to create Metal compute encoder.") }
            enc.setBuffers(buffers, offsets: bufferZeroOffsets, range: 0..<6)
            enc.setBuffer(configBuffer, offset: 0, index: OakMetalLayout.kernelConfigBufferIndex)
            for index in 0..<pipelines.count {
                let pip = pipelines[index]
                enc.setComputePipelineState(pip)
                if metalDebugSignposts { enc.insertDebugSignpost(Self.phaseKernelNames[index]) }
                let tg = min(max(pip.threadExecutionWidth, 1), pip.maxTotalThreadsPerThreadgroup)
                let w = alignedGridWidth(threadCount: grid[index], pipeline: pip)
                enc.dispatchThreads(MTLSize(width: w, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(tg, w), height: 1, depth: 1))
            }
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            if let error = cb.error { throw OakError.message("Metal command failed: \(error)") }
            return
        }
        for index in 0..<pipelines.count {
            let pip = pipelines[index]
            guard let cb = queue.makeCommandBuffer(descriptor: commandBufferDescriptor) else { throw OakError.message("Failed to create Metal command buffer.") }
            guard let enc = cb.makeComputeCommandEncoder() else { throw OakError.message("Failed to create Metal compute encoder.") }
            enc.setComputePipelineState(pip)
            enc.setBuffers(buffers, offsets: bufferZeroOffsets, range: 0..<6)
            enc.setBuffer(configBuffer, offset: 0, index: OakMetalLayout.kernelConfigBufferIndex)
            if metalDebugSignposts { enc.insertDebugSignpost(Self.phaseKernelNames[index]) }
            let tg = min(max(pip.threadExecutionWidth, 1), pip.maxTotalThreadsPerThreadgroup)
            let w = alignedGridWidth(threadCount: grid[index], pipeline: pip)
            enc.dispatchThreads(MTLSize(width: w, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(tg, w), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            if let error = cb.error { throw OakError.message("Metal command failed: \(error)") }
            let s = cb.gpuStartTime, e = cb.gpuEndTime
            if s > 0, e >= s { phaseGpuTimeSumSeconds[index] += e - s }
            else { phaseGpuInvalidStepCount[index] += 1 }
        }
    }
}

private enum RunSnapshot {
    private static let persistLock = NSLock()

    private static func storageFromEnvironment() -> URL? {
        guard let raw = ProcessInfo.processInfo.environment["OAK_METRICS_STORAGE"]?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
        return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
    }

    fileprivate static func resolveMetricsDirectory(explicit: URL?) -> URL {
        if let explicit { return explicit }
        if let s = storageFromEnvironment() { return s }
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return root.appendingPathComponent(".oak-benchmarks/\(Main.oakBenchmarksRunId())", isDirectory: true)
    }

    static func persist(_ file: RunMetricsFile, metricsDir: URL) throws -> URL {
        let stem = snapshotStem(envId: file.envId, policy: file.policy, seed: file.seed)
        let leaf = file.heartbeatSequence.map { "\(stem).heartbeat-\(String(format: "%06d", $0)).json" } ?? "\(stem).json"
        let seedDir = metricsDir
            .appendingPathComponent(file.envId, isDirectory: true)
            .appendingPathComponent(seedFolderName(file.seed), isDirectory: true)
        let url = seedDir.appendingPathComponent(leaf, isDirectory: false)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try enc.encode(file)
        persistLock.lock(); defer { persistLock.unlock() }
        try FileManager.default.createDirectory(at: seedDir, withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
        return url
    }

    fileprivate static func snapshotStem(envId: String, policy: Policy, seed: Int?) -> String {
        let raw = "\(envId)-\(policy.rawValue)-\(seedFolderName(seed))"
        return raw.replacingOccurrences(of: #"[^A-Za-z0-9]+"#, with: "-", options: .regularExpression)
    }

    fileprivate static func seedFolderName(_ seed: Int?) -> String {
        seed.map { "seed\($0)" } ?? "seedn"
    }
}

@main struct Main {
    static func main() {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            switch args.first {
            case "benchmark": try benchmark()
            case "diagnose": try diagnose(Array(args.dropFirst()))
            case "run": try run(args.dropFirst().first)
            default: try run(args.first)
            }
        } catch {
            fputs("\(error)\n", stderr); Foundation.exit(1)
        }
    }

    private static func run(_ envArg: String?) throws {
        let envId = ProcessInfo.processInfo.environment["OAK_ENV"] ?? envArg ?? ""
        guard !envId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OakError.message("Missing environment id. Use `OaK run <env-id>` or set OAK_ENV.")
        }
        let policy = try parsePolicy(ProcessInfo.processInfo.environment["OAK_POLICY"] ?? Policy.oak.rawValue)
        let seed = ProcessInfo.processInfo.environment["OAK_SEED"].flatMap(Int.init)
        let warmup = max(ProcessInfo.processInfo.environment["OAK_WARMUP_STEPS"].flatMap(Int.init) ?? 0, 0)
        let device = try metalDevice()
        let probe = try EnvPoolEnv(id: envId, numEnvs: 1, seed: seed)
        let defaultNumEnvs = try directNumEnvs(obsDim: probe.obsDim, actDim: probe.actDim, actionDiscreteCount: probe.actionDiscreteCount, device: device)
        let numEnvs = max(ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init) ?? defaultNumEnvs, 1)
        let stopAfter = max(ProcessInfo.processInfo.environment["OAK_STOP_AFTER_STEPS"].flatMap(Int.init) ?? numEnvs, numEnvs)
        let baseline = max(ProcessInfo.processInfo.environment["OAK_BASELINE_RANDOM_STEPS"].flatMap(Int.init) ?? 0, 0)
        if policy == .oak, ProcessInfo.processInfo.environment["OAK_STOP_AFTER_STEPS"] == nil {
            fputs("OaK: OAK_STOP_AFTER_STEPS unset (short run). See AGENTS.md (Long runs).\n", stderr)
        }
        let dir = RunSnapshot.resolveMetricsDirectory(explicit: nil)
        let phases: [(Policy, Int)] = policy == .oak && baseline > 0 ? [(.random, baseline), (policy, stopAfter)] : [(policy, stopAfter)]
        var metricsURL: URL!
        for (p, steps) in phases {
            (_, metricsURL) = try runSingle(envId: envId, policy: p, seed: seed, warmup: warmup, stopAfter: steps, numEnvs: numEnvs, device: device, command: "run", metricsDir: dir)
        }
        fputs("\(metricsURL.path)\n", stderr)
    }

    private static func diagnose(_ args: [String]) throws {
        var envId: String?, policy = Policy.oak, stopAfter: Int?, forcedNumEnvs: Int?
        var index = 0
        while index < args.count {
            switch args[index] {
            case "--policy":
                index += 1
                guard index < args.count else { throw OakError.message("Missing value for --policy.") }
                policy = try parsePolicy(args[index])
            case "--steps":
                index += 1
                guard index < args.count, let value = Int(args[index]), value > 0 else { throw OakError.message("Invalid value for --steps.") }
                stopAfter = value
            case "--num-envs":
                index += 1
                guard index < args.count, let value = Int(args[index]), value > 0 else { throw OakError.message("Invalid value for --num-envs.") }
                forcedNumEnvs = value
            default:
                guard envId == nil else { throw OakError.message("Unexpected argument '\(args[index])'.") }
                envId = args[index]
            }
            index += 1
        }
        let resolvedEnv = ProcessInfo.processInfo.environment["OAK_ENV"] ?? envId ?? ""
        guard !resolvedEnv.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw OakError.message("Use `OaK diagnose <env-id>`.") }
        let seed = ProcessInfo.processInfo.environment["OAK_SEED"].flatMap(Int.init)
        let warmup = max(ProcessInfo.processInfo.environment["OAK_DIAG_WARMUP_STEPS"].flatMap(Int.init) ?? 0, 0)
        let device = try metalDevice()
        let probe = try EnvPoolEnv(id: resolvedEnv, numEnvs: 1, seed: seed)
        let defaultNumEnvs = try directNumEnvs(obsDim: probe.obsDim, actDim: probe.actDim, actionDiscreteCount: probe.actionDiscreteCount, device: device)
        let numEnvs = max(forcedNumEnvs ?? ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init) ?? min(defaultNumEnvs, ProcessInfo.processInfo.environment["OAK_DIAG_MAX_ENVS"].flatMap(Int.init) ?? 8), 1)
        let (_, metricsURL) = try runSingle(
            envId: resolvedEnv, policy: policy, seed: seed, warmup: warmup,
            stopAfter: max(stopAfter ?? ProcessInfo.processInfo.environment["OAK_DIAG_STEPS"].flatMap(Int.init) ?? (numEnvs * 8), numEnvs),
            numEnvs: numEnvs, device: device, command: "diagnose", includeDiagnostic: true)
        fputs("\(metricsURL.path)\n", stderr)
    }

    private static func snapshotJsonCount(in metricsDir: URL) -> Int {
        let fm = FileManager.default
        func isSubdirectory(_ url: URL) -> Bool {
            var isDir: ObjCBool = false
            return fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
        }
        guard let envs = try? fm.contentsOfDirectory(at: metricsDir, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else { return 0 }
        var count = 0
        for envDir in envs where isSubdirectory(envDir) {
            guard let seeds = try? fm.contentsOfDirectory(at: envDir, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else { continue }
            for seedDir in seeds where isSubdirectory(seedDir) {
                guard let files = try? fm.contentsOfDirectory(at: seedDir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { continue }
                count += files.filter { $0.pathExtension.lowercased() == "json" && !$0.lastPathComponent.contains(".heartbeat-") }.count
            }
        }
        return count
    }

    private static func resolveBenchmarkWorkingDirectory(root: URL, expectedTasks: Int) -> URL {
        if let raw = ProcessInfo.processInfo.environment["OAK_BENCH_DIR"]?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
        }
        if let raw = ProcessInfo.processInfo.environment["OAK_METRICS_STORAGE"]?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            let storage = URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
            if snapshotJsonCount(in: storage) < expectedTasks { return storage }
            return root.appendingPathComponent(".oak-benchmarks/\(oakBenchmarksRunId())", isDirectory: true)
        }
        let fresh = { root.appendingPathComponent(".oak-benchmarks/\(oakBenchmarksRunId())", isDirectory: true) }
        let box = root.appendingPathComponent(".oak-benchmarks", isDirectory: true)
        guard let runs = try? FileManager.default.contentsOfDirectory(at: box, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return fresh() }
        // Only resume folders whose name matches the auto-generated run id shape
        // (yyyyMMdd-HHmmss-<pid>). User-curated labels like "gpt 5.4" or "composerMuse"
        // are left alone — start a fresh run instead.
        let runIdPattern = try? NSRegularExpression(pattern: #"^\d{8}-\d{6}-\d+$"#)
        var bestDir: URL?, bestDate = Date.distantPast
        for run in runs {
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: run.path, isDirectory: &isDir), isDir.boolValue else { continue }
            let name = run.lastPathComponent
            let range = NSRange(name.startIndex..<name.endIndex, in: name)
            guard runIdPattern?.firstMatch(in: name, options: [], range: range) != nil else { continue }
            guard snapshotJsonCount(in: run) < expectedTasks else { continue }
            let d = (try? run.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            if d >= bestDate { bestDate = d; bestDir = run }
        }
        return bestDir ?? fresh()
    }

    private static func benchmark() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let seedStart = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_START"].flatMap(Int.init) ?? 0
        let seedCount = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 10
        let warmup = max(ProcessInfo.processInfo.environment["OAK_BENCH_WARMUP_STEPS"].flatMap(Int.init) ?? 1000, 0)
        let stopAfter = max(ProcessInfo.processInfo.environment["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 50000, 1)
        let policies = try (ProcessInfo.processInfo.environment["OAK_BENCH_POLICIES"] ?? "oak,random").split(separator: ",").map { try parsePolicy(String($0)) }
        let device = try metalDevice()
        let forcedNumEnvs = ProcessInfo.processInfo.environment["OAK_BENCH_NUM_ENVS"].flatMap(Int.init) ?? ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init)
        let envOrder = try benchmarkEnvOrder(seed: seedStart)
        let expectedTasks = envOrder.count * seedCount * policies.count
        let benchDir = resolveBenchmarkWorkingDirectory(root: root, expectedTasks: expectedTasks)
        try FileManager.default.createDirectory(at: benchDir, withIntermediateDirectories: true)
        for benchmarkEnv in envOrder {
            let envId = benchmarkEnv.envId
            let defaultNumEnvs = try directNumEnvs(obsDim: benchmarkEnv.schema.observationDim, actDim: benchmarkEnv.schema.actionDim, actionDiscreteCount: benchmarkEnv.schema.actionDiscreteCount, device: device)
            let numEnvs = max(forcedNumEnvs ?? defaultNumEnvs, 1)
            for seed in seedStart..<(seedStart + seedCount) {
                for policy in policies {
                    let stem = RunSnapshot.snapshotStem(envId: envId, policy: policy, seed: seed)
                    let snapURL = benchDir
                        .appendingPathComponent(envId, isDirectory: true)
                        .appendingPathComponent(RunSnapshot.seedFolderName(seed), isDirectory: true)
                        .appendingPathComponent("\(stem).json", isDirectory: false)
                    if FileManager.default.fileExists(atPath: snapURL.path) { continue }
                    let logURL = benchDir.appendingPathComponent("\(stem).log")
                    do {
                        _ = try runSingle(envId: envId, policy: policy, seed: seed, warmup: warmup, stopAfter: stopAfter, numEnvs: numEnvs, device: device, command: "benchmark", metricsDir: benchDir).0
                    } catch {
                        try "\(error)\n".data(using: .utf8)?.write(to: logURL)
                        throw error
                    }
                }
            }
        }
    }

    private static func parsePolicy(_ raw: String) throws -> Policy {
        guard let policy = Policy(rawValue: raw.lowercased()) else { throw OakError.message("Unknown policy '\(raw)'.") }
        return policy
    }

    private static func metalDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else { throw OakError.message("Metal device unavailable.") }
        return device
    }

    private static func benchmarkEnvOrder(seed: Int) throws -> [BenchmarkEnv] {
        var buckets = [Schema: [BenchmarkEnv]]()
        for envId in NativeBridge.shared.benchmarkEnvIds() {
            let probe = try EnvPoolEnv(id: envId, numEnvs: 1, seed: seed)
            let schema = Schema(observationDim: probe.obsDim, actionDim: probe.actDim, actionDiscreteCount: probe.actionDiscreteCount)
            buckets[schema, default: []].append(BenchmarkEnv(envId: envId, schema: schema))
        }
        let orderedSchemas = buckets.keys.sorted()
        var groups = orderedSchemas.map { buckets[$0]!.sorted { $0.envId < $1.envId } }
        var ordered = [BenchmarkEnv]()
        while true {
            var appended = false
            for index in groups.indices where !groups[index].isEmpty {
                ordered.append(groups[index].removeFirst()); appended = true
            }
            if !appended { return ordered }
        }
    }

    private static func randomAction(for env: EnvPoolEnv, numEnvs: Int, rng: GKMersenneTwisterRandomSource?, floats: inout [Float], indices: inout [Int32]) -> ActionBatch {
        for index in floats.indices { floats[index] = 0 }
        if let count = env.actionDiscreteCount {
            for row in 0..<numEnvs {
                let choice = rng?.nextInt(upperBound: count) ?? Int.random(in: 0..<count)
                indices[row] = Int32(choice)
                floats[row * env.actDim + choice] = 1
            }
            return ActionBatch(floats: floats, indices: indices)
        }
        for index in floats.indices {
            let dim = index % env.actDim
            floats[index] = env.actionLow[dim] + (env.actionHigh[dim] - env.actionLow[dim]) * (rng?.nextUniform() ?? Float.random(in: 0..<1))
        }
        return ActionBatch(floats: floats, indices: nil)
    }

    private static func runSingle(envId: String, policy: Policy, seed: Int?, warmup: Int, stopAfter: Int, numEnvs: Int, device: MTLDevice, command: String, includeDiagnostic: Bool = false, metricsDir: URL? = nil) throws -> (RunMetricsFile, URL) {
        let metricsDirResolved = RunSnapshot.resolveMetricsDirectory(explicit: metricsDir)
        try FileManager.default.createDirectory(at: metricsDirResolved, withIntermediateDirectories: true)
        fputs("OaK metrics \(metricsDirResolved.path)\n", stderr); fflush(stderr)
        let env = try EnvPoolEnv(id: envId, numEnvs: numEnvs, seed: seed)
        let encoder = seed.map { GKMersenneTwisterRandomSource(seed: UInt64(bitPattern: Int64($0))) }
        let engine: OakMetalEngine? = policy == .oak ? try OakMetalEngine(
            obsDim: env.obsDim, baseObsDim: env.obsDim, actDim: env.actDim,
            batchSize: numEnvs, rank: env.obsDim,
            actionIsDiscrete: env.actionDiscreteCount != nil,
            actionLow: env.actionLow, actionHigh: env.actionHigh,
            device: device, seed: seed, diagnosticGpuPhaseTiming: includeDiagnostic) : nil
        try env.reset()
        var prevReward = Array(repeating: Float(0), count: numEnvs)
        var prevDone = Array(repeating: UInt8(0), count: numEnvs)
        var prevActionScratch = [Float](repeating: 0, count: env.actDim * numEnvs)
        var randomActionScratch = Array(repeating: Float(0), count: env.actDim * numEnvs)
        var randomIndexScratch = Array(repeating: Int32(0), count: numEnvs)
        var totalSteps = 0, measuredSteps = 0
        var measuredSeconds = 0.0
        var measurementStart: UInt64?
        var rewardSum = 0.0, doneCount = 0
        var agentMsSum = 0.0, envMsSum = 0.0
        var health = HealthAccumulator()
        let actionCol = MeasuredActionCollector(discreteCount: env.actionDiscreteCount, actDim: env.actDim)
        let defaultHb = command == "diagnose" ? 0 : max((stopAfter / 10 / numEnvs) * numEnvs, numEnvs)
        let hb = max(ProcessInfo.processInfo.environment["OAK_HEARTBEAT_STEPS"].flatMap(Int.init) ?? defaultHb, 0)
        var pulse = 0
        func snap(_ seq: Int?) -> RunMetricsFile {
            let u = max(measuredSteps / numEnvs, 1), a = agentMsSum / 1000, e = envMsSum / 1000, h = max(measuredSeconds - a - e, 0)
            let merged = health.diagnosticSnapshot().map { b in
                let (mm, mi) = engine?.metalPhaseGpuDiagnostics(measuredBatchSteps: u) ?? (nil, nil)
                return RunDiagnostic(nonFiniteByPhase: b.nonFiniteByPhase, maxAbsLatentPrediction: b.maxAbsLatentPrediction, maxAbsRewardPrediction: b.maxAbsRewardPrediction, maxAbsPolicyOutput: b.maxAbsPolicyOutput, agentShare: measuredSeconds > 0 ? a / measuredSeconds : nil, envShare: measuredSeconds > 0 ? e / measuredSeconds : nil, hostShare: measuredSeconds > 0 ? h / measuredSeconds : nil, metalPhaseGpuMsMean: mm, metalPhaseGpuTimingInvalid: mi)
            }
            let diagOut: RunDiagnostic? = ((seq != nil && policy == .oak) || (seq == nil && includeDiagnostic)) ? merged : nil
            let (dA, cA) = actionCol.partialMetrics(actDim: env.actDim)
            let sps = measuredSeconds > 0 ? Double(measuredSteps) / measuredSeconds : nil
            let reason = seq == nil ? "measured_steps" : "heartbeat"
            return RunMetricsFile(schemaVersion: 2, command: command, envId: envId, policy: policy, seed: seed, numEnvs: numEnvs, warmupSteps: warmup, stopAfterMeasuredSteps: stopAfter, observationDim: env.obsDim, actionDim: env.actDim, actionDiscreteCount: env.actionDiscreteCount, measuredStepCount: measuredSteps, measuredBatchCount: u, measuredSeconds: measuredSeconds, stepsPerSecond: sps, heartbeatSequence: seq, rewardSum: rewardSum, doneCount: doneCount, stopReason: reason, agentStepMsMean: agentMsSum / Double(u), envStepMsMean: envMsSum / Double(u), discreteAction: dA, continuousAction: cA, health: health.snapshot(), diagnostic: diagOut)
        }
        while measuredSteps < stopAfter {
            let nextTotal = totalSteps + numEnvs
            let measured = nextTotal > warmup && totalSteps > 0
            if measured && measurementStart == nil { measurementStart = DispatchTime.now().uptimeNanoseconds }
            let actionBatch: ActionBatch, agentMs: Double, stats: KernelStats?
            if let engine {
                let output = try env.observationScratch.withUnsafeBufferPointer { buf in
                    try engine.step(obs: buf, prevAction: prevActionScratch, prevReward: prevReward, prevDone: prevDone, accumulatePhaseGpu: measured)
                }
                actionBatch = output.0; agentMs = output.1; stats = output.2
            } else {
                let t0 = DispatchTime.now().uptimeNanoseconds
                actionBatch = randomAction(for: env, numEnvs: numEnvs, rng: encoder, floats: &randomActionScratch, indices: &randomIndexScratch)
                agentMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000
                stats = nil
            }
            let envStart = DispatchTime.now().uptimeNanoseconds
            try env.step(actionBatch)
            let envMs = Double(DispatchTime.now().uptimeNanoseconds - envStart) / 1_000_000
            totalSteps = nextTotal
            if measured {
                measuredSteps += numEnvs
                if let measurementStart { measuredSeconds = Double(DispatchTime.now().uptimeNanoseconds - measurementStart) / 1_000_000_000 }
                if hb > 0, (measuredSteps - numEnvs) / hb < measuredSteps / hb {
                    pulse += 1
                    _ = try RunSnapshot.persist(snap(pulse), metricsDir: metricsDirResolved)
                }
                doneCount += env.doneScratch.reduce(into: 0) { $0 += Int($1) }
                for value in env.rewardScratch {
                    guard value.isFinite else { throw OakError.message("Environment produced a non-finite reward.") }
                    rewardSum += Double(value)
                }
                agentMsSum += agentMs
                envMsSum += envMs
                if let stats { health.record(stats) }
                actionCol.addMeasured(actionBatch, numEnvs: numEnvs, actDim: env.actDim)
            }
            memcpy(&prevReward, env.rewardScratch, numEnvs * MemoryLayout<Float>.stride)
            memcpy(&prevDone, env.doneScratch, numEnvs * MemoryLayout<UInt8>.stride)
            memcpy(&prevActionScratch, actionBatch.floats, actionBatch.floats.count * MemoryLayout<Float>.stride)
        }
        let file = snap(nil)
        let metricsURL = try RunSnapshot.persist(file, metricsDir: metricsDirResolved)
        return (file, metricsURL)
    }

    private static let runIdFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .iso8601)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()

    fileprivate static func oakBenchmarksRunId() -> String {
        runIdFormatter.string(from: Date()) + "-\(ProcessInfo.processInfo.processIdentifier)"
    }
}
