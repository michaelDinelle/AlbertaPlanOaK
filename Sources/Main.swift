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
    /// Online average-reward estimate ρ (`KernelScalars.rewardRate`) at end of the measured window—not a Monte Carlo mean of batch rewards (see metrics export `rewardSum`).
    let rewardRate: Double?
    let latentPredictionMSE: Double?
    let rewardPredictionMSE: Double?
    /// Mean squared sanitized latent components (per bookkeeping pass), averaged over measured steps; not a variance or covariance trace.
    let latentStateMeanSquare: Double?
    let activeRank: Double?
    let actionDeltaL2Mean: Double?
    let planningActionChangeL2Mean: Double?
    /// Mean squared error between realized latent `state` and option forward model `rolloutState` (STOMP option-model quality).
    let optionModelPredictionMSE: Double?
    /// Telemetry: planning imagination uses option-conditioned dynamics (1.0 when Oak planning uses option slices).
    let planningUsedOptionDynamicsMean: Double?
    /// Count of option dynamics slices zero-reinited in bookkeeping (continual replacement).
    let oakOptionReinitCount: Double?
    let nonFiniteRawTotal: Int
    let nonFiniteSanitizedTotal: Int

    enum CodingKeys: String, CodingKey {
        case rewardRate
        case latentPredictionMSE
        case rewardPredictionMSE
        case latentStateMeanSquare = "latentVarianceTrace"
        case activeRank
        case actionDeltaL2Mean
        case planningActionChangeL2Mean
        case optionModelPredictionMSE
        case planningUsedOptionDynamicsMean
        case oakOptionReinitCount
        case nonFiniteRawTotal
        case nonFiniteSanitizedTotal
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
    /// Mean GPU time (ms) per measured batch step for each `oak_step` phase; from `MTLCommandBuffer` gpu start/end (one buffer per phase). Omitted when policy is not oak or timing unavailable.
    let metalPhaseGpuMsMean: [String: Double]?
    /// Phases where `gpuStartTime`/`gpuEndTime` were invalid for at least one measured step.
    let metalPhaseGpuTimingInvalid: [String]?
}

private struct DiscreteActionMetrics: Encodable { let histogram: [Int]; let uniqueCount: Int }
private struct ContinuousActionMetrics: Encodable { let meanPerDim: [Double]; let stdPerDim: [Double]; let meanL2: Double; let stdL2: Double }

/// Pre-allocated accumulators for measured-window action telemetry (see **`AGENTS.md`**).
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
                s[d] += v
                q[d] += v * v
                l2s += v * v
            }
            let l2 = sqrt(l2s)
            l2sum += l2
            l2sumsq += l2 * l2
            n += 1
        }
        sum = s
        sumsq = q
    }

    func finish(actDim: Int) -> (DiscreteActionMetrics?, ContinuousActionMetrics?) {
        let d = discrete.map { DiscreteActionMetrics(histogram: $0, uniqueCount: $0.filter { $0 > 0 }.count) }
        guard let s = sum, let q = sumsq, n > 0 else { return (d, nil) }
        let nn = Double(n)
        let mean = (0..<actDim).map { s[$0] / nn }
        let std = (0..<actDim).map { let m = mean[$0]; return sqrt(max(0, q[$0] / nn - m * m)) }
        let ml = l2sum / nn
        return (d, ContinuousActionMetrics(meanPerDim: mean, stdPerDim: std, meanL2: ml, stdL2: sqrt(max(0, l2sumsq / nn - ml * ml))))
    }
}

/// One pretty-printed JSON file per run: flat fields (no duplicate config), optional keys omitted, stable key order via `JSONEncoder` **`sortedKeys`** (Swift's encoder does not preserve manual `encode` order for large objects).
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
    /// Batch iterations in the measured window (`measuredStepCount / numEnvs` when divisible).
    let measuredBatchCount: Int
    let measuredSeconds: Double
    let stepsPerSecond: Double?
    let rewardSum: Double
    let doneCount: Int
    let stopReason: String
    let agentStepMsMean: Double
    let envStepMsMean: Double
    /// Present when **`actionDiscreteCount`** is set; indices match the batch passed to **`env.step`**.
    let discreteAction: DiscreteActionMetrics?
    /// Present for **continuous** control (`actionDiscreteCount` nil); summarizes **`actionBatch.floats`** over the measured window.
    let continuousAction: ContinuousActionMetrics?
    /// Present when the policy recorded kernel stats over the measured window (typically `oak`; omitted for `random`).
    let health: RunHealth?
    let diagnostic: RunDiagnostic?
}

struct ActionBatch { let floats: [Float]; let indices: [Int32]? }

private struct Schema: Hashable, Comparable {
    let observationDim: Int
    let actionDim: Int
    let actionDiscreteCount: Int?

    static func < (lhs: Schema, rhs: Schema) -> Bool {
        (lhs.observationDim, lhs.actionDim, lhs.actionDiscreteCount ?? -1) < (rhs.observationDim, rhs.actionDim, rhs.actionDiscreteCount ?? -1)
    }
}

private struct BenchmarkEnv {
    let envId: String
    let schema: Schema
}

private func oakRoots() -> [URL] {
    let fileManager = FileManager.default
    let executable = URL(fileURLWithPath: CommandLine.arguments[0], relativeTo: URL(fileURLWithPath: fileManager.currentDirectoryPath)).standardizedFileURL
    return [
        fileManager.currentDirectoryPath,
        executable.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().path
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
    var power = 1
    while power <= value / 2 { power *= 2 }
    return power
}

private func directNumEnvs(obsDim: Int, actDim: Int, actionDiscreteCount: Int?, device: MTLDevice) throws -> Int {
    let width = try OakMetalEngine.executionWidth(device: device)
    let rank = obsDim
    let rewardSpan = actionDiscreteCount == nil ? rank + actDim : actDim * (rank + 1)
    let packed = OakMetalLayout.packedStateWeightsSize(rank: rank, actDim: actDim)
    let oc = OakMetalLayout.derivedOptionCount(rank: rank, actDim: actDim)
    let ocCrit = OakMetalLayout.optionCriticBufferSize(rank: rank, actDim: actDim)
    let phase1Span = obsDim + rewardSpan + packed + ocCrit + rank + actDim * rank + (oc * rank) + (oc * rank * actDim)
    let computeSpan = max(phase1Span, OakMetalLayout.learnPhaseThreadCount(rank: rank, actDim: actDim), width * max(rank, actDim * actDim, width))
    let nativeBytesPerEnv = actionDiscreteCount == nil ? (1 << 20) : (1 << 14)
    let memoryBudget = max(Int(ProcessInfo.processInfo.physicalMemory / 16), width * nativeBytesPerEnv)
    let memorySpan = max(memoryBudget / nativeBytesPerEnv, width)
    let raw = max(roundDownPowerOfTwo(min(computeSpan, memorySpan)), width)
    // Hard ceiling: huge batch × rank grid (e.g. MiniGrid × 64k envs) destroys throughput; override with OAK_MAX_PARALLEL_ENVS.
    let ceiling = max(ProcessInfo.processInfo.environment["OAK_MAX_PARALLEL_ENVS"].flatMap(Int.init) ?? 4096, width)
    return min(raw, ceiling)
}

private struct HealthAccumulator {
    private struct Mean {
        var sum = 0.0
        var count = 0
        mutating func add(_ value: Float, raw: inout Int, sanitized: inout Int) {
            let double = Double(value)
            guard double.isFinite else {
                raw += 1
                sanitized += 1
                return
            }
            sum += double
            count += 1
        }
        var snapshot: Double? { count > 0 ? sum / Double(count) : nil }
    }

    var steps = 0
    /// Last finite online ρ snapshot per measured step; `RunHealth.rewardRate` uses the **final** value (not mean of snapshots—see Alberta average-reward TD).
    private var lastOnlineRewardRate: Double?
    private var latentPrediction = Mean(), rewardPrediction = Mean(), latentStateMS = Mean(), activeRank = Mean(), actionDelta = Mean(), planningDelta = Mean()
    private var optionModelPred = Mean(), planningOptDyn = Mean(), oakReinit = Mean()
    var rawNonFinite = 0, sanitizedNonFinite = 0
    private var encodeNonFinite = 0, learnNonFinite = 0, planNonFinite = 0
    private var maxAbsLatentPrediction = 0.0, maxAbsRewardPrediction = 0.0, maxAbsPolicyOutput = 0.0

    mutating func record(_ stats: KernelStats) {
        steps += 1
        let rr = Double(stats.rewardRate)
        if rr.isFinite { lastOnlineRewardRate = rr }
        latentPrediction.add(stats.latentPredictionMSE, raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
        rewardPrediction.add(stats.rewardPredictionMSE, raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
        latentStateMS.add(stats.latentStateMeanSquare, raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
        activeRank.add(stats.activeRank, raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
        actionDelta.add(stats.actionDeltaL2Mean, raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
        planningDelta.add(stats.planningActionChangeL2Mean, raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
        optionModelPred.add(stats.optionModelPredictionMSE, raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
        planningOptDyn.add(stats.planningUsedOptionDynamicsMean, raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
        oakReinit.add(stats.oakOptionReinitCount, raw: &rawNonFinite, sanitized: &sanitizedNonFinite)
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
        return RunHealth(
            rewardRate: lastOnlineRewardRate,
            latentPredictionMSE: latentPrediction.snapshot,
            rewardPredictionMSE: rewardPrediction.snapshot,
            latentStateMeanSquare: latentStateMS.snapshot,
            activeRank: activeRank.snapshot,
            actionDeltaL2Mean: actionDelta.snapshot,
            planningActionChangeL2Mean: planningDelta.snapshot,
            optionModelPredictionMSE: optionModelPred.snapshot,
            planningUsedOptionDynamicsMean: planningOptDyn.snapshot,
            oakOptionReinitCount: oakReinit.snapshot,
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
            agentShare: nil,
            envShare: nil,
            hostShare: nil,
            metalPhaseGpuMsMean: nil,
            metalPhaseGpuTimingInvalid: nil
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

    static let shared: NativeBridge = {
        do { return try NativeBridge() }
        catch { fatalError("\(error)") }
    }()

    private let dylib: UnsafeMutableRawPointer
    private let assetBasePath: String
    private let openFn: OpenFn
    private let closeFn: CloseFn
    private let obsDimFn: IntFn
    private let actDimFn: IntFn
    private let discreteCountFn: IntFn
    private let boundsFn: BoundsFn
    private let resetFn: ResetFn
    private let stepContFn: StepContFn
    private let stepDiscFn: StepDiscFn
    private let takeWarningFn: TakeWarningFn
    private let benchmarkCountFn: BenchmarkCountFn
    private let benchmarkIdFn: BenchmarkIdFn

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
        let count = Int(discreteCountFn(handle))
        return count > 0 ? count : nil
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
        (0..<Int(benchmarkCountFn())).compactMap { index in
            benchmarkIdFn(Int32(index)).map { String(cString: $0) }
        }
    }
}

final class EnvPoolEnv {
    private let bridge = NativeBridge.shared
    private let handle: UnsafeMutableRawPointer
    /// Filled in-place by `reset` / `step`; exposed to avoid CoW copies of full batch observations on every tick.
    private(set) var observationScratch: [Float]
    private(set) var rewardScratch: [Float]
    private(set) var doneScratch: [UInt8]
    let numEnvs: Int
    let obsDim: Int
    let actDim: Int
    let actionDiscreteCount: Int?
    let actionLow: [Float]
    let actionHigh: [Float]

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

private struct KernelConfig {
    var phase: UInt32 = 0
    var obsDim: UInt32 = 0
    var rank: UInt32 = 0
    var actDim: UInt32 = 0
    var batchSize: UInt32 = 0
    var actionIsDiscrete: UInt32 = 0
    /// Interface-derived width: min(rank, actDim); must match Metal `KernelConfig.optionCount`.
    var optionCount: UInt32 = 0
    var primitivePackedSize: UInt32 = 0
    var optionForwardStride: UInt32 = 0
    var planK: UInt32 = 0
    /// Env observation width before optional aux concat (`OAK_AUX_DIM`); encoder NLMS only for `feat < baseObsDim`.
    var baseObsDim: UInt32 = 0
}
private struct KernelScalars { var stepCount: UInt32 = 0, rewardCount: UInt32 = 0, activeCount: UInt32 = 0, rewardRate: Float = 0 }
/// Phase ids must match `kPhase*` in `OakKernel.metal`.
private enum OakKernelPhase: UInt32 {
    case encode = 0
    case prediction = 1
    case learn = 2
    case metaPolicy = 3
    case planning = 4
    case forwardModel = 6
    /// Matches `kPhaseBookkeeping` in Metal (prune + ρ + stats, single dispatch).
    case bookkeeping = 7
}

/// Buffer sizes and thread counts derived from `(rank, actDim)`; keeps Swift aligned with Metal packing.
private enum OakMetalLayout {
    /// Three learn bands in Metal (value head, GVF block, forward block); structural, not a tuned hyperparameter.
    static let learnParallelRowBands = 3
    /// Must match `constant KernelConfig &config [[buffer(N)]]` in `OakKernel.metal` (after STOMP/Oak buffers).
    static let kernelConfigBufferIndex = 30

    /// Option slots = min(signal ranks) from the interface; no fixed width (see project invariants).
    static func derivedOptionCount(rank: Int, actDim: Int) -> Int {
        max(1, min(rank, actDim))
    }

    /// GVF bank only (`rank × rank`); forward dynamics live in contiguous per-option blocks immediately after.
    static func gvfBankFloatCount(rank: Int) -> Int {
        rank * rank
    }

    /// One forward block: `rank` rows × `(rank + actDim)` weights per row (same layout as former “primitive” segment).
    static func optionForwardStride(rank: Int, actDim: Int) -> Int {
        rank * (rank + actDim)
    }

    /// Packed `stateWeights`: `[GVF bank][option 0 dynamics][option 1]…[option oc−1]]` — **no duplicate** primitive forward;
    /// option **0** is the shared one-step dynamics slice used when `pick_option_slice` returns 0 (rollout/planning/learning agree).
    static func packedStateWeightsSize(rank: Int, actDim: Int) -> Int {
        let oc = derivedOptionCount(rank: rank, actDim: actDim)
        return gvfBankFloatCount(rank: rank) + oc * optionForwardStride(rank: rank, actDim: actDim)
    }

    /// Float offset to the first option dynamics block (= GVF bank size). Passed as `KernelConfig.primitivePackedSize`.
    static func optionDynamicsRegionBase(rank: Int) -> Int {
        gvfBankFloatCount(rank: rank)
    }

    /// Structural K-step planning depth from rank only: `min(4, max(rank, 1))`.
    static func structuralPlanK(rank: Int) -> Int {
        min(4, max(rank, 1))
    }

    static func optionPolicyParameterCount(rank: Int, actDim: Int) -> Int {
        derivedOptionCount(rank: rank, actDim: actDim) * rank * actDim
    }

    static func optionValueParameterCount(rank: Int, actDim: Int) -> Int {
        derivedOptionCount(rank: rank, actDim: actDim) * rank
    }

    /// Secondary critics + online initiation-track tail (`optionCount` floats at end of the same buffer).
    static func optionCriticBufferSize(rank: Int, actDim: Int) -> Int {
        let oc = derivedOptionCount(rank: rank, actDim: actDim)
        return oc * rank + oc
    }

    /// `latentPower` tail packs: keyboard (`batch*oc` floats) + option feedback EMA (`3*oc`: model error, forward-use, planning-share) after `2*batch` NLMS slots (Metal `kbd0` / `ofb0`).
    static func latentPowerPackedSize(rank: Int, actDim: Int, batchSize: Int) -> Int {
        let oc = derivedOptionCount(rank: rank, actDim: actDim)
        return 2 * batchSize + batchSize * oc + 3 * oc
    }

    /// `stompState` uints: `4*batch` STOMP row state + `batch` batch-dynamics tags (packed, buffer bound).
    static func stompStatePackedSize(batchSize: Int) -> Int {
        max(5 * batchSize, 5)
    }

    /// Matches Metal `kPhaseLearn`: value + GVF + fused option forward NLMS + policy + option critics (no separate primitive band).
    static func learnPhaseThreadCount(rank: Int, actDim: Int) -> Int {
        let oc = derivedOptionCount(rank: rank, actDim: actDim)
        return max(4 * rank + oc * rank, 1)
    }
}

fileprivate struct KernelStats {
    var rewardRate: Float = 0
    var latentPredictionMSE: Float = 0
    var rewardPredictionMSE: Float = 0
    var latentStateMeanSquare: Float = 0
    var activeRank: Float = 0
    var actionDeltaL2Mean: Float = 0
    var planningActionChangeL2Mean: Float = 0
    var rawNonFiniteCount: Float = 0
    var sanitizedNonFiniteCount: Float = 0
    var encodeNonFiniteCount: Float = 0
    var learnNonFiniteCount: Float = 0
    var planNonFiniteCount: Float = 0
    var maxAbsLatentPrediction: Float = 0
    var maxAbsRewardPrediction: Float = 0
    var maxAbsPolicyOutput: Float = 0
    var optionModelPredictionMSE: Float = 0
    var planningUsedOptionDynamicsMean: Float = 0
    var oakOptionReinitCount: Float = 0
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

    func copy(from values: [Element]) { values.withUnsafeBufferPointer { guard let base = $0.baseAddress else { return }; pointer.update(from: base, count: count) } }
    func fill(with value: Element) { pointer.update(repeating: value, count: count) }
}

private typealias SharedFloats = SharedArray<Float>
private typealias SharedUInts = SharedArray<UInt32>

/// Softmax sample + one-hot host view for discrete envs. Continuous runs omit this object entirely.
private final class DiscreteSoftmaxBehavior {
    private let rng: GKMersenneTwisterRandomSource
    private var expScratch: [Float]
    private let actDim: Int

    init(actDim: Int, engineSeed: Int?) {
        self.actDim = actDim
        expScratch = [Float](repeating: 0, count: actDim)
        let seed64: UInt64
        if let s = engineSeed {
            seed64 = UInt64(bitPattern: Int64(s))
        } else {
            var g = SystemRandomNumberGenerator()
            seed64 = UInt64.random(in: UInt64.min ... UInt64.max, using: &g)
        }
        rng = GKMersenneTwisterRandomSource(seed: seed64)
    }

    func apply(hostAction: inout [Float], hostDiscrete: inout [Int32], batchSize: Int) {
        for row in 0..<batchSize {
            let o = row * actDim
            var mx = hostAction[o]
            for i in 1..<actDim { mx = max(mx, hostAction[o + i]) }
            var s: Float = 0
            for i in 0..<actDim {
                let e = exp(hostAction[o + i] - mx)
                expScratch[i] = e
                s += e
            }
            let invS = 1 / s
            let u = rng.nextUniform()
            var c: Float = 0
            var pick = actDim - 1
            for i in 0..<actDim {
                c += expScratch[i] * invS
                if u < c { pick = i; break }
            }
            hostDiscrete[row] = Int32(pick)
            for i in 0..<actDim { hostAction[o + i] = i == pick ? 1 : 0 }
        }
    }
}

final class OakMetalEngine {
    /// Total observation width (env flat dim + optional aux); equals `rank` (interface-derived capacity).
    private let obsDim: Int
    private let baseObsDim: Int
    private let actDim: Int
    private let batchSize: Int
    private let rank: Int
    private let actionIsDiscrete: Bool
    private let queue: MTLCommandQueue
    /// Unretained command-buffer refs: kernel buffers stay alive on `self` until `waitUntilCompleted` returns.
    private let commandBufferDescriptor: MTLCommandBufferDescriptor
    private let pipeline: MTLComputePipelineState
    private let actionBoundsBuffer: MTLBuffer
    private let scalarsBuffer: MTLBuffer
    private let statsBuffer: MTLBuffer
    private let scalarsPointer: UnsafeMutablePointer<KernelScalars>
    private let statsPointer: UnsafeMutablePointer<KernelStats>
    /// Fixed zero offsets for `setBuffers` (one allocation in `init`, none per dispatch).
    private let bufferZeroOffsets: [Int]

    private let observationMean, encoder, policy, valueWeights, rewardWeights, stateWeights, optionPolicyWeights, optionValues, featureSecondMoment, featureResidual: SharedFloats
    private let observation, reward, state, previousState, previousAction, latentError, rewardPrediction, td, actionGrad, actionBuffer, rolloutState, latentPower: SharedFloats
    private let optionCriticWeights: SharedFloats
    private let done, activeMask, counters: SharedUInts
    private let stompState: SharedUInts
    /// When `OAK_METAL_SIGNPOSTS=1`, inserts `insertDebugSignpost` between phase dispatches for Instruments / GPU capture (no effect on math).
    private let metalDebugSignposts: Bool
    /// When true (e.g. `diagnose`), measured steps use one command buffer per phase so `MTLCommandBuffer.gpuStartTime`/`gpuEndTime` yield per-phase GPU time in JSON. Warmup uses the fast single-buffer path.
    private let diagnosticGpuPhaseTiming: Bool
    /// Sum of GPU intervals (seconds) per phase index; only updated when `diagnosticGpuPhaseTiming` and the step is measured.
    private var phaseGpuTimeSumSeconds: [Double]
    /// Count of measured steps where GPU interval was missing/invalid, per phase.
    private var phaseGpuInvalidStepCount: [Int]
    private var hostAction: [Float]
    private var hostDiscrete: [Int32]
    private let discreteBehavior: DiscreteSoftmaxBehavior?

    private static func makePipeline(device: MTLDevice) throws -> MTLComputePipelineState {
        let library = try device.makeLibrary(URL: try kernelURL())
        guard let function = library.makeFunction(name: "oak_step") else { throw OakError.message("Missing oak_step in OakKernel.metallib.") }
        return try device.makeComputePipelineState(function: function)
    }

    static func executionWidth(device: MTLDevice) throws -> Int {
        let pipeline = try makePipeline(device: device)
        return max(pipeline.threadExecutionWidth, 1)
    }

    private static func kernelURL() throws -> URL {
        // `OakKernel.metal` is excluded from SwiftPM; rebuild `.build/OakKernel.metallib` after editing it (see `setup.sh`).
        try requiredBuildPath(".build/OakKernel.metallib", description: "non-empty .build/OakKernel.metallib", nonEmpty: true)
    }

    /// - Parameters:
    ///   - obsDim: Total flat observation width (must equal `rank`): env observation + optional aux (`OAK_AUX_DIM`).
    ///   - baseObsDim: Raw env observation width (encoder updates only for `feat < baseObsDim`).
    init(obsDim: Int, baseObsDim: Int, actDim: Int, batchSize: Int, rank: Int, actionIsDiscrete: Bool, actionLow: [Float], actionHigh: [Float], device: MTLDevice, seed: Int?, diagnosticGpuPhaseTiming: Bool = false) throws {
        precondition(obsDim == rank, "obsDim must equal rank (interface-derived latent width).")
        precondition(baseObsDim <= obsDim && baseObsDim >= 1, "baseObsDim must be positive and ≤ obsDim.")
        precondition(rank <= 384, "rank must be ≤ 384 (Metal planning scratch).")
        precondition(actDim <= 384, "actDim must be ≤ 384 (Metal learn-phase forward accumulators).")
        self.obsDim = obsDim
        self.baseObsDim = baseObsDim
        self.actDim = actDim
        self.batchSize = batchSize
        self.rank = rank
        self.actionIsDiscrete = actionIsDiscrete
        guard let queue = device.makeCommandQueue() else { throw OakError.message("Failed to create Metal command queue.") }
        self.queue = queue
        let cbDesc = MTLCommandBufferDescriptor()
        cbDesc.retainedReferences = false
        commandBufferDescriptor = cbDesc
        pipeline = try Self.makePipeline(device: device)
        // Discrete: env does not expose box bounds; use wide finite limits so `clamp` in Metal is a single path
        // (logits are not squeezed to 0,0 — the old placeholder broke planning + bookkeeping).
        let wideLo = -Float.greatestFiniteMagnitude
        let wideHi = Float.greatestFiniteMagnitude
        let lows = actionIsDiscrete ? Array(repeating: wideLo, count: actDim) : actionLow
        let highs = actionIsDiscrete ? Array(repeating: wideHi, count: actDim) : actionHigh
        actionBoundsBuffer = device.makeBuffer(bytes: lows + highs, length: 2 * actDim * MemoryLayout<Float>.stride, options: [.storageModeShared])!
        scalarsBuffer = device.makeBuffer(length: MemoryLayout<KernelScalars>.stride, options: [.storageModeShared])!
        statsBuffer = device.makeBuffer(length: MemoryLayout<KernelStats>.stride, options: [.storageModeShared])!
        scalarsPointer = scalarsBuffer.contents().bindMemory(to: KernelScalars.self, capacity: 1)
        statsPointer = statsBuffer.contents().bindMemory(to: KernelStats.self, capacity: 1)
        scalarsPointer.initialize(to: KernelScalars())
        statsPointer.initialize(to: KernelStats())
        bufferZeroOffsets = [Int](repeating: 0, count: 30)

        func floats(_ count: Int) -> SharedFloats { SharedFloats(device: device, count: count, repeating: 0) }
        func uints(_ count: Int) -> SharedUInts { SharedUInts(device: device, count: count, repeating: 0) }
        observationMean = floats(obsDim)
        encoder = floats(obsDim)
        policy = floats(actDim * rank)
        valueWeights = floats(rank)
        rewardWeights = floats(actionIsDiscrete ? actDim * (rank + 1) : rank + actDim)
        stateWeights = floats(OakMetalLayout.packedStateWeightsSize(rank: rank, actDim: actDim))
        optionPolicyWeights = floats(OakMetalLayout.optionPolicyParameterCount(rank: rank, actDim: actDim))
        optionValues = floats(OakMetalLayout.optionValueParameterCount(rank: rank, actDim: actDim))
        optionCriticWeights = floats(OakMetalLayout.optionCriticBufferSize(rank: rank, actDim: actDim))
        featureSecondMoment = floats(rank)
        featureResidual = floats(rank)
        observation = floats(batchSize * obsDim)
        reward = floats(batchSize)
        state = floats(batchSize * rank)
        previousState = floats(batchSize * rank)
        previousAction = floats(batchSize * actDim)
        latentError = floats(batchSize * rank)
        rewardPrediction = floats(batchSize)
        td = floats(batchSize)
        actionGrad = floats(batchSize * actDim)
        actionBuffer = floats(batchSize * actDim)
        rolloutState = floats(batchSize * rank)
        latentPower = floats(OakMetalLayout.latentPowerPackedSize(rank: rank, actDim: actDim, batchSize: batchSize))
        done = uints(batchSize)
        activeMask = uints(2 * rank)
        counters = uints(5)
        let ocInit = OakMetalLayout.derivedOptionCount(rank: rank, actDim: actDim)
        stompState = uints(OakMetalLayout.stompStatePackedSize(batchSize: batchSize))
        metalDebugSignposts = ProcessInfo.processInfo.environment["OAK_METAL_SIGNPOSTS"] == "1"
        self.diagnosticGpuPhaseTiming = diagnosticGpuPhaseTiming
        phaseGpuTimeSumSeconds = Array(repeating: 0, count: 7)
        phaseGpuInvalidStepCount = Array(repeating: 0, count: 7)
        hostAction = Array(repeating: 0, count: batchSize * actDim)
        hostDiscrete = Array(repeating: 0, count: batchSize)
        discreteBehavior = actionIsDiscrete ? DiscreteSoftmaxBehavior(actDim: actDim, engineSeed: seed) : nil
        scalarsPointer.pointee.activeCount = UInt32(rank)
        for feature in 0..<rank {
            activeMask.pointer[feature] = 1
            activeMask.pointer[rank + feature] = UInt32(feature)
        }

        let kbd0 = 2 * batchSize
        let ofb0 = kbd0 + batchSize * ocInit
        let invKey = 1.0 / Float(max(ocInit, 1))
        for row in 0..<batchSize {
            for k in 0..<ocInit {
                latentPower.pointer[kbd0 + row * ocInit + k] = invKey
            }
        }
        for i in 0..<(3 * ocInit) {
            latentPower.pointer[ofb0 + i] = 0
        }

        let rng = seed.map { GKMersenneTwisterRandomSource(seed: UInt64(bitPattern: Int64($0))) }
        let encoderScale: Float = 1.0
        let transitionScale = Float(1.0 / sqrt(Double(max(1 + actDim, 1))))
        for (storage, scale) in [(encoder, encoderScale), (policy, transitionScale), (valueWeights, transitionScale), (rewardWeights, transitionScale), (stateWeights, transitionScale), (optionPolicyWeights, transitionScale), (optionValues, transitionScale), (optionCriticWeights, transitionScale)] {
            for index in 0..<storage.count { storage.pointer[index] = ((rng?.nextUniform() ?? Float.random(in: 0..<1)) * 2 - 1) * scale }
        }
        // Subtask / option diagonal emphasis (STOMP): align option `k` with latent dimension `k` at init (signal-only, no env IDs).
        for opt in 0..<ocInit {
            optionValues.pointer[opt * rank + opt] += 1.0
        }
    }

    /// Observation must be exactly `batchSize * obsDim` contiguous floats (typically `env.observationScratch` or padded aux view).
    fileprivate func step(obs: UnsafeBufferPointer<Float>, prevAction: [Float], prevReward: [Float], prevDone: [UInt8], accumulatePhaseGpu: Bool) throws -> (ActionBatch, Double, KernelStats) {
        let start = DispatchTime.now().uptimeNanoseconds
        memcpy(latentPower.pointer, latentPower.pointer.advanced(by: batchSize), batchSize * MemoryLayout<Float>.stride)
        let obsCount = batchSize * obsDim
        precondition(obs.count == obsCount)
        memcpy(observation.pointer, obs.baseAddress!, obsCount * MemoryLayout<Float>.stride)
        precondition(prevReward.count == batchSize)
        _ = prevReward.withUnsafeBufferPointer { buf in
            memcpy(reward.pointer, buf.baseAddress!, batchSize * MemoryLayout<Float>.stride)
        }
        let paCount = batchSize * actDim
        precondition(prevAction.count == paCount)
        _ = prevAction.withUnsafeBufferPointer { buf in
            memcpy(previousAction.pointer, buf.baseAddress!, paCount * MemoryLayout<Float>.stride)
        }
        for index in 0..<batchSize { done.pointer[index] = UInt32(prevDone[index]) }
        // Latent predecessor for TD / forward model / GVF targets (must match last step's `state` after encode).
        memcpy(previousState.pointer, state.pointer, batchSize * rank * MemoryLayout<Float>.stride)
        counters.fill(with: 0)
        statsPointer.pointee = KernelStats()
        try runKernel(accumulatePhaseGpu: accumulatePhaseGpu)
        _ = hostAction.withUnsafeMutableBufferPointer { buf in
            memcpy(buf.baseAddress!, actionBuffer.pointer, batchSize * actDim * MemoryLayout<Float>.stride)
        }
        discreteBehavior?.apply(hostAction: &hostAction, hostDiscrete: &hostDiscrete, batchSize: batchSize)
        return (ActionBatch(floats: hostAction, indices: actionIsDiscrete ? hostDiscrete : nil), Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000, statsPointer.pointee)
    }

    private func kernelBuffers() -> [MTLBuffer?] {
        [
            observation.buffer, observationMean.buffer, encoder.buffer, policy.buffer, valueWeights.buffer, rewardWeights.buffer, stateWeights.buffer,
            optionPolicyWeights.buffer, optionValues.buffer, previousState.buffer, previousAction.buffer, reward.buffer, done.buffer, state.buffer, actionBuffer.buffer,
            latentError.buffer, rewardPrediction.buffer, td.buffer, actionGrad.buffer, scalarsBuffer, statsBuffer,
            featureSecondMoment.buffer, featureResidual.buffer, activeMask.buffer, rolloutState.buffer, latentPower.buffer, actionBoundsBuffer, counters.buffer,
            optionCriticWeights.buffer, stompState.buffer
        ]
    }

    private func phaseThreadPlan() -> [(OakKernelPhase, Int)] {
        [
            (.encode, batchSize * rank),
            (.prediction, batchSize),
            (.forwardModel, batchSize * rank),
            (.learn, OakMetalLayout.learnPhaseThreadCount(rank: rank, actDim: actDim)),
            (.metaPolicy, batchSize * actDim),
            (.planning, batchSize),
            (.bookkeeping, 1)
        ]
    }

    private func makeKernelConfig(prim: Int, ofs: Int) -> KernelConfig {
        KernelConfig(
            phase: 0,
            obsDim: UInt32(obsDim),
            rank: UInt32(rank),
            actDim: UInt32(actDim),
            batchSize: UInt32(batchSize),
            actionIsDiscrete: actionIsDiscrete ? 1 : 0,
            optionCount: UInt32(OakMetalLayout.derivedOptionCount(rank: rank, actDim: actDim)),
            primitivePackedSize: UInt32(prim),
            optionForwardStride: UInt32(ofs),
            planK: UInt32(OakMetalLayout.structuralPlanK(rank: rank)),
            baseObsDim: UInt32(baseObsDim)
        )
    }

    /// Per-phase GPU means (ms per batch step) and phase names that had invalid `gpuStartTime`/`gpuEndTime` at least once.
    fileprivate func metalPhaseGpuDiagnostics(measuredBatchSteps: Int) -> (meanMs: [String: Double]?, invalidPhases: [String]?) {
        guard diagnosticGpuPhaseTiming, measuredBatchSteps > 0 else { return (nil, nil) }
        let phaseThreads = phaseThreadPlan()
        var mean: [String: Double] = [:]
        var invalid: [String] = []
        let d = Double(measuredBatchSteps)
        for index in 0..<phaseThreads.count {
            let name = String(describing: phaseThreads[index].0)
            mean[name] = phaseGpuTimeSumSeconds[index] / d * 1000
            if phaseGpuInvalidStepCount[index] > 0 { invalid.append(name) }
        }
        return (mean, invalid.isEmpty ? nil : invalid)
    }

    private func runKernel(accumulatePhaseGpu: Bool) throws {
        if diagnosticGpuPhaseTiming && accumulatePhaseGpu {
            try runKernelSplitRecordingGpu()
        } else {
            try runKernelSingleBuffer()
        }
    }

    private func runKernelSingleBuffer() throws {
        guard let commandBuffer = queue.makeCommandBuffer(descriptor: commandBufferDescriptor) else { throw OakError.message("Failed to create Metal command buffer.") }
        let width = min(max(pipeline.threadExecutionWidth, 1), pipeline.maxTotalThreadsPerThreadgroup)
        let buffers = kernelBuffers()
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw OakError.message("Failed to create Metal compute encoder.") }
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffers(buffers, offsets: bufferZeroOffsets, range: 0..<buffers.count)
        let phaseThreads = phaseThreadPlan()
        let dynBase = OakMetalLayout.optionDynamicsRegionBase(rank: rank)
        let ofs = OakMetalLayout.optionForwardStride(rank: rank, actDim: actDim)
        var config = makeKernelConfig(prim: dynBase, ofs: ofs)
        for (phase, threadCount) in phaseThreads {
            if metalDebugSignposts { computeEncoder.insertDebugSignpost(String(describing: phase)) }
            config.phase = phase.rawValue
            computeEncoder.setBytes(&config, length: MemoryLayout<KernelConfig>.stride, index: OakMetalLayout.kernelConfigBufferIndex)
            let w = max(threadCount, 1)
            computeEncoder.dispatchThreads(MTLSize(width: w, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(width, w), height: 1, depth: 1))
        }
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw OakError.message("Metal command failed: \(error)") }
    }

    private func runKernelSplitRecordingGpu() throws {
        let width = min(max(pipeline.threadExecutionWidth, 1), pipeline.maxTotalThreadsPerThreadgroup)
        let buffers = kernelBuffers()
        let phaseThreads = phaseThreadPlan()
        let dynBase = OakMetalLayout.optionDynamicsRegionBase(rank: rank)
        let ofs = OakMetalLayout.optionForwardStride(rank: rank, actDim: actDim)
        var config = makeKernelConfig(prim: dynBase, ofs: ofs)
        for (index, (phase, threadCount)) in phaseThreads.enumerated() {
            guard let commandBuffer = queue.makeCommandBuffer(descriptor: commandBufferDescriptor) else { throw OakError.message("Failed to create Metal command buffer.") }
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw OakError.message("Failed to create Metal compute encoder.") }
            computeEncoder.setComputePipelineState(pipeline)
            computeEncoder.setBuffers(buffers, offsets: bufferZeroOffsets, range: 0..<buffers.count)
            if metalDebugSignposts { computeEncoder.insertDebugSignpost(String(describing: phase)) }
            config.phase = phase.rawValue
            computeEncoder.setBytes(&config, length: MemoryLayout<KernelConfig>.stride, index: OakMetalLayout.kernelConfigBufferIndex)
            let w = max(threadCount, 1)
            computeEncoder.dispatchThreads(MTLSize(width: w, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(width, w), height: 1, depth: 1))
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw OakError.message("Metal command failed: \(error)") }
            let gpuStart = commandBuffer.gpuStartTime
            let gpuEnd = commandBuffer.gpuEndTime
            if gpuStart > 0, gpuEnd >= gpuStart {
                phaseGpuTimeSumSeconds[index] += gpuEnd - gpuStart
            } else {
                phaseGpuInvalidStepCount[index] += 1
            }
        }
    }
}

private enum RunSnapshot {
    private static let persistLock = NSLock()

    static func write(_ file: RunMetricsFile, metricsDir: URL?) throws -> URL {
        let dir = metricsDir ?? storageFromEnvironment() ?? Main.defaultMetricsDirectory()
        let stem = snapshotStem(envId: file.envId, policy: file.policy, seed: file.seed)
        let url = dir.appendingPathComponent("\(stem).json", isDirectory: false)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try enc.encode(file)
        persistLock.lock()
        defer { persistLock.unlock() }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
        return url
    }

    private static func storageFromEnvironment() -> URL? {
        guard let raw = ProcessInfo.processInfo.environment["OAK_METRICS_STORAGE"]?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
        return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
    }

    fileprivate static func snapshotStem(envId: String, policy: Policy, seed: Int?) -> String {
        let seedPart = seed.map { "seed\($0)" } ?? "seedn"
        let raw = "\(envId)-\(policy.rawValue)-\(seedPart)"
        return raw.replacingOccurrences(of: #"[^A-Za-z0-9]+"#, with: "-", options: .regularExpression)
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
            fputs("\(error)\n", stderr)
            Foundation.exit(1)
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
        let auxProbe = max(ProcessInfo.processInfo.environment["OAK_AUX_DIM"].flatMap(Int.init) ?? 0, 0)
        let defaultNumEnvs = try directNumEnvs(obsDim: probe.obsDim + auxProbe, actDim: probe.actDim, actionDiscreteCount: probe.actionDiscreteCount, device: device)
        let numEnvs = max(ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init) ?? defaultNumEnvs, 1)
        _ = try runSingle(envId: envId, policy: policy, seed: seed, warmup: warmup, stopAfter: max(ProcessInfo.processInfo.environment["OAK_STOP_AFTER_STEPS"].flatMap(Int.init) ?? numEnvs, numEnvs), numEnvs: numEnvs, device: device, command: "run").0
    }

    private static func diagnose(_ args: [String]) throws {
        var envId: String?
        var policy = Policy.oak
        var stopAfter: Int?
        var forcedNumEnvs: Int?
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
        guard !resolvedEnv.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OakError.message("Use `OaK diagnose <env-id>`.")
        }
        let seed = ProcessInfo.processInfo.environment["OAK_SEED"].flatMap(Int.init)
        let warmup = max(ProcessInfo.processInfo.environment["OAK_DIAG_WARMUP_STEPS"].flatMap(Int.init) ?? 0, 0)
        let device = try metalDevice()
        let probe = try EnvPoolEnv(id: resolvedEnv, numEnvs: 1, seed: seed)
        let auxDiag = max(ProcessInfo.processInfo.environment["OAK_AUX_DIM"].flatMap(Int.init) ?? 0, 0)
        let defaultNumEnvs = try directNumEnvs(obsDim: probe.obsDim + auxDiag, actDim: probe.actDim, actionDiscreteCount: probe.actionDiscreteCount, device: device)
        let numEnvs = max(
            forcedNumEnvs
                ?? ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init)
                ?? min(defaultNumEnvs, ProcessInfo.processInfo.environment["OAK_DIAG_MAX_ENVS"].flatMap(Int.init) ?? 8),
            1
        )
        let (_, metricsURL) = try runSingle(
            envId: resolvedEnv,
            policy: policy,
            seed: seed,
            warmup: warmup,
            stopAfter: max(stopAfter ?? ProcessInfo.processInfo.environment["OAK_DIAG_STEPS"].flatMap(Int.init) ?? (numEnvs * 8), numEnvs),
            numEnvs: numEnvs,
            device: device,
            command: "diagnose",
            includeDiagnostic: true
        )
        fputs("\(metricsURL.path)\n", stderr)
    }

    private static func benchmark() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let benchDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["OAK_BENCH_DIR"] ?? root.appendingPathComponent(".oak-benchmarks/\(oakBenchmarksRunId())").path)
        try FileManager.default.createDirectory(at: benchDir, withIntermediateDirectories: true)
        let benchMetrics: URL = {
            if let raw = ProcessInfo.processInfo.environment["OAK_METRICS_STORAGE"]?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
                return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
            }
            return benchDir.appendingPathComponent("metrics", isDirectory: true)
        }()
        let seedStart = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_START"].flatMap(Int.init) ?? 0
        let seedCount = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 10
        let warmup = max(ProcessInfo.processInfo.environment["OAK_BENCH_WARMUP_STEPS"].flatMap(Int.init) ?? 1000, 0)
        let stopAfter = max(ProcessInfo.processInfo.environment["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 50000, 1)
        let policies = try (ProcessInfo.processInfo.environment["OAK_BENCH_POLICIES"] ?? "oak,random").split(separator: ",").map { try parsePolicy(String($0)) }
        let device = try metalDevice()
        let forcedNumEnvs = ProcessInfo.processInfo.environment["OAK_BENCH_NUM_ENVS"].flatMap(Int.init) ?? ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init)
        let auxBench = max(ProcessInfo.processInfo.environment["OAK_AUX_DIM"].flatMap(Int.init) ?? 0, 0)
        for benchmarkEnv in try benchmarkEnvOrder(seed: seedStart) {
            let envId = benchmarkEnv.envId
            let defaultNumEnvs = try directNumEnvs(
                obsDim: benchmarkEnv.schema.observationDim + auxBench,
                actDim: benchmarkEnv.schema.actionDim,
                actionDiscreteCount: benchmarkEnv.schema.actionDiscreteCount,
                device: device
            )
            let numEnvs = max(forcedNumEnvs ?? defaultNumEnvs, 1)
            for seed in seedStart..<(seedStart + seedCount) {
                for policy in policies {
                    let stem = RunSnapshot.snapshotStem(envId: envId, policy: policy, seed: seed)
                    let logURL = benchDir.appendingPathComponent("\(stem).log")
                    do {
                        _ = try runSingle(
                            envId: envId,
                            policy: policy,
                            seed: seed,
                            warmup: warmup,
                            stopAfter: stopAfter,
                            numEnvs: numEnvs,
                            device: device,
                            command: "benchmark",
                            metricsDir: benchMetrics
                        ).0
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
                ordered.append(groups[index].removeFirst())
                appended = true
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
        let env = try EnvPoolEnv(id: envId, numEnvs: numEnvs, seed: seed)
        let encoder = seed.map { GKMersenneTwisterRandomSource(seed: UInt64(bitPattern: Int64($0))) }
        let auxDim = max(ProcessInfo.processInfo.environment["OAK_AUX_DIM"].flatMap(Int.init) ?? 0, 0)
        let totalObsDim = env.obsDim + auxDim
        // Invariant: representation rank equals the observation interface width (env + optional aux); not a fixed table size.
        let engine = policy == .oak
            ? try OakMetalEngine(
                obsDim: totalObsDim,
                baseObsDim: env.obsDim,
                actDim: env.actDim,
                batchSize: numEnvs,
                rank: totalObsDim,
                actionIsDiscrete: env.actionDiscreteCount != nil,
                actionLow: env.actionLow,
                actionHigh: env.actionHigh,
                device: device,
                seed: seed,
                diagnosticGpuPhaseTiming: includeDiagnostic
            )
            : nil
        try env.reset()
        let rowObsBytes = env.obsDim * MemoryLayout<Float>.stride
        let auxPadBytes = max(totalObsDim - env.obsDim, 0) * MemoryLayout<Float>.stride
        /// When `auxDim == 0`, rank equals env obs width: feed Metal from `env.observationScratch` directly (no duplicate host buffer + memcpy per step).
        let useDirectEnvObs = auxDim == 0
        var obsPadded: [Float]?
        if !useDirectEnvObs {
            var padded = [Float](repeating: 0, count: numEnvs * totalObsDim)
            env.observationScratch.withUnsafeBufferPointer { src in
                padded.withUnsafeMutableBufferPointer { dst in
                    guard let sb = src.baseAddress, let db = dst.baseAddress else { return }
                    for row in 0..<numEnvs {
                        memcpy(db.advanced(by: row * totalObsDim), sb.advanced(by: row * env.obsDim), rowObsBytes)
                        if auxPadBytes > 0 { memset(db.advanced(by: row * totalObsDim + env.obsDim), 0, auxPadBytes) }
                    }
                }
            }
            obsPadded = padded
        }
        var prevReward = Array(repeating: Float(0), count: numEnvs)
        var prevDone = Array(repeating: UInt8(0), count: numEnvs)
        var prevActionScratch = [Float](repeating: 0, count: env.actDim * numEnvs)
        var randomActionScratch = Array(repeating: Float(0), count: env.actDim * numEnvs)
        var randomIndexScratch = Array(repeating: Int32(0), count: numEnvs)
        var totalSteps = 0
        var measuredSteps = 0
        var measuredSeconds = 0.0
        var measurementStart: UInt64?
        var rewardSum = 0.0
        var doneCount = 0
        var agentMsSum = 0.0
        var envMsSum = 0.0
        var health = HealthAccumulator()
        let actionCol = MeasuredActionCollector(discreteCount: env.actionDiscreteCount, actDim: env.actDim)
        while measuredSteps < stopAfter {
            let nextTotal = totalSteps + numEnvs
            let measured = nextTotal > warmup && totalSteps > 0
            if measured && measurementStart == nil { measurementStart = DispatchTime.now().uptimeNanoseconds }
            let actionBatch: ActionBatch
            let agentMs: Double
            let stats: KernelStats?
            if let engine {
                let output: (ActionBatch, Double, KernelStats)
                if useDirectEnvObs {
                    output = try env.observationScratch.withUnsafeBufferPointer { buf in
                        try engine.step(obs: buf, prevAction: prevActionScratch, prevReward: prevReward, prevDone: prevDone, accumulatePhaseGpu: measured)
                    }
                } else {
                    output = try obsPadded!.withUnsafeBufferPointer { buf in
                        try engine.step(obs: buf, prevAction: prevActionScratch, prevReward: prevReward, prevDone: prevDone, accumulatePhaseGpu: measured)
                    }
                }
                actionBatch = output.0
                agentMs = output.1
                stats = output.2
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
            if !useDirectEnvObs {
                env.observationScratch.withUnsafeBufferPointer { src in
                    obsPadded!.withUnsafeMutableBufferPointer { dst in
                        guard let sb = src.baseAddress, let db = dst.baseAddress else { return }
                        for row in 0..<numEnvs {
                            memcpy(db.advanced(by: row * totalObsDim), sb.advanced(by: row * env.obsDim), rowObsBytes)
                            if auxPadBytes > 0 { memset(db.advanced(by: row * totalObsDim + env.obsDim), 0, auxPadBytes) }
                        }
                    }
                }
            }
            _ = prevReward.withUnsafeMutableBufferPointer { dst in
                env.rewardScratch.withUnsafeBufferPointer { src in
                    memcpy(dst.baseAddress!, src.baseAddress!, numEnvs * MemoryLayout<Float>.stride)
                }
            }
            _ = prevDone.withUnsafeMutableBufferPointer { dst in
                env.doneScratch.withUnsafeBufferPointer { src in
                    memcpy(dst.baseAddress!, src.baseAddress!, numEnvs * MemoryLayout<UInt8>.stride)
                }
            }
            prevActionScratch.withUnsafeMutableBufferPointer { dst in
                actionBatch.floats.withUnsafeBufferPointer { src in
                    guard let d = dst.baseAddress, let s = src.baseAddress, src.count == dst.count else { return }
                    memcpy(d, s, src.count * MemoryLayout<Float>.stride)
                }
            }
        }
        let updates = max(measuredSteps / numEnvs, 1)
        let agentSeconds = agentMsSum / 1000
        let envSeconds = envMsSum / 1000
        let hostSeconds = max(measuredSeconds - agentSeconds - envSeconds, 0)
        let diagnostic: RunDiagnostic? = health.diagnosticSnapshot().map { base in
            let (metalMean, metalInvalid) = engine?.metalPhaseGpuDiagnostics(measuredBatchSteps: updates) ?? (nil, nil)
            return RunDiagnostic(
                nonFiniteByPhase: base.nonFiniteByPhase,
                maxAbsLatentPrediction: base.maxAbsLatentPrediction,
                maxAbsRewardPrediction: base.maxAbsRewardPrediction,
                maxAbsPolicyOutput: base.maxAbsPolicyOutput,
                agentShare: measuredSeconds > 0 ? agentSeconds / measuredSeconds : nil,
                envShare: measuredSeconds > 0 ? envSeconds / measuredSeconds : nil,
                hostShare: measuredSeconds > 0 ? hostSeconds / measuredSeconds : nil,
                metalPhaseGpuMsMean: metalMean,
                metalPhaseGpuTimingInvalid: metalInvalid
            )
        }
        let stepsPerSecond = measuredSeconds > 0 ? Double(measuredSteps) / measuredSeconds : nil
        let (discreteAction, continuousAction) = actionCol.finish(actDim: env.actDim)
        let file = RunMetricsFile(
            schemaVersion: 2,
            command: command,
            envId: envId,
            policy: policy,
            seed: seed,
            numEnvs: numEnvs,
            warmupSteps: warmup,
            stopAfterMeasuredSteps: stopAfter,
            observationDim: totalObsDim,
            actionDim: env.actDim,
            actionDiscreteCount: env.actionDiscreteCount,
            measuredStepCount: measuredSteps,
            measuredBatchCount: updates,
            measuredSeconds: measuredSeconds,
            stepsPerSecond: stepsPerSecond,
            rewardSum: rewardSum,
            doneCount: doneCount,
            stopReason: "measured_steps",
            agentStepMsMean: agentMsSum / Double(updates),
            envStepMsMean: envMsSum / Double(updates),
            discreteAction: discreteAction,
            continuousAction: continuousAction,
            health: health.snapshot(),
            diagnostic: includeDiagnostic ? diagnostic : nil
        )
        let metricsURL = try RunSnapshot.write(file, metricsDir: metricsDir)
        return (file, metricsURL)
    }

    fileprivate static func oakBenchmarksRunId() -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .iso8601)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date()) + "-\(ProcessInfo.processInfo.processIdentifier)"
    }

    fileprivate static func defaultMetricsDirectory() -> URL {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return root.appendingPathComponent(".oak-benchmarks/\(oakBenchmarksRunId())/metrics", isDirectory: true)
    }
}
