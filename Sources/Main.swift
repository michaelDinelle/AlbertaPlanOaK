import Foundation
import Metal
import Darwin

// =================================================================================================
// Orthodox Step 11 OaK — host-side dispatcher. See Sources/OakKernel.metal for the full citation
// block and the nine-invariant Alberta-Plan contract. This file owns: arena sizing, Metal engine,
// env pool FFI, CLI commands (run/diagnose/benchmark), and run-metrics JSON emission.
// =================================================================================================

enum OakError: Error, CustomStringConvertible {
    case message(String)
    var description: String { if case let .message(m) = self { return m }; return "" }
}

enum Policy: String, Codable { case oak, random }

// ------------------------------------------------------------------------------------------------ Run metrics JSON (AGENTS.md: schemaVersion 2).
struct RunHealth: Codable {
    let rewardRate: Double?
    let latentStateMeanSquare: Double?
    let activeRank: Double?
    let actionDeltaL2Mean: Double?
    let planningActionChangeL2Mean: Double?
    let optionModelPredictionMSE: Double?
    let oakOptionReinitCount: Double?
    let nonFiniteRawTotal: Int
    let nonFiniteSanitizedTotal: Int
}

struct RunDiagnostic: Codable {
    let nonFiniteByPhase: [String: Int]
    let maxAbsLatentPrediction: Double?
    let maxAbsPolicyOutput: Double?
    let agentShare: Double?
    let envShare: Double?
    let hostShare: Double?
    let metalPhaseGpuMsMean: [String: Double]?
    let metalPhaseGpuTimingInvalid: [String]?
}

private struct DiscreteActionMetrics: Encodable { let histogram: [Int]; let uniqueCount: Int }
private struct ContinuousActionMetrics: Encodable {
    let meanPerDim: [Double]; let stdPerDim: [Double]; let meanL2: Double; let stdL2: Double
}

private struct RunMetricsFile: Encodable {
    let schemaVersion: Int, command: String, envId: String, policy: Policy
    let seed: Int?
    let numEnvs: Int, warmupSteps: Int, stopAfterMeasuredSteps: Int
    let observationDim: Int, actionDim: Int, actionDiscreteCount: Int?
    let measuredStepCount: Int, measuredBatchCount: Int, measuredSeconds: Double
    let stepsPerSecond: Double?, heartbeatSequence: Int?
    let rewardSum: Double, doneCount: Int, stopReason: String
    let agentStepMsMean: Double, envStepMsMean: Double
    let discreteAction: DiscreteActionMetrics?, continuousAction: ContinuousActionMetrics?
    let health: RunHealth?, diagnostic: RunDiagnostic?
}

struct ActionBatch { let floats: [Float]; let indices: [Int32]? }

private struct Schema: Hashable, Comparable {
    let observationDim: Int, actionDim: Int, actionDiscreteCount: Int?
    static func < (lhs: Schema, rhs: Schema) -> Bool {
        (lhs.observationDim, lhs.actionDim, lhs.actionDiscreteCount ?? -1)
            < (rhs.observationDim, rhs.actionDim, rhs.actionDiscreteCount ?? -1)
    }
}

private struct BenchmarkEnv { let envId: String; let schema: Schema }

// Structural neutral seed: integer 0. Used only when the caller does not supply an explicit
// seed for the splitmix32 hash that initializes parameter weights and derives kernel-side RNG.
// Under the Big World Hypothesis the agent's learning dynamics are invariant to seed in
// expectation; this constant just fixes "no seed" → "deterministic, reproducible" so two
// unseeded runs are bit-identical for debugging. No magic: 0 is the additive identity of the
// seedBase multiplication, the most neutral value possible.
private let oakUnseededSeed: Int32 = 0

// ------------------------------------------------------------------------------------------------
// Filesystem helpers.
// ------------------------------------------------------------------------------------------------
private func oakRoots() -> [URL] {
    let fm = FileManager.default
    let exe = URL(fileURLWithPath: CommandLine.arguments[0],
                  relativeTo: URL(fileURLWithPath: fm.currentDirectoryPath)).standardizedFileURL
    return [
        fm.currentDirectoryPath,
        exe.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().path,
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

// ------------------------------------------------------------------------------------------------
// Health accumulator — lean fold over KernelStats. Seven slots, one per reported mean.
// ------------------------------------------------------------------------------------------------
private struct HealthAccumulator {
    struct Slot { var sum = 0.0; var count = 0 }
    // Slot order must mirror `values` in `record()` and the field order in `snapshot()`:
    //   0 latentStateMeanSquare, 1 activeRank, 2 actionDeltaL2Mean,
    //   3 planningActionChangeL2Mean, 4 optionModelPredictionMSE, 5 oakOptionReinitCount
    var slots = [Slot](repeating: Slot(), count: 6)
    var steps = 0
    var rawNonFinite = 0, sanitizedNonFinite = 0
    var encodeNF = 0, learnNF = 0, planNF = 0
    var maxLat = 0.0, maxPol = 0.0
    var lastRho: Double?

    mutating func record(_ stats: KernelStats) {
        steps += 1
        let rr = Double(stats.rewardRate); if rr.isFinite { lastRho = rr }
        let values: [Float] = [
            stats.latentStateMeanSquare,
            stats.activeRank,
            stats.actionDeltaL2Mean,
            stats.planningActionChangeL2Mean,
            stats.optionModelPredictionMSE,
            stats.oakOptionReinitCount,
        ]
        for (i, f) in values.enumerated() {
            let v = Double(f)
            if v.isFinite { slots[i].sum += v; slots[i].count += 1 }
            else { rawNonFinite += 1; sanitizedNonFinite += 1 }
        }
        rawNonFinite += Int(stats.rawNonFiniteCount)
        sanitizedNonFinite += Int(stats.sanitizedNonFiniteCount)
        encodeNF += Int(stats.encodeNonFiniteCount)
        learnNF += Int(stats.learnNonFiniteCount)
        planNF += Int(stats.planNonFiniteCount)
        maxLat = max(maxLat, Double(stats.maxAbsLatentPrediction))
        maxPol = max(maxPol, Double(stats.maxAbsPolicyOutput))
    }

    private func mean(_ i: Int) -> Double? { slots[i].count > 0 ? slots[i].sum / Double(slots[i].count) : nil }

    func snapshot() -> RunHealth? {
        guard steps > 0 else { return nil }
        return RunHealth(
            rewardRate: lastRho,
            latentStateMeanSquare: mean(0),
            activeRank: mean(1),
            actionDeltaL2Mean: mean(2),
            planningActionChangeL2Mean: mean(3),
            optionModelPredictionMSE: mean(4),
            oakOptionReinitCount: mean(5),
            nonFiniteRawTotal: rawNonFinite,
            nonFiniteSanitizedTotal: sanitizedNonFinite)
    }

    func diagnosticSnapshot() -> RunDiagnostic? {
        guard steps > 0 else { return nil }
        return RunDiagnostic(
            nonFiniteByPhase: ["encode": encodeNF, "learn": learnNF, "plan": planNF],
            maxAbsLatentPrediction: maxLat, maxAbsPolicyOutput: maxPol,
            agentShare: nil, envShare: nil, hostShare: nil,
            metalPhaseGpuMsMean: nil, metalPhaseGpuTimingInvalid: nil)
    }
}

// ------------------------------------------------------------------------------------------------
// Measured action collector.
// ------------------------------------------------------------------------------------------------
private final class MeasuredActionCollector {
    private var discrete: [Int]?
    private var sum: [Double]?, sumsq: [Double]?
    private var l2s = 0.0, l2sq = 0.0, n = 0

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
            var l = 0.0
            let b = row * actDim
            for d in 0..<actDim { let v = Double(f[b + d]); s[d] += v; q[d] += v * v; l += v * v }
            let r = sqrt(l); l2s += r; l2sq += r * r; n += 1
        }
        sum = s; sumsq = q
    }

    func partialMetrics(actDim: Int) -> (DiscreteActionMetrics?, ContinuousActionMetrics?) {
        let d = discrete.map { DiscreteActionMetrics(histogram: $0, uniqueCount: $0.filter { $0 > 0 }.count) }
        guard let s = sum, let q = sumsq, n > 0 else { return (d, nil) }
        let nn = Double(n)
        let mean = (0..<actDim).map { s[$0] / nn }
        let std  = (0..<actDim).map { let m = mean[$0]; return sqrt(max(0, q[$0] / nn - m * m)) }
        let ml = l2s / nn
        return (d, ContinuousActionMetrics(meanPerDim: mean, stdPerDim: std, meanL2: ml,
                                           stdL2: sqrt(max(0, l2sq / nn - ml * ml))))
    }
}

// ================================================================================================
// Native env bridge (envpool FFI) — unchanged contract.
// ================================================================================================
final class NativeBridge: @unchecked Sendable {
    typealias OpenFn = @convention(c) (UnsafePointer<CChar>, Int32, Int32, UnsafePointer<CChar>,
                                       UnsafeMutablePointer<CChar>?, Int32) -> UnsafeMutableRawPointer?
    typealias CloseFn = @convention(c) (UnsafeMutableRawPointer?) -> Void
    typealias IntFn = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    typealias BoundsFn = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<Float>?,
                                         UnsafeMutablePointer<Float>?) -> Int32
    typealias ResetFn = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<Float>?,
                                        UnsafeMutablePointer<CChar>?, Int32) -> Int32
    typealias StepContFn = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Float>?,
                                           UnsafeMutablePointer<Float>?, UnsafeMutablePointer<Float>?,
                                           UnsafeMutablePointer<UInt8>?, UnsafeMutablePointer<CChar>?,
                                           Int32) -> Int32
    typealias StepDiscFn = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Int32>?,
                                           UnsafeMutablePointer<Float>?, UnsafeMutablePointer<Float>?,
                                           UnsafeMutablePointer<UInt8>?, UnsafeMutablePointer<CChar>?,
                                           Int32) -> Int32
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
        let url = try requiredBuildPath(".build/oak-native/liboakenv.dylib",
                                        description: ".build/oak-native/liboakenv.dylib")
        guard let handle = dlopen(url.path, RTLD_NOW | RTLD_LOCAL) else {
            throw OakError.message(String(cString: dlerror()))
        }
        dylib = handle
        func load<T>(_ name: String) throws -> T {
            guard let raw = dlsym(handle, name) else {
                throw OakError.message("Missing symbol \(name) in native env bridge.")
            }
            return unsafeBitCast(raw, to: T.self)
        }
        openFn = try load("oak_env_open"); closeFn = try load("oak_env_close")
        obsDimFn = try load("oak_env_obs_dim"); actDimFn = try load("oak_env_act_dim")
        discreteCountFn = try load("oak_env_discrete_count")
        boundsFn = try load("oak_env_action_bounds"); resetFn = try load("oak_env_reset")
        stepContFn = try load("oak_env_step_continuous"); stepDiscFn = try load("oak_env_step_discrete")
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

    func takeWarning() -> String? {
        var error = errorBuffer()
        return takeWarningFn(&error, Int32(error.count)) == 1 ? decodeCString(error) : nil
    }
    func open(envId: String, numEnvs: Int, seed: Int?) throws -> UnsafeMutableRawPointer {
        var error = errorBuffer()
        let actualSeed = Int32(seed ?? Int(oakUnseededSeed))
        let handle = envId.withCString { e in
            assetBasePath.withCString { b in
                openFn(e, Int32(numEnvs), actualSeed, b, &error, Int32(error.count))
            }
        }
        guard let handle else { throw OakError.message(decodeCString(error)) }
        return handle
    }
    func close(_ h: UnsafeMutableRawPointer) { closeFn(h) }
    func obsDim(_ h: UnsafeMutableRawPointer) -> Int { Int(obsDimFn(h)) }
    func actDim(_ h: UnsafeMutableRawPointer) -> Int { Int(actDimFn(h)) }
    func discreteCount(_ h: UnsafeMutableRawPointer) -> Int? {
        let c = Int(discreteCountFn(h)); return c > 0 ? c : nil
    }
    func actionBounds(_ h: UnsafeMutableRawPointer, actDim: Int) -> ([Float], [Float]) {
        var lo = Array(repeating: Float(0), count: actDim)
        var hi = Array(repeating: Float(0), count: actDim)
        let c = Int(boundsFn(h, &lo, &hi)); return c > 0 ? (lo, hi) : ([], [])
    }
    func reset(_ h: UnsafeMutableRawPointer, obs: inout [Float]) throws {
        try checked { e, cap in obs.withUnsafeMutableBufferPointer { resetFn(h, $0.baseAddress, e, cap) } }
    }
    func stepContinuous(_ h: UnsafeMutableRawPointer, action: [Float],
                        obs: inout [Float], reward: inout [Float], done: inout [UInt8]) throws {
        try checked { e, cap in
            action.withUnsafeBufferPointer { ab in
                obs.withUnsafeMutableBufferPointer { ob in
                    reward.withUnsafeMutableBufferPointer { rb in
                        done.withUnsafeMutableBufferPointer { db in
                            stepContFn(h, ab.baseAddress, ob.baseAddress, rb.baseAddress,
                                       db.baseAddress, e, cap)
                        }}}}}
    }
    func stepDiscrete(_ h: UnsafeMutableRawPointer, action: [Int32],
                      obs: inout [Float], reward: inout [Float], done: inout [UInt8]) throws {
        try checked { e, cap in
            action.withUnsafeBufferPointer { ab in
                obs.withUnsafeMutableBufferPointer { ob in
                    reward.withUnsafeMutableBufferPointer { rb in
                        done.withUnsafeMutableBufferPointer { db in
                            stepDiscFn(h, ab.baseAddress, ob.baseAddress, rb.baseAddress,
                                       db.baseAddress, e, cap)
                        }}}}}
    }
    func benchmarkEnvIds() -> [String] {
        (0..<Int(benchmarkCountFn())).compactMap { i in benchmarkIdFn(Int32(i)).map { String(cString: $0) } }
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
        obsDim = bridge.obsDim(handle); actDim = bridge.actDim(handle)
        actionDiscreteCount = bridge.discreteCount(handle)
        (actionLow, actionHigh) = bridge.actionBounds(handle, actDim: actDim)
        observationScratch = Array(repeating: 0, count: numEnvs * obsDim)
        rewardScratch = Array(repeating: 0, count: numEnvs)
        doneScratch = Array(repeating: 0, count: numEnvs)
    }
    deinit { bridge.close(handle) }

    private func checked<T>(_ body: () throws -> T) throws -> T {
        let v = try body()
        if let w = bridge.takeWarning() { throw OakError.message("MuJoCo warning: \(w)") }
        return v
    }
    func reset() throws { try checked { try bridge.reset(handle, obs: &observationScratch) } }
    func step(_ action: ActionBatch) throws {
        try checked {
            if let idx = action.indices {
                try bridge.stepDiscrete(handle, action: idx, obs: &observationScratch,
                                        reward: &rewardScratch, done: &doneScratch)
            } else {
                try bridge.stepContinuous(handle, action: action.floats, obs: &observationScratch,
                                          reward: &rewardScratch, done: &doneScratch)
            }
        }
    }
}

// ================================================================================================
// Kernel config + scalars + stats — mirrors OakKernel.metal structs byte-for-byte. Keep in lockstep.
// KernelStats has 16 fields (4 dead fields removed from previous 20).
// ================================================================================================
private struct KernelConfig {
    var obsDim: UInt32 = 0, rank: UInt32 = 0, actDim: UInt32 = 0, batch: UInt32 = 0
    var optionCount: UInt32 = 0, baseObsDim: UInt32 = 0
    var actionIsDiscrete: UInt32 = 0
    // actionParamCount = numActions (discrete one-hot) | 2·actDim (continuous Gaussian [μ, log_σ]).
    // Determines D in optPolicy[K · D · R] and scoreGrad[B · D]. Set at init from action type.
    var actionParamCount: UInt32 = 0
    var pEncoder: UInt32 = 0, pValue: UInt32 = 0, pPsi: UInt32 = 0, pOptPolicy: UInt32 = 0
    var sObservation: UInt32 = 0, sObsMean: UInt32 = 0, sState: UInt32 = 0
    var sPrevAction: UInt32 = 0, sAction: UInt32 = 0, sScoreGrad: UInt32 = 0
    var sReward: UInt32 = 0, sTd: UInt32 = 0, sTdPsi: UInt32 = 0
    var uDone: UInt32 = 0, uActiveMask: UInt32 = 0, uActiveList: UInt32 = 0
    var uOption: UInt32 = 0, uCounters: UInt32 = 0, uSubtaskFeatureOf: UInt32 = 0
    var aFeatSM: UInt32 = 0, aWBar: UInt32 = 0, aFeatMean: UInt32 = 0
    var aUtilFeat: UInt32 = 0, aUtilOpt: UInt32 = 0
    var cap: Float = 0, alpha0: Float = 0, ridge: Float = 0
    var boundLat: Float = 0, boundRew: Float = 0, boundAct: Float = 0
    var invBoundLat: Float = 0, invBoundRew: Float = 0, invBoundAct: Float = 0
}

private struct KernelScalars {
    var stepCount: UInt32 = 0, doneCount: UInt32 = 0, activeCount: UInt32 = 0
    var stateSlot: UInt32 = 0, optionSlot: UInt32 = 0
    var _pad: UInt32 = 0
    var rho: Float = 0, pEff: Float = 0
    var alphaImp: Float = 0, alphaEnc: Float = 0
}

struct KernelStats {
    var rewardRate: Float = 0, latentStateMeanSquare: Float = 0, activeRank: Float = 0
    var rawNonFiniteCount: Float = 0, sanitizedNonFiniteCount: Float = 0
    var encodeNonFiniteCount: Float = 0, learnNonFiniteCount: Float = 0, planNonFiniteCount: Float = 0
    var maxAbsLatentPrediction: Float = 0, maxAbsPolicyOutput: Float = 0
    var actionDeltaL2Mean: Float = 0, planningActionChangeL2Mean: Float = 0
    var optionModelPredictionMSE: Float = 0
    var oakOptionReinitCount: Float = 0
    var utilityFeatureMean: Float = 0, utilityOptionMean: Float = 0
}

// ================================================================================================
// Arena layout — single source of truth (Metal reads the offsets from KernelConfig).
// ================================================================================================
private enum OakMetalLayout {
    static let kernelConfigBufferIndex = 6
    // Structural cap: one SIMD group of options per row in oak_plan (hardware-derived, not tuned).
    static let maxOptionCount = 32

    static func optionCount(rank: Int, actDim: Int) -> Int {
        max(1, min(rank, actDim, maxOptionCount))
    }
    // Concat dim M = R + K: shared chord-conditional ψ(s, λ) = W_ψ · concat[s, λ] (Alberta Plan
    // Design 2; OBB 2505.00787 §2.4 Eq.4). Replaces K independent K·R·R per-option SF matrices.
    static func concatDim(rank: Int, actDim: Int) -> Int {
        rank + optionCount(rank: rank, actDim: actDim)
    }
    static func psiCount(rank: Int, actDim: Int) -> Int {
        optionCount(rank: rank, actDim: actDim) * concatDim(rank: rank, actDim: actDim)
    }
    // D_params: numActions for discrete (one logit per action), 2·actDim for continuous Gaussian
    // ([μ, log_σ] per action dim). Structural — derived purely from action-type signal.
    static func actionParamCount(actDim: Int, actionDiscreteCount: Int?) -> Int {
        if let d = actionDiscreteCount, d > 0 { return d }
        return 2 * actDim
    }
    static func optPolicyCount(rank: Int, actDim: Int, actionDiscreteCount: Int?) -> Int {
        optionCount(rank: rank, actDim: actDim)
            * actionParamCount(actDim: actDim, actionDiscreteCount: actionDiscreteCount)
            * rank
    }
    static func discreteLogitHalfWidth(rank: Int, actDim: Int) -> Float {
        min(sqrtf(Float(max(rank * actDim, 1))), logf(Float.greatestFiniteMagnitude))
    }
    static func statsAbsBoundForSumSquares(termCount: Int) -> Float {
        sqrtf(Float.greatestFiniteMagnitude / Float(max(termCount, 1)))
    }

    struct Param { let encoder, value, psi, optPolicy, total: Int }
    static func paramLayout(obsDim: Int, rank: Int, actDim: Int, actionDiscreteCount: Int?) -> Param {
        var o = 0
        let enc = o; o += rank
        let val = o; o += rank
        let psi = o; o += psiCount(rank: rank, actDim: actDim)
        let op  = o; o += optPolicyCount(rank: rank, actDim: actDim,
                                          actionDiscreteCount: actionDiscreteCount)
        return Param(encoder: enc, value: val, psi: psi, optPolicy: op, total: o)
    }

    struct State {
        let observation, obsMean, state, prevAction, action, scoreGrad: Int
        let reward, td, tdPsi: Int
        let featSM, wBar, featMean, utilFeat, utilOpt: Int
        let total: Int
    }
    static func stateLayout(obsDim: Int, rank: Int, actDim: Int, batch: Int,
                            actionDiscreteCount: Int?) -> State {
        let K = optionCount(rank: rank, actDim: actDim)
        let D = actionParamCount(actDim: actDim, actionDiscreteCount: actionDiscreteCount)
        var o = 0
        let obs = o; o += batch * obsDim
        let mean = o; o += obsDim
        let st = o; o += 2 * batch * rank
        let prA = o; o += batch * actDim
        let act = o; o += batch * actDim
        let sg  = o; o += batch * D       // score function ∇log π(a|s) for the SAMPLED action
        let rew = o; o += batch
        let td  = o; o += batch
        let tdP = o; o += batch * K       // K cumulant TDs per row (Design 2 collapse)
        let fs  = o; o += rank
        let wb  = o; o += rank
        let fm  = o; o += rank            // per-feature running mean (cumulant centering ρ_k)
        let uf  = o; o += rank
        let uo  = o; o += K
        return State(observation: obs, obsMean: mean, state: st, prevAction: prA, action: act,
                     scoreGrad: sg, reward: rew, td: td, tdPsi: tdP, featSM: fs, wBar: wb,
                     featMean: fm, utilFeat: uf, utilOpt: uo, total: o)
    }

    struct UInts { let done, activeMask, activeList, option, counters, subtaskFeatureOf, total: Int }
    static func uintLayout(rank: Int, actDim: Int, batch: Int) -> UInts {
        let K = optionCount(rank: rank, actDim: actDim)
        var o = 0
        let d = o; o += batch
        let am = o; o += rank
        let al = o; o += rank
        let op = o; o += 2 * batch
        let c  = o; o += 6
        let sf = o; o += K  // sf[k] = state-feature index that cumulant k tracks (top-K by utility)
        return UInts(done: d, activeMask: am, activeList: al, option: op, counters: c,
                     subtaskFeatureOf: sf, total: o)
    }
}

// ================================================================================================
// Shared-storage helpers.
// ================================================================================================
private final class Shared<Element> {
    let buffer: MTLBuffer
    let pointer: UnsafeMutablePointer<Element>
    let count: Int
    init(device: MTLDevice, count: Int, zero: Element) {
        self.count = count
        buffer = device.makeBuffer(length: max(count, 1) * MemoryLayout<Element>.stride,
                                   options: [.storageModeShared])!
        pointer = buffer.contents().bindMemory(to: Element.self, capacity: max(count, 1))
        pointer.initialize(repeating: zero, count: max(count, 1))
    }
}

// ================================================================================================
// Metal engine — orthodox Step 11 OaK. Twelve phases, one dispatch per step.
// ================================================================================================
final class OakMetalEngine {
    // Phase metadata: kernel name + threadgroup size (0 = pipeline-default dispatchThreads).
    // Reduction kernels use fixed 128-thread TGs (4 SIMD groups × 32) with hierarchical simd_sum;
    // plan uses 32 (one SIMD group per row, SIMD-shuffle argmax).
    private struct Phase { let name: String; let tgSize: Int }
    // Phase ordering: meta_action runs LAST among learn-time kernels so that learn_pi reads
    // the score function written at the *previous* step's meta_action — i.e., the score for
    // action_{N-1} taken at state_{N-1}, which is the action that produced the transition
    // whose TD δ_N is being learned from. No prevScoreGrad slot needed; arena holds a single
    // scoreGrad slot that is read by learn_pi (this step) before being overwritten by
    // meta_action (this step) for the next step's learning.
    private static let phases: [Phase] = [
        Phase(name: "oak_encode",             tgSize: 0),
        Phase(name: "oak_predict",            tgSize: 0),
        Phase(name: "oak_plan",               tgSize: 32),
        Phase(name: "oak_reduce_alpha",       tgSize: 128),
        Phase(name: "oak_learn_value",        tgSize: 32),
        Phase(name: "oak_learn_encoder",      tgSize: 0),
        Phase(name: "oak_learn_psi",          tgSize: 0),
        Phase(name: "oak_learn_pi",           tgSize: 0),
        Phase(name: "oak_meta_action",        tgSize: 0),
        Phase(name: "oak_bk_reduce",          tgSize: 128),
        Phase(name: "oak_bk_feature",         tgSize: 32),
        Phase(name: "oak_bk_option",          tgSize: 128),
        Phase(name: "oak_bk_admit_commit",    tgSize: 0),
    ]

    private let obsDim, baseObsDim, actDim, batch, rank: Int
    private let actionIsDiscrete: Bool
    private let actionParamCount: Int
    private let queue: MTLCommandQueue
    private let cbDesc: MTLCommandBufferDescriptor
    private let pipelines: [MTLComputePipelineState]
    private let actionBoundsBuffer: MTLBuffer
    private let scalarsBuffer: MTLBuffer
    private let statsBuffer: MTLBuffer
    private let configBuffer: MTLBuffer
    private let scalarsPointer: UnsafeMutablePointer<KernelScalars>
    private let statsPointer: UnsafeMutablePointer<KernelStats>
    private let configPointer: UnsafeMutablePointer<KernelConfig>
    private let paramsArena: Shared<Float16>
    private let stateArena: Shared<Float>
    private let uintArena: Shared<UInt32>
    private let stateLayout: OakMetalLayout.State
    private let uintLayout: OakMetalLayout.UInts
    private let phaseGridWidths: [Int]
    private let diagnosticGpuPhaseTiming: Bool
    private var phaseGpuTimeSumSeconds: [Double]
    private var phaseGpuInvalidStepCount: [Int]
    private var hostAction: [Float]
    private var hostDiscrete: [Int32]
    // Per-action-dim env bounds. Continuous actions clamped here at FFI boundary so kernel can
    // emit raw policy logits without internal range projection (Tier 2.2 / 6.2).
    private let envActionLow: [Float]
    private let envActionHigh: [Float]
    // Metal 4 ICB + MTLResidencySet (macOS 26). Pre-encoded 12 dispatches execute in one encoder
    // on the hot path, eliminating per-phase encoder setup (~50µs × 12 = ~600µs/step host savings).
    private let indirectCommandBuffer: MTLIndirectCommandBuffer
    private let residencySet: MTLResidencySet
    private let resourceList: [MTLBuffer]

    private static func kernelURL() throws -> URL {
        try requiredBuildPath(".build/OakKernel.metallib",
                              description: "non-empty .build/OakKernel.metallib", nonEmpty: true)
    }

    private static func makePipelines(device: MTLDevice, planningIsDiscrete: Bool) throws -> [MTLComputePipelineState] {
        let library = try device.makeLibrary(URL: try kernelURL())
        return try phases.map { phase in
            let fc = MTLFunctionConstantValues()
            var discrete = planningIsDiscrete
            fc.setConstantValue(&discrete, type: .bool, index: 0)
            guard let fn = try? library.makeFunction(name: phase.name, constantValues: fc) else {
                throw OakError.message("Missing \(phase.name) in OakKernel.metallib.")
            }
            // supportIndirectCommandBuffers is required for ICB pre-encoding (Metal 3+).
            let desc = MTLComputePipelineDescriptor()
            desc.computeFunction = fn
            desc.supportIndirectCommandBuffers = true
            return try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
        }
    }

    // Apple GPUs have a fixed 32-thread SIMD width. Returning this constant avoids compiling all
    // pipelines just to read `threadExecutionWidth` from one of them.
    static func executionWidth(device: MTLDevice) throws -> Int { 32 }

    init(obsDim: Int, baseObsDim: Int, actDim: Int, batch: Int, rank: Int,
         actionIsDiscrete: Bool, actionLow: [Float], actionHigh: [Float],
         device: MTLDevice, seed: Int?, diagnosticGpuPhaseTiming: Bool = false) throws {
        precondition(obsDim == rank, "obsDim must equal rank (interface-derived latent width).")
        precondition(baseObsDim <= obsDim && baseObsDim >= 1, "baseObsDim must be positive and ≤ obsDim.")
        self.obsDim = obsDim; self.baseObsDim = baseObsDim; self.actDim = actDim
        self.batch = batch; self.rank = rank; self.actionIsDiscrete = actionIsDiscrete
        guard let queue = device.makeCommandQueue() else {
            throw OakError.message("Failed to create Metal command queue.")
        }
        self.queue = queue
        let d = MTLCommandBufferDescriptor(); d.retainedReferences = false
        cbDesc = d
        pipelines = try Self.makePipelines(device: device, planningIsDiscrete: actionIsDiscrete)

        let half = OakMetalLayout.discreteLogitHalfWidth(rank: rank, actDim: actDim)
        let lows  = actionIsDiscrete ? Array(repeating: -half, count: actDim) : actionLow
        let highs = actionIsDiscrete ? Array(repeating:  half, count: actDim) : actionHigh
        actionBoundsBuffer = device.makeBuffer(bytes: lows + highs,
                                               length: 2 * actDim * MemoryLayout<Float>.stride,
                                               options: [.storageModeShared])!
        scalarsBuffer = device.makeBuffer(length: MemoryLayout<KernelScalars>.stride, options: [.storageModeShared])!
        statsBuffer   = device.makeBuffer(length: MemoryLayout<KernelStats>.stride,   options: [.storageModeShared])!
        configBuffer  = device.makeBuffer(length: MemoryLayout<KernelConfig>.stride,  options: [.storageModeShared])!
        scalarsPointer = scalarsBuffer.contents().bindMemory(to: KernelScalars.self, capacity: 1)
        statsPointer   = statsBuffer.contents().bindMemory(to: KernelStats.self, capacity: 1)
        configPointer  = configBuffer.contents().bindMemory(to: KernelConfig.self, capacity: 1)
        scalarsPointer.initialize(to: KernelScalars())
        scalarsPointer.pointee.pEff = Float(rank)
        statsPointer.initialize(to: KernelStats())
        configPointer.initialize(to: KernelConfig())

        let actionDiscreteCount = actionIsDiscrete ? actDim : nil
        let paramLayout = OakMetalLayout.paramLayout(obsDim: obsDim, rank: rank, actDim: actDim,
                                                     actionDiscreteCount: actionDiscreteCount)
        stateLayout = OakMetalLayout.stateLayout(obsDim: obsDim, rank: rank, actDim: actDim,
                                                 batch: batch,
                                                 actionDiscreteCount: actionDiscreteCount)
        let actionParamCount = OakMetalLayout.actionParamCount(actDim: actDim,
                                                                actionDiscreteCount: actionDiscreteCount)
        self.actionParamCount = actionParamCount
        uintLayout  = OakMetalLayout.uintLayout(rank: rank, actDim: actDim, batch: batch)
        paramsArena = Shared(device: device, count: paramLayout.total, zero: Float16(0))
        stateArena  = Shared(device: device, count: stateLayout.total, zero: Float(0))
        uintArena   = Shared(device: device, count: uintLayout.total,  zero: UInt32(0))

        // Cache per-phase grid width so there's no allocation on the hot path.
        let K = OakMetalLayout.optionCount(rank: rank, actDim: actDim)
        let M = OakMetalLayout.concatDim(rank: rank, actDim: actDim)
        // meta_action: discrete = 1 thread/row (samples one index); continuous = 1 thread/(row·dim).
        let metaActionGrid = actionIsDiscrete ? batch : batch * actDim
        phaseGridWidths = [
            batch * rank,                     // oak_encode              [dispatchThreads]
            batch,                            // oak_predict             [dispatchThreads]
            batch,                            // oak_plan                [B TGs × 32]
            1,                                // oak_reduce_alpha        [1 TG × 128]
            rank,                             // oak_learn_value         [R TGs × 32]
            max(baseObsDim, 1),               // oak_learn_encoder       [dispatchThreads]
            K * M,                            // oak_learn_psi           [dispatchThreads]
            K * actionParamCount * rank,      // oak_learn_pi            [dispatchThreads]
            metaActionGrid,                   // oak_meta_action         [dispatchThreads]
            1,                                // oak_bk_reduce           [1 TG × 128]
            rank,                             // oak_bk_feature          [R TGs × 32]
            K,                                // oak_bk_option           [K TGs × 128]
            1,                                // oak_bk_admit_commit     [dispatchThreads]
        ]

        self.diagnosticGpuPhaseTiming = diagnosticGpuPhaseTiming
        phaseGpuTimeSumSeconds = Array(repeating: 0, count: Self.phases.count)
        phaseGpuInvalidStepCount = Array(repeating: 0, count: Self.phases.count)
        hostAction = Array(repeating: 0, count: batch * actDim)
        hostDiscrete = Array(repeating: 0, count: batch)
        envActionLow = lows
        envActionHigh = highs

        // --- Metal 4 ICB + MTLResidencySet setup ----------------------------------------------
        let icbDesc = MTLIndirectCommandBufferDescriptor()
        icbDesc.commandTypes = [.concurrentDispatch]
        icbDesc.inheritBuffers = false
        icbDesc.inheritPipelineState = false
        icbDesc.maxKernelBufferBindCount = 7
        guard let icb = device.makeIndirectCommandBuffer(descriptor: icbDesc,
                                                         maxCommandCount: Self.phases.count,
                                                         options: [.storageModeShared]) else {
            throw OakError.message("Failed to create Metal indirect command buffer.")
        }
        indirectCommandBuffer = icb
        let rsDesc = MTLResidencySetDescriptor()
        rsDesc.initialCapacity = 7
        residencySet = try device.makeResidencySet(descriptor: rsDesc)
        resourceList = [paramsArena.buffer, stateArena.buffer, uintArena.buffer,
                        scalarsBuffer, statsBuffer, actionBoundsBuffer, configBuffer]
        for buf in resourceList { residencySet.addAllocation(buf) }
        residencySet.commit()
        queue.addResidencySet(residencySet)

        // --- Parameter init: structural 1/√R scales on every linear head. Deterministic counter
        // hash (splitmix32 of seed ⊕ index) provides small centered probes without a host RNG
        // object allocated on the hot path.
        let invR: Float = 1.0 / sqrt(Float(max(rank, 1)))
        let seedBase = UInt32(bitPattern: Int32(truncatingIfNeeded: seed ?? Int(oakUnseededSeed))) &* 0x9e3779b9
        func sample(_ off: Int, _ count: Int, _ scale: Float) {
            let base = paramsArena.pointer.advanced(by: off)
            for i in 0..<count {
                var x = seedBase &+ UInt32(off &+ i) &* 0x7feb352d
                x = (x ^ (x &>> 16)) &* 0x846ca68b
                x ^= x &>> 16
                let u = Float(x >> 8) * 5.9604645e-8 * 2 - 1
                base[i] = Float16(u * scale)
            }
        }
        sample(paramLayout.encoder, rank, invR)
        sample(paramLayout.value, rank, invR)
        sample(paramLayout.psi, OakMetalLayout.psiCount(rank: rank, actDim: actDim), invR)
        sample(paramLayout.optPolicy,
               OakMetalLayout.optPolicyCount(rank: rank, actDim: actDim,
                                             actionDiscreteCount: actionDiscreteCount),
               invR)
        // ψ diagonal seed: cumulant k's W_ψ[k] · concat[s, λ] gets a +1/√R coefficient on its own
        // state feature index k (RRS feature-attainment initialization; Alberta Plan A.6 / OBB
        // §2.4). The first R weights of W_ψ[k] are state weights; index [k][k] is "predict
        // cumulant k mostly from feature k". No magic — invR is the structural per-feature scale.
        let psiPtr = paramsArena.pointer.advanced(by: paramLayout.psi)
        for k in 0..<K where k < rank {
            let idx = k * M + k
            psiPtr[idx] = Float16(Float(psiPtr[idx]) + invR)
        }

        // All features admitted on step 0.
        scalarsPointer.pointee.activeCount = UInt32(rank)
        let activeMask = uintArena.pointer.advanced(by: uintLayout.activeMask)
        let activeList = uintArena.pointer.advanced(by: uintLayout.activeList)
        for f in 0..<rank { activeMask[f] = 1; activeList[f] = UInt32(f) }
        // Initialize subtaskFeatureOf to identity sf[k] = k. bk_admit_commit re-ranks each step
        // by per-feature utility (Tier 7 RED dynamic top-K cumulants).
        let sfPtr = uintArena.pointer.advanced(by: uintLayout.subtaskFeatureOf)
        for k in 0..<K { sfPtr[k] = UInt32(min(k, rank - 1)) }

        writeKernelConfig(paramLayout: paramLayout)

        // Pre-encode the 12-phase dispatch graph into the ICB (grids are fixed for the engine's
        // lifetime; only KernelScalars/KernelStats/params/state data changes between steps).
        let buffers = resourceList
        for i in 0..<Self.phases.count {
            let cmd = icb.indirectComputeCommandAt(i)
            cmd.setComputePipelineState(pipelines[i])
            for j in 0..<6 { cmd.setKernelBuffer(buffers[j], offset: 0, at: j) }
            cmd.setKernelBuffer(configBuffer, offset: 0, at: OakMetalLayout.kernelConfigBufferIndex)
            let tgSize = Self.phases[i].tgSize
            if tgSize > 0 {
                let tgCount = max(phaseGridWidths[i], 1)
                cmd.concurrentDispatchThreadgroups(
                    MTLSize(width: tgCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            } else {
                let pip = pipelines[i]
                let tg = min(max(pip.threadExecutionWidth, 1), pip.maxTotalThreadsPerThreadgroup)
                let w = alignedGridWidth(threadCount: phaseGridWidths[i], pipeline: pip)
                cmd.concurrentDispatchThreads(
                    MTLSize(width: w, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(tg, w), height: 1, depth: 1))
            }
        }
    }

    deinit {
        queue.removeResidencySet(residencySet)
        residencySet.removeAllAllocations()
    }

    func metalPhaseGpuDiagnostics(measuredBatchSteps: Int) -> (meanMs: [String: Double]?, invalidPhases: [String]?) {
        guard diagnosticGpuPhaseTiming, measuredBatchSteps > 0 else { return (nil, nil) }
        var mean: [String: Double] = [:]; var invalid: [String] = []
        let d = Double(measuredBatchSteps)
        for (i, phase) in Self.phases.enumerated() {
            mean[phase.name] = phaseGpuTimeSumSeconds[i] / d * 1000
            if phaseGpuInvalidStepCount[i] > 0 { invalid.append(phase.name) }
        }
        return (mean, invalid.isEmpty ? nil : invalid)
    }

    // ---- Step dispatch -----------------------------------------------------------------------
    func step(obs: UnsafeBufferPointer<Float>, prevAction: [Float], prevReward: [Float],
              prevDone: [UInt8], accumulatePhaseGpu: Bool) throws -> (ActionBatch, Double, KernelStats) {
        let start = DispatchTime.now().uptimeNanoseconds

        precondition(obs.count == batch * obsDim)
        precondition(prevReward.count == batch)
        precondition(prevAction.count == batch * actDim)
        memcpy(stateArena.pointer.advanced(by: stateLayout.observation), obs.baseAddress!,
               batch * obsDim * MemoryLayout<Float>.stride)
        memcpy(stateArena.pointer.advanced(by: stateLayout.reward), prevReward,
               batch * MemoryLayout<Float>.stride)
        memcpy(stateArena.pointer.advanced(by: stateLayout.prevAction), prevAction,
               batch * actDim * MemoryLayout<Float>.stride)
        let donePtr = uintArena.pointer.advanced(by: uintLayout.done)
        for i in 0..<batch { donePtr[i] = UInt32(prevDone[i]) }

        let ctr = uintArena.pointer.advanced(by: uintLayout.counters)
        // Reset per-step non-finite counters (offsets OAK_CNT_OBS_NF, LEARN_NF, PLAN_NF).
        // counters[OAK_CNT_REINIT_OPT] is cleared by bk_admit_commit at end of step.
        // counters[OAK_CNT_REINIT_FEAT] is a cumulative life-of-engine count (not reset).
        ctr[0] = 0; ctr[1] = 0; ctr[2] = 0
        statsPointer.pointee = KernelStats()

        try runKernel(accumulatePhaseGpu: accumulatePhaseGpu)

        memcpy(&hostAction, stateArena.pointer.advanced(by: stateLayout.action),
               batch * actDim * MemoryLayout<Float>.stride)
        if actionIsDiscrete {
            for row in 0..<batch {
                var best = 0; var bestV: Float = -.greatestFiniteMagnitude
                let b = row * actDim
                for d in 0..<actDim { let v = hostAction[b + d]; if v > bestV { bestV = v; best = d } }
                hostDiscrete[row] = Int32(best)
            }
        } else {
            for row in 0..<batch {
                let b = row * actDim
                for d in 0..<actDim {
                    let v = hostAction[b + d]
                    if !v.isFinite { hostAction[b + d] = 0 }
                    else if v < envActionLow[d] { hostAction[b + d] = envActionLow[d] }
                    else if v > envActionHigh[d] { hostAction[b + d] = envActionHigh[d] }
                }
            }
        }
        return (ActionBatch(floats: hostAction, indices: actionIsDiscrete ? hostDiscrete : nil),
                Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000,
                statsPointer.pointee)
    }

    // Probe accessors (Tier 0; removed in Tier 12.3). Reads named arena slot for verification.
    var probeBatch: Int { batch }
    var probeRank: Int { rank }
    var probeActDim: Int { actDim }
    var probeOptionCount: Int { OakMetalLayout.optionCount(rank: rank, actDim: actDim) }

    func probeSlot(_ name: String) throws -> [Float] {
        let s = stateLayout
        let actionDiscreteCount = actionIsDiscrete ? actDim : nil
        let p = OakMetalLayout.paramLayout(obsDim: obsDim, rank: rank, actDim: actDim,
                                           actionDiscreteCount: actionDiscreteCount)
        let K = OakMetalLayout.optionCount(rank: rank, actDim: actDim)
        let D = actionParamCount
        // bk_admit_commit toggles stateSlot at end of each step; the just-written ("current")
        // state therefore lives at offset stateSlot, the previous slot at 1 - stateSlot.
        let curSlot = Int(scalarsPointer.pointee.stateSlot)
        let prevSlot = 1 - curSlot
        switch name {
        case "state":
            return Array(UnsafeBufferPointer(
                start: stateArena.pointer.advanced(by: s.state + curSlot * batch * rank),
                count: batch * rank))
        case "state_prev":
            return Array(UnsafeBufferPointer(
                start: stateArena.pointer.advanced(by: s.state + prevSlot * batch * rank),
                count: batch * rank))
        case "td":
            return Array(UnsafeBufferPointer(
                start: stateArena.pointer.advanced(by: s.td), count: batch))
        case "tdPsi":
            return Array(UnsafeBufferPointer(
                start: stateArena.pointer.advanced(by: s.tdPsi), count: batch * K))
        case "action":
            return Array(UnsafeBufferPointer(
                start: stateArena.pointer.advanced(by: s.action), count: batch * actDim))
        case "scoreGrad":
            return Array(UnsafeBufferPointer(
                start: stateArena.pointer.advanced(by: s.scoreGrad), count: batch * D))
        case "actionParamCount":
            return [Float(D)]
        case "valueW":
            return (0..<rank).map { Float(paramsArena.pointer[p.value + $0]) }
        case "encoderW":
            return (0..<rank).map { Float(paramsArena.pointer[p.encoder + $0]) }
        case "psi_total_count":
            return [Float(OakMetalLayout.psiCount(rank: rank, actDim: actDim))]
        case "policy_total_count":
            return [Float(OakMetalLayout.optPolicyCount(rank: rank, actDim: actDim,
                                                        actionDiscreteCount: actionDiscreteCount))]
        case "rho":
            return [scalarsPointer.pointee.rho]
        case "activeRank":
            return [statsPointer.pointee.activeRank]
        case "rewardRate":
            return [statsPointer.pointee.rewardRate]
        case "featMean":
            return Array(UnsafeBufferPointer(
                start: stateArena.pointer.advanced(by: s.featMean), count: rank))
        case "psi_bytes":
            // Total bytes of the ψ parameter block (fp16).
            return [Float(OakMetalLayout.psiCount(rank: rank, actDim: actDim) * 2)]
        case "concat_dim":
            return [Float(OakMetalLayout.concatDim(rank: rank, actDim: actDim))]
        case "subtaskFeatureOf":
            return (0..<K).map {
                Float(uintArena.pointer[uintLayout.subtaskFeatureOf + $0])
            }
        case "chord_slot":
            // Tier 3: chord = one-hot at active option index. Materialize on demand.
            let optBase = uintArena.pointer.advanced(by:
                uintLayout.option + Int(scalarsPointer.pointee.optionSlot) * batch)
            var out = [Float](repeating: 0, count: batch * K)
            for r in 0..<batch {
                let k = Int(min(optBase[r], UInt32(K - 1)))
                out[r * K + k] = 1
            }
            return out
        case "utilOpt":
            return Array(UnsafeBufferPointer(
                start: stateArena.pointer.advanced(by: s.utilOpt), count: K))
        case "utilFeat":
            return Array(UnsafeBufferPointer(
                start: stateArena.pointer.advanced(by: s.utilFeat), count: rank))
        default:
            throw OakError.message("Unknown probe slot: \(name)")
        }
    }

    private func writeKernelConfig(paramLayout: OakMetalLayout.Param) {
        let K = OakMetalLayout.optionCount(rank: rank, actDim: actDim)
        let cap = Float(max(rank * max(batch, 1), 1))
        let p = paramLayout, s = stateLayout, u = uintLayout
        var c = KernelConfig()
        c.obsDim = UInt32(obsDim); c.rank = UInt32(rank); c.actDim = UInt32(actDim)
        c.batch = UInt32(batch); c.optionCount = UInt32(K); c.baseObsDim = UInt32(baseObsDim)
        c.actionIsDiscrete = actionIsDiscrete ? 1 : 0
        c.actionParamCount = UInt32(actionParamCount)
        c.pEncoder = UInt32(p.encoder); c.pValue = UInt32(p.value)
        c.pPsi = UInt32(p.psi);         c.pOptPolicy = UInt32(p.optPolicy)
        c.sObservation = UInt32(s.observation); c.sObsMean = UInt32(s.obsMean)
        c.sState = UInt32(s.state); c.sPrevAction = UInt32(s.prevAction)
        c.sAction = UInt32(s.action); c.sScoreGrad = UInt32(s.scoreGrad)
        c.sReward = UInt32(s.reward); c.sTd = UInt32(s.td); c.sTdPsi = UInt32(s.tdPsi)
        c.uDone = UInt32(u.done); c.uActiveMask = UInt32(u.activeMask)
        c.uActiveList = UInt32(u.activeList); c.uOption = UInt32(u.option); c.uCounters = UInt32(u.counters)
        c.uSubtaskFeatureOf = UInt32(u.subtaskFeatureOf)
        c.aFeatSM = UInt32(s.featSM); c.aWBar = UInt32(s.wBar); c.aFeatMean = UInt32(s.featMean)
        c.aUtilFeat = UInt32(s.utilFeat); c.aUtilOpt = UInt32(s.utilOpt)
        c.cap = cap; c.alpha0 = 1.0 / cap; c.ridge = 1.0 / (cap + 1.0)
        let nLat = max(batch * rank, 1), nRew = max(batch, 1), nAct = max(batch * actDim, 1)
        c.boundLat = OakMetalLayout.statsAbsBoundForSumSquares(termCount: nLat)
        c.boundRew = OakMetalLayout.statsAbsBoundForSumSquares(termCount: nRew)
        c.boundAct = OakMetalLayout.statsAbsBoundForSumSquares(termCount: nAct)
        c.invBoundLat = 1.0 / c.boundLat; c.invBoundRew = 1.0 / c.boundRew; c.invBoundAct = 1.0 / c.boundAct
        configPointer.pointee = c
    }

    private func alignedGridWidth(threadCount: Int, pipeline: MTLComputePipelineState) -> Int {
        let align = max(Int(pipeline.threadExecutionWidth), 32)
        let n = max(threadCount, 1)
        return ((n + align - 1) / align) * align
    }

    private func encodePhase(into cb: MTLCommandBuffer, index: Int,
                             buffers: [MTLBuffer?], offsets: [Int]) throws {
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw OakError.message("Failed to create Metal compute encoder.")
        }
        let pip = pipelines[index]
        enc.setBuffers(buffers, offsets: offsets, range: 0..<6)
        enc.setBuffer(configBuffer, offset: 0, index: OakMetalLayout.kernelConfigBufferIndex)
        enc.setComputePipelineState(pip)
        let tgSize = Self.phases[index].tgSize
        if tgSize > 0 {
            // Fixed-TG reduction kernel; one TG per output element.
            let tgCount = max(phaseGridWidths[index], 1)
            enc.dispatchThreadgroups(MTLSize(width: tgCount, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        } else {
            let tg = min(max(pip.threadExecutionWidth, 1), pip.maxTotalThreadsPerThreadgroup)
            let w = alignedGridWidth(threadCount: phaseGridWidths[index], pipeline: pip)
            enc.dispatchThreads(MTLSize(width: w, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(tg, w), height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    private func runKernel(accumulatePhaseGpu: Bool) throws {
        let split = diagnosticGpuPhaseTiming && accumulatePhaseGpu

        // Hot path: execute the pre-encoded ICB in one compute encoder. No per-phase encoder
        // setup, no useResource calls (MTLResidencySet owns residency).
        if !split {
            guard let cb = queue.makeCommandBuffer(descriptor: cbDesc) else {
                throw OakError.message("Failed to create Metal command buffer.")
            }
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw OakError.message("Failed to create Metal compute encoder.")
            }
            enc.executeCommandsInBuffer(indirectCommandBuffer, range: 0..<pipelines.count)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            if let e = cb.error { throw OakError.message("Metal command failed: \(e)") }
            return
        }
        // Diagnostic path: per-phase command buffers to capture gpuStartTime/gpuEndTime.
        let buffers: [MTLBuffer?] = [paramsArena.buffer, stateArena.buffer, uintArena.buffer,
                                     scalarsBuffer, statsBuffer, actionBoundsBuffer]
        let offsets = [Int](repeating: 0, count: 6)
        for i in 0..<pipelines.count {
            guard let cb = queue.makeCommandBuffer(descriptor: cbDesc) else {
                throw OakError.message("Failed to create Metal command buffer.")
            }
            try encodePhase(into: cb, index: i, buffers: buffers, offsets: offsets)
            cb.commit(); cb.waitUntilCompleted()
            if let e = cb.error { throw OakError.message("Metal command failed: \(e)") }
            let s = cb.gpuStartTime, end = cb.gpuEndTime
            if s > 0, end >= s { phaseGpuTimeSumSeconds[i] += end - s }
            else { phaseGpuInvalidStepCount[i] += 1 }
        }
    }
}

// ================================================================================================
// Run-metrics persistence.
// ================================================================================================
private enum RunSnapshot {
    private static let persistLock = NSLock()

    private static func storageFromEnvironment() -> URL? {
        guard let raw = ProcessInfo.processInfo.environment["OAK_METRICS_STORAGE"]?
                            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty
        else { return nil }
        return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
    }

    fileprivate static func resolveMetricsDirectory(explicit: URL?) -> URL {
        if let e = explicit { return e }
        if let s = storageFromEnvironment() { return s }
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return root.appendingPathComponent(".oak-benchmarks/\(Main.oakBenchmarksRunId())", isDirectory: true)
    }

    static func persist(_ file: RunMetricsFile, metricsDir: URL) throws -> URL {
        let stem = snapshotStem(envId: file.envId, policy: file.policy, seed: file.seed)
        let leaf = file.heartbeatSequence.map { "\(stem).heartbeat-\(String(format: "%06d", $0)).json" }
                    ?? "\(stem).json"
        let seedDir = metricsDir
            .appendingPathComponent(file.envId, isDirectory: true)
            .appendingPathComponent(seedFolderName(file.seed), isDirectory: true)
        let url = seedDir.appendingPathComponent(leaf, isDirectory: false)
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
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
    fileprivate static func seedFolderName(_ seed: Int?) -> String { seed.map { "seed\($0)" } ?? "seedn" }
}

// ================================================================================================
// Sizing: directNumEnvs (S9) — structural memory budget, no bytesPerEnv heuristic.
// ================================================================================================
private func directNumEnvs(obsDim: Int, actDim: Int, actionDiscreteCount: Int?, device: MTLDevice) throws -> Int {
    let width = try OakMetalEngine.executionWidth(device: device)
    let rank = obsDim
    let K = OakMetalLayout.optionCount(rank: rank, actDim: actDim)
    let D = OakMetalLayout.actionParamCount(actDim: actDim, actionDiscreteCount: actionDiscreteCount)
    let perEnvFloats =  // state double-buffer (2R), obs (O), action+prevAction (2A), scoreGrad (D),
                        // tdPsi (K), reward (1), td (1).
        obsDim + 2 * rank + 2 * actDim + D + K + 2
    let perEnvUints = 1 + 2  // done + option slots
    let perEnvBytes = perEnvFloats * MemoryLayout<Float>.stride
                    + perEnvUints  * MemoryLayout<UInt32>.stride
    let paramBytes = OakMetalLayout.paramLayout(obsDim: obsDim, rank: rank, actDim: actDim,
                                                 actionDiscreteCount: actionDiscreteCount).total
                    * MemoryLayout<Float16>.stride
    // Fixed (per-engine, not per-env): featSM + wBar + featMean + utilFeat (4·R fp32) + utilOpt (K fp32).
    let fixedBytes = paramBytes + (4 * rank + K) * MemoryLayout<Float>.stride
    // Device-derived budget. Apple's `recommendedMaxWorkingSetSize` is the GPU's reported safe
    // working-set ceiling — structural to the hardware, not a magic divisor of system memory.
    let deviceBudget = Int(device.recommendedMaxWorkingSetSize)
    let budget = max(deviceBudget, width * perEnvBytes + fixedBytes)
    let memEnvs = max((budget - fixedBytes) / max(perEnvBytes, 1), width)
    // Optional user override via OAK_MAX_PARALLEL_ENVS; otherwise no synthetic ceiling — the
    // memory budget IS the cap.
    let userCap = ProcessInfo.processInfo.environment["OAK_MAX_PARALLEL_ENVS"].flatMap(Int.init)
    let ceiling = userCap.map { max($0, width) } ?? memEnvs
    func powerOfTwoLE(_ v: Int) -> Int { var p = 1; while p <= v / 2 { p *= 2 }; return max(p, 1) }
    return powerOfTwoLE(min(memEnvs, ceiling))
}

// ================================================================================================
// Run context and runner — a single-env/single-policy measurement loop, factored for clarity.
// ================================================================================================
private struct RunContext {
    let envId: String
    let policy: Policy
    let seed: Int?
    let warmup: Int
    let stopAfter: Int
    let numEnvs: Int
    let command: String
    let includeDiagnostic: Bool
    let metricsDir: URL
    let env: EnvPoolEnv
    let engine: OakMetalEngine?
}

private final class RunState {
    var prevReward: [Float]
    var prevDone: [UInt8]
    var prevAction: [Float]
    var randFloats: [Float]
    var randIdx: [Int32]
    var randSeed: UInt64
    var totalSteps = 0
    var measuredSteps = 0
    var measuredSeconds = 0.0
    var measurementStart: UInt64?
    var rewardSum = 0.0
    var doneCount = 0
    var agentMsSum = 0.0
    var envMsSum = 0.0
    var pulse = 0
    var health = HealthAccumulator()
    let actionCol: MeasuredActionCollector

    init(env: EnvPoolEnv, seed: Int?) {
        prevReward = Array(repeating: 0, count: env.numEnvs)
        prevDone = Array(repeating: 0, count: env.numEnvs)
        prevAction = Array(repeating: 0, count: env.actDim * env.numEnvs)
        randFloats = Array(repeating: 0, count: env.actDim * env.numEnvs)
        randIdx = Array(repeating: 0, count: env.numEnvs)
        randSeed = UInt64(bitPattern: Int64(seed ?? Int(oakUnseededSeed)))
        actionCol = MeasuredActionCollector(discreteCount: env.actionDiscreteCount, actDim: env.actDim)
    }
}

// ================================================================================================
// CLI main.
// ================================================================================================
@main struct Main {
    static func main() {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            switch args.first {
            case "benchmark": try benchmark()
            case "diagnose":  try diagnose(Array(args.dropFirst()))
            case "probe":     try probe(Array(args.dropFirst()))
            case "run":       try run(args.dropFirst().first)
            default:          try run(args.first)
            }
        } catch { fputs("\(error)\n", stderr); Foundation.exit(1) }
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
        let defaultN = try directNumEnvs(obsDim: probe.obsDim, actDim: probe.actDim,
                                         actionDiscreteCount: probe.actionDiscreteCount, device: device)
        let numEnvs = max(ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init) ?? defaultN, 1)
        let stopAfter = max(ProcessInfo.processInfo.environment["OAK_STOP_AFTER_STEPS"].flatMap(Int.init) ?? numEnvs, numEnvs)
        let baseline = max(ProcessInfo.processInfo.environment["OAK_BASELINE_RANDOM_STEPS"].flatMap(Int.init) ?? 0, 0)
        if policy == .oak, ProcessInfo.processInfo.environment["OAK_STOP_AFTER_STEPS"] == nil {
            fputs("OaK: OAK_STOP_AFTER_STEPS unset (short run). See AGENTS.md (Long runs).\n", stderr)
        }
        let dir = RunSnapshot.resolveMetricsDirectory(explicit: nil)
        let phases: [(Policy, Int)] = policy == .oak && baseline > 0
            ? [(.random, baseline), (policy, stopAfter)] : [(policy, stopAfter)]
        var metricsURL: URL!
        for (p, steps) in phases {
            (_, metricsURL) = try runSingle(envId: envId, policy: p, seed: seed, warmup: warmup,
                                             stopAfter: steps, numEnvs: numEnvs, device: device,
                                             command: "run", metricsDir: dir)
        }
        fputs("\(metricsURL.path)\n", stderr)
    }

    private static func diagnose(_ args: [String]) throws {
        var envId: String?, policy = Policy.oak, stopAfter: Int?, forcedN: Int?
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--policy":
                i += 1; guard i < args.count else { throw OakError.message("Missing value for --policy.") }
                policy = try parsePolicy(args[i])
            case "--steps":
                i += 1; guard i < args.count, let v = Int(args[i]), v > 0 else {
                    throw OakError.message("Invalid value for --steps.")
                }; stopAfter = v
            case "--num-envs":
                i += 1; guard i < args.count, let v = Int(args[i]), v > 0 else {
                    throw OakError.message("Invalid value for --num-envs.")
                }; forcedN = v
            default:
                guard envId == nil else { throw OakError.message("Unexpected argument '\(args[i])'.") }
                envId = args[i]
            }
            i += 1
        }
        let resolved = ProcessInfo.processInfo.environment["OAK_ENV"] ?? envId ?? ""
        guard !resolved.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OakError.message("Use `OaK diagnose <env-id>`.")
        }
        let seed = ProcessInfo.processInfo.environment["OAK_SEED"].flatMap(Int.init)
        let warmup = max(ProcessInfo.processInfo.environment["OAK_DIAG_WARMUP_STEPS"].flatMap(Int.init) ?? 0, 0)
        let device = try metalDevice()
        let probe = try EnvPoolEnv(id: resolved, numEnvs: 1, seed: seed)
        let defaultN = try directNumEnvs(obsDim: probe.obsDim, actDim: probe.actDim,
                                         actionDiscreteCount: probe.actionDiscreteCount, device: device)
        // diagnose uses the same memory-derived defaultN as run; OAK_NUM_ENVS overrides.
        let numEnvs = max(forcedN
                          ?? ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init)
                          ?? defaultN, 1)
        // OAK_DIAG_STEPS overrides; default = numEnvs (one batch). Caller adjusts per goal.
        let steps = max(stopAfter
                        ?? ProcessInfo.processInfo.environment["OAK_DIAG_STEPS"].flatMap(Int.init)
                        ?? numEnvs, numEnvs)
        let (_, metricsURL) = try runSingle(
            envId: resolved, policy: policy, seed: seed, warmup: warmup,
            stopAfter: steps,
            numEnvs: numEnvs, device: device, command: "diagnose", includeDiagnostic: true)
        fputs("\(metricsURL.path)\n", stderr)
    }

    // Tier 0 probe (removed in Tier 12.3): runs N steps on a synthetic deterministic env
    // (uniform fixed obs, constant reward) and dumps a named arena slot as JSON to stdout.
    // Math invariants are checked by external Python scripts; this is a one-shot diagnostic.
    private static func probe(_ args: [String]) throws {
        var slot = "state", steps = 10, rank = 32, actDim = 4, batch = 32
        var seed = Int(oakUnseededSeed), discrete = false
        var i = 0
        while i < args.count {
            let a = args[i]
            func eat(_ key: String) throws -> String {
                if a.hasPrefix(key + "=") { return String(a.dropFirst(key.count + 1)) }
                i += 1
                guard i < args.count else { throw OakError.message("Missing value for \(key).") }
                return args[i]
            }
            switch true {
            case a.hasPrefix("--slot"):     slot = try eat("--slot")
            case a.hasPrefix("--steps"):    steps = Int(try eat("--steps")) ?? steps
            case a.hasPrefix("--rank"):     rank = Int(try eat("--rank")) ?? rank
            case a.hasPrefix("--act-dim"):  actDim = Int(try eat("--act-dim")) ?? actDim
            case a.hasPrefix("--batch"):    batch = Int(try eat("--batch")) ?? batch
            case a.hasPrefix("--seed"):     seed = Int(try eat("--seed")) ?? seed
            case a == "--discrete":         discrete = true
            default: throw OakError.message("Unknown probe arg '\(a)'.")
            }
            i += 1
        }
        let device = try metalDevice()
        let lows  = Array(repeating: Float(-1), count: actDim)
        let highs = Array(repeating: Float( 1), count: actDim)
        let engine = try OakMetalEngine(
            obsDim: rank, baseObsDim: rank, actDim: actDim, batch: batch, rank: rank,
            actionIsDiscrete: discrete, actionLow: lows, actionHigh: highs,
            device: device, seed: seed)
        var obs = [Float](repeating: 0, count: batch * rank)
        var st = UInt32(bitPattern: Int32(seed)) &* 0x9e3779b9
        for n in 0..<obs.count {
            st &+= 0x9e3779b9
            var x = st; x = (x ^ (x &>> 16)) &* 0x85ebca6b
            x = (x ^ (x &>> 13)) &* 0xc2b2ae35; x ^= x &>> 16
            obs[n] = Float(x >> 8) * 5.9604645e-8 - 0.5
        }
        let zeroAction = [Float](repeating: 0, count: batch * actDim)
        let constReward = [Float](repeating: 0.1, count: batch)
        let zeroDone = [UInt8](repeating: 0, count: batch)
        var lastAction = zeroAction
        for _ in 0..<steps {
            let r = try obs.withUnsafeBufferPointer { p in
                try engine.step(obs: p, prevAction: lastAction, prevReward: constReward,
                                prevDone: zeroDone, accumulatePhaseGpu: false)
            }
            lastAction = r.0.floats
        }
        let values = try engine.probeSlot(slot)
        struct ProbeOut: Encodable {
            let slot: String, steps: Int, rank: Int, actDim: Int, batch: Int
            let optionCount: Int, count: Int, values: [Float]
        }
        let out = ProbeOut(slot: slot, steps: steps, rank: rank, actDim: actDim,
                           batch: batch, optionCount: engine.probeOptionCount,
                           count: values.count, values: values)
        let enc = JSONEncoder(); enc.outputFormatting = [.sortedKeys]
        let data = try enc.encode(out)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0a]))
    }

    private static func snapshotJsonCount(in metricsDir: URL) -> Int {
        let fm = FileManager.default
        func isDir(_ url: URL) -> Bool {
            var d: ObjCBool = false
            return fm.fileExists(atPath: url.path, isDirectory: &d) && d.boolValue
        }
        guard let envs = try? fm.contentsOfDirectory(at: metricsDir,
                includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
        else { return 0 }
        var count = 0
        for envDir in envs where isDir(envDir) {
            guard let seeds = try? fm.contentsOfDirectory(at: envDir,
                    includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
            else { continue }
            for seedDir in seeds where isDir(seedDir) {
                guard let files = try? fm.contentsOfDirectory(at: seedDir,
                        includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
                else { continue }
                count += files.filter { $0.pathExtension.lowercased() == "json"
                                     && !$0.lastPathComponent.contains(".heartbeat-") }.count
            }
        }
        return count
    }

    private static func resolveBenchmarkWorkingDirectory(root: URL, expectedTasks: Int) -> URL {
        if let raw = ProcessInfo.processInfo.environment["OAK_BENCH_DIR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
        }
        if let raw = ProcessInfo.processInfo.environment["OAK_METRICS_STORAGE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            let storage = URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
            if snapshotJsonCount(in: storage) < expectedTasks { return storage }
            return root.appendingPathComponent(".oak-benchmarks/\(oakBenchmarksRunId())", isDirectory: true)
        }
        let fresh = { root.appendingPathComponent(".oak-benchmarks/\(oakBenchmarksRunId())", isDirectory: true) }
        let box = root.appendingPathComponent(".oak-benchmarks", isDirectory: true)
        guard let runs = try? FileManager.default.contentsOfDirectory(at: box,
                includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return fresh() }
        let runIdPattern = try? NSRegularExpression(pattern: #"^\d{8}-\d{6}-\d+$"#)
        var best: URL?; var bestDate = Date.distantPast
        for r in runs {
            var d: ObjCBool = false
            guard FileManager.default.fileExists(atPath: r.path, isDirectory: &d), d.boolValue else { continue }
            let n = r.lastPathComponent
            let rng = NSRange(n.startIndex..<n.endIndex, in: n)
            guard runIdPattern?.firstMatch(in: n, options: [], range: rng) != nil else { continue }
            guard snapshotJsonCount(in: r) < expectedTasks else { continue }
            let dt = (try? r.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            if dt >= bestDate { bestDate = dt; best = r }
        }
        return best ?? fresh()
    }

    private static func benchmark() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let seedStart = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_START"].flatMap(Int.init) ?? 0
        let seedCount = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 10
        let warmup = max(ProcessInfo.processInfo.environment["OAK_BENCH_WARMUP_STEPS"].flatMap(Int.init) ?? 1000, 0)
        let stopAfter = max(ProcessInfo.processInfo.environment["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 50000, 1)
        let policies = try (ProcessInfo.processInfo.environment["OAK_BENCH_POLICIES"] ?? "oak,random")
            .split(separator: ",").map { try parsePolicy(String($0)) }
        let device = try metalDevice()
        let forcedN = ProcessInfo.processInfo.environment["OAK_BENCH_NUM_ENVS"].flatMap(Int.init)
                   ?? ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init)
        let envOrder = try benchmarkEnvOrder(seed: seedStart)
        let expected = envOrder.count * seedCount * policies.count
        let benchDir = resolveBenchmarkWorkingDirectory(root: root, expectedTasks: expected)
        try FileManager.default.createDirectory(at: benchDir, withIntermediateDirectories: true)
        for env in envOrder {
            let defaultN = try directNumEnvs(obsDim: env.schema.observationDim, actDim: env.schema.actionDim,
                                             actionDiscreteCount: env.schema.actionDiscreteCount, device: device)
            let numEnvs = max(forcedN ?? defaultN, 1)
            for seed in seedStart..<(seedStart + seedCount) {
                for policy in policies {
                    let stem = RunSnapshot.snapshotStem(envId: env.envId, policy: policy, seed: seed)
                    let snapURL = benchDir
                        .appendingPathComponent(env.envId, isDirectory: true)
                        .appendingPathComponent(RunSnapshot.seedFolderName(seed), isDirectory: true)
                        .appendingPathComponent("\(stem).json", isDirectory: false)
                    if FileManager.default.fileExists(atPath: snapURL.path) { continue }
                    let logURL = benchDir.appendingPathComponent("\(stem).log")
                    do {
                        _ = try runSingle(envId: env.envId, policy: policy, seed: seed, warmup: warmup,
                                           stopAfter: stopAfter, numEnvs: numEnvs, device: device,
                                           command: "benchmark", metricsDir: benchDir).0
                    } catch {
                        try "\(error)\n".data(using: .utf8)?.write(to: logURL)
                        throw error
                    }
                }
            }
        }
    }

    private static func parsePolicy(_ raw: String) throws -> Policy {
        guard let p = Policy(rawValue: raw.lowercased()) else {
            throw OakError.message("Unknown policy '\(raw)'.")
        }
        return p
    }

    private static func metalDevice() throws -> MTLDevice {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw OakError.message("Metal device unavailable.")
        }
        return d
    }

    private static func benchmarkEnvOrder(seed: Int) throws -> [BenchmarkEnv] {
        var buckets = [Schema: [BenchmarkEnv]]()
        for id in NativeBridge.shared.benchmarkEnvIds() {
            let probe = try EnvPoolEnv(id: id, numEnvs: 1, seed: seed)
            let schema = Schema(observationDim: probe.obsDim, actionDim: probe.actDim,
                                actionDiscreteCount: probe.actionDiscreteCount)
            buckets[schema, default: []].append(BenchmarkEnv(envId: id, schema: schema))
        }
        let ordered = buckets.keys.sorted()
        var groups = ordered.map { buckets[$0]!.sorted { $0.envId < $1.envId } }
        var out = [BenchmarkEnv]()
        while true {
            var any = false
            for i in groups.indices where !groups[i].isEmpty {
                out.append(groups[i].removeFirst()); any = true
            }
            if !any { return out }
        }
    }

    private static func randomAction(for env: EnvPoolEnv, numEnvs: Int, stepCount: inout UInt64,
                                     floats: inout [Float], indices: inout [Int32]) -> ActionBatch {
        // Deterministic counter-based RNG — no host RNG state, purely structural per Inv 9.
        func mix(_ x: UInt64) -> UInt64 {
            var z = x &+ 0x9e3779b97f4a7c15
            z = (z ^ (z &>> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z &>> 27)) &* 0x94d049bb133111eb
            return z ^ (z &>> 31)
        }
        for i in floats.indices { floats[i] = 0 }
        if let count = env.actionDiscreteCount {
            for row in 0..<numEnvs {
                stepCount &+= 1
                let choice = Int(mix(stepCount) % UInt64(count))
                indices[row] = Int32(choice)
                floats[row * env.actDim + choice] = 1
            }
            return ActionBatch(floats: floats, indices: indices)
        }
        for i in floats.indices {
            stepCount &+= 1
            let u = Float(mix(stepCount) >> 40) * (1.0 / Float(1 << 24))
            let dim = i % env.actDim
            floats[i] = env.actionLow[dim] + (env.actionHigh[dim] - env.actionLow[dim]) * u
        }
        return ActionBatch(floats: floats, indices: nil)
    }

    // --- single-run orchestration (setup → loop → snapshot) --------------------------------

    private static func setupRun(envId: String, policy: Policy, seed: Int?, warmup: Int,
                                 stopAfter: Int, numEnvs: Int, device: MTLDevice, command: String,
                                 includeDiagnostic: Bool, metricsDir: URL?) throws -> RunContext {
        let dir = RunSnapshot.resolveMetricsDirectory(explicit: metricsDir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        fputs("OaK metrics \(dir.path)\n", stderr); fflush(stderr)
        let env = try EnvPoolEnv(id: envId, numEnvs: numEnvs, seed: seed)
        let engine: OakMetalEngine? = policy == .oak ? try OakMetalEngine(
            obsDim: env.obsDim, baseObsDim: env.obsDim, actDim: env.actDim,
            batch: numEnvs, rank: env.obsDim,
            actionIsDiscrete: env.actionDiscreteCount != nil,
            actionLow: env.actionLow, actionHigh: env.actionHigh,
            device: device, seed: seed, diagnosticGpuPhaseTiming: includeDiagnostic) : nil
        try env.reset()
        return RunContext(envId: envId, policy: policy, seed: seed, warmup: warmup,
                          stopAfter: stopAfter, numEnvs: numEnvs, command: command,
                          includeDiagnostic: includeDiagnostic, metricsDir: dir,
                          env: env, engine: engine)
    }

    private static func emitSnapshot(_ ctx: RunContext, _ state: RunState, seq: Int?) -> RunMetricsFile {
        let u = max(state.measuredSteps / ctx.numEnvs, 1)
        let a = state.agentMsSum / 1000, e = state.envMsSum / 1000
        let h = max(state.measuredSeconds - a - e, 0)
        let merged = state.health.diagnosticSnapshot().map { b in
            let (mm, mi) = ctx.engine?.metalPhaseGpuDiagnostics(measuredBatchSteps: u) ?? (nil, nil)
            return RunDiagnostic(
                nonFiniteByPhase: b.nonFiniteByPhase,
                maxAbsLatentPrediction: b.maxAbsLatentPrediction,
                maxAbsPolicyOutput: b.maxAbsPolicyOutput,
                agentShare: state.measuredSeconds > 0 ? a / state.measuredSeconds : nil,
                envShare:   state.measuredSeconds > 0 ? e / state.measuredSeconds : nil,
                hostShare:  state.measuredSeconds > 0 ? h / state.measuredSeconds : nil,
                metalPhaseGpuMsMean: mm, metalPhaseGpuTimingInvalid: mi)
        }
        let diagOut: RunDiagnostic? = ((seq != nil && ctx.policy == .oak) || (seq == nil && ctx.includeDiagnostic))
                                       ? merged : nil
        let (dA, cA) = state.actionCol.partialMetrics(actDim: ctx.env.actDim)
        let sps = state.measuredSeconds > 0 ? Double(state.measuredSteps) / state.measuredSeconds : nil
        let reason = seq == nil ? "measured_steps" : "heartbeat"
        return RunMetricsFile(
            schemaVersion: 2, command: ctx.command, envId: ctx.envId, policy: ctx.policy, seed: ctx.seed,
            numEnvs: ctx.numEnvs, warmupSteps: ctx.warmup, stopAfterMeasuredSteps: ctx.stopAfter,
            observationDim: ctx.env.obsDim, actionDim: ctx.env.actDim,
            actionDiscreteCount: ctx.env.actionDiscreteCount,
            measuredStepCount: state.measuredSteps, measuredBatchCount: u,
            measuredSeconds: state.measuredSeconds, stepsPerSecond: sps,
            heartbeatSequence: seq, rewardSum: state.rewardSum, doneCount: state.doneCount,
            stopReason: reason, agentStepMsMean: state.agentMsSum / Double(u),
            envStepMsMean: state.envMsSum / Double(u),
            discreteAction: dA, continuousAction: cA,
            health: state.health.snapshot(), diagnostic: diagOut)
    }

    private static func measurementLoop(_ ctx: RunContext, _ state: RunState) throws {
        let defaultHb = ctx.command == "diagnose" ? 0 : max((ctx.stopAfter / 10 / ctx.numEnvs) * ctx.numEnvs, ctx.numEnvs)
        let hb = max(ProcessInfo.processInfo.environment["OAK_HEARTBEAT_STEPS"].flatMap(Int.init) ?? defaultHb, 0)
        while state.measuredSteps < ctx.stopAfter {
            let nextTotal = state.totalSteps + ctx.numEnvs
            let measured = nextTotal > ctx.warmup && state.totalSteps > 0
            if measured && state.measurementStart == nil {
                state.measurementStart = DispatchTime.now().uptimeNanoseconds
            }
            let actionBatch: ActionBatch, agentMs: Double, stats: KernelStats?
            if let engine = ctx.engine {
                let out = try ctx.env.observationScratch.withUnsafeBufferPointer { buf in
                    try engine.step(obs: buf, prevAction: state.prevAction, prevReward: state.prevReward,
                                    prevDone: state.prevDone, accumulatePhaseGpu: measured)
                }
                actionBatch = out.0; agentMs = out.1; stats = out.2
            } else {
                let t0 = DispatchTime.now().uptimeNanoseconds
                actionBatch = randomAction(for: ctx.env, numEnvs: ctx.numEnvs, stepCount: &state.randSeed,
                                           floats: &state.randFloats, indices: &state.randIdx)
                agentMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000
                stats = nil
            }
            let envStart = DispatchTime.now().uptimeNanoseconds
            try ctx.env.step(actionBatch)
            let envMs = Double(DispatchTime.now().uptimeNanoseconds - envStart) / 1_000_000
            state.totalSteps = nextTotal
            if measured {
                state.measuredSteps += ctx.numEnvs
                if let s = state.measurementStart {
                    state.measuredSeconds = Double(DispatchTime.now().uptimeNanoseconds - s) / 1_000_000_000
                }
                if hb > 0, (state.measuredSteps - ctx.numEnvs) / hb < state.measuredSteps / hb {
                    state.pulse += 1
                    _ = try RunSnapshot.persist(emitSnapshot(ctx, state, seq: state.pulse), metricsDir: ctx.metricsDir)
                }
                state.doneCount += ctx.env.doneScratch.reduce(into: 0) { $0 += Int($1) }
                for v in ctx.env.rewardScratch {
                    guard v.isFinite else { throw OakError.message("Environment produced a non-finite reward.") }
                    state.rewardSum += Double(v)
                }
                state.agentMsSum += agentMs; state.envMsSum += envMs
                if let s = stats { state.health.record(s) }
                state.actionCol.addMeasured(actionBatch, numEnvs: ctx.numEnvs, actDim: ctx.env.actDim)
            }
            memcpy(&state.prevReward, ctx.env.rewardScratch, ctx.numEnvs * MemoryLayout<Float>.stride)
            memcpy(&state.prevDone,   ctx.env.doneScratch,   ctx.numEnvs * MemoryLayout<UInt8>.stride)
            memcpy(&state.prevAction, actionBatch.floats, actionBatch.floats.count * MemoryLayout<Float>.stride)
        }
    }

    private static func runSingle(envId: String, policy: Policy, seed: Int?, warmup: Int,
                                  stopAfter: Int, numEnvs: Int, device: MTLDevice, command: String,
                                  includeDiagnostic: Bool = false, metricsDir: URL? = nil)
                                  throws -> (RunMetricsFile, URL) {
        let ctx = try setupRun(envId: envId, policy: policy, seed: seed, warmup: warmup,
                               stopAfter: stopAfter, numEnvs: numEnvs, device: device,
                               command: command, includeDiagnostic: includeDiagnostic,
                               metricsDir: metricsDir)
        let state = RunState(env: ctx.env, seed: seed)
        try measurementLoop(ctx, state)
        let file = emitSnapshot(ctx, state, seq: nil)
        let url = try RunSnapshot.persist(file, metricsDir: ctx.metricsDir)
        return (file, url)
    }

    private static let runIdFormatter: DateFormatter = {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .iso8601)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(secondsFromGMT: 0)
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f
    }()

    fileprivate static func oakBenchmarksRunId() -> String {
        runIdFormatter.string(from: Date()) + "-\(ProcessInfo.processInfo.processIdentifier)"
    }
}
