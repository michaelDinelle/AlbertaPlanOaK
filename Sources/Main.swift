import Foundation
import Metal
import Darwin

// =================================================================================================
// Orthodox Step 11 Oak - host-side dispatcher. See Sources/OakKernel.metal for the architectural
// citation block. This file owns: arena sizing, Metal engine, env-pool FFI, CLI commands
// (run/diagnose/benchmark), and run-metrics JSON emission.
//
// Single-stream commitment: ONE agent instance owns ONE env (numEnvs=1) and processes ONE
// observation per step. Concurrent benchmarking is achieved by running N independent agent
// instances in parallel via OperationQueue, never by batching B>1 into one engine.
// =================================================================================================

enum OakError: Error, CustomStringConvertible {
    case message(String)
    var description: String { if case let .message(m) = self { return m }; return "" }
}

enum Policy: String, Codable { case oak, random }

// =================================================================================================
// Run metrics JSON (schemaVersion 3).
// =================================================================================================
struct RunHealth: Codable {
    // Host-computed (works for ANY policy, including random):
    let rewardRate: Double?
    let actionDeltaL2Mean: Double?
    let scoreGradL2Mean: Double?
    // Kernel-side (only present for oak):
    let oakOptionReinitCountFinal: Int?, oakFeatureReinitCountFinal: Int?
    let nonFiniteByPhase: [String: Int]?
    // Latest chord effective-K = 1/Σ chord². Range [1, K]; K means chord is uniform,
    // 1 means one option carries the chord.
    let chordEffectiveK: Double?
    let rrBonusMean: Double?
    let psiStopGapMean: Double?
    let terminationRate: Double?
    let optionDurationMean: Double?
    let neverUsedFeatureCount: Double?
    let replacementAgeMean: Double?
    let stateEffectiveRank: Double?
}

struct RunDiagnostic: Codable {
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
    let numEnvs: Int, stopAfterMeasuredSteps: Int
    let observationDim: Int, actionDim: Int, actionDiscreteCount: Int?
    let measuredStepCount: Int, measuredBatchCount: Int, measuredSeconds: Double
    let workerDepth: Int                // = measuredStepCount (single-stream).
    let stepsPerSecond: Double?, heartbeatSequence: Int?
    let rewardSum: Double, doneCount: Int, stopReason: String
    let agentStepMsMean: Double, envStepMsMean: Double
    let discreteAction: DiscreteActionMetrics?, continuousAction: ContinuousActionMetrics?
    let health: RunHealth?, diagnostic: RunDiagnostic?
}

// =================================================================================================
// Action sampling - the SOLE site that knows action-space type (Inv-1 boundary). Agent emits
// policyMean[D]; wrapper composes per-component sampling and matching score function.
// =================================================================================================
struct PolicyMeanBatch { let values: [Float]; let dPolicy: Int }

enum ActionComponent {
    case discrete(n: Int)
    case box(n: Int, low: [Float], high: [Float])

    var paramCount: Int {
        switch self {
        case .discrete(let n): return n
        case .box(let n, _, _): return 2 * n
        }
    }
    var envActionWidth: Int {
        switch self {
        case .discrete: return 1
        case .box(let n, _, _): return n
        }
    }
}

extension Sequence where Element == ActionComponent {
    var totalPolicyParamCount: Int { reduce(0) { $0 + $1.paramCount } }
    var totalEnvActionWidth: Int   { reduce(0) { $0 + $1.envActionWidth } }
}

// Wrapper-side action sampling. Discrete: Gumbel-max with score 1{a=d}-π_d.
// Box: Gaussian with μ and log-σ policy coordinates; mean score uses Williams variance scaling.
// The wrapper also emits the sampled action in D_policy coordinates for the agent's experience stream.
func sampleAndScoreActions(spec: [ActionComponent], numEnvs: Int, dPolicy: Int,
                           envActionWidth: Int,
                           policyMean: UnsafePointer<Float>,
                           envAction: UnsafeMutablePointer<Float>,
                           actionSignal: UnsafeMutablePointer<Float>,
                           scoreGrad: UnsafeMutablePointer<Float>,
                           rngState: inout UInt64) {
    func nextU01(_ s: inout UInt64) -> Float {
        s &+= 0x9e3779b97f4a7c15
        var z = s
        z = (z ^ (z &>> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z &>> 27)) &* 0x94d049bb133111eb
        z ^= z &>> 31
        return Float(z >> 40) * (1.0 / Float(1 << 24))
    }
    func nextNormal(_ s: inout UInt64) -> Float {
        let u1 = max(nextU01(&s), 1e-7)
        let u2 = nextU01(&s)
        return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
    }
    for r in 0..<numEnvs {
        let mBase = r * dPolicy
        let aBase = r * envActionWidth
        var pIdx = 0, aIdx = 0
        for component in spec {
            switch component {
            case .discrete(let n):
                var maxLogit = -Float.greatestFiniteMagnitude
                for d in 0..<n { maxLogit = max(maxLogit, policyMean[mBase + pIdx + d]) }
                var bestIdx = 0
                var bestKey = -Float.greatestFiniteMagnitude
                for d in 0..<n {
                    let l = policyMean[mBase + pIdx + d]
                    let u = max(nextU01(&rngState), 1e-7)
                    let g = -log(-log(u))
                    let key = l + g
                    if key > bestKey { bestKey = key; bestIdx = d }
                }
                var Z: Float = 0
                for d in 0..<n { Z += exp(policyMean[mBase + pIdx + d] - maxLogit) }
                let invZ: Float = 1.0 / max(Z, .leastNormalMagnitude)
                for d in 0..<n {
                    let pi_d = exp(policyMean[mBase + pIdx + d] - maxLogit) * invZ
                    scoreGrad[mBase + pIdx + d] = ((d == bestIdx) ? 1.0 : 0.0) - pi_d
                    actionSignal[mBase + pIdx + d] = (d == bestIdx) ? 1.0 : 0.0
                }
                envAction[aBase + aIdx] = Float(bestIdx)
                pIdx += n; aIdx += 1
            case .box(let a, let low, let high):
                // Continuous actor-critic policy (Williams 1992; Degris-Pilarski-Sutton 2012):
                // μ and log-σ are separate policy outputs. σ starts at one policy-coordinate
                // unit, 1/D_policy, then the actor can adapt it.
                let sigmaBase: Float = 1.0 / Float(max(dPolicy, 1))
                let scaleBase = pIdx + a
                for d in 0..<a {
                    let z = nextNormal(&rngState)
                    let mu = policyMean[mBase + pIdx + d]
                    let rawLogSigma = policyMean[mBase + scaleBase + d]
                    let logSigma = rawLogSigma.isFinite ? rawLogSigma : 0
                    let sigmaRaw = sigmaBase * exp(logSigma)
                    let sigma = (sigmaRaw.isFinite && sigmaRaw > 0) ? sigmaRaw : sigmaBase
                    let raw = mu + sigma * z
                    let clamped = max(low[d], min(high[d], raw.isFinite ? raw : 0))
                    envAction[aBase + aIdx + d] = clamped
                    actionSignal[mBase + pIdx + d] = clamped
                    actionSignal[mBase + scaleBase + d] = 0
                    // Env-side clamp affects the env's view only. Williams/Degris AC-S scales
                    // the Gaussian mean score by σ², so (z/σ)·σ² = z·σ.
                    scoreGrad[mBase + pIdx + d] = z * sigma
                    scoreGrad[mBase + scaleBase + d] = z * z - 1.0
                }
                pIdx += 2 * a; aIdx += a
            }
        }
    }
}

private struct Schema: Hashable, Comparable {
    let observationDim: Int, actionDim: Int, actionDiscreteCount: Int?
    static func < (lhs: Schema, rhs: Schema) -> Bool {
        (lhs.observationDim, lhs.actionDim, lhs.actionDiscreteCount ?? -1)
            < (rhs.observationDim, rhs.actionDim, rhs.actionDiscreteCount ?? -1)
    }
}

private struct BenchmarkEnv { let envId: String; let schema: Schema }

let oakUnseededSeed: Int32 = 0

func metalDevice() throws -> MTLDevice {
    guard let d = MTLCreateSystemDefaultDevice() else {
        throw OakError.message("Metal device unavailable.")
    }
    return d
}

// =================================================================================================
// Filesystem helpers.
// =================================================================================================
private func oakRoots() -> [URL] {
    let fm = FileManager.default
    let exe = URL(fileURLWithPath: CommandLine.arguments[0],
                  relativeTo: URL(fileURLWithPath: fm.currentDirectoryPath)).standardizedFileURL
    return [
        fm.currentDirectoryPath,
        exe.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().path,
    ].map { URL(fileURLWithPath: $0).standardizedFileURL }
}

func requiredBuildPath(_ path: String, description: String, nonEmpty: Bool = false) throws -> URL {
    guard let url = oakRoots().map({ $0.appendingPathComponent(path) }).first(where: {
        guard FileManager.default.fileExists(atPath: $0.path) else { return false }
        return !nonEmpty || ((try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 0
    }) else { throw OakError.message("Missing \(description). Run ./setup.sh first.") }
    return url
}

func decodeCString(_ buffer: [CChar]) -> String {
    String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

// =================================================================================================
// HealthAccumulator - fixed-size struct fold (Inv-6: no per-step heap allocation).
//
// recordEnvSignals: host-computable from env-wrapper signals; populated for ANY policy.
//   Tracks rho-bar (Reward Centering 2405.09999, incremental EMA), action delta L2, score grad L2.
// recordKernelStats: kernel-internal counters; only OakMetalEngine populates these.
//   Per-phase non-finite counts + cumulative reinit counts (Plan Step 11 evidence).
// =================================================================================================
private struct HealthAccumulator {
    private var envSteps = 0, kernelSteps = 0
    private var rhoBar = 0.0
    private var actDeltaSum = 0.0, scoreGradSum = 0.0
    private var encodeNF = 0, learnNF = 0, planNF = 0, tdPsiNF = 0
    private var oakOptionReinitFinal = 0, oakFeatureReinitFinal = 0
    private var chordEffectiveKLatest: Double = 0
    private var rrBonusMeanLatest = 0.0, psiStopGapMeanLatest = 0.0
    private var terminationRateLatest = 0.0, optionDurationMeanLatest = 0.0
    private var neverUsedFeatureCountLatest = 0.0, replacementAgeMeanLatest = 0.0
    private var stateEffectiveRankLatest = 0.0
    private let trackFloor: Double

    init(obsDim: Int) { trackFloor = 1.0 / Double(max(obsDim, 1)) }

    mutating func recordEnvSignals(reward: Float,
                                   prevPolicyMean: UnsafeBufferPointer<Float>,
                                   currPolicyMean: UnsafeBufferPointer<Float>,
                                   scoreGrad: UnsafeBufferPointer<Float>) {
        envSteps += 1
        let r = Double(reward)
        if r.isFinite {
            let invT = max(1.0 / Double(envSteps), trackFloor)
            rhoBar += invT * (r - rhoBar)
        }
        var dl = 0.0, sg = 0.0
        for d in 0..<currPolicyMean.count {
            let delta = currPolicyMean[d] - prevPolicyMean[d]
            dl += Double(delta) * Double(delta)
        }
        for d in 0..<scoreGrad.count {
            sg += Double(scoreGrad[d]) * Double(scoreGrad[d])
        }
        actDeltaSum  += dl / Double(max(currPolicyMean.count, 1))
        scoreGradSum += sg / Double(max(scoreGrad.count, 1))
    }

    mutating func recordKernelStats(_ stats: KernelStats) {
        kernelSteps += 1
        encodeNF += Int(stats.encodeNonFiniteCount)
        learnNF  += Int(stats.learnNonFiniteCount)
        planNF   += Int(stats.planNonFiniteCount)
        tdPsiNF  += Int(stats.tdPsiNonFiniteCount)
        oakOptionReinitFinal  = Int(stats.oakOptionReinitCount)
        oakFeatureReinitFinal = Int(stats.oakFeatureReinitCount)
        chordEffectiveKLatest = Double(stats.chordEffectiveK)
        rrBonusMeanLatest = Double(stats.rrBonusMean)
        psiStopGapMeanLatest = Double(stats.psiStopGapMean)
        terminationRateLatest = Double(stats.terminationRate)
        optionDurationMeanLatest = Double(stats.optionDurationMean)
        neverUsedFeatureCountLatest = Double(stats.neverUsedFeatureCount)
        replacementAgeMeanLatest = Double(stats.replacementAgeMean)
        stateEffectiveRankLatest = Double(stats.stateEffectiveRank)
    }

    var hasMeasuredSteps: Bool { envSteps > 0 }

    func snapshot() -> RunHealth? {
        guard envSteps > 0 else { return nil }
        let n = Double(envSteps)
        let nf: [String: Int]? = kernelSteps > 0
            ? ["encode": encodeNF, "learn": learnNF, "plan": planNF, "tdPsi": tdPsiNF]
            : nil
        return RunHealth(
            rewardRate: rhoBar,
            actionDeltaL2Mean: actDeltaSum / n,
            scoreGradL2Mean: scoreGradSum / n,
            oakOptionReinitCountFinal: kernelSteps > 0 ? oakOptionReinitFinal : nil,
            oakFeatureReinitCountFinal: kernelSteps > 0 ? oakFeatureReinitFinal : nil,
            nonFiniteByPhase: nf,
            chordEffectiveK: kernelSteps > 0 ? chordEffectiveKLatest : nil,
            rrBonusMean: kernelSteps > 0 ? rrBonusMeanLatest : nil,
            psiStopGapMean: kernelSteps > 0 ? psiStopGapMeanLatest : nil,
            terminationRate: kernelSteps > 0 ? terminationRateLatest : nil,
            optionDurationMean: kernelSteps > 0 ? optionDurationMeanLatest : nil,
            neverUsedFeatureCount: kernelSteps > 0 ? neverUsedFeatureCountLatest : nil,
            replacementAgeMean: kernelSteps > 0 ? replacementAgeMeanLatest : nil,
            stateEffectiveRank: kernelSteps > 0 ? stateEffectiveRankLatest : nil)
    }
}

// =================================================================================================
// Measured action collector (single-stream).
// =================================================================================================
private final class MeasuredActionCollector {
    private var discrete: [Int]?
    private var sum: [Double]?, sumsq: [Double]?
    private var l2s = 0.0, l2sq = 0.0, n = 0
    private let envActionWidth: Int

    init(discreteCount: Int?, envActionWidth: Int) {
        self.envActionWidth = envActionWidth
        if let c = discreteCount { discrete = [Int](repeating: 0, count: c) }
        else { sum = [Double](repeating: 0, count: envActionWidth)
               sumsq = [Double](repeating: 0, count: envActionWidth) }
    }

    func addMeasured(envAction: [Float]) {
        if var h = discrete {
            let idx = max(0, min(h.count - 1, Int(envAction[0])))
            h[idx] &+= 1
            discrete = h
            return
        }
        guard var s = sum, var q = sumsq else { return }
        var l = 0.0
        for d in 0..<envActionWidth { let v = Double(envAction[d]); s[d] += v; q[d] += v * v; l += v * v }
        let r = sqrt(l); l2s += r; l2sq += r * r; n += 1
        sum = s; sumsq = q
    }

    func partialMetrics() -> (DiscreteActionMetrics?, ContinuousActionMetrics?) {
        let d = discrete.map { DiscreteActionMetrics(histogram: $0, uniqueCount: $0.filter { $0 > 0 }.count) }
        guard let s = sum, let q = sumsq, n > 0 else { return (d, nil) }
        let nn = Double(n)
        let mean = (0..<envActionWidth).map { s[$0] / nn }
        let std  = (0..<envActionWidth).map { let m = mean[$0]; return sqrt(max(0, q[$0] / nn - m * m)) }
        let ml = l2s / nn
        return (d, ContinuousActionMetrics(meanPerDim: mean, stdPerDim: std, meanL2: ml,
                                           stdL2: sqrt(max(0, l2sq / nn - ml * ml))))
    }
}

// =================================================================================================
// Kernel structs - mirror OakKernel.metal byte-for-byte. Keep in lockstep when editing either.
// =================================================================================================
private struct KernelConfig {
    var obsDim: UInt32 = 0, rank: UInt32 = 0
    var optionCount: UInt32 = 0, baseObsDim: UInt32 = 0
    var actionParamCount: UInt32 = 0
    // Non-instantaneous trace count M−1 where M=floor(log2(baseO))+1.
    // Encoder width = M·baseO + D_policy + 1 for obs traces plus action/reward.
    var baseObsTraceLevels: UInt32 = 0
    var pEncoder: UInt32 = 0, pPhase: UInt32 = 0, pValue: UInt32 = 0, pPsi: UInt32 = 0
    var pOptPolicy: UInt32 = 0, pOptReward: UInt32 = 0, pOptTransition: UInt32 = 0
    var pOptDuration: UInt32 = 0
    var sObservation: UInt32 = 0, sObsMean: UInt32 = 0, sObsSM: UInt32 = 0
    var sState: UInt32 = 0
    var sPolicyMean: UInt32 = 0, sActionSignal: UInt32 = 0, sScoreGrad: UInt32 = 0
    var sReward: UInt32 = 0, sTd: UInt32 = 0, sTdPsi: UInt32 = 0
    var sChord: UInt32 = 0, sChordPrev: UInt32 = 0
    var sUtilFeat: UInt32 = 0, sUtilOpt: UInt32 = 0
    var sBonusW: UInt32 = 0
    var sPlanAdv: UInt32 = 0
    var sAlpha: UInt32 = 0, sBeta: UInt32 = 0, sH: UInt32 = 0
    var sNTdMat: UInt32 = 0
    var sOptRewardTd: UInt32 = 0
    var sDurationTd: UInt32 = 0
    var sDurationHat: UInt32 = 0
    var sQOption: UInt32 = 0     // Per-option Q^k cache (K floats), oak_plan → oak_plan_reduce.
    var sObsTraceMulti: UInt32 = 0  // Tier I1: (M−1)·baseO multi-timescale obs traces.
    var sPrioOpt: UInt32 = 0
    var sContinuation: UInt32 = 0 // RR-derived per-option continuation indicator/probability.
    var sRhoOpt: UInt32 = 0
    var sReplacementAgeMean: UInt32 = 0
    var uCounters: UInt32 = 0
    var uOptFeature: UInt32 = 0
    var uFeatureAge: UInt32 = 0
    var uOptAge: UInt32 = 0
    var uFeatureEverUsed: UInt32 = 0
}

private struct KernelScalars {
    var stepCount: UInt32 = 0
    var stateSlot: UInt32 = 0
    var rho: Float = 0
    var alphaPi: Float = 0
    // Tier D2 (Plan Step 9, prioritized sweeping): mean of prioOpt[K], computed once per step
    // in oak_plan_reduce via simd_sum and broadcast-cached for every oak_learn_heads Head 4
    // thread to read (avoids every Head 4 thread computing the same K-element reduction).
    var meanPrio: Float = 0
}

struct KernelStats {
    // Only telemetry the host literally cannot derive from env signals.
    var encodeNonFiniteCount: Float = 0, learnNonFiniteCount: Float = 0
    var planNonFiniteCount: Float = 0, tdPsiNonFiniteCount: Float = 0
    var oakOptionReinitCount: Float = 0, oakFeatureReinitCount: Float = 0
    // Chord effective-K = 1 / Σ_k chord[k]². Range [1, K]. K = uniform chord;
    // 1 = one-hot; intermediate = moderate concentration.
    var chordEffectiveK: Float = 0
    var rrBonusMean: Float = 0, psiStopGapMean: Float = 0
    var terminationRate: Float = 0, optionDurationMean: Float = 0
    var neverUsedFeatureCount: Float = 0, replacementAgeMean: Float = 0
    var stateEffectiveRank: Float = 0
}

// =================================================================================================
// Arena layout - single source of truth for offsets. Single-stream: no `batch` parameter.
// =================================================================================================
private enum OakMetalLayout {
    static let kernelConfigBufferIndex = 5
    static let maxOptionCount = 32

    static func optionCount(rank: Int) -> Int { max(1, min(rank, maxOptionCount)) }
    static func psiCount(rank: Int) -> Int { optionCount(rank: rank) * rank }
    static func optPolicyCount(rank: Int, dPolicy: Int) -> Int {
        optionCount(rank: rank) * dPolicy * rank
    }
    static func optRewardCount(rank: Int) -> Int { optionCount(rank: rank) * rank }
    static func optDurationCount(rank: Int) -> Int { optionCount(rank: rank) * rank }
    // DENSE transition (RR-Subtasks Sutton-Machado-Holland-Timbers-Tanner-White 2022 eq. 17):
    // one weight per (option, j_out, i_in). n_hat^k_j(s) = sum_i w^k_{j,i} * s[i] gives options
    // real cross-feature dynamics for planning (V(n_hat^k(s)) = sum_j valueW[j] * sum_i w^k_{j,i} * s[i]).
    // K*R*R weights; the K*R^2 inner loop in oak_plan is absorbed by K-thread parallel +
    // simd_max reduction. Metal caps K to the hardware SIMD width.
    static func optTransitionCount(rank: Int) -> Int { optionCount(rank: rank) * rank * rank }
    static func postLearnThreadCount(rank: Int, dPolicy: Int) -> Int {
        let K = optionCount(rank: rank)
        let actBase = ((rank + K + 31) / 32) * 32
        return actBase + dPolicy * maxOptionCount
    }
    // Deep Fourier features (Lewandowski-Schuurmans-Machado 2024, arXiv:2410.20634): R/2
    // pre-activations, each producing a paired (cos, sin) channel into the R-wide state.
    // Encoder rows match the pre-activation count; phase is one scalar per pre-activation.
    //
    // Tier I1 (Plan Step 3 partial — "incorporation of older signals and traces"): each
    // pre-activation reads M concatenated input regions:
    //   region m=0: `std_obs[o]` (instantaneous, baseO dims)
    //   region m=1..M−1: `obsTraceMulti[m·baseO + o]` at decay 1/τ_m, τ_m = 2^m
    // Total encoder row width = M·baseO + D_policy + 1. M = floor(log2(baseO)) + 1
    // (structural; τ_max ≈ baseO). The extra D+1 signals are previous action descriptor
    // and previous reward.
    //
    // Tier C3 (recurrent self-feedback in the encoder) was attempted in two forms — RFF-1
    // across all three regions (v1) and variance-balanced W_state at sqrt(baseO/R) (v2) —
    // and both regressed by ~52% on the 10×1.5M paired bench. Diagnosis: recurrent dynamics
    // create double non-stationarity for the TD bootstrap (state distribution shifts both
    // from policy change AND from recurrent evolution). ESN/LSM works because the readout
    // is supervised against a stationary target; OaK's TD heads can't track both. Future
    // Tier I2 reservoir layer sidesteps this by FIXING the recurrent weights (no gradient
    // flow through r); encoder still trains via G&T as currently.
    static func fourierPreActCount(rank: Int) -> Int { rank / 2 }
    // Tier I1: M = floor(log2(baseO)) + 1, M ≥ 1. Number of trace levels including the
    // instantaneous std_obs at τ=1. (M−1) is the count of NON-INSTANTANEOUS levels stored
    // in obsTraceMulti.
    static func obsTraceLevelCount(baseObsDim: Int) -> Int {
        let b = max(baseObsDim, 1)
        var m = 0
        var v = b
        while v > 1 { v >>= 1; m += 1 }     // floor(log2(b))
        return m + 1                         // +1 for the instantaneous level
    }
    static func encoderInputWidth(baseObsDim: Int, dPolicy: Int) -> Int {
        obsTraceLevelCount(baseObsDim: baseObsDim) * max(baseObsDim, 1) + max(dPolicy, 0) + 1
    }
    static func encoderCount(obsDim: Int, baseObsDim: Int, rank: Int, dPolicy: Int) -> Int {
        fourierPreActCount(rank: rank) * encoderInputWidth(baseObsDim: baseObsDim, dPolicy: dPolicy)
    }
    static func phaseCount(rank: Int) -> Int { fourierPreActCount(rank: rank) }

    struct Param {
        let encoder, phase, value, psi, optPolicy, optReward, optTransition, optDuration, total: Int
    }
    static func paramLayout(obsDim: Int, baseObsDim: Int, rank: Int, dPolicy: Int) -> Param {
        var o = 0
        let enc = o; o += encoderCount(obsDim: obsDim, baseObsDim: baseObsDim, rank: rank, dPolicy: dPolicy)
        let ph  = o; o += phaseCount(rank: rank)
        let val = o; o += rank
        let psi = o; o += psiCount(rank: rank)
        let op  = o; o += optPolicyCount(rank: rank, dPolicy: dPolicy)
        let or_ = o; o += optRewardCount(rank: rank)
        let ot  = o; o += optTransitionCount(rank: rank)
        let od  = o; o += optDurationCount(rank: rank)
        return Param(encoder: enc, phase: ph, value: val, psi: psi, optPolicy: op,
                     optReward: or_, optTransition: ot, optDuration: od, total: o)
    }

    struct State {
        let observation, obsMean, obsSM, state: Int
        let policyMean, actionSignal, scoreGrad: Int
        let reward, td, tdPsi: Int
        let chord, chordPrev: Int
        let utilFeat, utilOpt: Int
        let bonusW: Int
        let planAdv: Int
        let alpha, beta, h: Int       // MetaOptimize L-approximation buffers (R floats each).
        let nTdMat: Int               // Dense option-transition TD targets, K*R floats, written by oak_plan.
        let optRewardTd: Int          // Option reward-model TD targets, K floats, written once by oak_plan.
        let durationTd, durationHat: Int
        let qOption: Int              // Per-option Q^k cache, K floats, oak_plan → oak_plan_reduce.
        let obsTraceMulti: Int        // Tier I1: (M−1)·baseO multi-timescale obs traces.
        let prioOpt: Int              // Tier D2: per-option priority for plan_reduce / Head 4.
        let continuation: Int          // RR-derived continuation, K floats.
        let rhoOpt: Int
        let replacementAgeMean: Int
        let total: Int
    }
    static func stateLayout(obsDim: Int, baseObsDim: Int, rank: Int, dPolicy: Int) -> State {
        let K = optionCount(rank: rank)
        let M = obsTraceLevelCount(baseObsDim: baseObsDim)
        var o = 0
        let obs = o; o += obsDim
        let mean = o; o += obsDim
        let osm  = o; o += obsDim
        let st = o; o += 2 * rank
        let pol = o; o += dPolicy
        let act = o; o += dPolicy
        let sg  = o; o += dPolicy
        let rew = o; o += 1
        let td  = o; o += 1
        let tdP = o; o += K
        let cs  = o; o += K
        let csP = o; o += K
        let uf  = o; o += rank
        let uo  = o; o += K
        let bw  = o; o += rank
        let pa  = o; o += 1
        let al  = o; o += rank
        let be  = o; o += rank
        let h   = o; o += rank
        let ntd = o; o += K * rank
        let ortd = o; o += K
        let dtd = o; o += K
        let dhat = o; o += K
        let qo  = o; o += K
        // Tier I1: (M−1)·baseO non-instantaneous trace levels, τ_m = 2^m.
        let otr = o; o += max(M - 1, 0) * max(baseObsDim, 1)
        let prio = o; o += K
        let cont = o; o += K
        let rhoOpt = o; o += K
        let ram = o; o += 1
        return State(observation: obs, obsMean: mean, obsSM: osm, state: st,
                     policyMean: pol, actionSignal: act, scoreGrad: sg,
                     reward: rew, td: td, tdPsi: tdP,
                     chord: cs, chordPrev: csP,
                     utilFeat: uf, utilOpt: uo, bonusW: bw,
                     planAdv: pa,
                     alpha: al, beta: be, h: h,
                     nTdMat: ntd, optRewardTd: ortd, durationTd: dtd, durationHat: dhat,
                     qOption: qo, obsTraceMulti: otr,
                     prioOpt: prio, continuation: cont, rhoOpt: rhoOpt,
                     replacementAgeMean: ram, total: o)
    }

    struct UInts { let counters, optFeature, featureAge, optAge, featureEverUsed, total: Int }
    static func uintLayout(rank: Int) -> UInts {
        // 6 counters: OBS_NF, LEARN_NF, PLAN_NF, REINIT_OPT, REINIT_FEAT, TDPSI_NF.
        // (Indices match the constants at the top of OakKernel.metal.)
        let K = optionCount(rank: rank)
        var o = 0
        let ctr = o; o += 6
        let of = o; o += K
        let fa = o; o += rank
        let oa = o; o += K
        let feu = o; o += rank
        return UInts(counters: ctr, optFeature: of, featureAge: fa, optAge: oa,
                     featureEverUsed: feu, total: o)
    }
}

// =================================================================================================
// Shared-storage helper.
// =================================================================================================
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

// =================================================================================================
// Metal engine - orthodox Step 11 Oak. Five fused GPU phases per step, dispatched as one ICB:
//   bk_encode(+meta_predict) -> plan -> plan_reduce -> learn_heads -> post_learn(+G&T commit)
// =================================================================================================
final class OakMetalEngine {
    private struct Phase { let name: String; let tgSize: Int }
    private static let phases: [Phase] = [
        // FUSED oak_bk_obs + oak_encode: per-feature Welford EMA + obsTrace EMA (bk_obs
        // phase, threads with tid < O), threadgroup_barrier(mem_device), Fourier features
        // (encode phase, all R=rank threads compute one feat each), then meta_predict in
        // the first hardware simdgroup. Single TG of R threads.
        Phase(name: "oak_bk_encode",       tgSize: -1),     // -1 → host fills with rank (1 TG of R threads).
        // K threadgroups × R threads each (one TG per option). Each thread (j) computes
        // one row of the dense option transition; per-TG simd_sum + threadgroup_barrier
        // reduces to vHat^k; lane 0 emits Q^k = r̂^k − ρ·l̂^k + vHat^k into qOption[K].
        Phase(name: "oak_plan",            tgSize: -1),     // -1 → host fills with rank
        // K-thread cross-option reduction: simd_max(qOption) + planAdv soft-bound.
        Phase(name: "oak_plan_reduce",     tgSize: 0),
        Phase(name: "oak_learn_heads",     tgSize: 0),
        // FUSED post_learn + admit_commit. A single TG keeps utility updates, action μ, and
        // G&T replacement in one dispatch; the action region is padded to a simdgroup boundary
        // so each action dim starts at lane 0 even when K is below the hardware lane width.
        //
        // Performance pass on top of D2: oak_post_learn's action region used one thread
        // per action dim, each doing K·R sequential fma and leaving most hardware lanes idle.
        // Now parallel: each active (p, k) is its own lane, with one hardware-width group per
        // action dim. Lane 0 of each p-simdgroup reads the
        // K dot products via simd_shuffle and applies chord with fma in the same kk=0..K-1
        // order as the original sequential code — bit-id preserved per Inv-10. Explicit
        // tgSize = postLearnThreadCount locks the simdgroup-aligned layout and gives the
        // in-kernel admit/commit phase enough lanes to reseed cooperatively.
        Phase(name: "oak_post_learn",      tgSize: -2),      // -2 → host fills with padded postLearnThreadCount.
    ]

    private let obsDim, baseObsDim, actDim, rank, dPolicy: Int
    private let queue: MTLCommandQueue
    private let pipelines: [MTLComputePipelineState]
    private let scalarsBuffer: MTLBuffer
    private let statsBuffer: MTLBuffer
    private let configBuffer: MTLBuffer
    private let scalarsPointer: UnsafeMutablePointer<KernelScalars>
    private let statsPointer: UnsafeMutablePointer<KernelStats>
    private let configPointer: UnsafeMutablePointer<KernelConfig>
    private let paramsArena: Shared<Float>
    private let stateArena: Shared<Float>
    private let uintArena: Shared<UInt32>
    private let stateLayout: OakMetalLayout.State
    private let uintLayout: OakMetalLayout.UInts
    private let phaseGridWidths: [Int]
    private let phaseThreadgroupSizes: [Int]
    private let diagnosticGpuPhaseTiming: Bool
    private var phaseGpuTimeSumSeconds: [Double]
    private var phaseGpuInvalidStepCount: [Int]
    private var hostAction: [Float]
    private let indirectCommandBuffer: MTLIndirectCommandBuffer
    private let residencySet: MTLResidencySet
    private let resourceList: [MTLBuffer]

    private static func kernelURL() throws -> URL {
        try requiredBuildPath(".build/OakKernel.metallib",
                              description: "non-empty .build/OakKernel.metallib", nonEmpty: true)
    }

    private static func makePipelines(device: MTLDevice) throws -> [MTLComputePipelineState] {
        let library = try device.makeLibrary(URL: try kernelURL())
        return try phases.map { phase in
            guard let fn = library.makeFunction(name: phase.name) else {
                throw OakError.message("Missing \(phase.name) in OakKernel.metallib.")
            }
            let desc = MTLComputePipelineDescriptor()
            desc.computeFunction = fn
            desc.supportIndirectCommandBuffers = true
            return try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
        }
    }

    static func executionWidth(device: MTLDevice) throws -> Int { 32 }

    init(obsDim: Int, baseObsDim: Int, actDim: Int, rank: Int, dPolicy: Int,
         device: MTLDevice, seed: Int?, diagnosticGpuPhaseTiming: Bool = false) throws {
        // R (rank) is a COMPUTE BUDGET (Plan Step 2: "performance depends on the resource
        // budget — how many features can be considered in parallel"), independent of the env's
        // input dim. baseObsDim is the only env-given quantity. R must be even because each
        // Fourier pre-activation produces a paired (cos, sin) into the R-wide state.
        precondition(rank >= 2, "rank must be >= 2 (compute budget; Fourier pair).")
        precondition(rank % 2 == 0, "rank must be even (Fourier pair encoding); host must round up.")
        precondition(rank <= 256, "rank must be <= 256 (oak_plan threadgroup storage cap).")
        precondition(baseObsDim >= 1, "baseObsDim must be positive (env input dim).")
        precondition(obsDim >= baseObsDim, "obsDim must be >= baseObsDim.")
        precondition(dPolicy >= 1, "dPolicy must be positive.")
        self.obsDim = obsDim; self.baseObsDim = baseObsDim; self.actDim = actDim
        self.rank = rank; self.dPolicy = dPolicy
        guard let queue = device.makeCommandQueue() else {
            throw OakError.message("Failed to create Metal command queue.")
        }
        self.queue = queue
        pipelines = try Self.makePipelines(device: device)

        scalarsBuffer = device.makeBuffer(length: MemoryLayout<KernelScalars>.stride,
                                          options: [.storageModeShared])!
        statsBuffer   = device.makeBuffer(length: MemoryLayout<KernelStats>.stride,
                                          options: [.storageModeShared])!
        configBuffer  = device.makeBuffer(length: MemoryLayout<KernelConfig>.stride,
                                          options: [.storageModeShared])!
        scalarsPointer = scalarsBuffer.contents().bindMemory(to: KernelScalars.self, capacity: 1)
        statsPointer   = statsBuffer.contents().bindMemory(to: KernelStats.self, capacity: 1)
        configPointer  = configBuffer.contents().bindMemory(to: KernelConfig.self, capacity: 1)
        scalarsPointer.initialize(to: KernelScalars())
        statsPointer.initialize(to: KernelStats())
        configPointer.initialize(to: KernelConfig())

        let paramLayout = OakMetalLayout.paramLayout(obsDim: obsDim, baseObsDim: baseObsDim, rank: rank, dPolicy: dPolicy)
        stateLayout = OakMetalLayout.stateLayout(obsDim: obsDim, baseObsDim: baseObsDim, rank: rank, dPolicy: dPolicy)
        uintLayout  = OakMetalLayout.uintLayout(rank: rank)
        paramsArena = Shared(device: device, count: paramLayout.total, zero: Float(0))
        stateArena  = Shared(device: device, count: stateLayout.total, zero: Float(0))
        uintArena   = Shared(device: device, count: uintLayout.total,  zero: UInt32(0))

        let K = OakMetalLayout.optionCount(rank: rank)
        // Grid widths sized for one row (single-stream).
        let learnTotal = rank
                       + OakMetalLayout.psiCount(rank: rank)
                       + OakMetalLayout.optPolicyCount(rank: rank, dPolicy: dPolicy)
                       + OakMetalLayout.optRewardCount(rank: rank)
                       + OakMetalLayout.optTransitionCount(rank: rank)
                       + OakMetalLayout.optDurationCount(rank: rank)
        let postLearnTotal = OakMetalLayout.postLearnThreadCount(rank: rank, dPolicy: dPolicy)
        if let postIndex = Self.phases.firstIndex(where: { $0.name == "oak_post_learn" }) {
            precondition(postLearnTotal <= pipelines[postIndex].maxTotalThreadsPerThreadgroup,
                         "post_learn fused TG exceeds Metal threadgroup cap; reduce OAK_RANK/action dim or redesign post_learn.")
        }
        let grids = [
            1,                  // bk_encode         [FUSED: 1 TG of R threads, bk_obs phase (tid<O) + threadgroup_barrier + encode phase (all R)]
            K,                  // plan              [K TGs (one per option) x R threads (one per output dim)]
            K,                  // plan_reduce       [K threads, simd_max(qOption) + planAdv soft-bound]
            learnTotal,         // learn_heads       [N threads, dense optTransition adds K*R^2]
            1,                  // post_learn        [1 TG: util/metaopt | util_opt | padded action | admit_commit]
        ]
        phaseGridWidths = grids

        phaseThreadgroupSizes = Self.phases.map { phase in
            if phase.tgSize == -1 { return rank }
            if phase.tgSize == -2 { return postLearnTotal }
            return phase.tgSize
        }

        self.diagnosticGpuPhaseTiming = diagnosticGpuPhaseTiming
        phaseGpuTimeSumSeconds = Array(repeating: 0, count: Self.phases.count)
        phaseGpuInvalidStepCount = Array(repeating: 0, count: Self.phases.count)
        hostAction = Array(repeating: 0, count: dPolicy)

        // Metal 4 ICB + MTLResidencySet setup. 6 binds: params, state, uint, scalars, stats, config.
        let icbDesc = MTLIndirectCommandBufferDescriptor()
        icbDesc.commandTypes = [.concurrentDispatch]
        icbDesc.inheritBuffers = false
        icbDesc.inheritPipelineState = false
        icbDesc.maxKernelBufferBindCount = 6
        guard let icb = device.makeIndirectCommandBuffer(descriptor: icbDesc,
                                                         maxCommandCount: Self.phases.count,
                                                         options: [.storageModeShared]) else {
            throw OakError.message("Failed to create Metal indirect command buffer.")
        }
        indirectCommandBuffer = icb
        let rsDesc = MTLResidencySetDescriptor()
        rsDesc.initialCapacity = 6
        residencySet = try device.makeResidencySet(descriptor: rsDesc)
        resourceList = [paramsArena.buffer, stateArena.buffer, uintArena.buffer,
                        scalarsBuffer, statsBuffer, configBuffer]
        for buf in resourceList { residencySet.addAllocation(buf) }
        residencySet.commit()
        queue.addResidencySet(residencySet)

        // Parameter init: structural 1/sqrt(R) scales on every linear head. The encoder uses
        // a Random-Fourier-Features bandwidth-1 init (Rahimi-Recht 2007 §3): with standardized
        // unit-variance inputs, sampling `W_enc ~ U[-1, 1]` gives Var(z) = baseO/3, putting |z|
        // squarely in the strongly-nonlinear cos/sin regime and collapsing the cos DC bias
        // E[cos(z)] = exp(-σ_z²/2) so the basis is effectively zero-centered. Plan-orthodox:
        // bandwidth=1 is structurally derived from "the basis spans its nonlinearity over the
        // input's natural scale"; no tunable, no per-env constant.
        let invR: Float = 1.0 / sqrt(Float(max(rank, 1)))
        let encScale: Float = 1.0
        let seedBase = UInt32(bitPattern: Int32(truncatingIfNeeded: seed ?? Int(oakUnseededSeed)))
                       &* 0x9e3779b9
        func sample(_ off: Int, _ count: Int, _ scale: Float) {
            let base = paramsArena.pointer.advanced(by: off)
            for i in 0..<count {
                var x = seedBase &+ UInt32(off &+ i) &* 0x7feb352d
                x = (x ^ (x &>> 16)) &* 0x846ca68b
                x ^= x &>> 16
                let u = Float(x >> 8) * 5.9604645e-8 * 2 - 1
                base[i] = u * scale
            }
        }
        sample(paramLayout.encoder,
               OakMetalLayout.encoderCount(obsDim: obsDim, baseObsDim: baseObsDim,
                                            rank: rank, dPolicy: dPolicy),
               encScale)
        // Fourier phase: uniform on [0, 2π) — structural, no tunable. Reuses the same
        // splitmix-style hash as `sample` but rescales (u + 1) ∈ [0, 2) → [0, 2π).
        do {
            let base = paramsArena.pointer.advanced(by: paramLayout.phase)
            let count = OakMetalLayout.phaseCount(rank: rank)
            let twoPi: Float = 6.283185307179586
            for i in 0..<count {
                var x = seedBase &+ UInt32(paramLayout.phase &+ i) &* 0x7feb352d
                x = (x ^ (x &>> 16)) &* 0x846ca68b
                x ^= x &>> 16
                let u01 = Float(x >> 8) * 5.9604645e-8           // u01 ∈ [0, 1).
                base[i] = u01 * twoPi
            }
        }
        sample(paramLayout.value, rank, invR)
        sample(paramLayout.psi, OakMetalLayout.psiCount(rank: rank), invR)
        let policyPtr = paramsArena.pointer.advanced(by: paramLayout.optPolicy)
        for i in 0..<OakMetalLayout.optPolicyCount(rank: rank, dPolicy: dPolicy) { policyPtr[i] = 0 }
        sample(paramLayout.optReward, OakMetalLayout.optRewardCount(rank: rank), invR)
        sample(paramLayout.optTransition, OakMetalLayout.optTransitionCount(rank: rank), invR)
        // Duration defaults to the structural one-step lower bound.
        let durationPtr = paramsArena.pointer.advanced(by: paramLayout.optDuration)
        for i in 0..<OakMetalLayout.optDurationCount(rank: rank) { durationPtr[i] = 0 }
        // psi feature-attainment seed: cumulant k gets +invR on its anchor feature index k.
        let psiPtr = paramsArena.pointer.advanced(by: paramLayout.psi)
        for k in 0..<K where k < rank {
            psiPtr[k * rank + k] += invR
        }
        let bonusSlot = stateArena.pointer.advanced(by: stateLayout.bonusW)
        let valuePtr = paramsArena.pointer.advanced(by: paramLayout.value)
        for i in 0..<rank { bonusSlot[i] = valuePtr[i] }

        // Chord seed: uniform 1/K (max-entropy prior). meta_policy recomputes each step.
        let chordSlot = stateArena.pointer.advanced(by: stateLayout.chord)
        let invK: Float = 1.0 / Float(max(K, 1))
        for k in 0..<K { chordSlot[k] = invK }

        // Per-option utility prior: utilOpt[k] = R (capacity-volume). The EMA in
        // oak_post_learn decays this toward true |δ_k|², making low-utility old options
        // eligible for Step-11 remove/replace without any tunable warmup gate.
        let utilOptSlot = stateArena.pointer.advanced(by: stateLayout.utilOpt)
        let utilOptPrior: Float = Float(rank)
        for k in 0..<K { utilOptSlot[k] = utilOptPrior }
        let optFeatureSlot = uintArena.pointer.advanced(by: uintLayout.optFeature)
        let featureEverUsedSlot = uintArena.pointer.advanced(by: uintLayout.featureEverUsed)
        for k in 0..<K {
            optFeatureSlot[k] = UInt32(k % rank)
            featureEverUsedSlot[k % rank] = 1
        }

        // MetaOptimize L-approximation seed (Sharifnassab-Salehkaleybar-Sutton 2024
        // arXiv:2402.02342). α[i] = 1/baseO is the prior step size, β[i] = log α[i] is
        // its log so the meta-update is multiplicative, h[i] = 0 is the initial accumulator
        // (no meta-gradient history at t=0).
        let alphaInit: Float = 1.0 / Float(max(baseObsDim, 1))
        let betaInit: Float  = log(alphaInit)
        let alphaSlot = stateArena.pointer.advanced(by: stateLayout.alpha)
        let betaSlot  = stateArena.pointer.advanced(by: stateLayout.beta)
        let hSlot     = stateArena.pointer.advanced(by: stateLayout.h)
        for i in 0..<rank {
            alphaSlot[i] = alphaInit
            betaSlot[i]  = betaInit
            hSlot[i]     = 0
        }

        writeKernelConfig(paramLayout: paramLayout)

        // Pre-encode all fixed phases into the ICB. Grids are fixed for the engine's lifetime.
        let buffers = resourceList
        for i in 0..<Self.phases.count {
            let cmd = icb.indirectComputeCommandAt(i)
            cmd.setComputePipelineState(pipelines[i])
            for j in 0..<buffers.count { cmd.setKernelBuffer(buffers[j], offset: 0, at: j) }
            // tgSize sentinel: -1 means "use runtime rank as TG size" (oak_plan needs R
            // threads per option-TG; rank is the only Plan-orthodox knob that maps onto a
            // GPU SIMD width, so all R-sized TGs key off it directly).
            let phaseTgSize = phaseThreadgroupSizes[i]
            if phaseTgSize > 0 {
                let tgCount = max(phaseGridWidths[i], 1)
                cmd.concurrentDispatchThreadgroups(
                    MTLSize(width: tgCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: phaseTgSize, height: 1, depth: 1))
            } else {
                let tg = min(max(pipelines[i].threadExecutionWidth, 1),
                             pipelines[i].maxTotalThreadsPerThreadgroup)
                let w = alignedGridWidth(threadCount: phaseGridWidths[i], pipeline: pipelines[i])
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

    func metalPhaseGpuDiagnostics(measuredSteps: Int) -> (meanMs: [String: Double]?, invalidPhases: [String]?) {
        guard diagnosticGpuPhaseTiming, measuredSteps > 0 else { return (nil, nil) }
        var mean: [String: Double] = [:]; var invalid: [String] = []
        let n = Double(measuredSteps)
        for (i, phase) in Self.phases.enumerated() {
            mean[phase.name] = phaseGpuTimeSumSeconds[i] / n * 1000
            if phaseGpuInvalidStepCount[i] > 0 { invalid.append(phase.name) }
        }
        return (mean, invalid.isEmpty ? nil : invalid)
    }

    // Single-stream step. Inputs are scalar per-step values + dPolicy-wide vectors.
    // prevDone is intentionally NOT plumbed through: the agent is continual (Inv-4); env resets
    // are wrapper events.
    func step(obs: UnsafeBufferPointer<Float>,
              prevActionSignal: [Float], prevScoreGrad: [Float], prevReward: Float,
              accumulatePhaseGpu: Bool) throws -> (PolicyMeanBatch, Double, KernelStats) {
        let start = DispatchTime.now().uptimeNanoseconds
        precondition(obs.count == obsDim)
        precondition(prevActionSignal.count == dPolicy)
        precondition(prevScoreGrad.count == dPolicy)
        memcpy(stateArena.pointer.advanced(by: stateLayout.observation), obs.baseAddress!,
               obsDim * MemoryLayout<Float>.stride)
        stateArena.pointer.advanced(by: stateLayout.reward).pointee = prevReward
        memcpy(stateArena.pointer.advanced(by: stateLayout.actionSignal), prevActionSignal,
               dPolicy * MemoryLayout<Float>.stride)
        memcpy(stateArena.pointer.advanced(by: stateLayout.scoreGrad), prevScoreGrad,
               dPolicy * MemoryLayout<Float>.stride)

        // Reset per-step non-finite counters (cumulative reinit counters at indices 3, 4 stay).
        let ctr = uintArena.pointer.advanced(by: uintLayout.counters)
        ctr[0] = 0; ctr[1] = 0; ctr[2] = 0; ctr[5] = 0

        try runKernel(accumulatePhaseGpu: accumulatePhaseGpu)

        memcpy(&hostAction, stateArena.pointer.advanced(by: stateLayout.policyMean),
               dPolicy * MemoryLayout<Float>.stride)
        return (PolicyMeanBatch(values: hostAction, dPolicy: dPolicy),
                Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000,
                statsPointer.pointee)
    }

    private func writeKernelConfig(paramLayout: OakMetalLayout.Param) {
        let K = OakMetalLayout.optionCount(rank: rank)
        let p = paramLayout, s = stateLayout, u = uintLayout
        var c = KernelConfig()
        c.obsDim = UInt32(obsDim); c.rank = UInt32(rank)
        c.optionCount = UInt32(K); c.baseObsDim = UInt32(baseObsDim)
        c.actionParamCount = UInt32(dPolicy)
        // Tier I1: M = floor(log2(baseO)) + 1; baseObsTraceLevels = M − 1.
        c.baseObsTraceLevels = UInt32(max(OakMetalLayout.obsTraceLevelCount(baseObsDim: baseObsDim) - 1, 0))
        c.pEncoder = UInt32(p.encoder); c.pPhase = UInt32(p.phase)
        c.pValue = UInt32(p.value); c.pPsi = UInt32(p.psi)
        c.pOptPolicy = UInt32(p.optPolicy)
        c.pOptReward = UInt32(p.optReward); c.pOptTransition = UInt32(p.optTransition)
        c.pOptDuration = UInt32(p.optDuration)
        c.sObservation = UInt32(s.observation); c.sObsMean = UInt32(s.obsMean); c.sObsSM = UInt32(s.obsSM)
        c.sState = UInt32(s.state)
        c.sPolicyMean = UInt32(s.policyMean)
        c.sActionSignal = UInt32(s.actionSignal)
        c.sScoreGrad = UInt32(s.scoreGrad)
        c.sReward = UInt32(s.reward); c.sTd = UInt32(s.td); c.sTdPsi = UInt32(s.tdPsi)
        c.sChord = UInt32(s.chord); c.sChordPrev = UInt32(s.chordPrev)
        c.sUtilFeat = UInt32(s.utilFeat); c.sUtilOpt = UInt32(s.utilOpt)
        c.sBonusW = UInt32(s.bonusW)
        c.sPlanAdv = UInt32(s.planAdv)
        c.sAlpha = UInt32(s.alpha); c.sBeta = UInt32(s.beta); c.sH = UInt32(s.h)
        c.sNTdMat = UInt32(s.nTdMat)
        c.sOptRewardTd = UInt32(s.optRewardTd)
        c.sDurationTd = UInt32(s.durationTd)
        c.sDurationHat = UInt32(s.durationHat)
        c.sQOption = UInt32(s.qOption)
        c.sObsTraceMulti = UInt32(s.obsTraceMulti)
        c.sPrioOpt = UInt32(s.prioOpt)
        c.sContinuation = UInt32(s.continuation)
        c.sRhoOpt = UInt32(s.rhoOpt)
        c.sReplacementAgeMean = UInt32(s.replacementAgeMean)
        c.uCounters = UInt32(u.counters)
        c.uOptFeature = UInt32(u.optFeature)
        c.uFeatureAge = UInt32(u.featureAge)
        c.uOptAge = UInt32(u.optAge)
        c.uFeatureEverUsed = UInt32(u.featureEverUsed)
        configPointer.pointee = c
    }

    private func alignedGridWidth(threadCount: Int, pipeline: MTLComputePipelineState) -> Int {
        let align = max(Int(pipeline.threadExecutionWidth), 32)
        let n = max(threadCount, 1)
        return ((n + align - 1) / align) * align
    }

    private func runKernel(accumulatePhaseGpu: Bool) throws {
        let split = diagnosticGpuPhaseTiming && accumulatePhaseGpu
        if !split {
            // Hot path: one command buffer, one encoder, one ICB execution per phase with a
            // memory barrier between phases (required for RAW/WAW correctness on shared buffers).
            //
            // The MTLCommandBuffer + MTLComputeCommandEncoder are NSObject subclasses (Metal's
            // Objective-C interop). In a tight long-running CLI loop with no run loop, Swift's
            // ARC cannot drain autoreleased Obj-C objects on its own — each CB keeps GPU-side
            // command storage alive until the pool drains. Wrapping the per-step allocation in
            // an explicit `autoreleasepool` bounds the allocation lifetime to one step and
            // eliminates the slow resident-set growth observed in Activity Monitor over 10M+
            // step runs.
            try autoreleasepool {
                guard let cb = queue.makeCommandBufferWithUnretainedReferences() else {
                    throw OakError.message("Failed to create Metal command buffer.")
                }
                guard let enc = cb.makeComputeCommandEncoder() else {
                    throw OakError.message("Failed to create Metal compute encoder.")
                }
                for i in 0..<pipelines.count {
                    enc.executeCommandsInBuffer(indirectCommandBuffer, range: i..<(i + 1))
                    if i + 1 < pipelines.count { enc.memoryBarrier(scope: [.buffers]) }
                }
                enc.endEncoding()
                cb.commit(); cb.waitUntilCompleted()
                if let e = cb.error { throw OakError.message("Metal command failed: \(e)") }
            }
            return
        }
        // Diagnostic path: per-phase command buffers to capture gpuStartTime/gpuEndTime.
        // Each CB gets its own autoreleasepool so 9 per-phase CBs drain per step, not 10M
        // accumulated across a diagnostic run.
        for i in 0..<pipelines.count {
            try autoreleasepool {
                guard let cb = queue.makeCommandBufferWithUnretainedReferences() else {
                    throw OakError.message("Failed to create Metal command buffer.")
                }
                guard let enc = cb.makeComputeCommandEncoder() else {
                    throw OakError.message("Failed to create Metal compute encoder.")
                }
                enc.executeCommandsInBuffer(indirectCommandBuffer, range: i..<(i + 1))
                enc.endEncoding()
                cb.commit(); cb.waitUntilCompleted()
                if let e = cb.error { throw OakError.message("Metal command failed: \(e)") }
                let s = cb.gpuStartTime, end = cb.gpuEndTime
                if s > 0, end >= s { phaseGpuTimeSumSeconds[i] += end - s }
                else { phaseGpuInvalidStepCount[i] += 1 }
            }
        }
    }
}

// =================================================================================================
// Run-metrics persistence.
// =================================================================================================
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

// =================================================================================================
// Run context and runner.
// =================================================================================================
private struct RunContext {
    let envId: String
    let policy: Policy
    let seed: Int?
    let stopAfter: Int
    let command: String
    let includeDiagnostic: Bool
    let metricsDir: URL
    let env: EnvPoolEnv
    let engine: OakMetalEngine?
}

private final class RunState {
    var prevReward: Float = 0
    var prevPolicyMean: [Float]
    var prevActionSignal: [Float]
    var prevScoreGrad: [Float]
    var randPolicyMean: [Float]
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
    var health: HealthAccumulator
    let actionCol: MeasuredActionCollector

    init(env: EnvPoolEnv, seed: Int?) {
        prevPolicyMean = Array(repeating: 0, count: env.dPolicy)
        prevActionSignal = Array(repeating: 0, count: env.dPolicy)
        prevScoreGrad  = Array(repeating: 0, count: env.dPolicy)
        randPolicyMean = Array(repeating: 0, count: env.dPolicy)
        randSeed = UInt64(bitPattern: Int64(seed ?? Int(oakUnseededSeed)))
        // rank = obsDim per the agent's structural choice (see OakMetalEngine init precondition).
        health = HealthAccumulator(obsDim: env.obsDim)
        actionCol = MeasuredActionCollector(discreteCount: env.actionDiscreteCount,
                                             envActionWidth: env.envActionWidth)
    }
}

// =================================================================================================
// CLI main.
// =================================================================================================
@main struct Main {
    static func main() {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            switch args.first {
            case "benchmark": try benchmark()
            case "diagnose":  try diagnose(Array(args.dropFirst()))
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
        let device = try metalDevice()
        let stopAfter = max(ProcessInfo.processInfo.environment["OAK_STOP_AFTER_STEPS"].flatMap(Int.init) ?? 1, 1)
        let baseline = max(ProcessInfo.processInfo.environment["OAK_BASELINE_RANDOM_STEPS"].flatMap(Int.init) ?? 0, 0)
        if policy == .oak, ProcessInfo.processInfo.environment["OAK_STOP_AFTER_STEPS"] == nil {
            fputs("OaK: OAK_STOP_AFTER_STEPS unset (short run). See AGENTS.md.\n", stderr)
        }
        let dir = RunSnapshot.resolveMetricsDirectory(explicit: nil)
        let phases: [(Policy, Int)] = policy == .oak && baseline > 0
            ? [(.random, baseline), (policy, stopAfter)] : [(policy, stopAfter)]
        var metricsURL: URL!
        for (p, steps) in phases {
            (_, metricsURL) = try runSingle(envId: envId, policy: p, seed: seed,
                                             stopAfter: steps, device: device,
                                             command: "run", metricsDir: dir)
        }
        fputs("\(metricsURL.path)\n", stderr)
    }

    private static func diagnose(_ args: [String]) throws {
        var envId: String?, policy = Policy.oak, stopAfter: Int?
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
        let device = try metalDevice()
        let steps = max(stopAfter
                        ?? ProcessInfo.processInfo.environment["OAK_DIAG_STEPS"].flatMap(Int.init)
                        ?? 1, 1)
        let (_, metricsURL) = try runSingle(
            envId: resolved, policy: policy, seed: seed,
            stopAfter: steps, device: device,
            command: "diagnose", includeDiagnostic: true)
        fputs("\(metricsURL.path)\n", stderr)
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
        let seedCount = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 5
        let stopAfter = max(ProcessInfo.processInfo.environment["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 500000, 1)
        let policies = try (ProcessInfo.processInfo.environment["OAK_BENCH_POLICIES"] ?? "oak,random")
            .split(separator: ",").map { try parsePolicy(String($0)) }
        let device = try metalDevice()
        let concurrency = max(ProcessInfo.processInfo.environment["OAK_BENCH_CONCURRENCY"].flatMap(Int.init) ?? 8, 1)
        let envOrder = try benchmarkEnvOrder(seed: seedStart)
        let expected = envOrder.count * seedCount * policies.count
        let benchDir = resolveBenchmarkWorkingDirectory(root: root, expectedTasks: expected)
        try FileManager.default.createDirectory(at: benchDir, withIntermediateDirectories: true)

        struct WorkItem { let env: BenchmarkEnv; let seed: Int; let policy: Policy }
        var work: [WorkItem] = []
        for env in envOrder {
            for seed in seedStart..<(seedStart + seedCount) {
                for policy in policies {
                    let stem = RunSnapshot.snapshotStem(envId: env.envId, policy: policy, seed: seed)
                    let snapURL = benchDir
                        .appendingPathComponent(env.envId, isDirectory: true)
                        .appendingPathComponent(RunSnapshot.seedFolderName(seed), isDirectory: true)
                        .appendingPathComponent("\(stem).json", isDirectory: false)
                    if FileManager.default.fileExists(atPath: snapURL.path) { continue }
                    work.append(WorkItem(env: env, seed: seed, policy: policy))
                }
            }
        }
        let totalWork = work.count
        fputs("OaK benchmark: \(totalWork) work items, single-stream B=1, N=\(concurrency) concurrent (steps=\(stopAfter), seeds=\(seedCount))\n", stderr)
        fflush(stderr)

        final class SyncState: @unchecked Sendable {
            var completed = 0
            var firstError: Error?
            let lock = NSLock()
        }
        let state = SyncState()
        let progressEvery = max(1, totalWork / 20)
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = concurrency

        for item in work {
            queue.addOperation {
                do {
                    _ = try runSingle(envId: item.env.envId, policy: item.policy, seed: item.seed,
                                      stopAfter: stopAfter,
                                      device: device,
                                      command: "benchmark", metricsDir: benchDir).0
                    state.lock.lock()
                    state.completed += 1
                    let n = state.completed
                    state.lock.unlock()
                    if n % progressEvery == 0 || n == totalWork {
                        fputs("OaK benchmark progress: \(n)/\(totalWork)\n", stderr)
                        fflush(stderr)
                    }
                } catch {
                    let stem = RunSnapshot.snapshotStem(envId: item.env.envId, policy: item.policy, seed: item.seed)
                    let logURL = benchDir.appendingPathComponent("\(stem).log")
                    try? "\(error)\n".data(using: .utf8)?.write(to: logURL)
                    state.lock.lock()
                    if state.firstError == nil { state.firstError = error }
                    state.lock.unlock()
                }
            }
        }
        queue.waitUntilAllOperationsAreFinished()
        if let e = state.firstError { throw e }
    }

    private static func parsePolicy(_ raw: String) throws -> Policy {
        guard let p = Policy(rawValue: raw.lowercased()) else {
            throw OakError.message("Unknown policy '\(raw)'.")
        }
        return p
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

    private static func randomPolicyMean(env: EnvPoolEnv, stepCount: inout UInt64,
                                          floats: inout [Float]) -> PolicyMeanBatch {
        func mix(_ x: UInt64) -> UInt64 {
            var z = x &+ 0x9e3779b97f4a7c15
            z = (z ^ (z &>> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z &>> 27)) &* 0x94d049bb133111eb
            return z ^ (z &>> 31)
        }
        for i in floats.indices {
            stepCount &+= 1
            let u = Float(mix(stepCount) >> 40) * (1.0 / Float(1 << 24))
            floats[i] = u * 2 - 1
        }
        return PolicyMeanBatch(values: floats, dPolicy: env.dPolicy)
    }

    // -- single-run orchestration (setup -> loop -> snapshot) --------------------------------

    private static func setupRun(envId: String, policy: Policy, seed: Int?,
                                 stopAfter: Int, device: MTLDevice,
                                 command: String,
                                 includeDiagnostic: Bool, metricsDir: URL?) throws -> RunContext {
        let dir = RunSnapshot.resolveMetricsDirectory(explicit: metricsDir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        fputs("OaK metrics \(dir.path)\n", stderr); fflush(stderr)
        let env = try EnvPoolEnv(id: envId, numEnvs: 1, seed: seed)
        // Plan Step 2: rank R is a COMPUTE BUDGET, independent of env input dim. OAK_RANK
        // overrides; default scales R = max(env.obsDim, 64) so small envs still get a
        // meaningful feature budget while large envs get at least their input dim. Rounded
        // up to the nearest even value because the deep Fourier features encoder
        // (Lewandowski-Schuurmans-Machado 2024) emits a paired (cos, sin) per pre-activation.
        let rawRank = max(ProcessInfo.processInfo.environment["OAK_RANK"].flatMap(Int.init)
                          ?? max(env.obsDim, 64), 2)
        let rank = (rawRank + 1) & ~1   // round up to even.
        guard rank <= 256 else {
            throw OakError.message("OAK rank \(rank) exceeds oak_plan's 256-threadgroup-storage cap. Set OAK_RANK<=256 or redesign oak_plan storage.")
        }
        let engine = policy == .oak
            ? try OakMetalEngine(obsDim: env.obsDim, baseObsDim: env.obsDim,
                                 actDim: env.actDim, rank: rank, dPolicy: env.dPolicy,
                                 device: device, seed: seed,
                                 diagnosticGpuPhaseTiming: includeDiagnostic)
            : nil
        try env.reset()
        return RunContext(envId: envId, policy: policy, seed: seed,
                          stopAfter: stopAfter, command: command,
                          includeDiagnostic: includeDiagnostic, metricsDir: dir,
                          env: env, engine: engine)
    }

    private static func emitSnapshot(_ ctx: RunContext, _ state: RunState, seq: Int?) -> RunMetricsFile {
        let measured = max(state.measuredSteps, 1)
        let a = state.agentMsSum / 1000, e = state.envMsSum / 1000
        let h = max(state.measuredSeconds - a - e, 0)
        let merged: RunDiagnostic? = state.health.hasMeasuredSteps ? {
            let (mm, mi) = ctx.engine?.metalPhaseGpuDiagnostics(measuredSteps: measured) ?? (nil, nil)
            return RunDiagnostic(
                agentShare: state.measuredSeconds > 0 ? a / state.measuredSeconds : nil,
                envShare:   state.measuredSeconds > 0 ? e / state.measuredSeconds : nil,
                hostShare:  state.measuredSeconds > 0 ? h / state.measuredSeconds : nil,
                metalPhaseGpuMsMean: mm, metalPhaseGpuTimingInvalid: mi)
        }() : nil
        let diagOut: RunDiagnostic? = ((seq != nil && ctx.policy == .oak) || (seq == nil && ctx.includeDiagnostic))
                                       ? merged : nil
        let (dA, cA) = state.actionCol.partialMetrics()
        let sps = state.measuredSeconds > 0 ? Double(state.measuredSteps) / state.measuredSeconds : nil
        let reason = seq == nil ? "measured_steps" : "heartbeat"
        return RunMetricsFile(
            schemaVersion: 3, command: ctx.command, envId: ctx.envId, policy: ctx.policy, seed: ctx.seed,
            numEnvs: 1, stopAfterMeasuredSteps: ctx.stopAfter,
            observationDim: ctx.env.obsDim, actionDim: ctx.env.actDim,
            actionDiscreteCount: ctx.env.actionDiscreteCount,
            measuredStepCount: state.measuredSteps, measuredBatchCount: measured,
            measuredSeconds: state.measuredSeconds,
            workerDepth: state.measuredSteps,
            stepsPerSecond: sps,
            heartbeatSequence: seq, rewardSum: state.rewardSum, doneCount: state.doneCount,
            stopReason: reason, agentStepMsMean: state.agentMsSum / Double(measured),
            envStepMsMean: state.envMsSum / Double(measured),
            discreteAction: dA, continuousAction: cA,
            health: state.health.snapshot(), diagnostic: diagOut)
    }

    private static func measurementLoop(_ ctx: RunContext, _ state: RunState) throws {
        let defaultHb = ctx.command == "diagnose" ? 0 : max(ctx.stopAfter / 10, 1)
        let hb = max(ProcessInfo.processInfo.environment["OAK_HEARTBEAT_STEPS"].flatMap(Int.init) ?? defaultHb, 0)
        // Continual: every step is a measured step, no warmup. Plan Inv-5: no special training
        // phases. Step 0 timer is established on the first iteration.
        if state.measurementStart == nil { state.measurementStart = DispatchTime.now().uptimeNanoseconds }
        while state.measuredSteps < ctx.stopAfter {
            let measured = true
            let policyMean: PolicyMeanBatch, agentMs: Double, stats: KernelStats?
            if let engine = ctx.engine {
                let out = try ctx.env.observationScratch.withUnsafeBufferPointer { buf in
                    try engine.step(obs: buf, prevActionSignal: state.prevActionSignal,
                                    prevScoreGrad: state.prevScoreGrad,
                                    prevReward: state.prevReward,
                                    accumulatePhaseGpu: measured)
                }
                policyMean = out.0; agentMs = out.1; stats = out.2
            } else {
                let t0 = DispatchTime.now().uptimeNanoseconds
                policyMean = randomPolicyMean(env: ctx.env, stepCount: &state.randSeed,
                                              floats: &state.randPolicyMean)
                agentMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000
                stats = nil
            }
            let envStart = DispatchTime.now().uptimeNanoseconds
            try ctx.env.sampleAndStep(policyMean: policyMean.values)
            let envMs = Double(DispatchTime.now().uptimeNanoseconds - envStart) / 1_000_000
            state.totalSteps += 1
            if measured {
                state.measuredSteps += 1
                if let s = state.measurementStart {
                    state.measuredSeconds = Double(DispatchTime.now().uptimeNanoseconds - s) / 1_000_000_000
                }
                if hb > 0, (state.measuredSteps - 1) / hb < state.measuredSteps / hb {
                    state.pulse += 1
                    _ = try RunSnapshot.persist(emitSnapshot(ctx, state, seq: state.pulse), metricsDir: ctx.metricsDir)
                }
                state.doneCount += Int(ctx.env.doneScratch[0])
                let v = ctx.env.rewardScratch[0]
                guard v.isFinite else { throw OakError.message("Environment produced a non-finite reward.") }
                state.rewardSum += Double(v)
                state.agentMsSum += agentMs; state.envMsSum += envMs
                // Host-computable health (works for ANY policy, including random).
                state.prevPolicyMean.withUnsafeBufferPointer { prev in
                    policyMean.values.withUnsafeBufferPointer { curr in
                        ctx.env.scoreGradScratch.withUnsafeBufferPointer { sg in
                            state.health.recordEnvSignals(reward: v, prevPolicyMean: prev,
                                                          currPolicyMean: curr, scoreGrad: sg)
                        }
                    }
                }
                if let s = stats { state.health.recordKernelStats(s) }
                state.actionCol.addMeasured(envAction: ctx.env.envActionScratch)
            }
            state.prevReward = ctx.env.rewardScratch[0]
            _ = state.prevPolicyMean.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress!, policyMean.values, policyMean.values.count * MemoryLayout<Float>.stride)
            }
            _ = state.prevScoreGrad.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress!, ctx.env.scoreGradScratch,
                       ctx.env.scoreGradScratch.count * MemoryLayout<Float>.stride)
            }
            _ = state.prevActionSignal.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress!, ctx.env.actionSignalScratch,
                       ctx.env.actionSignalScratch.count * MemoryLayout<Float>.stride)
            }
        }
    }

    private static func runSingle(envId: String, policy: Policy, seed: Int?,
                                  stopAfter: Int, device: MTLDevice,
                                  command: String,
                                  includeDiagnostic: Bool = false, metricsDir: URL? = nil)
                                  throws -> (RunMetricsFile, URL) {
        let ctx = try setupRun(envId: envId, policy: policy, seed: seed,
                               stopAfter: stopAfter, device: device,
                               command: command,
                               includeDiagnostic: includeDiagnostic, metricsDir: metricsDir)
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
