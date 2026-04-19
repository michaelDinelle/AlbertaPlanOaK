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
    let rewardRate: Double?
    let latentStateMeanSquare: Double?
    let activeRank: Double?            // legacy alias for activeCount
    let activeCount: Double?
    let pEffectiveRank: Double?
    let actionDeltaL2Mean: Double?
    let scoreGradL2Mean: Double?
    let optionModelPredictionMSE: Double?
    let oakOptionReinitCount: Double?
    let nonFiniteRawTotal: Int
    let nonFiniteSanitizedTotal: Int
    let tdPsiNonFiniteTotal: Int?
    let utilityFeatureMean: Double?, utilityOptionMean: Double?
    let modelErrEMA: Double?, modelErrEMAPrev: Double?
    let alphaPi: Double?
    let oakFeatureReinitCountFinal: Int?
    let valueWMax: Double?, psiMax: Double?, optPolicyMax: Double?
    let optRewardMax: Double?, optTransitionMax: Double?
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
        case .box(let n, _, _): return n
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

// Per-row per-component sampling. Discrete N: Gumbel-max categorical, scoreGrad = 1{a=d} - pi_d.
// Box A: structural-sigma=1 Gaussian, scoreGrad = z (Williams 1992 REINFORCE; the *pre-clamp*
// z is the literal score function for the sampled `raw = mu + z`, regardless of any subsequent
// env-side clamp - the clamp affects only the env's view of the action, not the agent's
// gradient signal).
func sampleAndScoreActions(spec: [ActionComponent], numEnvs: Int, dPolicy: Int,
                           envActionWidth: Int,
                           policyMean: UnsafePointer<Float>,
                           envAction: UnsafeMutablePointer<Float>,
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
                }
                envAction[aBase + aIdx] = Float(bestIdx)
                pIdx += n; aIdx += 1
            case .box(let a, let low, let high):
                for d in 0..<a {
                    let z = nextNormal(&rngState)
                    let mu = policyMean[mBase + pIdx + d]
                    let raw = mu + z
                    let clamped = max(low[d], min(high[d], raw.isFinite ? raw : 0))
                    envAction[aBase + aIdx + d] = clamped
                    // Pre-clamp z is the literal Williams 1992 REINFORCE score for sigma=1
                    // Gaussian: d log pi(a|s) / d mu = (raw - mu) = z. Env-side clamp does
                    // not change the score function the agent's policy parameterized.
                    scoreGrad[mBase + pIdx + d] = z
                }
                pIdx += a; aIdx += a
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
// HealthAccumulator - fixed-size struct fold over KernelStats. NO per-step heap allocation
// (Inv-6). Runs only on measured steps; no work otherwise.
// =================================================================================================
private struct HealthAccumulator {
    private struct Slot { var sum = 0.0; var count = 0 }
    private var slots = [Slot](repeating: Slot(), count: 11)
    private var steps = 0
    private var rawNonFinite = 0, sanitizedNonFinite = 0
    private var encodeNF = 0, learnNF = 0, planNF = 0
    private var maxLat = 0.0, maxPol = 0.0
    private var lastRho: Double?
    private var tdPsiNF = 0
    private var oakFeatureReinitFinal = 0
    private var valueWMax = 0.0, psiMax = 0.0, optPolicyMax = 0.0
    private var optRewardMax = 0.0, optTransitionMax = 0.0

    private mutating func acc(_ i: Int, _ v: Float) {
        let d = Double(v)
        if d.isFinite { slots[i].sum += d; slots[i].count += 1 }
        else { rawNonFinite += 1; sanitizedNonFinite += 1 }
    }

    mutating func record(_ stats: KernelStats) {
        steps += 1
        let rr = Double(stats.rewardRate); if rr.isFinite { lastRho = rr }
        acc(0,  stats.latentStateMeanSquare)
        acc(1,  stats.activeCount)
        acc(2,  stats.actionDeltaL2Mean)
        acc(3,  stats.scoreGradL2Mean)
        acc(4,  stats.optionModelPredictionMSE)
        acc(5,  stats.oakOptionReinitCount)
        acc(6,  stats.utilityFeatureMean)
        acc(7,  stats.utilityOptionMean)
        acc(8,  stats.pEffectiveRank)
        acc(9,  stats.modelErrEMA)
        acc(10, stats.alphaPiStat)
        rawNonFinite += Int(stats.rawNonFiniteCount)
        sanitizedNonFinite += Int(stats.sanitizedNonFiniteCount)
        encodeNF += Int(stats.encodeNonFiniteCount)
        learnNF += Int(stats.learnNonFiniteCount)
        planNF += Int(stats.planNonFiniteCount)
        tdPsiNF += Int(stats.tdPsiNonFiniteCount)
        oakFeatureReinitFinal = Int(stats.oakFeatureReinitCount)
        maxLat = max(maxLat, Double(stats.maxAbsLatentPrediction))
        maxPol = max(maxPol, Double(stats.maxAbsPolicyOutput))
        valueWMax        = max(valueWMax,        Double(stats.valueWMax))
        psiMax           = max(psiMax,           Double(stats.psiMax))
        optPolicyMax     = max(optPolicyMax,     Double(stats.optPolicyMax))
        optRewardMax     = max(optRewardMax,     Double(stats.optRewardMax))
        optTransitionMax = max(optTransitionMax, Double(stats.optTransitionMax))
    }

    private func mean(_ i: Int) -> Double? {
        slots[i].count > 0 ? slots[i].sum / Double(slots[i].count) : nil
    }

    func snapshot() -> RunHealth? {
        guard steps > 0 else { return nil }
        return RunHealth(
            rewardRate: lastRho,
            latentStateMeanSquare: mean(0),
            activeRank: mean(1), activeCount: mean(1),
            pEffectiveRank: mean(8),
            actionDeltaL2Mean: mean(2), scoreGradL2Mean: mean(3),
            optionModelPredictionMSE: mean(4),
            oakOptionReinitCount: mean(5),
            nonFiniteRawTotal: rawNonFinite,
            nonFiniteSanitizedTotal: sanitizedNonFinite,
            tdPsiNonFiniteTotal: tdPsiNF,
            utilityFeatureMean: mean(6), utilityOptionMean: mean(7),
            modelErrEMA: mean(9), modelErrEMAPrev: nil,
            alphaPi: mean(10),
            oakFeatureReinitCountFinal: oakFeatureReinitFinal,
            valueWMax:    valueWMax    > 0 ? valueWMax    : nil,
            psiMax:       psiMax       > 0 ? psiMax       : nil,
            optPolicyMax: optPolicyMax > 0 ? optPolicyMax : nil,
            optRewardMax: optRewardMax > 0 ? optRewardMax : nil,
            optTransitionMax: optTransitionMax > 0 ? optTransitionMax : nil)
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
    var obsDim: UInt32 = 0, rank: UInt32 = 0, actDim: UInt32 = 0
    var optionCount: UInt32 = 0, baseObsDim: UInt32 = 0
    var actionParamCount: UInt32 = 0
    var pEncoder: UInt32 = 0, pValue: UInt32 = 0, pPsi: UInt32 = 0
    var pOptPolicy: UInt32 = 0
    var pOptReward: UInt32 = 0, pOptTransition: UInt32 = 0
    var sObservation: UInt32 = 0, sObsMean: UInt32 = 0, sObsSM: UInt32 = 0
    var sState: UInt32 = 0
    var sPrevPolicyMean: UInt32 = 0, sPolicyMean: UInt32 = 0, sScoreGrad: UInt32 = 0
    var sReward: UInt32 = 0, sTd: UInt32 = 0, sTdPsi: UInt32 = 0, sVCurr: UInt32 = 0
    var sChord: UInt32 = 0, sChordPrev: UInt32 = 0, sBetaProb: UInt32 = 0
    var sFeatSM: UInt32 = 0, sWBar: UInt32 = 0, sFeatMean: UInt32 = 0
    var sUtilFeat: UInt32 = 0, sUtilOpt: UInt32 = 0, sUtilPlan: UInt32 = 0, sOptFracEMA: UInt32 = 0
    var sAlphaImpFeat: UInt32 = 0
    var sPlanAdv: UInt32 = 0
    var uDone: UInt32 = 0, uActiveMask: UInt32 = 0, uActiveList: UInt32 = 0
    var uOption: UInt32 = 0, uCounters: UInt32 = 0, uSubtaskFeatureOf: UInt32 = 0
    var uOptionAge: UInt32 = 0, uPlanArgmax: UInt32 = 0
    var cap: Float = 0, alpha0: Float = 0, ridge: Float = 0
}

private struct KernelScalars {
    var stepCount: UInt32 = 0, activeCount: UInt32 = 0
    var stateSlot: UInt32 = 0, optionSlot: UInt32 = 0
    var activeListIsIdentity: UInt32 = 0
    var rho: Float = 0
    var pEff: Float = 0
    var alphaPi: Float = 0
    var modelErrEMA: Float = 0, modelErrEMAPrev: Float = 0
}

struct KernelStats {
    var rewardRate: Float = 0, latentStateMeanSquare: Float = 0, activeCount: Float = 0
    var rawNonFiniteCount: Float = 0, sanitizedNonFiniteCount: Float = 0
    var encodeNonFiniteCount: Float = 0, learnNonFiniteCount: Float = 0, planNonFiniteCount: Float = 0
    var maxAbsLatentPrediction: Float = 0, maxAbsPolicyOutput: Float = 0
    var actionDeltaL2Mean: Float = 0, scoreGradL2Mean: Float = 0
    var optionModelPredictionMSE: Float = 0
    var oakOptionReinitCount: Float = 0
    var utilityFeatureMean: Float = 0, utilityOptionMean: Float = 0
    var tdPsiNonFiniteCount: Float = 0
    var pEffectiveRank: Float = 0
    var modelErrEMA: Float = 0, modelErrEMAPrev: Float = 0
    var alphaPiStat: Float = 0
    var oakFeatureReinitCount: Float = 0
    var valueWMax: Float = 0, psiMax: Float = 0, optPolicyMax: Float = 0
    var optRewardMax: Float = 0, optTransitionMax: Float = 0
}

// =================================================================================================
// Arena layout - single source of truth for offsets. Single-stream: no `batch` parameter.
// =================================================================================================
private enum OakMetalLayout {
    static let kernelConfigBufferIndex = 5
    static let maxOptionCount = 32

    static func optionCount(rank: Int) -> Int { max(1, min(rank, maxOptionCount)) }
    static func concatDim(rank: Int) -> Int { rank + optionCount(rank: rank) }
    static func psiCount(rank: Int) -> Int { optionCount(rank: rank) * concatDim(rank: rank) }
    static func optPolicyCount(rank: Int, dPolicy: Int) -> Int {
        optionCount(rank: rank) * dPolicy * rank
    }
    static func optRewardCount(rank: Int) -> Int { optionCount(rank: rank) * rank }
    static func optTransitionCount(rank: Int) -> Int { optionCount(rank: rank) * rank * rank }
    static func encoderCount(obsDim: Int, rank: Int) -> Int { rank * max(obsDim, 1) }

    struct Param {
        let encoder, value, psi, optPolicy, optReward, optTransition, total: Int
    }
    static func paramLayout(obsDim: Int, rank: Int, dPolicy: Int) -> Param {
        var o = 0
        let enc = o; o += encoderCount(obsDim: obsDim, rank: rank)
        let val = o; o += rank
        let psi = o; o += psiCount(rank: rank)
        let op  = o; o += optPolicyCount(rank: rank, dPolicy: dPolicy)
        let or_ = o; o += optRewardCount(rank: rank)
        let ot  = o; o += optTransitionCount(rank: rank)
        return Param(encoder: enc, value: val, psi: psi, optPolicy: op,
                     optReward: or_, optTransition: ot, total: o)
    }

    struct State {
        let observation, obsMean, obsSM, state: Int
        let prevPolicyMean, policyMean, scoreGrad: Int
        let reward, td, tdPsi, vCurr: Int
        let chord, chordPrev, betaProb: Int
        let featSM, wBar, featMean: Int
        let utilFeat, utilOpt, utilPlan, optFracEMA: Int
        let alphaImpFeat: Int
        let planAdv: Int
        let total: Int
    }
    static func stateLayout(obsDim: Int, rank: Int, dPolicy: Int) -> State {
        let K = optionCount(rank: rank)
        var o = 0
        let obs = o; o += obsDim
        let mean = o; o += obsDim
        let osm  = o; o += obsDim
        let st = o; o += 2 * rank
        let prP = o; o += dPolicy
        let pol = o; o += dPolicy
        let sg  = o; o += dPolicy
        let rew = o; o += 1
        let td  = o; o += 1
        let tdP = o; o += K
        let vc  = o; o += 1
        let cs  = o; o += K
        let csP = o; o += K
        let bp  = o; o += 1
        let fs  = o; o += rank
        let wb  = o; o += rank
        let fm  = o; o += rank
        let uf  = o; o += rank
        let uo  = o; o += K
        let up  = o; o += rank
        let ofE = o; o += K
        let aif = o; o += rank
        let pa  = o; o += 1
        return State(observation: obs, obsMean: mean, obsSM: osm, state: st,
                     prevPolicyMean: prP, policyMean: pol, scoreGrad: sg,
                     reward: rew, td: td, tdPsi: tdP, vCurr: vc,
                     chord: cs, chordPrev: csP, betaProb: bp,
                     featSM: fs, wBar: wb, featMean: fm,
                     utilFeat: uf, utilOpt: uo, utilPlan: up, optFracEMA: ofE,
                     alphaImpFeat: aif,
                     planAdv: pa, total: o)
    }

    struct UInts {
        let done, activeMask, activeList, option, counters, subtaskFeatureOf, optionAge,
            planArgmax, total: Int
    }
    static func uintLayout(rank: Int) -> UInts {
        let K = optionCount(rank: rank)
        var o = 0
        let d = o; o += 1
        let am = o; o += rank
        let al = o; o += rank
        let op = o; o += 2
        let c  = o; o += 8
        let sf = o; o += K
        let oa = o; o += 1
        let pa = o; o += 1
        return UInts(done: d, activeMask: am, activeList: al, option: op,
                     counters: c, subtaskFeatureOf: sf, optionAge: oa,
                     planArgmax: pa, total: o)
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
// Metal engine - orthodox Step 11 Oak. 12 phases per step, dispatched as one ICB.
//   bk_obs -> encode -> meta_policy -> predict -> plan -> reduce_alpha -> learn_heads
//   -> meta_action -> bk_reduce -> bk_utility -> bk_admit_commit
// (11 phases listed; meta_action runs LAST among learn-time kernels so learn_heads reads the
// score function written at the previous step's meta_action.)
// =================================================================================================
final class OakMetalEngine {
    private struct Phase { let name: String; let tgSize: Int }
    private static let phases: [Phase] = [
        Phase(name: "oak_bk_obs",          tgSize: 32),
        Phase(name: "oak_encode",          tgSize: 0),
        Phase(name: "oak_meta_policy",     tgSize: 0),
        Phase(name: "oak_predict",         tgSize: 0),
        Phase(name: "oak_plan",            tgSize: 0),
        Phase(name: "oak_reduce_alpha",    tgSize: 128),
        Phase(name: "oak_learn_heads",     tgSize: 0),
        Phase(name: "oak_meta_action",     tgSize: 0),
        Phase(name: "oak_bk_reduce",       tgSize: 128),
        Phase(name: "oak_bk_utility",      tgSize: 0),
        Phase(name: "oak_bk_admit_commit", tgSize: 0),
    ]

    private let obsDim, baseObsDim, actDim, rank, dPolicy: Int
    private let queue: MTLCommandQueue
    private let cbDesc: MTLCommandBufferDescriptor
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
        precondition(obsDim == rank, "obsDim must equal rank (interface-derived latent width).")
        precondition(baseObsDim <= obsDim && baseObsDim >= 1, "baseObsDim must be positive and <= obsDim.")
        precondition(dPolicy >= 1, "dPolicy must be positive.")
        self.obsDim = obsDim; self.baseObsDim = baseObsDim; self.actDim = actDim
        self.rank = rank; self.dPolicy = dPolicy
        guard let queue = device.makeCommandQueue() else {
            throw OakError.message("Failed to create Metal command queue.")
        }
        self.queue = queue
        let d = MTLCommandBufferDescriptor(); d.retainedReferences = false
        cbDesc = d
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
        scalarsPointer.pointee.pEff = Float(rank)
        statsPointer.initialize(to: KernelStats())
        configPointer.initialize(to: KernelConfig())

        let paramLayout = OakMetalLayout.paramLayout(obsDim: obsDim, rank: rank, dPolicy: dPolicy)
        stateLayout = OakMetalLayout.stateLayout(obsDim: obsDim, rank: rank, dPolicy: dPolicy)
        uintLayout  = OakMetalLayout.uintLayout(rank: rank)
        paramsArena = Shared(device: device, count: paramLayout.total, zero: Float(0))
        stateArena  = Shared(device: device, count: stateLayout.total, zero: Float(0))
        uintArena   = Shared(device: device, count: uintLayout.total,  zero: UInt32(0))

        let K = OakMetalLayout.optionCount(rank: rank)
        let M = OakMetalLayout.concatDim(rank: rank)
        // Grid widths sized for one row (single-stream).
        let learnTotal = OakMetalLayout.encoderCount(obsDim: obsDim, rank: rank)
                       + rank
                       + OakMetalLayout.psiCount(rank: rank)
                       + OakMetalLayout.optPolicyCount(rank: rank, dPolicy: dPolicy)
                       + OakMetalLayout.optRewardCount(rank: rank)
                       + OakMetalLayout.optTransitionCount(rank: rank)
        phaseGridWidths = [
            1,                  // bk_obs            [1 TG x 32]
            rank,               // encode            [R threads]
            1,                  // meta_policy       [1 thread]
            1,                  // predict           [1 thread]
            1,                  // plan              [1 thread]
            1,                  // reduce_alpha      [1 TG x 128]
            learnTotal,         // learn_heads       [N threads]
            dPolicy,            // meta_action       [D threads]
            1,                  // bk_reduce         [1 TG x 128]
            rank + K + 5,       // bk_utility        [(R + K + 5) threads, 5 learned heads]
            1,                  // bk_admit_commit   [1 thread]
        ]

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

        // Parameter init: structural 1/sqrt(R) scales on every linear head.
        let invR: Float = 1.0 / sqrt(Float(max(rank, 1)))
        let invO: Float = 1.0 / sqrt(Float(max(obsDim, 1)))
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
        sample(paramLayout.encoder, OakMetalLayout.encoderCount(obsDim: obsDim, rank: rank), invO)
        sample(paramLayout.value, rank, invR)
        sample(paramLayout.psi, OakMetalLayout.psiCount(rank: rank), invR)
        sample(paramLayout.optPolicy,
               OakMetalLayout.optPolicyCount(rank: rank, dPolicy: dPolicy), invR)
        sample(paramLayout.optReward, OakMetalLayout.optRewardCount(rank: rank), invR)
        sample(paramLayout.optTransition, OakMetalLayout.optTransitionCount(rank: rank), invR)
        // psi feature-attainment seed: cumulant k gets a +1/sqrt(R) coefficient on its anchor.
        let psiPtr = paramsArena.pointer.advanced(by: paramLayout.psi)
        for k in 0..<K where k < rank {
            let idx = k * M + k
            psiPtr[idx] = psiPtr[idx] + invR
        }

        // Active-feature seed: all admitted from step 0.
        scalarsPointer.pointee.activeCount = UInt32(rank)
        scalarsPointer.pointee.activeListIsIdentity = 1
        let activeMask = uintArena.pointer.advanced(by: uintLayout.activeMask)
        let activeList = uintArena.pointer.advanced(by: uintLayout.activeList)
        for f in 0..<rank { activeMask[f] = 1; activeList[f] = UInt32(f) }

        // Chord seed: uniform 1/K (max-entropy prior). optionAge = 0 forces meta_policy to
        // resample the chord on step 1.
        let chordSlot = stateArena.pointer.advanced(by: stateLayout.chord)
        let invK: Float = 1.0 / Float(max(K, 1))
        for k in 0..<K { chordSlot[k] = invK }
        let optFrac = stateArena.pointer.advanced(by: stateLayout.optFracEMA)
        for k in 0..<K { optFrac[k] = invK }

        // Per-feature implicit alpha seed: alpha_i = alpha_0 (= 1/cap). bk_admit_commit
        // recomputes from featSM each step (parameter-free; MetaOptimize 2402.02342).
        let initAlpha0: Float = 1.0 / Float(max(rank, 1))
        let aifPtr = stateArena.pointer.advanced(by: stateLayout.alphaImpFeat)
        for i in 0..<rank { aifPtr[i] = initAlpha0 }

        // sf seed: identity (k -> k). bk_admit_commit reranks each step.
        let sfPtr = uintArena.pointer.advanced(by: uintLayout.subtaskFeatureOf)
        for k in 0..<K { sfPtr[k] = UInt32(min(k, rank - 1)) }

        writeKernelConfig(paramLayout: paramLayout)

        // Pre-encode all 11 phases into the ICB. Grids are fixed for the engine's lifetime.
        let buffers = resourceList
        for i in 0..<Self.phases.count {
            let cmd = icb.indirectComputeCommandAt(i)
            cmd.setComputePipelineState(pipelines[i])
            for j in 0..<buffers.count { cmd.setKernelBuffer(buffers[j], offset: 0, at: j) }
            let tgSize = Self.phases[i].tgSize
            if tgSize > 0 {
                let tgCount = max(phaseGridWidths[i], 1)
                cmd.concurrentDispatchThreadgroups(
                    MTLSize(width: tgCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
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
    func step(obs: UnsafeBufferPointer<Float>, prevPolicyMean: [Float],
              prevScoreGrad: [Float], prevReward: Float, prevDone: UInt8,
              accumulatePhaseGpu: Bool) throws -> (PolicyMeanBatch, Double, KernelStats) {
        let start = DispatchTime.now().uptimeNanoseconds
        precondition(obs.count == obsDim)
        precondition(prevPolicyMean.count == dPolicy)
        precondition(prevScoreGrad.count == dPolicy)
        memcpy(stateArena.pointer.advanced(by: stateLayout.observation), obs.baseAddress!,
               obsDim * MemoryLayout<Float>.stride)
        stateArena.pointer.advanced(by: stateLayout.reward).pointee = prevReward
        memcpy(stateArena.pointer.advanced(by: stateLayout.prevPolicyMean), prevPolicyMean,
               dPolicy * MemoryLayout<Float>.stride)
        memcpy(stateArena.pointer.advanced(by: stateLayout.scoreGrad), prevScoreGrad,
               dPolicy * MemoryLayout<Float>.stride)
        uintArena.pointer.advanced(by: uintLayout.done).pointee = UInt32(prevDone)

        // Reset per-step non-finite counters; bk_admit_commit clears REINIT_OPT life-counter
        // additions are appended (cumulative).
        let ctr = uintArena.pointer.advanced(by: uintLayout.counters)
        ctr[0] = 0; ctr[1] = 0; ctr[2] = 0; ctr[5] = 0
        statsPointer.pointee = KernelStats()

        try runKernel(accumulatePhaseGpu: accumulatePhaseGpu)

        memcpy(&hostAction, stateArena.pointer.advanced(by: stateLayout.policyMean),
               dPolicy * MemoryLayout<Float>.stride)
        return (PolicyMeanBatch(values: hostAction, dPolicy: dPolicy),
                Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000,
                statsPointer.pointee)
    }

    private func writeKernelConfig(paramLayout: OakMetalLayout.Param) {
        let K = OakMetalLayout.optionCount(rank: rank)
        let cap = Float(max(rank, 1))
        let p = paramLayout, s = stateLayout, u = uintLayout
        var c = KernelConfig()
        c.obsDim = UInt32(obsDim); c.rank = UInt32(rank); c.actDim = UInt32(actDim)
        c.optionCount = UInt32(K); c.baseObsDim = UInt32(baseObsDim)
        c.actionParamCount = UInt32(dPolicy)
        c.pEncoder = UInt32(p.encoder); c.pValue = UInt32(p.value); c.pPsi = UInt32(p.psi)
        c.pOptPolicy = UInt32(p.optPolicy)
        c.pOptReward = UInt32(p.optReward); c.pOptTransition = UInt32(p.optTransition)
        c.sObservation = UInt32(s.observation); c.sObsMean = UInt32(s.obsMean); c.sObsSM = UInt32(s.obsSM)
        c.sState = UInt32(s.state)
        c.sPrevPolicyMean = UInt32(s.prevPolicyMean); c.sPolicyMean = UInt32(s.policyMean)
        c.sScoreGrad = UInt32(s.scoreGrad)
        c.sReward = UInt32(s.reward); c.sTd = UInt32(s.td); c.sTdPsi = UInt32(s.tdPsi)
        c.sVCurr = UInt32(s.vCurr)
        c.sChord = UInt32(s.chord); c.sChordPrev = UInt32(s.chordPrev); c.sBetaProb = UInt32(s.betaProb)
        c.sFeatSM = UInt32(s.featSM); c.sWBar = UInt32(s.wBar); c.sFeatMean = UInt32(s.featMean)
        c.sUtilFeat = UInt32(s.utilFeat); c.sUtilOpt = UInt32(s.utilOpt)
        c.sUtilPlan = UInt32(s.utilPlan); c.sOptFracEMA = UInt32(s.optFracEMA)
        c.sAlphaImpFeat = UInt32(s.alphaImpFeat)
        c.sPlanAdv = UInt32(s.planAdv)
        c.uDone = UInt32(u.done); c.uActiveMask = UInt32(u.activeMask)
        c.uActiveList = UInt32(u.activeList); c.uOption = UInt32(u.option)
        c.uCounters = UInt32(u.counters); c.uSubtaskFeatureOf = UInt32(u.subtaskFeatureOf)
        c.uOptionAge = UInt32(u.optionAge); c.uPlanArgmax = UInt32(u.planArgmax)
        c.cap = cap; c.alpha0 = 1.0 / cap; c.ridge = 1.0 / (cap + 1.0)
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
            guard let cb = queue.makeCommandBuffer(descriptor: cbDesc) else {
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
            return
        }
        // Diagnostic path: per-phase command buffers to capture gpuStartTime/gpuEndTime.
        for i in 0..<pipelines.count {
            guard let cb = queue.makeCommandBuffer(descriptor: cbDesc) else {
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
    let warmup: Int
    let stopAfter: Int
    let command: String
    let includeDiagnostic: Bool
    let metricsDir: URL
    let env: EnvPoolEnv
    let engine: OakMetalEngine?
}

private final class RunState {
    var prevReward: Float = 0
    var prevDone: UInt8 = 0
    var prevPolicyMean: [Float]
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
    var health = HealthAccumulator()
    let actionCol: MeasuredActionCollector

    init(env: EnvPoolEnv, seed: Int?) {
        prevPolicyMean = Array(repeating: 0, count: env.dPolicy)
        prevScoreGrad  = Array(repeating: 0, count: env.dPolicy)
        randPolicyMean = Array(repeating: 0, count: env.dPolicy)
        randSeed = UInt64(bitPattern: Int64(seed ?? Int(oakUnseededSeed)))
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
        let warmup = max(ProcessInfo.processInfo.environment["OAK_WARMUP_STEPS"].flatMap(Int.init) ?? 0, 0)
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
            (_, metricsURL) = try runSingle(envId: envId, policy: p, seed: seed, warmup: warmup,
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
        let warmup = max(ProcessInfo.processInfo.environment["OAK_DIAG_WARMUP_STEPS"].flatMap(Int.init) ?? 0, 0)
        let device = try metalDevice()
        let steps = max(stopAfter
                        ?? ProcessInfo.processInfo.environment["OAK_DIAG_STEPS"].flatMap(Int.init)
                        ?? 1, 1)
        let (_, metricsURL) = try runSingle(
            envId: resolved, policy: policy, seed: seed, warmup: warmup,
            stopAfter: steps, device: device, command: "diagnose", includeDiagnostic: true)
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
        let warmup = max(ProcessInfo.processInfo.environment["OAK_BENCH_WARMUP_STEPS"].flatMap(Int.init) ?? 1000, 0)
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
                                      warmup: warmup, stopAfter: stopAfter,
                                      device: device, command: "benchmark", metricsDir: benchDir).0
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

    private static func setupRun(envId: String, policy: Policy, seed: Int?, warmup: Int,
                                 stopAfter: Int, device: MTLDevice, command: String,
                                 includeDiagnostic: Bool, metricsDir: URL?) throws -> RunContext {
        let dir = RunSnapshot.resolveMetricsDirectory(explicit: metricsDir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        fputs("OaK metrics \(dir.path)\n", stderr); fflush(stderr)
        let env = try EnvPoolEnv(id: envId, numEnvs: 1, seed: seed)
        let engine: OakMetalEngine? = policy == .oak ? try OakMetalEngine(
            obsDim: env.obsDim, baseObsDim: env.obsDim, actDim: env.actDim,
            rank: env.obsDim, dPolicy: env.dPolicy,
            device: device, seed: seed, diagnosticGpuPhaseTiming: includeDiagnostic) : nil
        try env.reset()
        return RunContext(envId: envId, policy: policy, seed: seed, warmup: warmup,
                          stopAfter: stopAfter, command: command,
                          includeDiagnostic: includeDiagnostic, metricsDir: dir,
                          env: env, engine: engine)
    }

    private static func emitSnapshot(_ ctx: RunContext, _ state: RunState, seq: Int?) -> RunMetricsFile {
        let measured = max(state.measuredSteps, 1)
        let a = state.agentMsSum / 1000, e = state.envMsSum / 1000
        let h = max(state.measuredSeconds - a - e, 0)
        let merged = state.health.diagnosticSnapshot().map { b in
            let (mm, mi) = ctx.engine?.metalPhaseGpuDiagnostics(measuredSteps: measured) ?? (nil, nil)
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
        let (dA, cA) = state.actionCol.partialMetrics()
        let sps = state.measuredSeconds > 0 ? Double(state.measuredSteps) / state.measuredSeconds : nil
        let reason = seq == nil ? "measured_steps" : "heartbeat"
        return RunMetricsFile(
            schemaVersion: 3, command: ctx.command, envId: ctx.envId, policy: ctx.policy, seed: ctx.seed,
            numEnvs: 1, warmupSteps: ctx.warmup, stopAfterMeasuredSteps: ctx.stopAfter,
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
        while state.measuredSteps < ctx.stopAfter {
            let measured = state.totalSteps + 1 > ctx.warmup && state.totalSteps > 0
            if measured && state.measurementStart == nil {
                state.measurementStart = DispatchTime.now().uptimeNanoseconds
            }
            let policyMean: PolicyMeanBatch, agentMs: Double, stats: KernelStats?
            if let engine = ctx.engine {
                let out = try ctx.env.observationScratch.withUnsafeBufferPointer { buf in
                    try engine.step(obs: buf, prevPolicyMean: state.prevPolicyMean,
                                    prevScoreGrad: state.prevScoreGrad,
                                    prevReward: state.prevReward, prevDone: state.prevDone,
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
                if let s = stats { state.health.record(s) }
                state.actionCol.addMeasured(envAction: ctx.env.envActionScratch)
            }
            state.prevReward = ctx.env.rewardScratch[0]
            state.prevDone   = ctx.env.doneScratch[0]
            memcpy(&state.prevPolicyMean, policyMean.values, policyMean.values.count * MemoryLayout<Float>.stride)
            memcpy(&state.prevScoreGrad, ctx.env.scoreGradScratch,
                   ctx.env.scoreGradScratch.count * MemoryLayout<Float>.stride)
        }
    }

    private static func runSingle(envId: String, policy: Policy, seed: Int?, warmup: Int,
                                  stopAfter: Int, device: MTLDevice, command: String,
                                  includeDiagnostic: Bool = false, metricsDir: URL? = nil)
                                  throws -> (RunMetricsFile, URL) {
        let ctx = try setupRun(envId: envId, policy: policy, seed: seed, warmup: warmup,
                               stopAfter: stopAfter, device: device, command: command,
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
