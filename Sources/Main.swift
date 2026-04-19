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
    // Pass 15 Tier 1.2: `activeRank` retained (= activeCount value) for backwards
    // compatibility with Pass 12-14 reports. New canonical names (`activeCount`,
    // `pEffectiveRank`) carry the semantically correct quantities and should be used
    // by all new analysis scripts.
    let activeRank: Double?            // legacy alias for activeCount (count of admitted features)
    let activeCount: Double?           // Pass 15 T1.2: explicit count of admitted features
    let pEffectiveRank: Double?        // Pass 15 T1.2: real effective rank (trace/max of featSM)
    let actionDeltaL2Mean: Double?
    let scoreGradL2Mean: Double?      // mean ‖∇log π(a|s)‖² across rows × params (REINFORCE)
    let optionModelPredictionMSE: Double?
    let oakOptionReinitCount: Double?
    let nonFiniteRawTotal: Int
    let nonFiniteSanitizedTotal: Int
    // Pass 15 Tier 0: numerical-safety telemetry surfaced into the bench JSON. The (B)
    // hard rule is `tdPsiNonFiniteTotal == 0 && halfSaturationTotalFinal == 0` over the
    // full bench. Both default to 0 for older schemas (Codable optionality + default).
    let tdPsiNonFiniteTotal: Int?
    let halfSaturationTotalFinal: Int?
    // Pass 15 Tier 1: surfaced KernelScalars + per-buffer weight-magnitude maxima. The (C)
    // bounded-magnitude hard rule consumes `*Max` fields. The KernelScalars surfacing
    // (modelErrEMA, tdBaseline, alphaImp, etc.) lets analysis scripts inspect the agent's
    // internal stability state without re-running with diagnostic flags.
    let utilityFeatureMean: Double?, utilityOptionMean: Double?
    let modelErrEMA: Double?, modelErrEMAPrev: Double?
    let tdBaseline: Double?, alphaImp: Double?, alphaEnc: Double?, alphaPi: Double?
    let rhoBaseline: Double?         // Pass 15 Tier 2: ϖ-RTD centering baseline (cumulative reward mean)
    let oakFeatureReinitCountFinal: Int?
    let encoderMax: Double?, valueWMax: Double?, psiMax: Double?, optPolicyMax: Double?
    let outcomeHeadMax: Double?, betaWMax: Double?, omegaWMax: Double?, sprDecoderMax: Double?
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
    // Pass 15 Tier 5: workerDepth = measuredStepCount / numEnvs (per-env trajectory length).
    // The canonical depth metric for the BWH-orthodox bench protocol — at B=1 (Pass 15
    // default), workerDepth = measuredStepCount; at B=2048 (Pass 14 batched protocol)
    // workerDepth = measuredStepCount / 2048, exposing the depth-vs-width trade-off that
    // Pass 14 forensic data showed was masked by the prior `measuredStepCount` field
    // (50K total steps gave only 24 worker-steps per env at B=2048; explained the broken
    // strict-gate verdicts at B>1).
    let workerDepth: Int
    let stepsPerSecond: Double?, heartbeatSequence: Int?
    let rewardSum: Double, doneCount: Int, stopReason: String
    let agentStepMsMean: Double, envStepMsMean: Double
    let discreteAction: DiscreteActionMetrics?, continuousAction: ContinuousActionMetrics?
    let health: RunHealth?, diagnostic: RunDiagnostic?
}

// Pass 11 Tier 1: agent emits ONLY policyMean[B · D_policy]. The wrapper layer samples per
// component and produces both the env-flat action and the matching scoreGrad.
//
//   `PolicyMeanBatch` carries the kernel-emitted μ vector + dPolicy width.
//   `ActionComponent`  describes one component of the env's action spec (Discrete N | Box A).
//   `sampleAndScore`   composes per-component sampling: Gumbel-max + categorical scoreGrad for
//                      Discrete (Williams 1992; Maddison 2014); structural-σ-1 Gaussian +
//                      score = z for Box. The wrapper-layer sampling distribution and the
//                      score function are mathematically matched, regardless of action-space
//                      type, fixing the Pass-10 unified-Gaussian-then-argmax score-function
//                      mismatch on discrete envs.
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

// Per-row per-component sampling: writes envActionScratch (env-flat) and scoreGradScratch
// (size = dPolicy). Pure structural splitmix64 + Box-Muller; no allocation on the hot path.
//   Discrete N component: Gumbel-max categorical sample of softmax(μ); score = 1{a=d} − π_d.
//   Box A component:      structural-σ-1 Gaussian sample μ + z, clamped to env bounds; score = z.
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
                // Gumbel-max sample over logits at policyMean[mBase + pIdx ..< pIdx + n].
                // Stable softmax: subtract max-logit; score = 1{a=d} − π_d sums to 0 per row.
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
                for d in 0..<n {
                    Z += exp(policyMean[mBase + pIdx + d] - maxLogit)
                }
                let invZ: Float = 1.0 / max(Z, .leastNormalMagnitude)
                for d in 0..<n {
                    let pi_d = exp(policyMean[mBase + pIdx + d] - maxLogit) * invZ
                    scoreGrad[mBase + pIdx + d] = ((d == bestIdx) ? 1.0 : 0.0) - pi_d
                }
                envAction[aBase + aIdx] = Float(bestIdx)   // env reads as integer index
                pIdx += n; aIdx += 1
            case .box(let a, let low, let high):
                for d in 0..<a {
                    let z = nextNormal(&rngState)
                    let mu = policyMean[mBase + pIdx + d]
                    let raw = mu + z   // structural σ = 1
                    let clamped = max(low[d], min(high[d], raw.isFinite ? raw : 0))
                    envAction[aBase + aIdx + d] = clamped
                    // Pass 14 Tier 0.7: REINFORCE score for Gaussian policy with σ=1 is
                    // ∂log π(a|s) / ∂μ = (a - μ) / σ² = (a - μ). When the env clamps `raw`
                    // to [low, high], the ACTUAL action is `clamped`, not μ+z. Using `z`
                    // (= raw - μ) was biased on Box envs that hit bounds (Pass 13 bug).
                    scoreGrad[mBase + pIdx + d] = clamped - mu
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

// Structural neutral seed: integer 0. Used only when the caller does not supply an explicit
// seed for the splitmix32 hash that initializes parameter weights and derives kernel-side RNG.
// Under the Big World Hypothesis the agent's learning dynamics are invariant to seed in
// expectation; this constant just fixes "no seed" → "deterministic, reproducible" so two
// unseeded runs are bit-identical for debugging. No magic: 0 is the additive identity of the
// seedBase multiplication, the most neutral value possible.
let oakUnseededSeed: Int32 = 0

// Pass 12 Tier 1: env-var-driven kernel ablation. OAK_ABLATE_KERNEL=name1,name2 replaces the
// named kernel(s) with oak_noop in the ICB. Used by run/benchmark; probe also accepts
// --ablate-kernel=NAME on the CLI. Empty / unset = no ablation (baseline).
func metalDevice() throws -> MTLDevice {
    guard let d = MTLCreateSystemDefaultDevice() else {
        throw OakError.message("Metal device unavailable.")
    }
    return d
}

func ablatedKernelsFromEnv() -> Set<String> {
    guard let raw = ProcessInfo.processInfo.environment["OAK_ABLATE_KERNEL"]?
                            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty
    else { return [] }
    return Set(raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                                           .filter { !$0.isEmpty })
}

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

// ------------------------------------------------------------------------------------------------
// Health accumulator — lean fold over KernelStats. Six slots, one per reported mean.
// ------------------------------------------------------------------------------------------------
private struct HealthAccumulator {
    struct Slot { var sum = 0.0; var count = 0 }
    // Slot order must mirror `values` in `record()` and the field order in `snapshot()`.
    // Pass 15 Tier 1: extended to 16 slots covering all surfaced telemetry channels.
    //   0  latentStateMeanSquare
    //   1  activeCount             (formerly activeRank — the same quantity, semantically renamed)
    //   2  actionDeltaL2Mean
    //   3  scoreGradL2Mean
    //   4  optionModelPredictionMSE
    //   5  oakOptionReinitCount
    //   6  utilityFeatureMean      (T1.1: was written by GPU but never consumed by host)
    //   7  utilityOptionMean       (T1.1: same)
    //   8  pEffectiveRank          (T1.2: real effective rank from KernelScalars.pEff)
    //   9  modelErrEMA             (T1.3: surfaced from KernelScalars)
    //  10  modelErrEMAPrev         (T1.3)
    //  11  tdBaseline              (T1.3)
    //  12  alphaImp                (T1.3)
    //  13  alphaEnc                (T1.3)
    //  14  alphaPi                 (T1.3 — Pass 15 T0.3 ‖scoreGrad‖² implicit step)
    //  15  rhoBaseline             (T2  — ϖ-RTD baseline; Sakha-Kamalapurkar-Meyn 2603.27874v2)
    var slots = [Slot](repeating: Slot(), count: 16)
    var steps = 0
    var rawNonFinite = 0, sanitizedNonFinite = 0
    var encodeNF = 0, learnNF = 0, planNF = 0
    var maxLat = 0.0, maxPol = 0.0
    var lastRho: Double?
    // Pass 15 Tier 0: numerical-safety telemetry. tdPsiNF accumulates per-step counts
    // (an unbiased total over the run); halfSaturationCountFinal is read from the LAST
    // step's stats since the underlying counter is cumulative life-of-engine and the
    // delta over the bench window equals the final value (counter starts at 0 per init).
    var tdPsiNF = 0
    var halfSaturationFinal = 0
    var oakFeatureReinitFinal = 0
    // Pass 15 Tier 1.4: per-buffer weight-magnitude max over the bench window. Each is
    // the max-of-maxima across all recorded steps — a monotone signal for the (C)
    // bounded-magnitude gate. `*Max` reflects the worst-case at any point during the bench.
    var encoderMax = 0.0, valueWMax = 0.0, psiMax = 0.0, optPolicyMax = 0.0
    var outcomeHeadMax = 0.0, betaWMax = 0.0, omegaWMax = 0.0, sprDecoderMax = 0.0

    mutating func record(_ stats: KernelStats) {
        steps += 1
        let rr = Double(stats.rewardRate); if rr.isFinite { lastRho = rr }
        let values: [Float] = [
            stats.latentStateMeanSquare,
            stats.activeCount,
            stats.actionDeltaL2Mean,
            stats.scoreGradL2Mean,
            stats.optionModelPredictionMSE,
            stats.oakOptionReinitCount,
            stats.utilityFeatureMean,
            stats.utilityOptionMean,
            stats.pEffectiveRank,
            stats.modelErrEMA,
            stats.modelErrEMAPrev,
            stats.tdBaselineStat,
            stats.alphaImpStat,
            stats.alphaEncStat,
            stats.alphaPiStat,
            stats.rhoBaselineStat,
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
        tdPsiNF += Int(stats.tdPsiNonFiniteCount)
        halfSaturationFinal = Int(stats.halfSaturationCount)
        oakFeatureReinitFinal = Int(stats.oakFeatureReinitCount)
        maxLat = max(maxLat, Double(stats.maxAbsLatentPrediction))
        maxPol = max(maxPol, Double(stats.maxAbsPolicyOutput))
        encoderMax     = max(encoderMax,     Double(stats.encoderMax))
        valueWMax      = max(valueWMax,      Double(stats.valueWMax))
        psiMax         = max(psiMax,         Double(stats.psiMax))
        optPolicyMax   = max(optPolicyMax,   Double(stats.optPolicyMax))
        outcomeHeadMax = max(outcomeHeadMax, Double(stats.outcomeHeadMax))
        betaWMax       = max(betaWMax,       Double(stats.betaWMax))
        omegaWMax      = max(omegaWMax,      Double(stats.omegaWMax))
        sprDecoderMax  = max(sprDecoderMax,  Double(stats.sprDecoderMax))
    }

    private func mean(_ i: Int) -> Double? { slots[i].count > 0 ? slots[i].sum / Double(slots[i].count) : nil }

    func snapshot() -> RunHealth? {
        guard steps > 0 else { return nil }
        return RunHealth(
            rewardRate: lastRho,
            latentStateMeanSquare: mean(0),
            activeRank: mean(1),
            activeCount: mean(1),
            pEffectiveRank: mean(8),
            actionDeltaL2Mean: mean(2),
            scoreGradL2Mean: mean(3),
            optionModelPredictionMSE: mean(4),
            oakOptionReinitCount: mean(5),
            nonFiniteRawTotal: rawNonFinite,
            nonFiniteSanitizedTotal: sanitizedNonFinite,
            tdPsiNonFiniteTotal: tdPsiNF,
            halfSaturationTotalFinal: halfSaturationFinal,
            utilityFeatureMean: mean(6),
            utilityOptionMean: mean(7),
            modelErrEMA: mean(9),
            modelErrEMAPrev: mean(10),
            tdBaseline: mean(11),
            alphaImp: mean(12),
            alphaEnc: mean(13),
            alphaPi: mean(14),
            rhoBaseline: mean(15),
            oakFeatureReinitCountFinal: oakFeatureReinitFinal,
            encoderMax: encoderMax > 0 ? encoderMax : nil,
            valueWMax: valueWMax > 0 ? valueWMax : nil,
            psiMax: psiMax > 0 ? psiMax : nil,
            optPolicyMax: optPolicyMax > 0 ? optPolicyMax : nil,
            outcomeHeadMax: outcomeHeadMax > 0 ? outcomeHeadMax : nil,
            betaWMax: betaWMax > 0 ? betaWMax : nil,
            omegaWMax: omegaWMax > 0 ? omegaWMax : nil,
            sprDecoderMax: sprDecoderMax > 0 ? sprDecoderMax : nil)
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
    private let envActionWidth: Int

    // Pass 11 Tier 1: consumes env-flat action vectors (the wrapper-sampled actions actually
    // sent to the env). For discrete envs the env-flat width is 1 (the chosen index); for
    // continuous it is actDim. The agent's policyMean is never seen here — telemetry tracks
    // what the env actually received, which is the BWH-orthodox observable.
    init(discreteCount: Int?, envActionWidth: Int) {
        self.envActionWidth = envActionWidth
        if let c = discreteCount { discrete = [Int](repeating: 0, count: c) }
        else { sum = [Double](repeating: 0, count: envActionWidth); sumsq = [Double](repeating: 0, count: envActionWidth) }
    }

    func addMeasured(envAction: [Float], numEnvs: Int) {
        if var h = discrete {
            for row in 0..<numEnvs {
                let idx = max(0, min(h.count - 1, Int(envAction[row * envActionWidth])))
                h[idx] &+= 1
            }
            discrete = h
        }
        guard var s = sum, var q = sumsq else { return }
        for row in 0..<numEnvs {
            var l = 0.0
            let b = row * envActionWidth
            for d in 0..<envActionWidth { let v = Double(envAction[b + d]); s[d] += v; q[d] += v * v; l += v * v }
            let r = sqrt(l); l2s += r; l2sq += r * r; n += 1
        }
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


// ================================================================================================
// Kernel config + scalars + stats — mirrors OakKernel.metal structs byte-for-byte. Keep in lockstep.
// Pass 10 unification removed env-type fields: actionIsDiscrete and actionParamCount are gone
// from KernelConfig — the agent has no notion of action-space type, and D = 2·actDim is
// computed structurally in the kernel from cfg.actDim. boundLat/Rew/Act + invBoundLat/Rew/Act
// were also already gone from Tier 9 (Tier 2 deleted all clamps; implicit α + ρ̄ centering
// replaced them). KernelScalars dropped doneCount + _pad (host-side state.doneCount tracks it).
// KernelStats has 16 Float fields.
// ================================================================================================
private struct KernelConfig {
    var obsDim: UInt32 = 0, rank: UInt32 = 0, actDim: UInt32 = 0, batch: UInt32 = 0
    var optionCount: UInt32 = 0, baseObsDim: UInt32 = 0
    // Pass 11 Tier 1: env-declared D_policy. Sum of per-component policy-param widths from the
    // env's action spec (Discrete-N → N; Box-A → A). The agent has no notion of WHAT the
    // components are — only the integer width.
    var actionParamCount: UInt32 = 0
    // Pass 11 Tier 2: outcomeHead[K · E] for OPSR state augmentation (Vereecken 2604.07016).
    // Pass 11 Tier 3: betaW[E + K] termination head, omegaW[K · E] chord generator,
    // chord/optionAge/betaProb arena slots for temporal abstraction (Bacon-Precup 1609.05140;
    // Wan-Naik-Sutton 2110.13855).
    var pEncoder: UInt32 = 0, pValue: UInt32 = 0, pPsi: UInt32 = 0, pOptPolicy: UInt32 = 0
    var pOutcomeHead: UInt32 = 0, pBetaW: UInt32 = 0, pOmegaW: UInt32 = 0
    // Pass 13 Tier 3: SPR-Stream chord-conditioned diagonal next-state predictor.
    var pSprDecoder: UInt32 = 0
    var sObservation: UInt32 = 0, sObsMean: UInt32 = 0, sState: UInt32 = 0
    var sPrevPolicyMean: UInt32 = 0, sPolicyMean: UInt32 = 0, sScoreGrad: UInt32 = 0
    var sReward: UInt32 = 0, sTd: UInt32 = 0, sTdPsi: UInt32 = 0, sVCurr: UInt32 = 0
    var sChordSlot: UInt32 = 0, sChordSlotPrev: UInt32 = 0, sBetaProb: UInt32 = 0
    // Pass 14 Tier 1: sSprPred removed — oak_spr_fused recomputes ẑ_N inline.
    var uDone: UInt32 = 0, uActiveMask: UInt32 = 0, uActiveList: UInt32 = 0
    var uOption: UInt32 = 0, uCounters: UInt32 = 0, uSubtaskFeatureOf: UInt32 = 0
    var uOptionAge: UInt32 = 0
    var aFeatSM: UInt32 = 0, aWBar: UInt32 = 0, aFeatMean: UInt32 = 0
    var aUtilFeat: UInt32 = 0, aUtilOpt: UInt32 = 0
    // Pass 11 Tier 5: optFracEMA[K] for the structural-1/K dormancy reinit gate.
    var aOptFracEMA: UInt32 = 0
    // Pass 13 Tier 2: per-feature TIDBD α (SwiftSarsa 2507.19539 + MetaOptimize 2402.02342).
    var aAlphaImpFeat: UInt32 = 0, aAlphaImpBeta: UInt32 = 0, aAlphaImpH: UInt32 = 0
    // Pass 15 Tier 2: per-cumulant ϖ-RTD ψ-centering baseline (Sakha 2603.27874v2 §1).
    var aPsiMeanBaseline: UInt32 = 0
    // Pass 15 Tier 4: TD(λ) eligibility traces for value head + ψ.
    var aValueE: UInt32 = 0, aPsiE: UInt32 = 0
    var cap: Float = 0, alpha0: Float = 0, ridge: Float = 0
}

private struct KernelScalars {
    var stepCount: UInt32 = 0, activeCount: UInt32 = 0
    var stateSlot: UInt32 = 0, optionSlot: UInt32 = 0
    var rho: Float = 0, pEff: Float = 0
    var alphaImp: Float = 0, alphaEnc: Float = 0
    // Pass 15 Tier 0.3: REINFORCE policy implicit step. α₀ / (1 + α₀ · ⟨‖x‖²⟩·⟨‖scoreGrad‖²⟩).
    // Computed in oak_reduce_alpha; consumed by oak_learn_pi as the shared denominator that
    // includes the full outer-product gradient norm (Cauchy-Schwarz upper bound). Fixes the
    // Pass-14 200M MSE blowup root cause (per-thread n2 missed cross-row scoreGrad²).
    var alphaPi: Float = 0
    // Pass 11 Tier 4: REINFORCE running-mean TD baseline (Williams 1992 variance reduction).
    var tdBaseline: Float = 0
    // Pass 12 Tier 4.2/4.3 + Pass 13 Tier 0: continual EMA of model-prediction error
    // (= optionModelPredictionMSE) plus a SLOW snapshot lagged by ~N steps. The Pass-13
    // gate uses lpmRate = max(0, modelErrEMAPrev - modelErrEMA) as the information-gain
    // proxy (Hou-An-Du 2509.25438 ICLR 2026 LPM second-derivative). Replaces Pass-12's
    // absolute-scale gate which over-shrunk the planning bonus on stochastic envs.
    var modelErrEMA: Float = 0
    var modelErrEMAPrev: Float = 0
    // Pass 13 Tier 6.2: dense-fast-path flag (1 iff activeCount == E and activeList[i] == i for all i).
    // Set by oak_bk_admit_commit; consumed by active_dot_half_auto in the next step's kernels.
    var activeListIsIdentity: UInt32 = 0
    // Pass 15 Tier 2: ϖ-RTD baseline (cumulative Welford mean of reward; no floor) +
    // <valueW, featMean> baseline (the expected V(s) under empirical state distribution).
    // Sakha-Kamalapurkar-Meyn 2603.27874v2 Theorem 2.4(ii) — first published linear-FA
    // stability proof for relative-TD. The full Theorem requires BOTH terms in the TD
    // centering (per research-report §11); the rhoBaseline-only form is insufficient
    // (encoder · obs grows unbounded → V(s) grows unbounded → weights saturate). Both
    // are computed in oak_bk_admit_commit and consumed by oak_predict.
    var rhoBaseline: Float = 0
    var vMeanBaseline: Float = 0
    // Pass 15 Tier 4: TD(λ) eligibility-trace decay rate (Sutton-Barto Ch 12.7). Set in
    // oak_bk_admit_commit each step from λ = 1 − 1/cap (structural, Inv-9). Recovers
    // variance reduction at B=1 by accumulating gradient depth INSIDE the TD target.
    var lambdaTrace: Float = 0
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
    // Pass 15 Tier 0: numerical-safety telemetry.
    //   tdPsiNonFiniteCount  per-step NF in tdPsi (was untracked, contaminated MSE)
    //   halfSaturationCount  cumulative life-of-engine count of safe_half(NaN/Inf) substitutions
    var tdPsiNonFiniteCount: Float = 0, halfSaturationCount: Float = 0
    // Pass 15 Tier 1: telemetry overhaul. Surfaced KernelScalars + cumulative counters
    // + per-buffer weight-magnitude maxima. Layout MIRRORS OakKernel.metal struct exactly.
    var pEffectiveRank: Float = 0
    var modelErrEMA: Float = 0, modelErrEMAPrev: Float = 0
    var tdBaselineStat: Float = 0
    var alphaImpStat: Float = 0, alphaEncStat: Float = 0, alphaPiStat: Float = 0
    var rhoBaselineStat: Float = 0
    var oakFeatureReinitCount: Float = 0
    var encoderMax: Float = 0, valueWMax: Float = 0, psiMax: Float = 0, optPolicyMax: Float = 0
    var outcomeHeadMax: Float = 0, betaWMax: Float = 0, omegaWMax: Float = 0, sprDecoderMax: Float = 0
}

// ================================================================================================
// Arena layout — single source of truth (Metal reads the offsets from KernelConfig).
// ================================================================================================
private enum OakMetalLayout {
    static let kernelConfigBufferIndex = 5
    // Structural cap: one SIMD group of options per row in oak_plan (hardware-derived, not tuned).
    static let maxOptionCount = 32

    static func optionCount(rank: Int, actDim: Int) -> Int {
        max(1, min(rank, actDim, maxOptionCount))
    }
    // Pass 11 Tier 2: extended-state width E = R + K (encoder output + OPSR outcome dims).
    // All linear heads (value, ψ, π, outcomeHead) span the full E-dim state.
    static func extendedDim(rank: Int, actDim: Int) -> Int {
        rank + optionCount(rank: rank, actDim: actDim)
    }
    // Concat dim M = E + K: shared chord-conditional ψ(s, λ) = W_ψ · concat[s_E, λ] (Alberta
    // Plan Design 2; OBB 2505.00787 §2.4 Eq.4). State weights span the OPSR-extended E.
    static func concatDim(rank: Int, actDim: Int) -> Int {
        extendedDim(rank: rank, actDim: actDim) + optionCount(rank: rank, actDim: actDim)
    }
    static func psiCount(rank: Int, actDim: Int) -> Int {
        optionCount(rank: rank, actDim: actDim) * concatDim(rank: rank, actDim: actDim)
    }
    // Pass 11 Tier 1: D_policy is env-declared. Pass 11 Tier 2: optPolicy spans E (not R).
    static func optPolicyCount(rank: Int, actDim: Int, dPolicy: Int) -> Int {
        optionCount(rank: rank, actDim: actDim) * dPolicy * extendedDim(rank: rank, actDim: actDim)
    }
    // Pass 11 Tier 2: outcomeHead[K · E] — one E-dim row per outcome dimension (Vereecken 2604.07016).
    static func outcomeHeadCount(rank: Int, actDim: Int) -> Int {
        optionCount(rank: rank, actDim: actDim) * extendedDim(rank: rank, actDim: actDim)
    }
    // Pass 11 Tier 3: termination head betaW[E + K] (state weights + chord weights);
    // chord generator omegaW[K · E] (one row per option's chord-logit producer).
    static func betaWCount(rank: Int, actDim: Int) -> Int {
        extendedDim(rank: rank, actDim: actDim) + optionCount(rank: rank, actDim: actDim)
    }
    static func omegaWCount(rank: Int, actDim: Int) -> Int {
        optionCount(rank: rank, actDim: actDim) * extendedDim(rank: rank, actDim: actDim)
    }
    // Pass 13 Tier 3: SPR-Stream chord-conditioned diagonal next-state predictor (Nilaksh
    // et al. 2602.09396 ICML 2026). sprDecoder[k][e] models the per-option transition gain
    // applied to z_{N-1}[e] to predict z_N[e]. K · R fp16 ≈ 2 KB at K=32, R=64.
    static func sprDecoderCount(rank: Int, actDim: Int) -> Int {
        optionCount(rank: rank, actDim: actDim) * rank
    }

    struct Param { let encoder, value, psi, optPolicy, outcomeHead, betaW, omegaW, sprDecoder, total: Int }
    static func paramLayout(obsDim: Int, rank: Int, actDim: Int, dPolicy: Int) -> Param {
        let E = extendedDim(rank: rank, actDim: actDim)
        var o = 0
        let enc = o; o += rank                    // encoder: R weights (obs → state[:R])
        let val = o; o += E                       // valueW: E weights (over OPSR-extended state)
        let psi = o; o += psiCount(rank: rank, actDim: actDim)
        let op  = o; o += optPolicyCount(rank: rank, actDim: actDim, dPolicy: dPolicy)
        let oh  = o; o += outcomeHeadCount(rank: rank, actDim: actDim)
        let bw  = o; o += betaWCount(rank: rank, actDim: actDim)
        let ow  = o; o += omegaWCount(rank: rank, actDim: actDim)
        let sd  = o; o += sprDecoderCount(rank: rank, actDim: actDim)
        return Param(encoder: enc, value: val, psi: psi, optPolicy: op, outcomeHead: oh,
                     betaW: bw, omegaW: ow, sprDecoder: sd, total: o)
    }

    struct State {
        let observation, obsMean, state, prevPolicyMean, policyMean, scoreGrad: Int
        let reward, td, tdPsi, vCurr: Int
        let chordSlot, chordSlotPrev, betaProb: Int
        let featSM, wBar, featMean, utilFeat, utilOpt, optFracEMA: Int
        // Pass 13 Tier 2: per-feature TIDBD α (SwiftSarsa 2507.19539 + MetaOptimize 2402.02342).
        // alphaImpFeat[E] = exp(beta_i) / (1 + exp(beta_i) · featSM[i]) per-feature implicit step.
        // alphaImpBeta[E] = log-step meta-parameter; alphaImpH[E] = decay trace of past Δw.
        let alphaImpFeat, alphaImpBeta, alphaImpH: Int
        // Pass 14 Tier 1: sprPred[B·R] arena slot REMOVED (the fused oak_spr_fused kernel
        // recomputes ẑ_N inline per (row, feat) and never materializes it).
        // Pass 15 Tier 2: psiMeanBaseline[K] = <ψ_k, featMean> per cumulant — the ϖ-RTD
        // centering baseline read by oak_predict; updated in oak_bk_admit_commit.
        let psiMeanBaseline: Int
        // Pass 15 Tier 4: TD(λ) eligibility traces. valueE[E] for w_main; psiE[K · M] for ψ.
        // Engine-init zero; in-place updated by oak_learn_value / oak_learn_psi at B=1.
        let valueE, psiE: Int
        let total: Int
    }
    static func stateLayout(obsDim: Int, rank: Int, actDim: Int, batch: Int, dPolicy: Int) -> State {
        let K = optionCount(rank: rank, actDim: actDim)
        let E = extendedDim(rank: rank, actDim: actDim)
        var o = 0
        let obs = o; o += batch * obsDim
        let mean = o; o += obsDim
        let st = o; o += 2 * batch * E      // OPSR-extended state (E = R + K)
        let prP = o; o += batch * dPolicy   // prev step's policyMean (host-written, telemetry)
        let pol = o; o += batch * dPolicy   // current step's policyMean (kernel-written)
        let sg  = o; o += batch * dPolicy   // host-written ∇log π for the wrapper-sampled action
        let rew = o; o += batch
        let td  = o; o += batch
        let tdP = o; o += batch * K         // K cumulant TDs per row (Design 2 collapse)
        let vc  = o; o += batch             // vCurr = w_main · curr_state (cached in predict, read by plan)
        // Pass 11 Tier 3: temporal abstraction state — persistent soft chord per row.
        let cs  = o; o += batch * K         // current step's chord (∈ Δ^K simplex per row)
        let csP = o; o += batch * K         // prev step's chord (for learn_psi/_pi attribution)
        let bp  = o; o += batch             // β output per row (termination probability telemetry)
        // featSM/wBar/featMean/utilFeat are encoder-only stats (cumulant ranking + plan optimism).
        let fs  = o; o += rank
        let wb  = o; o += rank
        let fm  = o; o += rank
        let uf  = o; o += rank
        let uo  = o; o += K
        let ofE = o; o += K          // Pass 11 Tier 5: soft-chord per-option mass EMA
        let aif = o; o += E          // Pass 13 Tier 2: alphaImpFeat[E] (per-feature implicit step)
        let aib = o; o += E          // alphaImpBeta[E] = log meta-parameter, init log(alpha0)
        let aih = o; o += E          // alphaImpH[E] = decay trace of past Δw, init 0
        let pmb = o; o += K          // Pass 15 Tier 2: psiMeanBaseline[K] = <ψ_k, featMean>
        let vE  = o; o += E          // Pass 15 Tier 4: valueE[E] eligibility trace for w_main
        let pE  = o; o += K * (E + K)// Pass 15 Tier 4: psiE[K * M] eligibility traces for ψ (M = E + K)
        return State(observation: obs, obsMean: mean, state: st, prevPolicyMean: prP, policyMean: pol,
                     scoreGrad: sg, reward: rew, td: td, tdPsi: tdP, vCurr: vc,
                     chordSlot: cs, chordSlotPrev: csP, betaProb: bp,
                     featSM: fs, wBar: wb, featMean: fm, utilFeat: uf, utilOpt: uo,
                     optFracEMA: ofE,
                     alphaImpFeat: aif, alphaImpBeta: aib, alphaImpH: aih,
                     psiMeanBaseline: pmb,
                     valueE: vE, psiE: pE,
                     total: o)
    }

    struct UInts { let done, activeMask, activeList, option, counters, subtaskFeatureOf, optionAge, total: Int }
    static func uintLayout(rank: Int, actDim: Int, batch: Int) -> UInts {
        let K = optionCount(rank: rank, actDim: actDim)
        let E = extendedDim(rank: rank, actDim: actDim)
        var o = 0
        let d = o; o += batch
        let am = o; o += E      // active mask spans E (outcome dims R..E always-active)
        let al = o; o += E      // active list also sized E
        let op = o; o += 2 * batch     // legacy option_slot — argmax of chord, telemetry-only in Tier 3
        // Pass 15 Tier 0: counters[8] expanded from counters[6] to host two new
        // numerical-safety counters (OAK_CNT_HALF_SAT cumulative + OAK_CNT_TDPSI_NF
        // per-step). Layout matches OakKernel.metal OAK_CNT_* constants 0..7.
        let c  = o; o += 8
        let sf = o; o += K
        let oa = o; o += batch  // Pass 11 Tier 3: optionAge[r] — steps since last termination
        return UInts(done: d, activeMask: am, activeList: al, option: op, counters: c,
                     subtaskFeatureOf: sf, optionAge: oa, total: o)
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
// Metal engine — orthodox Step 11 OaK. Pass 15 Tier 1: 20 phases per step, dispatched as
// one ICB.
//   encode → predict_outcome → meta_policy → predict → plan → reduce_alpha
//   → learn_value → learn_encoder → spr_fused → learn_outcome → learn_psi → learn_pi
//   → learn_beta → learn_omega → meta_action
//   → bk_reduce → bk_feature → bk_option → bk_admit_commit → bk_weight_magnitudes
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
    // Pass 13 Tier 3: 20-kernel pipeline. SPR-Stream adds oak_predict_next_latent (after
    // meta_policy so it can read chordPrev = chord_{N-1} that drove the just-completed
    // transition) and oak_learn_encoder_spr (after oak_learn_encoder so the orthogonal
    // projection has the just-applied RL gradient available for de-correlation).
    private static let phases: [Phase] = [
        Phase(name: "oak_encode",             tgSize: 0),
        Phase(name: "oak_predict_outcome",    tgSize: 0),
        Phase(name: "oak_meta_policy",        tgSize: 0),
        // Pass 14 T1: oak_predict_next_latent + oak_learn_encoder_spr fused into oak_spr_fused.
        Phase(name: "oak_predict",            tgSize: 0),  // Pass 14 T2: per-thread dispatchThreads
        Phase(name: "oak_plan",               tgSize: 32),
        Phase(name: "oak_reduce_alpha",       tgSize: 128),
        Phase(name: "oak_learn_value",        tgSize: 32),
        Phase(name: "oak_learn_encoder",      tgSize: 0),
        Phase(name: "oak_spr_fused",          tgSize: 0),   // Pass 14 T1: fused predict + learn
        Phase(name: "oak_learn_outcome",      tgSize: 0),
        Phase(name: "oak_learn_psi",          tgSize: 0),
        Phase(name: "oak_learn_pi",           tgSize: 0),
        Phase(name: "oak_learn_beta",         tgSize: 0),
        Phase(name: "oak_learn_omega",        tgSize: 0),
        Phase(name: "oak_meta_action",        tgSize: 0),  // Pass 14 T0.5+T2 (REVISED): B*D
                                                            // dispatchThreads (no stack array,
                                                            // no lane waste, e-outer-k-inner)
        Phase(name: "oak_bk_reduce",          tgSize: 128),
        Phase(name: "oak_bk_feature",         tgSize: 32),
        Phase(name: "oak_bk_option",          tgSize: 128),
        Phase(name: "oak_bk_admit_commit",    tgSize: 0),
        // Pass 15 Tier 1.4: per-buffer max(|w|) for the (C) bounded-magnitude gate.
        // 8 TGs × 32 threads (one TG per fp16 weight buffer; simd_max reduction).
        Phase(name: "oak_bk_weight_magnitudes", tgSize: 32),
    ]

    private let obsDim, baseObsDim, actDim, batch, rank, dPolicy: Int
    private let queue: MTLCommandQueue
    private let cbDesc: MTLCommandBufferDescriptor
    private let pipelines: [MTLComputePipelineState]
    private let noopPipeline: MTLComputePipelineState   // Pass 12 Tier 1: ablation hot-swap target
    private let ablatedPhaseIndices: Set<Int>           // phase indices replaced with noop in the ICB
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
    // Metal 4 ICB + MTLResidencySet (macOS 26). Pre-encoded `phases.count` dispatches execute in
    // one encoder on the hot path, eliminating per-phase encoder setup overhead.
    private let indirectCommandBuffer: MTLIndirectCommandBuffer
    private let residencySet: MTLResidencySet
    private let resourceList: [MTLBuffer]

    private static func kernelURL() throws -> URL {
        try requiredBuildPath(".build/OakKernel.metallib",
                              description: "non-empty .build/OakKernel.metallib", nonEmpty: true)
    }

    private static func makePipelines(device: MTLDevice) throws -> (real: [MTLComputePipelineState], noop: MTLComputePipelineState) {
        let library = try device.makeLibrary(URL: try kernelURL())
        func make(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw OakError.message("Missing \(name) in OakKernel.metallib.")
            }
            let desc = MTLComputePipelineDescriptor()
            desc.computeFunction = fn
            desc.supportIndirectCommandBuffers = true
            return try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
        }
        let real = try phases.map { try make($0.name) }
        // Pass 12 Tier 1: oak_noop hot-swap target for per-kernel ablation.
        let noop = try make("oak_noop")
        return (real, noop)
    }

    // Apple GPUs have a fixed 32-thread SIMD width. Returning this constant avoids compiling all
    // pipelines just to read `threadExecutionWidth` from one of them.
    static func executionWidth(device: MTLDevice) throws -> Int { 32 }

    // Pass 11 Tier 1: agent constructor takes signal-shape parameters {obsDim, baseObsDim, actDim,
    // batch, rank, dPolicy}. dPolicy is the env-declared D_policy (sum of per-component policy-
    // param widths from the env's ActionSpec). The agent has no notion of WHAT the components
    // are — only the integer width. Inv-1 + Bitter Lesson.
    init(obsDim: Int, baseObsDim: Int, actDim: Int, batch: Int, rank: Int, dPolicy: Int,
         device: MTLDevice, seed: Int?, diagnosticGpuPhaseTiming: Bool = false,
         ablatedKernels: Set<String> = []) throws {
        precondition(obsDim == rank, "obsDim must equal rank (interface-derived latent width).")
        precondition(baseObsDim <= obsDim && baseObsDim >= 1, "baseObsDim must be positive and ≤ obsDim.")
        precondition(dPolicy >= 1, "dPolicy must be positive (env's action-spec param-count sum).")
        self.obsDim = obsDim; self.baseObsDim = baseObsDim; self.actDim = actDim
        self.batch = batch; self.rank = rank; self.dPolicy = dPolicy
        guard let queue = device.makeCommandQueue() else {
            throw OakError.message("Failed to create Metal command queue.")
        }
        self.queue = queue
        let d = MTLCommandBufferDescriptor(); d.retainedReferences = false
        cbDesc = d
        let built = try Self.makePipelines(device: device)
        pipelines = built.real
        noopPipeline = built.noop
        // Pass 12 Tier 1: resolve ablated kernel names to phase indices. Unknown names are warned
        // about (so a typo doesn't silently produce a baseline run). Per the multi-signal Pareto
        // gate, removing an "ablated" kernel from the ICB simulates "what if this kernel didn't run"
        // — the arena slot it would have written stays at its previous-step value.
        let phaseNames = Set(Self.phases.map { $0.name })
        for k in ablatedKernels where !phaseNames.contains(k) {
            fputs("OaK: warning — ablate-kernel '\(k)' is not a known phase; ignored.\n", stderr)
        }
        ablatedPhaseIndices = Set(Self.phases.enumerated()
            .filter { ablatedKernels.contains($0.element.name) }.map { $0.offset })
        if !ablatedPhaseIndices.isEmpty {
            let names = ablatedPhaseIndices.sorted().map { Self.phases[$0].name }.joined(separator: ", ")
            fputs("OaK: ablating kernels [\(names)] — replaced with oak_noop in the ICB.\n", stderr)
        }

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

        let paramLayout = OakMetalLayout.paramLayout(obsDim: obsDim, rank: rank, actDim: actDim,
                                                     dPolicy: dPolicy)
        stateLayout = OakMetalLayout.stateLayout(obsDim: obsDim, rank: rank, actDim: actDim,
                                                 batch: batch, dPolicy: dPolicy)
        uintLayout  = OakMetalLayout.uintLayout(rank: rank, actDim: actDim, batch: batch)
        paramsArena = Shared(device: device, count: paramLayout.total, zero: Float16(0))
        stateArena  = Shared(device: device, count: stateLayout.total, zero: Float(0))
        uintArena   = Shared(device: device, count: uintLayout.total,  zero: UInt32(0))

        // Cache per-phase grid width so there's no allocation on the hot path.
        let K = OakMetalLayout.optionCount(rank: rank, actDim: actDim)
        let E = OakMetalLayout.extendedDim(rank: rank, actDim: actDim)
        let M = OakMetalLayout.concatDim(rank: rank, actDim: actDim)
        // Pass 15 Tier 1.4: 20-kernel pipeline (was 19 at Pass 14 T1).
        // oak_bk_weight_magnitudes appended for the (C) bounded-magnitude gate.
        phaseGridWidths = [
            batch * rank,                     // oak_encode              [dispatchThreads]
            batch * K,                        // oak_predict_outcome     [dispatchThreads]
            batch,                            // oak_meta_policy         [dispatchThreads]
            batch,                            // oak_predict             [dispatchThreads]
            batch,                            // oak_plan                [B TGs × 32]
            1,                                // oak_reduce_alpha        [1 TG × 128]
            E,                                // oak_learn_value         [E TGs × 32]
            max(baseObsDim, 1),               // oak_learn_encoder       [dispatchThreads]
            max(baseObsDim, 1),               // oak_spr_fused           [dispatchThreads baseO]
            K * E,                            // oak_learn_outcome       [dispatchThreads]
            K * M,                            // oak_learn_psi           [dispatchThreads]
            K * dPolicy * E,                  // oak_learn_pi            [dispatchThreads]
            E + K,                            // oak_learn_beta          [dispatchThreads]
            K * E,                            // oak_learn_omega         [dispatchThreads]
            batch * dPolicy,                  // oak_meta_action         [dispatchThreads B*D]
            1,                                // oak_bk_reduce           [1 TG × 128]
            rank,                             // oak_bk_feature          [R TGs × 32]
            K,                                // oak_bk_option           [K TGs × 128]
            1,                                // oak_bk_admit_commit     [dispatchThreads]
            8,                                // oak_bk_weight_magnitudes [8 TGs × 32]
        ]

        self.diagnosticGpuPhaseTiming = diagnosticGpuPhaseTiming
        phaseGpuTimeSumSeconds = Array(repeating: 0, count: Self.phases.count)
        phaseGpuInvalidStepCount = Array(repeating: 0, count: Self.phases.count)
        hostAction = Array(repeating: 0, count: batch * dPolicy)

        // --- Metal 4 ICB + MTLResidencySet setup ----------------------------------------------
        // Pass 10 dropped the bounds buffer. 6 binds: params, stateArena, uintArena, scalars,
        // stats, config (config is at OakMetalLayout.kernelConfigBufferIndex = 5).
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
        // Pass 11 Tier 2: valueW has E weights (was R) — extends over the OPSR-augmented state.
        sample(paramLayout.value, OakMetalLayout.extendedDim(rank: rank, actDim: actDim), invR)
        sample(paramLayout.psi, OakMetalLayout.psiCount(rank: rank, actDim: actDim), invR)
        sample(paramLayout.optPolicy,
               OakMetalLayout.optPolicyCount(rank: rank, actDim: actDim, dPolicy: dPolicy), invR)
        sample(paramLayout.outcomeHead,
               OakMetalLayout.outcomeHeadCount(rank: rank, actDim: actDim), invR)
        sample(paramLayout.betaW,
               OakMetalLayout.betaWCount(rank: rank, actDim: actDim), invR)
        sample(paramLayout.omegaW,
               OakMetalLayout.omegaWCount(rank: rank, actDim: actDim), invR)
        // Pass 13 Tier 3: SPR-Stream decoder. Init at the unit-gain identity (every chord
        // predicts ẑ_N = z_{N-1}). The chord-conditioned linear predictor learns the actual
        // transition gains from the prediction errors.
        let sprDecoderCount = OakMetalLayout.sprDecoderCount(rank: rank, actDim: actDim)
        let sprBase = paramsArena.pointer.advanced(by: paramLayout.sprDecoder)
        for i in 0..<sprDecoderCount { sprBase[i] = Float16(1) }
        // ψ diagonal seed: cumulant k's W_ψ[k] · concat[s, λ] gets a +1/√R coefficient on its own
        // state feature index k (RRS feature-attainment initialization; Alberta Plan A.6 / OBB
        // §2.4). The first E weights of W_ψ[k] are state weights; index [k][k] is "predict
        // cumulant k mostly from feature k". No magic — invR is the structural per-feature scale.
        let psiPtr = paramsArena.pointer.advanced(by: paramLayout.psi)
        for k in 0..<K where k < rank {
            let idx = k * M + k
            psiPtr[idx] = Float16(Float(psiPtr[idx]) + invR)
        }

        // Pass 11 Tier 2: all encoder features (0..R) admitted on step 0; outcome dims R..E
        // are ALWAYS-ACTIVE by construction (they're agent-internal predictions, not env
        // features that need utility-ranked admission). bk_admit_commit only re-ranks [0, R).
        scalarsPointer.pointee.activeCount = UInt32(E)
        let activeMask = uintArena.pointer.advanced(by: uintLayout.activeMask)
        let activeList = uintArena.pointer.advanced(by: uintLayout.activeList)
        for f in 0..<rank { activeMask[f] = 1; activeList[f] = UInt32(f) }
        for k in 0..<K { activeMask[rank + k] = 1; activeList[rank + k] = UInt32(rank + k) }

        // Pass 11 Tier 3: seed chord uniformly = 1/K per option (the maximum-entropy prior).
        // optionAge = 0 forces meta_policy to resample on step 1 (so chord moves off uniform).
        let chordSlot = stateArena.pointer.advanced(by: stateLayout.chordSlot)
        let invK: Float = 1.0 / Float(max(K, 1))
        for r in 0..<batch {
            for k in 0..<K { chordSlot[r * K + k] = invK }
        }
        // Pass 11 Tier 5: seed optFracEMA = 1/K per option (the structural uniform-share, same
        // as the dormancy threshold's reference value). The gate fires when an option's mass
        // drops below 0.5/K, so this initial value is well above threshold.
        let optFrac = stateArena.pointer.advanced(by: stateLayout.optFracEMA)
        for k in 0..<K { optFrac[k] = invK }
        // Pass 13 Tier 2: per-feature TIDBD α (SwiftSarsa 2507.19539 + MetaOptimize 2402.02342).
        // Init β_i = log(α₀) so exp(β_i) = α₀ at step 0 — recovers Pass-12 global-α behavior
        // before the meta-update has had any data to act on. h_i = 0 (no history yet).
        // alphaImpFeat[i] = α₀ initially (overwritten by oak_reduce_alpha on step 1).
        let extDim = rank + K   // E = R + K (OPSR-extended state)
        let initAlpha0: Float = 1.0 / Float(max(rank * max(batch, 1), 1))
        let logAlpha0 = Float(log(Double(initAlpha0)))
        let aifPtr = stateArena.pointer.advanced(by: stateLayout.alphaImpFeat)
        let aibPtr = stateArena.pointer.advanced(by: stateLayout.alphaImpBeta)
        let aihPtr = stateArena.pointer.advanced(by: stateLayout.alphaImpH)
        for i in 0..<extDim {
            aifPtr[i] = initAlpha0
            aibPtr[i] = logAlpha0
            aihPtr[i] = 0
        }
        // optionAge starts at 0 (force termination); chordSlotPrev/betaProb start at 0.
        // Initialize subtaskFeatureOf to identity sf[k] = k. bk_admit_commit re-ranks each step
        // by per-feature utility (Tier 7 RED dynamic top-K cumulants).
        let sfPtr = uintArena.pointer.advanced(by: uintLayout.subtaskFeatureOf)
        for k in 0..<K { sfPtr[k] = UInt32(min(k, rank - 1)) }

        writeKernelConfig(paramLayout: paramLayout)

        // Pre-encode the phase dispatch graph into the ICB (`phases.count` phases; Pass 14 = 20).
        // Grids are fixed for the engine's lifetime; only KernelScalars/KernelStats/params/state
        // data changes between steps. resourceList is laid out
        // [0..4 = params, state, uint, scalars, stats; 5 = config].
        let buffers = resourceList
        for i in 0..<Self.phases.count {
            let cmd = icb.indirectComputeCommandAt(i)
            // Pass 12 Tier 1: hot-swap the noop pipeline for ablated phases. Same buffer
            // bindings + same dispatch geometry — the kernel just returns immediately, so
            // the arena slot the ablated kernel would have written stays at its previous
            // step's value (which is what "what if this kernel didn't run" means).
            let isAblated = ablatedPhaseIndices.contains(i)
            let pip = isAblated ? noopPipeline : pipelines[i]
            cmd.setComputePipelineState(pip)
            for j in 0..<buffers.count { cmd.setKernelBuffer(buffers[j], offset: 0, at: j) }
            let tgSize = Self.phases[i].tgSize
            if tgSize > 0 {
                let tgCount = max(phaseGridWidths[i], 1)
                cmd.concurrentDispatchThreadgroups(
                    MTLSize(width: tgCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            } else {
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

    // ---- Probe-only accessor (Tier-0 verification harness; deleted in Tier 7) ----------------
    // Pass 15 Tier 5: probeSlot deleted (~85 LOC). Probe harness was used for kernel-
    // correctness validation in Pass 11-13; that role is now covered by the (B) numerical-
    // safety hard gate (nonFiniteRawTotal == 0 + halfSaturationTotalFinal == 0 over the
    // full bench window) and the (C) bounded-magnitude hard gate (per-buffer max(|w|) <
    // √HALF_MAX), both surfaced into RunHealth and verifiable from any bench JSON.

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
    // Pass 11 Tier 1: takes prevPolicyMean[B·D] + prevScoreGrad[B·D] (host-written scoreGrad
    // for last step's wrapper-sampled action), returns this step's policyMean[B·D]. The wrapper
    // (EnvPoolEnv.sampleAndStep) then samples per-component and produces (action, scoreGrad).
    func step(obs: UnsafeBufferPointer<Float>, prevPolicyMean: [Float],
              prevScoreGrad: [Float], prevReward: [Float], prevDone: [UInt8],
              accumulatePhaseGpu: Bool) throws -> (PolicyMeanBatch, Double, KernelStats) {
        let start = DispatchTime.now().uptimeNanoseconds

        precondition(obs.count == batch * obsDim)
        precondition(prevReward.count == batch)
        precondition(prevPolicyMean.count == batch * dPolicy)
        precondition(prevScoreGrad.count == batch * dPolicy)
        memcpy(stateArena.pointer.advanced(by: stateLayout.observation), obs.baseAddress!,
               batch * obsDim * MemoryLayout<Float>.stride)
        memcpy(stateArena.pointer.advanced(by: stateLayout.reward), prevReward,
               batch * MemoryLayout<Float>.stride)
        memcpy(stateArena.pointer.advanced(by: stateLayout.prevPolicyMean), prevPolicyMean,
               batch * dPolicy * MemoryLayout<Float>.stride)
        memcpy(stateArena.pointer.advanced(by: stateLayout.scoreGrad), prevScoreGrad,
               batch * dPolicy * MemoryLayout<Float>.stride)
        let donePtr = uintArena.pointer.advanced(by: uintLayout.done)
        for i in 0..<batch { donePtr[i] = UInt32(prevDone[i]) }

        let ctr = uintArena.pointer.advanced(by: uintLayout.counters)
        // Reset per-step non-finite counters (offsets OAK_CNT_OBS_NF=0, LEARN_NF=1,
        // PLAN_NF=2, TDPSI_NF=6). counters[OAK_CNT_REINIT_OPT=3] is cleared by
        // bk_admit_commit at end of step. counters[OAK_CNT_REINIT_FEAT=4] and
        // counters[OAK_CNT_HALF_SAT=5] are cumulative life-of-engine counts.
        ctr[0] = 0; ctr[1] = 0; ctr[2] = 0; ctr[6] = 0
        statsPointer.pointee = KernelStats()

        try runKernel(accumulatePhaseGpu: accumulatePhaseGpu)

        memcpy(&hostAction, stateArena.pointer.advanced(by: stateLayout.policyMean),
               batch * dPolicy * MemoryLayout<Float>.stride)
        return (PolicyMeanBatch(values: hostAction, dPolicy: dPolicy),
                Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000,
                statsPointer.pointee)
    }

    private func writeKernelConfig(paramLayout: OakMetalLayout.Param) {
        let K = OakMetalLayout.optionCount(rank: rank, actDim: actDim)
        let cap = Float(max(rank * max(batch, 1), 1))
        let p = paramLayout, s = stateLayout, u = uintLayout
        var c = KernelConfig()
        c.obsDim = UInt32(obsDim); c.rank = UInt32(rank); c.actDim = UInt32(actDim)
        c.batch = UInt32(batch); c.optionCount = UInt32(K); c.baseObsDim = UInt32(baseObsDim)
        c.actionParamCount = UInt32(dPolicy)
        c.pEncoder = UInt32(p.encoder); c.pValue = UInt32(p.value)
        c.pPsi = UInt32(p.psi);         c.pOptPolicy = UInt32(p.optPolicy)
        c.pOutcomeHead = UInt32(p.outcomeHead)
        c.pBetaW = UInt32(p.betaW); c.pOmegaW = UInt32(p.omegaW)
        c.pSprDecoder = UInt32(p.sprDecoder)
        c.sObservation = UInt32(s.observation); c.sObsMean = UInt32(s.obsMean)
        c.sState = UInt32(s.state); c.sPrevPolicyMean = UInt32(s.prevPolicyMean)
        c.sPolicyMean = UInt32(s.policyMean); c.sScoreGrad = UInt32(s.scoreGrad)
        c.sReward = UInt32(s.reward); c.sTd = UInt32(s.td); c.sTdPsi = UInt32(s.tdPsi)
        c.sVCurr = UInt32(s.vCurr)
        c.sChordSlot = UInt32(s.chordSlot); c.sChordSlotPrev = UInt32(s.chordSlotPrev)
        c.sBetaProb = UInt32(s.betaProb)
        // Pass 14 Tier 1: sSprPred removed (oak_spr_fused inlines ẑ_N).
        c.uDone = UInt32(u.done); c.uActiveMask = UInt32(u.activeMask)
        c.uActiveList = UInt32(u.activeList); c.uOption = UInt32(u.option); c.uCounters = UInt32(u.counters)
        c.uSubtaskFeatureOf = UInt32(u.subtaskFeatureOf); c.uOptionAge = UInt32(u.optionAge)
        c.aFeatSM = UInt32(s.featSM); c.aWBar = UInt32(s.wBar); c.aFeatMean = UInt32(s.featMean)
        c.aAlphaImpFeat = UInt32(s.alphaImpFeat); c.aAlphaImpBeta = UInt32(s.alphaImpBeta); c.aAlphaImpH = UInt32(s.alphaImpH)
        c.aPsiMeanBaseline = UInt32(s.psiMeanBaseline)
        c.aValueE = UInt32(s.valueE); c.aPsiE = UInt32(s.psiE)
        c.aUtilFeat = UInt32(s.utilFeat); c.aUtilOpt = UInt32(s.utilOpt)
        c.aOptFracEMA = UInt32(s.optFracEMA)
        c.cap = cap; c.alpha0 = 1.0 / cap; c.ridge = 1.0 / (cap + 1.0)
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
        // 5 array buffers (params, state, uint, scalars, stats) + config at index 5.
        enc.setBuffers(buffers, offsets: offsets, range: 0..<5)
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
        // 5 array buffers at indices 0..4 (params, state, uint, scalars, stats); config at 5.
        let buffers: [MTLBuffer?] = [paramsArena.buffer, stateArena.buffer, uintArena.buffer,
                                     scalarsBuffer, statsBuffer]
        let offsets = [Int](repeating: 0, count: 5)
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
private func directNumEnvs(obsDim: Int, actDim: Int, dPolicy: Int, device: MTLDevice) throws -> Int {
    let width = try OakMetalEngine.executionWidth(device: device)
    let rank = obsDim
    // Pass 14 Tier 0.6: derive per-env and fixed memory from OakMetalLayout PROGRAMMATICALLY.
    // The hand-rolled formulae below were off by sprPred (B·R from Pass 13 Tier 3) and by
    // alphaImpFeat / Beta / H (3·E from Pass 13 Tier 2) which silently risked GPU memory
    // over-subscription on large envs. Now we compute total bytes from the canonical layout
    // so any new arena slot is automatically counted.
    let probeStateLayout = OakMetalLayout.stateLayout(obsDim: obsDim, rank: rank,
                                                       actDim: actDim, batch: 1, dPolicy: dPolicy)
    let probeFixedStateLayout = OakMetalLayout.stateLayout(obsDim: obsDim, rank: rank,
                                                            actDim: actDim, batch: 0, dPolicy: dPolicy)
    let perEnvFloats = max(probeStateLayout.total - probeFixedStateLayout.total, 0)
    // perEnvUints from uintLayout(batch=1) - uintLayout(batch=0).
    let probeUintLayout = OakMetalLayout.uintLayout(rank: rank, actDim: actDim, batch: 1)
    let probeFixedUintLayout = OakMetalLayout.uintLayout(rank: rank, actDim: actDim, batch: 0)
    let perEnvUints = max(probeUintLayout.total - probeFixedUintLayout.total, 0)
    let perEnvBytes = perEnvFloats * MemoryLayout<Float>.stride
                    + perEnvUints  * MemoryLayout<UInt32>.stride
    let paramBytes = OakMetalLayout.paramLayout(obsDim: obsDim, rank: rank, actDim: actDim,
                                                 dPolicy: dPolicy).total
                    * MemoryLayout<Float16>.stride
    let fixedBytes = paramBytes + probeFixedStateLayout.total * MemoryLayout<Float>.stride
                                + probeFixedUintLayout.total  * MemoryLayout<UInt32>.stride
    // Memory ceiling from the GPU's reported safe working-set size (Apple Metal). Structural
    // to the hardware. This is the *upper bound* on what the agent could ever allocate.
    let memEnvs = max(
        (Int(device.recommendedMaxWorkingSetSize) - fixedBytes) / max(perEnvBytes, 1),
        width)
    // Practical concurrency ceiling (default). One TG = one row in the per-row kernels
    // (oak_plan, oak_meta_action, …); beyond ~`width × width` rows the GPU dispatch wave is
    // already saturated and additional rows just inflate per-batch latency without raising
    // throughput. Set to `width × width × 2 = 2048` (= 64 SIMD groups of `width` threads,
    // which is the empirically-tuned saturation point across M1/M2/M3 hardware: with the
    // current default and the user's hardware, OAK_BENCH_NUM_ENVS=2048 matches the
    // dispatch-wave knee on real env stepping). Override with OAK_MAX_PARALLEL_ENVS.
    let practicalDefault = width * width * 2
    let userCap = ProcessInfo.processInfo.environment["OAK_MAX_PARALLEL_ENVS"].flatMap(Int.init)
    let ceiling = userCap.map { max($0, width) } ?? practicalDefault
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
    var prevPolicyMean: [Float]   // size B · dPolicy
    var prevScoreGrad: [Float]    // size B · dPolicy
    var randPolicyMean: [Float]   // scratch for random-policy μ emission
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
        prevPolicyMean = Array(repeating: 0, count: env.numEnvs * env.dPolicy)
        prevScoreGrad  = Array(repeating: 0, count: env.numEnvs * env.dPolicy)
        randPolicyMean = Array(repeating: 0, count: env.numEnvs * env.dPolicy)
        randSeed = UInt64(bitPattern: Int64(seed ?? Int(oakUnseededSeed)))
        actionCol = MeasuredActionCollector(discreteCount: env.actionDiscreteCount,
                                             envActionWidth: env.envActionWidth)
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
            case "run":       try run(args.dropFirst().first)
            default:          try run(args.first)
            // Pass 15 Tier 5: `probe` subcommand removed. Probe harness (Probe.swift) was
            // a kernel-correctness validation tool for Pass 11-13 deferred slots; its
            // role is now covered by the (B) numerical-safety + (C) bounded-magnitude
            // hard gates surfaced into bench JSON via RunHealth (see emitSnapshot).
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
                                         dPolicy: probe.dPolicy, device: device)
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
                                         dPolicy: probe.dPolicy, device: device)
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
        // Pass 15 Tier 3: default seedCount lowered 10 → 5 (per plan §9.1; concurrent
        // execution makes more seeds cheap if needed but 5 is the canonical Pass-15 scope).
        let seedCount = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 5
        let warmup = max(ProcessInfo.processInfo.environment["OAK_BENCH_WARMUP_STEPS"].flatMap(Int.init) ?? 1000, 0)
        // Pass 15 Tier 3: default stopAfter raised 50K → 500K (per plan §9.1 — the new canonical
        // workerDepth at B=1 is 500K = the full env-step count in a single continuing trajectory,
        // not 50K/2048 = 24 effective steps per env that Pass 12-14 batched bench produced).
        let stopAfter = max(ProcessInfo.processInfo.environment["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 500000, 1)
        let policies = try (ProcessInfo.processInfo.environment["OAK_BENCH_POLICIES"] ?? "oak,random")
            .split(separator: ",").map { try parsePolicy(String($0)) }
        let device = try metalDevice()
        // Pass 15 Tier 3: B=1 single-stream per env (BWH-orthodox per Sutton continuing-task
        // formulation). Each agent owns one continuous trajectory per env. OAK_BENCH_NUM_ENVS
        // override retained for back-compat with batched Pass-14 bench (set to 64 etc. to
        // re-run pre-Pass-15 protocol for direct comparison).
        let numEnvs = max(ProcessInfo.processInfo.environment["OAK_BENCH_NUM_ENVS"].flatMap(Int.init)
                       ?? ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init) ?? 1, 1)
        // Pass 15 Tier 3: ConcurrentBenchHarness — N work-items run in parallel via
        // OperationQueue (one MTLCommandQueue per OakMetalEngine instance; Apple's Metal
        // driver schedules them across the GPU's ~4096 ALUs). N=8 default (per plan §3.3);
        // OAK_BENCH_CONCURRENCY=1 forces serial execution for debugging.
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
        fputs("OaK benchmark: \(totalWork) work items, B=\(numEnvs), N=\(concurrency) concurrent (steps=\(stopAfter), seeds=\(seedCount))\n", stderr)
        fflush(stderr)

        // Shared progress + first-error tracker. Reference type so the @Sendable closure
        // captures the wrapper, not the mutable value (Swift 6 strict concurrency).
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
                                      warmup: warmup, stopAfter: stopAfter, numEnvs: numEnvs,
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

    private static func randomPolicyMean(for env: EnvPoolEnv, numEnvs: Int,
                                          stepCount: inout UInt64,
                                          floats: inout [Float]) -> PolicyMeanBatch {
        // Pass 11 Tier 1: the random "policy" emits a uniform-random μ vector in [-1, 1] of
        // width B · dPolicy. The wrapper then samples per-component (Gumbel-max for Discrete:
        // a uniform-random μ produces a near-uniform categorical → effectively random index;
        // structural-σ-1 Gaussian for Box: μ + N(0,1) ≈ N(0,1) → bound-clamped). This keeps
        // the random baseline shape-equivalent to the agent and exercises the same wrapper
        // sampling path, so the (oak vs random) comparison is a clean policy-only ablation.
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

    // --- single-run orchestration (setup → loop → snapshot) --------------------------------

    private static func setupRun(envId: String, policy: Policy, seed: Int?, warmup: Int,
                                 stopAfter: Int, numEnvs: Int, device: MTLDevice, command: String,
                                 includeDiagnostic: Bool, metricsDir: URL?) throws -> RunContext {
        let dir = RunSnapshot.resolveMetricsDirectory(explicit: metricsDir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        fputs("OaK metrics \(dir.path)\n", stderr); fflush(stderr)
        let env = try EnvPoolEnv(id: envId, numEnvs: numEnvs, seed: seed)
        // Pass 11 Tier 1: agent constructor takes signal-shape params + dPolicy (env-declared).
        // Action-space type, per-dim bounds, and per-component sampling all live in EnvPoolEnv.
        let engine: OakMetalEngine? = policy == .oak ? try OakMetalEngine(
            obsDim: env.obsDim, baseObsDim: env.obsDim, actDim: env.actDim,
            batch: numEnvs, rank: env.obsDim, dPolicy: env.dPolicy,
            device: device, seed: seed, diagnosticGpuPhaseTiming: includeDiagnostic,
            ablatedKernels: ablatedKernelsFromEnv()) : nil
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
        let (dA, cA) = state.actionCol.partialMetrics()
        let sps = state.measuredSeconds > 0 ? Double(state.measuredSteps) / state.measuredSeconds : nil
        let reason = seq == nil ? "measured_steps" : "heartbeat"
        return RunMetricsFile(
            schemaVersion: 3, command: ctx.command, envId: ctx.envId, policy: ctx.policy, seed: ctx.seed,
            numEnvs: ctx.numEnvs, warmupSteps: ctx.warmup, stopAfterMeasuredSteps: ctx.stopAfter,
            observationDim: ctx.env.obsDim, actionDim: ctx.env.actDim,
            actionDiscreteCount: ctx.env.actionDiscreteCount,
            measuredStepCount: state.measuredSteps, measuredBatchCount: u,
            measuredSeconds: state.measuredSeconds,
            workerDepth: state.measuredSteps / max(ctx.numEnvs, 1),
            stepsPerSecond: sps,
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
            let policyMean: PolicyMeanBatch, agentMs: Double, stats: KernelStats?
            if let engine = ctx.engine {
                let out = try ctx.env.observationScratch.withUnsafeBufferPointer { buf in
                    try engine.step(obs: buf, prevPolicyMean: state.prevPolicyMean,
                                    prevScoreGrad: state.prevScoreGrad, prevReward: state.prevReward,
                                    prevDone: state.prevDone, accumulatePhaseGpu: measured)
                }
                policyMean = out.0; agentMs = out.1; stats = out.2
            } else {
                let t0 = DispatchTime.now().uptimeNanoseconds
                policyMean = randomPolicyMean(for: ctx.env, numEnvs: ctx.numEnvs, stepCount: &state.randSeed,
                                              floats: &state.randPolicyMean)
                agentMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000
                stats = nil
            }
            let envStart = DispatchTime.now().uptimeNanoseconds
            try ctx.env.sampleAndStep(policyMean: policyMean.values)
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
                state.actionCol.addMeasured(envAction: ctx.env.envActionScratch, numEnvs: ctx.numEnvs)
            }
            memcpy(&state.prevReward, ctx.env.rewardScratch, ctx.numEnvs * MemoryLayout<Float>.stride)
            memcpy(&state.prevDone,   ctx.env.doneScratch,   ctx.numEnvs * MemoryLayout<UInt8>.stride)
            memcpy(&state.prevPolicyMean, policyMean.values, policyMean.values.count * MemoryLayout<Float>.stride)
            memcpy(&state.prevScoreGrad, ctx.env.scoreGradScratch,
                   ctx.env.scoreGradScratch.count * MemoryLayout<Float>.stride)
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
