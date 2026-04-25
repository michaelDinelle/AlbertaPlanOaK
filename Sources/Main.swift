import Foundation
import Metal
import Darwin

// =================================================================================================
// Orthodox Step 11 Oak - host-side dispatcher. See Sources/OakKernel.metal for the architectural
// citation block. This file owns: arena sizing, Metal engine, env-pool FFI, CLI commands
// (run/diagnose/benchmark), and run-metrics JSON emission.
//
// Current stream semantics: ONE agent instance owns ONE env (numEnvs=1) and processes ONE
// observation per step. Concurrent benchmarking runs N independent worker processes by default
// so native env/global state cannot leak across streams; future batched kernels must preserve
// disjoint per-agent state.
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
    let continuingRestartCount: Double?
    // Kernel-side (only present for oak):
    let oakOptionReinitCountFinal: Int?, oakFeatureReinitCountFinal: Int?
    let nonFiniteByPhase: [String: Int]?
    let rrBonusMean: Double?
    let psiStopGapMean: Double?
    let psiStopGapAbsMean: Double?
    let psiAbsMean: Double?
    let rrStopAbsMean: Double?
    let valueAbsMean: Double?
    let terminationRate: Double?
    let optionDurationMean: Double?
    let stateEffectiveRank: Double?
    let importanceEss: Double?
    let gvfTdMean: Double?
    let lambdaEffectiveRank: Double?
    let actorUpdateNormMean: Double?
    let actorTdMean: Double?
    let actorSupportEntropy: Double?
    let modelPriorityMax: Double?
}

struct RunDiagnostic: Codable {
    let metalPhaseGpuMsMean: [String: Double]?
    let metalPhaseGpuTimingInvalid: [String]?
}

private struct RunMetricsFile: Encodable {
    let schemaVersion: Int, command: String, envId: String, policy: Policy
    let seed: Int?
    let numEnvs: Int, stopAfterMeasuredSteps: Int
    let observationDim: Int, actionDim: Int, actionDiscreteCount: Int?
    let measuredStepCount: Int, measuredSeconds: Double
    let stepsPerSecond: Double?, heartbeatSequence: Int?
    let rewardSum: Double, doneCount: Int, stopReason: String
    let health: RunHealth?, diagnostic: RunDiagnostic?
}

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
    private var lastKernelStats: KernelStats?
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
        let dlMean = dl / Double(max(currPolicyMean.count, 1))
        let sgMean = sg / Double(max(scoreGrad.count, 1))
        let metricMax = Double.greatestFiniteMagnitude / 4
        let dlSafe = dlMean.isFinite ? min(dlMean, metricMax) : metricMax
        let sgSafe = sgMean.isFinite ? min(sgMean, metricMax) : metricMax
        actDeltaSum = min(metricMax, actDeltaSum + dlSafe)
        scoreGradSum = min(metricMax, scoreGradSum + sgSafe)
    }

    mutating func recordKernelStats(_ stats: KernelStats) {
        kernelSteps += 1
        lastKernelStats = stats
        encodeNF += Int(stats.encodeNonFiniteCount)
        learnNF  += Int(stats.learnNonFiniteCount)
        planNF   += Int(stats.planNonFiniteCount)
        tdPsiNF  += Int(stats.tdPsiNonFiniteCount)
        oakOptionReinitFinal  = Int(stats.oakOptionReinitCount)
        oakFeatureReinitFinal = Int(stats.oakFeatureReinitCount)
    }

    var hasMeasuredSteps: Bool { envSteps > 0 }

    func snapshot(continuingRestartCount: Int = 0) -> RunHealth? {
        guard envSteps > 0 else { return nil }
        let n = Double(envSteps)
        let nf: [String: Int]? = kernelSteps > 0
            ? ["encode": encodeNF, "learn": learnNF, "plan": planNF, "tdPsi": tdPsiNF]
            : nil
        func stat(_ keyPath: KeyPath<KernelStats, Float>) -> Double? {
            guard kernelSteps > 0, let stats = lastKernelStats else { return nil }
            let v = Double(stats[keyPath: keyPath])
            return v.isFinite ? v : 0.0
        }
        return RunHealth(
            rewardRate: rhoBar,
            actionDeltaL2Mean: actDeltaSum / n,
            scoreGradL2Mean: scoreGradSum / n,
            continuingRestartCount: Double(continuingRestartCount),
            oakOptionReinitCountFinal: kernelSteps > 0 ? oakOptionReinitFinal : nil,
            oakFeatureReinitCountFinal: kernelSteps > 0 ? oakFeatureReinitFinal : nil,
            nonFiniteByPhase: nf,
            rrBonusMean: stat(\.rrBonusMean),
            psiStopGapMean: stat(\.psiStopGapMean),
            psiStopGapAbsMean: stat(\.psiStopGapAbsMean),
            psiAbsMean: stat(\.psiAbsMean),
            rrStopAbsMean: stat(\.rrStopAbsMean),
            valueAbsMean: stat(\.valueAbsMean),
            terminationRate: stat(\.terminationRate),
            optionDurationMean: stat(\.optionDurationMean),
            stateEffectiveRank: stat(\.stateEffectiveRank),
            importanceEss: stat(\.importanceEss),
            gvfTdMean: stat(\.gvfTdMean),
            lambdaEffectiveRank: stat(\.lambdaEffectiveRank),
            actorUpdateNormMean: stat(\.actorUpdateNormMean),
            actorTdMean: stat(\.actorTdMean),
            actorSupportEntropy: stat(\.actorSupportEntropy),
            modelPriorityMax: stat(\.modelPriorityMax))
    }
}

// =================================================================================================
// Kernel structs - mirror OakKernel.metal byte-for-byte. Keep in lockstep when editing either.
// =================================================================================================
private struct KernelConfig {
    var obsDim: UInt32 = 0, rank: UInt32 = 0
    var optionCount: UInt32 = 0
    var actionParamCount: UInt32 = 0
    var obsTraceLevels: UInt32 = 0
    var linearAnchorCount: UInt32 = 0
    var optionPageWidth: UInt32 = 0
    var invRank: Float = 0, invOptionCount: Float = 0
    var invOneInputWidth: Float = 0, metaEta: Float = 0
    var pValue: UInt32 = 0, pPsi: UInt32 = 0
    var pOptPolicy: UInt32 = 0, pOptReward: UInt32 = 0, pOptTransition: UInt32 = 0
    var pOptDuration: UInt32 = 0
    var pPsiCross: UInt32 = 0
    var pOneReward: UInt32 = 0
    var pGvfRate: UInt32 = 0, pGvfValue: UInt32 = 0
    var sObservation: UInt32 = 0, sObsMean: UInt32 = 0, sObsSM: UInt32 = 0
    var sState: UInt32 = 0
    var sPolicyMean: UInt32 = 0, sActionSignal: UInt32 = 0
    var sOptionScoreGrad: UInt32 = 0
    var sOptionPolicyMean: UInt32 = 0
    var sImportance: UInt32 = 0
    var sReward: UInt32 = 0, sTd: UInt32 = 0, sTdPsi: UInt32 = 0
    var sUtilFeat: UInt32 = 0, sUtilOpt: UInt32 = 0
    var sBonusW: UInt32 = 0
    var sFeatureMean: UInt32 = 0, sFeatureSM: UInt32 = 0
    var sAlpha: UInt32 = 0, sBeta: UInt32 = 0, sH: UInt32 = 0
    var sGradEnergy: UInt32 = 0
    var sAlphaPsi: UInt32 = 0, sBetaPsi: UInt32 = 0, sHPsi: UInt32 = 0, sGradEnergyPsi: UInt32 = 0
    var sAlphaModel: UInt32 = 0, sBetaModel: UInt32 = 0, sHModel: UInt32 = 0, sGradEnergyModel: UInt32 = 0
    var sAlphaGvf: UInt32 = 0, sBetaGvf: UInt32 = 0, sHGvf: UInt32 = 0, sGradEnergyGvf: UInt32 = 0
    var sActorAdv: UInt32 = 0
    var sNTdMat: UInt32 = 0
    var sOptRewardTd: UInt32 = 0
    var sHeadDenom: UInt32 = 0, sCrossTd: UInt32 = 0
    var sOneRewardTd: UInt32 = 0, sOneStepDenom: UInt32 = 0
    var sDurationTd: UInt32 = 0
    var sDurationHat: UInt32 = 0
    var sObsTraceMulti: UInt32 = 0
    var sPrioOpt: UInt32 = 0
    var sContinuation: UInt32 = 0 // RR-derived per-option continuation indicator/probability.
    var sLambda: UInt32 = 0, sKeyboardScore: UInt32 = 0
    var sGvfTd: UInt32 = 0, sFeatureTraceDenom: UInt32 = 0
    var sModelDenom: UInt32 = 0
    var sOptionPlanningUse: UInt32 = 0
    var sValueGauge: UInt32 = 0
    var uCounters: UInt32 = 0
    var uOptFeature: UInt32 = 0
    var uFeatureAge: UInt32 = 0
    var uOptAge: UInt32 = 0
    var uFeatureParentA: UInt32 = 0, uFeatureParentB: UInt32 = 0, uFeatureKind: UInt32 = 0
    var uFeatureBirthCount: UInt32 = 0
    var uSelectedOption: UInt32 = 0, uHighModelOption: UInt32 = 0
    var uModelAge: UInt32 = 0
    var uGvfAge: UInt32 = 0, uBadFeatureAge: UInt32 = 0
    var uPrimitiveChannel: UInt32 = 0
}

private struct KernelScalars {
    var stepCount: UInt32 = 0
    var stateSlot: UInt32 = 0
    var rho: Float = 0
}

struct KernelStats {
    // Only telemetry the host literally cannot derive from env signals.
    var encodeNonFiniteCount: Float = 0, learnNonFiniteCount: Float = 0
    var planNonFiniteCount: Float = 0, tdPsiNonFiniteCount: Float = 0
    var oakOptionReinitCount: Float = 0, oakFeatureReinitCount: Float = 0
    var rrBonusMean: Float = 0, psiStopGapMean: Float = 0
    var psiStopGapAbsMean: Float = 0, psiAbsMean: Float = 0
    var rrStopAbsMean: Float = 0, valueAbsMean: Float = 0
    var terminationRate: Float = 0, optionDurationMean: Float = 0
    var stateEffectiveRank: Float = 0
    var importanceEss: Float = 0
    var gvfTdMean: Float = 0
    var lambdaEffectiveRank: Float = 0
    var actorUpdateNormMean: Float = 0
    var actorTdMean: Float = 0
    var actorSupportEntropy: Float = 0
    var modelPriorityMax: Float = 0
}

// =================================================================================================
// Arena layout - single source of truth for offsets. Single-stream: no `batch` parameter.
// =================================================================================================
private enum OakMetalLayout {
    static let kernelConfigBufferIndex = 5
    static let optionPageWidth = 32

    static func optionCount(rank: Int) -> Int { max(1, rank) }
    static func psiCount(rank: Int) -> Int { optionCount(rank: rank) * rank }
    static func psiCrossCount(rank: Int) -> Int {
        let K = optionCount(rank: rank)
        return K * K * rank
    }
    static func optPolicyCount(rank: Int, dPolicy: Int) -> Int {
        optionCount(rank: rank) * dPolicy * rank
    }
    static func optionPolicyMeanCount(rank: Int, dPolicy: Int) -> Int {
        optionCount(rank: rank) * dPolicy
    }
    static func optRewardCount(rank: Int) -> Int { optionCount(rank: rank) * rank }
    static func optDurationCount(rank: Int) -> Int { optionCount(rank: rank) * rank }
    static func optTransitionCount(rank: Int) -> Int { optionCount(rank: rank) * rank * rank }
    static func oneRewardCount(rank: Int) -> Int { rank }
    static func gvfCount(rank: Int) -> Int {
        rank + optionCount(rank: rank) + 2
    }
    static func gvfUpdateCount(rank: Int) -> Int {
        min(optionPageWidth, gvfCount(rank: rank))
    }
    static func gvfValueUpdateCount(rank: Int) -> Int {
        gvfUpdateCount(rank: rank) * rank
    }
    static func gvfValueCount(rank: Int) -> Int {
        gvfCount(rank: rank) * rank
    }
    static func obsTraceLevelCount(obsDim: Int) -> Int {
        let b = max(obsDim, 1)
        var m = 0
        var v = b
        while v > 1 { v >>= 1; m += 1 }     // floor(log2(b))
        return m + 1                         // +1 for the instantaneous level
    }
    static func linearInputWidth(obsDim: Int, dPolicy: Int) -> Int {
        obsTraceLevelCount(obsDim: obsDim) * max(obsDim, 1) + max(dPolicy, 0) + 2
    }
    static func linearAnchorCount(rank: Int) -> Int {
        min(rank, max(rank / 2, 1))
    }
    struct Param {
        let value, psi, optPolicy, optReward, optTransition, optDuration: Int
        let psiCross, oneReward, gvfRate, gvfValue: Int
        let total: Int
    }
    static func paramLayout(rank: Int, dPolicy: Int) -> Param {
        var o = 0
        let val = o; o += rank
        let psi = o; o += psiCount(rank: rank)
        let op  = o; o += optPolicyCount(rank: rank, dPolicy: dPolicy)
        let or_ = o; o += optRewardCount(rank: rank)
        let ot  = o; o += optTransitionCount(rank: rank)
        let od  = o; o += optDurationCount(rank: rank)
        let pc  = o; o += psiCrossCount(rank: rank)
        let oneR = o; o += oneRewardCount(rank: rank)
        let gr = o; o += gvfCount(rank: rank)
        let gv = o; o += gvfValueCount(rank: rank)
        return Param(value: val, psi: psi, optPolicy: op, optReward: or_,
                     optTransition: ot, optDuration: od,
                     psiCross: pc, oneReward: oneR,
                     gvfRate: gr, gvfValue: gv, total: o)
    }

    struct State {
        let observation, obsMean, obsSM, state: Int
        let policyMean, actionSignal, optionScoreGrad: Int
        let optionPolicyMean, importance: Int
        let reward, td, tdPsi: Int
        let utilFeat, utilOpt: Int
        let bonusW, featureMean, featureSM: Int
        let alpha, beta, h: Int       // MetaOptimize L/Hessian-free approximation buffers (R floats each).
        let gradEnergy: Int
        let alphaPsi, betaPsi, hPsi, gradEnergyPsi: Int
        let alphaModel, betaModel, hModel, gradEnergyModel: Int
        let alphaGvf, betaGvf, hGvf, gradEnergyGvf: Int
        let actorAdv: Int
        let nTdMat: Int               // Dense option-transition TD targets, K*R floats, written by oak_plan.
        let optRewardTd: Int          // Option reward-model TD targets, K floats, written once by oak_plan.
        let headDenom, crossTd: Int
        let oneRewardTd, oneStepDenom: Int
        let durationTd, durationHat: Int
        let obsTraceMulti: Int
        let prioOpt: Int              // Tier D2: per-option priority for plan_reduce / Head 4.
        let continuation: Int          // RR-derived continuation, K floats.
        let lambda, keyboardScore: Int
        let gvfTd, featureTraceDenom: Int
        let modelDenom: Int
        let optionPlanningUse: Int
        let valueGauge: Int
        let total: Int
    }
    static func stateLayout(obsDim: Int, rank: Int, dPolicy: Int) -> State {
        let K = optionCount(rank: rank)
        let M = obsTraceLevelCount(obsDim: obsDim)
        let G = gvfCount(rank: rank)
        var o = 0
        let obs = o; o += obsDim
        let mean = o; o += obsDim
        let osm  = o; o += obsDim
        let st = o; o += 2 * rank
        let pol = o; o += dPolicy
        let act = o; o += dPolicy
        let osg = o; o += K * dPolicy
        let opm = o; o += K * dPolicy
        let imp = o; o += K
        let rew = o; o += 1
        let td  = o; o += 1
        let tdP = o; o += K
        let uf  = o; o += rank
        let uo  = o; o += K
        let bw  = o; o += rank
        let fm  = o; o += rank
        let fsm = o; o += rank
        let al  = o; o += rank
        let be  = o; o += rank
        let h   = o; o += rank
        let ge  = o; o += rank
        let aps = o; o += rank
        let bps = o; o += rank
        let hps = o; o += rank
        let gps = o; o += rank
        let apm = o; o += rank
        let bpm = o; o += rank
        let hpm = o; o += rank
        let gpm = o; o += rank
        let apg = o; o += rank
        let bpg = o; o += rank
        let hpg = o; o += rank
        let gpg = o; o += rank
        let aadv = o; o += K
        let ntd = o; o += K * rank
        let ortd = o; o += K
        let hden = o; o += K
        let ctd = o; o += K
        let onertd = o; o += 1
        let oneden = o; o += 1
        let dtd = o; o += K
        let dhat = o; o += K
        let otr = o; o += max(M - 1, 0) * max(obsDim, 1)
        let prio = o; o += K
        let cont = o; o += K
        let lambda = o; o += K
        let kscore = o; o += K
        let gvftd = o; o += G
        let ftd = o; o += 1
        let modelDenom = o; o += K
        let planUse = o; o += K
        let vg = o; o += 1
        return State(observation: obs, obsMean: mean, obsSM: osm, state: st,
                     policyMean: pol, actionSignal: act, optionScoreGrad: osg,
                     optionPolicyMean: opm, importance: imp,
                     reward: rew, td: td, tdPsi: tdP,
                     utilFeat: uf, utilOpt: uo, bonusW: bw, featureMean: fm, featureSM: fsm,
                     alpha: al, beta: be, h: h,
                     gradEnergy: ge,
                     alphaPsi: aps, betaPsi: bps, hPsi: hps, gradEnergyPsi: gps,
                     alphaModel: apm, betaModel: bpm, hModel: hpm, gradEnergyModel: gpm,
                     alphaGvf: apg, betaGvf: bpg, hGvf: hpg, gradEnergyGvf: gpg,
                     actorAdv: aadv,
                     nTdMat: ntd, optRewardTd: ortd, headDenom: hden, crossTd: ctd,
                     oneRewardTd: onertd, oneStepDenom: oneden,
                     durationTd: dtd, durationHat: dhat,
                     obsTraceMulti: otr,
                     prioOpt: prio, continuation: cont,
                     lambda: lambda, keyboardScore: kscore,
                     gvfTd: gvftd, featureTraceDenom: ftd,
                     modelDenom: modelDenom, optionPlanningUse: planUse,
                     valueGauge: vg, total: o)
    }

    struct UInts {
        let counters, optFeature, featureAge, optAge: Int
        let featureParentA, featureParentB, featureKind, featureBirthCount: Int
        let selectedOption, highModelOption, modelAge: Int
        let gvfAge, badFeatureAge: Int
        let primitiveChannel: Int
        let total: Int
    }
    static func uintLayout(rank: Int) -> UInts {
        let K = optionCount(rank: rank)
        let G = gvfCount(rank: rank)
        var o = 0
        let ctr = o; o += 7
        let of = o; o += K
        let fa = o; o += rank
        let oa = o; o += K
        let fpa = o; o += rank
        let fpb = o; o += rank
        let fkind = o; o += rank
        let fbc = o; o += rank
        let sel = o; o += 1
        let high = o; o += 1
        let ma = o; o += K
        let ga = o; o += G
        let bfa = o; o += rank
        let pc = o; o += rank
        return UInts(counters: ctr, optFeature: of, featureAge: fa, optAge: oa,
                     featureParentA: fpa, featureParentB: fpb,
                     featureKind: fkind, featureBirthCount: fbc,
                     selectedOption: sel, highModelOption: high,
                     modelAge: ma, gvfAge: ga, badFeatureAge: bfa,
                     primitiveChannel: pc, total: o)
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
// Metal engine - Step 1-11 Oak rebuild. Focused every-step phases keep the Plan mechanisms
// visible while preserving one command-buffer ICB dispatch ordering.
// =================================================================================================
final class OakMetalEngine {
    private struct Phase { let name: String; let tgSize: Int }
    private static let phases: [Phase] = [
        Phase(name: "oak_encode",           tgSize: -1),
        Phase(name: "oak_plan",             tgSize: -1),
        Phase(name: "oak_plan_reduce",      tgSize: -1),
        Phase(name: "oak_learn_heads",      tgSize: 0),
        Phase(name: "oak_replace",          tgSize: -1),
    ]

    private let obsDim, rank, dPolicy: Int
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
    private var hostOptionAction: [Float]
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
            return try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
        }
    }

    static func executionWidth(device: MTLDevice) throws -> Int { 32 }

    init(obsDim: Int, rank: Int, dPolicy: Int,
         device: MTLDevice, seed: Int?, diagnosticGpuPhaseTiming: Bool = false) throws {
        // R (rank) is a compute budget and the active feature/subtask/option count. It is no
        // longer capped to one SIMD group; 32 is only the scheduling page width.
        precondition(rank >= 2, "rank must be >= 2 (feature/subtask/option budget).")
        precondition(rank <= 256, "rank must be <= 256 (oak_plan threadgroup storage cap).")
        precondition(obsDim >= 1, "obsDim must be positive.")
        precondition(dPolicy >= 1, "dPolicy must be positive.")
        self.obsDim = obsDim
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

        let paramLayout = OakMetalLayout.paramLayout(rank: rank, dPolicy: dPolicy)
        stateLayout = OakMetalLayout.stateLayout(obsDim: obsDim, rank: rank, dPolicy: dPolicy)
        uintLayout  = OakMetalLayout.uintLayout(rank: rank)
        paramsArena = Shared(device: device, count: paramLayout.total, zero: Float(0))
        stateArena  = Shared(device: device, count: stateLayout.total, zero: Float(0))
        uintArena   = Shared(device: device, count: uintLayout.total,  zero: UInt32(0))

        let K = OakMetalLayout.optionCount(rank: rank)
        // Grid widths sized for one row (single-stream).
        let crossPageCount = K * rank
        let learnTotal = rank
                       + OakMetalLayout.psiCount(rank: rank)
                       + crossPageCount
                       + OakMetalLayout.optPolicyCount(rank: rank, dPolicy: dPolicy)
                       + OakMetalLayout.optRewardCount(rank: rank)
                       + OakMetalLayout.optTransitionCount(rank: rank)
                       + OakMetalLayout.optDurationCount(rank: rank)
                       + rank
                       + OakMetalLayout.gvfUpdateCount(rank: rank)
                       + OakMetalLayout.gvfValueUpdateCount(rank: rank)
        let replaceGroups = max(1, (K * dPolicy + OakMetalLayout.optionPageWidth - 1) / OakMetalLayout.optionPageWidth)
        let grids = [
            1,                  // encode + keyboard_predict
            K <= OakMetalLayout.optionPageWidth ? K : OakMetalLayout.optionPageWidth,
            1,                  // plan_reduce
            learnTotal,         // learn_heads
            replaceGroups,      // replace + action emission
        ]
        phaseGridWidths = grids

        phaseThreadgroupSizes = Self.phases.map { phase in
            if phase.tgSize == -1 { return max(rank, OakMetalLayout.optionPageWidth) }
            return phase.tgSize
        }

        self.diagnosticGpuPhaseTiming = diagnosticGpuPhaseTiming
        phaseGpuTimeSumSeconds = Array(repeating: 0, count: Self.phases.count)
        phaseGpuInvalidStepCount = Array(repeating: 0, count: Self.phases.count)
        hostAction = Array(repeating: 0, count: dPolicy)
        hostOptionAction = Array(repeating: 0, count: K * dPolicy)

        resourceList = [paramsArena.buffer, stateArena.buffer, uintArena.buffer,
                        scalarsBuffer, statsBuffer, configBuffer]

        let invR: Float = 1.0 / sqrt(Float(max(rank, 1)))
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
        sample(paramLayout.value, rank, invR)
        sample(paramLayout.psi, OakMetalLayout.psiCount(rank: rank), invR)
        let policyPtr = paramsArena.pointer.advanced(by: paramLayout.optPolicy)
        for i in 0..<OakMetalLayout.optPolicyCount(rank: rank, dPolicy: dPolicy) { policyPtr[i] = 0 }
        sample(paramLayout.optReward, OakMetalLayout.optRewardCount(rank: rank), invR)
        sample(paramLayout.optTransition, OakMetalLayout.optTransitionCount(rank: rank), invR)
        let psiCrossPtr = paramsArena.pointer.advanced(by: paramLayout.psiCross)
        for i in 0..<OakMetalLayout.psiCrossCount(rank: rank) { psiCrossPtr[i] = 0 }
        let oneRewardPtr = paramsArena.pointer.advanced(by: paramLayout.oneReward)
        for i in 0..<OakMetalLayout.oneRewardCount(rank: rank) { oneRewardPtr[i] = 0 }
        sample(paramLayout.gvfValue,
               OakMetalLayout.gvfValueCount(rank: rank),
               invR)
        // Duration defaults to the structural one-step lower bound.
        let durationPtr = paramsArena.pointer.advanced(by: paramLayout.optDuration)
        for i in 0..<OakMetalLayout.optDurationCount(rank: rank) { durationPtr[i] = 0 }
        // psi feature-attainment seed: cumulant k gets +invR on its anchor feature index k.
        let psiPtr = paramsArena.pointer.advanced(by: paramLayout.psi)
        for k in 0..<K where k < rank {
            psiPtr[k * rank + k] += invR
        }
        for k in 0..<K where k < rank {
            psiCrossPtr[(k * K + k) * rank + k] += invR
        }
        let bonusSlot = stateArena.pointer.advanced(by: stateLayout.bonusW)
        let valuePtr = paramsArena.pointer.advanced(by: paramLayout.value)
        for i in 0..<rank { bonusSlot[i] = valuePtr[i] }

        // Lambda seed: uniform 1/K (max-entropy prior). Keyboard recomputes each step.
        let lambdaSlot = stateArena.pointer.advanced(by: stateLayout.lambda)
        let scoreSlot = stateArena.pointer.advanced(by: stateLayout.keyboardScore)
        let invK: Float = 1.0 / Float(max(K, 1))
        for k in 0..<K {
            lambdaSlot[k] = invK
            scoreSlot[k] = 0
        }
        let importanceSlot = stateArena.pointer.advanced(by: stateLayout.importance)
        for k in 0..<K { importanceSlot[k] = 1 }

        // Per-option utility prior: utilOpt[k] = R (capacity-volume). The EMA in
        // oak_post_learn decays this toward true |δ_k|², making low-utility old options
        // eligible for Step-11 remove/replace without any tunable warmup gate.
        let utilOptSlot = stateArena.pointer.advanced(by: stateLayout.utilOpt)
        let utilOptPrior: Float = Float(rank)
        for k in 0..<K { utilOptSlot[k] = utilOptPrior }
        let optFeatureSlot = uintArena.pointer.advanced(by: uintLayout.optFeature)
        let parentA = uintArena.pointer.advanced(by: uintLayout.featureParentA)
        let parentB = uintArena.pointer.advanced(by: uintLayout.featureParentB)
        let kind = uintArena.pointer.advanced(by: uintLayout.featureKind)
        let primitiveChannel = uintArena.pointer.advanced(by: uintLayout.primitiveChannel)
        let anchorCount = OakMetalLayout.linearAnchorCount(rank: rank)
        let linearWidth = OakMetalLayout.linearInputWidth(obsDim: obsDim, dPolicy: dPolicy)
        for k in 0..<K {
            optFeatureSlot[k] = UInt32(k)
        }
        for i in 0..<rank {
            if i < anchorCount {
                primitiveChannel[i] = i == 0
                    ? UInt32(max(linearWidth - 1, 0))
                    : UInt32(((i * max(linearWidth - 1, 1)) / max(anchorCount, 1)) % max(linearWidth - 1, 1))
            } else {
                primitiveChannel[i] = 0
            }
            let pa = max(0, min(anchorCount - 1, i % max(anchorCount, 1)))
            let pb = max(0, min(anchorCount - 1, (i &* 7 &+ 3) % max(anchorCount, 1)))
            parentA[i] = UInt32(pa)
            parentB[i] = UInt32(pb)
            kind[i] = UInt32(i % 3)
        }

        // MetaOptimize L-approximation seed (Sharifnassab-Salehkaleybar-Sutton 2024
        // arXiv:2402.02342). α[i] = 1/obsDim is the prior step size, β[i] = log α[i] is
        // its log so the meta-update is multiplicative, h[i] = 0 is the initial accumulator
        // (no meta-gradient history at t=0).
        let alphaInit: Float = 1.0 / Float(max(obsDim, 1))
        let betaInit: Float  = log(alphaInit)
        let alphaSlot = stateArena.pointer.advanced(by: stateLayout.alpha)
        let betaSlot  = stateArena.pointer.advanced(by: stateLayout.beta)
        let hSlot     = stateArena.pointer.advanced(by: stateLayout.h)
        let alphaPsiSlot = stateArena.pointer.advanced(by: stateLayout.alphaPsi)
        let betaPsiSlot  = stateArena.pointer.advanced(by: stateLayout.betaPsi)
        let hPsiSlot     = stateArena.pointer.advanced(by: stateLayout.hPsi)
        let alphaModelSlot = stateArena.pointer.advanced(by: stateLayout.alphaModel)
        let betaModelSlot  = stateArena.pointer.advanced(by: stateLayout.betaModel)
        let hModelSlot     = stateArena.pointer.advanced(by: stateLayout.hModel)
        let alphaGvfSlot = stateArena.pointer.advanced(by: stateLayout.alphaGvf)
        let betaGvfSlot  = stateArena.pointer.advanced(by: stateLayout.betaGvf)
        let hGvfSlot     = stateArena.pointer.advanced(by: stateLayout.hGvf)
        for i in 0..<rank {
            alphaSlot[i] = alphaInit
            betaSlot[i]  = betaInit
            hSlot[i]     = 0
            alphaPsiSlot[i] = alphaInit
            betaPsiSlot[i] = betaInit
            hPsiSlot[i] = 0
            alphaModelSlot[i] = alphaInit
            betaModelSlot[i] = betaInit
            hModelSlot[i] = 0
            alphaGvfSlot[i] = alphaInit
            betaGvfSlot[i] = betaInit
            hGvfSlot[i] = 0
        }
        writeKernelConfig(paramLayout: paramLayout)

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
              prevActionSignal: [Float],
              prevOptionScoreGrad: [Float],
              prevImportance: [Float],
              prevReward: Float,
              accumulatePhaseGpu: Bool,
              captureStats: Bool) throws -> (PolicyMeanBatch, KernelStats?) {
        precondition(obs.count == obsDim)
        precondition(prevActionSignal.count == dPolicy)
        let K = OakMetalLayout.optionCount(rank: rank)
        precondition(prevOptionScoreGrad.count == K * dPolicy)
        memcpy(stateArena.pointer.advanced(by: stateLayout.observation), obs.baseAddress!,
               obsDim * MemoryLayout<Float>.stride)
        stateArena.pointer.advanced(by: stateLayout.reward).pointee = prevReward
        memcpy(stateArena.pointer.advanced(by: stateLayout.actionSignal), prevActionSignal,
               dPolicy * MemoryLayout<Float>.stride)
        memcpy(stateArena.pointer.advanced(by: stateLayout.optionScoreGrad), prevOptionScoreGrad,
               K * dPolicy * MemoryLayout<Float>.stride)
        if prevImportance.count == K {
            memcpy(stateArena.pointer.advanced(by: stateLayout.importance), prevImportance,
                   K * MemoryLayout<Float>.stride)
        }
        // Reset per-step non-finite counters (cumulative reinit counters at indices 3, 4 stay).
        let ctr = uintArena.pointer.advanced(by: uintLayout.counters)
        ctr[0] = 0; ctr[1] = 0; ctr[2] = 0; ctr[5] = 0; ctr[6] = 0

        try runKernel(accumulatePhaseGpu: accumulatePhaseGpu)

        memcpy(&hostAction, stateArena.pointer.advanced(by: stateLayout.policyMean),
               dPolicy * MemoryLayout<Float>.stride)
        memcpy(&hostOptionAction, stateArena.pointer.advanced(by: stateLayout.optionPolicyMean),
               hostOptionAction.count * MemoryLayout<Float>.stride)
        return (PolicyMeanBatch(values: hostAction, optionValues: hostOptionAction),
                captureStats ? statsPointer.pointee : nil)
    }

    private func writeKernelConfig(paramLayout: OakMetalLayout.Param) {
        let K = OakMetalLayout.optionCount(rank: rank)
        let p = paramLayout, s = stateLayout, u = uintLayout
        var c = KernelConfig()
        c.obsDim = UInt32(obsDim); c.rank = UInt32(rank)
        c.optionCount = UInt32(K)
        c.actionParamCount = UInt32(dPolicy)
        c.obsTraceLevels = UInt32(max(OakMetalLayout.obsTraceLevelCount(obsDim: obsDim) - 1, 0))
        c.linearAnchorCount = UInt32(OakMetalLayout.linearAnchorCount(rank: rank))
        c.optionPageWidth = UInt32(OakMetalLayout.optionPageWidth)
        c.invRank = 1.0 / Float(max(rank, 1))
        c.invOptionCount = 1.0 / Float(max(K, 1))
        c.invOneInputWidth = 1.0 / Float(max(rank + dPolicy, 1))
        c.metaEta = c.invRank / Float(max(rank + dPolicy, 1))
        c.pValue = UInt32(p.value); c.pPsi = UInt32(p.psi)
        c.pOptPolicy = UInt32(p.optPolicy)
        c.pOptReward = UInt32(p.optReward); c.pOptTransition = UInt32(p.optTransition)
        c.pOptDuration = UInt32(p.optDuration)
        c.pPsiCross = UInt32(p.psiCross)
        c.pOneReward = UInt32(p.oneReward)
        c.pGvfRate = UInt32(p.gvfRate)
        c.pGvfValue = UInt32(p.gvfValue)
        c.sObservation = UInt32(s.observation); c.sObsMean = UInt32(s.obsMean); c.sObsSM = UInt32(s.obsSM)
        c.sState = UInt32(s.state)
        c.sPolicyMean = UInt32(s.policyMean)
        c.sActionSignal = UInt32(s.actionSignal)
        c.sOptionScoreGrad = UInt32(s.optionScoreGrad)
        c.sOptionPolicyMean = UInt32(s.optionPolicyMean)
        c.sImportance = UInt32(s.importance)
        c.sReward = UInt32(s.reward); c.sTd = UInt32(s.td); c.sTdPsi = UInt32(s.tdPsi)
        c.sUtilFeat = UInt32(s.utilFeat); c.sUtilOpt = UInt32(s.utilOpt)
        c.sBonusW = UInt32(s.bonusW)
        c.sFeatureMean = UInt32(s.featureMean)
        c.sFeatureSM = UInt32(s.featureSM)
        c.sAlpha = UInt32(s.alpha); c.sBeta = UInt32(s.beta); c.sH = UInt32(s.h)
        c.sGradEnergy = UInt32(s.gradEnergy)
        c.sAlphaPsi = UInt32(s.alphaPsi)
        c.sBetaPsi = UInt32(s.betaPsi)
        c.sHPsi = UInt32(s.hPsi)
        c.sGradEnergyPsi = UInt32(s.gradEnergyPsi)
        c.sAlphaModel = UInt32(s.alphaModel)
        c.sBetaModel = UInt32(s.betaModel)
        c.sHModel = UInt32(s.hModel)
        c.sGradEnergyModel = UInt32(s.gradEnergyModel)
        c.sAlphaGvf = UInt32(s.alphaGvf)
        c.sBetaGvf = UInt32(s.betaGvf)
        c.sHGvf = UInt32(s.hGvf)
        c.sGradEnergyGvf = UInt32(s.gradEnergyGvf)
        c.sActorAdv = UInt32(s.actorAdv)
        c.sNTdMat = UInt32(s.nTdMat)
        c.sOptRewardTd = UInt32(s.optRewardTd)
        c.sHeadDenom = UInt32(s.headDenom)
        c.sCrossTd = UInt32(s.crossTd)
        c.sOneRewardTd = UInt32(s.oneRewardTd)
        c.sOneStepDenom = UInt32(s.oneStepDenom)
        c.sDurationTd = UInt32(s.durationTd)
        c.sDurationHat = UInt32(s.durationHat)
        c.sObsTraceMulti = UInt32(s.obsTraceMulti)
        c.sPrioOpt = UInt32(s.prioOpt)
        c.sContinuation = UInt32(s.continuation)
        c.sLambda = UInt32(s.lambda)
        c.sKeyboardScore = UInt32(s.keyboardScore)
        c.sGvfTd = UInt32(s.gvfTd)
        c.sFeatureTraceDenom = UInt32(s.featureTraceDenom)
        c.sModelDenom = UInt32(s.modelDenom)
        c.sOptionPlanningUse = UInt32(s.optionPlanningUse)
        c.sValueGauge = UInt32(s.valueGauge)
        c.uCounters = UInt32(u.counters)
        c.uOptFeature = UInt32(u.optFeature)
        c.uFeatureAge = UInt32(u.featureAge)
        c.uOptAge = UInt32(u.optAge)
        c.uFeatureParentA = UInt32(u.featureParentA)
        c.uFeatureParentB = UInt32(u.featureParentB)
        c.uFeatureKind = UInt32(u.featureKind)
        c.uFeatureBirthCount = UInt32(u.featureBirthCount)
        c.uSelectedOption = UInt32(u.selectedOption)
        c.uHighModelOption = UInt32(u.highModelOption)
        c.uModelAge = UInt32(u.modelAge)
        c.uGvfAge = UInt32(u.gvfAge)
        c.uBadFeatureAge = UInt32(u.badFeatureAge)
        c.uPrimitiveChannel = UInt32(u.primitiveChannel)
        configPointer.pointee = c
    }

    private func alignedGridWidth(threadCount: Int, pipeline: MTLComputePipelineState) -> Int {
        let align = max(Int(pipeline.threadExecutionWidth), 32)
        let n = max(threadCount, 1)
        return ((n + align - 1) / align) * align
    }

    private func dispatchPhase(_ i: Int, on enc: MTLComputeCommandEncoder) {
        enc.setComputePipelineState(pipelines[i])
        for j in 0..<resourceList.count { enc.setBuffer(resourceList[j], offset: 0, index: j) }
        let phaseTgSize = phaseThreadgroupSizes[i]
        if phaseTgSize > 0 {
            enc.dispatchThreadgroups(
                MTLSize(width: max(phaseGridWidths[i], 1), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: phaseTgSize, height: 1, depth: 1))
        } else {
            let tg = min(max(pipelines[i].threadExecutionWidth, 1),
                         pipelines[i].maxTotalThreadsPerThreadgroup)
            let w = alignedGridWidth(threadCount: phaseGridWidths[i], pipeline: pipelines[i])
            enc.dispatchThreads(MTLSize(width: w, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(tg, w), height: 1, depth: 1))
        }
    }

    private func runKernel(accumulatePhaseGpu: Bool) throws {
        let split = diagnosticGpuPhaseTiming && accumulatePhaseGpu
        if !split {
            // Hot path: one command buffer, one encoder, one dispatch per phase with a memory
            // barrier between phases (required for RAW/WAW correctness on shared buffers).
            try autoreleasepool {
                guard let cb = queue.makeCommandBufferWithUnretainedReferences() else {
                    throw OakError.message("Failed to create Metal command buffer.")
                }
                guard let enc = cb.makeComputeCommandEncoder() else {
                    throw OakError.message("Failed to create Metal compute encoder.")
                }
                for i in 0..<pipelines.count {
                    dispatchPhase(i, on: enc)
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
                dispatchPhase(i, on: enc)
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
    let optionCount: Int
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
    var prevOptionScoreGrad: [Float]
    var prevImportance: [Float]
    var randPolicyMean: [Float]
    var randSeed: UInt64
    var measuredSteps = 0
    var measuredSeconds = 0.0
    var measurementStart: UInt64?
    var rewardSum = 0.0
    var doneCount = 0
    var pulse = 0
    var health: HealthAccumulator

    init(env: EnvPoolEnv, seed: Int?, optionCount: Int) {
        prevPolicyMean = Array(repeating: 0, count: env.dPolicy)
        prevActionSignal = Array(repeating: 0, count: env.dPolicy)
        prevOptionScoreGrad = Array(repeating: 0, count: max(optionCount, 0) * env.dPolicy)
        prevImportance = Array(repeating: 0, count: max(optionCount, 0))
        randPolicyMean = Array(repeating: 0, count: env.dPolicy)
        randSeed = UInt64(bitPattern: Int64(seed ?? Int(oakUnseededSeed)))
        // rank = obsDim per the agent's structural choice (see OakMetalEngine init precondition).
        health = HealthAccumulator(obsDim: env.obsDim)
    }
}

// =================================================================================================
// CLI main.
// =================================================================================================
@main struct Main {
    @discardableResult
    private static func configureHotPathQoS() -> Int32 {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    }

    static func main() {
        _ = configureHotPathQoS()
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            switch args.first {
            case "benchmark": try benchmark()
            case "benchmark-worker": try benchmarkWorker(args.dropFirst().first)
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
        if policy == .oak, ProcessInfo.processInfo.environment["OAK_STOP_AFTER_STEPS"] == nil {
            fputs("OaK: OAK_STOP_AFTER_STEPS unset (short run). See AGENTS.md.\n", stderr)
        }
        let dir = RunSnapshot.resolveMetricsDirectory(explicit: nil)
        let (_, metricsURL) = try runSingle(envId: envId, policy: policy, seed: seed,
                                            stopAfter: stopAfter, device: device,
                                            command: "run", metricsDir: dir)
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

    private static func resolveBenchmarkWorkingDirectory(root: URL) -> URL {
        if let raw = ProcessInfo.processInfo.environment["OAK_BENCH_DIR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
        }
        if let raw = ProcessInfo.processInfo.environment["OAK_METRICS_STORAGE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
        }
        return root.appendingPathComponent(".oak-benchmarks/\(oakBenchmarksRunId())", isDirectory: true)
    }

    private static func benchmark() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let env = ProcessInfo.processInfo.environment
        let envId = env["OAK_BENCH_ENV"] ?? env["OAK_ENV"] ?? "HalfCheetah-v4"
        guard !envId.contains(",") else {
            throw OakError.message("OAK_BENCH_ENV is single-env by design; run separate benchmarks for multiple envs.")
        }
        let seedStart = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_START"].flatMap(Int.init) ?? 0
        let seedCount = ProcessInfo.processInfo.environment["OAK_BENCH_SEED_COUNT"].flatMap(Int.init) ?? 5
        let stopAfter = max(ProcessInfo.processInfo.environment["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 500000, 1)
        let policies = try (ProcessInfo.processInfo.environment["OAK_BENCH_POLICIES"] ?? "oak,random")
            .split(separator: ",").map { try parsePolicy(String($0)) }
        let concurrency = max(ProcessInfo.processInfo.environment["OAK_BENCH_CONCURRENCY"].flatMap(Int.init) ?? 8, 1)
        let benchDir = resolveBenchmarkWorkingDirectory(root: root)
        try FileManager.default.createDirectory(at: benchDir, withIntermediateDirectories: true)

        struct WorkItem { let seed: Int; let policy: Policy }
        var work: [WorkItem] = []
        for seed in seedStart..<(seedStart + seedCount) {
            for policy in policies {
                let stem = RunSnapshot.snapshotStem(envId: envId, policy: policy, seed: seed)
                let snapURL = benchDir
                    .appendingPathComponent(envId, isDirectory: true)
                    .appendingPathComponent(RunSnapshot.seedFolderName(seed), isDirectory: true)
                    .appendingPathComponent("\(stem).json", isDirectory: false)
                if FileManager.default.fileExists(atPath: snapURL.path) { continue }
                work.append(WorkItem(seed: seed, policy: policy))
            }
        }
        let totalWork = work.count
        fputs("OaK benchmark: \(totalWork) \(envId) work items, single-stream B=1, N=\(concurrency) worker processes (steps=\(stopAfter), seeds=\(seedCount))\n", stderr)
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
        queue.qualityOfService = .userInteractive

        for item in work {
            queue.addOperation {
                _ = configureHotPathQoS()
                do {
                    try runBenchmarkWorkerProcess(envId: envId, policy: item.policy,
                                                  seed: item.seed, stopAfter: stopAfter,
                                                  metricsDir: benchDir)
                    state.lock.lock()
                    state.completed += 1
                    let n = state.completed
                    state.lock.unlock()
                    if n % progressEvery == 0 || n == totalWork {
                        fputs("OaK benchmark progress: \(n)/\(totalWork)\n", stderr)
                        fflush(stderr)
                    }
                } catch {
                    let stem = RunSnapshot.snapshotStem(envId: envId, policy: item.policy, seed: item.seed)
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

    private static func benchmarkWorker(_ envArg: String?) throws {
        let env = ProcessInfo.processInfo.environment
        let envId = env["OAK_ENV"] ?? envArg ?? ""
        guard !envId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OakError.message("Missing benchmark-worker environment id.")
        }
        let policy = try parsePolicy(env["OAK_POLICY"] ?? Policy.oak.rawValue)
        let seed = env["OAK_SEED"].flatMap(Int.init)
        let stopAfter = max(env["OAK_STOP_AFTER_STEPS"].flatMap(Int.init)
                            ?? env["OAK_BENCH_STEPS"].flatMap(Int.init) ?? 1, 1)
        let device = try metalDevice()
        let dir = RunSnapshot.resolveMetricsDirectory(explicit: nil)
        _ = try runSingle(envId: envId, policy: policy, seed: seed,
                          stopAfter: stopAfter, device: device,
                          command: "benchmark", metricsDir: dir).0
    }

    private static func currentExecutableURL() throws -> URL {
        let raw = CommandLine.arguments[0]
        let url = raw.hasPrefix("/")
            ? URL(fileURLWithPath: raw)
            : URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(raw)
        guard FileManager.default.isExecutableFile(atPath: url.path) else {
            throw OakError.message("Cannot locate benchmark worker executable at \(url.path)")
        }
        return url
    }

    private static func runBenchmarkWorkerProcess(envId: String, policy: Policy, seed: Int,
                                                  stopAfter: Int, metricsDir: URL) throws {
        let stem = RunSnapshot.snapshotStem(envId: envId, policy: policy, seed: seed)
        let logURL = metricsDir.appendingPathComponent("\(stem).worker.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: logURL)
        defer { try? log.close() }

        var env = ProcessInfo.processInfo.environment
        env["OAK_ENV"] = envId
        env["OAK_POLICY"] = policy.rawValue
        env["OAK_SEED"] = "\(seed)"
        env["OAK_STOP_AFTER_STEPS"] = "\(stopAfter)"
        env["OAK_METRICS_STORAGE"] = metricsDir.path

        let p = Process()
        p.executableURL = try currentExecutableURL()
        p.arguments = ["benchmark-worker", envId]
        p.environment = env
        p.standardOutput = log
        p.standardError = log
        try p.run()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            throw OakError.message("Benchmark worker failed for \(envId) seed \(seed) policy \(policy.rawValue) (status \(p.terminationStatus)); see \(logURL.path)")
        }
    }

    private static func parsePolicy(_ raw: String) throws -> Policy {
        guard let p = Policy(rawValue: raw.lowercased()) else {
            throw OakError.message("Unknown policy '\(raw)'.")
        }
        return p
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
        return PolicyMeanBatch(values: floats, optionValues: nil)
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
        // Plan Step 2: rank R is the active feature/subtask/option budget, independent of
        // env input dim. K = R; the 32-wide page is only the hardware scheduling width.
        let defaultRank = max(env.obsDim, env.dPolicy * 2)
        let rawRank = max(ProcessInfo.processInfo.environment["OAK_RANK"].flatMap(Int.init)
                          ?? defaultRank, 2)
        let rank = rawRank
        guard rank <= 256 else {
            throw OakError.message("OAK rank \(rank) exceeds oak_plan's 256-threadgroup-storage cap. Set OAK_RANK<=256 or redesign oak_plan storage.")
        }
        let engine = policy == .oak
            ? try OakMetalEngine(obsDim: env.obsDim, rank: rank, dPolicy: env.dPolicy,
                                 device: device, seed: seed,
                                 diagnosticGpuPhaseTiming: includeDiagnostic)
            : nil
        try env.reset()
        return RunContext(envId: envId, policy: policy, seed: seed,
                          stopAfter: stopAfter,
                          optionCount: policy == .oak ? OakMetalLayout.optionCount(rank: rank) : 0,
                          command: command,
                          includeDiagnostic: includeDiagnostic, metricsDir: dir,
                          env: env, engine: engine)
    }

    private static func emitSnapshot(_ ctx: RunContext, _ state: RunState, seq: Int?) -> RunMetricsFile {
        let measured = max(state.measuredSteps, 1)
        let merged: RunDiagnostic? = state.health.hasMeasuredSteps ? {
            let (mm, mi) = ctx.engine?.metalPhaseGpuDiagnostics(measuredSteps: measured) ?? (nil, nil)
            return RunDiagnostic(
                metalPhaseGpuMsMean: mm, metalPhaseGpuTimingInvalid: mi)
        }() : nil
        let diagOut: RunDiagnostic? = ((seq != nil && ctx.policy == .oak) || (seq == nil && ctx.includeDiagnostic))
                                       ? merged : nil
        let sps = state.measuredSeconds > 0 ? Double(state.measuredSteps) / state.measuredSeconds : nil
        let reason = seq == nil ? "measured_steps" : "heartbeat"
        return RunMetricsFile(
            schemaVersion: 3, command: ctx.command, envId: ctx.envId, policy: ctx.policy, seed: ctx.seed,
            numEnvs: 1, stopAfterMeasuredSteps: ctx.stopAfter,
            observationDim: ctx.env.obsDim, actionDim: ctx.env.actDim,
            actionDiscreteCount: ctx.env.actionDiscreteCount,
            measuredStepCount: state.measuredSteps,
            measuredSeconds: state.measuredSeconds,
            stepsPerSecond: sps,
            heartbeatSequence: seq, rewardSum: state.rewardSum, doneCount: state.doneCount,
            stopReason: reason,
            health: state.health.snapshot(continuingRestartCount: state.doneCount),
            diagnostic: diagOut)
    }

    private static func measurementLoop(_ ctx: RunContext, _ state: RunState) throws {
        let defaultHb = ctx.command == "diagnose" ? 0 : max(ctx.stopAfter / 10, 1)
        let hb = max(ProcessInfo.processInfo.environment["OAK_HEARTBEAT_STEPS"].flatMap(Int.init) ?? defaultHb, 0)
        // Continual: every step is a measured step, no warmup. Plan Inv-5: no special training
        // phases. Step 0 timer is established on the first iteration.
        if state.measurementStart == nil { state.measurementStart = DispatchTime.now().uptimeNanoseconds }
        while state.measuredSteps < ctx.stopAfter {
            let nextMeasuredStep = state.measuredSteps + 1
            let willHeartbeat = hb > 0 && state.measuredSteps / hb < nextMeasuredStep / hb
            let willFinish = nextMeasuredStep == ctx.stopAfter
            let captureKernelStats = ctx.policy == .oak && (willHeartbeat || willFinish)
            let policyMean: PolicyMeanBatch, stats: KernelStats?
            if let engine = ctx.engine {
                let out = try ctx.env.observationScratch.withUnsafeBufferPointer { buf in
                    try engine.step(obs: buf, prevActionSignal: state.prevActionSignal,
                                    prevOptionScoreGrad: state.prevOptionScoreGrad,
                                    prevImportance: state.prevImportance,
                                    prevReward: state.prevReward,
                                    accumulatePhaseGpu: true,
                                    captureStats: captureKernelStats)
                }
                policyMean = out.0; stats = out.1
            } else {
                policyMean = randomPolicyMean(env: ctx.env, stepCount: &state.randSeed,
                                              floats: &state.randPolicyMean)
                stats = nil
            }
            try ctx.env.sampleAndStep(policy: policyMean)
            state.measuredSteps += 1
            if let s = state.measurementStart {
                state.measuredSeconds = Double(DispatchTime.now().uptimeNanoseconds - s) / 1_000_000_000
            }
            state.doneCount += Int(ctx.env.doneScratch[0])
            let v = ctx.env.rewardScratch[0]
            guard v.isFinite else { throw OakError.message("Environment produced a non-finite reward.") }
            state.rewardSum += Double(v)
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
            if willHeartbeat {
                state.pulse += 1
                _ = try RunSnapshot.persist(emitSnapshot(ctx, state, seq: state.pulse), metricsDir: ctx.metricsDir)
            }
            state.prevReward = ctx.env.rewardScratch[0]
            _ = state.prevPolicyMean.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress!, policyMean.values, policyMean.values.count * MemoryLayout<Float>.stride)
            }
            if ctx.env.optionScoreGradScratch.count == state.prevOptionScoreGrad.count {
                _ = state.prevOptionScoreGrad.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress!, ctx.env.optionScoreGradScratch,
                           ctx.env.optionScoreGradScratch.count * MemoryLayout<Float>.stride)
                }
            }
            _ = state.prevActionSignal.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress!, ctx.env.actionSignalScratch,
                       ctx.env.actionSignalScratch.count * MemoryLayout<Float>.stride)
            }
            if ctx.env.importanceScratch.count == state.prevImportance.count {
                _ = state.prevImportance.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress!, ctx.env.importanceScratch,
                           ctx.env.importanceScratch.count * MemoryLayout<Float>.stride)
                }
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
        let state = RunState(env: ctx.env, seed: seed, optionCount: ctx.optionCount)
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
