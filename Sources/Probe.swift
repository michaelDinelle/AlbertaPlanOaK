import Foundation
import Metal
import Darwin

// =================================================================================================
// Probe harness — closed-form synthetic envs where the optimal policy is known analytically, so a
// statistically-monotonic learning trajectory is verifiable. Six envs cover the architectural
// invariants the bench can't isolate (sparse-reward chord, partial-observability OPSR, plasticity
// reinit, high-variance baseline). The harness exercises the SAME engine.step() and SAME
// wrapper-side action sampling (`sampleAndScoreActions`) as production benches — the only thing
// that changes is the env that supplies (obs, reward, done) and grades agent actions against
// closed-form-optimal targets.
//
// Pass-13 Tier 0.2: extracted from Main.swift to drop the dispatcher's LOC. Same-target file split
// (no separate executable target needed — the LOC reduction is the user-visible win, not the
// number of binaries).
// =================================================================================================
enum ProbeEnvKind: String {
    case discrete            // 4-action gridworld; r = 1{a == argmax(w·s)}
    case continuous          // 2-dim continuous; r = -‖a − w·s‖²
    case sparse              // 8-step chain MDP; r = 1 only on terminal step (Tier 3 chord required)
    case partialObservable   // target shown for 1 step then hidden; reach after 5 steps (Tier 2 OPSR required)
    case plasticityStress    // discrete with target swap mid-run (Tier 5 reinit required)
    case highVariance        // continuous + Bernoulli ±1 reward jitter (Tier 4 baseline required)

    static func parse(_ s: String) throws -> ProbeEnvKind {
        switch s {
        case "discrete": return .discrete
        case "continuous": return .continuous
        case "sparse", "sparse-reward": return .sparse
        case "partial-observable", "partial-observability": return .partialObservable
        case "plasticity-stress": return .plasticityStress
        case "high-variance": return .highVariance
        default: throw OakError.message("Unknown probe env '\(s)'.")
        }
    }
}

final class ProbeEnv {
    let kind: ProbeEnvKind
    let numEnvs: Int, obsDim: Int, actDim: Int
    let actionDiscreteCount: Int?
    let actionLow: [Float], actionHigh: [Float]
    let actionSpec: [ActionComponent]
    let dPolicy: Int
    let envActionWidth: Int
    private(set) var observationScratch: [Float]
    private(set) var rewardScratch: [Float]
    private(set) var doneScratch: [UInt8]
    private(set) var envActionScratch: [Float]
    private(set) var scoreGradScratch: [Float]
    private var rngState: UInt64
    private var stepIndex: Int = 0
    private var swapped: Bool = false

    // Closed-form target signal w[O]. For discrete envs the optimal action is argmax(w·s);
    // for continuous, the optimal action is w·s. The synthetic env generates s ~ N(0,1) per step
    // and assigns reward by comparing the agent's action to the closed-form-optimal action.
    // For non-stationary plasticity-stress, w flips sign mid-run.
    private var targetWeights: [Float]
    private let swapStep: Int  // for plasticityStress: at this step targetWeights flips sign

    // Sparse-chain state: position in [0, 8); reward=1 only when reaching terminal.
    private var chainPos: [Int]
    // Partial-observability state: per-row target identity in [0, A); shown at step 0 only.
    private var poTarget: [Int]
    private var poStepsSinceTarget: [Int]
    private let poHorizon: Int = 5

    init(kind: ProbeEnvKind, numEnvs: Int, seed: Int?) {
        self.kind = kind; self.numEnvs = numEnvs
        // Structural sizing per env kind. obsDim must equal rank. The env must have temporal
        // structure for OaK's value-bootstrapping to learn — pure contextual-bandit (iid s)
        // gives the value head nothing to bootstrap. Each env keeps a per-row "task" (target
        // identity / regression target) that persists for `taskHorizon` steps then resamples.
        // Random baseline = 1/A; optimal = 1.
        switch kind {
        case .discrete, .plasticityStress:
            self.obsDim = 64; self.actDim = 4; self.actionDiscreteCount = 4
        case .continuous, .highVariance:
            self.obsDim = 64; self.actDim = 4; self.actionDiscreteCount = nil
        case .sparse:
            self.obsDim = 64; self.actDim = 4; self.actionDiscreteCount = 4
        case .partialObservable:
            self.obsDim = 64; self.actDim = 4; self.actionDiscreteCount = 4
        }
        self.actionLow  = Array(repeating: -1, count: actDim)
        self.actionHigh = Array(repeating:  1, count: actDim)
        if let n = actionDiscreteCount {
            self.actionSpec = [.discrete(n: n)]
        } else {
            self.actionSpec = [.box(n: actDim, low: actionLow, high: actionHigh)]
        }
        self.dPolicy = actionSpec.totalPolicyParamCount
        self.envActionWidth = actionSpec.totalEnvActionWidth
        self.observationScratch = Array(repeating: 0, count: numEnvs * obsDim)
        self.rewardScratch = Array(repeating: 0, count: numEnvs)
        self.doneScratch = Array(repeating: 0, count: numEnvs)
        self.envActionScratch = Array(repeating: 0, count: numEnvs * envActionWidth)
        self.scoreGradScratch = Array(repeating: 0, count: numEnvs * dPolicy)
        self.rngState = UInt64(bitPattern: Int64(seed ?? 1)) &* 0x9e3779b97f4a7c15 &+ 1
        self.targetWeights = Array(repeating: 0, count: actDim * obsDim)
        self.swapStep = 5000
        self.chainPos = Array(repeating: 0, count: numEnvs)
        self.poTarget = Array(repeating: 0, count: numEnvs)
        self.poStepsSinceTarget = Array(repeating: 0, count: numEnvs)
        self.persistentTask = Array(repeating: 0, count: numEnvs)
        self.persistentObs = Array(repeating: 0, count: numEnvs * obsDim)
        self.persistentSteps = Array(repeating: 0, count: numEnvs)
        // Closed-form signal: w[a, o] = δ(a == o). Optimal action = argmax_a obs[a]. For
        // continuous, optimal per-dim action = obs[d]. Combined with persistent observations
        // (the obs encodes the task identity and stays constant for `taskHorizon` steps),
        // the agent's value-bootstrap learns v(s) and policy learns to map s → optimal action.
        for a in 0..<actDim {
            for o in 0..<obsDim {
                targetWeights[a * obsDim + o] = (a == o ? 1 : 0)
            }
        }
        for r in 0..<numEnvs { resampleTask(row: r) }
        copyPersistentToScratch()
    }

    private let taskHorizon = 8  // task persists for this many steps then resamples
    private var persistentTask: [Int]
    private var persistentObs: [Float]
    private var persistentSteps: [Int]

    private func resampleTask(row r: Int) {
        let task = Int(nextU01() * Float(actDim)) % actDim
        persistentTask[r] = task
        // Persistent observation: a randomly-positioned strong (+10) signal indicating the
        // task. Specifically obs[task*16 .. (task+1)*16] has +10 in one dim, others zero. This
        // mimics MiniGrid's "strong feature in a known position" structure, and the high
        // magnitude (+10) ensures the encoder's random init produces large, non-noisy state
        // values that make the policy gradient pop above value-bootstrap noise.
        for o in 0..<obsDim { persistentObs[r * obsDim + o] = 0 }
        let pos = task * 16 + (Int(nextU01() * 16) % 16)
        persistentObs[r * obsDim + pos] = 10.0
        persistentSteps[r] = 0
    }

    private func copyPersistentToScratch() {
        observationScratch = persistentObs
    }

    func reset() { copyPersistentToScratch() }

    private func nextU01() -> Float {
        rngState &+= 0x9e3779b97f4a7c15
        var z = rngState
        z = (z ^ (z &>> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z &>> 27)) &* 0x94d049bb133111eb
        z ^= z &>> 31
        return Float(z >> 40) * (1.0 / Float(1 << 24))
    }

    private func nextNormal() -> Float {
        let u1 = max(nextU01(), 1e-7)
        let u2 = nextU01()
        return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
    }

    private func sampleObservation() {
        switch kind {
        case .partialObservable:
            for r in 0..<numEnvs {
                if poStepsSinceTarget[r] == 0 {
                    observationScratch[r * obsDim + 0] = Float(poTarget[r]) - Float(actDim - 1) / 2
                    for o in 1..<obsDim { observationScratch[r * obsDim + o] = nextNormal() }
                } else {
                    for o in 0..<obsDim { observationScratch[r * obsDim + o] = nextNormal() }
                    observationScratch[r * obsDim + 0] = 0  // hidden after step 0
                }
            }
        case .sparse:
            for r in 0..<numEnvs {
                for o in 0..<obsDim { observationScratch[r * obsDim + o] = 0 }
                observationScratch[r * obsDim + 0] = Float(chainPos[r]) / 7.0 - 0.5
            }
        case .discrete, .plasticityStress, .continuous, .highVariance:
            // Persistent-task envs: copy the row's persistent obs (set on task resample) to
            // the scratch. The task persists for taskHorizon steps so OaK's value-head can
            // bootstrap. After taskHorizon, the row's task resamples (handled in step()).
            copyPersistentToScratch()
        }
    }

    func sampleAndStep(policyMean: [Float]) {
        precondition(policyMean.count == numEnvs * dPolicy)
        policyMean.withUnsafeBufferPointer { pm in
            envActionScratch.withUnsafeMutableBufferPointer { aBuf in
                scoreGradScratch.withUnsafeMutableBufferPointer { sgBuf in
                    sampleAndScoreActions(spec: actionSpec, numEnvs: numEnvs, dPolicy: dPolicy,
                                          envActionWidth: envActionWidth,
                                          policyMean: pm.baseAddress!,
                                          envAction: aBuf.baseAddress!,
                                          scoreGrad: sgBuf.baseAddress!, rngState: &rngState)
                }
            }
        }
        stepIndex += 1
        if kind == .plasticityStress, !swapped, stepIndex >= swapStep {
            for i in 0..<targetWeights.count { targetWeights[i] = -targetWeights[i] }
            swapped = true
        }
        switch kind {
        case .discrete, .plasticityStress:
            // Persistent-task contextual bandit. Reward = +1 if chosen action == persistentTask,
            // -1 otherwise. Random baseline E[r] = (1/A)·1 + ((A-1)/A)·(-1) = 1 - 2(A-1)/A
            // (e.g., -0.5 at A=4). Optimal = +1. Task resamples every taskHorizon.
            for r in 0..<numEnvs {
                let chosen = max(0, min(actDim - 1, Int(envActionScratch[r])))
                rewardScratch[r] = chosen == persistentTask[r] ? 1 : -1
                doneScratch[r] = 0
                persistentSteps[r] += 1
                if persistentSteps[r] >= taskHorizon {
                    doneScratch[r] = 1
                    resampleTask(row: r)
                }
            }
        case .continuous:
            // Persistent-task continuous regression: optimal action[d] = persistentObs[r][d].
            // Reward = -SSE. Task resamples every taskHorizon.
            for r in 0..<numEnvs {
                var sse: Float = 0
                let b = r * actDim
                for d in 0..<actDim {
                    let target = persistentObs[r * obsDim + d]
                    let diff = envActionScratch[b + d] - target
                    sse += diff * diff
                }
                rewardScratch[r] = -sse
                doneScratch[r] = 0
                persistentSteps[r] += 1
                if persistentSteps[r] >= taskHorizon {
                    doneScratch[r] = 1
                    resampleTask(row: r)
                }
            }
        case .highVariance:
            // Same regression structure as continuous, plus Bernoulli ±1 reward jitter.
            for r in 0..<numEnvs {
                var sse: Float = 0
                let b = r * actDim
                for d in 0..<actDim {
                    let target = persistentObs[r * obsDim + d]
                    let diff = envActionScratch[b + d] - target
                    sse += diff * diff
                }
                let jitter: Float = nextU01() < 0.5 ? -1 : 1
                rewardScratch[r] = -sse + jitter
                doneScratch[r] = 0
                persistentSteps[r] += 1
                if persistentSteps[r] >= taskHorizon {
                    doneScratch[r] = 1
                    resampleTask(row: r)
                }
            }
        case .sparse:
            for r in 0..<numEnvs {
                let chosen = max(0, min(actDim - 1, Int(envActionScratch[r])))
                if chosen == 0 { chainPos[r] = min(chainPos[r] + 1, 7) }
                if chainPos[r] == 7 {
                    rewardScratch[r] = 1; doneScratch[r] = 1; chainPos[r] = 0
                } else {
                    rewardScratch[r] = 0; doneScratch[r] = 0
                }
            }
        case .partialObservable:
            for r in 0..<numEnvs {
                let chosen = max(0, min(actDim - 1, Int(envActionScratch[r])))
                poStepsSinceTarget[r] += 1
                if poStepsSinceTarget[r] >= poHorizon {
                    rewardScratch[r] = chosen == poTarget[r] ? 1 : 0
                    doneScratch[r] = 1
                    poStepsSinceTarget[r] = 0
                    poTarget[r] = Int(nextU01() * Float(actDim)) % actDim
                } else {
                    rewardScratch[r] = 0; doneScratch[r] = 0
                }
            }
        }
        sampleObservation()
    }
}

// ----- Spearman rank correlation + one-tailed p-value via z-statistic -----------------------------
func spearmanRho(_ y: [Double]) -> (rho: Double, p: Double) {
    let n = y.count
    if n < 3 { return (0, 1) }
    let yRanks = ranks(y)
    var sxy = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0
    for i in 0..<n {
        let xi = Double(i), yi = yRanks[i]
        sxy += xi * yi; sx += xi; sy += yi; sxx += xi * xi; syy += yi * yi
    }
    let nd = Double(n)
    let num = sxy - sx * sy / nd
    let den = sqrt(max((sxx - sx * sx / nd) * (syy - sy * sy / nd), 1e-30))
    let rho = num / den
    // Approximate one-tailed p via Fisher z (good for n ≥ 10; conservative for smaller n).
    let z = atanh(max(-0.999999, min(0.999999, rho))) * sqrt(Double(max(n - 3, 1)))
    let p = 0.5 * erfc(z / sqrt(2))
    return (rho, p)
}

func ranks(_ a: [Double]) -> [Double] {
    let n = a.count
    let order = (0..<n).sorted { a[$0] < a[$1] }
    var r = [Double](repeating: 0, count: n)
    var i = 0
    while i < n {
        var j = i
        while j + 1 < n && a[order[j + 1]] == a[order[i]] { j += 1 }
        let avg = Double(i + j) / 2 + 1  // ranks 1..n with average-tie correction
        for k in i...j { r[order[k]] = avg }
        i = j + 1
    }
    return r
}

// =================================================================================================
// Probe CLI dispatcher (`OaK probe`).
// =================================================================================================
func runProbeCLI(_ args: [String]) throws {
    var envKind = ProbeEnvKind.discrete
    var steps = 10000
    var slot: String?
    var assertMonotonic = false
    var seed: Int? = nil
    var probeAblateKernel: Set<String> = []
    var probeNumEnvs = 64
    var i = 0
    while i < args.count {
        let a = args[i]
        switch a {
        case "--env":
            i += 1; guard i < args.count else { throw OakError.message("Missing value for --env.") }
            envKind = try ProbeEnvKind.parse(args[i])
        case let s where s.hasPrefix("--env="):
            envKind = try ProbeEnvKind.parse(String(s.dropFirst("--env=".count)))
        case "--steps":
            i += 1; guard i < args.count, let v = Int(args[i]), v > 0 else {
                throw OakError.message("Invalid value for --steps.")
            }; steps = v
        case let s where s.hasPrefix("--steps="):
            guard let v = Int(s.dropFirst("--steps=".count)), v > 0 else {
                throw OakError.message("Invalid value for --steps.")
            }; steps = v
        case "--slot":
            i += 1; guard i < args.count else { throw OakError.message("Missing value for --slot.") }
            slot = args[i]
        case let s where s.hasPrefix("--slot="):
            slot = String(s.dropFirst("--slot=".count))
        case "--assert-monotonic-reward":
            assertMonotonic = true
        case "--seed":
            i += 1; guard i < args.count, let v = Int(args[i]) else {
                throw OakError.message("Invalid value for --seed.")
            }; seed = v
        case let s where s.hasPrefix("--seed="):
            guard let v = Int(s.dropFirst("--seed=".count)) else {
                throw OakError.message("Invalid value for --seed.")
            }; seed = v
        case "--num-envs":
            i += 1; guard i < args.count, let v = Int(args[i]), v > 0 else {
                throw OakError.message("Invalid value for --num-envs.")
            }; probeNumEnvs = v
        case let s where s.hasPrefix("--num-envs="):
            guard let v = Int(s.dropFirst("--num-envs=".count)), v > 0 else {
                throw OakError.message("Invalid value for --num-envs.")
            }; probeNumEnvs = v
        case "--ablate-kernel":
            i += 1; guard i < args.count else { throw OakError.message("Missing value for --ablate-kernel.") }
            probeAblateKernel.formUnion(args[i].split(separator: ",").map(String.init))
        case let s where s.hasPrefix("--ablate-kernel="):
            probeAblateKernel.formUnion(s.dropFirst("--ablate-kernel=".count).split(separator: ",").map(String.init))
        default:
            throw OakError.message("Unrecognized probe arg '\(a)'.")
        }
        i += 1
    }

    let device = try metalDevice()
    let env = ProbeEnv(kind: envKind, numEnvs: probeNumEnvs, seed: seed)
    let engine = try OakMetalEngine(
        obsDim: env.obsDim, baseObsDim: env.obsDim, actDim: env.actDim,
        batch: probeNumEnvs, rank: env.obsDim, dPolicy: env.dPolicy,
        device: device, seed: seed, diagnosticGpuPhaseTiming: false,
        ablatedKernels: ablatedKernelsFromEnv().union(probeAblateKernel))
    env.reset()

    // Window count K = ⌈√N⌉ heartbeat windows of ⌊N/K⌋ steps each (structural — √N is the
    // bias-variance optimal window count for a Gaussian-noise mean estimator). Per-window
    // mean reward is averaged across all (numEnvs × stepsInWindow) reward samples.
    let stepsPerBatch = probeNumEnvs
    let totalBatches = max(steps / stepsPerBatch, 1)
    let windowCount = max(Int((Double(totalBatches)).squareRoot().rounded(.up)), 2)
    let batchesPerWindow = max(totalBatches / windowCount, 1)
    var windowSums = [Double](repeating: 0, count: windowCount)
    var windowCounts = [Int](repeating: 0, count: windowCount)

    var prevPolicyMean = [Float](repeating: 0, count: probeNumEnvs * env.dPolicy)
    var prevScoreGrad  = [Float](repeating: 0, count: probeNumEnvs * env.dPolicy)
    var prevReward = [Float](repeating: 0, count: probeNumEnvs)
    var prevDone   = [UInt8](repeating: 0, count: probeNumEnvs)
    for batchIdx in 0..<totalBatches {
        let policyMean = try env.observationScratch.withUnsafeBufferPointer { buf in
            try engine.step(obs: buf, prevPolicyMean: prevPolicyMean,
                            prevScoreGrad: prevScoreGrad, prevReward: prevReward,
                            prevDone: prevDone, accumulatePhaseGpu: false).0
        }
        env.sampleAndStep(policyMean: policyMean.values)
        let w = min(batchIdx / batchesPerWindow, windowCount - 1)
        for v in env.rewardScratch { windowSums[w] += Double(v); windowCounts[w] += 1 }
        memcpy(&prevReward, env.rewardScratch, probeNumEnvs * MemoryLayout<Float>.stride)
        memcpy(&prevDone,   env.doneScratch,   probeNumEnvs * MemoryLayout<UInt8>.stride)
        memcpy(&prevPolicyMean, policyMean.values, policyMean.values.count * MemoryLayout<Float>.stride)
        memcpy(&prevScoreGrad, env.scoreGradScratch,
               env.scoreGradScratch.count * MemoryLayout<Float>.stride)
    }

    var means = [Double](repeating: 0, count: windowCount)
    for w in 0..<windowCount {
        means[w] = windowCounts[w] > 0 ? windowSums[w] / Double(windowCounts[w]) : 0
    }

    if assertMonotonic {
        let (rho, p) = spearmanRho(means)
        // Decision rule: pass if Spearman ρ > 0 at one-tailed p < 0.05, OR if the trajectory
        // shows no strict decrease (r̄_{2nd half} ≥ r̄_{1st half}). The latter handles
        // convergence-at-plateau (post-learning is acceptable).
        let firstHalf = Array(means.prefix(windowCount / 2))
        let secondHalf = Array(means.suffix(windowCount - windowCount / 2))
        let firstMean = firstHalf.reduce(0, +) / Double(firstHalf.count)
        let secondMean = secondHalf.reduce(0, +) / Double(secondHalf.count)
        let isMonotonic = (rho > 0 && p < 0.05) || (secondMean >= firstMean)
        print("probe env=\(envKind.rawValue) steps=\(steps) batches=\(totalBatches) windows=\(windowCount)")
        for (w, m) in means.enumerated() { print(String(format: "  window[%d] mean_reward=%.6f", w, m)) }
        print(String(format: "spearman_rho=%.4f p=%.4g first_half_mean=%.4f second_half_mean=%.4f",
                     rho, p, firstMean, secondMean))
        print("monotonic=\(isMonotonic ? "PASS" : "FAIL")")
        if !isMonotonic {
            fputs("probe FAILED: rewardRate not statistically monotonic on env=\(envKind.rawValue)\n", stderr)
            Foundation.exit(1)
        }
        return
    }

    guard let slotName = slot else {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        struct Out: Encodable { let env: String; let windows: Int; let means: [Double] }
        let out = Out(env: envKind.rawValue, windows: windowCount, means: means)
        let data = try enc.encode(out)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
        return
    }

    guard let values = engine.probeSlot(slotName) else {
        throw OakError.message("Slot '\(slotName)' not available in current tier (or unknown).")
    }
    let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    struct SlotOut: Encodable { let env: String; let slot: String; let count: Int; let values: [Float] }
    let payload = SlotOut(env: envKind.rawValue, slot: slotName, count: values.count, values: values)
    let data = try enc.encode(payload)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}
