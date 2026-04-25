import Foundation
import Darwin

// =================================================================================================
// Native env bridge (envpool FFI) — wrapper-side action sampling lives here too. This is the SOLE
// boundary where env-type knowledge is allowed (Inv-1; AGENTS.md hard rule). The agent is told
// only an integer dPolicy and never knows whether the env is discrete, continuous, or mixed.
//
// Pass-13 Tier 0.2: extracted from Main.swift (LOC reduction; same target — internal access stays
// the same as before).
// =================================================================================================
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

    static let shared: NativeBridge = { do { return try NativeBridge() } catch { fatalError("\(error)") } }()

    private let dylib: UnsafeMutableRawPointer
    private let assetBasePath: String
    private let openFn: OpenFn; private let closeFn: CloseFn
    private let obsDimFn: IntFn; private let actDimFn: IntFn; private let discreteCountFn: IntFn
    private let boundsFn: BoundsFn; private let resetFn: ResetFn
    private let stepContFn: StepContFn; private let stepDiscFn: StepDiscFn
    private let takeWarningFn: TakeWarningFn

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
    }

    deinit { dlclose(dylib) }

    func open(envId: String, numEnvs: Int, seed: Int?) throws -> UnsafeMutableRawPointer {
        var buf = [CChar](repeating: 0, count: 4096)
        let cap = Int32(buf.count)
        let actualSeed = Int32(seed ?? Int(oakUnseededSeed))
        let handle = envId.withCString { e in
            assetBasePath.withCString { b in
                buf.withUnsafeMutableBufferPointer { ptr in
                    openFn(e, Int32(numEnvs), actualSeed, b, ptr.baseAddress, cap)
                }
            }
        }
        guard let handle else { throw OakError.message(decodeCString(buf)) }
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
    func reset(_ h: UnsafeMutableRawPointer, obs: inout [Float],
               error: UnsafeMutablePointer<CChar>?, cap: Int32) -> Int32 {
        obs.withUnsafeMutableBufferPointer { resetFn(h, $0.baseAddress, error, cap) }
    }
    func stepContinuous(_ h: UnsafeMutableRawPointer, action: [Float],
                        obs: inout [Float], reward: inout [Float], done: inout [UInt8],
                        error: UnsafeMutablePointer<CChar>?, cap: Int32) -> Int32 {
        action.withUnsafeBufferPointer { ab in
            obs.withUnsafeMutableBufferPointer { ob in
                reward.withUnsafeMutableBufferPointer { rb in
                    done.withUnsafeMutableBufferPointer { db in
                        stepContFn(h, ab.baseAddress, ob.baseAddress, rb.baseAddress,
                                   db.baseAddress, error, cap)
                    }}}}
    }
    func stepDiscrete(_ h: UnsafeMutableRawPointer, action: [Int32],
                      obs: inout [Float], reward: inout [Float], done: inout [UInt8],
                      error: UnsafeMutablePointer<CChar>?, cap: Int32) -> Int32 {
        action.withUnsafeBufferPointer { ab in
            obs.withUnsafeMutableBufferPointer { ob in
                reward.withUnsafeMutableBufferPointer { rb in
                    done.withUnsafeMutableBufferPointer { db in
                        stepDiscFn(h, ab.baseAddress, ob.baseAddress, rb.baseAddress,
                                   db.baseAddress, error, cap)
                    }}}}
    }
    func takeWarning(error: UnsafeMutablePointer<CChar>?, cap: Int32) -> Int32 {
        takeWarningFn(error, cap)
    }
}

final class EnvPoolEnv {
    private let bridge = NativeBridge.shared
    private let handle: UnsafeMutableRawPointer
    private(set) var observationScratch: [Float]
    private(set) var rewardScratch: [Float]
    private(set) var doneScratch: [UInt8]
    let obsDim: Int, actDim: Int
    let actionDiscreteCount: Int?
    let actionLow: [Float], actionHigh: [Float]
    // Pass 11 Tier 1: env-declared structured action spec composed of one or more components
    // (Discrete N | Box A). Single-component envs derive the spec from existing bridge fields
    // (discreteCount + actionBounds); future multi-component envs (StarCraft 2, mixed) declare
    // their spec via a C++ ABI extension. The agent only sees `dPolicy`.
    let actionSpec: [ActionComponent]
    let dPolicy: Int
    let envActionWidth: Int
    private(set) var envActionScratch: [Float]
    private(set) var actionSignalScratch: [Float]
    private(set) var scoreGradScratch: [Float]
    private(set) var optionScoreGradScratch: [Float]
    private(set) var importanceScratch: [Float]
    private var optionLogProbScratch: [Float]
    private var discreteIndexScratch: [Int32]
    private var ffiMessageScratch: [CChar]
    private var rngState: UInt64

    init(id: String, numEnvs: Int, seed: Int?) throws {
        precondition(numEnvs == 1, "OaK runs one stream per agent; benchmark concurrency uses independent processes.")
        handle = try bridge.open(envId: id, numEnvs: numEnvs, seed: seed)
        obsDim = bridge.obsDim(handle); actDim = bridge.actDim(handle)
        actionDiscreteCount = bridge.discreteCount(handle)
        (actionLow, actionHigh) = bridge.actionBounds(handle, actDim: actDim)
        if let n = actionDiscreteCount {
            self.actionSpec = [.discrete(n: n)]
        } else {
            self.actionSpec = [.box(n: actDim, low: actionLow, high: actionHigh)]
        }
        self.dPolicy = actionSpec.totalPolicyParamCount
        self.envActionWidth = actionSpec.totalEnvActionWidth
        observationScratch = Array(repeating: 0, count: obsDim)
        rewardScratch = Array(repeating: 0, count: 1)
        doneScratch = Array(repeating: 0, count: 1)
        envActionScratch  = Array(repeating: 0, count: envActionWidth)
        actionSignalScratch = Array(repeating: 0, count: dPolicy)
        scoreGradScratch  = Array(repeating: 0, count: dPolicy)
        optionScoreGradScratch = []
        importanceScratch = []
        optionLogProbScratch = []
        discreteIndexScratch = Array(repeating: 0, count: 1)
        ffiMessageScratch = Array(repeating: 0, count: 4096)
        rngState = UInt64(bitPattern: Int64(seed ?? Int(oakUnseededSeed))) &* 0x9e3779b97f4a7c15 &+ 1
    }
    deinit { bridge.close(handle) }

    private func checked(_ body: (UnsafeMutablePointer<CChar>?, Int32) -> Int32) throws {
        let cap = Int32(ffiMessageScratch.count)
        let ok = ffiMessageScratch.withUnsafeMutableBufferPointer { body($0.baseAddress, cap) }
        guard ok == 1 else { throw OakError.message(decodeCString(ffiMessageScratch)) }
        let warning = ffiMessageScratch.withUnsafeMutableBufferPointer {
            bridge.takeWarning(error: $0.baseAddress, cap: cap)
        }
        if warning == 1 { throw OakError.message("MuJoCo warning: \(decodeCString(ffiMessageScratch))") }
    }
    func reset() throws {
        try checked { e, cap in bridge.reset(handle, obs: &observationScratch, error: e, cap: cap) }
    }

    private func ensureOptionScratch(optionCount: Int) {
        let n = max(0, optionCount)
        if importanceScratch.count != n {
            importanceScratch = Array(repeating: 0, count: n)
            optionLogProbScratch = Array(repeating: 0, count: n)
        }
        let ng = n * dPolicy
        if optionScoreGradScratch.count != ng {
            optionScoreGradScratch = Array(repeating: 0, count: ng)
        }
    }

    // Wrapper-owned sampling + scoreGrad. Reads policyMean[B · dPolicy], samples per-component,
    // writes envActionScratch + actionSignalScratch + scoreGradScratch, dispatches to the appropriate bridge step
    // function. This is the SOLE site that knows the env's action-space type — Inv-1 boundary.
    func sampleAndStep(policy: PolicyMeanBatch) throws {
        let policyMean = policy.values
        precondition(policyMean.count == dPolicy)
        let optionCount = policy.optionValues.map { $0.count / max(dPolicy, 1) } ?? 0
        ensureOptionScratch(optionCount: optionCount)
        policyMean.withUnsafeBufferPointer { pmBuf in
            envActionScratch.withUnsafeMutableBufferPointer { aBuf in
                actionSignalScratch.withUnsafeMutableBufferPointer { actBuf in
                    scoreGradScratch.withUnsafeMutableBufferPointer { sgBuf in
                        optionScoreGradScratch.withUnsafeMutableBufferPointer { osgBuf in
                            importanceScratch.withUnsafeMutableBufferPointer { impBuf in
                                optionLogProbScratch.withUnsafeMutableBufferPointer { optLogBuf in
                                    if let optionValues = policy.optionValues {
                                        optionValues.withUnsafeBufferPointer { optBuf in
                                            ActionTransducer.sampleAndScore(
                                                spec: actionSpec, dPolicy: dPolicy, envActionWidth: envActionWidth,
                                                policyMean: pmBuf.baseAddress!,
                                                optionPolicyMean: optBuf.baseAddress,
                                                optionCount: optionCount,
                                                envAction: aBuf.baseAddress!,
                                                actionSignal: actBuf.baseAddress!,
                                                scoreGrad: sgBuf.baseAddress!,
                                                optionScoreGrad: osgBuf.baseAddress,
                                                importance: impBuf.baseAddress,
                                                optionLogProbScratch: optLogBuf.baseAddress,
                                                rngState: &rngState)
                                        }
                                    } else {
                                        ActionTransducer.sampleAndScore(
                                            spec: actionSpec, dPolicy: dPolicy, envActionWidth: envActionWidth,
                                            policyMean: pmBuf.baseAddress!,
                                            optionPolicyMean: nil, optionCount: 0,
                                            envAction: aBuf.baseAddress!,
                                            actionSignal: actBuf.baseAddress!,
                                            scoreGrad: sgBuf.baseAddress!,
                                            optionScoreGrad: nil,
                                            importance: nil,
                                            optionLogProbScratch: nil,
                                            rngState: &rngState)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        switch actionSpec[0] {
        case .discrete:
            discreteIndexScratch[0] = Int32(envActionScratch[0])
            try checked { e, cap in
                bridge.stepDiscrete(handle, action: discreteIndexScratch,
                                    obs: &observationScratch,
                                    reward: &rewardScratch, done: &doneScratch,
                                    error: e, cap: cap)
            }
        case .box:
            try checked { e, cap in
                bridge.stepContinuous(handle, action: envActionScratch,
                                      obs: &observationScratch,
                                      reward: &rewardScratch, done: &doneScratch,
                                      error: e, cap: cap)
            }
        }
    }
}
