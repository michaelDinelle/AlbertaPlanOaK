import Darwin
import Foundation

struct AgentStepResult {
    let action: [Float]
    let telemetry: AgentTelemetry
}

/// Replaces non-finite scalars where a numeric default is required (e.g. timing strings). Dynamic telemetry uses ``DynamicTelemetryChannel`` instead.
func metricsFinite(_ value: Double, default defaultValue: Double = 0) -> Double {
    value.isFinite ? value : defaultValue
}

/// One dynamic-telemetry channel in metrics JSON: a finite number, or JSON `null` (non-finite source value, or no finite samples in an aggregate).
enum DynamicTelemetryChannel: Codable, Equatable, Sendable {
    case finite(Double)
    case invalid

    init(finiteOrNan raw: Double) {
        if raw.isFinite {
            self = .finite(raw)
        } else {
            self = .invalid
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .invalid
            return
        }
        let d = try container.decode(Double.self)
        self = d.isFinite ? .finite(d) : .invalid
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .finite(let d):
            try container.encode(d)
        case .invalid:
            try container.encodeNil()
        }
    }

    /// For ``PerChannelRunningStats`` only: invalid steps aggregate as `nan` internally (never written to JSON).
    var doubleForAggregation: Double {
        switch self {
        case .finite(let x):
            return x
        case .invalid:
            return .nan
        }
    }
}

struct DynamicTelemetryDescriptor: Codable, Equatable {
    let key: String
    let unit: String
}

struct DynamicTelemetrySchema: Codable, Equatable {
    let descriptors: [DynamicTelemetryDescriptor]

    static let empty = DynamicTelemetrySchema(descriptors: [])
}

extension DynamicTelemetrySchema {
    /// Single source of truth for `BaseSelfScalingCore` dynamic telemetry: descriptor order matches `dynamicTelemetryValues` in `step`.
    ///
    /// **Inventory (learning paths → channels):** budget/capacity (`budget.*`, `capacity.*`); perception encode (`perception.*` except utilities); predictive knowledge (`predictive_knowledge.*`); world model (`world_model.*`); reactive control TD/reward rate (`control.*`); search/planner (`search.*`); learning gate (`learning.step`); signal summaries (`signal.*`); timings (`timing.*`); structural snapshots (`snapshot.*`); perception utility roll-up (`perception.feature_utility_mean`); option bank utility roll-up (`options.bank_utility_mean`).
    static let oakSelfScalingCore = DynamicTelemetrySchema(
        descriptors: [
            DynamicTelemetryDescriptor(key: "budget.input_ns", unit: "ns"),
            DynamicTelemetryDescriptor(key: "budget.work_units", unit: "work_units"),
            DynamicTelemetryDescriptor(key: "capacity.allowed_rank", unit: "rank"),
            DynamicTelemetryDescriptor(key: "perception.active_rank", unit: "rank"),
            DynamicTelemetryDescriptor(key: "perception.residual_energy", unit: "energy"),
            DynamicTelemetryDescriptor(key: "perception.total_energy", unit: "energy"),
            DynamicTelemetryDescriptor(key: "predictive_knowledge.error_mean", unit: "loss"),
            DynamicTelemetryDescriptor(key: "world_model.error_mean", unit: "loss"),
            DynamicTelemetryDescriptor(key: "control.td_abs_mean", unit: "delta"),
            DynamicTelemetryDescriptor(key: "control.reward_rate", unit: "reward"),
            DynamicTelemetryDescriptor(key: "search.calls", unit: "count"),
            DynamicTelemetryDescriptor(key: "search.expansions", unit: "count"),
            DynamicTelemetryDescriptor(key: "search.best_score", unit: "score"),
            DynamicTelemetryDescriptor(key: "search.budget_ns", unit: "ns"),
            DynamicTelemetryDescriptor(key: "search.invocations", unit: "count"),
            DynamicTelemetryDescriptor(key: "learning.step", unit: "unitless"),
            DynamicTelemetryDescriptor(key: "signal.reward_input", unit: "reward"),
            DynamicTelemetryDescriptor(key: "signal.obs_mean", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.obs_std", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.obs_l2", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.action_mean", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.action_std", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.action_l2", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.action_min", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.action_max", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.state_mean", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.state_std", unit: "signal"),
            DynamicTelemetryDescriptor(key: "signal.state_l2", unit: "signal"),
            DynamicTelemetryDescriptor(key: "timing.agent_step_ms", unit: "ms"),
            DynamicTelemetryDescriptor(key: "timing.planner_step_ms", unit: "ms"),
            DynamicTelemetryDescriptor(key: "snapshot.active_rank", unit: "rank"),
            DynamicTelemetryDescriptor(key: "snapshot.active_gvfs", unit: "count"),
            DynamicTelemetryDescriptor(key: "snapshot.active_options", unit: "count"),
            DynamicTelemetryDescriptor(key: "perception.feature_utility_mean", unit: "utility"),
            DynamicTelemetryDescriptor(key: "options.bank_utility_mean", unit: "utility"),
        ]
    )
}


struct TelemetryVectorSummary: Equatable {
    let mean: Double
    let std: Double
    let l2: Double
    let min: Double
    let max: Double

    static let zero = TelemetryVectorSummary(mean: 0, std: 0, l2: 0, min: 0, max: 0)

    init(mean: Double, std: Double, l2: Double, min: Double, max: Double) {
        self.mean = metricsFinite(mean)
        self.std = metricsFinite(std)
        self.l2 = metricsFinite(l2)
        self.min = metricsFinite(min)
        self.max = metricsFinite(max)
    }
}

struct AgentTelemetry: Equatable {
    let rewardInput: Double
    let obsMean: Double
    let obsStd: Double
    let obsL2: Double
    let actionMean: Double
    let actionStd: Double
    let actionL2: Double
    let actionMin: Double
    let actionMax: Double
    let stateMean: Double
    let stateStd: Double
    let stateL2: Double
    let agentStepMs: Double
    let backend: String
    let plannerStepMs: Double
    let computeBudgetNs: Double
    let activeRank: Int
    let activeGVFs: Int
    let activeOptions: Int
    let snapshotToken: Int
    let cmd1Ms: Double?
    let cmd2Ms: Double?
    let cmd3Ms: Double?
    let cmd4Ms: Double?
    let cmd5Ms: Double?
    /// `nil` when the policy does not emit the oak schema vector. Otherwise one ``DynamicTelemetryChannel`` per schema row (invalid encodes as JSON `null`).
    let dynamicTelemetryValues: [DynamicTelemetryChannel]?

    init(
        rewardInput: Float,
        obs: UnsafeBufferPointer<Float>,
        state: UnsafeBufferPointer<Float>,
        action: UnsafeBufferPointer<Float>,
        agentStepMs: Double,
        backend: String,
        metalTimings: MetalCommandTimings? = nil,
        plannerStepMs: Double = 0,
        computeBudgetNs: UInt64 = 0,
        activeRank: Int = 0,
        activeGVFs: Int = 0,
        activeOptions: Int = 0,
        snapshotToken: Int = 0,
        dynamicTelemetryValues: [DynamicTelemetryChannel]? = nil
    ) {
        let obsStats = VectorSummary(obs)
        let actionStats = VectorSummary(action)
        let stateStats = VectorSummary(state)

        self.init(
            rewardInput: Double(rewardInput),
            obsSummary: obsStats.telemetrySummary,
            stateSummary: stateStats.telemetrySummary,
            actionSummary: actionStats.telemetrySummary,
            agentStepMs: agentStepMs,
            backend: backend,
            metalTimings: metalTimings,
            plannerStepMs: plannerStepMs,
            computeBudgetNs: computeBudgetNs,
            activeRank: activeRank,
            activeGVFs: activeGVFs,
            activeOptions: activeOptions,
            snapshotToken: snapshotToken,
            dynamicTelemetryValues: dynamicTelemetryValues
        )
    }

    init(
        rewardInput: Double,
        obsSummary: TelemetryVectorSummary,
        stateSummary: TelemetryVectorSummary,
        actionSummary: TelemetryVectorSummary,
        agentStepMs: Double,
        backend: String,
        metalTimings: MetalCommandTimings? = nil,
        plannerStepMs: Double = 0,
        computeBudgetNs: UInt64 = 0,
        activeRank: Int = 0,
        activeGVFs: Int = 0,
        activeOptions: Int = 0,
        snapshotToken: Int = 0,
        dynamicTelemetryValues: [DynamicTelemetryChannel]? = nil
    ) {
        self.rewardInput = metricsFinite(rewardInput)
        obsMean = obsSummary.mean
        obsStd = obsSummary.std
        obsL2 = obsSummary.l2
        actionMean = actionSummary.mean
        actionStd = actionSummary.std
        actionL2 = actionSummary.l2
        actionMin = actionSummary.min
        actionMax = actionSummary.max
        stateMean = stateSummary.mean
        stateStd = stateSummary.std
        stateL2 = stateSummary.l2
        self.agentStepMs = metricsFinite(agentStepMs)
        self.backend = backend
        self.plannerStepMs = metricsFinite(plannerStepMs)
        self.computeBudgetNs = metricsFinite(Double(computeBudgetNs))
        self.activeRank = activeRank
        self.activeGVFs = activeGVFs
        self.activeOptions = activeOptions
        self.snapshotToken = snapshotToken
        cmd1Ms = metalTimings?.cmd1Ms
        cmd2Ms = metalTimings?.cmd2Ms
        cmd3Ms = metalTimings?.cmd3Ms
        cmd4Ms = metalTimings?.cmd4Ms
        cmd5Ms = metalTimings?.cmd5Ms
        self.dynamicTelemetryValues = dynamicTelemetryValues
    }

    static func randomPolicy(
        rewardInput: Float,
        obs: UnsafeBufferPointer<Float>,
        action: UnsafeBufferPointer<Float>,
        agentStepMs: Double,
        dynamicTelemetryValues: [DynamicTelemetryChannel]? = nil
    ) -> AgentTelemetry {
        AgentTelemetry(
            rewardInput: rewardInput,
            obs: obs,
            state: UnsafeBufferPointer<Float>(start: nil, count: 0),
            action: action,
            agentStepMs: agentStepMs,
            backend: "random",
            plannerStepMs: 0,
            computeBudgetNs: 0,
            activeRank: 0,
            activeGVFs: 0,
            activeOptions: 0,
            snapshotToken: 0,
            dynamicTelemetryValues: dynamicTelemetryValues
        )
    }

    static func == (lhs: AgentTelemetry, rhs: AgentTelemetry) -> Bool {
        lhs.rewardInput == rhs.rewardInput
            && lhs.obsMean == rhs.obsMean
            && lhs.obsStd == rhs.obsStd
            && lhs.obsL2 == rhs.obsL2
            && lhs.actionMean == rhs.actionMean
            && lhs.actionStd == rhs.actionStd
            && lhs.actionL2 == rhs.actionL2
            && lhs.actionMin == rhs.actionMin
            && lhs.actionMax == rhs.actionMax
            && lhs.stateMean == rhs.stateMean
            && lhs.stateStd == rhs.stateStd
            && lhs.stateL2 == rhs.stateL2
            && lhs.agentStepMs == rhs.agentStepMs
            && lhs.backend == rhs.backend
            && lhs.plannerStepMs == rhs.plannerStepMs
            && lhs.computeBudgetNs == rhs.computeBudgetNs
            && lhs.activeRank == rhs.activeRank
            && lhs.activeGVFs == rhs.activeGVFs
            && lhs.activeOptions == rhs.activeOptions
            && lhs.snapshotToken == rhs.snapshotToken
            && lhs.cmd1Ms == rhs.cmd1Ms
            && lhs.cmd2Ms == rhs.cmd2Ms
            && lhs.cmd3Ms == rhs.cmd3Ms
            && lhs.cmd4Ms == rhs.cmd4Ms
            && lhs.cmd5Ms == rhs.cmd5Ms
            && lhs.dynamicTelemetryValues == rhs.dynamicTelemetryValues
    }
}

struct MetalCommandTimings: Equatable {
    let cmd1Ms: Double
    let cmd2Ms: Double
    let cmd3Ms: Double
    let cmd4Ms: Double
    let cmd5Ms: Double
}

/// Per-channel running mean, temporal population std, min/max over **finite** samples, plus non-finite observation counts.
struct PerChannelRunningStats: Equatable {
    private var finiteSum: [Double] = []
    private var finiteSumSq: [Double] = []
    private var finiteCount: [Int] = []
    private var nonFiniteCount: [Int] = []
    private var minVal: [Double] = []
    private var maxVal: [Double] = []

    mutating func record(values: [Double]) {
        if values.isEmpty { return }
        if finiteSum.count < values.count {
            let old = finiteSum.count
            for _ in old..<values.count {
                finiteSum.append(0)
                finiteSumSq.append(0)
                finiteCount.append(0)
                nonFiniteCount.append(0)
                minVal.append(.greatestFiniteMagnitude)
                maxVal.append(-.greatestFiniteMagnitude)
            }
        }
        for i in 0..<values.count {
            let v = values[i]
            if v.isFinite {
                finiteSum[i] += v
                finiteSumSq[i] += v * v
                finiteCount[i] += 1
                minVal[i] = Swift.min(minVal[i], v)
                maxVal[i] = Swift.max(maxVal[i], v)
            } else {
                nonFiniteCount[i] += 1
            }
        }
    }

    /// Per-channel mean over **finite** samples only. ``DynamicTelemetryChannel/invalid`` means no finite samples (not numeric zero).
    var means: [DynamicTelemetryChannel]? {
        guard !finiteSum.isEmpty else { return nil }
        return (0..<finiteSum.count).map { i in
            if finiteCount[i] > 0 {
                return .finite(finiteSum[i] / Double(finiteCount[i]))
            }
            return .invalid
        }
    }

    var temporalStd: [DynamicTelemetryChannel]? {
        guard !finiteSum.isEmpty else { return nil }
        return (0..<finiteSum.count).map { i in
            let c = finiteCount[i]
            guard c > 0 else { return .invalid }
            let m = finiteSum[i] / Double(c)
            let variance = max(finiteSumSq[i] / Double(c) - m * m, 0)
            return .finite(sqrt(variance))
        }
    }

    var mins: [DynamicTelemetryChannel]? {
        guard !minVal.isEmpty else { return nil }
        return (0..<minVal.count).map { i in
            finiteCount[i] > 0 ? .finite(minVal[i]) : .invalid
        }
    }

    var maxs: [DynamicTelemetryChannel]? {
        guard !maxVal.isEmpty else { return nil }
        return (0..<maxVal.count).map { i in
            finiteCount[i] > 0 ? .finite(maxVal[i]) : .invalid
        }
    }

    var nonFiniteCounts: [Int]? {
        guard !nonFiniteCount.isEmpty else { return nil }
        return nonFiniteCount
    }

    mutating func reset() {
        self = PerChannelRunningStats()
    }
}

struct RollingWindowStats: Equatable {
    private(set) var count = 0
    private(set) var startStep = 0
    private(set) var endStep = 0
    private(set) var globalEnvStepCount = 0
    private var rewardSum = 0.0
    private var rewardSquaredSum = 0.0
    private(set) var doneCount = 0
    private(set) var resetCount = 0
    private var agentStepMsSum = 0.0
    private var envStepMsSum = 0.0
    private var totalStepMsSum = 0.0
    private var actionL2Sum = 0.0
    private var stateL2Sum = 0.0
    private var plannerStepMsSum = 0.0
    private var plannerActiveStepCount = 0
    private var activeRankSum = 0.0
    private var activeRankObservedMin = Int.max
    private var activeRankObservedMax = 0
    private var activeGVFsSum = 0.0
    private var activeOptionsSum = 0.0
    private var dynamicTelemetry = PerChannelRunningStats()

    mutating func record(step: Int, rewards: [Float], doneCountBatch: Int, telemetry: AgentTelemetry, envStepMs: Double, totalStepMs: Double) {
        if count == 0 {
            startStep = step
        }

        count += 1
        endStep = step
        globalEnvStepCount += rewards.count
        for reward in rewards {
            let safeReward = metricsFinite(Double(reward))
            rewardSum += safeReward
            rewardSquaredSum += safeReward * safeReward
        }
        doneCount += doneCountBatch
        resetCount += doneCountBatch
        agentStepMsSum += metricsFinite(telemetry.agentStepMs)
        envStepMsSum += metricsFinite(envStepMs)
        totalStepMsSum += metricsFinite(totalStepMs)
        actionL2Sum += metricsFinite(telemetry.actionL2)
        stateL2Sum += metricsFinite(telemetry.stateL2)
        plannerStepMsSum += metricsFinite(telemetry.plannerStepMs)
        plannerActiveStepCount += telemetry.plannerStepMs > 0 ? 1 : 0
        activeRankSum += metricsFinite(Double(telemetry.activeRank))
        activeRankObservedMin = min(activeRankObservedMin, telemetry.activeRank)
        activeRankObservedMax = max(activeRankObservedMax, telemetry.activeRank)
        activeGVFsSum += metricsFinite(Double(telemetry.activeGVFs))
        activeOptionsSum += metricsFinite(Double(telemetry.activeOptions))
        if let dyn = telemetry.dynamicTelemetryValues {
            dynamicTelemetry.record(values: dyn.map(\.doubleForAggregation))
        }
    }

    var rewardMean: Double {
        globalEnvStepCount > 0 ? rewardSum / Double(globalEnvStepCount) : 0
    }

    var rewardStd: Double {
        guard globalEnvStepCount > 0 else {
            return 0
        }

        let mean = rewardMean
        let variance = max((rewardSquaredSum / Double(globalEnvStepCount)) - (mean * mean), 0)
        return sqrt(variance)
    }

    var stepsPerSecond: Double {
        guard totalStepMsSum > 0 else {
            return 0
        }
        return Double(globalEnvStepCount) / (totalStepMsSum / 1000)
    }

    var agentStepMsMean: Double {
        count > 0 ? agentStepMsSum / Double(count) : 0
    }

    var envStepMsMean: Double {
        count > 0 ? envStepMsSum / Double(count) : 0
    }

    var totalStepMsMean: Double {
        count > 0 ? totalStepMsSum / Double(count) : 0
    }

    var actionL2Mean: Double {
        count > 0 ? actionL2Sum / Double(count) : 0
    }

    var stateL2Mean: Double {
        count > 0 ? stateL2Sum / Double(count) : 0
    }

    var plannerStepMsMean: Double {
        count > 0 ? plannerStepMsSum / Double(count) : 0
    }

    var plannerActiveFraction: Double {
        count > 0 ? Double(plannerActiveStepCount) / Double(count) : 0
    }

    var activeRankMean: Double {
        count > 0 ? activeRankSum / Double(count) : 0
    }

    var activeRankMin: Int {
        count > 0 ? activeRankObservedMin : 0
    }

    var activeRankMax: Int {
        count > 0 ? activeRankObservedMax : 0
    }

    var activeGVFsMean: Double {
        count > 0 ? activeGVFsSum / Double(count) : 0
    }

    var activeOptionsMean: Double {
        count > 0 ? activeOptionsSum / Double(count) : 0
    }

    var dynamicTelemetryMeanValues: [DynamicTelemetryChannel]? {
        guard count > 0 else { return nil }
        return dynamicTelemetry.means
    }

    /// Population standard deviation of each dynamic telemetry channel over vectorized update steps in this window (finite samples).
    var dynamicTelemetryStdValues: [DynamicTelemetryChannel]? {
        guard count > 0 else { return nil }
        return dynamicTelemetry.temporalStd
    }

    var dynamicTelemetryMinValues: [DynamicTelemetryChannel]? {
        guard count > 0 else { return nil }
        return dynamicTelemetry.mins
    }

    var dynamicTelemetryMaxValues: [DynamicTelemetryChannel]? {
        guard count > 0 else { return nil }
        return dynamicTelemetry.maxs
    }

    /// Per-channel count of non-finite step observations in this window (parallel to schema).
    var dynamicTelemetryNonFiniteCounts: [Int]? {
        guard count > 0 else { return nil }
        return dynamicTelemetry.nonFiniteCounts
    }

    var rewardTotal: Double {
        rewardSum
    }

    mutating func reset() {
        self = RollingWindowStats()
    }
}

struct MetricsConfiguration {
    let windowSize: Int
    /// When `nil`, metrics are printed to the terminal only (no JSONL file).
    let fileURL: URL?

    init(windowSize: Int, fileURL: URL?) {
        self.windowSize = windowSize
        self.fileURL = fileURL
    }

    static func resolve(
        envName: String,
        policyName: String,
        runControl: RunControl,
        numEnvs: Int,
        rootDirectory: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        processId: Int32 = getpid()
    ) -> MetricsConfiguration {
        let fileManager = FileManager.default
        let baseDirectory = rootDirectory ?? environment["OAK_ROOT_DIR"] ?? fileManager.currentDirectoryPath
        let windowSize = deriveMetricsWindowSize(runControl: runControl, numEnvs: numEnvs)

        if let overridePath = environment["OAK_METRICS_JSONL_PATH"], !overridePath.isEmpty {
            return MetricsConfiguration(
                windowSize: windowSize,
                fileURL: resolveMetricsPath(overridePath, rootDirectory: baseDirectory)
            )
        }

        if let metricsDirectory = environment["OAK_METRICS_DIR"], !metricsDirectory.isEmpty {
            let directoryURL = resolveMetricsPath(metricsDirectory, rootDirectory: baseDirectory)
            let filename = "\(metricsFilenameTimestamp(Date()))-\(envName)-\(policyName)-\(processId).jsonl"
            return MetricsConfiguration(
                windowSize: windowSize,
                fileURL: directoryURL.appendingPathComponent(filename)
            )
        }

        return MetricsConfiguration(windowSize: windowSize, fileURL: nil)
    }
}

struct RunControl {
    let warmupSteps: Int
    let stopAfterMeasuredSteps: Int?
    let stopAfterMeasuredSeconds: Double?

    static let unbounded = RunControl(warmupSteps: 0, stopAfterMeasuredSteps: nil, stopAfterMeasuredSeconds: nil)

    static func resolve(environment: [String: String] = ProcessInfo.processInfo.environment) -> RunControl {
        RunControl(
            warmupSteps: max(environment["OAK_WARMUP_STEPS"].flatMap(Int.init) ?? 0, 0),
            stopAfterMeasuredSteps: positiveInt(from: environment["OAK_STOP_AFTER_STEPS"]),
            stopAfterMeasuredSeconds: positiveDouble(from: environment["OAK_STOP_AFTER_SECONDS"])
        )
    }
}

final class JSONLMetricsSink {
    let fileURL: URL

    private let handle: FileHandle
    private let encoder: JSONEncoder

    init(fileURL: URL) throws {
        let fileManager = FileManager.default
        let directoryURL = fileURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        if !fileManager.fileExists(atPath: fileURL.path) {
            fileManager.createFile(atPath: fileURL.path, contents: nil)
        }

        self.fileURL = fileURL
        handle = try FileHandle(forWritingTo: fileURL)
        try handle.seekToEnd()

        encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
    }

    deinit {
        try? handle.close()
    }

    func write(_ record: MetricsRecord) throws {
        let data = try encoder.encode(record)
        try handle.write(contentsOf: data)
        try handle.write(contentsOf: Data([0x0A]))
    }
}

struct RunMetadata: Equatable {
    let logicalEnvName: String
    let concreteEnvId: String
    let leagueName: String?
    let schema: EnvironmentSchemaFingerprint
    let dynamicTelemetrySchema: DynamicTelemetrySchema
}

final class RunMetricsAggregator {
    let runId: String
    let envName: String
    let metadata: RunMetadata
    let policyName: String
    let backend: String
    let windowSize: Int
    let seed: Int?
    let numEnvs: Int

    private(set) var stepCount = 0
    private(set) var doneCount = 0
    private(set) var resetCount = 0
    private var totalStats = RollingWindowStats()
    private var windowStats = RollingWindowStats()

    init(
        runId: String = UUID().uuidString,
        envName: String,
        logicalEnvName: String,
        concreteEnvId: String,
        leagueName: String? = nil,
        policyName: String,
        backend: String,
        schema: EnvironmentSchemaFingerprint,
        dynamicTelemetrySchema: DynamicTelemetrySchema = .empty,
        windowSize: Int,
        seed: Int? = nil,
        numEnvs: Int = 1
    ) {
        self.runId = runId
        self.envName = envName
        metadata = RunMetadata(
            logicalEnvName: logicalEnvName,
            concreteEnvId: concreteEnvId,
            leagueName: leagueName,
            schema: schema,
            dynamicTelemetrySchema: dynamicTelemetrySchema
        )
        self.policyName = policyName
        self.backend = backend
        self.windowSize = max(windowSize, 1)
        self.seed = seed
        self.numEnvs = max(numEnvs, 1)
    }

    func runStartRecord(control: RunControl = .unbounded, at date: Date = Date()) -> MetricsRecord {
        MetricsRecord(
            type: "run_start",
            timestamp: metricsRecordTimestamp(date),
            runId: runId,
            env: envName,
            logicalEnv: metadata.logicalEnvName,
            concreteEnvId: metadata.concreteEnvId,
            league: metadata.leagueName,
            policy: policyName,
            backend: backend,
            step: 0,
            seed: seed,
            numEnvs: numEnvs,
            warmupSteps: control.warmupSteps,
            stopAfterMeasuredSteps: control.stopAfterMeasuredSteps,
            stopAfterMeasuredSeconds: control.stopAfterMeasuredSeconds,
            observationSpaceKind: metadata.schema.observation.kind,
            observationFlatDim: metadata.schema.observation.flattenedDim,
            actionSpaceKind: metadata.schema.action.kind,
            actionFlatDim: metadata.schema.action.flattenedDim,
            actionDiscreteCount: metadata.schema.action.discreteCount,
            actionShape: metadata.schema.action.shape,
            schemaFingerprint: metadata.schema.schemaFingerprint,
            dynamicTelemetrySchema: metadata.dynamicTelemetrySchema
        )
    }

    func recordStep(rewards: [Float], dones: [Bool], telemetry: AgentTelemetry, envStepMs: Double, totalStepMs: Double, at date: Date = Date()) -> (step: MetricsRecord, summary: MetricsRecord?) {
        stepCount += 1
        let rewardValues = rewards.map { metricsFinite(Double($0)) }
        let rewardSumBatch = rewardValues.reduce(0, +)
        let rewardMeanBatch = rewardValues.isEmpty ? 0 : rewardSumBatch / Double(rewardValues.count)
        let doneCountBatch = dones.reduce(0) { $0 + ($1 ? 1 : 0) }
        let doneAnyBatch = doneCountBatch > 0
        doneCount += doneCountBatch
        resetCount += doneCountBatch

        totalStats.record(
            step: stepCount,
            rewards: rewards,
            doneCountBatch: doneCountBatch,
            telemetry: telemetry,
            envStepMs: envStepMs,
            totalStepMs: totalStepMs
        )
        windowStats.record(
            step: stepCount,
            rewards: rewards,
            doneCountBatch: doneCountBatch,
            telemetry: telemetry,
            envStepMs: envStepMs,
            totalStepMs: totalStepMs
        )

        let stepRecord = MetricsRecord.stepRecord(
            timestamp: metricsRecordTimestamp(date),
            runId: runId,
            env: envName,
            policy: policyName,
            backend: backend,
            step: stepCount,
            seed: seed,
            reward: rewardMeanBatch,
            done: doneAnyBatch,
            resetCount: resetCount,
            rewardSumBatch: rewardSumBatch,
            rewardMeanBatch: rewardMeanBatch,
            doneCountBatch: doneCountBatch,
            doneAnyBatch: doneAnyBatch,
            telemetry: telemetry,
            envStepMs: envStepMs,
            totalStepMs: totalStepMs
        )

        var summaryRecord: MetricsRecord?
        if windowStats.count >= windowSize {
            summaryRecord = MetricsRecord.summaryRecord(
                type: "summary",
                timestamp: metricsRecordTimestamp(date),
                runId: runId,
                env: envName,
                policy: policyName,
                backend: backend,
                step: stepCount,
                seed: seed,
                logicalEnv: metadata.logicalEnvName,
                concreteEnvId: metadata.concreteEnvId,
                league: metadata.leagueName,
                observationSpaceKind: metadata.schema.observation.kind,
                observationFlatDim: metadata.schema.observation.flattenedDim,
                actionSpaceKind: metadata.schema.action.kind,
                actionFlatDim: metadata.schema.action.flattenedDim,
                actionDiscreteCount: metadata.schema.action.discreteCount,
                actionShape: metadata.schema.action.shape,
                schemaFingerprint: metadata.schema.schemaFingerprint,
                dynamicTelemetryValues: windowStats.dynamicTelemetryMeanValues,
                dynamicTelemetryStd: windowStats.dynamicTelemetryStdValues,
                dynamicTelemetryMin: windowStats.dynamicTelemetryMinValues,
                dynamicTelemetryMax: windowStats.dynamicTelemetryMaxValues,
                stats: windowStats
            )
            windowStats.reset()
        }

        return (stepRecord, summaryRecord)
    }

    func runEndRecord(control: RunControl = .unbounded, stopReason: String = "completed", measuredSeconds: Double = 0, globalMeasuredEnvSteps: Int, at date: Date = Date()) -> MetricsRecord {
        MetricsRecord.summaryRecord(
            type: "run_end",
            timestamp: metricsRecordTimestamp(date),
            runId: runId,
            env: envName,
            policy: policyName,
            backend: backend,
            step: stepCount,
            seed: seed,
            logicalEnv: metadata.logicalEnvName,
            concreteEnvId: metadata.concreteEnvId,
            league: metadata.leagueName,
            warmupSteps: control.warmupSteps,
            measuredStepCount: globalMeasuredEnvSteps,
            vectorUpdateCount: stepCount,
            measuredSeconds: measuredSeconds,
            stopReason: stopReason,
            stopAfterMeasuredSteps: control.stopAfterMeasuredSteps,
            stopAfterMeasuredSeconds: control.stopAfterMeasuredSeconds,
            observationSpaceKind: metadata.schema.observation.kind,
            observationFlatDim: metadata.schema.observation.flattenedDim,
            actionSpaceKind: metadata.schema.action.kind,
            actionFlatDim: metadata.schema.action.flattenedDim,
            actionDiscreteCount: metadata.schema.action.discreteCount,
            actionShape: metadata.schema.action.shape,
            schemaFingerprint: metadata.schema.schemaFingerprint,
            dynamicTelemetryValues: totalStats.dynamicTelemetryMeanValues,
            dynamicTelemetryStd: totalStats.dynamicTelemetryStdValues,
            dynamicTelemetryMin: totalStats.dynamicTelemetryMinValues,
            dynamicTelemetryMax: totalStats.dynamicTelemetryMaxValues,
            stats: totalStats
        )
    }
}

struct MetricsRecord: Codable {
    let type: String
    let timestamp: String
    let runId: String
    let env: String
    let logicalEnv: String?
    let concreteEnvId: String?
    let league: String?
    let policy: String
    let backend: String
    let step: Int
    let seed: Int?
    /// Vectorized env batch width for this run (present on `run_start` for benchmark provenance).
    let numEnvs: Int?
    let warmupSteps: Int?
    let measuredStepCount: Int?
    let measuredSeconds: Double?
    let stopReason: String?
    let stopAfterMeasuredSteps: Int?
    let stopAfterMeasuredSeconds: Double?
    let reward: Double?
    let done: Bool?
    let resetCount: Int?
    let rewardSumBatch: Double?
    let rewardMeanBatch: Double?
    let doneCountBatch: Int?
    let doneAnyBatch: Bool?
    let rewardInput: Double?
    let observationSpaceKind: String?
    let observationFlatDim: Int?
    let obsMean: Double?
    let obsStd: Double?
    let obsL2: Double?
    let actionSpaceKind: String?
    let actionFlatDim: Int?
    let actionDiscreteCount: Int?
    let actionShape: [Int]?
    let actionMean: Double?
    let actionStd: Double?
    let actionL2: Double?
    let actionMin: Double?
    let actionMax: Double?
    let stateMean: Double?
    let stateStd: Double?
    let stateL2: Double?
    let agentStepMs: Double?
    let cmd1Ms: Double?
    let cmd2Ms: Double?
    let cmd3Ms: Double?
    let cmd4Ms: Double?
    let cmd5Ms: Double?
    let envStepMs: Double?
    let totalStepMs: Double?
    let windowStartStep: Int?
    let windowEndStep: Int?
    let rewardSum: Double?
    let rewardMean: Double?
    let rewardStd: Double?
    let doneCount: Int?
    let stepsPerSecond: Double?
    let agentStepMsMean: Double?
    let envStepMsMean: Double?
    let totalStepMsMean: Double?
    let actionL2Mean: Double?
    let stateL2Mean: Double?
    let plannerStepMs: Double?
    let computeBudgetNs: Double?
    let activeRank: Int?
    let activeGVFs: Int?
    let activeOptions: Int?
    let snapshotToken: Int?
    let plannerStepMsMean: Double?
    let plannerActiveFraction: Double?
    let activeRankMean: Double?
    let activeRankMin: Int?
    let activeRankMax: Int?
    let activeGVFsMean: Double?
    let activeOptionsMean: Double?
    let schemaFingerprint: String?
    let dynamicTelemetrySchema: DynamicTelemetrySchema?
    /// Per-channel aggregates (``DynamicTelemetryChannel/invalid`` is JSON `null`, not zero).
    let dynamicTelemetryValues: [DynamicTelemetryChannel]?
    /// Temporal population std of each `dynamicTelemetrySchema` channel over steps in the `summary` / `run_end` window.
    let dynamicTelemetryStd: [DynamicTelemetryChannel]?
    /// Temporal min per channel over steps in the window.
    let dynamicTelemetryMin: [DynamicTelemetryChannel]?
    /// Temporal max per channel over steps in the window.
    let dynamicTelemetryMax: [DynamicTelemetryChannel]?
    /// Per-channel count of non-finite step observations in the `summary` / `run_end` window (parallel to schema).
    let dynamicTelemetryNonFiniteCounts: [Int]?
    /// For `run_end`: number of vectorized agent+env update iterations (batch steps); `measuredStepCount` is global env steps.
    let vectorUpdateCount: Int?

    init(
        type: String,
        timestamp: String,
        runId: String,
        env: String,
        logicalEnv: String? = nil,
        concreteEnvId: String? = nil,
        league: String? = nil,
        policy: String,
        backend: String,
        step: Int,
        seed: Int? = nil,
        numEnvs: Int? = nil,
        warmupSteps: Int? = nil,
        measuredStepCount: Int? = nil,
        measuredSeconds: Double? = nil,
        stopReason: String? = nil,
        stopAfterMeasuredSteps: Int? = nil,
        stopAfterMeasuredSeconds: Double? = nil,
        reward: Double? = nil,
        done: Bool? = nil,
        resetCount: Int? = nil,
        rewardSumBatch: Double? = nil,
        rewardMeanBatch: Double? = nil,
        doneCountBatch: Int? = nil,
        doneAnyBatch: Bool? = nil,
        rewardInput: Double? = nil,
        observationSpaceKind: String? = nil,
        observationFlatDim: Int? = nil,
        obsMean: Double? = nil,
        obsStd: Double? = nil,
        obsL2: Double? = nil,
        actionSpaceKind: String? = nil,
        actionFlatDim: Int? = nil,
        actionDiscreteCount: Int? = nil,
        actionShape: [Int]? = nil,
        actionMean: Double? = nil,
        actionStd: Double? = nil,
        actionL2: Double? = nil,
        actionMin: Double? = nil,
        actionMax: Double? = nil,
        stateMean: Double? = nil,
        stateStd: Double? = nil,
        stateL2: Double? = nil,
        agentStepMs: Double? = nil,
        cmd1Ms: Double? = nil,
        cmd2Ms: Double? = nil,
        cmd3Ms: Double? = nil,
        cmd4Ms: Double? = nil,
        cmd5Ms: Double? = nil,
        envStepMs: Double? = nil,
        totalStepMs: Double? = nil,
        windowStartStep: Int? = nil,
        windowEndStep: Int? = nil,
        rewardSum: Double? = nil,
        rewardMean: Double? = nil,
        rewardStd: Double? = nil,
        doneCount: Int? = nil,
        stepsPerSecond: Double? = nil,
        agentStepMsMean: Double? = nil,
        envStepMsMean: Double? = nil,
        totalStepMsMean: Double? = nil,
        actionL2Mean: Double? = nil,
        stateL2Mean: Double? = nil,
        plannerStepMs: Double? = nil,
        computeBudgetNs: Double? = nil,
        activeRank: Int? = nil,
        activeGVFs: Int? = nil,
        activeOptions: Int? = nil,
        snapshotToken: Int? = nil,
        plannerStepMsMean: Double? = nil,
        plannerActiveFraction: Double? = nil,
        activeRankMean: Double? = nil,
        activeRankMin: Int? = nil,
        activeRankMax: Int? = nil,
        activeGVFsMean: Double? = nil,
        activeOptionsMean: Double? = nil,
        schemaFingerprint: String? = nil,
        dynamicTelemetrySchema: DynamicTelemetrySchema? = nil,
        dynamicTelemetryValues: [DynamicTelemetryChannel]? = nil,
        dynamicTelemetryStd: [DynamicTelemetryChannel]? = nil,
        dynamicTelemetryMin: [DynamicTelemetryChannel]? = nil,
        dynamicTelemetryMax: [DynamicTelemetryChannel]? = nil,
        dynamicTelemetryNonFiniteCounts: [Int]? = nil,
        vectorUpdateCount: Int? = nil
    ) {
        self.type = type
        self.timestamp = timestamp
        self.runId = runId
        self.env = env
        self.logicalEnv = logicalEnv
        self.concreteEnvId = concreteEnvId
        self.league = league
        self.policy = policy
        self.backend = backend
        self.step = step
        self.seed = seed
        self.numEnvs = numEnvs
        self.warmupSteps = warmupSteps
        self.measuredStepCount = measuredStepCount
        self.measuredSeconds = measuredSeconds
        self.stopReason = stopReason
        self.stopAfterMeasuredSteps = stopAfterMeasuredSteps
        self.stopAfterMeasuredSeconds = stopAfterMeasuredSeconds
        self.reward = reward
        self.done = done
        self.resetCount = resetCount
        self.rewardSumBatch = rewardSumBatch
        self.rewardMeanBatch = rewardMeanBatch
        self.doneCountBatch = doneCountBatch
        self.doneAnyBatch = doneAnyBatch
        self.rewardInput = rewardInput
        self.observationSpaceKind = observationSpaceKind
        self.observationFlatDim = observationFlatDim
        self.obsMean = obsMean
        self.obsStd = obsStd
        self.obsL2 = obsL2
        self.actionSpaceKind = actionSpaceKind
        self.actionFlatDim = actionFlatDim
        self.actionDiscreteCount = actionDiscreteCount
        self.actionShape = actionShape
        self.actionMean = actionMean
        self.actionStd = actionStd
        self.actionL2 = actionL2
        self.actionMin = actionMin
        self.actionMax = actionMax
        self.stateMean = stateMean
        self.stateStd = stateStd
        self.stateL2 = stateL2
        self.agentStepMs = agentStepMs
        self.cmd1Ms = cmd1Ms
        self.cmd2Ms = cmd2Ms
        self.cmd3Ms = cmd3Ms
        self.cmd4Ms = cmd4Ms
        self.cmd5Ms = cmd5Ms
        self.envStepMs = envStepMs
        self.totalStepMs = totalStepMs
        self.windowStartStep = windowStartStep
        self.windowEndStep = windowEndStep
        self.rewardSum = rewardSum
        self.rewardMean = rewardMean
        self.rewardStd = rewardStd
        self.doneCount = doneCount
        self.stepsPerSecond = stepsPerSecond
        self.agentStepMsMean = agentStepMsMean
        self.envStepMsMean = envStepMsMean
        self.totalStepMsMean = totalStepMsMean
        self.actionL2Mean = actionL2Mean
        self.stateL2Mean = stateL2Mean
        self.plannerStepMs = plannerStepMs
        self.computeBudgetNs = computeBudgetNs
        self.activeRank = activeRank
        self.activeGVFs = activeGVFs
        self.activeOptions = activeOptions
        self.snapshotToken = snapshotToken
        self.plannerStepMsMean = plannerStepMsMean
        self.plannerActiveFraction = plannerActiveFraction
        self.activeRankMean = activeRankMean
        self.activeRankMin = activeRankMin
        self.activeRankMax = activeRankMax
        self.activeGVFsMean = activeGVFsMean
        self.activeOptionsMean = activeOptionsMean
        self.schemaFingerprint = schemaFingerprint
        self.dynamicTelemetrySchema = dynamicTelemetrySchema
        self.dynamicTelemetryValues = dynamicTelemetryValues
        self.dynamicTelemetryStd = dynamicTelemetryStd
        self.dynamicTelemetryMin = dynamicTelemetryMin
        self.dynamicTelemetryMax = dynamicTelemetryMax
        self.dynamicTelemetryNonFiniteCounts = dynamicTelemetryNonFiniteCounts
        self.vectorUpdateCount = vectorUpdateCount
    }

    static func stepRecord(
        timestamp: String,
        runId: String,
        env: String,
        policy: String,
        backend: String,
        step: Int,
        seed: Int?,
        reward: Double,
        done: Bool,
        resetCount: Int,
        rewardSumBatch: Double,
        rewardMeanBatch: Double,
        doneCountBatch: Int,
        doneAnyBatch: Bool,
        telemetry: AgentTelemetry,
        envStepMs: Double,
        totalStepMs: Double
    ) -> MetricsRecord {
        MetricsRecord(
            type: "step",
            timestamp: timestamp,
            runId: runId,
            env: env,
            policy: policy,
            backend: backend,
            step: step,
            seed: seed,
            reward: reward,
            done: done,
            resetCount: resetCount,
            rewardSumBatch: rewardSumBatch,
            rewardMeanBatch: rewardMeanBatch,
            doneCountBatch: doneCountBatch,
            doneAnyBatch: doneAnyBatch,
            rewardInput: telemetry.rewardInput,
            obsMean: telemetry.obsMean,
            obsStd: telemetry.obsStd,
            obsL2: telemetry.obsL2,
            actionMean: telemetry.actionMean,
            actionStd: telemetry.actionStd,
            actionL2: telemetry.actionL2,
            actionMin: telemetry.actionMin,
            actionMax: telemetry.actionMax,
            stateMean: telemetry.stateMean,
            stateStd: telemetry.stateStd,
            stateL2: telemetry.stateL2,
            agentStepMs: telemetry.agentStepMs,
            cmd1Ms: telemetry.cmd1Ms,
            cmd2Ms: telemetry.cmd2Ms,
            cmd3Ms: telemetry.cmd3Ms,
            cmd4Ms: telemetry.cmd4Ms,
            cmd5Ms: telemetry.cmd5Ms,
            envStepMs: envStepMs,
            totalStepMs: totalStepMs,
            plannerStepMs: telemetry.plannerStepMs,
            computeBudgetNs: telemetry.computeBudgetNs,
            activeRank: telemetry.activeRank,
            activeGVFs: telemetry.activeGVFs,
            activeOptions: telemetry.activeOptions,
            snapshotToken: telemetry.snapshotToken,
            dynamicTelemetryValues: telemetry.dynamicTelemetryValues
        )
    }

    static func summaryRecord(
        type: String,
        timestamp: String,
        runId: String,
        env: String,
        policy: String,
        backend: String,
        step: Int,
        seed: Int? = nil,
        logicalEnv: String? = nil,
        concreteEnvId: String? = nil,
        league: String? = nil,
        warmupSteps: Int? = nil,
        measuredStepCount: Int? = nil,
        vectorUpdateCount: Int? = nil,
        measuredSeconds: Double? = nil,
        stopReason: String? = nil,
        stopAfterMeasuredSteps: Int? = nil,
        stopAfterMeasuredSeconds: Double? = nil,
        observationSpaceKind: String? = nil,
        observationFlatDim: Int? = nil,
        actionSpaceKind: String? = nil,
        actionFlatDim: Int? = nil,
        actionDiscreteCount: Int? = nil,
        actionShape: [Int]? = nil,
        schemaFingerprint: String? = nil,
        dynamicTelemetryValues: [DynamicTelemetryChannel]? = nil,
        dynamicTelemetryStd: [DynamicTelemetryChannel]? = nil,
        dynamicTelemetryMin: [DynamicTelemetryChannel]? = nil,
        dynamicTelemetryMax: [DynamicTelemetryChannel]? = nil,
        stats: RollingWindowStats
    ) -> MetricsRecord {
        MetricsRecord(
            type: type,
            timestamp: timestamp,
            runId: runId,
            env: env,
            logicalEnv: logicalEnv,
            concreteEnvId: concreteEnvId,
            league: league,
            policy: policy,
            backend: backend,
            step: step,
            seed: seed,
            warmupSteps: warmupSteps,
            measuredStepCount: measuredStepCount,
            measuredSeconds: measuredSeconds,
            stopReason: stopReason,
            stopAfterMeasuredSteps: stopAfterMeasuredSteps,
            stopAfterMeasuredSeconds: stopAfterMeasuredSeconds,
            resetCount: stats.resetCount,
            observationSpaceKind: observationSpaceKind,
            observationFlatDim: observationFlatDim,
            actionSpaceKind: actionSpaceKind,
            actionFlatDim: actionFlatDim,
            actionDiscreteCount: actionDiscreteCount,
            actionShape: actionShape,
            windowStartStep: stats.startStep,
            windowEndStep: stats.endStep,
            rewardSum: stats.rewardTotal,
            rewardMean: stats.rewardMean,
            rewardStd: stats.rewardStd,
            doneCount: stats.doneCount,
            stepsPerSecond: stats.stepsPerSecond,
            agentStepMsMean: stats.agentStepMsMean,
            envStepMsMean: stats.envStepMsMean,
            totalStepMsMean: stats.totalStepMsMean,
            actionL2Mean: stats.actionL2Mean,
            stateL2Mean: stats.stateL2Mean,
            plannerStepMsMean: stats.plannerStepMsMean,
            plannerActiveFraction: stats.plannerActiveFraction,
            activeRankMean: stats.activeRankMean,
            activeRankMin: stats.activeRankMin,
            activeRankMax: stats.activeRankMax,
            activeGVFsMean: stats.activeGVFsMean,
            activeOptionsMean: stats.activeOptionsMean,
            schemaFingerprint: schemaFingerprint,
            dynamicTelemetryValues: dynamicTelemetryValues,
            dynamicTelemetryStd: dynamicTelemetryStd,
            dynamicTelemetryMin: dynamicTelemetryMin,
            dynamicTelemetryMax: dynamicTelemetryMax,
            dynamicTelemetryNonFiniteCounts: stats.dynamicTelemetryNonFiniteCounts,
            vectorUpdateCount: vectorUpdateCount
        )
    }
}

func formatMetricsSummary(_ record: MetricsRecord) -> String {
    let label = record.type == "run_end" ? "final" : "window"
    let startStep = record.windowStartStep ?? record.step
    let endStep = record.windowEndStep ?? record.step
    let seedLabel = record.seed.map { " seed=\($0)" } ?? ""
    let globalEnvSteps = record.measuredStepCount ?? endStep
    let vectorUpdates = record.vectorUpdateCount ?? record.step
    let measuredSecondsLabel = record.measuredSeconds.map { String(format: " measured_s=%.2f", $0) } ?? ""
    let stopReasonLabel = record.stopReason.map { " stop=\($0)" } ?? ""
    let rewardSum = record.rewardSum ?? 0
    let rewardMean = record.rewardMean ?? 0
    let rewardStd = record.rewardStd ?? 0
    let doneCount = record.doneCount ?? 0
    let resetCount = record.resetCount ?? 0
    let stepsPerSecond = record.stepsPerSecond ?? 0
    let agentStepMsMean = record.agentStepMsMean ?? 0
    let envStepMsMean = record.envStepMsMean ?? 0
    let totalStepMsMean = record.totalStepMsMean ?? 0
    let actionL2Mean = record.actionL2Mean ?? 0
    let stateL2Mean = record.stateL2Mean ?? 0
    let plannerActiveFraction = record.plannerActiveFraction ?? 0
    let completionRate = globalEnvSteps > 0 ? Double(doneCount) / Double(globalEnvSteps) : 0

    return String(
        format: "[metrics] %@ env=%@%@ window_steps=%d-%d global_env_steps=%d vector_updates=%d%@%@ reward_sum=%.5f reward_mean=%.5f reward_std=%.5f done=%d resets=%d completion=%.5f sps=%.2f agent_ms=%.2f env_ms=%.2f total_ms=%.2f action_l2=%.5f state_l2=%.5f planner_active=%.5f",
        label,
        "\(record.env) policy=\(record.policy)",
        seedLabel,
        startStep,
        endStep,
        globalEnvSteps,
        vectorUpdates,
        measuredSecondsLabel,
        stopReasonLabel,
        rewardSum,
        rewardMean,
        rewardStd,
        doneCount,
        resetCount,
        completionRate,
        stepsPerSecond,
        agentStepMsMean,
        envStepMsMean,
        totalStepMsMean,
        actionL2Mean,
        stateL2Mean,
        plannerActiveFraction
    )
}

private func positiveInt(from rawValue: String?) -> Int? {
    guard let rawValue, let parsed = Int(rawValue), parsed > 0 else {
        return nil
    }
    return parsed
}

private func positiveDouble(from rawValue: String?) -> Double? {
    guard let rawValue, let parsed = Double(rawValue), parsed > 0 else {
        return nil
    }
    return parsed
}

private func deriveMetricsWindowSize(runControl: RunControl, numEnvs: Int) -> Int {
    let safeNumEnvs = max(numEnvs, 1)
    if let stopAfterMeasuredSteps = runControl.stopAfterMeasuredSteps {
        let estimatedVectorUpdates = max(Int(ceil(Double(stopAfterMeasuredSteps) / Double(safeNumEnvs))), 1)
        let targetSummaryCount = 8
        return max(Int(ceil(Double(estimatedVectorUpdates) / Double(targetSummaryCount))), 1)
    }
    if runControl.stopAfterMeasuredSeconds != nil {
        return 25
    }
    return 50
}

struct VectorSummary: Equatable {
    let mean: Double
    let std: Double
    let l2: Double
    let min: Double
    let max: Double

    init(_ values: UnsafeBufferPointer<Float>) {
        guard !values.isEmpty else {
            mean = 0
            std = 0
            l2 = 0
            min = 0
            max = 0
            return
        }

        var sum = 0.0
        var sumSquares = 0.0
        var minValue = metricsFinite(Double(values[0]))
        var maxValue = metricsFinite(Double(values[0]))

        for value in values {
            let numericValue = metricsFinite(Double(value))
            sum += numericValue
            sumSquares += numericValue * numericValue
            minValue = Swift.min(minValue, numericValue)
            maxValue = Swift.max(maxValue, numericValue)
        }

        let count = Double(values.count)
        mean = sum / count
        let variance = Swift.max((sumSquares / count) - (mean * mean), 0)
        std = sqrt(variance)
        l2 = sqrt(sumSquares)
        min = minValue
        max = maxValue
    }

    var telemetrySummary: TelemetryVectorSummary {
        TelemetryVectorSummary(mean: mean, std: std, l2: l2, min: min, max: max)
    }
}

private func resolveMetricsPath(_ path: String, rootDirectory: String) -> URL {
    if path.hasPrefix("/") {
        return URL(fileURLWithPath: path)
    }
    return URL(fileURLWithPath: rootDirectory).appendingPathComponent(path)
}

nonisolated(unsafe) private let metricsRecordFormatter: ISO8601DateFormatter = {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter
}()

private func metricsRecordTimestamp(_ date: Date) -> String {
    return metricsRecordFormatter.string(from: date)
}

private func metricsFilenameTimestamp(_ date: Date) -> String {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
    return formatter.string(from: date)
}
