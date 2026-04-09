import Darwin
import Foundation

struct AgentStepResult {
    let action: [Float]
    let telemetry: AgentTelemetry
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
    let cmd1Ms: Double?
    let cmd2Ms: Double?
    let cmd3Ms: Double?
    let cmd4Ms: Double?
    let cmd5Ms: Double?

    init(
        rewardInput: Float,
        obs: [Float],
        state: [Float],
        action: [Float],
        agentStepMs: Double,
        backend: String,
        metalTimings: MetalCommandTimings? = nil
    ) {
        let obsStats = VectorSummary(obs)
        let actionStats = VectorSummary(action)
        let stateStats = VectorSummary(state)

        self.rewardInput = Double(rewardInput)
        obsMean = obsStats.mean
        obsStd = obsStats.std
        obsL2 = obsStats.l2
        actionMean = actionStats.mean
        actionStd = actionStats.std
        actionL2 = actionStats.l2
        actionMin = actionStats.min
        actionMax = actionStats.max
        stateMean = stateStats.mean
        stateStd = stateStats.std
        stateL2 = stateStats.l2
        self.agentStepMs = agentStepMs
        self.backend = backend
        cmd1Ms = metalTimings?.cmd1Ms
        cmd2Ms = metalTimings?.cmd2Ms
        cmd3Ms = metalTimings?.cmd3Ms
        cmd4Ms = metalTimings?.cmd4Ms
        cmd5Ms = metalTimings?.cmd5Ms
    }

    static func randomPolicy(rewardInput: Float, obs: [Float], action: [Float], agentStepMs: Double) -> AgentTelemetry {
        AgentTelemetry(
            rewardInput: rewardInput,
            obs: obs,
            state: [],
            action: action,
            agentStepMs: agentStepMs,
            backend: "random"
        )
    }
}

struct MetalCommandTimings: Equatable {
    let cmd1Ms: Double
    let cmd2Ms: Double
    let cmd3Ms: Double
    let cmd4Ms: Double
    let cmd5Ms: Double
}

struct RollingWindowStats: Equatable {
    private(set) var count = 0
    private(set) var startStep = 0
    private(set) var endStep = 0
    private var rewardSum = 0.0
    private var rewardSquaredSum = 0.0
    private(set) var doneCount = 0
    private(set) var resetCount = 0
    private var agentStepMsSum = 0.0
    private var envStepMsSum = 0.0
    private var totalStepMsSum = 0.0
    private var actionL2Sum = 0.0
    private var stateL2Sum = 0.0

    mutating func record(step: Int, reward: Double, done: Bool, resetDelta: Int, telemetry: AgentTelemetry, envStepMs: Double, totalStepMs: Double) {
        if count == 0 {
            startStep = step
        }

        count += 1
        endStep = step
        rewardSum += reward
        rewardSquaredSum += reward * reward
        doneCount += done ? 1 : 0
        resetCount += resetDelta
        agentStepMsSum += telemetry.agentStepMs
        envStepMsSum += envStepMs
        totalStepMsSum += totalStepMs
        actionL2Sum += telemetry.actionL2
        stateL2Sum += telemetry.stateL2
    }

    var rewardMean: Double {
        count > 0 ? rewardSum / Double(count) : 0
    }

    var rewardStd: Double {
        guard count > 0 else {
            return 0
        }

        let mean = rewardMean
        let variance = max((rewardSquaredSum / Double(count)) - (mean * mean), 0)
        return sqrt(variance)
    }

    var stepsPerSecond: Double {
        guard totalStepMsSum > 0 else {
            return 0
        }
        return Double(count) / (totalStepMsSum / 1000)
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

    var rewardTotal: Double {
        rewardSum
    }

    mutating func reset() {
        self = RollingWindowStats()
    }
}

struct MetricsConfiguration {
    let windowSize: Int
    let fileURL: URL
    let emitStepRecords: Bool
    let emitSummaryRecords: Bool
    let emitStdoutSummaries: Bool

    init(windowSize: Int, fileURL: URL, emitStepRecords: Bool = true, emitSummaryRecords: Bool = true, emitStdoutSummaries: Bool = true) {
        self.windowSize = windowSize
        self.fileURL = fileURL
        self.emitStepRecords = emitStepRecords
        self.emitSummaryRecords = emitSummaryRecords
        self.emitStdoutSummaries = emitStdoutSummaries
    }

    static func resolve(envName: String, policyName: String, rootDirectory: String? = nil, environment: [String: String] = ProcessInfo.processInfo.environment, processId: Int32 = getpid()) -> MetricsConfiguration {
        let fileManager = FileManager.default
        let baseDirectory = rootDirectory ?? environment["OAK_ROOT_DIR"] ?? fileManager.currentDirectoryPath
        let windowSize = max(environment["OAK_METRICS_WINDOW"].flatMap(Int.init) ?? 100, 1)
        let emitStepRecords = metricsFlag(named: "OAK_METRICS_STEP_RECORDS", defaultValue: true, environment: environment)
        let emitSummaryRecords = metricsFlag(named: "OAK_METRICS_SUMMARY_RECORDS", defaultValue: true, environment: environment)
        let emitStdoutSummaries = metricsFlag(named: "OAK_METRICS_STDOUT_SUMMARIES", defaultValue: true, environment: environment)

        if let overridePath = environment["OAK_METRICS_JSONL_PATH"], !overridePath.isEmpty {
            return MetricsConfiguration(
                windowSize: windowSize,
                fileURL: resolveMetricsPath(overridePath, rootDirectory: baseDirectory),
                emitStepRecords: emitStepRecords,
                emitSummaryRecords: emitSummaryRecords,
                emitStdoutSummaries: emitStdoutSummaries
            )
        }

        let metricsDirectory = environment["OAK_METRICS_DIR"] ?? ".oak-metrics"
        let directoryURL = resolveMetricsPath(metricsDirectory, rootDirectory: baseDirectory)
        let filename = "\(metricsFilenameTimestamp(Date()))-\(envName)-\(policyName)-\(processId).jsonl"
        return MetricsConfiguration(
            windowSize: windowSize,
            fileURL: directoryURL.appendingPathComponent(filename),
            emitStepRecords: emitStepRecords,
            emitSummaryRecords: emitSummaryRecords,
            emitStdoutSummaries: emitStdoutSummaries
        )
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

    init(configuration: MetricsConfiguration) throws {
        let fileManager = FileManager.default
        let directoryURL = configuration.fileURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        if !fileManager.fileExists(atPath: configuration.fileURL.path) {
            fileManager.createFile(atPath: configuration.fileURL.path, contents: nil)
        }

        fileURL = configuration.fileURL
        handle = try FileHandle(forWritingTo: configuration.fileURL)
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

final class RunMetricsAggregator {
    let runId: String
    let envName: String
    let policyName: String
    let backend: String
    let windowSize: Int
    let seed: Int?

    private(set) var stepCount = 0
    private(set) var doneCount = 0
    private(set) var resetCount = 0
    private var totalStats = RollingWindowStats()
    private var windowStats = RollingWindowStats()

    init(runId: String = UUID().uuidString, envName: String, policyName: String, backend: String, windowSize: Int, seed: Int? = nil) {
        self.runId = runId
        self.envName = envName
        self.policyName = policyName
        self.backend = backend
        self.windowSize = max(windowSize, 1)
        self.seed = seed
    }

    func runStartRecord(control: RunControl = .unbounded, at date: Date = Date()) -> MetricsRecord {
        MetricsRecord(
            type: "run_start",
            timestamp: metricsRecordTimestamp(date),
            runId: runId,
            env: envName,
            policy: policyName,
            backend: backend,
            step: 0,
            seed: seed,
            warmupSteps: control.warmupSteps,
            stopAfterMeasuredSteps: control.stopAfterMeasuredSteps,
            stopAfterMeasuredSeconds: control.stopAfterMeasuredSeconds
        )
    }

    func recordStep(reward: Float, done: Bool, telemetry: AgentTelemetry, envStepMs: Double, totalStepMs: Double, at date: Date = Date()) -> (step: MetricsRecord, summary: MetricsRecord?) {
        stepCount += 1
        let rewardValue = Double(reward)
        let resetDelta = done ? 1 : 0
        doneCount += done ? 1 : 0
        resetCount += resetDelta

        totalStats.record(step: stepCount, reward: rewardValue, done: done, resetDelta: resetDelta, telemetry: telemetry, envStepMs: envStepMs, totalStepMs: totalStepMs)
        windowStats.record(step: stepCount, reward: rewardValue, done: done, resetDelta: resetDelta, telemetry: telemetry, envStepMs: envStepMs, totalStepMs: totalStepMs)

        let stepRecord = MetricsRecord.stepRecord(
            timestamp: metricsRecordTimestamp(date),
            runId: runId,
            env: envName,
            policy: policyName,
            backend: backend,
            step: stepCount,
            seed: seed,
            reward: rewardValue,
            done: done,
            resetCount: resetCount,
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
                stats: windowStats
            )
            windowStats.reset()
        }

        return (stepRecord, summaryRecord)
    }

    func runEndRecord(control: RunControl = .unbounded, stopReason: String = "completed", measuredSeconds: Double = 0, at date: Date = Date()) -> MetricsRecord {
        MetricsRecord.summaryRecord(
            type: "run_end",
            timestamp: metricsRecordTimestamp(date),
            runId: runId,
            env: envName,
            policy: policyName,
            backend: backend,
            step: stepCount,
            seed: seed,
            warmupSteps: control.warmupSteps,
            measuredStepCount: stepCount,
            measuredSeconds: measuredSeconds,
            stopReason: stopReason,
            stopAfterMeasuredSteps: control.stopAfterMeasuredSteps,
            stopAfterMeasuredSeconds: control.stopAfterMeasuredSeconds,
            stats: totalStats
        )
    }
}

struct MetricsRecord: Codable {
    let type: String
    let timestamp: String
    let runId: String
    let env: String
    let policy: String
    let backend: String
    let step: Int
    let seed: Int?
    let warmupSteps: Int?
    let measuredStepCount: Int?
    let measuredSeconds: Double?
    let stopReason: String?
    let stopAfterMeasuredSteps: Int?
    let stopAfterMeasuredSeconds: Double?
    let reward: Double?
    let done: Bool?
    let resetCount: Int?
    let rewardInput: Double?
    let obsMean: Double?
    let obsStd: Double?
    let obsL2: Double?
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

    init(
        type: String,
        timestamp: String,
        runId: String,
        env: String,
        policy: String,
        backend: String,
        step: Int,
        seed: Int? = nil,
        warmupSteps: Int? = nil,
        measuredStepCount: Int? = nil,
        measuredSeconds: Double? = nil,
        stopReason: String? = nil,
        stopAfterMeasuredSteps: Int? = nil,
        stopAfterMeasuredSeconds: Double? = nil,
        reward: Double? = nil,
        done: Bool? = nil,
        resetCount: Int? = nil,
        rewardInput: Double? = nil,
        obsMean: Double? = nil,
        obsStd: Double? = nil,
        obsL2: Double? = nil,
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
        stateL2Mean: Double? = nil
    ) {
        self.type = type
        self.timestamp = timestamp
        self.runId = runId
        self.env = env
        self.policy = policy
        self.backend = backend
        self.step = step
        self.seed = seed
        self.warmupSteps = warmupSteps
        self.measuredStepCount = measuredStepCount
        self.measuredSeconds = measuredSeconds
        self.stopReason = stopReason
        self.stopAfterMeasuredSteps = stopAfterMeasuredSteps
        self.stopAfterMeasuredSeconds = stopAfterMeasuredSeconds
        self.reward = reward
        self.done = done
        self.resetCount = resetCount
        self.rewardInput = rewardInput
        self.obsMean = obsMean
        self.obsStd = obsStd
        self.obsL2 = obsL2
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
            totalStepMs: totalStepMs
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
        warmupSteps: Int? = nil,
        measuredStepCount: Int? = nil,
        measuredSeconds: Double? = nil,
        stopReason: String? = nil,
        stopAfterMeasuredSteps: Int? = nil,
        stopAfterMeasuredSeconds: Double? = nil,
        stats: RollingWindowStats
    ) -> MetricsRecord {
        MetricsRecord(
            type: type,
            timestamp: timestamp,
            runId: runId,
            env: env,
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
            stateL2Mean: stats.stateL2Mean
        )
    }
}

func formatMetricsSummary(_ record: MetricsRecord) -> String {
    let label = record.type == "run_end" ? "final" : "window"
    let startStep = record.windowStartStep ?? record.step
    let endStep = record.windowEndStep ?? record.step
    let seedLabel = record.seed.map { " seed=\($0)" } ?? ""
    let measuredStepCount = record.measuredStepCount ?? endStep
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

    return String(
        format: "[metrics] %@ env=%@%@ steps=%d-%d measured_steps=%d%@%@ reward_sum=%.5f reward_mean=%.5f reward_std=%.5f done=%d resets=%d sps=%.2f agent_ms=%.2f env_ms=%.2f total_ms=%.2f action_l2=%.5f state_l2=%.5f",
        label,
        "\(record.env) policy=\(record.policy)",
        seedLabel,
        startStep,
        endStep,
        measuredStepCount,
        measuredSecondsLabel,
        stopReasonLabel,
        rewardSum,
        rewardMean,
        rewardStd,
        doneCount,
        resetCount,
        stepsPerSecond,
        agentStepMsMean,
        envStepMsMean,
        totalStepMsMean,
        actionL2Mean,
        stateL2Mean
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

private func metricsFlag(named name: String, defaultValue: Bool, environment: [String: String]) -> Bool {
    guard let rawValue = environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !rawValue.isEmpty else {
        return defaultValue
    }

    switch rawValue {
    case "1", "true", "yes", "on":
        return true
    case "0", "false", "no", "off":
        return false
    default:
        return defaultValue
    }
}

private struct VectorSummary: Equatable {
    let mean: Double
    let std: Double
    let l2: Double
    let min: Double
    let max: Double

    init(_ values: [Float]) {
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
        var minValue = Double(values[0])
        var maxValue = Double(values[0])

        for value in values {
            let numericValue = Double(value)
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
}

private func resolveMetricsPath(_ path: String, rootDirectory: String) -> URL {
    if path.hasPrefix("/") {
        return URL(fileURLWithPath: path)
    }
    return URL(fileURLWithPath: rootDirectory).appendingPathComponent(path)
}

private func metricsRecordTimestamp(_ date: Date) -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter.string(from: date)
}

private func metricsFilenameTimestamp(_ date: Date) -> String {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
    return formatter.string(from: date)
}
