import Foundation
import XCTest
@testable import OaK

final class MetricsTests: XCTestCase {
    func testDiscreteActionBatchUsesPerEnvArgmax() {
        let decoded = decodeDiscreteActionBatch(
            [
                -1.0, 0.25, 0.1, 0.9,
                3.5, 2.0, 1.5, -0.25,
            ],
            numEnvs: 2,
            actionCount: 4
        )

        XCTAssertEqual(decoded, [3, 0])
    }

    func testSchemaFingerprintIsStableAndSensitiveToSchemaChanges() {
        let baseline = EnvironmentSchemaFingerprint(
            observation: .box(flattenedDim: 12, shape: [3, 4]),
            action: .discrete(count: 7)
        )
        let same = EnvironmentSchemaFingerprint(
            observation: .box(flattenedDim: 12, shape: [3, 4]),
            action: .discrete(count: 7)
        )
        let changed = EnvironmentSchemaFingerprint(
            observation: .box(flattenedDim: 13, shape: [13]),
            action: .discrete(count: 7)
        )

        XCTAssertEqual(baseline.schemaFingerprint, same.schemaFingerprint)
        XCTAssertNotEqual(baseline.schemaFingerprint, changed.schemaFingerprint)
    }

    func testSmokeLeagueResolutionUsesConcreteTasks() throws {
        let rows = try resolveBenchmarkRows("smoke")
        let triples = Set(rows.map { "\($0.league)|\($0.logicalEnv)|\($0.concreteEnvId)" })

        XCTAssertTrue(triples.contains("smoke|cheetah|HalfCheetah-v4"))
        XCTAssertTrue(triples.contains("smoke|walker|Walker2d-v4"))
        XCTAssertTrue(triples.contains("smoke|humanoid|Humanoid-v4"))
        XCTAssertTrue(triples.contains("smoke|minigrid|MiniGrid-Empty-8x8-v0"))
        XCTAssertTrue(triples.contains("smoke|minigrid|MiniGrid-DoorKey-8x8-v0"))
        XCTAssertTrue(triples.contains("smoke|minigrid|MiniGrid-MemoryS9-v0"))
    }

    func testCoreLeagueIgnoresAmbientMinigridSelection() throws {
        let rows = try resolveBenchmarkRows(
            "core",
            environment: ["OAK_ENVPOOL_TASK_ID": "MiniGrid-Empty-8x8-v0"]
        )
        let minigridTasks = rows
            .filter { $0.league == "core" && $0.logicalEnv == "minigrid" }
            .map(\.concreteEnvId)

        XCTAssertGreaterThan(minigridTasks.count, 20)
        XCTAssertTrue(minigridTasks.contains("MiniGrid-Empty-8x8-v0"))
        XCTAssertTrue(minigridTasks.contains("MiniGrid-DoorKey-8x8-v0"))
        XCTAssertTrue(minigridTasks.contains("MiniGrid-MemoryS9-v0"))
    }

    func testAdhocMinigridSelectionUsesExplicitConcreteTask() throws {
        let rows = try resolveBenchmarkRows(
            "minigrid",
            environment: ["OAK_ENVPOOL_TASK_ID": "MiniGrid-Empty-8x8-v0"]
        )

        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows.first?.league, "adhoc")
        XCTAssertEqual(rows.first?.logicalEnv, "minigrid")
        XCTAssertEqual(rows.first?.concreteEnvId, "MiniGrid-Empty-8x8-v0")
    }

    func testBenchmarkReportGeneratesTaskAndFamilyArtifacts() throws {
        let tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let benchDirectory = tempDirectory.appendingPathComponent("bench", isDirectory: true)
        let runsDirectory = benchDirectory.appendingPathComponent("runs", isDirectory: true)
        try FileManager.default.createDirectory(at: runsDirectory, withIntermediateDirectories: true)

        let taskId = "MiniGrid-Empty-8x8-v0"
        let schemaFingerprint = "schema-test"
        let oakMetricsPath = runsDirectory.appendingPathComponent("oak.jsonl")
        let randomMetricsPath = runsDirectory.appendingPathComponent("random.jsonl")

        try writeJSONL(
            [
                [
                    "type": "run_start",
                    "timestamp": "2026-04-10T00:00:00Z",
                    "runId": "oak-run",
                    "env": taskId,
                    "logicalEnv": "minigrid",
                    "concreteEnvId": taskId,
                    "league": "smoke",
                    "policy": "oak",
                    "backend": "cpu",
                    "step": 0,
                    "seed": 0,
                    "schemaFingerprint": schemaFingerprint,
                ],
                [
                    "type": "run_end",
                    "timestamp": "2026-04-10T00:00:01Z",
                    "runId": "oak-run",
                    "env": taskId,
                    "logicalEnv": "minigrid",
                    "concreteEnvId": taskId,
                    "league": "smoke",
                    "policy": "oak",
                    "backend": "cpu",
                    "step": 1,
                    "seed": 0,
                    "rewardSum": 5.0,
                    "rewardMean": 2.5,
                    "rewardStd": 0.5,
                    "doneCount": 1,
                    "resetCount": 1,
                    "measuredStepCount": 2,
                    "vectorUpdateCount": 1,
                    "stepsPerSecond": 10.0,
                    "plannerStepMsMean": 1.0,
                    "plannerActiveFraction": 0.5,
                    "activeRankMean": 2.0,
                    "activeRankMin": 1,
                    "activeRankMax": 3,
                    "stopReason": "measured_steps",
                ],
            ],
            to: oakMetricsPath
        )

        try writeJSONL(
            [
                [
                    "type": "run_start",
                    "timestamp": "2026-04-10T00:00:00Z",
                    "runId": "random-run",
                    "env": taskId,
                    "logicalEnv": "minigrid",
                    "concreteEnvId": taskId,
                    "league": "smoke",
                    "policy": "random",
                    "backend": "cpu",
                    "step": 0,
                    "seed": 0,
                    "schemaFingerprint": schemaFingerprint,
                ],
                [
                    "type": "run_end",
                    "timestamp": "2026-04-10T00:00:01Z",
                    "runId": "random-run",
                    "env": taskId,
                    "logicalEnv": "minigrid",
                    "concreteEnvId": taskId,
                    "league": "smoke",
                    "policy": "random",
                    "backend": "cpu",
                    "step": 1,
                    "seed": 0,
                    "rewardSum": 1.0,
                    "rewardMean": 0.5,
                    "rewardStd": 0.25,
                    "doneCount": 0,
                    "resetCount": 0,
                    "measuredStepCount": 2,
                    "vectorUpdateCount": 1,
                    "stepsPerSecond": 12.0,
                    "plannerStepMsMean": 0.0,
                    "plannerActiveFraction": 0.0,
                    "activeRankMean": 1.0,
                    "activeRankMin": 1,
                    "activeRankMax": 1,
                    "stopReason": "measured_steps",
                ],
            ],
            to: randomMetricsPath
        )

        let manifestPath = benchDirectory.appendingPathComponent("resolved_manifest.tsv")
        let header = [
            "league", "mode", "logical_env", "concrete_env_id", "backend", "policy", "seed",
            "num_envs", "warmup_steps", "stop_steps", "stop_seconds", "run_id", "metrics_path",
            "log_path", "schema_fingerprint", "observation_space_kind", "observation_flat_dim",
            "action_space_kind", "action_flat_dim", "action_discrete_count", "action_shape",
        ].joined(separator: "\t")
        let oakRow = [
            "smoke", "quality", "minigrid", taskId, "cpu", "oak", "0",
            "2", "1", "2", "", "oak-run", oakMetricsPath.path,
            runsDirectory.appendingPathComponent("oak.log").path, schemaFingerprint, "dict", "11",
            "discrete", "7", "7", "[7]",
        ].joined(separator: "\t")
        let randomRow = [
            "smoke", "quality", "minigrid", taskId, "cpu", "random", "0",
            "2", "1", "2", "", "random-run", randomMetricsPath.path,
            runsDirectory.appendingPathComponent("random.log").path, schemaFingerprint, "dict", "11",
            "discrete", "7", "7", "[7]",
        ].joined(separator: "\t")
        try (header + "\n" + oakRow + "\n" + randomRow + "\n").write(to: manifestPath, atomically: true, encoding: .utf8)

        let summaryJSON = benchDirectory.appendingPathComponent("summary.json")
        let taskSummaryJSON = benchDirectory.appendingPathComponent("task_summary.json")
        let familySummaryJSON = benchDirectory.appendingPathComponent("family_summary.json")
        let researchVerdictJSON = benchDirectory.appendingPathComponent("research_verdict.json")
        let summaryMD = benchDirectory.appendingPathComponent("summary.md")
        let reportMD = benchDirectory.appendingPathComponent("REPORT.md")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [
            repoRoot.appendingPathComponent("scripts/benchmark_report.py").path,
            "--bench-dir", benchDirectory.path,
            "--manifest", manifestPath.path,
            "--summary-json", summaryJSON.path,
            "--task-summary-json", taskSummaryJSON.path,
            "--family-summary-json", familySummaryJSON.path,
            "--research-verdict-json", researchVerdictJSON.path,
            "--summary-md", summaryMD.path,
            "--report-md", reportMD.path,
            "--git-revision", "test-revision",
            "--python-bin", "/usr/bin/python3",
            "--baseline-policy", "random",
        ]
        process.currentDirectoryURL = repoRoot
        let stderr = Pipe()
        process.standardError = stderr

        try process.run()
        process.waitUntilExit()

        let stderrText = String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        XCTAssertEqual(process.terminationStatus, 0, stderrText)

        let summaryObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: summaryJSON)) as? [String: Any]
        )
        let taskSummary = try XCTUnwrap(summaryObject["taskSummary"] as? [String: Any])
        let taskSummaries = try XCTUnwrap(taskSummary["taskSummaries"] as? [[String: Any]])
        XCTAssertEqual(taskSummaries.count, 1)
        XCTAssertEqual(taskSummaries.first?["concreteEnvId"] as? String, taskId)

        let researchVerdict = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: researchVerdictJSON)) as? [String: Any]
        )
        XCTAssertEqual(researchVerdict["baselinePolicy"] as? String, "random")
        XCTAssertEqual(researchVerdict["promotionEligible"] as? Bool, false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: familySummaryJSON.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: summaryMD.path))
    }

    func testRollingSummaryAggregatesKnownSequence() throws {
        let aggregator = makeAggregator(runId: "run-1", envName: "MiniGrid-Empty-8x8-v0", logicalEnvName: "minigrid", policyName: "oak", backend: "metal", windowSize: 2)

        let step1 = aggregator.recordStep(
            rewards: [1, 3],
            dones: [false, false],
            telemetry: makeTelemetry(rewardInput: 0, agentStepMs: 2, state: [0.5, 0.5], action: [0.2, -0.2]),
            envStepMs: 3,
            totalStepMs: 5,
            at: Date(timeIntervalSince1970: 10)
        )
        XCTAssertEqual(step1.step.type, "step")
        XCTAssertNil(step1.summary)
        XCTAssertEqual(step1.step.resetCount, 0)
        XCTAssertEqual(step1.step.rewardSumBatch ?? 0, 4, accuracy: 0.0001)
        XCTAssertEqual(step1.step.rewardMeanBatch ?? 0, 2, accuracy: 0.0001)
        XCTAssertEqual(step1.step.doneCountBatch, 0)
        XCTAssertEqual(step1.step.doneAnyBatch, false)

        let step2 = aggregator.recordStep(
            rewards: [5, 7],
            dones: [true, false],
            telemetry: makeTelemetry(rewardInput: 1, agentStepMs: 4, state: [1.0, 0.0], action: [0.4, -0.4]),
            envStepMs: 5,
            totalStepMs: 7,
            at: Date(timeIntervalSince1970: 11)
        )

        let summaryRecord: MetricsRecord? = step2.summary
        let summary = try XCTUnwrap(summaryRecord)
        XCTAssertEqual(summary.type, "summary")
        XCTAssertEqual(summary.windowStartStep, 1)
        XCTAssertEqual(summary.windowEndStep, 2)
        XCTAssertEqual(summary.rewardSum ?? 0, 16, accuracy: 0.0001)
        XCTAssertEqual(summary.rewardMean ?? 0, 4, accuracy: 0.0001)
        XCTAssertEqual(summary.rewardStd ?? 0, sqrt(5), accuracy: 0.0001)
        XCTAssertEqual(summary.doneCount, 1)
        XCTAssertEqual(summary.resetCount, 1)
        XCTAssertEqual(summary.agentStepMsMean ?? 0, 3, accuracy: 0.0001)
        XCTAssertEqual(summary.envStepMsMean ?? 0, 4, accuracy: 0.0001)
        XCTAssertEqual(summary.totalStepMsMean ?? 0, 6, accuracy: 0.0001)
        XCTAssertEqual(summary.plannerActiveFraction ?? 0, 0, accuracy: 0.0001)
        XCTAssertEqual(summary.activeRankMin, 0)
        XCTAssertEqual(summary.activeRankMax, 0)
        XCTAssertEqual(step2.step.resetCount, 1)
        XCTAssertGreaterThan(summary.stepsPerSecond ?? 0, 0)
    }

    func testJSONLSinkWritesExpectedRecordShapes() throws {
        let tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let fileURL = tempDirectory.appendingPathComponent("metrics.jsonl")
        let sink = try JSONLMetricsSink(configuration: MetricsConfiguration(windowSize: 1, fileURL: fileURL))
        let aggregator = makeAggregator(runId: "run-jsonl", envName: "pyflyt", logicalEnvName: "pyflyt", policyName: "random", backend: "cpu", windowSize: 1, seed: 7)
        let control = RunControl(warmupSteps: 5, stopAfterMeasuredSteps: 10, stopAfterMeasuredSeconds: 3)

        try sink.write(aggregator.runStartRecord(control: control, at: Date(timeIntervalSince1970: 20)))

        let stepRecords = aggregator.recordStep(
            rewards: [2, 4],
            dones: [false, true],
            telemetry: makeTelemetry(rewardInput: 0, backend: "cpu"),
            envStepMs: 3,
            totalStepMs: 4,
            at: Date(timeIntervalSince1970: 21)
        )
        try sink.write(stepRecords.step)
        try sink.write(try XCTUnwrap(stepRecords.summary))
        try sink.write(aggregator.runEndRecord(control: control, stopReason: "measured_steps", measuredSeconds: 1.5, globalMeasuredEnvSteps: 2, at: Date(timeIntervalSince1970: 22)))

        let fileContents = try String(contentsOf: fileURL, encoding: .utf8)
        let lines = fileContents.split(separator: "\n")
        XCTAssertEqual(lines.count, 4)

        let runStart = try jsonObject(from: lines[0])
        XCTAssertEqual(runStart["type"] as? String, "run_start")
        XCTAssertEqual(runStart["runId"] as? String, "run-jsonl")
        XCTAssertEqual(runStart["policy"] as? String, "random")
        XCTAssertEqual(runStart["seed"] as? Int, 7)
        XCTAssertEqual(runStart["warmupSteps"] as? Int, 5)
        XCTAssertEqual(runStart["stopAfterMeasuredSteps"] as? Int, 10)
        XCTAssertEqual(runStart["logicalEnv"] as? String, "pyflyt")
        XCTAssertEqual(runStart["concreteEnvId"] as? String, "pyflyt")
        XCTAssertEqual(runStart["league"] as? String, "test")
        XCTAssertNotNil(runStart["schemaFingerprint"])
        XCTAssertNil(runStart["reward"])

        let step = try jsonObject(from: lines[1])
        XCTAssertEqual(step["type"] as? String, "step")
        XCTAssertEqual(step["env"] as? String, "pyflyt")
        XCTAssertEqual(step["policy"] as? String, "random")
        XCTAssertEqual(step["backend"] as? String, "cpu")
        XCTAssertNotNil(step["reward"])
        XCTAssertNotNil(step["done"])
        XCTAssertNotNil(step["resetCount"])
        XCTAssertEqual(try XCTUnwrap(step["reward"] as? Double), 3, accuracy: 0.0001)
        XCTAssertEqual(try XCTUnwrap(step["rewardSumBatch"] as? Double), 6, accuracy: 0.0001)
        XCTAssertEqual(try XCTUnwrap(step["rewardMeanBatch"] as? Double), 3, accuracy: 0.0001)
        XCTAssertEqual(step["doneCountBatch"] as? Int, 1)
        XCTAssertEqual(step["doneAnyBatch"] as? Bool, true)
        XCTAssertNotNil(step["rewardInput"])
        XCTAssertNotNil(step["obsMean"])
        XCTAssertNotNil(step["actionMean"])
        XCTAssertNotNil(step["stateMean"])
        XCTAssertNotNil(step["agentStepMs"])
        XCTAssertNotNil(step["envStepMs"])
        XCTAssertNotNil(step["totalStepMs"])
        XCTAssertNil(step["obs"])
        XCTAssertNil(step["state"])
        XCTAssertNil(step["action"])

        let summary = try jsonObject(from: lines[2])
        XCTAssertEqual(summary["type"] as? String, "summary")
        XCTAssertNotNil(summary["rewardSum"])
        XCTAssertNotNil(summary["rewardMean"])
        XCTAssertNotNil(summary["rewardStd"])
        XCTAssertNotNil(summary["doneCount"])
        XCTAssertNotNil(summary["stepsPerSecond"])
        XCTAssertNotNil(summary["plannerActiveFraction"])
        XCTAssertNotNil(summary["activeRankMin"])
        XCTAssertNotNil(summary["activeRankMax"])

        let runEnd = try jsonObject(from: lines[3])
        XCTAssertEqual(runEnd["type"] as? String, "run_end")
        XCTAssertNotNil(runEnd["rewardSum"])
        XCTAssertNotNil(runEnd["windowStartStep"])
        XCTAssertNotNil(runEnd["windowEndStep"])
        XCTAssertEqual(runEnd["stopReason"] as? String, "measured_steps")
        XCTAssertEqual(runEnd["measuredStepCount"] as? Int, 2)
        XCTAssertEqual(runEnd["measuredSeconds"] as? Double, 1.5)
        XCTAssertEqual(runEnd["vectorUpdateCount"] as? Int, 1)
    }

    func testResetAndDoneAccountingCarryAcrossContinuousRun() {
        let aggregator = makeAggregator(runId: "run-reset", envName: "HalfCheetah-v4", logicalEnvName: "cheetah", policyName: "oak", backend: "metal", windowSize: 10)

        let step1 = aggregator.recordStep(
            rewards: [0.5, 1.5],
            dones: [false, false],
            telemetry: makeTelemetry(rewardInput: 0),
            envStepMs: 1,
            totalStepMs: 2
        )
        XCTAssertEqual(step1.step.resetCount, 0)

        let step2 = aggregator.recordStep(
            rewards: [1.5, 2.5],
            dones: [true, false],
            telemetry: makeTelemetry(rewardInput: 0.5),
            envStepMs: 1,
            totalStepMs: 2
        )
        XCTAssertEqual(step2.step.resetCount, 1)

        let step3 = aggregator.recordStep(
            rewards: [2.5, 3.5],
            dones: [false, false],
            telemetry: makeTelemetry(rewardInput: 0),
            envStepMs: 1,
            totalStepMs: 2
        )
        XCTAssertEqual(step3.step.resetCount, 1)

        let runEnd = aggregator.runEndRecord(globalMeasuredEnvSteps: 6)
        XCTAssertEqual(runEnd.type, "run_end")
        XCTAssertEqual(runEnd.doneCount, 1)
        XCTAssertEqual(runEnd.resetCount, 1)
        XCTAssertEqual(runEnd.windowStartStep, 1)
        XCTAssertEqual(runEnd.windowEndStep, 3)
        XCTAssertEqual(runEnd.measuredStepCount, 6)
        XCTAssertEqual(runEnd.vectorUpdateCount, 3)
    }

    func testPlannerActivityAndActiveRankRangeAreTracked() throws {
        let aggregator = makeAggregator(runId: "run-plan", envName: "HalfCheetah-v4", logicalEnvName: "cheetah", policyName: "oak", backend: "metal", windowSize: 2)

        _ = aggregator.recordStep(
            rewards: [1, 1],
            dones: [false, false],
            telemetry: makeTelemetry(rewardInput: 0, plannerStepMs: 0, activeRank: 1),
            envStepMs: 1,
            totalStepMs: 2
        )
        let second = aggregator.recordStep(
            rewards: [2, 2],
            dones: [false, false],
            telemetry: makeTelemetry(rewardInput: 1, plannerStepMs: 1.5, activeRank: 3),
            envStepMs: 1,
            totalStepMs: 2
        )

        let summary = try XCTUnwrap(second.summary)
        XCTAssertEqual(summary.plannerActiveFraction ?? 0, 0.5, accuracy: 0.0001)
        XCTAssertEqual(summary.activeRankMin, 1)
        XCTAssertEqual(summary.activeRankMax, 3)
    }

    func testSchemaFingerprintAppearsOnRunEnd() {
        let aggregator = makeAggregator(runId: "run-schema", envName: "HalfCheetah-v4", logicalEnvName: "cheetah", policyName: "oak", backend: "metal", windowSize: 10)
        _ = aggregator.recordStep(
            rewards: [1, 2],
            dones: [false, false],
            telemetry: makeTelemetry(rewardInput: 0),
            envStepMs: 1,
            totalStepMs: 2
        )
        let runEnd = aggregator.runEndRecord(globalMeasuredEnvSteps: 2)
        XCTAssertEqual(runEnd.logicalEnv, "cheetah")
        XCTAssertEqual(runEnd.concreteEnvId, "HalfCheetah-v4")
        XCTAssertEqual(runEnd.league, "test")
        XCTAssertEqual(runEnd.observationSpaceKind, "box")
        XCTAssertNotNil(runEnd.schemaFingerprint)
    }

    private func makeTelemetry(
        rewardInput: Float,
        agentStepMs: Double = 1.5,
        backend: String = "metal",
        plannerStepMs: Double = 0,
        activeRank: Int = 0,
        obs: [Float] = [1, 2, 3],
        state: [Float] = [0.25, 0.75],
        action: [Float] = [0.1, -0.2]
    ) -> AgentTelemetry {
        let mutableObs = obs
        let mutableState = state
        let mutableAction = action
        return mutableObs.withUnsafeBufferPointer { obsPtr in
            mutableState.withUnsafeBufferPointer { statePtr in
                mutableAction.withUnsafeBufferPointer { actionPtr in
                    AgentTelemetry(
                        rewardInput: rewardInput,
                        obs: obsPtr,
                        state: statePtr,
                        action: actionPtr,
                        agentStepMs: agentStepMs,
                        backend: backend,
                        metalTimings: backend == "metal"
                            ? MetalCommandTimings(cmd1Ms: 0.1, cmd2Ms: 0.2, cmd3Ms: 0.3, cmd4Ms: 0.4, cmd5Ms: 0.5)
                            : nil,
                        plannerStepMs: plannerStepMs,
                        activeRank: activeRank
                    )
                }
            }
        }
    }

    private func jsonObject(from line: Substring) throws -> [String: Any] {
        let data = Data(line.utf8)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func makeAggregator(
        runId: String,
        envName: String,
        logicalEnvName: String,
        policyName: String,
        backend: String,
        windowSize: Int,
        seed: Int? = nil
    ) -> RunMetricsAggregator {
        RunMetricsAggregator(
            runId: runId,
            envName: envName,
            logicalEnvName: logicalEnvName,
            concreteEnvId: envName,
            leagueName: "test",
            policyName: policyName,
            backend: backend,
            schema: EnvironmentSchemaFingerprint(
                observation: .box(flattenedDim: 4, shape: [4]),
                action: .box(flattenedDim: 2, shape: [2])
            ),
            windowSize: windowSize,
            seed: seed
        )
    }

    private struct ResolvedBenchmarkRow: Equatable {
        let league: String
        let logicalEnv: String
        let concreteEnvId: String
    }

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func resolveBenchmarkRows(_ request: String, environment: [String: String] = [:]) throws -> [ResolvedBenchmarkRow] {
        let scriptPath = repoRoot.appendingPathComponent("benchmark.sh").path

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [scriptPath, "--resolve-only", request]
        process.currentDirectoryURL = repoRoot
        process.environment = ProcessInfo.processInfo.environment.merging(environment) { _, new in new }

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        try process.run()
        process.waitUntilExit()

        let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        let stdoutText = String(decoding: stdoutData, as: UTF8.self)
        let stderrText = String(decoding: stderrData, as: UTF8.self)

        XCTAssertEqual(process.terminationStatus, 0, "benchmark resolver failed: \(stderrText)")
        XCTAssertFalse(stdoutText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "benchmark resolver produced no output")

        return stdoutText
            .split(whereSeparator: \.isNewline)
            .compactMap { line -> ResolvedBenchmarkRow? in
                let columns = line.split(separator: "\t", omittingEmptySubsequences: false)
                guard columns.count == 3 else { return nil }
                return ResolvedBenchmarkRow(
                    league: String(columns[0]),
                    logicalEnv: String(columns[1]),
                    concreteEnvId: String(columns[2])
                )
            }
    }

    private func writeJSONL(_ records: [[String: Any]], to url: URL) throws {
        let lines = try records.map { record in
            let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
            return try XCTUnwrap(String(data: data, encoding: .utf8))
        }
        try (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
    }
}
