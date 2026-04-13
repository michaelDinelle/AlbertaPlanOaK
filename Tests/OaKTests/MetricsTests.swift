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

    func testBenchmarkDefaultsToCoreLeague() throws {
        let rows = try resolveBenchmarkRows()
        let triples = Set(rows.map { "\($0.league)|\($0.logicalEnv)|\($0.concreteEnvId)" })
        let minigridRows = rows.filter { $0.league == "core" && $0.logicalEnv == "minigrid" }

        XCTAssertTrue(triples.contains("core|cheetah|HalfCheetah-v4"))
        XCTAssertTrue(triples.contains("core|walker|Walker2d-v4"))
        XCTAssertTrue(triples.contains("core|humanoid|Humanoid-v4"))
        XCTAssertGreaterThan(minigridRows.count, 20)
    }

    func testMetricsConfigurationDerivesWindowFromStepBudget() {
        let config = MetricsConfiguration.resolve(
            envName: "HalfCheetah-v4",
            policyName: "oak",
            runControl: RunControl(warmupSteps: 100, stopAfterMeasuredSteps: 50_000, stopAfterMeasuredSeconds: nil),
            numEnvs: 1024,
            rootDirectory: repoRoot.path,
            environment: ["OAK_METRICS_JSONL_PATH": repoRoot.appendingPathComponent("tmp/metrics.jsonl").path],
            processId: 123
        )

        XCTAssertEqual(config.windowSize, 7)
    }

    func testBenchmarkReportGeneratesSingleCanonicalJSONArtifact() throws {
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
                    "backend": "metal",
                    "step": 0,
                    "seed": 0,
                    "schemaFingerprint": schemaFingerprint,
                    "dynamicTelemetrySchema": [
                        "descriptors": [
                            ["key": "budget.input_ns", "unit": "ns"],
                            ["key": "perception.active_rank", "unit": "rank"],
                        ],
                    ],
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
                    "backend": "metal",
                    "step": 1,
                    "seed": 0,
                    "rewardSum": 5.0,
                    "rewardMean": 2.5,
                    "rewardStd": 0.5,
                    "doneCount": 1,
                    "resetCount": 1,
                    "measuredStepCount": 2,
                    "measuredSeconds": 0.2,
                    "vectorUpdateCount": 1,
                    "stepsPerSecond": 10.0,
                    "agentStepMsMean": 1.2,
                    "envStepMsMean": 2.4,
                    "totalStepMsMean": 3.6,
                    "actionL2Mean": 0.7,
                    "stateL2Mean": 1.4,
                    "plannerStepMsMean": 1.0,
                    "plannerActiveFraction": 0.5,
                    "activeRankMean": 2.0,
                    "activeRankMin": 1,
                    "activeRankMax": 3,
                    "activeGVFsMean": 4.0,
                    "activeOptionsMean": 5.0,
                    "dynamicTelemetryValues": [10_000.0, 2.0],
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
                    "backend": "random",
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
                    "backend": "random",
                    "step": 1,
                    "seed": 0,
                    "rewardSum": 1.0,
                    "rewardMean": 0.5,
                    "rewardStd": 0.25,
                    "doneCount": 0,
                    "resetCount": 0,
                    "measuredStepCount": 2,
                    "measuredSeconds": 0.15,
                    "vectorUpdateCount": 1,
                    "stepsPerSecond": 12.0,
                    "agentStepMsMean": 1.0,
                    "envStepMsMean": 2.0,
                    "totalStepMsMean": 3.0,
                    "actionL2Mean": 0.4,
                    "stateL2Mean": 0.8,
                    "plannerStepMsMean": 0.0,
                    "plannerActiveFraction": 0.0,
                    "activeRankMean": 1.0,
                    "activeRankMin": 1,
                    "activeRankMax": 1,
                    "activeGVFsMean": 0.0,
                    "activeOptionsMean": 0.0,
                    "stopReason": "measured_steps",
                ],
            ],
            to: randomMetricsPath
        )

        let manifestPath = try writeBenchmarkManifest(
            rows: [
                benchmarkManifestRow(
                    league: "smoke",
                    logicalEnv: "minigrid",
                    concreteEnvId: taskId,
                    backend: "metal",
                    policy: "oak",
                    seed: 0,
                    numEnvs: 2,
                    warmupSteps: 1,
                    stopSteps: 2,
                    runId: "oak-run",
                    metricsPath: oakMetricsPath,
                    logPath: runsDirectory.appendingPathComponent("oak.log"),
                    schemaFingerprint: schemaFingerprint,
                    observationSpaceKind: "dict",
                    observationFlatDim: 11,
                    actionSpaceKind: "discrete",
                    actionFlatDim: 7,
                    actionDiscreteCount: 7,
                    actionShape: [7]
                ),
                benchmarkManifestRow(
                    league: "smoke",
                    logicalEnv: "minigrid",
                    concreteEnvId: taskId,
                    backend: "random",
                    policy: "random",
                    seed: 0,
                    numEnvs: 2,
                    warmupSteps: 1,
                    stopSteps: 2,
                    runId: "random-run",
                    metricsPath: randomMetricsPath,
                    logPath: runsDirectory.appendingPathComponent("random.log"),
                    schemaFingerprint: schemaFingerprint,
                    observationSpaceKind: "dict",
                    observationFlatDim: 11,
                    actionSpaceKind: "discrete",
                    actionFlatDim: 7,
                    actionDiscreteCount: 7,
                    actionShape: [7]
                ),
            ],
            benchDirectory: benchDirectory
        )

        let result = try runBenchmarkReport(manifestPath: manifestPath, benchDirectory: benchDirectory)
        XCTAssertEqual(result.terminationStatus, 0, result.stderr)

        let benchmarkObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: result.benchmarkJSON)) as? [String: Any]
        )
        let benchmarkText = try String(contentsOf: result.benchmarkJSON, encoding: .utf8)
        XCTAssertEqual(
            Set(benchmarkObject.keys),
            [
                "baselinePolicy",
                "benchDir",
                "config",
                "configHash",
                "generatedAt",
                "gitRevision",
                "policies",
                "pythonVersions",
                "runs",
                "tasks",
            ]
        )
        XCTAssertEqual(benchmarkObject["baselinePolicy"] as? String, "random")
        XCTAssertNil(benchmarkObject["promotionEligible"])
        XCTAssertNil(benchmarkObject["dualAxisComplete"])
        XCTAssertNil(benchmarkObject["topImprovements"])
        XCTAssertFalse(benchmarkText.contains("planner_never_engaged"))
        XCTAssertFalse(benchmarkText.contains("active_rank_stuck_at_minimum"))
        XCTAssertFalse(benchmarkText.contains("zero_completions_sparse_suite"))

        let config = try XCTUnwrap(benchmarkObject["config"] as? [String: Any])
        XCTAssertEqual(config["resolvedTaskCount"] as? Int, 1)
        XCTAssertEqual(config["runCount"] as? Int, 2)

        let runs = try XCTUnwrap(benchmarkObject["runs"] as? [[String: Any]])
        XCTAssertEqual(runs.count, 2)
        XCTAssertNil(runs.first?["errors"])

        let tasks = try XCTUnwrap(benchmarkObject["tasks"] as? [[String: Any]])
        XCTAssertEqual(tasks.count, 1)

        let task = try XCTUnwrap(tasks.first)
        XCTAssertEqual(task["concreteEnvId"] as? String, taskId)
        XCTAssertEqual(task["logicalEnv"] as? String, "minigrid")
        XCTAssertNil(task["errors"])

        let policies = try XCTUnwrap(task["policies"] as? [String: Any])
        let oakPolicy = try XCTUnwrap(policies["oak"] as? [String: Any])
        XCTAssertNotNil(oakPolicy["plannerStepMsMean"])
        XCTAssertNotNil(oakPolicy["plannerActiveFraction"])
        XCTAssertNotNil(oakPolicy["activeGVFsMean"])
        XCTAssertNotNil(oakPolicy["activeOptionsMean"])
        XCTAssertNotNil(oakPolicy["actionL2Mean"])
        XCTAssertNotNil(oakPolicy["stateL2Mean"])
        XCTAssertNotNil(oakPolicy["agentStepMsMean"])
        XCTAssertNotNil(oakPolicy["envStepMsMean"])
        XCTAssertNotNil(oakPolicy["totalStepMsMean"])
        XCTAssertNotNil(oakPolicy["stopReasonCounts"])
        let oakDynamicSchema = try XCTUnwrap(oakPolicy["dynamicTelemetrySchema"] as? [[String: Any]])
        XCTAssertEqual(oakDynamicSchema.count, 2)
        let oakDynamicTelemetry = try XCTUnwrap(oakPolicy["dynamicTelemetry"] as? [String: Any])
        XCTAssertNotNil(oakDynamicTelemetry["budget.input_ns"])
        XCTAssertNotNil(oakDynamicTelemetry["perception.active_rank"])

        let comparisons = try XCTUnwrap(task["comparisons"] as? [String: Any])
        let comparison = try XCTUnwrap(comparisons["oak__vs__random"] as? [String: Any])
        XCTAssertNotNil(comparison["rewardSumDelta"])
        XCTAssertNotNil(comparison["rewardMeanDelta"])
        XCTAssertNotNil(comparison["stepsPerSecondDelta"])
        XCTAssertNotNil(comparison["completionRateDelta"])

        XCTAssertFalse(FileManager.default.fileExists(atPath: benchDirectory.appendingPathComponent("summary.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: benchDirectory.appendingPathComponent("task_summary.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: benchDirectory.appendingPathComponent("family_summary.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: benchDirectory.appendingPathComponent("research_verdict.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: benchDirectory.appendingPathComponent("summary.md").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: benchDirectory.appendingPathComponent("REPORT.md").path))
    }

    func testBenchmarkReportTreatsSchemaDriftAsError() throws {
        let tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let benchDirectory = tempDirectory.appendingPathComponent("bench", isDirectory: true)
        let runsDirectory = benchDirectory.appendingPathComponent("runs", isDirectory: true)
        try FileManager.default.createDirectory(at: runsDirectory, withIntermediateDirectories: true)

        let taskId = "MiniGrid-Empty-8x8-v0"
        let metricsPath0 = runsDirectory.appendingPathComponent("oak-seed0.jsonl")
        let metricsPath1 = runsDirectory.appendingPathComponent("oak-seed1.jsonl")

        try writeJSONL([benchmarkRunStart(runId: "oak-0", logicalEnv: "minigrid", concreteEnvId: taskId, league: "smoke", policy: "oak", backend: "metal", seed: 0, schemaFingerprint: "schema-a"), benchmarkRunEnd(runId: "oak-0", logicalEnv: "minigrid", concreteEnvId: taskId, league: "smoke", policy: "oak", backend: "metal", seed: 0)], to: metricsPath0)
        try writeJSONL([benchmarkRunStart(runId: "oak-1", logicalEnv: "minigrid", concreteEnvId: taskId, league: "smoke", policy: "oak", backend: "metal", seed: 1, schemaFingerprint: "schema-b"), benchmarkRunEnd(runId: "oak-1", logicalEnv: "minigrid", concreteEnvId: taskId, league: "smoke", policy: "oak", backend: "metal", seed: 1)], to: metricsPath1)

        let manifestPath = try writeBenchmarkManifest(
            rows: [
                benchmarkManifestRow(league: "smoke", logicalEnv: "minigrid", concreteEnvId: taskId, backend: "metal", policy: "oak", seed: 0, numEnvs: 1, warmupSteps: 1, stopSteps: 2, runId: "oak-0", metricsPath: metricsPath0, logPath: runsDirectory.appendingPathComponent("oak-seed0.log"), schemaFingerprint: "schema-a", observationSpaceKind: "dict", observationFlatDim: 11, actionSpaceKind: "discrete", actionFlatDim: 7, actionDiscreteCount: 7, actionShape: [7]),
                benchmarkManifestRow(league: "smoke", logicalEnv: "minigrid", concreteEnvId: taskId, backend: "metal", policy: "oak", seed: 1, numEnvs: 1, warmupSteps: 1, stopSteps: 2, runId: "oak-1", metricsPath: metricsPath1, logPath: runsDirectory.appendingPathComponent("oak-seed1.log"), schemaFingerprint: "schema-b", observationSpaceKind: "dict", observationFlatDim: 11, actionSpaceKind: "discrete", actionFlatDim: 7, actionDiscreteCount: 7, actionShape: [7]),
            ],
            benchDirectory: benchDirectory
        )

        let result = try runBenchmarkReport(manifestPath: manifestPath, benchDirectory: benchDirectory, baselinePolicy: "oak")
        XCTAssertEqual(result.terminationStatus, 1, result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: result.benchmarkJSON.path))
        XCTAssertTrue(result.stderr.contains("schema drift within task"))
    }

    func testBenchmarkReportFailsOnIncompleteRunCoverage() throws {
        let tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let benchDirectory = tempDirectory.appendingPathComponent("bench", isDirectory: true)
        let runsDirectory = benchDirectory.appendingPathComponent("runs", isDirectory: true)
        try FileManager.default.createDirectory(at: runsDirectory, withIntermediateDirectories: true)

        let taskId = "MiniGrid-Empty-8x8-v0"
        let metricsPath0 = runsDirectory.appendingPathComponent("oak-seed0.jsonl")
        let metricsPath1 = runsDirectory.appendingPathComponent("oak-seed1.jsonl")

        try writeJSONL(
            [
                benchmarkRunStart(runId: "oak-0", logicalEnv: "minigrid", concreteEnvId: taskId, league: "smoke", policy: "oak", backend: "metal", seed: 0, schemaFingerprint: "schema-a"),
                benchmarkRunEnd(runId: "oak-0", logicalEnv: "minigrid", concreteEnvId: taskId, league: "smoke", policy: "oak", backend: "metal", seed: 0),
            ],
            to: metricsPath0
        )
        try writeJSONL(
            [
                benchmarkRunStart(runId: "oak-1", logicalEnv: "minigrid", concreteEnvId: taskId, league: "smoke", policy: "oak", backend: "metal", seed: 1, schemaFingerprint: "schema-a"),
            ],
            to: metricsPath1
        )

        let manifestPath = try writeBenchmarkManifest(
            rows: [
                benchmarkManifestRow(league: "smoke", logicalEnv: "minigrid", concreteEnvId: taskId, backend: "metal", policy: "oak", seed: 0, numEnvs: 1, warmupSteps: 1, stopSteps: 2, runId: "oak-0", metricsPath: metricsPath0, logPath: runsDirectory.appendingPathComponent("oak-seed0.log"), schemaFingerprint: "schema-a", observationSpaceKind: "dict", observationFlatDim: 11, actionSpaceKind: "discrete", actionFlatDim: 7, actionDiscreteCount: 7, actionShape: [7]),
                benchmarkManifestRow(league: "smoke", logicalEnv: "minigrid", concreteEnvId: taskId, backend: "metal", policy: "oak", seed: 1, numEnvs: 1, warmupSteps: 1, stopSteps: 2, runId: "oak-1", metricsPath: metricsPath1, logPath: runsDirectory.appendingPathComponent("oak-seed1.log"), schemaFingerprint: "schema-a", observationSpaceKind: "dict", observationFlatDim: 11, actionSpaceKind: "discrete", actionFlatDim: 7, actionDiscreteCount: 7, actionShape: [7]),
            ],
            benchDirectory: benchDirectory
        )

        let result = try runBenchmarkReport(manifestPath: manifestPath, benchDirectory: benchDirectory, baselinePolicy: "oak")
        XCTAssertEqual(result.terminationStatus, 1, result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: result.benchmarkJSON.path))
        XCTAssertTrue(result.stderr.contains("incomplete run coverage"))
    }

    func testBenchmarkReportTreatsTaskResolutionMismatchAsError() throws {
        let tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let benchDirectory = tempDirectory.appendingPathComponent("bench", isDirectory: true)
        let runsDirectory = benchDirectory.appendingPathComponent("runs", isDirectory: true)
        try FileManager.default.createDirectory(at: runsDirectory, withIntermediateDirectories: true)

        let declaredTaskId = "MiniGrid-Empty-8x8-v0"
        let recordedTaskId = "MiniGrid-DoorKey-8x8-v0"
        let metricsPath = runsDirectory.appendingPathComponent("oak.jsonl")

        try writeJSONL([benchmarkRunStart(runId: "oak-run", logicalEnv: "minigrid", concreteEnvId: recordedTaskId, league: "smoke", policy: "oak", backend: "metal", seed: 0, schemaFingerprint: "schema-a"), benchmarkRunEnd(runId: "oak-run", logicalEnv: "minigrid", concreteEnvId: recordedTaskId, league: "smoke", policy: "oak", backend: "metal", seed: 0)], to: metricsPath)

        let manifestPath = try writeBenchmarkManifest(
            rows: [
                benchmarkManifestRow(league: "smoke", logicalEnv: "minigrid", concreteEnvId: declaredTaskId, backend: "metal", policy: "oak", seed: 0, numEnvs: 1, warmupSteps: 1, stopSteps: 2, runId: "oak-run", metricsPath: metricsPath, logPath: runsDirectory.appendingPathComponent("oak.log"), schemaFingerprint: "schema-a", observationSpaceKind: "dict", observationFlatDim: 11, actionSpaceKind: "discrete", actionFlatDim: 7, actionDiscreteCount: 7, actionShape: [7]),
            ],
            benchDirectory: benchDirectory
        )

        let result = try runBenchmarkReport(manifestPath: manifestPath, benchDirectory: benchDirectory, baselinePolicy: "oak")
        XCTAssertEqual(result.terminationStatus, 1, result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: result.benchmarkJSON.path))
        XCTAssertTrue(result.stderr.contains("task resolution mismatch"))
    }

    func testBenchmarkReportTreatsNonFiniteMetricAsError() throws {
        let tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let benchDirectory = tempDirectory.appendingPathComponent("bench", isDirectory: true)
        let runsDirectory = benchDirectory.appendingPathComponent("runs", isDirectory: true)
        try FileManager.default.createDirectory(at: runsDirectory, withIntermediateDirectories: true)

        let taskId = "MiniGrid-Empty-8x8-v0"
        let metricsPath = runsDirectory.appendingPathComponent("oak.jsonl")
        let lines = [
            #"{"backend":"metal","concreteEnvId":"MiniGrid-Empty-8x8-v0","env":"MiniGrid-Empty-8x8-v0","league":"smoke","logicalEnv":"minigrid","policy":"oak","runId":"oak-run","schemaFingerprint":"schema-a","seed":0,"step":0,"timestamp":"2026-04-10T00:00:00Z","type":"run_start"}"#,
            #"{"actionL2Mean":0.5,"activeGVFsMean":3.0,"activeOptionsMean":4.0,"activeRankMax":2,"activeRankMean":2.0,"activeRankMin":1,"agentStepMsMean":1.0,"backend":"metal","concreteEnvId":"MiniGrid-Empty-8x8-v0","doneCount":1,"env":"MiniGrid-Empty-8x8-v0","envStepMsMean":2.0,"league":"smoke","logicalEnv":"minigrid","measuredSeconds":0.2,"measuredStepCount":2,"plannerActiveFraction":0.25,"plannerStepMsMean":0.5,"policy":"oak","resetCount":1,"rewardMean":0.5,"rewardStd":NaN,"rewardSum":1.0,"runId":"oak-run","seed":0,"stateL2Mean":1.0,"step":1,"stopReason":"measured_steps","stepsPerSecond":10.0,"timestamp":"2026-04-10T00:00:01Z","totalStepMsMean":3.0,"type":"run_end","vectorUpdateCount":1}"#,
        ]
        try (lines.joined(separator: "\n") + "\n").write(to: metricsPath, atomically: true, encoding: .utf8)

        let manifestPath = try writeBenchmarkManifest(
            rows: [
                benchmarkManifestRow(league: "smoke", logicalEnv: "minigrid", concreteEnvId: taskId, backend: "metal", policy: "oak", seed: 0, numEnvs: 1, warmupSteps: 1, stopSteps: 2, runId: "oak-run", metricsPath: metricsPath, logPath: runsDirectory.appendingPathComponent("oak.log"), schemaFingerprint: "schema-a", observationSpaceKind: "dict", observationFlatDim: 11, actionSpaceKind: "discrete", actionFlatDim: 7, actionDiscreteCount: 7, actionShape: [7]),
            ],
            benchDirectory: benchDirectory
        )

        let result = try runBenchmarkReport(manifestPath: manifestPath, benchDirectory: benchDirectory, baselinePolicy: "oak")
        XCTAssertEqual(result.terminationStatus, 1, result.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: result.benchmarkJSON.path))
        XCTAssertTrue(result.stderr.contains("non-finite numeric value"))
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
        let telemetrySchema = DynamicTelemetrySchema(
            descriptors: [
                DynamicTelemetryDescriptor(key: "budget.input_ns", unit: "ns"),
                DynamicTelemetryDescriptor(key: "perception.active_rank", unit: "rank"),
            ]
        )
        let aggregator = makeAggregator(runId: "run-jsonl", envName: "pyflyt", logicalEnvName: "pyflyt", policyName: "random", backend: "random", windowSize: 1, dynamicTelemetrySchema: telemetrySchema, seed: 7)
        let control = RunControl(warmupSteps: 5, stopAfterMeasuredSteps: 10, stopAfterMeasuredSeconds: 3)

        try sink.write(aggregator.runStartRecord(control: control, at: Date(timeIntervalSince1970: 20)))

        let stepRecords = aggregator.recordStep(
            rewards: [2, 4],
            dones: [false, true],
            telemetry: makeTelemetry(rewardInput: 0, backend: "random", dynamicTelemetryValues: [10_000.0, 1.0]),
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
        XCTAssertNotNil(runStart["dynamicTelemetrySchema"])
        XCTAssertNil(runStart["reward"])

        let step = try jsonObject(from: lines[1])
        XCTAssertEqual(step["type"] as? String, "step")
        XCTAssertEqual(step["env"] as? String, "pyflyt")
        XCTAssertEqual(step["policy"] as? String, "random")
        XCTAssertEqual(step["backend"] as? String, "random")
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
        XCTAssertEqual(step["dynamicTelemetryValues"] as? [Double], [10_000.0, 1.0])
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
        XCTAssertEqual(summary["dynamicTelemetryValues"] as? [Double], [10_000.0, 1.0])

        let runEnd = try jsonObject(from: lines[3])
        XCTAssertEqual(runEnd["type"] as? String, "run_end")
        XCTAssertNotNil(runEnd["rewardSum"])
        XCTAssertNotNil(runEnd["windowStartStep"])
        XCTAssertNotNil(runEnd["windowEndStep"])
        XCTAssertEqual(runEnd["stopReason"] as? String, "measured_steps")
        XCTAssertEqual(runEnd["measuredStepCount"] as? Int, 2)
        XCTAssertEqual(runEnd["measuredSeconds"] as? Double, 1.5)
        XCTAssertEqual(runEnd["vectorUpdateCount"] as? Int, 1)
        XCTAssertEqual(runEnd["dynamicTelemetryValues"] as? [Double], [10_000.0, 1.0])
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
        dynamicTelemetryValues: [Double] = [],
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
                        activeRank: activeRank,
                        dynamicTelemetryValues: dynamicTelemetryValues
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
        dynamicTelemetrySchema: DynamicTelemetrySchema = .empty,
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
            dynamicTelemetrySchema: dynamicTelemetrySchema,
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

    private func resolveBenchmarkRows(_ request: String? = nil, environment: [String: String] = [:]) throws -> [ResolvedBenchmarkRow] {
        let scriptPath = repoRoot.appendingPathComponent("benchmark.sh").path

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [scriptPath, "--resolve-only"] + (request.map { [$0] } ?? [])
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

    private func benchmarkRunStart(
        runId: String,
        logicalEnv: String,
        concreteEnvId: String,
        league: String,
        policy: String,
        backend: String,
        seed: Int,
        schemaFingerprint: String
    ) -> [String: Any] {
        [
            "type": "run_start",
            "timestamp": "2026-04-10T00:00:00Z",
            "runId": runId,
            "env": concreteEnvId,
            "logicalEnv": logicalEnv,
            "concreteEnvId": concreteEnvId,
            "league": league,
            "policy": policy,
            "backend": backend,
            "step": 0,
            "seed": seed,
            "schemaFingerprint": schemaFingerprint,
            "observationSpaceKind": "dict",
            "observationFlatDim": 11,
            "actionSpaceKind": "discrete",
            "actionFlatDim": 7,
            "actionDiscreteCount": 7,
            "actionShape": [7],
        ]
    }

    private func benchmarkRunEnd(
        runId: String,
        logicalEnv: String,
        concreteEnvId: String,
        league: String,
        policy: String,
        backend: String,
        seed: Int
    ) -> [String: Any] {
        [
            "type": "run_end",
            "timestamp": "2026-04-10T00:00:01Z",
            "runId": runId,
            "env": concreteEnvId,
            "logicalEnv": logicalEnv,
            "concreteEnvId": concreteEnvId,
            "league": league,
            "policy": policy,
            "backend": backend,
            "step": 1,
            "seed": seed,
            "rewardSum": 1.0,
            "rewardMean": 0.5,
            "rewardStd": 0.25,
            "doneCount": 1,
            "resetCount": 1,
            "measuredStepCount": 2,
            "measuredSeconds": 0.2,
            "vectorUpdateCount": 1,
            "stepsPerSecond": 10.0,
            "agentStepMsMean": 1.0,
            "envStepMsMean": 2.0,
            "totalStepMsMean": 3.0,
            "actionL2Mean": 0.5,
            "stateL2Mean": 1.0,
            "plannerStepMsMean": 0.5,
            "plannerActiveFraction": 0.25,
            "activeRankMean": 2.0,
            "activeRankMin": 1,
            "activeRankMax": 2,
            "activeGVFsMean": 3.0,
            "activeOptionsMean": 4.0,
            "stopReason": "measured_steps",
        ]
    }

    private func benchmarkManifestRow(
        league: String,
        logicalEnv: String,
        concreteEnvId: String,
        backend: String,
        policy: String,
        seed: Int,
        numEnvs: Int,
        warmupSteps: Int,
        stopSteps: Int,
        runId: String,
        metricsPath: URL,
        logPath: URL,
        schemaFingerprint: String,
        observationSpaceKind: String,
        observationFlatDim: Int,
        actionSpaceKind: String,
        actionFlatDim: Int,
        actionDiscreteCount: Int,
        actionShape: [Int]
    ) -> String {
        [
            league,
            logicalEnv,
            concreteEnvId,
            backend,
            policy,
            String(seed),
            String(numEnvs),
            String(warmupSteps),
            String(stopSteps),
            runId,
            metricsPath.path,
            logPath.path,
            schemaFingerprint,
            observationSpaceKind,
            String(observationFlatDim),
            actionSpaceKind,
            String(actionFlatDim),
            String(actionDiscreteCount),
            String(data: try! JSONSerialization.data(withJSONObject: actionShape, options: []), encoding: .utf8)!,
        ].joined(separator: "\t")
    }

    private func writeBenchmarkManifest(rows: [String], benchDirectory: URL) throws -> URL {
        let manifestPath = benchDirectory.appendingPathComponent("manifest.tsv")
        let header = [
            "league", "logical_env", "concrete_env_id", "backend", "policy", "seed",
            "num_envs", "warmup_steps", "stop_steps", "run_id", "metrics_path",
            "log_path", "schema_fingerprint", "observation_space_kind", "observation_flat_dim",
            "action_space_kind", "action_flat_dim", "action_discrete_count", "action_shape",
        ].joined(separator: "\t")
        try ([header] + rows).joined(separator: "\n").appending("\n").write(to: manifestPath, atomically: true, encoding: .utf8)
        return manifestPath
    }

    private func runBenchmarkReport(
        manifestPath: URL,
        benchDirectory: URL,
        baselinePolicy: String = "random"
    ) throws -> (terminationStatus: Int32, stderr: String, benchmarkJSON: URL) {
        let benchmarkJSON = benchDirectory.appendingPathComponent("benchmark.json")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [
            repoRoot.appendingPathComponent("scripts/benchmark_report.py").path,
            "--bench-dir", benchDirectory.path,
            "--manifest", manifestPath.path,
            "--benchmark-json", benchmarkJSON.path,
            "--git-revision", "test-revision",
            "--python-bin", "/usr/bin/python3",
            "--baseline-policy", baselinePolicy,
        ]
        process.currentDirectoryURL = repoRoot
        let stderr = Pipe()
        process.standardError = stderr

        try process.run()
        process.waitUntilExit()

        let stderrText = String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        return (process.terminationStatus, stderrText, benchmarkJSON)
    }
}
