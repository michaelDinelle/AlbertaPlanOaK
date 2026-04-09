import Foundation
import XCTest
@testable import OaK

final class MetricsTests: XCTestCase {
    func testRollingSummaryAggregatesKnownSequence() throws {
        let aggregator = RunMetricsAggregator(runId: "run-1", envName: "minigrid", policyName: "oak", backend: "metal", windowSize: 2)

        let step1 = aggregator.recordStep(
            reward: 1,
            done: false,
            telemetry: makeTelemetry(rewardInput: 0, agentStepMs: 2, state: [0.5, 0.5], action: [0.2, -0.2]),
            envStepMs: 3,
            totalStepMs: 5,
            at: Date(timeIntervalSince1970: 10)
        )
        XCTAssertEqual(step1.step.type, "step")
        XCTAssertNil(step1.summary)
        XCTAssertEqual(step1.step.resetCount, 0)

        let step2 = aggregator.recordStep(
            reward: 3,
            done: true,
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
        XCTAssertEqual(summary.rewardSum ?? 0, 4, accuracy: 0.0001)
        XCTAssertEqual(summary.rewardMean ?? 0, 2, accuracy: 0.0001)
        XCTAssertEqual(summary.rewardStd ?? 0, 1, accuracy: 0.0001)
        XCTAssertEqual(summary.doneCount, 1)
        XCTAssertEqual(summary.resetCount, 1)
        XCTAssertEqual(summary.agentStepMsMean ?? 0, 3, accuracy: 0.0001)
        XCTAssertEqual(summary.envStepMsMean ?? 0, 4, accuracy: 0.0001)
        XCTAssertEqual(summary.totalStepMsMean ?? 0, 6, accuracy: 0.0001)
        XCTAssertEqual(step2.step.resetCount, 1)
        XCTAssertGreaterThan(summary.stepsPerSecond ?? 0, 0)
    }

    func testJSONLSinkWritesExpectedRecordShapes() throws {
        let tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let fileURL = tempDirectory.appendingPathComponent("metrics.jsonl")
        let sink = try JSONLMetricsSink(configuration: MetricsConfiguration(windowSize: 1, fileURL: fileURL))
        let aggregator = RunMetricsAggregator(runId: "run-jsonl", envName: "pyflyt", policyName: "random", backend: "cpu", windowSize: 1, seed: 7)
        let control = RunControl(warmupSteps: 5, stopAfterMeasuredSteps: 10, stopAfterMeasuredSeconds: 3)

        try sink.write(aggregator.runStartRecord(control: control, at: Date(timeIntervalSince1970: 20)))

        let stepRecords = aggregator.recordStep(
            reward: 2,
            done: false,
            telemetry: makeTelemetry(rewardInput: 0, backend: "cpu"),
            envStepMs: 3,
            totalStepMs: 4,
            at: Date(timeIntervalSince1970: 21)
        )
        try sink.write(stepRecords.step)
        try sink.write(try XCTUnwrap(stepRecords.summary))
        try sink.write(aggregator.runEndRecord(control: control, stopReason: "measured_steps", measuredSeconds: 1.5, at: Date(timeIntervalSince1970: 22)))

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
        XCTAssertNil(runStart["reward"])

        let step = try jsonObject(from: lines[1])
        XCTAssertEqual(step["type"] as? String, "step")
        XCTAssertEqual(step["env"] as? String, "pyflyt")
        XCTAssertEqual(step["policy"] as? String, "random")
        XCTAssertEqual(step["backend"] as? String, "cpu")
        XCTAssertNotNil(step["reward"])
        XCTAssertNotNil(step["done"])
        XCTAssertNotNil(step["resetCount"])
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

        let runEnd = try jsonObject(from: lines[3])
        XCTAssertEqual(runEnd["type"] as? String, "run_end")
        XCTAssertNotNil(runEnd["rewardSum"])
        XCTAssertNotNil(runEnd["windowStartStep"])
        XCTAssertNotNil(runEnd["windowEndStep"])
        XCTAssertEqual(runEnd["stopReason"] as? String, "measured_steps")
        XCTAssertEqual(runEnd["measuredStepCount"] as? Int, 1)
        XCTAssertEqual(runEnd["measuredSeconds"] as? Double, 1.5)
    }

    func testResetAndDoneAccountingCarryAcrossContinuousRun() {
        let aggregator = RunMetricsAggregator(runId: "run-reset", envName: "cheetah", policyName: "oak", backend: "metal", windowSize: 10)

        let step1 = aggregator.recordStep(
            reward: 0.5,
            done: false,
            telemetry: makeTelemetry(rewardInput: 0),
            envStepMs: 1,
            totalStepMs: 2
        )
        XCTAssertEqual(step1.step.resetCount, 0)

        let step2 = aggregator.recordStep(
            reward: 1.5,
            done: true,
            telemetry: makeTelemetry(rewardInput: 0.5),
            envStepMs: 1,
            totalStepMs: 2
        )
        XCTAssertEqual(step2.step.resetCount, 1)

        let step3 = aggregator.recordStep(
            reward: 2.5,
            done: false,
            telemetry: makeTelemetry(rewardInput: 0),
            envStepMs: 1,
            totalStepMs: 2
        )
        XCTAssertEqual(step3.step.resetCount, 1)

        let runEnd = aggregator.runEndRecord()
        XCTAssertEqual(runEnd.type, "run_end")
        XCTAssertEqual(runEnd.doneCount, 1)
        XCTAssertEqual(runEnd.resetCount, 1)
        XCTAssertEqual(runEnd.windowStartStep, 1)
        XCTAssertEqual(runEnd.windowEndStep, 3)
    }

    private func makeTelemetry(
        rewardInput: Float,
        agentStepMs: Double = 1.5,
        backend: String = "metal",
        obs: [Float] = [1, 2, 3],
        state: [Float] = [0.25, 0.75],
        action: [Float] = [0.1, -0.2]
    ) -> AgentTelemetry {
        AgentTelemetry(
            rewardInput: rewardInput,
            obs: obs,
            state: state,
            action: action,
            agentStepMs: agentStepMs,
            backend: backend,
            metalTimings: backend == "metal"
                ? MetalCommandTimings(cmd1Ms: 0.1, cmd2Ms: 0.2, cmd3Ms: 0.3, cmd4Ms: 0.4, cmd5Ms: 0.5)
                : nil
        )
    }

    private func jsonObject(from line: Substring) throws -> [String: Any] {
        let data = Data(line.utf8)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
