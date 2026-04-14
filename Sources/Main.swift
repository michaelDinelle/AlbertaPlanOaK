import Foundation
import Metal
import PythonKit

typealias JSON = [String: Any]

final class PythonRuntime: @unchecked Sendable {
    static let shared = PythonRuntime()

    let numpy: PythonObject
    let envpool: PythonObject

    private init() {
        do {
            let sys = try Python.attemptImport("sys")
            let currentDir = ProcessInfo.processInfo.environment["OAK_ROOT_DIR"] ?? FileManager.default.currentDirectoryPath
            sys.path.append("\(currentDir)/scripts")
            sys.path.append("/opt/homebrew/lib/python3.11/site-packages")
            numpy = try Python.attemptImport("numpy")
            envpool = try Python.attemptImport("envpool")
            _ = try Python.attemptImport("envpool.mujoco.gym.registration")
            _ = try Python.attemptImport("envpool.minigrid.registration")
        } catch {
            fatalError("Python runtime unavailable. Run ./setup.sh. Underlying error: \(error)")
        }
    }
}

func pythonFlatArray(_ object: PythonObject) -> [Float] {
    Array(PythonRuntime.shared.numpy.asarray(object, dtype: PythonRuntime.shared.numpy.float32).flatten()).map {
        let scalar = Float($0) ?? 0
        return scalar.isFinite ? scalar : 0
    }
}

func json(_ pairs: (String, Any?)...) -> JSON {
    var out: JSON = [:]
    out.reserveCapacity(pairs.count)
    for (key, value) in pairs {
        if let value { out[key] = value }
    }
    return out
}

func jsonReady(_ value: Any) -> Any {
    switch value {
    case let number as Double:
        return number.isFinite ? number : NSNull()
    case let number as Float:
        return number.isFinite ? Double(number) : NSNull()
    case let dict as JSON:
        var out: JSON = [:]
        out.reserveCapacity(dict.count)
        for (key, child) in dict {
            out[key] = jsonReady(child)
        }
        return out
    case let array as [Any]:
        return array.map(jsonReady)
    default:
        return value
    }
}

private func mixedSeed(_ seed: Int?, salt: UInt64, fallback: UInt64) -> UInt64 {
    guard let seed else { return fallback }
    var value = UInt64(bitPattern: Int64(seed)) &+ salt &+ 0x9e3779b97f4a7c15
    value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
    value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
    let mixed = value ^ (value >> 31)
    return mixed == 0 ? fallback : mixed
}

@main struct Main {
    static func main() {
        run()
    }

    private static func run() {
        let cliArguments = Array(CommandLine.arguments.dropFirst())
        print("OaK - Alberta Plan")
        guard let envId = ProcessInfo.processInfo.environment["OAK_ENV"] ?? cliArguments.first,
              !envId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            fputs("Missing environment id. Pass a concrete envpool task id as the first argument or set OAK_ENV.\n", stderr)
            Foundation.exit(1)
        }
        let requestedPolicy = (ProcessInfo.processInfo.environment["OAK_POLICY"] ?? "oak").lowercased()
        guard ["oak", "random"].contains(requestedPolicy) else {
            fputs("Unknown policy '\(requestedPolicy)'. Choose one of: oak, random\n", stderr)
            Foundation.exit(1)
        }

        let numEnvs = ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init) ?? 1
        let runSeed = ProcessInfo.processInfo.environment["OAK_SEED"].flatMap(Int.init)
        let warmupSteps = max(ProcessInfo.processInfo.environment["OAK_WARMUP_STEPS"].flatMap(Int.init) ?? 0, 0)
        let stopAfterMeasuredSteps = max(ProcessInfo.processInfo.environment["OAK_STOP_AFTER_STEPS"].flatMap(Int.init) ?? numEnvs, numEnvs)
        let env = EnvPoolEnv(id: envId, num_envs: numEnvs, seed: runSeed)
        let randomRNG = requestedPolicy == "random"
            ? OakRNG(seed: mixedSeed(runSeed, salt: 0x3001, fallback: 0xa0761d6478bd642f))
            : nil
        let oakDiscreteRNG = requestedPolicy == "oak" && env.actionDiscreteCount != nil
            ? OakRNG(seed: mixedSeed(runSeed, salt: 0x0a6f, fallback: 0xe7037ed1a0b428db))
            : nil
        print("\n\(envId) (\(type(of: env))) [policy=\(requestedPolicy)\(runSeed.map { " seed=\($0)" } ?? "")]")
        var totalEnvironmentSteps = 0
        var measuredSteps = 0
        var measuredSeconds = 0.0
        var measurementStartMs: Double?
        var stopReason = "completed"
        var observedBudgetCount: UInt64 = 0
        var cumulativeObservedBudgetNs = 0.0
        var agentStepStats = (count: 0, sum: 0.0)
        var envStepStats = (count: 0, sum: 0.0)
        var totalStepStats = (count: 0, sum: 0.0)
        var rewardSum = 0.0
        var doneCount = 0
        var vectorUpdateCount = 0
        var lastSummaryVectorUpdate = 0
        var lastHealth: JSON?
        let metricsPath = ProcessInfo.processInfo.environment["OAK_METRICS_JSONL_PATH"].map(URL.init(fileURLWithPath:))
        let measuredTarget = max(stopAfterMeasuredSteps, numEnvs)
        let summaryWindowSize = max(1, Int(sqrt(Double(max(measuredTarget / max(numEnvs, 1), 1)))))
        let oakEngine: OakMetalEngine? = {
            guard requestedPolicy == "oak" else { return nil }
            let resolvedDevice = MTLCreateSystemDefaultDevice()
                ?? MTLCopyAllDevices().first(where: { !$0.isHeadless })
                ?? MTLCopyAllDevices().first
            guard let device = resolvedDevice else {
                fatalError("Metal device unavailable. OaK is a Metal-only runtime.")
            }
            return OakMetalEngine(obsDim: env.obsDim, actDim: env.actDim, device: device, seed: runSeed)
        }()
        let runId = UUID().uuidString
        var metricsHandle: FileHandle?
        func observe(_ stat: inout (count: Int, sum: Double), _ value: Double) {
            guard value.isFinite else { return }
            stat.count += 1
            stat.sum += value
        }
        func mean(_ stat: (count: Int, sum: Double)) -> Double? {
            stat.count > 0 ? stat.sum / Double(stat.count) : nil
        }
        func writeRecord(_ record: JSON) throws {
            guard let metricsHandle else { return }
            metricsHandle.write(try JSONSerialization.data(withJSONObject: jsonReady(record), options: [.sortedKeys]))
            metricsHandle.write(Data([0x0a]))
        }
        func makeRecord(type: String, warmup: Int? = nil, stopAfter: Int? = nil, measuredSeconds: Double? = nil, stopReason: String? = nil, health: JSON? = nil) -> JSON {
            let isStart = type == "run_start"
            return json(
                ("type", type),
                ("runId", runId),
                ("envId", envId),
                ("policy", requestedPolicy),
                ("seed", runSeed),
                ("numEnvs", numEnvs),
                ("warmupSteps", warmup),
                ("stopAfterMeasuredSteps", stopAfter),
                ("summaryWindowSize", summaryWindowSize),
                ("observationDim", env.obsDim),
                ("actionDim", env.actDim),
                ("actionDiscreteCount", env.actionDiscreteCount),
                ("rewardSum", isStart ? nil : rewardSum),
                ("doneCount", isStart ? nil : doneCount),
                ("measuredStepCount", isStart ? nil : measuredSteps),
                ("vectorUpdateCount", isStart ? nil : vectorUpdateCount),
                ("measuredSeconds", isStart ? nil : measuredSeconds),
                ("agentStepMsMean", isStart ? nil : mean(agentStepStats)),
                ("envStepMsMean", isStart ? nil : mean(envStepStats)),
                ("totalStepMsMean", isStart ? nil : mean(totalStepStats)),
                ("stopReason", stopReason),
                ("health", health)
            )
        }
        func discreteBehaviorAction(_ scores: [Float], rng: OakRNG?) -> [Float] {
            guard let count = env.actionDiscreteCount, count > 0 else { return scores }
            var oneHot = Array(repeating: Float(0), count: scores.count)
            for envIndex in 0..<numEnvs {
                let start = envIndex * count
                var bestIndex = 0
                var bestValue = -Double.greatestFiniteMagnitude
                for actionIndex in 0..<count {
                    let score = start + actionIndex < scores.count ? Double(scores[start + actionIndex]) : 0
                    let value = score + (rng?.nextGumbel() ?? 0)
                    if value > bestValue {
                        bestValue = value
                        bestIndex = actionIndex
                    }
                }
                if start + bestIndex < oneHot.count {
                    oneHot[start + bestIndex] = 1
                }
            }
            return oneHot
        }
        if let fileURL = metricsPath {
            do {
                FileManager.default.createFile(atPath: fileURL.path, contents: nil)
                metricsHandle = try FileHandle(forWritingTo: fileURL)
                try writeRecord(makeRecord(type: "run_start", warmup: warmupSteps, stopAfter: stopAfterMeasuredSteps))
            } catch {
                fatalError("Failed to initialize metrics: \(error)")
            }
        }
        defer { try? metricsHandle?.close() }
        var obs = env.reset()
        var pendingRewards = Array(repeating: Float(0), count: numEnvs)
        var pendingDone = Array(repeating: false, count: numEnvs)
        var lastAction = Array(repeating: Float(0), count: env.actDim * numEnvs)
        defer {
        let finalRecord = makeRecord(type: "run_end", measuredSeconds: measuredSeconds, stopReason: stopReason, health: lastHealth)
            do {
                try writeRecord(finalRecord)
            } catch {
                fatalError("Failed to write metrics run_end record: \(error)")
            }
        }
        while true {
            let nextEnvironmentStep = totalEnvironmentSteps + numEnvs
            let warmupSatisfied = nextEnvironmentStep > warmupSteps
            let currentBudgetNs = observedBudgetCount > 0
                ? UInt64((cumulativeObservedBudgetNs / Double(observedBudgetCount)).rounded(.down))
                : 0
            let isMeasuredStep = warmupSatisfied && observedBudgetCount > 0
            let totalStepStart = CFAbsoluteTimeGetCurrent() * 1000
            if isMeasuredStep && measurementStartMs == nil {
                measurementStartMs = totalStepStart
                if warmupSteps > 0 {
                    print("[metrics] warmup_complete env=\(envId) policy=\(requestedPolicy) warmup_steps=\(warmupSteps) budget_bootstrapped=1")
                }
            }
            let agentAction: [Float]
            let agentTelemetry: (agentStepMs: Double, health: JSON?)
            if let oakEngine {
                let output = oakEngine.step(
                    obs: obs,
                    prevAction: lastAction,
                    prevReward: pendingRewards,
                    prevDone: pendingDone,
                    batchSize: numEnvs,
                    wallClockBudgetNs: currentBudgetNs
                )
                let behaviorAction = discreteBehaviorAction(output.action, rng: oakDiscreteRNG)
                lastAction = behaviorAction
                agentAction = behaviorAction
                agentTelemetry = output.telemetry
            } else {
                let stepStart = CFAbsoluteTimeGetCurrent() * 1000
                let action: [Float]
                if let count = env.actionDiscreteCount, count > 0 {
                    action = (0..<numEnvs).flatMap { _ -> [Float] in
                        let selected = randomRNG?.nextInt(upperBound: count) ?? Int.random(in: 0..<count)
                        return (0..<count).map { $0 == selected ? 1 : 0 }
                    }
                } else {
                    action = (0..<(env.actDim * numEnvs)).map { index in
                        let dim = env.actDim > 0 ? index % env.actDim : 0
                        let low = dim < env.actionLow.count ? env.actionLow[dim] : -1
                        let high = dim < env.actionHigh.count ? env.actionHigh[dim] : 1
                        return randomRNG?.nextFloat(in: low...high) ?? Float.random(in: low...high)
                    }
                }
                lastAction = action
                agentAction = action
                agentTelemetry = (max((CFAbsoluteTimeGetCurrent() * 1000) - stepStart, 0), nil)
            }
            let envStepStart = CFAbsoluteTimeGetCurrent() * 1000
            let result = env.step(agentAction)
            let envStepMs = (CFAbsoluteTimeGetCurrent() * 1000) - envStepStart
            let totalStepMs = (CFAbsoluteTimeGetCurrent() * 1000) - totalStepStart
            cumulativeObservedBudgetNs += max(totalStepMs, 0) * 1_000_000
            observedBudgetCount += 1
            totalEnvironmentSteps = nextEnvironmentStep
            var shouldStop = false
            if isMeasuredStep {
                measuredSteps += numEnvs
                if let measurementStartMs {
                    measuredSeconds = max(((CFAbsoluteTimeGetCurrent() * 1000) - measurementStartMs) / 1000, 0)
                }
                do {
                    vectorUpdateCount += 1
                    let stepDoneCount = result.2.reduce(into: 0) { if $1 { $0 += 1 } }
                    doneCount += stepDoneCount
                    for reward in result.1 {
                        let value = reward.isFinite ? Double(reward) : 0
                        rewardSum += value
                    }
                    observe(&agentStepStats, agentTelemetry.agentStepMs)
                    observe(&envStepStats, envStepMs)
                    observe(&totalStepStats, totalStepMs)
                    lastHealth = agentTelemetry.health
                    if vectorUpdateCount - lastSummaryVectorUpdate >= summaryWindowSize {
                        lastSummaryVectorUpdate = vectorUpdateCount
                        let summaryRecord = makeRecord(type: "summary", health: agentTelemetry.health)
                        try writeRecord(summaryRecord)
                    }
                } catch {
                    fatalError("Failed to write metrics record: \(error)")
                }
                if measuredSteps >= stopAfterMeasuredSteps {
                    stopReason = "measured_steps"
                    shouldStop = true
                }
            }
            obs = result.0
            pendingRewards = result.1
            pendingDone = result.2
            if shouldStop {
                print("Stopping after \(measuredSteps) measured steps (\(totalEnvironmentSteps) total env steps), reason=\(stopReason)")
                break
            }
        }
    }
}

private func pythonSpaceInfo(_ space: PythonObject, discreteAsCount: Bool) -> (dim: Int, discreteCount: Int?, keys: [String]) {
    if Bool(Python.hasattr(space, "spaces")) == true {
        let pyKeys = Python.list(space.spaces.keys())
        let keyCount = Int(pyKeys.__len__()) ?? 0
        let keys = (0..<keyCount).map { String(pyKeys[$0]) ?? "\($0)" }.sorted()
        let dim = keys.reduce(0) { partial, key in
            partial + pythonSpaceInfo(space.spaces[PythonObject(key)], discreteAsCount: false).dim
        }
        return (max(dim, 1), nil, keys)
    }
    if Bool(Python.hasattr(space, "n")) == true, let count = Int(space.n) {
        return (discreteAsCount ? max(count, 1) : 1, count, [])
    }
    if Bool(Python.hasattr(space, "shape")) == true, let shape = [Int](space.shape) {
        let dim = max(shape.reduce(1, *), 1)
        return (dim, nil, [])
    }
    return (1, nil, [])
}

public final class EnvPoolEnv {
    private let env: PythonObject
    private let observationKeys: [String]
    public let numEnvs: Int
    public let obsDim: Int
    public let actDim: Int
    public let actionDiscreteCount: Int?
    public let actionLow: [Float]
    public let actionHigh: [Float]
    
    public init(id: String, num_envs: Int, seed: Int? = nil) {
        let runtime = PythonRuntime.shared
        let envpool = runtime.envpool
        
        numEnvs = num_envs
        
        if let seed = seed {
            env = envpool.make(id, "gymnasium", num_envs: num_envs, seed: seed)
        } else {
            env = envpool.make(id, "gymnasium", num_envs: num_envs)
        }
        
        let obsSpace = self.env.observation_space
        let actSpace = self.env.action_space
        let obsInfo = pythonSpaceInfo(obsSpace, discreteAsCount: false)
        let actInfo = pythonSpaceInfo(actSpace, discreteAsCount: true)
        observationKeys = obsInfo.keys
        obsDim = obsInfo.dim
        actDim = actInfo.dim
        actionDiscreteCount = actInfo.discreteCount
        if actInfo.discreteCount == nil, Bool(Python.hasattr(actSpace, "low")) == true, Bool(Python.hasattr(actSpace, "high")) == true {
            actionLow = pythonFlatArray(runtime.numpy.asarray(actSpace.low, dtype: runtime.numpy.float32))
            actionHigh = pythonFlatArray(runtime.numpy.asarray(actSpace.high, dtype: runtime.numpy.float32))
        } else {
            actionLow = []
            actionHigh = []
        }
    }

    public func reset() -> [Float] {
        let result = env.reset()
        return flattenObservationBatch(result[0])
    }
    
    public func step(_ action: [Float]) -> (obs: [Float], reward: [Float], done: [Bool]) {
        let np = PythonRuntime.shared.numpy
        
        let npAction: PythonObject
        if actionDiscreteCount != nil {
            let count = actionDiscreteCount ?? actDim
            let intActions = count > 0 ? (0..<numEnvs).map { envIndex -> Int32 in
                let start = envIndex * count
                var bestIndex = 0
                var bestValue = -Float.greatestFiniteMagnitude
                for actionIndex in 0..<count {
                    let raw = start + actionIndex < action.count ? action[start + actionIndex] : 0
                    let value = raw.isFinite ? raw : 0
                    if value > bestValue {
                        bestValue = value
                        bestIndex = actionIndex
                    }
                }
                return Int32(bestIndex)
            } : []
            npAction = np.array(intActions, dtype: np.int32).reshape(numEnvs)
        } else {
            npAction = np.array(action, dtype: np.float32).reshape(numEnvs, actDim)
        }
        
        let result = env.step(npAction)
        let pyObs = result[0]
        let pyReward = result[1]
        let pyTerminated = result[2]
        let pyTruncated = result[3]
        
        let pyDone = np.logical_or(pyTerminated, pyTruncated)
        
        let obs = flattenObservationBatch(pyObs)
        let reward = pythonFlatArray(np.asarray(pyReward, dtype: np.float32))
        let done = pythonFlatArray(np.asarray(pyDone, dtype: np.float32)).map { $0 != 0 }
        
        return (obs, reward, done)
    }

    private func flattenObservationBatch(_ pyObs: PythonObject) -> [Float] {
        if Bool(Python.hasattr(pyObs, "keys")) == true {
            var flat = Array(repeating: Float(0), count: numEnvs * obsDim)
            var offset = 0
            for key in observationKeys {
                let child = flattenObservationBatch(pyObs[PythonObject(key)])
                precondition(child.count % max(numEnvs, 1) == 0, "Observation key \(key) flattened to \(child.count) floats for \(numEnvs) envs")
                let childWidth = child.count / max(numEnvs, 1)
                for envIndex in 0..<numEnvs {
                    let source = envIndex * childWidth
                    let destination = envIndex * obsDim + offset
                    for index in 0..<childWidth {
                        flat[destination + index] = child[source + index]
                    }
                }
                offset += childWidth
            }
            precondition(offset == obsDim, "Observation dict flattened to \(offset) cols, expected \(obsDim)")
            return flat
        }

        let np = PythonRuntime.shared.numpy
        let array = np.asarray(pyObs, dtype: np.float32)
        guard let shape = [Int](array.shape), !shape.isEmpty else {
            let scalar = pythonFlatArray(array)
            return Array(repeating: scalar.first ?? 0, count: numEnvs)
        }

        let flat = pythonFlatArray(array)
        let rowWidth = shape.dropFirst().reduce(1, *) == 0 ? 1 : max(shape.dropFirst().reduce(1, *), 1)
        precondition(shape[0] == numEnvs, "Observation batch width \(shape[0]) did not match numEnvs \(numEnvs)")
        precondition(flat.count == numEnvs * rowWidth, "Observation flattened to \(flat.count) floats, expected \(numEnvs * rowWidth)")
        return flat
    }
}

private struct KernelConfig {
    var obsDim: UInt32, rank: UInt32, actDim: UInt32, batchSize: UInt32
    var offeredNs: Float
}

private struct KernelScalars {
    var activeRank: UInt32 = 1, rewardCount: UInt32 = 0, reserved0: UInt32 = 0, reserved1: UInt32 = 0
    var rewardRate: Float = 0
}

private struct KernelStats {
    var residualMean: Float = 0, totalMean: Float = 0, rewardRate: Float = 0
    var tdAbsMean: Float = 0, actionDeltaL2: Float = 0, gvfErrorMean: Float = 0, gvfPredictionAbsMean: Float = 0
    var modelErrorMean: Float = 0, rewardPredictionAbsMean: Float = 0, statePredictionAbsMean: Float = 0
    var optionSelectionFraction: Float = 0, optionTrainingSignalMean: Float = 0, optionScoreMean: Float = 0
    var planningCandidateCount: Float = 0, planningScoreMean: Float = 0, planningScoreMax: Float = 0, planningActionChangeL2: Float = 0
    var rawNonFiniteCount: Float = 0, sanitizedNonFiniteCount: Float = 0, fatalNonFiniteCount: Float = 0
}

private final class OakRNG {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0x9e3779b97f4a7c15 : seed
    }

    func nextUInt64() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return value ^ (value >> 31)
    }

    func nextSignedFloat(scale: Float) -> Float {
        let unit = nextUnitDouble()
        return Float(unit * 2 - 1) * scale
    }
    
    func nextFloat(in range: ClosedRange<Float>) -> Float {
        let unit = nextUnitDouble()
        let lower = Double(range.lowerBound)
        let upper = Double(range.upperBound)
        return Float(lower + (upper - lower) * unit)
    }

    func nextInt(upperBound: Int) -> Int {
        guard upperBound > 1 else { return 0 }
        return Int(nextUInt64() % UInt64(upperBound))
    }

    func nextGumbel() -> Double {
        let unit = min(max(nextUnitDouble(), Double.leastNonzeroMagnitude), 1.0 - Double.leastNonzeroMagnitude)
        return -log(-log(unit))
    }

    private func nextUnitDouble() -> Double {
        Double(nextUInt64() >> 11) / Double(1 << 53)
    }
}

private final class SharedFloats {
    private(set) var count: Int
    private(set) var buffer: MTLBuffer
    private(set) var pointer: UnsafeMutablePointer<Float>

    init(device: MTLDevice, count: Int) {
        self.count = max(count, 1)
        buffer = device.makeBuffer(length: self.count * MemoryLayout<Float>.stride, options: [.storageModeShared])!
        pointer = buffer.contents().bindMemory(to: Float.self, capacity: self.count)
        pointer.initialize(repeating: 0, count: self.count)
    }

    func ensureCapacity(device: MTLDevice, count newCount: Int) {
        guard newCount > count else { return }
        let replacement = device.makeBuffer(length: newCount * MemoryLayout<Float>.stride, options: [.storageModeShared])!
        let replacementPointer = replacement.contents().bindMemory(to: Float.self, capacity: newCount)
        replacementPointer.initialize(repeating: 0, count: newCount)
        replacementPointer.update(from: pointer, count: count)
        count = newCount
        buffer = replacement
        pointer = replacementPointer
    }

    func copy(from values: [Float], count copyCount: Int) {
        let safeCount = min(copyCount, min(values.count, count))
        values.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else { return }
            pointer.update(from: base, count: safeCount)
        }
        if safeCount < count {
            memset(pointer.advanced(by: safeCount), 0, (count - safeCount) * MemoryLayout<Float>.stride)
        }
    }

    func array(count copyCount: Int) -> [Float] {
        let safeCount = min(copyCount, count)
        return Array(UnsafeBufferPointer(start: pointer, count: safeCount))
    }
}

final class OakMetalEngine {
    private let obsDim: Int
    private let actDim: Int
    private let registeredRank: Int
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let kernelConfigBuffer: MTLBuffer
    private let scalarBuffer: MTLBuffer
    private let statsBuffer: MTLBuffer
    private let scalarPointer: UnsafeMutablePointer<KernelScalars>
    private let statsPointer: UnsafeMutablePointer<KernelStats>

    private let encoder: SharedFloats
    private let policy: SharedFloats
    private let valueWeights: SharedFloats
    private let gvfWeights: SharedFloats
    private let rewardModelWeights: SharedFloats
    private let stateModelWeights: SharedFloats
    private let optionPolicyWeights: SharedFloats
    private let optionValueWeights: SharedFloats

    private var observationBuffer: SharedFloats
    private var rewardBuffer: SharedFloats
    private var doneBuffer: SharedFloats
    private var stateBuffer: SharedFloats
    private var actionBuffer: SharedFloats
    private var previousStateBuffer: SharedFloats
    private var previousActionBuffer: SharedFloats

    private var cumulativeResponseNs = 0.0
    private var observedResponseCount = 0

    init(obsDim: Int, actDim: Int, device: MTLDevice, seed: Int?) {
        self.obsDim = obsDim
        self.actDim = actDim
        self.device = device
        registeredRank = max(1, min(obsDim, max(actDim, 1)))
        queue = device.makeCommandQueue()!
        let root = ProcessInfo.processInfo.environment["OAK_ROOT_DIR"] ?? FileManager.default.currentDirectoryPath
        let kernelPath = URL(fileURLWithPath: root).appendingPathComponent("Sources/OakKernel.metal")
        let kernelSource = try! String(contentsOf: kernelPath, encoding: .utf8)
        let library = try! device.makeLibrary(source: kernelSource, options: nil)
        let function = library.makeFunction(name: "oak_step")!
        pipeline = try! device.makeComputePipelineState(function: function)
        kernelConfigBuffer = device.makeBuffer(length: MemoryLayout<KernelConfig>.stride, options: [.storageModeShared])!
        scalarBuffer = device.makeBuffer(length: MemoryLayout<KernelScalars>.stride, options: [.storageModeShared])!
        statsBuffer = device.makeBuffer(length: MemoryLayout<KernelStats>.stride, options: [.storageModeShared])!
        scalarPointer = scalarBuffer.contents().bindMemory(to: KernelScalars.self, capacity: 1)
        statsPointer = statsBuffer.contents().bindMemory(to: KernelStats.self, capacity: 1)
        scalarPointer.initialize(to: KernelScalars())
        statsPointer.initialize(to: KernelStats())
        scalarPointer.pointee.activeRank = 1

        let batchCapacity = 1
        encoder = SharedFloats(device: device, count: registeredRank * obsDim)
        policy = SharedFloats(device: device, count: actDim * registeredRank)
        valueWeights = SharedFloats(device: device, count: registeredRank)
        gvfWeights = SharedFloats(device: device, count: (registeredRank + 1) * registeredRank)
        rewardModelWeights = SharedFloats(device: device, count: registeredRank + actDim)
        stateModelWeights = SharedFloats(device: device, count: registeredRank * (registeredRank + actDim))
        optionPolicyWeights = SharedFloats(device: device, count: registeredRank * actDim)
        optionValueWeights = SharedFloats(device: device, count: registeredRank * registeredRank)

        observationBuffer = SharedFloats(device: device, count: batchCapacity * obsDim)
        rewardBuffer = SharedFloats(device: device, count: batchCapacity)
        doneBuffer = SharedFloats(device: device, count: batchCapacity)
        stateBuffer = SharedFloats(device: device, count: batchCapacity * registeredRank)
        actionBuffer = SharedFloats(device: device, count: batchCapacity * actDim)
        previousStateBuffer = SharedFloats(device: device, count: batchCapacity * registeredRank)
        previousActionBuffer = SharedFloats(device: device, count: batchCapacity * actDim)

        let rng = OakRNG(seed: mixedSeed(seed, salt: 0x5eed, fallback: 0x9e3779b97f4a7c15))
        let encoderScale = Float(1.0 / sqrt(Double(max(obsDim, 1))))
        let rankScale = Float(1.0 / sqrt(Double(max(registeredRank, 1))))
        let transitionScale = Float(1.0 / sqrt(Double(max(registeredRank + actDim, 1))))
        let optionScale = Float(1.0 / sqrt(Double(max(actDim, 1))))
        for index in 0..<encoder.count { encoder.pointer[index] = rng.nextSignedFloat(scale: encoderScale) }
        for index in 0..<policy.count { policy.pointer[index] = rng.nextSignedFloat(scale: rankScale) }
        for index in 0..<valueWeights.count { valueWeights.pointer[index] = rng.nextSignedFloat(scale: rankScale) }
        for index in 0..<gvfWeights.count { gvfWeights.pointer[index] = rng.nextSignedFloat(scale: rankScale) }
        for index in 0..<rewardModelWeights.count { rewardModelWeights.pointer[index] = rng.nextSignedFloat(scale: transitionScale) }
        for index in 0..<stateModelWeights.count { stateModelWeights.pointer[index] = rng.nextSignedFloat(scale: transitionScale) }
        for index in 0..<optionPolicyWeights.count { optionPolicyWeights.pointer[index] = rng.nextSignedFloat(scale: optionScale) }
        for index in 0..<optionValueWeights.count { optionValueWeights.pointer[index] = rng.nextSignedFloat(scale: rankScale) }
    }

    func step(
        obs: [Float],
        prevAction: [Float],
        prevReward: [Float],
        prevDone: [Bool],
        batchSize: Int,
        wallClockBudgetNs: UInt64
    ) -> (action: [Float], telemetry: (agentStepMs: Double, health: JSON?)) {
        let stepStartNs = DispatchTime.now().uptimeNanoseconds
        ensureBatchCapacity(batchSize)
        observationBuffer.copy(from: obs, count: batchSize * obsDim)
        rewardBuffer.copy(from: prevReward, count: batchSize)
        doneBuffer.copy(from: prevDone.map { $0 ? 1 : 0 }, count: batchSize)
        previousActionBuffer.copy(from: prevAction, count: batchSize * actDim)

        let responseTimeNs = observedResponseCount > 0
            ? UInt64((cumulativeResponseNs / Double(observedResponseCount)).rounded(.down))
            : wallClockBudgetNs
        let allocation = (
            responseTimeNs: responseTimeNs,
            offeredNs: responseTimeNs > 0 ? responseTimeNs : UInt64(max(batchSize * max(registeredRank + actDim, 1), 1))
        )
        let kernelStartNs = DispatchTime.now().uptimeNanoseconds
        let kernelStats = runKernel(batchSize: batchSize, offeredNs: allocation.offeredNs)
        let consumedNs = DispatchTime.now().uptimeNanoseconds - kernelStartNs

        let activeRank = max(Int(scalarPointer.pointee.activeRank), 1)
        let action = actionBuffer.array(count: batchSize * actDim)
        let health = json(
            ("compute", json(
                ("responseTimeNs", Double(allocation.responseTimeNs)),
                ("offeredNs", Double(allocation.offeredNs)),
                ("consumedNs", Double(consumedNs))
            )),
            ("perception", json(
                ("activeRank", Double(activeRank)),
                ("residualEnergy", Double(kernelStats.residualMean)),
                ("totalEnergy", Double(kernelStats.totalMean))
            )),
            ("control", json(
                ("rewardRate", Double(kernelStats.rewardRate)),
                ("tdAbsMean", Double(kernelStats.tdAbsMean)),
                ("actionDeltaL2", Double(kernelStats.actionDeltaL2))
            )),
            ("gvfBank", json(
                ("errorMean", Double(kernelStats.gvfErrorMean)),
                ("predictionAbsMean", Double(kernelStats.gvfPredictionAbsMean))
            )),
            ("worldModel", json(
                ("errorMean", Double(kernelStats.modelErrorMean)),
                ("rewardPredictionAbsMean", Double(kernelStats.rewardPredictionAbsMean)),
                ("statePredictionAbsMean", Double(kernelStats.statePredictionAbsMean))
            )),
            ("optionBank", json(
                ("selectionFraction", Double(kernelStats.optionSelectionFraction)),
                ("trainingSignalMean", Double(kernelStats.optionTrainingSignalMean)),
                ("scoreMean", Double(kernelStats.optionScoreMean))
            )),
            ("planning", json(
                ("candidateCount", Double(kernelStats.planningCandidateCount)),
                ("scoreMean", Double(kernelStats.planningScoreMean)),
                ("scoreMax", Double(kernelStats.planningScoreMax)),
                ("actionChangeL2", Double(kernelStats.planningActionChangeL2))
            )),
            ("nonFinite", json(
                ("raw", Int(kernelStats.rawNonFiniteCount)),
                ("sanitized", Int(kernelStats.sanitizedNonFiniteCount)),
                ("fatal", Int(kernelStats.fatalNonFiniteCount))
            ))
        )

        let stepNs = DispatchTime.now().uptimeNanoseconds - stepStartNs
        cumulativeResponseNs += Double(stepNs)
        observedResponseCount += 1

        return (action, (Double(stepNs) / 1_000_000, health))
    }

    private func ensureBatchCapacity(_ batchSize: Int) {
        observationBuffer.ensureCapacity(device: device, count: batchSize * obsDim)
        rewardBuffer.ensureCapacity(device: device, count: batchSize)
        doneBuffer.ensureCapacity(device: device, count: batchSize)
        stateBuffer.ensureCapacity(device: device, count: batchSize * registeredRank)
        actionBuffer.ensureCapacity(device: device, count: batchSize * actDim)
        previousStateBuffer.ensureCapacity(device: device, count: batchSize * registeredRank)
        previousActionBuffer.ensureCapacity(device: device, count: batchSize * actDim)
    }

    private func runKernel(batchSize: Int, offeredNs: UInt64) -> KernelStats {
        var config = KernelConfig(
            obsDim: UInt32(obsDim),
            rank: UInt32(registeredRank),
            actDim: UInt32(actDim),
            batchSize: UInt32(batchSize),
            offeredNs: Float(offeredNs)
        )
        memcpy(kernelConfigBuffer.contents(), &config, MemoryLayout<KernelConfig>.stride)
        statsPointer.pointee = KernelStats()
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create Metal command encoder.")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(observationBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(self.encoder.buffer, offset: 0, index: 1)
        encoder.setBuffer(policy.buffer, offset: 0, index: 2)
        encoder.setBuffer(valueWeights.buffer, offset: 0, index: 3)
        encoder.setBuffer(gvfWeights.buffer, offset: 0, index: 4)
        encoder.setBuffer(rewardModelWeights.buffer, offset: 0, index: 5)
        encoder.setBuffer(stateModelWeights.buffer, offset: 0, index: 6)
        encoder.setBuffer(optionPolicyWeights.buffer, offset: 0, index: 7)
        encoder.setBuffer(optionValueWeights.buffer, offset: 0, index: 8)
        encoder.setBuffer(previousStateBuffer.buffer, offset: 0, index: 9)
        encoder.setBuffer(previousActionBuffer.buffer, offset: 0, index: 10)
        encoder.setBuffer(rewardBuffer.buffer, offset: 0, index: 11)
        encoder.setBuffer(doneBuffer.buffer, offset: 0, index: 12)
        encoder.setBuffer(stateBuffer.buffer, offset: 0, index: 13)
        encoder.setBuffer(actionBuffer.buffer, offset: 0, index: 14)
        encoder.setBuffer(scalarBuffer, offset: 0, index: 15)
        encoder.setBuffer(statsBuffer, offset: 0, index: 16)
        encoder.setBuffer(kernelConfigBuffer, offset: 0, index: 17)

        let threads = MTLSize(width: 1, height: 1, depth: 1)
        let groups = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: groups)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            fatalError("Metal command failed: \(error)")
        }
        return statsPointer.pointee
    }
}
