import Darwin
import Foundation
import Metal
import PythonKit

final class PythonRuntime: @unchecked Sendable {
    static let shared = PythonRuntime()

    private let lock = NSLock()
    private var configured = false
    private var moduleCache: [String: PythonObject] = [:]

    private init() {}

    var sys: PythonObject {
        module(named: "sys")
    }

    var gym: PythonObject {
        module(named: "gymnasium")
    }

    var numpy: PythonObject {
        module(named: "numpy")
    }

    var random: PythonObject {
        module(named: "random")
    }

    func module(named name: String) -> PythonObject {
        lock.lock()
        defer { lock.unlock() }

        configureIfNeededLocked()
        if let cached = moduleCache[name] {
            return cached
        }

        do {
            let imported = try Python.attemptImport(name)
            moduleCache[name] = imported
            return imported
        } catch {
            fatalError("Missing Python module '\(name)'. Run ./setup.sh. Underlying error: \(error)")
        }
    }

    func hasModule(named name: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        configureIfNeededLocked()
        if moduleCache[name] != nil {
            return true
        }

        do {
            let imported = try Python.attemptImport(name)
            moduleCache[name] = imported
            return true
        } catch {
            return false
        }
    }

    func require(modules names: [String]) {
        for name in names {
            _ = module(named: name)
        }
    }

    private func configureIfNeededLocked() {
        guard !configured else {
            return
        }

        do {
            let importedSys = try Python.attemptImport("sys")
            moduleCache["sys"] = importedSys
            let currentDir = ProcessInfo.processInfo.environment["OAK_ROOT_DIR"] ?? FileManager.default.currentDirectoryPath
            importedSys.path.append("\(currentDir)/scripts")
            importedSys.path.append("/opt/homebrew/lib/python3.11/site-packages")

            configured = true
        } catch {
            fatalError("Python runtime unavailable. Run ./setup.sh. Underlying error: \(error)")
        }
    }
}

private func currentTimeMilliseconds() -> Double {
    CFAbsoluteTimeGetCurrent() * 1000
}

private func sanitizeScalar(_ value: Float, clamp range: ClosedRange<Float>? = nil) -> Float {
    let finiteValue = value.isFinite ? value : 0
    guard let range else {
        return finiteValue
    }
    return min(max(finiteValue, range.lowerBound), range.upperBound)
}

private func sanitizeVector(_ values: [Float], count: Int? = nil, clamp range: ClosedRange<Float>? = nil) -> [Float] {
    let targetCount = count ?? values.count
    return (0..<targetCount).map { index in
        let value = index < values.count ? values[index] : 0
        return sanitizeScalar(value, clamp: range)
    }
}

private func clampAction(_ action: [Float], low: [Float], high: [Float]) -> [Float] {
    let count = min(low.count, high.count)
    return (0..<count).map { index in
        let value = index < action.count ? action[index] : 0
        return min(max(sanitizeScalar(value), low[index]), high[index])
    }
}

func pythonFlatArray(_ object: PythonObject) -> [Float] {
    let np = PythonRuntime.shared.numpy
    let flattened = np.asarray(object, dtype: np.float32).flatten()
    return Array(flattened).map { sanitizeScalar(Float($0) ?? 0) }
}

private func zeroBuffer(_ buffer: MTLBuffer) {
    memset(buffer.contents(), 0, buffer.length)
}

private func configuredRunSeed(environment: [String: String] = ProcessInfo.processInfo.environment) -> Int? {
    guard let rawSeed = environment["OAK_SEED"], !rawSeed.isEmpty else {
        return nil
    }
    guard let seed = Int(rawSeed) else {
        fatalError("Invalid OAK_SEED '\(rawSeed)'. Expected an integer.")
    }
    return seed
}

private func configuredStepBudgetNs(environment: [String: String] = ProcessInfo.processInfo.environment) -> UInt64? {
    guard let rawBudget = environment["OAK_STEP_BUDGET_NS"], !rawBudget.isEmpty else {
        return nil
    }
    guard let budget = UInt64(rawBudget), budget > 0 else {
        fatalError("Invalid OAK_STEP_BUDGET_NS '\(rawBudget)'. Expected a positive integer.")
    }
    return budget
}

private func currentProcessArchitecture() -> String {
#if arch(arm64)
    "arm64"
#elseif arch(x86_64)
    "x86_64"
#else
    "unknown"
#endif
}

private func runMetalDoctor(arguments: ArraySlice<String>) -> Int32 {
    let quiet = arguments.contains("--quiet")
    let devices = MTLCopyAllDevices()
    let defaultDevice = MTLCreateSystemDefaultDevice()

    func emit(_ message: String) {
        if !quiet {
            print(message)
        }
    }

    emit("metal_doctor arch=\(currentProcessArchitecture())")
    emit("metal_doctor default_device=\(defaultDevice?.name ?? "nil")")
    emit("metal_doctor device_count=\(devices.count)")
    for (index, device) in devices.enumerated() {
        emit("metal_doctor device[\(index)]=\(device.name) headless=\(device.isHeadless) low_power=\(device.isLowPower) removable=\(device.isRemovable)")
    }

    guard let device = defaultDevice ?? devices.first else {
        fputs("Metal doctor failed: no Metal device available to this process.\n", stderr)
        return 1
    }

    let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void probe(device float *output [[buffer(0)]], uint index [[thread_position_in_grid]]) {
        if (index == 0) {
            output[0] = 42.0f;
        }
    }
    """

    do {
        let library = try device.makeLibrary(source: kernelSource, options: nil)
        guard let function = library.makeFunction(name: "probe") else {
            throw NSError(domain: "MetalDoctor", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to compile probe function"])
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue() else {
            throw NSError(domain: "MetalDoctor", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal command queue"])
        }
        guard let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: [.storageModeShared]) else {
            throw NSError(domain: "MetalDoctor", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate Metal probe buffer"])
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalDoctor", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal command encoder"])
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        let gridSize = MTLSize(width: 1, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let commandError = commandBuffer.error {
            throw commandError
        }

        let result = outputBuffer.contents().assumingMemoryBound(to: Float.self)[0]
        guard abs(result - 42) < 0.001 else {
            throw NSError(domain: "MetalDoctor", code: 5, userInfo: [NSLocalizedDescriptionKey: "Probe kernel returned \(result), expected 42"])
        }

        emit("metal_doctor compute_submission=ok result=\(result)")
        return 0
    } catch {
        fputs("Metal doctor failed: \(error)\n", stderr)
        return 1
    }
}

private func seedPythonGlobals(_ seed: Int) {
    let runtime = PythonRuntime.shared
    runtime.random.seed(seed)
    runtime.numpy.random.seed(seed)
}

private func seedSpace(_ space: PythonObject, seed: Int?) {
    guard let seed else {
        return
    }
    _ = space.seed(seed)
}

private func resetPythonEnv(_ env: PythonObject, seed: Int?) -> PythonObject {
    if let seed {
        return env.reset(seed: seed)
    }
    return env.reset()
}

private func deriveSeed(_ seed: Int?, salt: UInt64) -> UInt64? {
    guard let seed else {
        return nil
    }

    var value = UInt64(bitPattern: Int64(seed)) &+ salt &+ 0x9E3779B97F4A7C15
    value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
    value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
    return value ^ (value >> 31)
}

private final class SeededRNG {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0xA0761D6478BD642F : seed
    }

    func nextUInt64() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }

    func nextFloat(in range: ClosedRange<Float>) -> Float {
        let unit = Double(nextUInt64() >> 11) / Double(1 << 53)
        let lowerBound = Double(range.lowerBound)
        let upperBound = Double(range.upperBound)
        return Float(lowerBound + (upperBound - lowerBound) * unit)
    }
}

private func seededFloat(in range: ClosedRange<Float>, rng: SeededRNG?) -> Float {
    guard let rng else {
        return Float.random(in: range)
    }
    return rng.nextFloat(in: range)
}

private struct ResetSeedSequence {
    private let baseSeed: Int?
    private var resetIndex = 0

    init(baseSeed: Int?) {
        self.baseSeed = baseSeed
    }

    mutating func nextResetSeed() -> Int? {
        guard let baseSeed else {
            return nil
        }

        defer {
            resetIndex += 1
        }
        return baseSeed &+ resetIndex
    }
}

struct Vector {
    var data: [Float]
    init(_ n: Int) {
        data = Array(repeating: 0, count: n)
    }
    init(_ a: [Float]) {
        data = a
    }
    var count: Int {
        data.count
    }
    subscript(i: Int) -> Float {
        get {
            data[i]
        }
        set {
            data[i] = newValue
        }
    }
}

protocol PyEnv {
    var obsDim: Int { get }
    var actDim: Int { get }
    var schema: EnvironmentSchemaFingerprint { get }
    func reset() -> [Float]
    func step(_ action: [Float]) -> ([Float], Float, Bool)
}

final class PyFlytEnv: PyEnv {
    private let env: PythonObject
    private var resetSeedSequence: ResetSeedSequence
    let obsDim: Int
    let actDim: Int
    let schema: EnvironmentSchemaFingerprint

    init() {
        let baseSeed = configuredRunSeed()
        let runtime = PythonRuntime.shared
        runtime.require(modules: ["PyFlyt.gym_envs", "gymnasium", "numpy"])
        env = runtime.gym.make("PyFlyt/QuadX-Hover-v0")
        resetSeedSequence = ResetSeedSequence(baseSeed: baseSeed)
        if let baseSeed {
            seedPythonGlobals(baseSeed)
            seedSpace(env.action_space, seed: baseSeed)
            seedSpace(env.observation_space, seed: baseSeed)
        }
        obsDim = Int(env.observation_space.shape[0])!
        actDim = Int(env.action_space.shape[0])!
        schema = EnvironmentSchemaFingerprint(
            observation: pythonSpaceSchema(env.observation_space),
            action: pythonSpaceSchema(env.action_space)
        )
    }

    func reset() -> [Float] {
        let result = resetPythonEnv(env, seed: resetSeedSequence.nextResetSeed())
        return pythonFlattenedObservation(result[0])
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let np = PythonRuntime.shared.numpy
        let result = env.step(np.array(action))
        let obs = pythonFlattenedObservation(result[0])
        let reward = sanitizeScalar(Float(result[1]) ?? 0)
        let terminated = Bool(result[2]) ?? false
        let truncated = Bool(result[3]) ?? false
        return (obs, reward, terminated || truncated)
    }
}

final class GymContinuousEnv: PyEnv {
    private let env: PythonObject
    private let actionLow: [Float]
    private let actionHigh: [Float]
    private var resetSeedSequence: ResetSeedSequence
    private var initialObservation: [Float]?
    let obsDim: Int
    let actDim: Int
    let schema: EnvironmentSchemaFingerprint

    init(id: String) {
        let baseSeed = configuredRunSeed()
        let runtime = PythonRuntime.shared
        runtime.require(modules: ["gymnasium", "numpy"])
        env = runtime.gym.make(id)
        resetSeedSequence = ResetSeedSequence(baseSeed: baseSeed)
        if let baseSeed {
            seedPythonGlobals(baseSeed)
            seedSpace(env.action_space, seed: baseSeed)
            seedSpace(env.observation_space, seed: baseSeed)
        }
        let actionSpace = env.action_space
        actDim = Int(actionSpace.shape[0])!
        actionLow = pythonFlatArray(actionSpace.low)
        actionHigh = pythonFlatArray(actionSpace.high)
        let result = resetPythonEnv(env, seed: resetSeedSequence.nextResetSeed())
        let initialObservation = pythonFlattenedObservation(result[0])
        self.initialObservation = initialObservation
        obsDim = initialObservation.count
        schema = EnvironmentSchemaFingerprint(
            observation: pythonSpaceSchema(env.observation_space),
            action: pythonSpaceSchema(actionSpace)
        )
    }

    func reset() -> [Float] {
        if let initialObservation {
            self.initialObservation = nil
            return initialObservation
        }

        let result = resetPythonEnv(env, seed: resetSeedSequence.nextResetSeed())
        return pythonFlatArray(result[0])
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let safeAction = clampAction(action, low: actionLow, high: actionHigh)
        let np = PythonRuntime.shared.numpy
        let result = env.step(np.array(safeAction, dtype: np.float32))
        let reward = sanitizeScalar(Float(result[1]) ?? 0)
        let terminated = Bool(result[2]) ?? false
        let truncated = Bool(result[3]) ?? false
        return (pythonFlattenedObservation(result[0]), reward, terminated || truncated)
    }
}

final class DMControlEnv: PyEnv {
    private let env: PythonObject
    private let actionLow: [Float]
    private let actionHigh: [Float]
    private var initialObservation: [Float]?
    let obsDim: Int
    let actDim: Int
    let schema: EnvironmentSchemaFingerprint

    init(domain: String, task: String) {
        let runtime = PythonRuntime.shared
        runtime.require(modules: ["dm_control.suite", "numpy", "random"])
        let suite = runtime.module(named: "dm_control.suite")
        let baseSeed = configuredRunSeed()
        if let baseSeed {
            seedPythonGlobals(baseSeed)
            env = suite.load(domain_name: domain, task_name: task, task_kwargs: ["random": baseSeed])
        } else {
            env = suite.load(domain_name: domain, task_name: task)
        }
        let actionSpec = env.action_spec()
        actDim = Int(actionSpec.shape[0])!
        actionLow = pythonFlatArray(actionSpec.minimum)
        actionHigh = pythonFlatArray(actionSpec.maximum)
        let timeStep = env.reset()
        let initialObservation = pythonFlattenedObservation(timeStep.observation)
        self.initialObservation = initialObservation
        obsDim = initialObservation.count
        schema = EnvironmentSchemaFingerprint(
            observation: .box(flattenedDim: obsDim, shape: [obsDim]),
            action: .box(flattenedDim: actDim, shape: [actDim])
        )
    }

    func reset() -> [Float] {
        if let initialObservation {
            self.initialObservation = nil
            return initialObservation
        }

        let timeStep = env.reset()
        return pythonFlattenedObservation(timeStep.observation)
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let safeAction = clampAction(action, low: actionLow, high: actionHigh)
        let np = PythonRuntime.shared.numpy
        let timeStep = env.step(np.array(safeAction, dtype: np.float32))
        let reward = timeStep.reward
        let rewardValue = sanitizeScalar(Float(reward) ?? 0)
        return (pythonFlattenedObservation(timeStep.observation), rewardValue, Bool(timeStep.last()) ?? false)
    }
}

private protocol PolicyRunner {
    var policyName: String { get }
    var backendName: String { get }
    var telemetrySchema: DynamicTelemetrySchema { get }
    func step(obs: [Float], reward: [Float], wallClockBudgetNs: UInt64, batchSize: Int) -> AgentStepResult
}

final class VectorEnv {
    private let envs: [PyEnv]?
    private let envpoolEnv: EnvPoolEnv?
    let numEnvs: Int
    let obsDim: Int
    let actDim: Int
    let schema: EnvironmentSchemaFingerprint
    
    init(name: String, count: Int, factory: (() -> PyEnv)? = nil, envpoolTaskId: String? = nil) {
        self.numEnvs = count
        
        let supportedEnvPoolEnvs = ["cheetah", "walker", "humanoid", "minigrid"]
        if supportedEnvPoolEnvs.contains(name) {
            let baseSeed = configuredRunSeed()
            let envId: String
            if let task = envpoolTaskId, !task.isEmpty {
                envId = task
            } else if let fixed = EnvPoolCatalog.logicalNameToFixedEnvId[name] {
                envId = fixed
            } else {
                fatalError(
                    "EnvPool logical name '\(name)' needs a concrete gym id. " +
                        "For suite-style families pass envpoolTaskId from the runner, or set OAK_ENVPOOL_TASK_ID."
                )
            }
            self.envpoolEnv = EnvPoolEnv(id: envId, numEnvs: count, seed: baseSeed)
            self.envs = nil
            self.obsDim = self.envpoolEnv!.obsDim
            self.actDim = self.envpoolEnv!.actDim
            self.schema = self.envpoolEnv!.schema
        } else {
            guard let factory else {
                fatalError("Environment \(name) requires a PyEnv factory but none was provided.")
            }
            self.envs = (0..<count).map { _ in factory() }
            self.envpoolEnv = nil
            self.obsDim = self.envs![0].obsDim
            self.actDim = self.envs![0].actDim
            self.schema = self.envs![0].schema
        }
    }
    
    func reset() -> [Float] {
        if let envpoolEnv {
            return envpoolEnv.reset()
        }
        
        var obs: [Float] = []
        obs.reserveCapacity(numEnvs * obsDim)
        for env in envs! {
            obs.append(contentsOf: env.reset())
        }
        return obs
    }
    
    func step(_ action: [Float]) -> ([Float], [Float], [Bool]) {
        if let envpoolEnv {
            return envpoolEnv.step(action)
        }
        
        var nextObs: [Float] = []
        var rewards: [Float] = []
        var dones: [Bool] = []
        nextObs.reserveCapacity(numEnvs * obsDim)
        rewards.reserveCapacity(numEnvs)
        dones.reserveCapacity(numEnvs)
        
        for i in 0..<numEnvs {
            let actSlice = Array(action[(i * actDim)..<((i + 1) * actDim)])
            var result = envs![i].step(actSlice)
            
            if result.2 {
                result.0 = envs![i].reset()
                result.1 = 0 // Continual abstraction
            }
            
            nextObs.append(contentsOf: result.0)
            rewards.append(result.1)
            dones.append(result.2)
        }
        
        return (nextObs, rewards, dones)
    }

    func close() {
        envpoolEnv?.close()
    }
}

final class OaK: PolicyRunner {
    private let obsDim: Int
    private let actDim: Int
    private let core: any ForegroundCore
    private var lastAction: [Float]

    init(obsDim: Int, actDim: Int, seed: Int? = configuredRunSeed()) {
        let resolvedDevice = MTLCreateSystemDefaultDevice()
            ?? MTLCopyAllDevices().first(where: { !$0.isHeadless })
            ?? MTLCopyAllDevices().first
        self.obsDim = obsDim
        self.actDim = actDim
        lastAction = []
        guard let device = resolvedDevice else {
            fatalError("Metal device unavailable. OaK is a Metal-only runtime.")
        }
        core = MetalForegroundCore(obsDim: obsDim, actDim: actDim, device: device, seed: seed)
    }

    var policyName: String {
        "oak"
    }

    var backendName: String {
        core.backendName
    }

    var telemetrySchema: DynamicTelemetrySchema {
        core.telemetrySchema
    }

    func step(obs: [Float], reward: [Float], wallClockBudgetNs: UInt64, batchSize: Int) -> AgentStepResult {
        if lastAction.count < actDim * batchSize {
            lastAction = Array(repeating: 0, count: actDim * batchSize)
        }
        let output = core.step(input: StepInput(
            obs: sanitizeVector(obs, count: obsDim * batchSize),
            prevAction: sanitizeVector(lastAction, count: actDim * batchSize, clamp: -1...1),
            prevReward: sanitizeVector(reward, count: batchSize),
            batchSize: batchSize,
            wallClockBudgetNs: wallClockBudgetNs
        ))
        lastAction = output.action
        return AgentStepResult(action: output.action, telemetry: output.telemetry)
    }
}

private final class RandomPolicyRunner: PolicyRunner {
    private let obsDim: Int
    private let actDim: Int
    private let rng: SeededRNG?

    let policyName = "random"
    let backendName = "random"
    let telemetrySchema = DynamicTelemetrySchema.oakSelfScalingCore

    init(obsDim: Int, actDim: Int, seed: Int? = configuredRunSeed()) {
        self.obsDim = obsDim
        self.actDim = actDim
        rng = deriveSeed(seed, salt: 0x3001).map(SeededRNG.init)
    }

    func step(obs: [Float], reward: [Float], wallClockBudgetNs _: UInt64, batchSize: Int) -> AgentStepResult {
        let stepStart = currentTimeMilliseconds()
        let safeObs = sanitizeVector(obs)
        let action = (0..<(actDim * batchSize)).map { _ in seededFloat(in: -1...1, rng: rng) }
        let rewardMean = reward.isEmpty ? 0 : reward.reduce(Float.zero, +) / Float(reward.count)
        
        let telemetry = safeObs.withUnsafeBufferPointer { obsPtr in
            action.withUnsafeBufferPointer { actPtr in
                AgentTelemetry.randomPolicy(
                    rewardInput: rewardMean,
                    obs: UnsafeBufferPointer(start: obsPtr.baseAddress, count: obsPtr.count),
                    action: UnsafeBufferPointer(start: actPtr.baseAddress, count: actPtr.count),
                    agentStepMs: currentTimeMilliseconds() - stepStart,
                    dynamicTelemetryValues: nil
                )
            }
        }
        
        return AgentStepResult(action: action, telemetry: telemetry)
    }
}

private struct EnvRun {
    let baseName: String
    let envpoolTaskId: String?
    let factory: (() -> PyEnv)?
}

private func envRunsForRequest(availableEnvs: [(String, (() -> PyEnv)?)], requestedEnv: String) -> [EnvRun] {
    let req = requestedEnv.lowercased()

    func expandSuiteRuns(logicalName: String, prefix: String) -> [EnvRun] {
        let ids = EnvPoolCatalog.registeredEnvIds(withPrefix: prefix)
        guard !ids.isEmpty else {
            fatalError("envpool has no registered ids with prefix '\(prefix)'. Check envpool build and prefix.")
        }
        return ids.map { EnvRun(baseName: logicalName, envpoolTaskId: $0, factory: nil) }
    }

    /// `minigrid` is a short name for "EnvPool tasks under the default MiniGrid family prefix"; suite vs one task is generic envpool config.
    func resolveMinigridRuns() -> [EnvRun] {
        let primary = ProcessInfo.processInfo.environment["OAK_ENVPOOL_TASK_ID"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let legacy = ProcessInfo.processInfo.environment["OAK_MINIGRID_ENV_ID"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let task: String
        if let p = primary, !p.isEmpty {
            task = p
        } else if let l = legacy, !l.isEmpty {
            task = l
        } else {
            task = ""
        }
        if !task.isEmpty, task.lowercased() != "suite" {
            return [EnvRun(baseName: "minigrid", envpoolTaskId: task, factory: nil)]
        }
        let prefixEnv = ProcessInfo.processInfo.environment["OAK_ENVPOOL_SUITE_PREFIX"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let prefix = (prefixEnv?.isEmpty == false) ? prefixEnv! : EnvPoolCatalog.minigridSuitePrefix
        return expandSuiteRuns(logicalName: "minigrid", prefix: prefix)
    }

    if req == "all" {
        var runs: [EnvRun] = []
        for (name, factory) in availableEnvs {
            if name == "minigrid" {
                runs.append(contentsOf: resolveMinigridRuns())
            } else if let fixed = EnvPoolCatalog.logicalNameToFixedEnvId[name] {
                runs.append(EnvRun(baseName: name, envpoolTaskId: fixed, factory: nil))
            } else {
                runs.append(EnvRun(baseName: name, envpoolTaskId: nil, factory: factory))
            }
        }
        return runs
    }

    if req == "minigrid" {
        return resolveMinigridRuns()
    }

    if let fixed = EnvPoolCatalog.logicalNameToFixedEnvId[req] {
        return [EnvRun(baseName: req, envpoolTaskId: fixed, factory: nil)]
    }

    return availableEnvs
        .filter { $0.0.lowercased() == req }
        .map { EnvRun(baseName: $0.0, envpoolTaskId: nil, factory: $0.1) }
}

@main struct Main {
    static func main() {
        run()
    }

    private static func run() {
        let cliArguments = Array(CommandLine.arguments.dropFirst())
        if let command = cliArguments.first, command == "doctor-metal" {
            Foundation.exit(runMetalDoctor(arguments: cliArguments.dropFirst()))
        }

        print("OaK - Alberta Plan")
        let availableEnvs: [(String, (() -> PyEnv)?)] = [
            ("pyflyt", { PyFlytEnv() }),
            ("cheetah", nil),
            ("walker", nil),
            ("humanoid", nil),
            ("minigrid", nil),
        ]

        let requestedEnv = ProcessInfo.processInfo.environment["OAK_ENV"]
            ?? cliArguments.first
            ?? "all"
        let requestedPolicy = (ProcessInfo.processInfo.environment["OAK_POLICY"] ?? "oak").lowercased()
        let runSeed = configuredRunSeed()
        let configuredBudgetNs = configuredStepBudgetNs()
        let runControl = RunControl.resolve()
        let maxEnvironmentSteps = ProcessInfo.processInfo.environment["OAK_MAX_STEPS"].flatMap(Int.init)
        let envsToRun = envRunsForRequest(availableEnvs: availableEnvs, requestedEnv: requestedEnv)

        guard ["oak", "random"].contains(requestedPolicy) else {
            fputs("Unknown policy '\(requestedPolicy)'. Choose one of: oak, random\n", stderr)
            Foundation.exit(1)
        }

        guard !envsToRun.isEmpty else {
            let choices = (availableEnvs.map(\.0) + ["all"]).joined(separator: ", ")
            fputs("Unknown environment '\(requestedEnv)'. Choose one of: \(choices)\n", stderr)
            Foundation.exit(1)
        }

        let numEnvs = ProcessInfo.processInfo.environment["OAK_NUM_ENVS"].flatMap(Int.init) ?? 1
        let benchmarkLeague = ProcessInfo.processInfo.environment["OAK_BENCH_LEAGUE"]?.trimmingCharacters(in: .whitespacesAndNewlines)

        for run in envsToRun {
            let name = run.baseName
            let metricsEnvName: String = run.envpoolTaskId ?? name
            let env = VectorEnv(name: name, count: numEnvs, factory: run.factory, envpoolTaskId: run.envpoolTaskId)
            let runner: any PolicyRunner = requestedPolicy == "random"
                ? RandomPolicyRunner(obsDim: env.obsDim, actDim: env.actDim, seed: runSeed)
                : OaK(obsDim: env.obsDim, actDim: env.actDim, seed: runSeed)
            let seedSuffix = runSeed.map { " seed=\($0)" } ?? ""
            print("\n\(metricsEnvName) (\(type(of: env))) [policy=\(runner.policyName)\(seedSuffix)]")
            let metricsConfiguration = MetricsConfiguration.resolve(
                envName: metricsEnvName,
                policyName: runner.policyName,
                runControl: runControl,
                numEnvs: numEnvs
            )
            let metrics = RunMetricsAggregator(
                envName: metricsEnvName,
                logicalEnvName: name,
                concreteEnvId: metricsEnvName,
                leagueName: benchmarkLeague?.isEmpty == false ? benchmarkLeague : nil,
                policyName: runner.policyName,
                backend: runner.backendName,
                schema: env.schema,
                dynamicTelemetrySchema: runner.telemetrySchema,
                windowSize: metricsConfiguration.windowSize,
                seed: runSeed,
                numEnvs: numEnvs
            )
            var metricsSink: JSONLMetricsSink?
            if let fileURL = metricsConfiguration.fileURL {
                do {
                    metricsSink = try JSONLMetricsSink(fileURL: fileURL)
                    try metricsSink?.write(metrics.runStartRecord(control: runControl))
                } catch {
                    fatalError("Failed to initialize metrics: \(error)")
                }
            }
            var obs = env.reset()
            var pendingRewards = Array(repeating: Float(0), count: numEnvs)
            var totalEnvironmentSteps = 0
            var measuredSteps = 0
            var measuredSeconds = 0.0
            var measurementStartMs: Double?
            var stopReason = "completed"
            var adaptiveBudgetNs: UInt64 = configuredBudgetNs ?? 1
            defer {
                let finalRecord = metrics.runEndRecord(
                    control: runControl,
                    stopReason: stopReason,
                    measuredSeconds: measuredSeconds,
                    globalMeasuredEnvSteps: measuredSteps
                )
                if let metricsSink {
                    do {
                        try metricsSink.write(finalRecord)
                    } catch {
                        fatalError("Failed to write metrics run_end record: \(error)")
                    }
                }
                print(formatMetricsSummary(finalRecord))
            }
            while true {
                let nextEnvironmentStep = totalEnvironmentSteps + numEnvs
                let isMeasuredStep = nextEnvironmentStep > runControl.warmupSteps
                let totalStepStart = currentTimeMilliseconds()
                if isMeasuredStep && measurementStartMs == nil {
                    measurementStartMs = totalStepStart
                    if runControl.warmupSteps > 0 {
                        print("[metrics] warmup_complete env=\(metricsEnvName) policy=\(runner.policyName) warmup_steps=\(runControl.warmupSteps)")
                    }
                }
                let agentResult = runner.step(obs: obs, reward: pendingRewards, wallClockBudgetNs: adaptiveBudgetNs, batchSize: numEnvs)
                let envStepStart = currentTimeMilliseconds()
                let cleanAction = sanitizeVector(agentResult.action, count: numEnvs * env.actDim, clamp: -1...1)
                let result = env.step(cleanAction)
                let envStepMs = currentTimeMilliseconds() - envStepStart
                let totalStepMs = currentTimeMilliseconds() - totalStepStart
                if configuredBudgetNs == nil {
                    adaptiveBudgetNs = 10_000 // Fixed 10us budget for continual planning
                }
                totalEnvironmentSteps = nextEnvironmentStep
                var shouldStop = false
                if isMeasuredStep {
                    measuredSteps += numEnvs
                    if let measurementStartMs {
                        measuredSeconds = max((currentTimeMilliseconds() - measurementStartMs) / 1000, 0)
                    }

                    let metricsRecords = metrics.recordStep(
                        rewards: result.1,
                        dones: result.2,
                        telemetry: agentResult.telemetry,
                        envStepMs: envStepMs,
                        totalStepMs: totalStepMs
                    )
                    do {
                        if let summaryRecord = metricsRecords.summary {
                            if let metricsSink {
                                try metricsSink.write(summaryRecord)
                            }
                            print(formatMetricsSummary(summaryRecord))
                        }
                    } catch {
                        fatalError("Failed to write metrics record: \(error)")
                    }

                    if let stopAfterMeasuredSteps = runControl.stopAfterMeasuredSteps, measuredSteps >= stopAfterMeasuredSteps {
                        stopReason = "measured_steps"
                        shouldStop = true
                    } else if let stopAfterMeasuredSeconds = runControl.stopAfterMeasuredSeconds, measuredSeconds >= stopAfterMeasuredSeconds {
                        stopReason = "measured_seconds"
                        shouldStop = true
                    }
                }
                if let maxEnvironmentSteps, totalEnvironmentSteps >= maxEnvironmentSteps {
                    stopReason = "max_env_steps"
                    shouldStop = true
                }

                obs = result.0
                pendingRewards = result.1

                if shouldStop {
                    print("Stopping after \(measuredSteps) measured steps (\(totalEnvironmentSteps) total env steps), reason=\(stopReason)")
                    env.close()
                    break
                }
            }
        }
    }
}
