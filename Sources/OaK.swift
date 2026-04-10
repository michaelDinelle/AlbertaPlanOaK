import Darwin
import Foundation
import Metal
import PythonKit

private func candidateCarlaRoots() -> [String] {
    let fileManager = FileManager.default
    let currentDirectory = ProcessInfo.processInfo.environment["OAK_ROOT_DIR"] ?? fileManager.currentDirectoryPath
    let homeDirectory = fileManager.homeDirectoryForCurrentUser.path

    return [
        "\(currentDirectory)/CARLA_0.9.15",
        "\(homeDirectory)/Applications/CARLA_0.9.15",
    ]
}

private func nativeHostCarlaPythonSearchPaths() -> [String] {
    let fileManager = FileManager.default

    return candidateCarlaRoots().flatMap { root in
        let pythonAPIPath = "\(root)/PythonAPI/carla"
        let distPath = "\(pythonAPIPath)/dist"
        var paths: [String] = []

        if fileManager.fileExists(atPath: pythonAPIPath) {
            paths.append(pythonAPIPath)
        }

        if let entries = try? fileManager.contentsOfDirectory(atPath: distPath) {
            paths.append(contentsOf: entries
                .filter { $0.hasSuffix(".egg") && !$0.contains("win-amd64") }
                .map { "\(distPath)/\($0)" })
        }

        return paths
    }
}

private func wineZDrivePath(for hostPath: String) -> String {
    "Z:" + hostPath.replacingOccurrences(of: "/", with: "\\")
}

private func shellQuoted(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

private final class PythonRuntime: @unchecked Sendable {
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
            fatalError("Missing Python module '\(name)'. Run ./doctor.sh python for details. Underlying error: \(error)")
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

            for path in nativeHostCarlaPythonSearchPaths() {
                importedSys.path.append(path)
            }

            configured = true
        } catch {
            fatalError("Python runtime unavailable. Run ./doctor.sh python for details. Underlying error: \(error)")
        }
    }
}

private func logWarning(_ message: String) {
    fputs("[OaK] \(message)\n", stderr)
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

private func pythonFlatArray(_ object: PythonObject) -> [Float] {
    let np = PythonRuntime.shared.numpy
    let flattened = np.asarray(object, dtype: np.float32).flatten()
    return Array(flattened).map { sanitizeScalar(Float($0) ?? 0) }
}

private func discreteActionIndex(from value: Float, count: Int) -> Int {
    let normalized = sanitizeScalar(value, clamp: -1...1)
    let scaled = ((normalized + 1) * 0.5) * Float(count)
    return min(max(Int(scaled), 0), count - 1)
}

private func zeroBuffer(_ buffer: MTLBuffer) {
    memset(buffer.contents(), 0, buffer.length)
}

private func carlaImageSize() -> Int {
    let configuredSize = ProcessInfo.processInfo.environment["OAK_CARLA_IMAGE_SIZE"].flatMap(Int.init) ?? 4
    return max(configuredSize, 4)
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
    func reset() -> [Float]
    func step(_ action: [Float]) -> ([Float], Float, Bool)
}

final class PyFlytEnv: PyEnv {
    private let env: PythonObject
    private var resetSeedSequence: ResetSeedSequence
    let obsDim: Int
    let actDim: Int

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
    }

    func reset() -> [Float] {
        let result = resetPythonEnv(env, seed: resetSeedSequence.nextResetSeed())
        return Array(result[0]).map { Float($0)! }
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let np = PythonRuntime.shared.numpy
        let result = env.step(np.array(action))
        let obs = Array(result[0]).map { Float($0)! }
        let reward = Float(result[1])!
        let terminated = Bool(result[2])!
        let truncated = Bool(result[3])!
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
        let initialObservation = pythonFlatArray(result[0])
        self.initialObservation = initialObservation
        obsDim = initialObservation.count
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
        return (pythonFlatArray(result[0]), reward, terminated || truncated)
    }
}

final class CoreSignalEnv: PyEnv {
    private let rewardProjector: [Float]
    private let driftKernel: [Float]
    private let baseObservation: [Float]
    private var observation: [Float]
    let obsDim: Int
    let actDim: Int

    init() {
        let environment = ProcessInfo.processInfo.environment
        let processorScale = max(ProcessInfo.processInfo.activeProcessorCount, 1)
        let resolvedObsDim = max(environment["OAK_CORE_OBS_DIM"].flatMap(Int.init) ?? (processorScale * processorScale), 1)
        let resolvedActDim = max(environment["OAK_CORE_ACT_DIM"].flatMap(Int.init) ?? processorScale, 1)
        obsDim = resolvedObsDim
        actDim = resolvedActDim

        let rng = deriveSeed(configuredRunSeed(environment: environment), salt: 0x4001).map(SeededRNG.init)
        let projectorScale = Float(1 / sqrt(Double(max(resolvedObsDim, 1))))
        let driftScale = Float(1 / sqrt(Double(max(resolvedActDim, 1))))
        rewardProjector = (0..<resolvedObsDim).map { index in
            let base = rng?.nextFloat(in: -projectorScale...projectorScale) ?? Float(index + 1) / Float(resolvedObsDim + 1)
            return sanitizeScalar(base)
        }
        driftKernel = (0..<resolvedActDim).map { index in
            let base = rng?.nextFloat(in: -driftScale...driftScale) ?? Float(index + 1) / Float(resolvedActDim + 1)
            return sanitizeScalar(base, clamp: -1...1)
        }
        baseObservation = (0..<resolvedObsDim).map { index in
            let phase = Float(index + 1) / Float(resolvedObsDim + 1)
            return sin(phase * .pi)
        }
        observation = baseObservation
    }

    func reset() -> [Float] {
        observation = baseObservation
        return observation
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let safeAction = sanitizeVector(action, count: actDim, clamp: -1...1)
        let projectedAction = (0..<obsDim).map { rewardProjector[$0] * safeAction[$0 % actDim] }
        let reward = zip(observation, projectedAction).reduce(Float(0)) { partial, pair in
            partial + pair.0 * pair.1
        } / Float(max(obsDim, 1))

        var nextObservation = Array(repeating: Float(0), count: obsDim)
        for index in 0..<obsDim {
            let neighbor = observation[(index + 1) % obsDim]
            let actionInfluence = driftKernel[index % actDim] * safeAction[index % actDim]
            nextObservation[index] = sanitizeScalar(tanh(observation[index] + neighbor + actionInfluence + reward))
        }
        observation = nextObservation
        return (observation, sanitizeScalar(reward), false)
    }
}

final class DirectCARLAEnv: PyEnv {
    private static let vehicleRoleName = "oak_vehicle"
    private static let cameraRoleName = "oak_camera"
    private let client: PythonObject
    private let world: PythonObject
    private let imageSize: Int
    private let originalSynchronousMode: Bool
    private let originalFixedDeltaSeconds: Double?
    private var vehicle: PythonObject?
    private var camera: PythonObject?
    private var image: [Float]
    let obsDim: Int
    let actDim: Int

    init() {
        let runtime = PythonRuntime.shared
        runtime.require(modules: ["carla", "numpy"])
        let carla = runtime.module(named: "carla")
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        imageSize = carlaImageSize()
        obsDim = imageSize * imageSize * 3
        actDim = 2
        image = Array(repeating: 0, count: obsDim)
        let settings = world.get_settings()
        originalSynchronousMode = Bool(settings.synchronous_mode) ?? false
        originalFixedDeltaSeconds = Double(settings.fixed_delta_seconds)
        settings.synchronous_mode = true
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
    }

    func reset() -> [Float] {
        let destroyedActors = destroyActors() || destroyStaleActors()
        if destroyedActors {
            world.tick()
        }
        let runtime = PythonRuntime.shared
        let carla = runtime.module(named: "carla")
        let np = runtime.numpy
        let blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name", Self.vehicleRoleName)
        vehicle = spawnVehicle(with: blueprint)
        let camBp = world.get_blueprint_library().find("sensor.camera.rgb")
        camBp.set_attribute("role_name", Self.cameraRoleName)
        camBp.set_attribute("image_size_x", "\(imageSize)")
        camBp.set_attribute("image_size_y", "\(imageSize)")
        let transform = carla.Transform(carla.Location(x: 1.5, z: 2.4))
        camera = world.spawn_actor(camBp, transform, attach_to: vehicle)
        let callback = PythonFunction { [weak self] (img: PythonObject) in
            let data = np.frombuffer(img.raw_data, dtype: np.uint8)
            var nextImage: [Float] = []
            nextImage.reserveCapacity(self?.obsDim ?? (carlaImageSize() * carlaImageSize() * 3))

            let rawBytes = Array(data)
            var index = 0
            while index + 2 < rawBytes.count {
                nextImage.append(Float(rawBytes[index + 2])! / 255.0)
                nextImage.append(Float(rawBytes[index + 1])! / 255.0)
                nextImage.append(Float(rawBytes[index])! / 255.0)
                index += 4
            }

            self?.image = nextImage
            return Python.None
        }.pythonObject
        camera!.listen(callback)
        world.tick()
        return image
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        guard let v = vehicle else {
            return (image, 0, true)
        }
        let safeAction = sanitizeVector(action, count: actDim, clamp: -1...1)
        let carla = PythonRuntime.shared.module(named: "carla")
        let control = carla.VehicleControl()
        control.steer = PythonObject(safeAction[0])
        control.throttle = PythonObject(max(0, safeAction[1]))
        control.brake = PythonObject(max(0, -safeAction[1]))
        v.apply_control(control)
        world.tick()
        let velocity = v.get_velocity()
        let vx = Double(velocity.x) ?? 0
        let vy = Double(velocity.y) ?? 0
        let vz = Double(velocity.z) ?? 0
        let speed = sqrt(vx * vx + vy * vy + vz * vz)
        let reward = Float(speed) * 0.01
        return (image, reward, false)
    }

    deinit {
        if destroyActors() {
            world.tick()
        }
        restoreWorldSettings()
    }

    private func destroyActors() -> Bool {
        var destroyed = false
        if let c = camera {
            _ = c.stop()
            c.destroy()
            camera = nil
            destroyed = true
        }
        if let v = vehicle {
            v.destroy()
            vehicle = nil
            destroyed = true
        }
        return destroyed
    }

    private func destroyStaleActors() -> Bool {
        var destroyed = false
        for actor in Array(world.get_actors()) {
            let roleName = String(actor.attributes["role_name"]) ?? ""
            if roleName != Self.vehicleRoleName && roleName != Self.cameraRoleName {
                continue
            }
            _ = actor.destroy()
            destroyed = true
        }
        return destroyed
    }

    private func spawnVehicle(with blueprint: PythonObject) -> PythonObject {
        for spawnPoint in Array(world.get_map().get_spawn_points()) {
            let spawnedVehicle = world.try_spawn_actor(blueprint, spawnPoint)
            if spawnedVehicle != Python.None {
                return spawnedVehicle
            }
        }
        fatalError("Failed to spawn OaK vehicle at any CARLA spawn point")
    }

    private func restoreWorldSettings() {
        let settings = world.get_settings()
        settings.synchronous_mode = PythonObject(originalSynchronousMode)
        if let originalFixedDeltaSeconds {
            settings.fixed_delta_seconds = PythonObject(originalFixedDeltaSeconds)
        } else {
            settings.fixed_delta_seconds = Python.None
        }
        world.apply_settings(settings)
    }
}

fileprivate struct WhiskyCARLAConfiguration {
    let whiskyCmd: String
    let bottleName: String
    let windowsPython: String
    let windowsCarlaEgg: String
    let windowsCarlaPythonAPI: String
    let windowsBridgeScript: String
    let bridgeHost: String
    let bridgePort: String
    let carlaHost: String
    let carlaPort: String

    static func resolve() -> WhiskyCARLAConfiguration? {
        let env = ProcessInfo.processInfo.environment
        let fileManager = FileManager.default
        let rootDirectory = env["OAK_ROOT_DIR"] ?? fileManager.currentDirectoryPath
        let bridgeScriptHostPath = "\(rootDirectory)/scripts/carla_whisky_bridge.py"

        guard
            let windowsPython = env["OAK_CARLA_WIN_PYTHON"],
            let windowsCarlaEgg = env["OAK_CARLA_WIN_EGG"],
            let windowsCarlaPythonAPI = env["OAK_CARLA_WIN_PYTHONAPI"],
            fileManager.fileExists(atPath: bridgeScriptHostPath)
        else {
            return nil
        }

        return WhiskyCARLAConfiguration(
            whiskyCmd: env["OAK_WHISKY_CMD"] ?? "/Applications/Whisky.app/Contents/Resources/WhiskyCmd",
            bottleName: env["OAK_WHISKY_BOTTLE"] ?? "CARLA",
            windowsPython: windowsPython,
            windowsCarlaEgg: windowsCarlaEgg,
            windowsCarlaPythonAPI: windowsCarlaPythonAPI,
            windowsBridgeScript: wineZDrivePath(for: bridgeScriptHostPath),
            bridgeHost: env["OAK_CARLA_BRIDGE_HOST"] ?? "127.0.0.1",
            bridgePort: env["OAK_CARLA_BRIDGE_PORT"] ?? String(Int.random(in: 23000...29999)),
            carlaHost: env["OAK_CARLA_HOST"] ?? "127.0.0.1",
            carlaPort: env["OAK_CARLA_PORT"] ?? "2000"
        )
    }
}

final class WhiskyCARLAEnv: PyEnv {
    private let process: Process
    private let stderrPipe: Pipe
    private let bridgeSocket: Int32
    private var bridgeReadBuffer: Data
    private let debugEnabled: Bool
    private(set) var obsDim: Int
    private(set) var actDim: Int

    fileprivate init(configuration: WhiskyCARLAConfiguration) {
        process = Process()
        stderrPipe = Pipe()
        bridgeReadBuffer = Data()
        debugEnabled = ProcessInfo.processInfo.environment["OAK_CARLA_DEBUG"] == "1"
        obsDim = 0
        actDim = 0

        let shellCommand = """
        eval "$(\(shellQuoted(configuration.whiskyCmd)) shellenv \(shellQuoted(configuration.bottleName)))" && exec wine64 \(shellQuoted(configuration.windowsPython)) \(shellQuoted(configuration.windowsBridgeScript)) serve \(shellQuoted(configuration.windowsCarlaEgg)) \(shellQuoted(configuration.windowsCarlaPythonAPI)) \(shellQuoted(configuration.bridgeHost)) \(shellQuoted(configuration.bridgePort)) \(shellQuoted(configuration.carlaHost)) \(shellQuoted(configuration.carlaPort))
        """

        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", shellCommand]
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "PYTHON_LIBRARY")
        environment.removeValue(forKey: "PYTHONEXECUTABLE")
        environment.removeValue(forKey: "PYTHONNOUSERSITE")
        process.environment = environment
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = stderrPipe

        do {
            try process.run()
            bridgeSocket = try Self.connectToBridge(
                host: configuration.bridgeHost,
                port: Int(configuration.bridgePort) ?? 0,
                process: process,
                stderrPipe: stderrPipe,
                timeout: 15,
                debugEnabled: debugEnabled
            )
            let readyResponse = try readResponse()
            let dimensions = try Self.parseReadyDimensions(from: readyResponse)
            obsDim = dimensions.obsDim
            actDim = dimensions.actDim
            if debugEnabled {
                logWarning("CARLA Whisky bridge ready with obsDim \(obsDim) and actDim \(actDim)")
            }
        } catch {
            fatalError("Failed to start CARLA Whisky bridge: \(error)")
        }
    }

    func reset() -> [Float] {
        do {
            if debugEnabled {
                logWarning("Sending CARLA reset request")
            }
            try send(request: ["cmd": "reset"])
            let response = try readResponse()
            guard let status = response["status"] as? String, status == "ok" else {
                fatalError(bridgeErrorMessage(from: response))
            }
            let image = decodeImage(from: response)
            if debugEnabled {
                logWarning("Decoded CARLA reset image with \(image.count) floats")
            }
            return image
        } catch {
            fatalError("Failed to reset CARLA via Whisky bridge: \(error)")
        }
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        do {
            if debugEnabled {
                logWarning("Sending CARLA step request")
            }
            try send(request: [
                "cmd": "step",
                "action": sanitizeVector(action, count: actDim, clamp: -1...1),
            ])
            let response = try readResponse()
            guard let status = response["status"] as? String, status == "ok" else {
                fatalError(bridgeErrorMessage(from: response))
            }

            let image = decodeImage(from: response)
            let reward = sanitizeScalar((response["reward"] as? NSNumber)?.floatValue ?? 0)
            let done = (response["done"] as? NSNumber)?.boolValue ?? false
            if debugEnabled {
                logWarning("Decoded CARLA step response with \(image.count) floats, reward \(reward), done \(done)")
            }
            return (image, reward, done)
        } catch {
            fatalError("Failed to step CARLA via Whisky bridge: \(error)")
        }
    }

    deinit {
        if process.isRunning {
            if debugEnabled {
                logWarning("Sending CARLA close request")
            }
            do {
                try send(request: ["cmd": "close"])
                _ = try readResponse()
                if debugEnabled {
                    logWarning("CARLA close request acknowledged")
                }
            } catch {
                if debugEnabled {
                    logWarning("CARLA bridge close handshake failed: \(error)")
                }
            }
        }

        _ = Darwin.shutdown(bridgeSocket, SHUT_RDWR)
        Darwin.close(bridgeSocket)

        if process.isRunning {
            let deadline = Date().addingTimeInterval(2)
            while process.isRunning && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
        }

        if process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
    }

    private func send(request: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: request)
        data.append(0x0A)
        try data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                return
            }

            var written = 0
            while written < rawBuffer.count {
                let result = Darwin.write(bridgeSocket, baseAddress.advanced(by: written), rawBuffer.count - written)
                if result < 0 {
                    throw NSError(
                        domain: "WhiskyCARLAEnv",
                        code: 4,
                        userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(errno))]
                    )
                }
                written += result
            }
        }
    }

    private func readResponse() throws -> [String: Any] {
        let payloadText = try readNextPayload()
        guard let data = payloadText.data(using: .utf8) else {
            throw NSError(domain: "WhiskyCARLAEnv", code: 1, userInfo: [NSLocalizedDescriptionKey: "Bridge returned non-UTF8 output"])
        }
        guard let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "WhiskyCARLAEnv", code: 2, userInfo: [NSLocalizedDescriptionKey: "Bridge returned invalid JSON"])
        }
        return payload
    }

    private func readNextPayload() throws -> String {
        while true {
            if let payload = extractBufferedJSONPayload() {
                if debugEnabled {
                    logWarning("CARLA bridge payload: \(summarizedForDebug(payload))")
                }
                return payload
            }

            var buffer = [UInt8](repeating: 0, count: 4096)
            let bytesRead = Darwin.read(bridgeSocket, &buffer, buffer.count)

            if bytesRead > 0 {
                let chunk = Data(buffer.prefix(bytesRead))
                if debugEnabled {
                    logWarning("CARLA bridge received \(bytesRead) bytes")
                }
                bridgeReadBuffer.append(chunk)
                continue
            }

            if bytesRead == 0 {
                if let payload = extractBufferedJSONPayload() {
                    if debugEnabled {
                        logWarning("CARLA bridge payload at EOF: \(summarizedForDebug(payload))")
                    }
                    return payload
                }
                throw try Self.bridgeProcessError(process: process, stderrPipe: stderrPipe, defaultMessage: "CARLA Whisky bridge closed the control socket")
            }

            if errno == EINTR {
                continue
            }

            throw NSError(
                domain: "WhiskyCARLAEnv",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(errno))]
            )
        }
    }

    private func decodeImage(from response: [String: Any]) -> [Float] {
        guard let image = response["image"] as? [NSNumber] else {
            fatalError("CARLA Whisky bridge did not return an image payload")
        }
        let decoded = image.map { sanitizeScalar($0.floatValue) }
        guard decoded.count == obsDim else {
            fatalError("CARLA Whisky bridge returned \(decoded.count) floats, expected \(obsDim)")
        }
        return decoded
    }

    private func bridgeErrorMessage(from response: [String: Any]) -> String {
        let error = (response["error"] as? String) ?? "Unknown CARLA bridge error"
        let traceback = (response["traceback"] as? String) ?? ""
        return traceback.isEmpty ? error : "\(error)\n\(traceback)"
    }

    private func extractBufferedJSONPayload() -> String? {
        guard let start = bridgeReadBuffer.firstIndex(of: 0x7B) else {
            return nil
        }
        guard let end = bridgeReadBuffer[start...].firstIndex(of: 0x7D) else {
            return nil
        }

        let payloadData = bridgeReadBuffer.subdata(in: start..<bridgeReadBuffer.index(after: end))
        bridgeReadBuffer.removeSubrange(bridgeReadBuffer.startIndex..<bridgeReadBuffer.index(after: end))
        return String(decoding: payloadData, as: UTF8.self)
    }

    private func summarizedForDebug(_ text: String, limit: Int = 240) -> String {
        guard text.count > limit else {
            return text
        }
        return "\(text.prefix(limit))... [\(text.count) chars]"
    }

    private static func connectToBridge(
        host: String,
        port: Int,
        process: Process,
        stderrPipe: Pipe,
        timeout: TimeInterval,
        debugEnabled: Bool
    ) throws -> Int32 {
        guard port > 0 else {
            throw NSError(domain: "WhiskyCARLAEnv", code: 6, userInfo: [NSLocalizedDescriptionKey: "Invalid CARLA bridge port \(port)"])
        }

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)

        guard inet_pton(AF_INET, host, &address.sin_addr) == 1 else {
            throw NSError(domain: "WhiskyCARLAEnv", code: 7, userInfo: [NSLocalizedDescriptionKey: "Invalid CARLA bridge host \(host)"])
        }

        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if !process.isRunning {
                throw try bridgeProcessError(process: process, stderrPipe: stderrPipe, defaultMessage: "CARLA Whisky bridge exited before opening its control socket")
            }

            let socketFD = Darwin.socket(AF_INET, SOCK_STREAM, 0)
            if socketFD >= 0 {
                var connectionAddress = address
                let result = withUnsafePointer(to: &connectionAddress) {
                    $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { pointer in
                        Darwin.connect(socketFD, pointer, socklen_t(MemoryLayout<sockaddr_in>.size))
                    }
                }

                if result == 0 {
                    if debugEnabled {
                        logWarning("Connected to CARLA bridge socket on \(host):\(port)")
                    }
                    return socketFD
                }

                Darwin.close(socketFD)
            }

            Thread.sleep(forTimeInterval: 0.1)
        }

        throw try bridgeProcessError(process: process, stderrPipe: stderrPipe, defaultMessage: "Timed out waiting for the CARLA Whisky bridge socket on \(host):\(port)")
    }

    private static func bridgeProcessError(process: Process, stderrPipe: Pipe, defaultMessage: String) throws -> NSError {
        let stderrData = try stderrPipe.fileHandleForReading.readToEnd() ?? Data()
        let stderrOutput = String(decoding: stderrData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        let message = stderrOutput.isEmpty ? defaultMessage : stderrOutput
        return NSError(domain: "WhiskyCARLAEnv", code: 3, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private static func parseReadyDimensions(from response: [String: Any]) throws -> (obsDim: Int, actDim: Int) {
        guard let status = response["status"] as? String, status == "ready" else {
            let error = (response["error"] as? String) ?? "Unexpected CARLA bridge handshake"
            throw NSError(domain: "WhiskyCARLAEnv", code: 8, userInfo: [NSLocalizedDescriptionKey: error])
        }

        guard
            let obsDimValue = (response["obs_dim"] as? NSNumber)?.intValue,
            let actDimValue = (response["act_dim"] as? NSNumber)?.intValue,
            obsDimValue > 0,
            actDimValue > 0
        else {
            throw NSError(domain: "WhiskyCARLAEnv", code: 9, userInfo: [NSLocalizedDescriptionKey: "CARLA bridge handshake did not include valid dimensions"])
        }

        return (obsDimValue, actDimValue)
    }
}

final class CARLAEnv: PyEnv {
    private let backend: PyEnv
    let obsDim: Int
    let actDim: Int

    init() {
        let environment = ProcessInfo.processInfo.environment
        if environment["OAK_CARLA_BRIDGE"] != "1", PythonRuntime.shared.hasModule(named: "carla") {
            backend = DirectCARLAEnv()
        } else if let configuration = WhiskyCARLAConfiguration.resolve() {
            backend = WhiskyCARLAEnv(configuration: configuration)
        } else {
            fatalError("CARLA is configured for Whisky but the bridge environment is missing. Run ./setup.sh and launch with ./run.sh.")
        }
        obsDim = backend.obsDim
        actDim = backend.actDim
    }

    func reset() -> [Float] {
        backend.reset()
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        backend.step(action)
    }
}

final class DMControlEnv: PyEnv {
    private let env: PythonObject
    private let actionLow: [Float]
    private let actionHigh: [Float]
    private var initialObservation: [Float]?
    let obsDim: Int
    let actDim: Int

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
        let initialObservation = Self.flattenObservation(timeStep.observation)
        self.initialObservation = initialObservation
        obsDim = initialObservation.count
    }

    func reset() -> [Float] {
        if let initialObservation {
            self.initialObservation = nil
            return initialObservation
        }

        let timeStep = env.reset()
        return Self.flattenObservation(timeStep.observation)
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let safeAction = clampAction(action, low: actionLow, high: actionHigh)
        let np = PythonRuntime.shared.numpy
        let timeStep = env.step(np.array(safeAction, dtype: np.float32))
        let reward = timeStep.reward
        let rewardValue = sanitizeScalar(Float(reward) ?? 0)
        return (Self.flattenObservation(timeStep.observation), rewardValue, Bool(timeStep.last()) ?? false)
    }

    private static func flattenObservation(_ observation: PythonObject) -> [Float] {
        var result: [Float] = []
        for key in observation.keys() {
            result.append(contentsOf: pythonFlatArray(observation[key]))
        }
        return result
    }
}

final class MiniGridEnv: PyEnv {
    private let env: PythonObject
    private var resetSeedSequence: ResetSeedSequence
    private var initialObservation: [Float]?
    let obsDim: Int
    let actDim: Int

    init(id: String = "MiniGrid-Empty-8x8-v0") {
        let baseSeed = configuredRunSeed()
        let runtime = PythonRuntime.shared
        runtime.require(modules: ["minigrid", "gymnasium", "numpy"])
        env = runtime.gym.make(id)
        resetSeedSequence = ResetSeedSequence(baseSeed: baseSeed)
        if let baseSeed {
            seedPythonGlobals(baseSeed)
            seedSpace(env.action_space, seed: baseSeed)
            seedSpace(env.observation_space, seed: baseSeed)
        }
        actDim = Int(env.action_space.n) ?? 7
        let result = resetPythonEnv(env, seed: resetSeedSequence.nextResetSeed())
        let initialObservation = Self.encodeObservation(result[0])
        self.initialObservation = initialObservation
        obsDim = initialObservation.count
    }

    func reset() -> [Float] {
        if let initialObservation {
            self.initialObservation = nil
            return initialObservation
        }

        let result = resetPythonEnv(env, seed: resetSeedSequence.nextResetSeed())
        return Self.encodeObservation(result[0])
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let actionIndex = discreteActionIndex(from: action.first ?? 0, count: actDim)
        let result = env.step(actionIndex)
        let obs = Self.encodeObservation(result[0])
        let reward = sanitizeScalar(Float(result[1]) ?? 0)
        let terminated = Bool(result[2]) ?? false
        let truncated = Bool(result[3]) ?? false
        return (obs, reward, terminated || truncated)
    }

    private static func encodeObservation(_ observation: PythonObject) -> [Float] {
        let image = observation["image"]
        return pythonFlatArray(image).map { $0 / 10.0 }
    }
}

private protocol PolicyRunner {
    var policyName: String { get }
    var backendName: String { get }
    func step(obs: [Float], reward: [Float], wallClockBudgetNs: UInt64, batchSize: Int) -> AgentStepResult
}

final class VectorEnv {
    private let envs: [PyEnv]?
    private let batchedEnv: PythonObject?
    let numEnvs: Int
    let obsDim: Int
    let actDim: Int
    
    init(name: String, factory: () -> PyEnv, count: Int) {
        self.numEnvs = count
        
        let supportedBatchEnvs = ["pyflyt", "minigrid", "cheetah", "walker", "humanoid"]
        if count > 1 && supportedBatchEnvs.contains(name) {
            let runtime = PythonRuntime.shared
            let batchedModule = runtime.module(named: "batched_env_bridge")
            let Bridge = batchedModule.BatchedEnvBridge
            
            let envType: String
            var envArgs: [PythonObject] = []
            let baseSeed = configuredRunSeed()
            let pySeed = baseSeed.map { PythonObject($0) } ?? PythonObject(Python.None)
            
            switch name {
            case "cheetah":
                envType = "dm_control"
                envArgs = ["cheetah", "run", pySeed]
            case "walker":
                envType = "dm_control"
                envArgs = ["walker", "walk", pySeed]
            case "humanoid":
                envType = "dm_control"
                envArgs = ["humanoid", "stand", pySeed]
            case "pyflyt":
                envType = "pyflyt"
                envArgs = ["PyFlyt/QuadX-Hover-v0", pySeed]
            case "minigrid":
                envType = "minigrid"
                envArgs = ["MiniGrid-Empty-8x8-v0", pySeed]
            default:
                fatalError("Unsupported batched env")
            }
            
            self.batchedEnv = Bridge(envType, envArgs, count)
            self.envs = nil
            self.obsDim = Int(batchedEnv!.get_obs_dim())!
            self.actDim = Int(batchedEnv!.get_act_dim())!
        } else {
            self.envs = (0..<count).map { _ in factory() }
            self.batchedEnv = nil
            self.obsDim = self.envs![0].obsDim
            self.actDim = self.envs![0].actDim
        }
    }
    
    func reset() -> [Float] {
        if let batchedEnv {
            let pyObs = batchedEnv.initial_reset()
            return pythonFlatArray(pyObs)
        }
        
        var obs: [Float] = []
        obs.reserveCapacity(numEnvs * obsDim)
        for env in envs! {
            obs.append(contentsOf: env.reset())
        }
        return obs
    }
    
    func step(_ action: [Float]) -> ([Float], [Float], [Bool]) {
        if let batchedEnv {
            let np = PythonRuntime.shared.numpy
            let npAction = np.array(action, dtype: np.float32)
            let result = batchedEnv.step(npAction)
            let nextObs = pythonFlatArray(result[0])
            let rewards = pythonFlatArray(result[1])
            let dones = pythonFlatArray(result[2]).map { $0 > 0.5 }
            return (nextObs, rewards, dones)
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
        batchedEnv?.close()
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
        if let device = resolvedDevice {
            core = MetalForegroundCore(obsDim: obsDim, actDim: actDim, device: device, seed: seed)
        } else if ProcessInfo.processInfo.environment["OAK_ALLOW_CPU_FALLBACK"] == "1" {
            logWarning("Metal is unavailable in this runtime; using the CPU OaK backend because OAK_ALLOW_CPU_FALLBACK=1.")
            core = CPUForegroundCore(obsDim: obsDim, actDim: actDim, seed: seed)
        } else {
            fatalError("Metal device unavailable. OaK requires Metal in normal runs.")
        }
    }

    var policyName: String {
        "oak"
    }

    var backendName: String {
        core.backendName
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

    init(obsDim: Int, actDim: Int, seed: Int? = configuredRunSeed()) {
        self.obsDim = obsDim
        self.actDim = actDim
        rng = deriveSeed(seed, salt: 0x3001).map(SeededRNG.init)
    }

    func step(obs: [Float], reward: [Float], wallClockBudgetNs _: UInt64, batchSize: Int) -> AgentStepResult {
        let stepStart = currentTimeMilliseconds()
        let safeObs = sanitizeVector(obs)
        let action = (0..<(actDim * batchSize)).map { _ in seededFloat(in: -1...1, rng: rng) }
        
        let telemetry = safeObs.withUnsafeBufferPointer { obsPtr in
            action.withUnsafeBufferPointer { actPtr in
                AgentTelemetry.randomPolicy(
                    rewardInput: reward.first ?? 0,
                    obs: UnsafeBufferPointer(start: obsPtr.baseAddress, count: obsDim),
                    action: UnsafeBufferPointer(start: actPtr.baseAddress, count: actDim),
                    agentStepMs: currentTimeMilliseconds() - stepStart
                )
            }
        }
        
        return AgentStepResult(action: action, telemetry: telemetry)
    }
}

@main struct Main {
    static func main() {
        let cliArguments = Array(CommandLine.arguments.dropFirst())
        if let command = cliArguments.first, command == "doctor-metal" {
            Foundation.exit(runMetalDoctor(arguments: cliArguments.dropFirst()))
        }

        print("OaK - Alberta Plan")
        let availableEnvs: [(String, () -> PyEnv)] = [
            ("core", { CoreSignalEnv() }),
            ("pyflyt", { PyFlytEnv() }),
            ("carla", { CARLAEnv() }),
            ("cheetah", { DMControlEnv(domain: "cheetah", task: "run") }),
            ("walker", { DMControlEnv(domain: "walker", task: "walk") }),
            ("humanoid", { DMControlEnv(domain: "humanoid", task: "stand") }),
            ("minigrid", { MiniGridEnv() }),
        ]

        let requestedEnv = ProcessInfo.processInfo.environment["OAK_ENV"]
            ?? cliArguments.first
            ?? "all"
        let requestedPolicy = (ProcessInfo.processInfo.environment["OAK_POLICY"] ?? "oak").lowercased()
        let runSeed = configuredRunSeed()
        let configuredBudgetNs = configuredStepBudgetNs()
        let runControl = RunControl.resolve()
        let maxEnvironmentSteps = ProcessInfo.processInfo.environment["OAK_MAX_STEPS"].flatMap(Int.init)
        let envsToRun = requestedEnv == "all"
            ? availableEnvs
            : availableEnvs.filter { $0.0 == requestedEnv.lowercased() }

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

        for (name, makeEnv) in envsToRun {
            let env = VectorEnv(name: name, factory: makeEnv, count: numEnvs)
            let runner: any PolicyRunner = requestedPolicy == "random"
                ? RandomPolicyRunner(obsDim: env.obsDim, actDim: env.actDim, seed: runSeed)
                : OaK(obsDim: env.obsDim, actDim: env.actDim, seed: runSeed)
            let seedSuffix = runSeed.map { " seed=\($0)" } ?? ""
            print("\n\(name) (\(type(of: env))) [policy=\(runner.policyName)\(seedSuffix)]")
            let metricsConfiguration = MetricsConfiguration.resolve(envName: name, policyName: runner.policyName)
            let metrics = RunMetricsAggregator(
                envName: name,
                policyName: runner.policyName,
                backend: runner.backendName,
                windowSize: metricsConfiguration.windowSize,
                seed: runSeed
            )
            let metricsSink: JSONLMetricsSink
            do {
                metricsSink = try JSONLMetricsSink(configuration: metricsConfiguration)
                try metricsSink.write(metrics.runStartRecord(control: runControl))
            } catch {
                fatalError("Failed to initialize metrics: \(error)")
            }
            let carlaDebugEnabled = ProcessInfo.processInfo.environment["OAK_CARLA_DEBUG"] == "1"
            if carlaDebugEnabled {
                logWarning("Resetting environment \(name)")
            }
            var obs = env.reset()
            var pendingRewards = Array(repeating: Float(0), count: numEnvs)
            if carlaDebugEnabled {
                logWarning("Environment \(name) reset complete")
            }
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
                    measuredSeconds: measuredSeconds
                )
                do {
                    try metricsSink.write(finalRecord)
                } catch {
                    fatalError("Failed to write metrics run_end record: \(error)")
                }
                if metricsConfiguration.emitStdoutSummaries {
                    print(formatMetricsSummary(finalRecord))
                }
            }
            while true {
                let nextEnvironmentStep = totalEnvironmentSteps + numEnvs
                let isMeasuredStep = nextEnvironmentStep > runControl.warmupSteps
                if carlaDebugEnabled {
                    logWarning("Computing action for environment \(name) step \(nextEnvironmentStep)")
                }
                let totalStepStart = currentTimeMilliseconds()
                if isMeasuredStep && measurementStartMs == nil {
                    measurementStartMs = totalStepStart
                    if runControl.warmupSteps > 0, metricsConfiguration.emitStdoutSummaries {
                        print("[metrics] warmup_complete env=\(name) policy=\(runner.policyName) warmup_steps=\(runControl.warmupSteps)")
                    }
                }
                let agentResult = runner.step(obs: obs, reward: pendingRewards, wallClockBudgetNs: adaptiveBudgetNs, batchSize: numEnvs)
                if carlaDebugEnabled {
                    logWarning("Stepping environment \(name) step \(nextEnvironmentStep)")
                }
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

                    let avgReward = result.1.reduce(0, +) / Float(numEnvs)
                    let anyDone = result.2.contains(true)
                    let metricsRecords = metrics.recordStep(
                        reward: avgReward,
                        done: anyDone,
                        telemetry: agentResult.telemetry,
                        envStepMs: envStepMs,
                        totalStepMs: totalStepMs
                    )
                    do {
                        if metricsConfiguration.emitStepRecords {
                            try metricsSink.write(metricsRecords.step)
                        }
                        if let summaryRecord = metricsRecords.summary {
                            if metricsConfiguration.emitSummaryRecords {
                                try metricsSink.write(summaryRecord)
                            }
                            if metricsConfiguration.emitStdoutSummaries {
                                print(formatMetricsSummary(summaryRecord))
                            }
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
                if carlaDebugEnabled {
                    logWarning("Environment \(name) step \(totalEnvironmentSteps) complete")
                }
                if let maxEnvironmentSteps, totalEnvironmentSteps >= maxEnvironmentSteps {
                    stopReason = "max_env_steps"
                    shouldStop = true
                }

                obs = result.0
                pendingRewards = result.1

                if shouldStop {
                    if metricsConfiguration.emitStdoutSummaries {
                        print("Stopping after \(measuredSteps) measured steps (\(totalEnvironmentSteps) total env steps), reason=\(stopReason)")
                    }
                    if carlaDebugEnabled {
                        logWarning("Stopping environment \(name) because \(stopReason)")
                    }
                    env.close()
                    break
                }
            }
        }
    }
}
