import Foundation
import PythonKit
import Metal

let sys = Python.import("sys")
sys.path.append("/opt/homebrew/lib/python3.11/site-packages")
sys.path.append("/Users/studio/oak/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-macosx-10.15-x86_64.egg")
sys.path.append("/Users/studio/oak/CARLA_0.9.15/PythonAPI/carla")
let gym = Python.import("gymnasium")
let np = Python.import("numpy")

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
    let obsDim: Int
    let actDim: Int

    init() {
        env = gym.make("PyFlyt/QuadX-Hover-v1")
        obsDim = Int(env.observation_space.shape[0])!
        actDim = Int(env.action_space.shape[0])!
    }

    func reset() -> [Float] {
        let result = env.reset()
        return Array(result[0]).map { Float($0)! }
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let result = env.step(np.array(action))
        let obs = Array(result[0]).map { Float($0)! }
        let reward = Float(result[1])!
        let terminated = Bool(result[2])!
        let truncated = Bool(result[3])!
        return (obs, reward, terminated || truncated)
    }
}

final class CARLAEnv: PyEnv {
    private let client: PythonObject
    private let world: PythonObject
    private var vehicle: PythonObject?
    private var camera: PythonObject?
    private var image: [Float]
    let obsDim: Int
    let actDim: Int

    init() {
        let carla = Python.import("carla")
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        obsDim = 84 * 84 * 3
        actDim = 2
        image = Array(repeating: 0, count: obsDim)
        let settings = world.get_settings()
        settings.synchronous_mode = true
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
    }

    func reset() -> [Float] {
        if let v = vehicle {
            v.destroy()
        }
        if let c = camera {
            c.destroy()
        }
        let carla = Python.import("carla")
        let blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
        let spawnPoints = world.get_map().get_spawn_points()
        let spawnPoint = spawnPoints[0]
        vehicle = world.spawn_actor(blueprint, spawnPoint)
        let camBp = world.get_blueprint_library().find("sensor.camera.rgb")
        camBp.set_attribute("image_size_x", "84")
        camBp.set_attribute("image_size_y", "84")
        let transform = carla.Transform(carla.Location(x: 1.5, z: 2.4))
        camera = world.spawn_actor(camBp, transform, attach_to: vehicle)
        camera!.listen(PythonObject { [weak self] (img: PythonObject) in
            let data = np.frombuffer(img.raw_data, dtype: np.uint8)
            self?.image = Array(data).map { Float($0)! / 255.0 }
            return Python.None
        })
        world.tick()
        return image
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        guard let v = vehicle else {
            return (image, 0, true)
        }
        let carla = Python.import("carla")
        let control = carla.VehicleControl()
        control.steer = PythonObject(action[0])
        control.throttle = PythonObject(max(0, action[1]))
        control.brake = PythonObject(max(0, -action[1]))
        v.apply_control(control)
        world.tick()
        let velocity = v.get_velocity()
        let speed = sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        let reward = Float(speed) * 0.01
        return (image, reward, false)
    }

    deinit {
        vehicle?.destroy()
        camera?.destroy()
    }
}

final class DMControlEnv: PyEnv {
    private let env: PythonObject
    let obsDim: Int
    let actDim: Int

    init(domain: String, task: String) {
        let suite = Python.import("dm_control.suite")
        env = suite.load(domain, task)
        let actionSpec = env.action_spec()
        actDim = Int(actionSpec.shape[0])!
        let timeStep = env.reset()
        var dimension = 0
        for key in timeStep.observation.keys() {
            let shape = timeStep.observation[key].shape
            dimension += Int(shape[0])!
        }
        obsDim = dimension
    }

    func reset() -> [Float] {
        let timeStep = env.reset()
        return flatten(timeStep.observation)
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let timeStep = env.step(np.array(action))
        let reward = timeStep.reward
        let rewardValue = Float(reward) ?? 0
        return (flatten(timeStep.observation), rewardValue, false)
    }

    private func flatten(_ observation: PythonObject) -> [Float] {
        var result: [Float] = []
        for key in observation.keys() {
            let array = Array(observation[key]).map { Float($0)! }
            result.append(contentsOf: array)
        }
        return result
    }
}

final class MiniGridEnv: PyEnv {
    private let env: PythonObject
    let obsDim: Int
    let actDim: Int

    init() {
        env = gym.make("MiniGrid-Empty-8x8-v0")
        obsDim = 147
        actDim = 7
    }

    func reset() -> [Float] {
        let result = env.reset()
        return encode(result[0])
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let actionIndex = Int(action[0] * Float(actDim)) % actDim
        let result = env.step(actionIndex)
        let obs = encode(result[0])
        let reward = Float(result[1])!
        let terminated = Bool(result[2])!
        let truncated = Bool(result[3])!
        return (obs, reward, terminated || truncated)
    }

    private func encode(_ observation: PythonObject) -> [Float] {
        let image = observation["image"]
        var result: [Float] = []
        for i in 0..<7 {
            for j in 0..<7 {
                for k in 0..<3 {
                    let value = Float(image[i][j][k])!
                    result.append(value / 10.0)
                }
            }
        }
        return result
    }
}

final class HabitatEnv: PyEnv {
    private let env: PythonObject
    let obsDim: Int
    let actDim: Int

    init() {
        let habitat = Python.import("habitat")
        let config = habitat.get_config("configs/tasks/pointnav.yaml")
        env = habitat.Env(config: config)
        obsDim = 256 * 256 * 3
        actDim = 4
    }

    func reset() -> [Float] {
        let observation = env.reset()
        let rgb = observation["rgb"]
        return Array(rgb).map { Float($0)! / 255.0 }
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let actionIndex = Int(action[0] * 4) % 4
        let actions = [0, 1, 2, 3]
        let result = env.step(actions[actionIndex])
        let rgb = result["rgb"]
        let obs = Array(rgb).map { Float($0)! / 255.0 }
        let reward = Float(result["reward"])!
        let done = Bool(result["done"])!
        return (obs, reward, done)
    }
}

final class FinRLEnv: PyEnv {
    private let env: PythonObject
    let obsDim: Int
    let actDim: Int

    init() {
        env = gym.make("FinRL-v0")
        obsDim = Int(env.observation_space.shape[0])!
        actDim = Int(env.action_space.shape[0])!
    }

    func reset() -> [Float] {
        let result = env.reset()
        return Array(result[0]).map { Float($0)! }
    }

    func step(_ action: [Float]) -> ([Float], Float, Bool) {
        let result = env.step(np.array(action))
        let obs = Array(result[0]).map { Float($0)! }
        let reward = Float(result[1])!
        let done = Bool(result[2])!
        return (obs, reward, done)
    }
}

final class OaK {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipelines: [String: MTLComputePipelineState]
    private var perceptionW: MTLBuffer
    private var perceptionAlpha: MTLBuffer
    private var policyW: MTLBuffer
    private var valueTheta: MTLBuffer
    private var valueRho: MTLBuffer
    private var valueWealth: MTLBuffer
    private var valueGSum: MTLBuffer
    private var modelW: MTLBuffer
    private var modelProj: MTLBuffer
    private let obsDim: Int
    private let actDim: Int
    private let stateDim: Int
    private var lastState: [Float]
    private var lastAction: [Float]
    private var stepCount: Int

    init(obsDim: Int, actDim: Int) {
        self.obsDim = obsDim
        self.actDim = actDim
        self.stateDim = obsDim
        self.stepCount = 0

        device = MTLCreateSystemDefaultDevice()!
        queue = device.makeCommandQueue()!
        let library = try! device.makeDefaultLibrary(bundle: Bundle.main)

        pipelines = [
            "fire": try! device.makeComputePipelineState(function: library.makeFunction(name: "fire")!),
            "perception_forward": try! device.makeComputePipelineState(function: library.makeFunction(name: "perception_forward")!),
            "perception_update": try! device.makeComputePipelineState(function: library.makeFunction(name: "perception_update")!),
            "value_update": try! device.makeComputePipelineState(function: library.makeFunction(name: "value_update")!),
            "policy_act": try! device.makeComputePipelineState(function: library.makeFunction(name: "policy_act")!),
            "policy_update": try! device.makeComputePipelineState(function: library.makeFunction(name: "policy_update")!),
            "model_predict": try! device.makeComputePipelineState(function: library.makeFunction(name: "model_predict")!),
            "model_update": try! device.makeComputePipelineState(function: library.makeFunction(name: "model_update")!),
        ]

        let perceptionSize = stateDim * (obsDim + actDim)
        perceptionW = device.makeBuffer(length: perceptionSize * 4, options:.storageModeShared)!
        perceptionAlpha = device.makeBuffer(length: perceptionSize * 4, options:.storageModeShared)!
        let perceptionPtr = perceptionW.contents().bindMemory(to: Float.self, capacity: perceptionSize)
        let alphaPtr = perceptionAlpha.contents().bindMemory(to: Float.self, capacity: perceptionSize)
        let perceptionScale = 1 / sqrt(Float(obsDim + actDim))
        for i in 0..<perceptionSize {
            perceptionPtr[i] = Float.random(in: -perceptionScale...perceptionScale)
            alphaPtr[i] = 1 / Float(perceptionSize)
        }

        let policySize = actDim * stateDim
        policyW = device.makeBuffer(length: policySize * 4, options:.storageModeShared)!
        let policyPtr = policyW.contents().bindMemory(to: Float.self, capacity: policySize)
        let policyScale = 1 / sqrt(Float(stateDim))
        for i in 0..<policySize {
            policyPtr[i] = Float.random(in: -policyScale...policyScale)
        }

        valueTheta = device.makeBuffer(length: stateDim * 4, options:.storageModeShared)!
        valueRho = device.makeBuffer(length: 4, options:.storageModeShared)!
        valueWealth = device.makeBuffer(length: 4, options:.storageModeShared)!
        valueGSum = device.makeBuffer(length: 4, options:.storageModeShared)!
        valueWealth.contents().bindMemory(to: Float.self, capacity: 1)[0] = 1

        let initialPdim = 1
        modelW = device.makeBuffer(length: stateDim * initialPdim * 4, options:.storageModeShared)!
        modelProj = device.makeBuffer(length: (stateDim + actDim) * initialPdim * 4, options:.storageModeShared)!
        let modelPtr = modelW.contents().bindMemory(to: Float.self, capacity: stateDim * initialPdim)
        let projPtr = modelProj.contents().bindMemory(to: Float.self, capacity: (stateDim + actDim) * initialPdim)
        let modelScale = 1 / sqrt(Float(initialPdim))
        for i in 0..<(stateDim * initialPdim) {
            modelPtr[i] = Float.random(in: -modelScale...modelScale)
        }
        let projScale = 1 / sqrt(Float(initialPdim))
        for i in 0..<((stateDim + actDim) * initialPdim) {
            projPtr[i] = Float.random(in: -projScale...projScale)
        }

        lastState = Array(repeating: 0, count: stateDim)
        lastAction = Array(repeating: 0, count: actDim)
    }

    func step(obs: [Float], reward: Float) -> [Float] {
        stepCount += 1
        let input = obs + lastAction
        let inputBuf = device.makeBuffer(bytes: input, length: input.count * 4, options:.storageModeShared)!
        let stateBuf = device.makeBuffer(length: stateDim * 4, options:.storageModeShared)!
        let cmd1 = queue.makeCommandBuffer()!
        let enc1 = cmd1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(pipelines["perception_forward"]!)
        enc1.setBuffer(perceptionW, offset: 0, index: 0)
        enc1.setBuffer(inputBuf, offset: 0, index: 1)
        enc1.setBuffer(stateBuf, offset: 0, index: 2)
        var outDim = UInt32(stateDim)
        var inDim = UInt32(input.count)
        enc1.setBytes(&outDim, length: 4, index: 3)
        enc1.setBytes(&inDim, length: 4, index: 4)
        let threadsPerGroup = min(stateDim, 256)
        enc1.dispatchThreads(MTLSize(width: stateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        enc1.endEncoding()
        cmd1.commit()
        cmd1.waitUntilCompleted()
        let statePtr = stateBuf.contents().bindMemory(to: Float.self, capacity: stateDim)
        let state = Array(UnsafeBufferPointer(start: statePtr, count: stateDim))

        let sBuf = device.makeBuffer(bytes: lastState, length: lastState.count * 4, options:.storageModeShared)!
        let sNextBuf = device.makeBuffer(bytes: state, length: state.count * 4, options:.storageModeShared)!
        let rewardBuf = device.makeBuffer(bytes: [reward], length: 4, options:.storageModeShared)!
        let cmd2 = queue.makeCommandBuffer()!
        let enc2 = cmd2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(pipelines["value_update"]!)
        enc2.setBuffer(valueTheta, offset: 0, index: 0)
        enc2.setBuffer(sBuf, offset: 0, index: 1)
        enc2.setBuffer(sNextBuf, offset: 0, index: 2)
        enc2.setBuffer(valueRho, offset: 0, index: 3)
        enc2.setBuffer(valueWealth, offset: 0, index: 4)
        enc2.setBuffer(valueGSum, offset: 0, index: 5)
        enc2.setBuffer(rewardBuf, offset: 0, index: 6)
        var dim = UInt32(stateDim)
        enc2.setBytes(&dim, length: 4, index: 7)
        enc2.dispatchThreads(MTLSize(width: stateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()

        let actionBuf = device.makeBuffer(length: actDim * 4, options:.storageModeShared)!
        let cmd3 = queue.makeCommandBuffer()!
        let enc3 = cmd3.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(pipelines["policy_act"]!)
        enc3.setBuffer(policyW, offset: 0, index: 0)
        enc3.setBuffer(stateBuf, offset: 0, index: 1)
        enc3.setBuffer(actionBuf, offset: 0, index: 2)
        var aDim = UInt32(actDim)
        var sDim = UInt32(stateDim)
        enc3.setBytes(&aDim, length: 4, index: 3)
        enc3.setBytes(&sDim, length: 4, index: 4)
        let policyThreads = min(actDim, 256)
        enc3.dispatchThreads(MTLSize(width: actDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: policyThreads, height: 1, depth: 1))
        enc3.endEncoding()
        cmd3.commit()
        cmd3.waitUntilCompleted()
        let actionPtr = actionBuf.contents().bindMemory(to: Float.self, capacity: actDim)
        let action = Array(UnsafeBufferPointer(start: actionPtr, count: actDim))

        let lastActionBuf = device.makeBuffer(bytes: lastAction, length: lastAction.count * 4, options:.storageModeShared)!
        let phiBuf = device.makeBuffer(length: 1 * 4, options:.storageModeShared)!
        let nextBuf = device.makeBuffer(length: stateDim * 4, options:.storageModeShared)!
        let errBuf = device.makeBuffer(length: stateDim * 4, options:.storageModeShared)!
        let errPtr = errBuf.contents().bindMemory(to: Float.self, capacity: stateDim)
        for i in 0..<stateDim {
            errPtr[i] = state[i] - lastState[i]
        }

        let cmd4 = queue.makeCommandBuffer()!
        let enc4 = cmd4.makeComputeCommandEncoder()!
        enc4.setComputePipelineState(pipelines["model_update"]!)
        enc4.setBuffer(modelW, offset: 0, index: 0)
        enc4.setBuffer(phiBuf, offset: 0, index: 1)
        enc4.setBuffer(errBuf, offset: 0, index: 2)
        enc4.setBytes(&sDim, length: 4, index: 3)
        var pDim = UInt32(1)
        enc4.setBytes(&pDim, length: 4, index: 4)
        enc4.dispatchThreads(MTLSize(width: stateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc4.endEncoding()
        cmd4.commit()
        cmd4.waitUntilCompleted()

        let cmd5 = queue.makeCommandBuffer()!
        let enc5 = cmd5.makeComputeCommandEncoder()!
        enc5.setComputePipelineState(pipelines["fire"]!)
        enc5.setBuffer(perceptionW, offset: 0, index: 0)
        var n = UInt32(stateDim)
        var d = UInt32(obsDim + actDim)
        enc5.setBytes(&n, length: 4, index: 1)
        enc5.setBytes(&d, length: 4, index: 2)
        let fireThreads = MTLSize(width: min(Int(n), 16), height: min(Int(d), 16), depth: 1)
        enc5.dispatchThreads(MTLSize(width: Int(n), height: Int(d), depth: 1), threadsPerThreadgroup: fireThreads)
        enc5.endEncoding()
        cmd5.commit()
        cmd5.waitUntilCompleted()

        lastState = state
        lastAction = action
        return action
    }
}

@main struct Main {
    static func main() {
        print("OaK - Alberta Plan")
        let envs: [PyEnv] = [
            PyFlytEnv(),
            CARLAEnv(),
            DMControlEnv(domain: "cheetah", task: "run"),
            DMControlEnv(domain: "walker", task: "walk"),
            DMControlEnv(domain: "humanoid", task: "stand"),
            MiniGridEnv(),
            HabitatEnv(),
            FinRLEnv(),
        ]
        for env in envs {
            print("\n\(type(of: env))")
            let agent = OaK(obsDim: env.obsDim, actDim: env.actDim)
            var obs = env.reset()
            while true {
                let action = agent.step(obs: obs, reward: 0)
                let result = env.step(action)
                obs = result.0
                if result.2 {
                    obs = env.reset()
                }
            }
        }
    }
}