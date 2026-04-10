import Foundation
import MLX
import PythonKit

public final class EnvPoolEnv {
    private let env: PythonObject
    public let numEnvs: Int
    public let obsDim: Int
    public let actDim: Int
    public let isDiscrete: Bool
    public let isDictImage: Bool
    
    public init(id: String, numEnvs: Int, seed: Int? = nil) {
        let runtime = PythonRuntime.shared
        runtime.require(modules: ["envpool", "numpy"])
        let envpool = runtime.module(named: "envpool")
        
        self.numEnvs = numEnvs
        
        if let seed = seed {
            self.env = envpool.make(id, "gymnasium", num_envs: numEnvs, seed: seed)
        } else {
            self.env = envpool.make(id, "gymnasium", num_envs: numEnvs)
        }
        
        let obsSpace = self.env.observation_space
        let actSpace = self.env.action_space
        
        var targetObsSpace = obsSpace
        self.isDictImage = Bool(Python.hasattr(obsSpace, "spaces")) == true && obsSpace.spaces["image"] != Python.None
        if isDictImage {
            targetObsSpace = obsSpace.spaces["image"]
        }
        
        if let shape = [Int](targetObsSpace.shape), !shape.isEmpty {
            self.obsDim = shape.reduce(1, *)
        } else {
            self.obsDim = 1
        }
        
        if let shape = [Int](actSpace.shape), !shape.isEmpty {
            self.actDim = shape.reduce(1, *)
            self.isDiscrete = false
        } else {
            // Discrete action space
            if let n = Int(actSpace.n) {
                self.actDim = n
                self.isDiscrete = true
            } else {
                self.actDim = 1
                self.isDiscrete = false
            }
        }
    }
    
    private func processObs(_ pyObs: PythonObject) -> MLXArray {
        let obsData = isDictImage ? pyObs["image"] : pyObs
        let mlxObs = numpyToMLXArray(obsData)
        return isDictImage ? mlxObs / 10.0 : mlxObs
    }

    public func reset() -> MLXArray {
        let result = env.reset()
        return processObs(result[0])
    }
    
    public func step(_ action: MLXArray) -> (obs: MLXArray, reward: MLXArray, done: MLXArray) {
        let np = PythonRuntime.shared.numpy
        let actionFloatArray = action.asArray(Float.self)
        
        let npAction: PythonObject
        if isDiscrete {
            let count = actDim
            let intActions = (0..<numEnvs).map { i -> Int32 in
                let val = actionFloatArray[i * actDim]
                let normalized = max(-1.0, min(1.0, val.isFinite ? val : 0.0))
                let scaled = ((normalized + 1.0) * 0.5) * Float(count)
                return min(max(Int32(scaled), 0), Int32(count - 1))
            }
            npAction = np.array(intActions, dtype: np.int32).reshape(numEnvs)
        } else {
            npAction = np.array(actionFloatArray, dtype: np.float32).reshape(numEnvs, actDim)
        }
        
        let result = env.step(npAction)
        let pyObs = result[0]
        let pyReward = result[1]
        let pyTerminated = result[2]
        let pyTruncated = result[3]
        
        let pyDone = np.logical_or(pyTerminated, pyTruncated)
        
        let obs = processObs(pyObs)
        let reward = numpyToMLXArray(pyReward)
        let done = numpyToMLXArray(pyDone)
        
        return (obs, reward, done)
    }
    
    private func numpyToMLXArray(_ npArray: PythonObject) -> MLXArray {
        let dataDict = npArray.__array_interface__
        let dataTuple = dataDict["data"]
        let dataPointerUInt = UInt(dataTuple[0])!
        
        guard let shapeArray = [Int](npArray.shape) else {
            fatalError("Could not extract shape")
        }
        let shape = shapeArray
        let count = shape.reduce(1, *)
        
        let rawPointer = UnsafeRawPointer(bitPattern: dataPointerUInt)!
        
        let typeStr = String(dataDict["typestr"])!
        if typeStr.contains("f4") {
            let floatPointer = rawPointer.assumingMemoryBound(to: Float.self)
            let buffer = UnsafeBufferPointer(start: floatPointer, count: count)
            return MLXArray(buffer, shape)
        } else if typeStr.contains("f8") {
            // envpool might return float64 for reward
            let doublePointer = rawPointer.assumingMemoryBound(to: Double.self)
            _ = UnsafeBufferPointer(start: doublePointer, count: count)
            // MLXArray doesn't currently support Double natively in all builds, but usually float32 is preferred.
            // If it returns float64, we might need to convert.
            // We can just cast using Numpy to float32 first.
            let float32Array = npArray.astype("float32")
            return numpyToMLXArray(float32Array)
        } else if typeStr.contains("b1") || typeStr.contains("i1") {
            // boolean
            let boolPointer = rawPointer.assumingMemoryBound(to: Bool.self)
            let buffer = UnsafeBufferPointer(start: boolPointer, count: count)
            return MLXArray(buffer, shape)
        } else {
            // Cast to float32 by default
            let float32Array = npArray.astype("float32")
            return numpyToMLXArray(float32Array)
        }
    }
    
    public func close() {
        // Any specific close logic for envpool? It usually handles its own lifetime.
    }
}
