import Foundation
import PythonKit

public final class EnvPoolEnv {
    private let env: PythonObject
    public let numEnvs: Int
    public let obsDim: Int
    public let actDim: Int
    public let isDiscrete: Bool
    let schema: EnvironmentSchemaFingerprint
    
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
        let observationSchema = pythonSpaceSchema(obsSpace)
        let actionSchema = pythonSpaceSchema(actSpace, flattenDiscreteAsCount: true)
        schema = EnvironmentSchemaFingerprint(observation: observationSchema, action: actionSchema)
        obsDim = observationSchema.flattenedDim
        actDim = actionSchema.flattenedDim
        isDiscrete = actionSchema.discreteCount != nil
    }

    public func reset() -> [Float] {
        let result = env.reset()
        return flattenObservationBatch(result[0])
    }
    
    public func step(_ action: [Float]) -> (obs: [Float], reward: [Float], done: [Bool]) {
        let np = PythonRuntime.shared.numpy
        let actionFloatArray = action
        
        let npAction: PythonObject
        if isDiscrete {
            let count = schema.action.discreteCount ?? actDim
            let intActions = decodeDiscreteActionBatch(actionFloatArray, numEnvs: numEnvs, actionCount: count)
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
        
        let obs = flattenObservationBatch(pyObs)
        let reward = pythonFlatArray(np.asarray(pyReward, dtype: np.float32))
        let done = pythonFlatArray(np.asarray(pyDone, dtype: np.float32)).map { $0 != 0 }
        
        return (obs, reward, done)
    }

    private func flattenObservationBatch(_ pyObs: PythonObject) -> [Float] {
        if Bool(Python.hasattr(pyObs, "keys")) == true {
            let keys = schema.observation.fieldNames
            var rows = Array(repeating: [Float](), count: numEnvs)
            rows.indices.forEach { rows[$0].reserveCapacity(obsDim) }
            for key in keys {
                let childRows = flattenObservationRows(pyObs[PythonObject(key)])
                precondition(childRows.count == numEnvs, "Observation key \(key) returned \(childRows.count) rows, expected \(numEnvs)")
                for envIndex in 0..<numEnvs {
                    rows[envIndex].append(contentsOf: childRows[envIndex])
                }
            }
            return rows.flatMap { $0 }
        }

        let rows = flattenObservationRows(pyObs)
        return rows.flatMap { $0 }
    }

    private func flattenObservationRows(_ pyObs: PythonObject) -> [[Float]] {
        let np = PythonRuntime.shared.numpy
        let array = np.asarray(pyObs, dtype: np.float32)
        guard let shape = [Int](array.shape), !shape.isEmpty else {
            let scalar = pythonFlatArray(array)
            return Array(repeating: scalar, count: numEnvs)
        }

        let flat = pythonFlatArray(array)
        let rowWidth = shape.dropFirst().reduce(1, *) == 0 ? 1 : max(shape.dropFirst().reduce(1, *), 1)
        precondition(shape[0] == numEnvs, "Observation batch width \(shape[0]) did not match numEnvs \(numEnvs)")
        precondition(flat.count == numEnvs * rowWidth, "Observation flattened to \(flat.count) floats, expected \(numEnvs * rowWidth)")
        return (0..<numEnvs).map { envIndex in
            let start = envIndex * rowWidth
            return Array(flat[start..<(start + rowWidth)])
        }
    }
    public func close() {
        // Any specific close logic for envpool? It usually handles its own lifetime.
    }
}
