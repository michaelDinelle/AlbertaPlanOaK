import Foundation
import PythonKit

struct EnvironmentSpaceSchema: Codable, Equatable {
    let kind: String
    let flattenedDim: Int
    let shape: [Int]
    let discreteCount: Int?
    let fieldNames: [String]

    static func box(flattenedDim: Int, shape: [Int]) -> EnvironmentSpaceSchema {
        EnvironmentSpaceSchema(
            kind: "box",
            flattenedDim: max(flattenedDim, 1),
            shape: shape,
            discreteCount: nil,
            fieldNames: []
        )
    }

    static func discrete(count: Int) -> EnvironmentSpaceSchema {
        EnvironmentSpaceSchema(
            kind: "discrete",
            flattenedDim: max(count, 1),
            shape: [],
            discreteCount: count,
            fieldNames: []
        )
    }
}

struct EnvironmentSchemaFingerprint: Codable, Equatable {
    let observation: EnvironmentSpaceSchema
    let action: EnvironmentSpaceSchema
    let schemaFingerprint: String

    init(observation: EnvironmentSpaceSchema, action: EnvironmentSpaceSchema) {
        self.observation = observation
        self.action = action
        schemaFingerprint = stableSchemaFingerprint(
            "obs[\(observation.kind):\(observation.flattenedDim):\(observation.shape):\(observation.discreteCount.map(String.init) ?? "nil"):\(observation.fieldNames.joined(separator: ","))]" +
                "|act[\(action.kind):\(action.flattenedDim):\(action.shape):\(action.discreteCount.map(String.init) ?? "nil"):\(action.fieldNames.joined(separator: ","))]"
        )
    }
}

func stableSchemaFingerprint(_ descriptor: String) -> String {
    let offsetBasis: UInt64 = 0xcbf29ce484222325
    let prime: UInt64 = 0x100000001b3
    var hash = offsetBasis
    for byte in descriptor.utf8 {
        hash ^= UInt64(byte)
        hash &*= prime
    }
    return String(format: "%016llx", hash)
}

private func pythonSchemaKeys(_ object: PythonObject) -> [String] {
    let keysObject: PythonObject
    if Bool(Python.hasattr(object, "keys")) == true {
        keysObject = Python.list(object.keys())
    } else {
        keysObject = Python.list(object)
    }
    let length = Int(keysObject.__len__()) ?? 0
    var keys: [String] = []
    keys.reserveCapacity(length)
    for index in 0..<length {
        keys.append(String(keysObject[index]) ?? "\(keys.count)")
    }
    return keys.sorted()
}

func pythonFlattenedObservation(_ object: PythonObject) -> [Float] {
    if Bool(Python.hasattr(object, "keys")) == true {
        let keys = pythonSchemaKeys(object)
        var flattened: [Float] = []
        for key in keys {
            flattened.append(contentsOf: pythonFlattenedObservation(object[PythonObject(key)]))
        }
        return flattened
    }
    return pythonFlatArray(object)
}

func pythonSpaceSchema(_ space: PythonObject, flattenDiscreteAsCount: Bool = false) -> EnvironmentSpaceSchema {
    if Bool(Python.hasattr(space, "spaces")) == true {
        let keys = pythonSchemaKeys(space.spaces)
        let childSchemas = keys.map { key in pythonSpaceSchema(space.spaces[PythonObject(key)], flattenDiscreteAsCount: false) }
        let flattenedDim = childSchemas.reduce(0) { $0 + $1.flattenedDim }
        return EnvironmentSpaceSchema(
            kind: "dict",
            flattenedDim: max(flattenedDim, 1),
            shape: [],
            discreteCount: nil,
            fieldNames: keys
        )
    }

    if Bool(Python.hasattr(space, "n")) == true, let discreteCount = Int(space.n) {
        return EnvironmentSpaceSchema(
            kind: "discrete",
            flattenedDim: flattenDiscreteAsCount ? max(discreteCount, 1) : 1,
            shape: [],
            discreteCount: discreteCount,
            fieldNames: []
        )
    }

    if Bool(Python.hasattr(space, "shape")) == true, let shape = [Int](space.shape) {
        let flattenedDim = max(shape.reduce(1, *), 1)
        let kind = String(space.__class__.__name__)?.lowercased() ?? "box"
        return EnvironmentSpaceSchema(
            kind: kind,
            flattenedDim: flattenedDim,
            shape: shape,
            discreteCount: nil,
            fieldNames: []
        )
    }

    return EnvironmentSpaceSchema(
        kind: String(space.__class__.__name__)?.lowercased() ?? "scalar",
        flattenedDim: 1,
        shape: [],
        discreteCount: nil,
        fieldNames: []
    )
}

func decodeDiscreteActionBatch(_ preferences: [Float], numEnvs: Int, actionCount: Int) -> [Int32] {
    guard numEnvs > 0, actionCount > 0 else {
        return []
    }

    return (0..<numEnvs).map { envIndex in
        let start = envIndex * actionCount
        var bestIndex = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for actionIndex in 0..<actionCount {
            let rawValue = start + actionIndex < preferences.count ? preferences[start + actionIndex] : 0
            let finiteValue = rawValue.isFinite ? rawValue : 0
            if finiteValue > bestValue {
                bestValue = finiteValue
                bestIndex = actionIndex
            }
        }
        return Int32(bestIndex)
    }
}
