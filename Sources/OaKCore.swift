import Darwin
import Foundation
import Metal
import MLX
import MLXNN

struct StepInput {
    let obs: [Float] // Flattened batchSize * obsDim
    let prevAction: [Float] // Flattened batchSize * actDim
    let prevReward: [Float] // batchSize
    let batchSize: Int
    let wallClockBudgetNs: UInt64
}

struct StepOutput {
    let action: [Float] // Flattened batchSize * actDim
    let telemetry: AgentTelemetry
    let snapshotToken: Int
}

protocol ComputeFuture {
    func wait()
}

class CPUComputeFuture: ComputeFuture {
    func wait() {}
}

protocol ForegroundCore {
    var backendName: String { get }
    func step(input: StepInput) -> StepOutput
}

private func oakCurrentTimeNanoseconds() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

private func oakCurrentTimeMilliseconds() -> Double {
    Double(oakCurrentTimeNanoseconds()) / 1_000_000
}

private func oakDeriveSeed(_ seed: Int?, salt: UInt64) -> UInt64? {
    guard let seed else { return nil }
    var value = UInt64(bitPattern: Int64(seed)) &+ salt &+ 0x9E3779B97F4A7C15
    value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
    value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
    return value ^ (value >> 31)
}

private func oakSafeScalar(_ value: Float, clamp range: ClosedRange<Float>? = nil) -> Float {
    let finiteValue = value.isFinite ? value : 0
    guard let range else { return finiteValue }
    return min(max(finiteValue, range.lowerBound), range.upperBound)
}

private func oakDot(_ lhs: UnsafeBufferPointer<Float>, _ rhs: UnsafeBufferPointer<Float>, count: Int) -> Float {
    guard count > 0 else { return 0 }
    var sum: Float = 0
    for index in 0..<count {
        sum += lhs[index] * rhs[index]
    }
    return sum
}

private func oakDotRow(_ storage: SharedFloatStorage, row: Int, stride: Int, vector: UnsafeBufferPointer<Float>, count: Int) -> Float {
    guard count > 0 else { return 0 }
    let rowOffset = row * stride
    var sum: Float = 0
    for index in 0..<count {
        sum += storage[rowOffset + index] * vector[index]
    }
    return sum
}

private func oakNormalizeRow(_ storage: SharedFloatStorage, row: Int, stride: Int, count: Int) {
    guard count > 0 else { return }
    let rowOffset = row * stride
    var norm: Double = 0
    for index in 0..<count {
        let value = Double(storage[rowOffset + index])
        norm += value * value
    }
    guard norm > Double(Float.ulpOfOne) else { return }
    let invNorm = Float(1 / sqrt(norm))
    for index in 0..<count {
        storage[rowOffset + index] *= invNorm
    }
}

final class CoreSeededRNG {
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
    func nextFloat() -> Float {
        let unit = Double(nextUInt64() >> 11) / Double(1 << 53)
        return Float(unit)
    }
    func nextSignedFloat(scale: Float) -> Float {
        (nextFloat() * 2 - 1) * scale
    }
}

final class SharedFloatStorage {
    let count: Int
    let pointer: UnsafeMutablePointer<Float>
    let metalBuffer: MTLBuffer?

    init(count: Int, device: MTLDevice? = nil) {
        self.count = count
        let size = count * MemoryLayout<Float>.stride
        if let device {
            let buffer = device.makeBuffer(length: size, options: [.storageModeShared])!
            metalBuffer = buffer
            pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
            pointer.initialize(repeating: 0, count: count)
        } else {
            metalBuffer = nil
            pointer = UnsafeMutablePointer<Float>.allocate(capacity: count)
            pointer.initialize(repeating: 0, count: count)
        }
    }

    deinit {
        pointer.deinitialize(count: count)
        if metalBuffer == nil {
            pointer.deallocate()
        }
    }

    subscript(index: Int) -> Float {
        get { pointer[index] }
        set { pointer[index] = newValue }
    }

    func zero() {
        memset(pointer, 0, count * MemoryLayout<Float>.stride)
    }

    func bufferPointer() -> UnsafeMutableBufferPointer<Float> {
        UnsafeMutableBufferPointer(start: pointer, count: count)
    }

    func immutableBufferPointer() -> UnsafeBufferPointer<Float> {
        UnsafeBufferPointer(start: pointer, count: count)
    }
    
    func immutableBufferPointer(offset: Int, length: Int) -> UnsafeBufferPointer<Float> {
        UnsafeBufferPointer(start: pointer.advanced(by: offset), count: length)
    }

    func copy(from values: [Float], count: Int? = nil) {
        let targetCount = min(count ?? values.count, self.count)
        values.withUnsafeBufferPointer { source in
            guard let sourceBase = source.baseAddress else { return }
            pointer.update(from: sourceBase, count: targetCount)
        }
        if targetCount < self.count {
            memset(pointer.advanced(by: targetCount), 0, (self.count - targetCount) * MemoryLayout<Float>.stride)
        }
    }

    func snapshot(offset: Int, prefix: Int) -> [Float] {
        let safePrefix = min(max(prefix, 0), count - offset)
        return Array(UnsafeBufferPointer(start: pointer.advanced(by: offset), count: safePrefix))
    }
}

struct CapacityDecision {
    let allowedRank: Int
    let budgetWorkUnits: Double
}

final class AdaptivePerception {
    let signalDim: Int
    let capacityLimit: Int
    let basis: SharedFloatStorage
    private(set) var activeRank: Int
    private var featureEnergy: [Double]
    private var featureUtility: [Double]

    init(signalDim: Int, capacityLimit: Int, device: MTLDevice?, rng: CoreSeededRNG) {
        self.signalDim = signalDim
        self.capacityLimit = capacityLimit
        basis = SharedFloatStorage(count: max(signalDim * capacityLimit, 1), device: device)
        activeRank = min(max(1, capacityLimit), capacityLimit)
        featureEnergy = Array(repeating: 0, count: capacityLimit)
        featureUtility = Array(repeating: 0, count: capacityLimit)

        let scale = Float(1 / sqrt(Double(max(signalDim, 1))))
        for row in 0..<capacityLimit {
            let rowOffset = row * signalDim
            for column in 0..<signalDim {
                basis[rowOffset + column] = rng.nextSignedFloat(scale: scale)
            }
            oakNormalizeRow(basis, row: row, stride: signalDim, count: signalDim)
            featureEnergy[row] = 1
            featureUtility[row] = 1 / Double(row + 1)
        }
    }

    func encode(
        signal: SharedFloatStorage, signalOffset: Int,
        state: SharedFloatStorage, stateOffset: Int,
        reconstruction: SharedFloatStorage, reconOffset: Int,
        residual: SharedFloatStorage, resOffset: Int,
        allowedRank: Int
    ) -> (residualEnergy: Double, totalEnergy: Double) {
        let boundedRank = max(1, min(allowedRank, capacityLimit))
        if activeRank > boundedRank {
            compact(to: boundedRank)
        }

        let signalBuffer = signal.immutableBufferPointer(offset: signalOffset, length: signalDim)

        for row in 0..<activeRank {
            let projection = tanh(oakDotRow(basis, row: row, stride: signalDim, vector: signalBuffer, count: signalDim))
            state[stateOffset + row] = oakSafeScalar(projection)
            featureEnergy[row] += Double(projection * projection)
            let rowOffset = row * signalDim
            for column in 0..<signalDim {
                reconstruction[reconOffset + column] += projection * basis[rowOffset + column]
            }
        }
        if activeRank < capacityLimit {
            for row in activeRank..<capacityLimit {
                state[stateOffset + row] = 0
            }
        }

        var totalEnergy = 0.0
        var residualEnergy = 0.0
        for column in 0..<signalDim {
            let residualValue = signal[signalOffset + column] - reconstruction[reconOffset + column]
            residual[resOffset + column] = residualValue
            let signalValue = Double(signal[signalOffset + column])
            let residualValueDouble = Double(residualValue)
            totalEnergy += signalValue * signalValue
            residualEnergy += residualValueDouble * residualValueDouble
        }

        updateBasis(signal: signalBuffer, state: state, stateOffset: stateOffset)
        adaptBasisSet(residual: residual, resOffset: resOffset, residualEnergy: residualEnergy, allowedRank: boundedRank)
        updateFeatureUtilities(state: state, stateOffset: stateOffset)
        return (residualEnergy, totalEnergy)
    }

    func currentActiveRank(budget: CapacityDecision) -> Int {
        let energySum = featureEnergy.prefix(activeRank).reduce(0, +)
        let energySquaredSum = featureEnergy.prefix(activeRank).reduce(0) { $0 + $1 * $1 }
        let effectiveRank = energySquaredSum > 0 ? Int(ceil((energySum * energySum) / energySquaredSum)) : 1
        return max(1, min(budget.allowedRank, max(effectiveRank, 1)))
    }

    func recordUtility(index: Int, value: Double) {
        if featureUtility.indices.contains(index) {
            featureUtility[index] += abs(value)
        }
    }

    private func updateBasis(signal: UnsafeBufferPointer<Float>, state: SharedFloatStorage, stateOffset: Int) {
        for row in 0..<activeRank {
            let rate = Float(1 / max(featureEnergy[row], Double(Float.ulpOfOne)))
            let activation = state[stateOffset + row]
            let rowOffset = row * signalDim
            for column in 0..<signalDim {
                let current = basis[rowOffset + column]
                let correction = signal[column] - activation * current
                basis[rowOffset + column] = current + rate * activation * correction
            }
            oakNormalizeRow(basis, row: row, stride: signalDim, count: signalDim)
        }
    }

    private func adaptBasisSet(residual: SharedFloatStorage, resOffset: Int, residualEnergy: Double, allowedRank: Int) {
        guard allowedRank > 0 else { return }
        let meanEnergy = featureEnergy.prefix(max(activeRank, 1)).reduce(0, +) / Double(max(activeRank, 1))
        guard residualEnergy > meanEnergy else { return }

        if activeRank < allowedRank {
            writeNormalizedResidual(residual, resOffset: resOffset, into: activeRank)
            featureEnergy[activeRank] = residualEnergy
            featureUtility[activeRank] = residualEnergy
            activeRank += 1
            return
        }

        guard let weakestIndex = (0..<activeRank).min(by: { featureUtility[$0] < featureUtility[$1] }) else { return }
        writeNormalizedResidual(residual, resOffset: resOffset, into: weakestIndex)
        featureEnergy[weakestIndex] = residualEnergy
        featureUtility[weakestIndex] = residualEnergy
    }

    private func writeNormalizedResidual(_ residual: SharedFloatStorage, resOffset: Int, into row: Int) {
        let rowOffset = row * signalDim
        for column in 0..<signalDim {
            basis[rowOffset + column] = residual[resOffset + column]
        }
        oakNormalizeRow(basis, row: row, stride: signalDim, count: signalDim)
    }

    private func compact(to targetRank: Int) {
        guard targetRank < activeRank else { return }
        let ranked = (0..<activeRank).sorted { featureUtility[$0] + featureEnergy[$0] > featureUtility[$1] + featureEnergy[$1] }
        let keep = Array(ranked.prefix(targetRank))
        let scratch = basis.snapshot(offset: 0, prefix: basis.count)
        for (newRow, oldRow) in keep.enumerated() {
            let srcOffset = oldRow * signalDim
            let dstOffset = newRow * signalDim
            for column in 0..<signalDim {
                basis[dstOffset + column] = scratch[srcOffset + column]
            }
            featureEnergy[newRow] = featureEnergy[oldRow]
            featureUtility[newRow] = featureUtility[oldRow]
        }
        for row in targetRank..<capacityLimit {
            let rowOffset = row * signalDim
            for column in 0..<signalDim { basis[rowOffset + column] = 0 }
            featureEnergy[row] = 0
            featureUtility[row] = 0
        }
        activeRank = max(targetRank, 1)
    }

    private func updateFeatureUtilities(state: SharedFloatStorage, stateOffset: Int) {
        for row in 0..<activeRank {
            featureUtility[row] += Double(abs(state[stateOffset + row]))
        }
    }
}

final class ReactiveControl {
    let actDim: Int
    let capacityLimit: Int
    let policy: SharedFloatStorage
    let valueWeights: SharedFloatStorage
    var rewardRate: Float

    init(actDim: Int, capacityLimit: Int, device: MTLDevice?, rng: CoreSeededRNG) {
        self.actDim = actDim
        self.capacityLimit = capacityLimit
        policy = SharedFloatStorage(count: max(actDim * capacityLimit, 1), device: device)
        valueWeights = SharedFloatStorage(count: max(capacityLimit, 1), device: nil)
        rewardRate = 0
        let scale = Float(1 / sqrt(Double(max(capacityLimit, 1))))
        for row in 0..<actDim {
            let rowOffset = row * capacityLimit
            for column in 0..<capacityLimit {
                policy[rowOffset + column] = rng.nextSignedFloat(scale: scale)
            }
        }
    }

    func actCPU(state: SharedFloatStorage, stateOffset: Int, plannedBias: SharedFloatStorage, plannedOffset: Int, activeRank: Int, action: SharedFloatStorage, actOffset: Int) {
        let stateBuffer = state.immutableBufferPointer(offset: stateOffset, length: capacityLimit)
        for row in 0..<actDim {
            var sum = plannedBias[plannedOffset + row]
            let rowOffset = row * capacityLimit
            for column in 0..<activeRank {
                sum += policy[rowOffset + column] * stateBuffer[column]
            }
            action[actOffset + row] = oakSafeScalar(tanh(sum), clamp: -1...1)
        }
    }

    func evaluate(state: SharedFloatStorage, stateOffset: Int, activeRank: Int) -> Float {
        let stateBuffer = state.immutableBufferPointer(offset: stateOffset, length: capacityLimit)
        let valueBuffer = valueWeights.immutableBufferPointer()
        return oakDot(stateBuffer, valueBuffer, count: activeRank)
    }

    func update(prevState: SharedFloatStorage, prevOffset: Int, state: SharedFloatStorage, stateOffset: Int, prevAction: SharedFloatStorage, prevActOffset: Int, reward: Float, activeRank: Int) -> Float {
        guard activeRank > 0 else { return 0 }
        let prevValue = evaluate(state: prevState, stateOffset: prevOffset, activeRank: activeRank)
        let nextValue = evaluate(state: state, stateOffset: stateOffset, activeRank: activeRank)
        let prevStateBuffer = prevState.immutableBufferPointer(offset: prevOffset, length: capacityLimit)
        let stateNorm = max(oakDot(prevStateBuffer, prevStateBuffer, count: activeRank), Float.ulpOfOne)
        let step = 1 / (1 + stateNorm)
        let delta = reward - rewardRate + nextValue - prevValue
        rewardRate += step * delta

        for index in 0..<activeRank {
            valueWeights[index] += step * delta * prevState[prevOffset + index]
        }

        for row in 0..<actDim {
            let actionValue = oakSafeScalar(prevAction[prevActOffset + row], clamp: -1...1)
            let sensitivity = 1 - actionValue * actionValue
            let rowOffset = row * capacityLimit
            for column in 0..<activeRank {
                policy[rowOffset + column] += step * delta * sensitivity * prevState[prevOffset + column]
            }
        }
        return delta
    }
}

final class PredictiveKnowledge: Module {
    let capacityLimit: Int
    @ModuleInfo var weights: MLXArray
    @ModuleInfo var alpha: MLXArray
    @ModuleInfo var h: MLXArray
    @ModuleInfo var v: MLXArray
    private(set) var utility: [Double]

    let mu: Float
    let tau: Float

    private var compiledUpdate: (([MLXArray]) -> [MLXArray])?
    private var compiledRank: Int = -1

    init(capacityLimit: Int) {
        self.capacityLimit = capacityLimit
        self.mu = 1.0 / Float(max(capacityLimit, 1))
        self.tau = Float(capacityLimit * capacityLimit)
        
        let numWeights = max(capacityLimit, 1)
        self.weights = MLXArray.zeros([numWeights, numWeights])
        self.alpha = MLXArray.zeros([numWeights, numWeights]) + 0.1
        self.h = MLXArray.zeros([numWeights, numWeights])
        self.v = MLXArray.zeros([numWeights, numWeights])
        
        self.utility = Array(repeating: 0, count: capacityLimit)
        super.init()
    }

    func updateBatched(prevState: SharedFloatStorage, prevIdx: Int, state: SharedFloatStorage, stateIdx: Int, batchSize: Int, capacityLimit: Int, activeRank: Int) -> Double {
        guard activeRank > 0 else { return 0 }
        
        let pOffset = prevIdx * batchSize * capacityLimit
        let sOffset = stateIdx * batchSize * capacityLimit
        
        let xArray = Array(prevState.immutableBufferPointer(offset: pOffset, length: batchSize * capacityLimit))
        let targetArray = Array(state.immutableBufferPointer(offset: sOffset, length: batchSize * capacityLimit))
        
        let xFull = MLXArray(xArray, [batchSize, capacityLimit])
        let targetFull = MLXArray(targetArray, [batchSize, capacityLimit])
        
        let x = xFull[0..<batchSize, 0..<activeRank]
        let target = targetFull[0..<batchSize, 0..<activeRank]
        
        if compiledUpdate == nil || compiledRank != activeRank {
            compiledUpdate = MLX.compile(inputs: [self], outputs: [self], shapeless: false) { [weak self] arrays in
                guard let self = self else { return [] }
                let x = arrays[0]
                let target = arrays[1]
                let batchSize = x.shape[0]
                let activeRank = x.shape[1]
                
                let w = self.weights[0..<activeRank, 0..<activeRank]
                let a = self.alpha[0..<activeRank, 0..<activeRank]
                let h_trace = self.h[0..<activeRank, 0..<activeRank]
                let v_trace = self.v[0..<activeRank, 0..<activeRank]
                
                let gradFn = valueAndGrad(model: self) { (model: PredictiveKnowledge, input: MLXArray, tgt: MLXArray) -> MLXArray in
                    let activeW = model.weights[0..<activeRank, 0..<activeRank]
                    // input is [batchSize, activeRank], activeW is [activeRank, activeRank]
                    // pred should be [batchSize, activeRank] -> matmul(input, activeW.T)
                    let pred = matmul(input, activeW.T)
                    return (0.5 / Float(batchSize)) * sum(square(pred - tgt))
                }
                let (_, grads) = gradFn(self, x, target)
                let grad = grads[unwrapping: "weights"]![0..<activeRank, 0..<activeRank]
                
                let pred = matmul(x, w.T)
                let error = pred - target
                
                let deltaXh = -grad * h_trace
                let absDeltaXh = abs(deltaXh)
                
                // x2 is averaged over the batch
                let x2 = mean(square(x), axes: [0]).reshaped([1, activeRank])
                
                let vDiff = absDeltaXh - v_trace
                let vUpdate = (1.0 / self.tau) * a * x2 * vDiff
                let newV = maximum(absDeltaXh, v_trace + vUpdate)
                let vZeroMask = newV .== 0.0
                let safeV = MLX.where(vZeroMask, MLXArray(1.0), newV)
                let muFactor = MLXArray(self.mu) * deltaXh / safeV
                var newA = a * exp(muFactor)
                newA = MLX.where(vZeroMask, a, newA)
                
                let currentM = sum(newA * x2, axes: [1], keepDims: true)
                let m_eff = maximum(currentM, MLXArray(1.0))
                
                newA = newA / m_eff
                
                let gradStep = newA * grad
                let newWeights = w - gradStep
                let hDecay = h_trace * (1.0 - newA * x2)
                let newH = hDecay - gradStep
                
                self.alpha[0..<activeRank, 0..<activeRank] = newA
                self.v[0..<activeRank, 0..<activeRank] = newV
                self.h[0..<activeRank, 0..<activeRank] = newH
                self.weights[0..<activeRank, 0..<activeRank] = newWeights
                
                return [error]
            }
            compiledRank = activeRank
        }
        
        let outputs = compiledUpdate!([x, target])
        let error = outputs[0]
        
        let totalError = Double(sum(abs(error)).item(Float.self))
        
        let errorMean = mean(abs(error), axes: [0]).asArray(Float.self)
        for i in 0..<activeRank {
            utility[i] += Double(errorMean[i])
        }
        
        eval(self, self.alpha, self.v, self.h, self.weights)
        
        return totalError
    }
}

final class WorldModel: Module {
    let capacityLimit: Int
    let actDim: Int
    let inputWidth: Int
    @ModuleInfo var weights: MLXArray
    @ModuleInfo var alpha: MLXArray
    @ModuleInfo var h: MLXArray
    @ModuleInfo var v: MLXArray
    private let inputStorage: SharedFloatStorage
    private let predictionStorage: SharedFloatStorage

    let mu: Float
    let tau: Float

    private var compiledUpdate: (([MLXArray]) -> [MLXArray])?
    private var compiledPredict: (([MLXArray]) -> [MLXArray])?
    private var compiledRank: Int = -1

    init(capacityLimit: Int, actDim: Int) {
        self.capacityLimit = capacityLimit
        self.actDim = actDim
        inputWidth = capacityLimit + actDim + 1
        
        self.mu = 1.0 / Float(max(capacityLimit, 1))
        self.tau = Float(capacityLimit * capacityLimit)
        
        let numWeights = max(capacityLimit + 1, 1)
        self.weights = MLXArray.zeros([numWeights, inputWidth])
        self.alpha = MLXArray.zeros([numWeights, inputWidth]) + 0.1
        self.h = MLXArray.zeros([numWeights, inputWidth])
        self.v = MLXArray.zeros([numWeights, inputWidth])
        
        inputStorage = SharedFloatStorage(count: max(inputWidth, 1), device: nil)
        predictionStorage = SharedFloatStorage(count: max(capacityLimit + 1, 1), device: nil)
        super.init()
    }

    func updateBatched(prevState: SharedFloatStorage, prevIdx: Int, state: SharedFloatStorage, stateIdx: Int, prevAction: SharedFloatStorage, activeIdx: Int, reward: SharedFloatStorage, batchSize: Int, capacityLimit: Int, activeRank: Int) -> Double {
        guard activeRank > 0 else { return 0 }
        
        let pOffset = prevIdx * batchSize * capacityLimit
        let sOffset = stateIdx * batchSize * capacityLimit
        let aOffset = prevIdx * batchSize * actDim
        let rOffset = activeIdx * batchSize
        
        let pArray = Array(prevState.immutableBufferPointer(offset: pOffset, length: batchSize * capacityLimit))
        let sArray = Array(state.immutableBufferPointer(offset: sOffset, length: batchSize * capacityLimit))
        let actArray = Array(prevAction.immutableBufferPointer(offset: aOffset, length: batchSize * actDim))
        let rArray = Array(reward.immutableBufferPointer(offset: rOffset, length: batchSize))
        
        let pFull = MLXArray(pArray, [batchSize, capacityLimit])
        let sFull = MLXArray(sArray, [batchSize, capacityLimit])
        let actFull = MLXArray(actArray, [batchSize, actDim])
        let rFull = MLXArray(rArray, [batchSize, 1])
        
        let pActive = pFull[0..<batchSize, 0..<activeRank]
        let sActive = sFull[0..<batchSize, 0..<activeRank]
        
        let padding = MLXArray.zeros([batchSize, capacityLimit - activeRank])
        let ones = MLXArray.ones([batchSize, 1])
        let x = concatenated([pActive, padding, actFull, ones], axis: 1)
        
        let target = concatenated([rFull, sActive], axis: 1)
        
        if compiledUpdate == nil || compiledRank != activeRank {
            compiledUpdate = MLX.compile(inputs: [self], outputs: [self], shapeless: false) { [weak self] arrays in
                guard let self = self else { return [] }
                let x = arrays[0]
                let target = arrays[1]
                let batchSize = x.shape[0]
                let activeRank = target.shape[1] - 1
                
                let w = self.weights[0...activeRank, 0..<self.inputWidth]
                let a = self.alpha[0...activeRank, 0..<self.inputWidth]
                let h_trace = self.h[0...activeRank, 0..<self.inputWidth]
                let v_trace = self.v[0...activeRank, 0..<self.inputWidth]
                
                let gradFn = valueAndGrad(model: self) { (model: WorldModel, input: MLXArray, tgt: MLXArray) -> MLXArray in
                    let activeW = model.weights[0...activeRank, 0..<model.inputWidth]
                    let pred = matmul(input, activeW.T)
                    return (0.5 / Float(batchSize)) * sum(square(pred - tgt))
                }
                let (_, grads) = gradFn(self, x, target)
                let grad = grads[unwrapping: "weights"]![0...activeRank, 0..<self.inputWidth]
                
                let pred = matmul(x, w.T)
                let error = pred - target
                
                let deltaXh = -grad * h_trace
                let absDeltaXh = abs(deltaXh)
                
                let x2 = mean(square(x), axes: [0]).reshaped([1, self.inputWidth])
                
                let vDiff = absDeltaXh - v_trace
                let vUpdate = (1.0 / self.tau) * a * x2 * vDiff
                let newV = maximum(absDeltaXh, v_trace + vUpdate)
                let vZeroMask = newV .== 0.0
                let safeV = MLX.where(vZeroMask, MLXArray(1.0), newV)
                let muFactor = MLXArray(self.mu) * deltaXh / safeV
                var newA = a * exp(muFactor)
                newA = MLX.where(vZeroMask, a, newA)
                
                let currentM = sum(newA * x2, axes: [1], keepDims: true)
                let m_eff = maximum(currentM, MLXArray(1.0))
                
                newA = newA / m_eff
                
                let gradStep = newA * grad
                let newWeights = w - gradStep
                let hDecay = h_trace * (1.0 - newA * x2)
                let newH = hDecay - gradStep
                
                self.alpha[0...activeRank, 0..<self.inputWidth] = newA
                self.v[0...activeRank, 0..<self.inputWidth] = newV
                self.h[0...activeRank, 0..<self.inputWidth] = newH
                self.weights[0...activeRank, 0..<self.inputWidth] = newWeights
                
                return [error]
            }
            compiledRank = activeRank
        }
        
        let outputs = compiledUpdate!([x, target])
        let error = outputs[0]
        
        let totalError = Double(sum(abs(error)).item(Float.self)) / Double(batchSize)
        
        eval(self, self.alpha, self.v, self.h, self.weights)
        
        return totalError
    }

    func predict(state: SharedFloatStorage, stateOffset: Int, action: SharedFloatStorage, actOffset: Int, activeRank: Int, nextState: SharedFloatStorage, nextOffset: Int) -> Float {
        guard activeRank > 0 else { return 0 }
        encodeInput(state: state, stateOffset: stateOffset, action: action, actOffset: actOffset, activeRank: activeRank)
        
        let xArray = Array(inputStorage.immutableBufferPointer(offset: 0, length: inputWidth))
        let x = MLXArray(xArray, [inputWidth])
        
        if compiledPredict == nil || compiledRank != activeRank {
            compiledPredict = MLX.compile(inputs: [self], outputs: [self], shapeless: false) { [weak self] arrays in
                guard let self = self else { return [] }
                let x = arrays[0]
                let activeRank = self.compiledRank
                let w = self.weights[0...activeRank, 0..<self.inputWidth]
                let pred = matmul(w, x)
                return [pred]
            }
            compiledRank = activeRank
        }
        
        let outputs = compiledPredict!([x])
        let pred = outputs[0]
        
        eval(pred)
        let predArray = pred.asArray(Float.self)
        
        let rewardPrediction = predArray[0]
        for row in 0..<activeRank {
            nextState[nextOffset + row] = predArray[row + 1]
        }
        if activeRank < capacityLimit {
            for row in activeRank..<capacityLimit {
                nextState[nextOffset + row] = 0
            }
        }
        return rewardPrediction
    }

    private func encodeInput(state: SharedFloatStorage, stateOffset: Int, action: SharedFloatStorage, actOffset: Int, activeRank: Int) {
        for index in 0..<capacityLimit {
            inputStorage[index] = index < activeRank ? state[stateOffset + index] : 0
        }
        for index in 0..<actDim {
            let actionValue = action[actOffset + index]
            inputStorage[capacityLimit + index] = oakSafeScalar(actionValue, clamp: -1...1)
        }
        inputStorage[inputWidth - 1] = 1
    }
}

final class OptionBank {
    let capacityLimit: Int
    let actDim: Int
    let weights: SharedFloatStorage
    private(set) var utility: [Double]

    init(capacityLimit: Int, actDim: Int, rng: CoreSeededRNG) {
        self.capacityLimit = capacityLimit
        self.actDim = actDim
        weights = SharedFloatStorage(count: max(capacityLimit * actDim, 1), device: nil)
        utility = Array(repeating: 0, count: capacityLimit)
        let scale = Float(1 / sqrt(Double(max(actDim, 1))))
        for row in 0..<capacityLimit {
            let rowOffset = row * actDim
            for column in 0..<actDim {
                weights[rowOffset + column] = rng.nextSignedFloat(scale: scale)
            }
        }
    }
    func recordUtility(index: Int, value: Double) {
        if utility.indices.contains(index) { utility[index] += abs(value) }
    }
}

final class SearchControl {
    private let optionBank: OptionBank
    private let worldModel: WorldModel
    private let control: ReactiveControl
    private let perception: AdaptivePerception
    private let plannedBias: SharedFloatStorage
    private let candidateAction: SharedFloatStorage
    private let predictedState: SharedFloatStorage
    private var cursor = 0

    init(optionBank: OptionBank, worldModel: WorldModel, control: ReactiveControl, perception: AdaptivePerception, plannedBias: SharedFloatStorage, candidateAction: SharedFloatStorage, predictedState: SharedFloatStorage) {
        self.optionBank = optionBank
        self.worldModel = worldModel
        self.control = control
        self.perception = perception
        self.plannedBias = plannedBias
        self.candidateAction = candidateAction
        self.predictedState = predictedState
    }

    func plan(state: SharedFloatStorage, stateOffset: Int, reactiveAction: SharedFloatStorage, actOffset: Int, plannedOffset: Int, activeRank: Int, budgetNs: UInt64) -> Double {
        guard budgetNs > 0, activeRank > 0 else { return 0 }
        let start = oakCurrentTimeNanoseconds()
        var bestScore = -Double.greatestFiniteMagnitude
        var bestIndex = 0

        while oakCurrentTimeNanoseconds() - start < budgetNs {
            let optionIndex = cursor % activeRank
            cursor += 1
            let rowOffset = optionIndex * optionBank.actDim
            for column in 0..<optionBank.actDim {
                let blended = reactiveAction[actOffset + column] + optionBank.weights[rowOffset + column]
                let bounded = oakSafeScalar(tanh(blended), clamp: -1...1)
                candidateAction[column] = bounded
            }

            // Unroll loop for temporally extended action search
            var accumulatedReward = 0.0
            var optionTrainingSignal = 0.0
            let horizon = 3
            
            for step in 0..<horizon {
                let s = step == 0 ? state : predictedState
                let sOff = step == 0 ? stateOffset : 0
                
                let stepReward = worldModel.predict(state: s, stateOffset: sOff, action: candidateAction, actOffset: 0, activeRank: activeRank, nextState: predictedState, nextOffset: 0)
                accumulatedReward += Double(stepReward)
                
                if step == 0 {
                    // Option training signal (representation bottleneck) from the first step
                    optionTrainingSignal = Double(predictedState[optionIndex] - state[stateOffset + optionIndex])
                }
                
                // If we ran out of budget during unrolling, break early
                if oakCurrentTimeNanoseconds() - start >= budgetNs {
                    break
                }
            }
            
            let currentValue = control.evaluate(state: state, stateOffset: stateOffset, activeRank: activeRank)
            let nextValue = control.evaluate(state: predictedState, stateOffset: 0, activeRank: activeRank)
            
            var optionNorm = 0.0
            for column in 0..<optionBank.actDim {
                let value = Double(optionBank.weights[rowOffset + column])
                optionNorm += value * value
            }
            optionNorm = max(sqrt(optionNorm), Double(Float.ulpOfOne))
            
            // Search score combines multi-step accumulated reward + value of final predicted state
            let searchScore = (accumulatedReward - Double(control.rewardRate) + Double(nextValue) - Double(currentValue)) / (1 + optionNorm)
            
            optionBank.recordUtility(index: optionIndex, value: abs(optionTrainingSignal))
            perception.recordUtility(index: optionIndex, value: abs(optionTrainingSignal))

            let adapt = Float(1 / max(1 + abs(optionTrainingSignal), Double(Float.ulpOfOne)))
            let featureActivation = state[stateOffset + optionIndex]
            for column in 0..<optionBank.actDim {
                let candidate = candidateAction[column]
                optionBank.weights[rowOffset + column] += adapt * featureActivation * Float(optionTrainingSignal) * (1 - candidate * candidate)
            }

            if searchScore > bestScore {
                bestScore = searchScore
                bestIndex = optionIndex
            }
            if cursor >= activeRank, oakCurrentTimeNanoseconds() - start >= budgetNs { break }
        }

        let winnerOffset = bestIndex * optionBank.actDim
        let blendDenominator = Float(1 + max(bestScore.magnitude, Double(Float.ulpOfOne)))
        for column in 0..<optionBank.actDim {
            let target = optionBank.weights[winnerOffset + column]
            plannedBias[plannedOffset + column] += (target - plannedBias[plannedOffset + column]) / blendDenominator
        }
        return Double(oakCurrentTimeNanoseconds() - start) / 1_000_000
    }
}

class BaseSelfScalingCore: ForegroundCore {
    let obsDim: Int
    let actDim: Int
    let signalDim: Int
    let capacityLimit: Int
    let backendName: String

    let perception: AdaptivePerception
    let control: ReactiveControl
    let predictiveKnowledge: PredictiveKnowledge
    let worldModel: WorldModel
    let optionBank: OptionBank
    let searchControl: SearchControl

    // Double-Buffered Shared Memory State
    var currentBatchSize = 1
    var bufferIndex = 0 // Toggles 0/1
    var stepCount = 0
    
    var signal: SharedFloatStorage
    var state: SharedFloatStorage
    var prevState: SharedFloatStorage
    var action: SharedFloatStorage
    var reward: SharedFloatStorage
    var reconstruction: SharedFloatStorage
    var residual: SharedFloatStorage
    var plannedBias: SharedFloatStorage
    let candidateAction: SharedFloatStorage
    let predictedState: SharedFloatStorage

    private var nsPerWorkUnitEstimate: Double
    private var snapshotToken: Int

    init(obsDim: Int, actDim: Int, device: MTLDevice?, backendName: String, seed: Int?) {
        self.obsDim = obsDim
        self.actDim = actDim
        signalDim = obsDim + actDim + 1
        capacityLimit = max(1, Int(ceil(sqrt(Double(max(signalDim, 1))))))
        self.backendName = backendName

        let rootSeed = oakDeriveSeed(seed, salt: 0x5000) ?? 0x9E3779B97F4A7C15
        let rng = CoreSeededRNG(seed: rootSeed)

        // Initialize with default size 1 * 2 buffers
        signal = SharedFloatStorage(count: max(signalDim * 2, 1), device: nil)
        state = SharedFloatStorage(count: max(capacityLimit * 2, 1), device: nil)
        prevState = SharedFloatStorage(count: max(capacityLimit * 2, 1), device: nil)
        action = SharedFloatStorage(count: max(actDim * 2, 1), device: nil)
        reward = SharedFloatStorage(count: max(2, 1), device: nil)
        
        plannedBias = SharedFloatStorage(count: max(actDim * 2, 1), device: device)
        reconstruction = SharedFloatStorage(count: max(signalDim * 2, 1), device: nil)
        residual = SharedFloatStorage(count: max(signalDim * 2, 1), device: nil)
        candidateAction = SharedFloatStorage(count: max(actDim, 1), device: nil)
        predictedState = SharedFloatStorage(count: max(capacityLimit, 1), device: nil)

        perception = AdaptivePerception(signalDim: signalDim, capacityLimit: capacityLimit, device: device, rng: rng)
        control = ReactiveControl(actDim: actDim, capacityLimit: capacityLimit, device: device, rng: rng)
        predictiveKnowledge = PredictiveKnowledge(capacityLimit: capacityLimit)
        worldModel = WorldModel(capacityLimit: capacityLimit, actDim: actDim)
        optionBank = OptionBank(capacityLimit: capacityLimit, actDim: actDim, rng: rng)
        searchControl = SearchControl(
            optionBank: optionBank, worldModel: worldModel, control: control,
            perception: perception, plannedBias: plannedBias, candidateAction: candidateAction, predictedState: predictedState
        )
        nsPerWorkUnitEstimate = 1
        snapshotToken = 0
    }

    func ensureBatchSize(_ batchSize: Int, device: MTLDevice?) {
        if batchSize > currentBatchSize {
            currentBatchSize = batchSize
            signal = SharedFloatStorage(count: max(signalDim * batchSize * 2, 1), device: device)
            state = SharedFloatStorage(count: max(capacityLimit * batchSize * 2, 1), device: device)
            prevState = SharedFloatStorage(count: max(capacityLimit * batchSize * 2, 1), device: device)
            action = SharedFloatStorage(count: max(actDim * batchSize * 2, 1), device: device)
            reward = SharedFloatStorage(count: max(batchSize * 2, 1), device: device)
            reconstruction = SharedFloatStorage(count: max(signalDim * batchSize * 2, 1), device: nil)
            residual = SharedFloatStorage(count: max(signalDim * batchSize * 2, 1), device: nil)
            plannedBias = SharedFloatStorage(count: max(actDim * batchSize * 2, 1), device: device)
        }
    }

    func step(input: StepInput) -> StepOutput {
        ensureBatchSize(input.batchSize, device: nil)
        let stepStartNs = oakCurrentTimeNanoseconds()
        
        let activeIdx = bufferIndex
        let prevIdx = 1 - bufferIndex
        
        // 1. Load incoming signal into active buffer
        loadSignal(obs: input.obs, prevAction: input.prevAction, prevReward: input.prevReward, batchSize: input.batchSize, bufferIndex: activeIdx)

        let capacityDecision = decideCapacity(budgetNs: input.wallClockBudgetNs)
        
        let sigOff0 = (activeIdx * input.batchSize + 0) * signalDim
        let stOff0 = (activeIdx * input.batchSize + 0) * capacityLimit
        let reconOff0 = (activeIdx * input.batchSize + 0) * signalDim
        let resOff0 = (activeIdx * input.batchSize + 0) * signalDim
        
        let encoding = perception.encode(
            signal: signal, signalOffset: sigOff0, state: state, stateOffset: stOff0,
            reconstruction: reconstruction, reconOffset: reconOff0, residual: residual, resOffset: resOff0,
            allowedRank: capacityDecision.allowedRank
        )
        let activeRank = perception.currentActiveRank(budget: capacityDecision)
        
        for b in 0..<input.batchSize {
            let sigOff = (activeIdx * input.batchSize + b) * signalDim
            let stOff = (activeIdx * input.batchSize + b) * capacityLimit
            let reconOff = (activeIdx * input.batchSize + b) * signalDim
            let resOff = (activeIdx * input.batchSize + b) * signalDim
            
            if b > 0 {
                _ = perception.encode(
                    signal: signal, signalOffset: sigOff, state: state, stateOffset: stOff,
                    reconstruction: reconstruction, reconOffset: reconOff, residual: residual, resOffset: resOff,
                    allowedRank: activeRank
                )
            } else if perception.activeRank > activeRank {
                _ = perception.encode(
                    signal: signal, signalOffset: sigOff0, state: state, stateOffset: stOff0,
                    reconstruction: reconstruction, reconOffset: reconOff0, residual: residual, resOffset: resOff0,
                    allowedRank: activeRank
                )
            }
        }

        let encodeStartMs = oakCurrentTimeMilliseconds()
        
        // 2. Dispatch GPU inference asynchronously
        let computeFuture = performActionInferenceAsync(activeRank: perception.activeRank, batchSize: input.batchSize, bufferIndex: activeIdx)
        
        // Await inference BEFORE updating weights. This prevents tearing data races where the GPU reads `basis` and `policy` while the CPU mutates them.
        computeFuture.wait()
        
        let foregroundComputeMs = oakCurrentTimeMilliseconds() - encodeStartMs

        // 3. Compute Learning Updates on the PREVIOUS buffer
        var planningMs = 0.0
        if stepCount > 0 {
            // Batched MLX Updates
            let gvfErrorMean = predictiveKnowledge.updateBatched(
                prevState: prevState, prevIdx: prevIdx, state: state, stateIdx: activeIdx,
                batchSize: input.batchSize, capacityLimit: capacityLimit, activeRank: perception.activeRank
            )
            
            let modelErrorMean = worldModel.updateBatched(
                prevState: prevState, prevIdx: prevIdx, state: state, stateIdx: activeIdx,
                prevAction: action, activeIdx: activeIdx, reward: reward,
                batchSize: input.batchSize, capacityLimit: capacityLimit, activeRank: perception.activeRank
            )
            
            let gvfErrorPerItem = gvfErrorMean
            let modelErrorPerItem = modelErrorMean

            for b in 0..<input.batchSize {
                let pStOff = (prevIdx * input.batchSize + b) * capacityLimit
                let pPrevStOff = (prevIdx * input.batchSize + b) * capacityLimit
                let pActOff = (prevIdx * input.batchSize + b) * actDim
                
                let r = reward[(activeIdx * input.batchSize) + b]
                
                let tdDelta = control.update(prevState: prevState, prevOffset: pPrevStOff, state: state, stateOffset: pStOff, prevAction: action, prevActOffset: pActOff, reward: r, activeRank: perception.activeRank)
                
                for index in 0..<perception.activeRank {
                    let utilitySignal = Double(abs(tdDelta * prevState[pPrevStOff + index])) + gvfErrorPerItem + modelErrorPerItem + encoding.residualEnergy
                    perception.recordUtility(index: index, value: utilitySignal)
                    optionBank.recordUtility(index: index, value: utilitySignal)
                }
                
                let elapsedBeforePlanning = oakCurrentTimeNanoseconds() - stepStartNs
                let remainingBudgetNs = input.wallClockBudgetNs > elapsedBeforePlanning ? (input.wallClockBudgetNs - elapsedBeforePlanning) / UInt64(input.batchSize - b) : 0
                
                let planOff = (activeIdx * input.batchSize + b) * actDim
                planningMs += searchControl.plan(
                    state: state, stateOffset: pStOff,
                    reactiveAction: action, actOffset: pActOff,
                    plannedOffset: planOff,
                    activeRank: perception.activeRank, budgetNs: remainingBudgetNs
                )
            }
        }
        
        for b in 0..<input.batchSize {
            let stOff = (activeIdx * input.batchSize + b) * capacityLimit
            let prevStOff = (activeIdx * input.batchSize + b) * capacityLimit
            for index in 0..<capacityLimit {
                prevState[prevStOff + index] = state[stOff + index]
            }
        }

        // 4. Finalize Agent Outputs
        snapshotToken &+= 1
        let stepNs = oakCurrentTimeNanoseconds() - stepStartNs
        let workUnits = max(Double(perception.activeRank * (signalDim + actDim + perception.activeRank) * input.batchSize), 1)
        let measuredNsPerWorkUnit = Double(stepNs) / workUnits
        nsPerWorkUnitEstimate = (nsPerWorkUnitEstimate + measuredNsPerWorkUnit) / 2

        let telemetry = input.obs.withUnsafeBufferPointer { obsPtr in
            AgentTelemetry(
                rewardInput: input.prevReward[0],
                obs: UnsafeBufferPointer(start: obsPtr.baseAddress, count: min(obsPtr.count, obsDim)),
                state: state.immutableBufferPointer(offset: (activeIdx * input.batchSize) * capacityLimit, length: perception.activeRank),
                action: action.immutableBufferPointer(offset: (activeIdx * input.batchSize) * actDim, length: actDim),
                agentStepMs: Double(stepNs) / 1_000_000,
                backend: backendName,
                metalTimings: MetalCommandTimings(cmd1Ms: foregroundComputeMs, cmd2Ms: 0, cmd3Ms: 0, cmd4Ms: 0, cmd5Ms: planningMs),
                plannerStepMs: planningMs,
                computeBudgetNs: input.wallClockBudgetNs,
                activeRank: perception.activeRank,
                activeGVFs: perception.activeRank,
                activeOptions: perception.activeRank,
                snapshotToken: snapshotToken
            )
        }
        
        bufferIndex = prevIdx
        stepCount += 1
        
        let outAction = action.snapshot(offset: (activeIdx * input.batchSize) * actDim, prefix: actDim * input.batchSize)
        return StepOutput(action: outAction, telemetry: telemetry, snapshotToken: snapshotToken)
    }

    func performActionInferenceAsync(activeRank: Int, batchSize: Int, bufferIndex: Int) -> ComputeFuture {
        for b in 0..<batchSize {
            control.actCPU(
                state: state, stateOffset: (bufferIndex * batchSize + b) * capacityLimit,
                plannedBias: plannedBias, plannedOffset: (bufferIndex * batchSize + b) * actDim,
                activeRank: activeRank,
                action: action, actOffset: (bufferIndex * batchSize + b) * actDim
            )
        }
        return CPUComputeFuture()
    }

    private func decideCapacity(budgetNs: UInt64) -> CapacityDecision {
        let safeBudget = max(Double(budgetNs), 1)
        let budgetWorkUnits = safeBudget / max(nsPerWorkUnitEstimate, Double(Float.ulpOfOne))
        let dimensionality = Double(signalDim + actDim)
        let rankByBudget = Int(max(1, floor((sqrt(dimensionality * dimensionality + 4 * budgetWorkUnits) - dimensionality) / 2)))
        return CapacityDecision(allowedRank: max(1, min(capacityLimit, rankByBudget)), budgetWorkUnits: budgetWorkUnits)
    }

    private func loadSignal(obs: [Float], prevAction: [Float], prevReward: [Float], batchSize: Int, bufferIndex: Int) {
        for b in 0..<batchSize {
            let sigOff = (bufferIndex * batchSize + b) * signalDim
            let obsOff = b * obsDim
            let actOff = b * actDim
            for index in 0..<obsDim {
                signal[sigOff + index] = oakSafeScalar(obs[obsOff + index])
            }
            for index in 0..<actDim {
                signal[sigOff + obsDim + index] = oakSafeScalar(prevAction[actOff + index], clamp: -1...1)
            }
            let r = oakSafeScalar(prevReward[b])
            signal[sigOff + signalDim - 1] = r
            reward[bufferIndex * batchSize + b] = r
        }
    }

    func saveCheckpoint(url: URL) throws {
        var stateDict: [String: MLXArray] = [:]
        
        let pkParams = predictiveKnowledge.parameters().flattened()
        for (k, v) in pkParams { stateDict["pk.\(k)"] = v }
        stateDict["pk.alpha"] = predictiveKnowledge.alpha
        stateDict["pk.h"] = predictiveKnowledge.h
        stateDict["pk.v"] = predictiveKnowledge.v
        
        let wmParams = worldModel.parameters().flattened()
        for (k, v) in wmParams { stateDict["wm.\(k)"] = v }
        stateDict["wm.alpha"] = worldModel.alpha
        stateDict["wm.h"] = worldModel.h
        stateDict["wm.v"] = worldModel.v
        
        let policyArray = Array(control.policy.immutableBufferPointer())
        stateDict["control.policy"] = MLXArray(policyArray, [control.actDim * capacityLimit])
        let valueWeightsArray = Array(control.valueWeights.immutableBufferPointer())
        stateDict["control.valueWeights"] = MLXArray(valueWeightsArray, [capacityLimit])
        stateDict["control.rewardRate"] = MLXArray(control.rewardRate)
        
        let basisArray = Array(perception.basis.immutableBufferPointer())
        stateDict["perception.basis"] = MLXArray(basisArray, [capacityLimit * signalDim])
        
        let optionWeightsArray = Array(optionBank.weights.immutableBufferPointer())
        stateDict["optionBank.weights"] = MLXArray(optionWeightsArray, [capacityLimit * actDim])
        
        try MLX.save(arrays: stateDict, url: url)
    }

    func loadCheckpoint(url: URL) throws {
        var loadedDict = try MLX.loadArrays(url: url)
        
        if let a = loadedDict.removeValue(forKey: "pk.alpha") { predictiveKnowledge.alpha = a }
        if let h = loadedDict.removeValue(forKey: "pk.h") { predictiveKnowledge.h = h }
        if let v = loadedDict.removeValue(forKey: "pk.v") { predictiveKnowledge.v = v }
        
        var pkParams: [String: MLXArray] = [:]
        for key in loadedDict.keys.filter({ $0.hasPrefix("pk.") }) {
            let val = loadedDict.removeValue(forKey: key)!
            let subKey = String(key.dropFirst(3))
            pkParams[subKey] = val
        }
        predictiveKnowledge.update(parameters: ModuleParameters.unflattened(pkParams))
        
        if let a = loadedDict.removeValue(forKey: "wm.alpha") { worldModel.alpha = a }
        if let h = loadedDict.removeValue(forKey: "wm.h") { worldModel.h = h }
        if let v = loadedDict.removeValue(forKey: "wm.v") { worldModel.v = v }
        
        var wmParams: [String: MLXArray] = [:]
        for key in loadedDict.keys.filter({ $0.hasPrefix("wm.") }) {
            let val = loadedDict.removeValue(forKey: key)!
            let subKey = String(key.dropFirst(3))
            wmParams[subKey] = val
        }
        worldModel.update(parameters: ModuleParameters.unflattened(wmParams))
        
        if let policyArr = loadedDict.removeValue(forKey: "control.policy") {
            control.policy.copy(from: policyArr.asArray(Float.self))
        }
        if let vwArr = loadedDict.removeValue(forKey: "control.valueWeights") {
            control.valueWeights.copy(from: vwArr.asArray(Float.self))
        }
        if let rrArr = loadedDict.removeValue(forKey: "control.rewardRate") {
            control.rewardRate = rrArr.item(Float.self)
        }
        
        if let basisArr = loadedDict.removeValue(forKey: "perception.basis") {
            perception.basis.copy(from: basisArr.asArray(Float.self))
        }
        
        if let owArr = loadedDict.removeValue(forKey: "optionBank.weights") {
            optionBank.weights.copy(from: owArr.asArray(Float.self))
        }
    }
}

final class CPUForegroundCore: BaseSelfScalingCore {
    init(obsDim: Int, actDim: Int, seed: Int?) {
        super.init(obsDim: obsDim, actDim: actDim, device: nil, backendName: "cpu", seed: seed)
    }
}

class MetalComputeFuture: ComputeFuture {
    let commandBuffer: MTLCommandBuffer
    init(commandBuffer: MTLCommandBuffer) { self.commandBuffer = commandBuffer }
    func wait() { commandBuffer.waitUntilCompleted() }
}

final class MetalForegroundCore: BaseSelfScalingCore {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let encodePipeline: MTLComputePipelineState
    private let actionPipeline: MTLComputePipelineState

    init(obsDim: Int, actDim: Int, device: MTLDevice, seed: Int?) {
        self.device = device
        queue = device.makeCommandQueue()!
        let library: MTLLibrary
        if let defaultLib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            library = defaultLib
        } else {
            let kernelsURL = Bundle.module.url(forResource: "Kernels", withExtension: "metal")!
            let kernelsSource = try! String(contentsOf: kernelsURL, encoding: .utf8)
            library = try! device.makeLibrary(source: kernelsSource, options: nil)
        }
        encodePipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "encode_state")!)
        actionPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "policy_action")!)
        super.init(obsDim: obsDim, actDim: actDim, device: device, backendName: "metal", seed: seed)
    }

    override func ensureBatchSize(_ batchSize: Int, device: MTLDevice?) {
        super.ensureBatchSize(batchSize, device: self.device)
    }

    override func performActionInferenceAsync(activeRank: Int, batchSize: Int, bufferIndex: Int) -> ComputeFuture {
        let commandBuffer = queue.makeCommandBuffer()!

        let encodeEncoder = commandBuffer.makeComputeCommandEncoder()!
        encodeEncoder.setComputePipelineState(encodePipeline)
        encodeEncoder.setBuffer(perception.basis.metalBuffer, offset: 0, index: 0)
        encodeEncoder.setBuffer(signal.metalBuffer, offset: 0, index: 1)
        encodeEncoder.setBuffer(state.metalBuffer, offset: 0, index: 2)
        var signalDimValue = UInt32(signalDim)
        var activeRankValue = UInt32(activeRank)
        var stateStrideValue = UInt32(capacityLimit)
        encodeEncoder.setBytes(&signalDimValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encodeEncoder.setBytes(&activeRankValue, length: MemoryLayout<UInt32>.stride, index: 4)
        encodeEncoder.setBytes(&stateStrideValue, length: MemoryLayout<UInt32>.stride, index: 5)
        var signalOffsetElement = UInt32(bufferIndex * batchSize * signalDim)
        var stateOffsetElement = UInt32(bufferIndex * batchSize * capacityLimit)
        encodeEncoder.setBytes(&signalOffsetElement, length: MemoryLayout<UInt32>.stride, index: 6)
        encodeEncoder.setBytes(&stateOffsetElement, length: MemoryLayout<UInt32>.stride, index: 7)
        
        let encodeThreads = MTLSize(width: max(activeRank, 1), height: batchSize, depth: 1)
        let wGroup = min(max(activeRank, 1), encodePipeline.maxTotalThreadsPerThreadgroup)
        let encodeGroup = MTLSize(width: wGroup, height: 1, depth: 1)
        encodeEncoder.dispatchThreads(encodeThreads, threadsPerThreadgroup: encodeGroup)
        encodeEncoder.endEncoding()

        let actionEncoder = commandBuffer.makeComputeCommandEncoder()!
        actionEncoder.setComputePipelineState(actionPipeline)
        actionEncoder.setBuffer(control.policy.metalBuffer, offset: 0, index: 0)
        actionEncoder.setBuffer(state.metalBuffer, offset: 0, index: 1)
        actionEncoder.setBuffer(plannedBias.metalBuffer, offset: 0, index: 2)
        actionEncoder.setBuffer(action.metalBuffer, offset: 0, index: 3)
        actionEncoder.setBytes(&activeRankValue, length: MemoryLayout<UInt32>.stride, index: 4)
        var actDimValue = UInt32(actDim)
        actionEncoder.setBytes(&actDimValue, length: MemoryLayout<UInt32>.stride, index: 5)
        var policyStride = UInt32(capacityLimit)
        actionEncoder.setBytes(&policyStride, length: MemoryLayout<UInt32>.stride, index: 6)
        actionEncoder.setBytes(&stateStrideValue, length: MemoryLayout<UInt32>.stride, index: 7)
        var plannedBiasOffsetElement = UInt32(bufferIndex * batchSize * actDim)
        var actionOffsetElement = UInt32(bufferIndex * batchSize * actDim)
        actionEncoder.setBytes(&stateOffsetElement, length: MemoryLayout<UInt32>.stride, index: 8)
        actionEncoder.setBytes(&plannedBiasOffsetElement, length: MemoryLayout<UInt32>.stride, index: 9)
        actionEncoder.setBytes(&actionOffsetElement, length: MemoryLayout<UInt32>.stride, index: 10)
        
        let actionThreads = MTLSize(width: max(actDim, 1), height: batchSize, depth: 1)
        let wActGroup = min(max(actDim, 1), actionPipeline.maxTotalThreadsPerThreadgroup)
        let actionGroup = MTLSize(width: wActGroup, height: 1, depth: 1)
        actionEncoder.dispatchThreads(actionThreads, threadsPerThreadgroup: actionGroup)
        actionEncoder.endEncoding()

        commandBuffer.commit()
        return MetalComputeFuture(commandBuffer: commandBuffer)
    }
}
