import Darwin
import Foundation
import Metal

struct StepInput {
    let obs: [Float]
    let prevAction: [Float]
    let prevReward: Float
    let wallClockBudgetNs: UInt64
}

struct StepOutput {
    let action: [Float]
    let telemetry: AgentTelemetry
    let snapshotToken: Int
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
    guard let seed else {
        return nil
    }

    var value = UInt64(bitPattern: Int64(seed)) &+ salt &+ 0x9E3779B97F4A7C15
    value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
    value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
    return value ^ (value >> 31)
}

private func oakSafeScalar(_ value: Float, clamp range: ClosedRange<Float>? = nil) -> Float {
    let finiteValue = value.isFinite ? value : 0
    guard let range else {
        return finiteValue
    }
    return min(max(finiteValue, range.lowerBound), range.upperBound)
}

private func oakMean(_ values: UnsafeBufferPointer<Float>, count: Int) -> Double {
    guard count > 0 else {
        return 0
    }
    var sum = 0.0
    for index in 0..<count {
        sum += Double(values[index])
    }
    return sum / Double(count)
}

private func oakL2(_ values: UnsafeBufferPointer<Float>, count: Int) -> Double {
    guard count > 0 else {
        return 0
    }
    var sum = 0.0
    for index in 0..<count {
        let value = Double(values[index])
        sum += value * value
    }
    return sqrt(sum)
}

private func oakDot(_ lhs: UnsafeBufferPointer<Float>, _ rhs: UnsafeBufferPointer<Float>, count: Int) -> Float {
    guard count > 0 else {
        return 0
    }
    var sum: Float = 0
    for index in 0..<count {
        sum += lhs[index] * rhs[index]
    }
    return sum
}

private func oakDotRow(
    _ storage: SharedFloatStorage,
    row: Int,
    stride: Int,
    vector: UnsafeBufferPointer<Float>,
    count: Int
) -> Float {
    guard count > 0 else {
        return 0
    }
    let rowOffset = row * stride
    var sum: Float = 0
    for index in 0..<count {
        sum += storage[rowOffset + index] * vector[index]
    }
    return sum
}

private func oakNormalizeRow(_ storage: SharedFloatStorage, row: Int, stride: Int, count: Int) {
    guard count > 0 else {
        return
    }
    let rowOffset = row * stride
    var norm: Double = 0
    for index in 0..<count {
        let value = Double(storage[rowOffset + index])
        norm += value * value
    }
    guard norm > Double(Float.ulpOfOne) else {
        return
    }
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
    private let pointer: UnsafeMutablePointer<Float>
    let metalBuffer: MTLBuffer?

    init(count: Int, device: MTLDevice? = nil) {
        self.count = count
        pointer = UnsafeMutablePointer<Float>.allocate(capacity: count)
        pointer.initialize(repeating: 0, count: count)
        if let device {
            metalBuffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: count * MemoryLayout<Float>.stride,
                options: [.storageModeShared],
                deallocator: nil
            )
        } else {
            metalBuffer = nil
        }
    }

    deinit {
        pointer.deinitialize(count: count)
        pointer.deallocate()
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

    func copy(from values: [Float], count: Int? = nil) {
        let targetCount = min(count ?? values.count, self.count)
        values.withUnsafeBufferPointer { source in
            guard let sourceBase = source.baseAddress else {
                return
            }
            pointer.update(from: sourceBase, count: targetCount)
        }
        if targetCount < self.count {
            memset(pointer.advanced(by: targetCount), 0, (self.count - targetCount) * MemoryLayout<Float>.stride)
        }
    }

    func snapshot(prefix: Int) -> [Float] {
        let safePrefix = min(max(prefix, 0), count)
        return Array(UnsafeBufferPointer(start: pointer, count: safePrefix))
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
        signal: SharedFloatStorage,
        state: SharedFloatStorage,
        reconstruction: SharedFloatStorage,
        residual: SharedFloatStorage,
        allowedRank: Int
    ) -> (residualEnergy: Double, totalEnergy: Double) {
        let boundedRank = max(1, min(allowedRank, capacityLimit))
        if activeRank > boundedRank {
            compact(to: boundedRank)
        }

        reconstruction.zero()
        let signalBuffer = signal.immutableBufferPointer()

        for row in 0..<activeRank {
            let projection = tanh(oakDotRow(basis, row: row, stride: signalDim, vector: signalBuffer, count: signalDim))
            state[row] = oakSafeScalar(projection)
            featureEnergy[row] += Double(projection * projection)
            let rowOffset = row * signalDim
            for column in 0..<signalDim {
                reconstruction[column] += projection * basis[rowOffset + column]
            }
        }
        if activeRank < capacityLimit {
            for row in activeRank..<capacityLimit {
                state[row] = 0
            }
        }

        var totalEnergy = 0.0
        var residualEnergy = 0.0
        for column in 0..<signalDim {
            let residualValue = signal[column] - reconstruction[column]
            residual[column] = residualValue
            let signalValue = Double(signal[column])
            let residualValueDouble = Double(residualValue)
            totalEnergy += signalValue * signalValue
            residualEnergy += residualValueDouble * residualValueDouble
        }

        updateBasis(signal: signal, state: state, residual: residual)
        adaptBasisSet(residual: residual, residualEnergy: residualEnergy, allowedRank: boundedRank)
        updateFeatureUtilities(from: state)
        return (residualEnergy, totalEnergy)
    }

    func currentActiveRank(budget: CapacityDecision) -> Int {
        let energySum = featureEnergy.prefix(activeRank).reduce(0, +)
        let energySquaredSum = featureEnergy.prefix(activeRank).reduce(0) { partial, value in
            partial + value * value
        }
        let effectiveRank = energySquaredSum > 0
            ? Int(ceil((energySum * energySum) / energySquaredSum))
            : 1
        return max(1, min(budget.allowedRank, max(effectiveRank, 1)))
    }

    func recordUtility(index: Int, value: Double) {
        guard featureUtility.indices.contains(index) else {
            return
        }
        featureUtility[index] += abs(value)
    }

    private func updateBasis(signal: SharedFloatStorage, state: SharedFloatStorage, residual: SharedFloatStorage) {
        let signalBuffer = signal.immutableBufferPointer()
        for row in 0..<activeRank {
            let rate = Float(1 / max(featureEnergy[row], Double(Float.ulpOfOne)))
            let activation = state[row]
            let rowOffset = row * signalDim
            for column in 0..<signalDim {
                let current = basis[rowOffset + column]
                let correction = signalBuffer[column] - activation * current
                basis[rowOffset + column] = current + rate * activation * correction
            }
            oakNormalizeRow(basis, row: row, stride: signalDim, count: signalDim)
        }
    }

    private func adaptBasisSet(residual: SharedFloatStorage, residualEnergy: Double, allowedRank: Int) {
        guard allowedRank > 0 else {
            return
        }
        let meanEnergy = featureEnergy.prefix(max(activeRank, 1)).reduce(0, +) / Double(max(activeRank, 1))
        guard residualEnergy > meanEnergy else {
            return
        }

        if activeRank < allowedRank {
            let targetRow = activeRank
            writeNormalizedResidual(residual, into: targetRow)
            featureEnergy[targetRow] = residualEnergy
            featureUtility[targetRow] = residualEnergy
            activeRank += 1
            return
        }

        guard let weakestIndex = (0..<activeRank).min(by: { featureUtility[$0] < featureUtility[$1] }) else {
            return
        }
        writeNormalizedResidual(residual, into: weakestIndex)
        featureEnergy[weakestIndex] = residualEnergy
        featureUtility[weakestIndex] = residualEnergy
    }

    private func writeNormalizedResidual(_ residual: SharedFloatStorage, into row: Int) {
        let rowOffset = row * signalDim
        for column in 0..<signalDim {
            basis[rowOffset + column] = residual[column]
        }
        oakNormalizeRow(basis, row: row, stride: signalDim, count: signalDim)
    }

    private func compact(to targetRank: Int) {
        guard targetRank < activeRank else {
            return
        }
        let ranked = (0..<activeRank).sorted { lhs, rhs in
            let lhsScore = featureUtility[lhs] + featureEnergy[lhs]
            let rhsScore = featureUtility[rhs] + featureEnergy[rhs]
            return lhsScore > rhsScore
        }
        let keep = Array(ranked.prefix(targetRank))
        let scratch = basis.snapshot(prefix: basis.count)
        for (newRow, oldRow) in keep.enumerated() {
            let srcOffset = oldRow * signalDim
            let dstOffset = newRow * signalDim
            for column in 0..<signalDim {
                basis[dstOffset + column] = scratch[srcOffset + column]
            }
            featureEnergy[newRow] = featureEnergy[oldRow]
            featureUtility[newRow] = featureUtility[oldRow]
        }
        if targetRank < capacityLimit {
            for row in targetRank..<capacityLimit {
                let rowOffset = row * signalDim
                for column in 0..<signalDim {
                    basis[rowOffset + column] = 0
                }
                featureEnergy[row] = 0
                featureUtility[row] = 0
            }
        }
        activeRank = max(targetRank, 1)
    }

    private func updateFeatureUtilities(from state: SharedFloatStorage) {
        for row in 0..<activeRank {
            featureUtility[row] += Double(abs(state[row]))
        }
    }
}

final class ReactiveControl {
    let actDim: Int
    let capacityLimit: Int
    let policy: SharedFloatStorage
    let valueWeights: SharedFloatStorage
    private(set) var rewardRate: Float

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

    func actCPU(state: SharedFloatStorage, plannedBias: SharedFloatStorage, activeRank: Int, action: SharedFloatStorage) {
        let stateBuffer = state.immutableBufferPointer()
        for row in 0..<actDim {
            var sum = plannedBias[row]
            let rowOffset = row * capacityLimit
            for column in 0..<activeRank {
                sum += policy[rowOffset + column] * stateBuffer[column]
            }
            action[row] = oakSafeScalar(tanh(sum), clamp: -1...1)
        }
    }

    func evaluate(state: SharedFloatStorage, activeRank: Int) -> Float {
        let stateBuffer = state.immutableBufferPointer()
        let valueBuffer = valueWeights.immutableBufferPointer()
        return oakDot(stateBuffer, valueBuffer, count: activeRank)
    }

    func update(prevState: SharedFloatStorage, state: SharedFloatStorage, prevAction: [Float], reward: Float, activeRank: Int) -> Float {
        guard activeRank > 0 else {
            return 0
        }
        let prevValue = evaluate(state: prevState, activeRank: activeRank)
        let nextValue = evaluate(state: state, activeRank: activeRank)
        let prevStateBuffer = prevState.immutableBufferPointer()
        let stateNorm = max(oakDot(prevStateBuffer, prevStateBuffer, count: activeRank), Float.ulpOfOne)
        let step = 1 / (1 + stateNorm)
        let delta = reward - rewardRate + nextValue - prevValue
        rewardRate += step * delta

        for index in 0..<activeRank {
            valueWeights[index] += step * delta * prevState[index]
        }

        for row in 0..<actDim {
            let actionValue = row < prevAction.count ? oakSafeScalar(prevAction[row], clamp: -1...1) : 0
            let sensitivity = 1 - actionValue * actionValue
            let rowOffset = row * capacityLimit
            for column in 0..<activeRank {
                policy[rowOffset + column] += step * delta * sensitivity * prevState[column]
            }
        }

        return delta
    }
}

final class PredictiveKnowledge {
    let capacityLimit: Int
    let weights: SharedFloatStorage
    private(set) var utility: [Double]

    init(capacityLimit: Int) {
        self.capacityLimit = capacityLimit
        weights = SharedFloatStorage(count: max(capacityLimit * capacityLimit, 1), device: nil)
        utility = Array(repeating: 0, count: capacityLimit)
    }

    func update(prevState: SharedFloatStorage, state: SharedFloatStorage, activeRank: Int) -> Double {
        guard activeRank > 0 else {
            return 0
        }
        let prevStateBuffer = prevState.immutableBufferPointer()
        let stateNorm = max(oakDot(prevStateBuffer, prevStateBuffer, count: activeRank), Float.ulpOfOne)
        let step = 1 / (1 + stateNorm)
        var totalError = 0.0
        for target in 0..<activeRank {
            var prediction: Float = 0
            let rowOffset = target * capacityLimit
            for column in 0..<activeRank {
                prediction += weights[rowOffset + column] * prevState[column]
            }
            let error = state[target] - prediction
            totalError += Double(abs(error))
            utility[target] += Double(abs(error))
            for column in 0..<activeRank {
                weights[rowOffset + column] += step * error * prevState[column]
            }
        }
        return totalError
    }
}

final class WorldModel {
    let capacityLimit: Int
    let actDim: Int
    let inputWidth: Int
    let weights: SharedFloatStorage
    private let inputStorage: SharedFloatStorage
    private let predictionStorage: SharedFloatStorage

    init(capacityLimit: Int, actDim: Int) {
        self.capacityLimit = capacityLimit
        self.actDim = actDim
        inputWidth = capacityLimit + actDim + 1
        weights = SharedFloatStorage(count: max((capacityLimit + 1) * inputWidth, 1), device: nil)
        inputStorage = SharedFloatStorage(count: max(inputWidth, 1), device: nil)
        predictionStorage = SharedFloatStorage(count: max(capacityLimit + 1, 1), device: nil)
    }

    func update(prevState: SharedFloatStorage, state: SharedFloatStorage, prevAction: [Float], reward: Float, activeRank: Int) -> Double {
        guard activeRank > 0 else {
            return 0
        }
        encodeInput(state: prevState, action: prevAction, activeRank: activeRank)
        let inputBuffer = inputStorage.immutableBufferPointer()
        let inputNorm = max(oakDot(inputBuffer, inputBuffer, count: inputWidth), Float.ulpOfOne)
        let step = 1 / (1 + inputNorm)
        var totalError = 0.0

        for row in 0...activeRank {
            let target: Float = row == 0 ? reward : state[row - 1]
            let prediction = predictRow(row: row, activeRank: activeRank)
            let error = target - prediction
            totalError += Double(abs(error))
            let rowOffset = row * inputWidth
            for column in 0..<inputWidth {
                weights[rowOffset + column] += step * error * inputStorage[column]
            }
        }

        return totalError
    }

    func predict(state: SharedFloatStorage, action: [Float], activeRank: Int, nextState: SharedFloatStorage) -> Float {
        guard activeRank > 0 else {
            nextState.zero()
            return 0
        }
        encodeInput(state: state, action: action, activeRank: activeRank)
        let rewardPrediction = predictRow(row: 0, activeRank: activeRank)
        for row in 0..<activeRank {
            nextState[row] = predictRow(row: row + 1, activeRank: activeRank)
        }
        if activeRank < capacityLimit {
            for row in activeRank..<capacityLimit {
                nextState[row] = 0
            }
        }
        return rewardPrediction
    }

    private func encodeInput(state: SharedFloatStorage, action: [Float], activeRank: Int) {
        for index in 0..<capacityLimit {
            inputStorage[index] = index < activeRank ? state[index] : 0
        }
        for index in 0..<actDim {
            let actionValue = index < action.count ? action[index] : 0
            inputStorage[capacityLimit + index] = oakSafeScalar(actionValue, clamp: -1...1)
        }
        inputStorage[inputWidth - 1] = 1
    }

    private func predictRow(row: Int, activeRank: Int) -> Float {
        let rowOffset = row * inputWidth
        var sum: Float = 0
        for column in 0..<inputWidth {
            sum += weights[rowOffset + column] * inputStorage[column]
        }
        predictionStorage[row] = sum
        return sum
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
        guard utility.indices.contains(index) else {
            return
        }
        utility[index] += abs(value)
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

    func plan(state: SharedFloatStorage, reactiveAction: SharedFloatStorage, activeRank: Int, budgetNs: UInt64) -> Double {
        guard budgetNs > 0, activeRank > 0 else {
            return 0
        }
        let start = oakCurrentTimeNanoseconds()
        var bestScore = -Double.greatestFiniteMagnitude
        var bestIndex = 0

        while oakCurrentTimeNanoseconds() - start < budgetNs {
            let optionIndex = cursor % activeRank
            cursor += 1

            let rowOffset = optionIndex * optionBank.actDim
            var candidateVector = Array(repeating: Float(0), count: optionBank.actDim)
            for column in 0..<optionBank.actDim {
                let blended = reactiveAction[column] + optionBank.weights[rowOffset + column]
                let bounded = oakSafeScalar(tanh(blended), clamp: -1...1)
                candidateAction[column] = bounded
                candidateVector[column] = bounded
            }

            let predictedReward = worldModel.predict(state: state, action: candidateVector, activeRank: activeRank, nextState: predictedState)
            let currentValue = control.evaluate(state: state, activeRank: activeRank)
            let nextValue = control.evaluate(state: predictedState, activeRank: activeRank)
            var optionNorm = 0.0
            for column in 0..<optionBank.actDim {
                let value = Double(optionBank.weights[rowOffset + column])
                optionNorm += value * value
            }
            optionNorm = max(sqrt(optionNorm), Double(Float.ulpOfOne))
            let score = Double(predictedReward - control.rewardRate + nextValue - currentValue) / (1 + optionNorm)
            optionBank.recordUtility(index: optionIndex, value: score)
            perception.recordUtility(index: optionIndex, value: score)

            let adapt = Float(1 / max(1 + abs(score), Double(Float.ulpOfOne)))
            let featureActivation = state[optionIndex]
            for column in 0..<optionBank.actDim {
                let candidate = candidateAction[column]
                optionBank.weights[rowOffset + column] += adapt * featureActivation * Float(score) * (1 - candidate * candidate)
            }

            if score > bestScore {
                bestScore = score
                bestIndex = optionIndex
            }

            if cursor >= activeRank, oakCurrentTimeNanoseconds() - start >= budgetNs {
                break
            }
        }

        let winnerOffset = bestIndex * optionBank.actDim
        let blendDenominator = Float(1 + max(bestScore.magnitude, Double(Float.ulpOfOne)))
        for column in 0..<optionBank.actDim {
            let target = optionBank.weights[winnerOffset + column]
            plannedBias[column] += (target - plannedBias[column]) / blendDenominator
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

    let signal: SharedFloatStorage
    let state: SharedFloatStorage
    let prevState: SharedFloatStorage
    let action: SharedFloatStorage
    let plannedBias: SharedFloatStorage
    let reconstruction: SharedFloatStorage
    let residual: SharedFloatStorage
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

        signal = SharedFloatStorage(count: max(signalDim, 1), device: nil)
        state = SharedFloatStorage(count: max(capacityLimit, 1), device: nil)
        prevState = SharedFloatStorage(count: max(capacityLimit, 1), device: nil)
        action = SharedFloatStorage(count: max(actDim, 1), device: nil)
        plannedBias = SharedFloatStorage(count: max(actDim, 1), device: device)
        reconstruction = SharedFloatStorage(count: max(signalDim, 1), device: nil)
        residual = SharedFloatStorage(count: max(signalDim, 1), device: nil)
        candidateAction = SharedFloatStorage(count: max(actDim, 1), device: nil)
        predictedState = SharedFloatStorage(count: max(capacityLimit, 1), device: nil)

        perception = AdaptivePerception(signalDim: signalDim, capacityLimit: capacityLimit, device: device, rng: rng)
        control = ReactiveControl(actDim: actDim, capacityLimit: capacityLimit, device: device, rng: rng)
        predictiveKnowledge = PredictiveKnowledge(capacityLimit: capacityLimit)
        worldModel = WorldModel(capacityLimit: capacityLimit, actDim: actDim)
        optionBank = OptionBank(capacityLimit: capacityLimit, actDim: actDim, rng: rng)
        searchControl = SearchControl(
            optionBank: optionBank,
            worldModel: worldModel,
            control: control,
            perception: perception,
            plannedBias: plannedBias,
            candidateAction: candidateAction,
            predictedState: predictedState
        )
        nsPerWorkUnitEstimate = 1
        snapshotToken = 0
    }

    func step(input: StepInput) -> StepOutput {
        let stepStartNs = oakCurrentTimeNanoseconds()
        loadSignal(obs: input.obs, prevAction: input.prevAction, prevReward: input.prevReward)

        let capacityDecision = decideCapacity(budgetNs: input.wallClockBudgetNs)
        let encoding = perception.encode(
            signal: signal,
            state: state,
            reconstruction: reconstruction,
            residual: residual,
            allowedRank: capacityDecision.allowedRank
        )
        let activeRank = perception.currentActiveRank(budget: capacityDecision)
        if perception.activeRank > activeRank {
            _ = perception.encode(
                signal: signal,
                state: state,
                reconstruction: reconstruction,
                residual: residual,
                allowedRank: activeRank
            )
        }

        let encodeStartMs = oakCurrentTimeMilliseconds()
        performActionInference(activeRank: perception.activeRank)
        let foregroundComputeMs = oakCurrentTimeMilliseconds() - encodeStartMs

        let tdDelta = control.update(prevState: prevState, state: state, prevAction: input.prevAction, reward: input.prevReward, activeRank: perception.activeRank)
        let gvfError = predictiveKnowledge.update(prevState: prevState, state: state, activeRank: perception.activeRank)
        let modelError = worldModel.update(prevState: prevState, state: state, prevAction: input.prevAction, reward: input.prevReward, activeRank: perception.activeRank)
        for index in 0..<perception.activeRank {
            let utilitySignal = Double(abs(tdDelta * prevState[index])) + gvfError + modelError + encoding.residualEnergy
            perception.recordUtility(index: index, value: utilitySignal)
            optionBank.recordUtility(index: index, value: utilitySignal)
        }

        let elapsedBeforePlanning = oakCurrentTimeNanoseconds() - stepStartNs
        let remainingBudgetNs = input.wallClockBudgetNs > elapsedBeforePlanning
            ? input.wallClockBudgetNs - elapsedBeforePlanning
            : 0
        let planningMs = searchControl.plan(
            state: state,
            reactiveAction: action,
            activeRank: perception.activeRank,
            budgetNs: remainingBudgetNs
        )

        for index in 0..<capacityLimit {
            prevState[index] = state[index]
        }

        snapshotToken &+= 1
        let stepNs = oakCurrentTimeNanoseconds() - stepStartNs
        let workUnits = max(Double(perception.activeRank * (signalDim + actDim + perception.activeRank)), 1)
        let measuredNsPerWorkUnit = Double(stepNs) / workUnits
        nsPerWorkUnitEstimate = (nsPerWorkUnitEstimate + measuredNsPerWorkUnit) / 2

        let telemetry = AgentTelemetry(
            rewardInput: input.prevReward,
            obs: input.obs,
            state: state.snapshot(prefix: perception.activeRank),
            action: action.snapshot(prefix: actDim),
            agentStepMs: Double(stepNs) / 1_000_000,
            backend: backendName,
            metalTimings: MetalCommandTimings(
                cmd1Ms: foregroundComputeMs,
                cmd2Ms: 0,
                cmd3Ms: 0,
                cmd4Ms: 0,
                cmd5Ms: planningMs
            ),
            plannerStepMs: planningMs,
            computeBudgetNs: input.wallClockBudgetNs,
            activeRank: perception.activeRank,
            activeGVFs: perception.activeRank,
            activeOptions: perception.activeRank,
            snapshotToken: snapshotToken
        )
        return StepOutput(action: action.snapshot(prefix: actDim), telemetry: telemetry, snapshotToken: snapshotToken)
    }

    func performActionInference(activeRank: Int) {
        control.actCPU(state: state, plannedBias: plannedBias, activeRank: activeRank, action: action)
    }

    private func decideCapacity(budgetNs: UInt64) -> CapacityDecision {
        let safeBudget = max(Double(budgetNs), 1)
        let budgetWorkUnits = safeBudget / max(nsPerWorkUnitEstimate, Double(Float.ulpOfOne))
        let dimensionality = Double(signalDim + actDim)
        let rankByBudget = Int(max(1, floor((sqrt(dimensionality * dimensionality + 4 * budgetWorkUnits) - dimensionality) / 2)))
        return CapacityDecision(
            allowedRank: max(1, min(capacityLimit, rankByBudget)),
            budgetWorkUnits: budgetWorkUnits
        )
    }

    private func loadSignal(obs: [Float], prevAction: [Float], prevReward: Float) {
        for index in 0..<obsDim {
            let value = index < obs.count ? obs[index] : 0
            signal[index] = oakSafeScalar(value)
        }
        for index in 0..<actDim {
            let value = index < prevAction.count ? prevAction[index] : 0
            signal[obsDim + index] = oakSafeScalar(value, clamp: -1...1)
        }
        signal[signalDim - 1] = oakSafeScalar(prevReward)
    }
}

final class CPUForegroundCore: BaseSelfScalingCore {
    init(obsDim: Int, actDim: Int, seed: Int?) {
        super.init(obsDim: obsDim, actDim: actDim, device: nil, backendName: "cpu", seed: seed)
    }
}

final class MetalForegroundCore: BaseSelfScalingCore {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let encodePipeline: MTLComputePipelineState
    private let actionPipeline: MTLComputePipelineState

    private let signalBuffer: SharedFloatStorage
    private let stateBuffer: SharedFloatStorage
    private let actionBuffer: SharedFloatStorage
    private let plannedBiasBuffer: SharedFloatStorage
    private let basisBuffer: SharedFloatStorage
    private let policyBuffer: SharedFloatStorage

    init(obsDim: Int, actDim: Int, device: MTLDevice, seed: Int?) {
        self.device = device
        queue = device.makeCommandQueue()!
        let kernelsURL = Bundle.module.url(forResource: "Kernels", withExtension: "metal")!
        let kernelsSource = try! String(contentsOf: kernelsURL, encoding: .utf8)
        let library = try! device.makeLibrary(source: kernelsSource, options: nil)
        encodePipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "encode_state")!)
        actionPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "policy_action")!)

        signalBuffer = SharedFloatStorage(count: max(obsDim + actDim + 1, 1), device: device)
        let capacityLimit = max(1, Int(ceil(sqrt(Double(max(obsDim + actDim + 1, 1))))))
        stateBuffer = SharedFloatStorage(count: max(capacityLimit, 1), device: device)
        actionBuffer = SharedFloatStorage(count: max(actDim, 1), device: device)
        plannedBiasBuffer = SharedFloatStorage(count: max(actDim, 1), device: device)
        basisBuffer = SharedFloatStorage(count: max((obsDim + actDim + 1) * capacityLimit, 1), device: device)
        policyBuffer = SharedFloatStorage(count: max(actDim * capacityLimit, 1), device: device)
        super.init(obsDim: obsDim, actDim: actDim, device: device, backendName: "metal", seed: seed)
    }

    override func performActionInference(activeRank: Int) {
        signalBuffer.copy(from: signal.snapshot(prefix: signal.count))
        plannedBiasBuffer.copy(from: plannedBias.snapshot(prefix: plannedBias.count))
        basisBuffer.copy(from: perception.basis.snapshot(prefix: basisBuffer.count))
        policyBuffer.copy(from: control.policy.snapshot(prefix: policyBuffer.count))

        let commandBuffer = queue.makeCommandBuffer()!

        let encodeEncoder = commandBuffer.makeComputeCommandEncoder()!
        encodeEncoder.setComputePipelineState(encodePipeline)
        encodeEncoder.setBuffer(basisBuffer.metalBuffer, offset: 0, index: 0)
        encodeEncoder.setBuffer(signalBuffer.metalBuffer, offset: 0, index: 1)
        encodeEncoder.setBuffer(stateBuffer.metalBuffer, offset: 0, index: 2)
        var signalDimValue = UInt32(signal.count)
        var activeRankValue = UInt32(activeRank)
        encodeEncoder.setBytes(&signalDimValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encodeEncoder.setBytes(&activeRankValue, length: MemoryLayout<UInt32>.stride, index: 4)
        let encodeThreads = MTLSize(width: max(activeRank, 1), height: 1, depth: 1)
        let encodeGroup = MTLSize(width: min(max(activeRank, 1), encodePipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        encodeEncoder.dispatchThreads(encodeThreads, threadsPerThreadgroup: encodeGroup)
        encodeEncoder.endEncoding()

        let actionEncoder = commandBuffer.makeComputeCommandEncoder()!
        actionEncoder.setComputePipelineState(actionPipeline)
        actionEncoder.setBuffer(policyBuffer.metalBuffer, offset: 0, index: 0)
        actionEncoder.setBuffer(stateBuffer.metalBuffer, offset: 0, index: 1)
        actionEncoder.setBuffer(plannedBiasBuffer.metalBuffer, offset: 0, index: 2)
        actionEncoder.setBuffer(actionBuffer.metalBuffer, offset: 0, index: 3)
        actionEncoder.setBytes(&activeRankValue, length: MemoryLayout<UInt32>.stride, index: 4)
        var actDimValue = UInt32(action.count)
        actionEncoder.setBytes(&actDimValue, length: MemoryLayout<UInt32>.stride, index: 5)
        var policyStride = UInt32(capacityLimit)
        actionEncoder.setBytes(&policyStride, length: MemoryLayout<UInt32>.stride, index: 6)
        let actionThreads = MTLSize(width: max(action.count, 1), height: 1, depth: 1)
        let actionGroup = MTLSize(width: min(max(action.count, 1), actionPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        actionEncoder.dispatchThreads(actionThreads, threadsPerThreadgroup: actionGroup)
        actionEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let encodedState = stateBuffer.snapshot(prefix: state.count)
        let encodedAction = actionBuffer.snapshot(prefix: action.count)
        state.copy(from: encodedState, count: state.count)
        action.copy(from: encodedAction, count: action.count)
    }
}
