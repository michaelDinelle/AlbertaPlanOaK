import Foundation

struct PolicyMeanBatch {
    let values: [Float]
    let optionValues: [Float]?
}

enum ActionComponent {
    case discrete(n: Int)
    case box(n: Int, low: [Float], high: [Float])

    var paramCount: Int {
        switch self {
        case .discrete(let n): return n
        case .box(let n, _, _): return 2 * n
        }
    }

    var envActionWidth: Int {
        switch self {
        case .discrete: return 1
        case .box(let n, _, _): return n
        }
    }
}

extension Sequence where Element == ActionComponent {
    var totalPolicyParamCount: Int { reduce(0) { $0 + $1.paramCount } }
    var totalEnvActionWidth: Int { reduce(0) { $0 + $1.envActionWidth } }
}

enum ActionTransducer {
    static func sampleAndScore(spec: [ActionComponent], dPolicy: Int,
                               envActionWidth: Int,
                               policyMean: UnsafePointer<Float>,
                               optionPolicyMean: UnsafePointer<Float>?,
                               optionCount: Int,
                               envAction: UnsafeMutablePointer<Float>,
                               actionSignal: UnsafeMutablePointer<Float>,
                               scoreGrad: UnsafeMutablePointer<Float>,
                               optionScoreGrad: UnsafeMutablePointer<Float>?,
                               importance: UnsafeMutablePointer<Float>?,
                               optionLogProbScratch: UnsafeMutablePointer<Float>?,
                               rngState: inout UInt64) {
        let uniformLatticeMin = 1.0 / Float(1 << 24)
        func nextU01(_ s: inout UInt64) -> Float {
            s &+= 0x9e3779b97f4a7c15
            var z = s
            z = (z ^ (z &>> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z &>> 27)) &* 0x94d049bb133111eb
            z ^= z &>> 31
            return Float(z >> 40) * (1.0 / Float(1 << 24))
        }

        func nextNormal(_ s: inout UInt64) -> Float {
            let u1 = max(nextU01(&s), uniformLatticeMin)
            let u2 = nextU01(&s)
            return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
        }

        var pIdx = 0, aIdx = 0
        for k in 0..<max(optionCount, 0) {
            importance?[k] = 0
            optionLogProbScratch?[k] = 0
        }
        var behaviorLogProb: Float = 0
        for component in spec {
            switch component {
            case .discrete(let n):
                var maxLogit = -Float.greatestFiniteMagnitude
                for d in 0..<n { maxLogit = max(maxLogit, policyMean[pIdx + d]) }
                var bestIdx = 0
                var bestKey = -Float.greatestFiniteMagnitude
                for d in 0..<n {
                    let u = max(nextU01(&rngState), uniformLatticeMin)
                    let key = policyMean[pIdx + d] - log(-log(u))
                    if key > bestKey { bestKey = key; bestIdx = d }
                }
                var z: Float = 0
                for d in 0..<n { z += exp(policyMean[pIdx + d] - maxLogit) }
                let invZ: Float = 1.0 / max(z, .leastNormalMagnitude)
                for d in 0..<n {
                    let pi = exp(policyMean[pIdx + d] - maxLogit) * invZ
                    scoreGrad[pIdx + d] = ((d == bestIdx) ? 1.0 : 0.0) - pi
                    actionSignal[pIdx + d] = (d == bestIdx) ? 1.0 : 0.0
                }
                behaviorLogProb += log(max(exp(policyMean[pIdx + bestIdx] - maxLogit) * invZ,
                                           .leastNormalMagnitude))
                if let opt = optionPolicyMean {
                    for k in 0..<optionCount {
                        let base = k * dPolicy + pIdx
                        var optMax = -Float.greatestFiniteMagnitude
                        for d in 0..<n { optMax = max(optMax, opt[base + d]) }
                        var optZ: Float = 0
                        for d in 0..<n { optZ += exp(opt[base + d] - optMax) }
                        let invOptZ = 1.0 / max(optZ, .leastNormalMagnitude)
                        let p = exp(opt[base + bestIdx] - optMax) * invOptZ
                        optionLogProbScratch?[k] += log(max(p, .leastNormalMagnitude))
                        if let osg = optionScoreGrad {
                            for d in 0..<n {
                                let pi = exp(opt[base + d] - optMax) * invOptZ
                                osg[base + d] = ((d == bestIdx) ? 1.0 : 0.0) - pi
                            }
                        }
                    }
                }
                envAction[aIdx] = Float(bestIdx)
                pIdx += n; aIdx += 1

            case .box(let a, let low, let high):
                let sigmaBase: Float = 1.0 / Float(max(dPolicy, 1))
                let scaleBase = pIdx + a
                for d in 0..<a {
                    let z = nextNormal(&rngState)
                    let mu = policyMean[pIdx + d]
                    let sigma = sigmaBase
                    let raw = mu + sigma * z
                    let clamped = max(low[d], min(high[d], raw.isFinite ? raw : 0))
                    envAction[aIdx + d] = clamped
                    actionSignal[pIdx + d] = clamped
                    actionSignal[scaleBase + d] = 0
                    scoreGrad[pIdx + d] = z * sigma
                    scoreGrad[scaleBase + d] = 0
                    behaviorLogProb += -log(sigma) - 0.5 * z * z
                    if let opt = optionPolicyMean {
                        for k in 0..<optionCount {
                            let base = k * dPolicy
                            let optMu = opt[base + pIdx + d]
                            let oz = (raw - optMu) / sigma
                            optionLogProbScratch?[k] += -log(sigma) - 0.5 * oz * oz
                            if let osg = optionScoreGrad {
                                osg[base + pIdx + d] = oz * sigma
                                osg[base + scaleBase + d] = 0
                            }
                        }
                    }
                }
                pIdx += 2 * a; aIdx += a
            }
        }
        for k in 0..<optionCount {
            let lp = optionPolicyMean == nil ? behaviorLogProb : (optionLogProbScratch?[k] ?? behaviorLogProb)
            importance?[k] = exp(lp - behaviorLogProb)
        }
    }
}
