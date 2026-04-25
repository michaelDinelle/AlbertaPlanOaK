import Foundation

struct PolicyMeanBatch {
    let values: [Float]
    let dPolicy: Int
    let optionValues: [Float]?
    let selectedOption: Int?
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
    static func sampleAndScore(spec: [ActionComponent], numEnvs: Int, dPolicy: Int,
                               envActionWidth: Int,
                               policyMean: UnsafePointer<Float>,
                               optionPolicyMean: UnsafePointer<Float>?,
                               optionCount: Int,
                               envAction: UnsafeMutablePointer<Float>,
                               actionSignal: UnsafeMutablePointer<Float>,
                               scoreGrad: UnsafeMutablePointer<Float>,
                               optionScoreGrad: UnsafeMutablePointer<Float>?,
                               importance: UnsafeMutablePointer<Float>?,
                               importanceLog: UnsafeMutablePointer<Float>?,
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

        for r in 0..<numEnvs {
            let mBase = r * dPolicy
            let aBase = r * envActionWidth
            var pIdx = 0, aIdx = 0
            for k in 0..<max(optionCount, 0) {
                importance?[r * optionCount + k] = 0
                importanceLog?[r * optionCount + k] = 0
                optionLogProbScratch?[r * optionCount + k] = 0
            }
            var behaviorLogProb: Float = 0
            for component in spec {
                switch component {
                case .discrete(let n):
                    var maxLogit = -Float.greatestFiniteMagnitude
                    for d in 0..<n { maxLogit = max(maxLogit, policyMean[mBase + pIdx + d]) }
                    var bestIdx = 0
                    var bestKey = -Float.greatestFiniteMagnitude
                    for d in 0..<n {
                        let l = policyMean[mBase + pIdx + d]
                        let u = max(nextU01(&rngState), uniformLatticeMin)
                        let g = -log(-log(u))
                        let key = l + g
                        if key > bestKey { bestKey = key; bestIdx = d }
                    }
                    var z: Float = 0
                    for d in 0..<n { z += exp(policyMean[mBase + pIdx + d] - maxLogit) }
                    let invZ: Float = 1.0 / max(z, .leastNormalMagnitude)
                    for d in 0..<n {
                        let pi = exp(policyMean[mBase + pIdx + d] - maxLogit) * invZ
                        scoreGrad[mBase + pIdx + d] = ((d == bestIdx) ? 1.0 : 0.0) - pi
                        actionSignal[mBase + pIdx + d] = (d == bestIdx) ? 1.0 : 0.0
                    }
                    behaviorLogProb += log(max(exp(policyMean[mBase + pIdx + bestIdx] - maxLogit) * invZ,
                                               .leastNormalMagnitude))
                    if let opt = optionPolicyMean {
                        for k in 0..<optionCount {
                            let base = (r * optionCount + k) * dPolicy + pIdx
                            var optMax = -Float.greatestFiniteMagnitude
                            for d in 0..<n { optMax = max(optMax, opt[base + d]) }
                            var optZ: Float = 0
                            for d in 0..<n { optZ += exp(opt[base + d] - optMax) }
                            let invOptZ = 1.0 / max(optZ, .leastNormalMagnitude)
                            let p = exp(opt[base + bestIdx] - optMax) * invOptZ
                            optionLogProbScratch?[r * optionCount + k] += log(max(p, .leastNormalMagnitude))
                            if let osg = optionScoreGrad {
                                for d in 0..<n {
                                    let pi = exp(opt[base + d] - optMax) * invOptZ
                                    osg[base + d] = ((d == bestIdx) ? 1.0 : 0.0) - pi
                                }
                            }
                        }
                    }
                    envAction[aBase + aIdx] = Float(bestIdx)
                    pIdx += n; aIdx += 1

                case .box(let a, let low, let high):
                    let sigmaBase: Float = 1.0 / Float(max(dPolicy, 1))
                    let scaleBase = pIdx + a
                    for d in 0..<a {
                        let z = nextNormal(&rngState)
                        let mu = policyMean[mBase + pIdx + d]
                        let rawLogSigma = policyMean[mBase + scaleBase + d]
                        let logSigma = rawLogSigma.isFinite ? rawLogSigma : 0
                        let sigmaRaw = sigmaBase * exp(logSigma)
                        let sigma = (sigmaRaw.isFinite && sigmaRaw > 0) ? sigmaRaw : sigmaBase
                        let raw = mu + sigma * z
                        let clamped = max(low[d], min(high[d], raw.isFinite ? raw : 0))
                        envAction[aBase + aIdx + d] = clamped
                        actionSignal[mBase + pIdx + d] = clamped
                        actionSignal[mBase + scaleBase + d] = 0
                        scoreGrad[mBase + pIdx + d] = z * sigma
                        scoreGrad[mBase + scaleBase + d] = z * z - 1.0
                        behaviorLogProb += -log(max(sigma, .leastNormalMagnitude)) - 0.5 * z * z
                        if let opt = optionPolicyMean {
                            for k in 0..<optionCount {
                                let base = (r * optionCount + k) * dPolicy
                                let optMu = opt[base + pIdx + d]
                                let optRawLogSigma = opt[base + scaleBase + d]
                                let optLogSigma = optRawLogSigma.isFinite ? optRawLogSigma : 0
                                let optSigmaRaw = sigmaBase * exp(optLogSigma)
                                let optSigma = (optSigmaRaw.isFinite && optSigmaRaw > 0)
                                    ? optSigmaRaw : sigmaBase
                                let oz = (raw - optMu) / max(optSigma, .leastNormalMagnitude)
                                optionLogProbScratch?[r * optionCount + k] +=
                                    -log(max(optSigma, .leastNormalMagnitude)) - 0.5 * oz * oz
                                if let osg = optionScoreGrad {
                                    osg[base + pIdx + d] = oz * optSigma
                                    osg[base + scaleBase + d] = oz * oz - 1.0
                                }
                            }
                        }
                    }
                    pIdx += 2 * a; aIdx += a
                }
            }
            if optionCount > 0 {
                for k in 0..<optionCount {
                    let lp = optionPolicyMean == nil
                        ? behaviorLogProb
                        : (optionLogProbScratch?[r * optionCount + k] ?? behaviorLogProb)
                    let logRho = lp - behaviorLogProb
                    let rho = exp(logRho)
                    importance?[r * optionCount + k] = rho.isFinite ? rho : 0
                    importanceLog?[r * optionCount + k] = logRho.isFinite ? logRho : 0
                }
            }
        }
    }
}
