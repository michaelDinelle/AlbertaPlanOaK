import Foundation
import PythonKit

/// EnvPool's Python registration list — no per-domain logic beyond optional prefix filters.
public enum EnvPoolCatalog {
    /// Short CLI names that map to a single fixed Gymnasium id (EnvPool `make` id).
    public static let logicalNameToFixedEnvId: [String: String] = [
        "cheetah": "HalfCheetah-v4",
        "walker": "Walker2d-v4",
        "humanoid": "Humanoid-v4",
    ]

    /// Default prefix used when the logical name `minigrid` runs in suite mode (all matching registered ids).
    public static let minigridSuitePrefix = "MiniGrid-"

    /// All gym ids returned by `envpool.registration.list_all_envs()`, sorted.
    public static func registeredEnvIds() -> [String] {
        allRegisteredStrings()
    }

    /// Subset whose id has the given prefix (e.g. `"MiniGrid-"` for the MiniGrid family).
    public static func registeredEnvIds(withPrefix prefix: String) -> [String] {
        allRegisteredStrings().filter { $0.hasPrefix(prefix) }
    }

    private static func allRegisteredStrings() -> [String] {
        let runtime = PythonRuntime.shared
        runtime.require(modules: ["envpool"])
        let registration = Python.import("envpool.registration")
        let pyList = registration.list_all_envs()
        let length = Int(pyList.__len__()) ?? 0
        var ids: [String] = []
        for i in 0..<length {
            if let s = String(pyList[i]) {
                ids.append(s)
            }
        }
        return ids.sorted()
    }
}
