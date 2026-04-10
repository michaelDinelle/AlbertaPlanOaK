// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "OaK",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(url: "https://github.com/pvieito/PythonKit", from: "0.5.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.14.1")
    ],
    targets: [
        .executableTarget(
            name: "OaK",
            dependencies: [
                "PythonKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift")
            ],
            path: "Sources",
            resources: [.process("Kernels.metal")]
        ),
        .testTarget(
            name: "OaKTests",
            dependencies: ["OaK"],
            path: "Tests/OaKTests"
        ),
    ]
)
