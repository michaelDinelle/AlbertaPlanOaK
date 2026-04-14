// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "OaK",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(url: "https://github.com/pvieito/PythonKit", from: "0.5.0")
    ],
    targets: [
        .executableTarget(
            name: "OaK",
            dependencies: [
                "PythonKit"
            ],
            path: "Sources",
            exclude: ["OakKernel.metal"]
        ),
    ]
)
