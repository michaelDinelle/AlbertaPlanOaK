// swift-tools-version:6.2
import PackageDescription

let package = Package(
  name: "OaK",
  platforms: [.macOS(.v26)],
  targets: [
    .executableTarget(
      name: "OaK",
      path: "Sources",
      exclude: ["OakKernel.metal"],
      linkerSettings: [
        .unsafeFlags([
          "-L", ".build/oak-native", "-loakenv",
          "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../oak-native",
        ])
      ]
    )
  ]
)
