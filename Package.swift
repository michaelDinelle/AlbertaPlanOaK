// swift-tools-version:6.2
import PackageDescription

let package = Package(
  name: "OaK",
  platforms: [.macOS(.v26)],
  targets: [
    .executableTarget(
      name: "OaK",
      path: "Sources",
      exclude: ["OakKernel.metal", "OakPhysics.metal", "HalfCheetahModel.metal"]
    )
  ]
)
