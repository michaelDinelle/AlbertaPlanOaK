// swift-tools-version:6.2
import PackageDescription

let package = Package(
  name: "OaK",
  platforms: [.macOS(.v26)],
  targets: [
    .executableTarget(
      name: "OaK",
      path: "Sources",
      exclude: ["OakKernel.metal", "OakPhysics.metal", "HalfCheetahModel.metal", "Gridworld.metal", "ClassicControl.metal", "Env.metal"],
      swiftSettings: [.unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK"])]
    )
  ]
)
