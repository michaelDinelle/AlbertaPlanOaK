// swift-tools-version:6.2
import PackageDescription

let package = Package(
  name: "OaK",
  platforms: [.macOS(.v26)],
  targets: [
    .executableTarget(
      name: "OaK",
      path: "Sources/OaK",
      swiftSettings: [.unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK"])]
    )
  ]
)
