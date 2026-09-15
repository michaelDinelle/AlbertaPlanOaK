import Foundation

struct MjBlob {
  let data: Data
  let nameHash: UInt32
  let stateWords: Int

  static func load() throws -> MjBlob {
    guard let url = Bundle.module.url(forResource: "mjmodels", withExtension: "bin") else {
      throw err("mjmodels.bin missing from bundle.")
    }
    let data = try Data(contentsOf: url)
    let words = data.withUnsafeBytes { $0.bindMemory(to: UInt32.self) }
    guard data.count >= 56, words[0] == 0x4C4D_4A4D else {
      throw err("mjmodels.bin header is not MJML.")
    }
    guard words[1] == 1 else { throw err("mjmodels.bin must hold exactly one model, found \(words[1]).") }
    let byteOffset = Int(words[3])
    guard byteOffset >= 56, byteOffset % 4 == 0, byteOffset < data.count else {
      throw err("mjmodels.bin model has an invalid payload offset.")
    }
    let nq = Int(words[4]), nv = Int(words[5]), nu = Int(words[6])
    return MjBlob(data: data, nameHash: words[2], stateWords: nq + 2 * nv + nu)
  }
}
