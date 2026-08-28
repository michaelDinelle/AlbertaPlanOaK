import Foundation

struct MjEntry {
  let nameHash: UInt32
  let stateWords: Int
  let scratchWords: Int
}

struct MjBlob {
  let data: Data
  let entries: [MjEntry]

  static func scratchWords(
    nv: Int, nbody: Int, ngeom: Int, nconMax: Int, nefcMax: Int, nM: Int
  ) -> Int {
    return 60 * nbody + 28 * nv + 6 * ngeom + 40 + 2 * nM + 27 * max(nconMax, 1)
      + nefcMax * (nv + 3)
  }

  static func load() throws -> MjBlob {
    guard let url = Bundle.module.url(forResource: "mjmodels", withExtension: "bin") else {
      throw err("mjmodels.bin missing from bundle.")
    }
    let data = try Data(contentsOf: url)
    let words = data.withUnsafeBytes { $0.bindMemory(to: UInt32.self) }
    guard data.count >= 8, words[0] == 0x4C4D_4A4D else {
      throw err("mjmodels.bin header is not MJML.")
    }
    let count = Int(words[1])
    guard count <= (data.count - 8) / 48 else {
      throw err("mjmodels.bin model table exceeds the file.")
    }
    let tableBytes = 8 + count * 48
    var entries: [MjEntry] = []
    var previousOffset = tableBytes
    for i in 0..<count {
      let base = 2 + i * 12
      let byteOffset = Int(words[base + 1])
      let ordered = i == 0 ? byteOffset >= previousOffset : byteOffset > previousOffset
      guard ordered, byteOffset % 4 == 0, byteOffset < data.count else {
        throw err("mjmodels.bin model (i) has an invalid payload offset.")
      }
      let nq = Int(words[base + 2])
      let nv = Int(words[base + 3])
      let nu = Int(words[base + 4])
      let nbody = Int(words[base + 5])
      let ngeom = Int(words[base + 6])
      let nconMax = Int(words[base + 9])
      let nefcMax = Int(words[base + 10])
      let nM = Int(words[base + 11])
      entries.append(
        MjEntry(
          nameHash: words[base], stateWords: nq + 2 * nv + nu,
          scratchWords: scratchWords(
            nv: nv, nbody: nbody, ngeom: ngeom, nconMax: nconMax,
            nefcMax: nefcMax, nM: nM)))
      previousOffset = byteOffset
    }
    return MjBlob(data: data, entries: entries)
  }
}
