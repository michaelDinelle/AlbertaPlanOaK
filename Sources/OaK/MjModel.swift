import Foundation

struct MjEntry {
  let nameHash: UInt32
  let byteOffset: Int
  let nq, nv, nu, nbody, ngeom, njnt, npair, nconMax, nefcMax, nM: Int
  var stateWords: Int { nq + nv + nu + nv }
  var scratchWords: Int {
    MjBlob.scratchWords(
      nv: nv, nbody: nbody, ngeom: ngeom, nconMax: nconMax, nefcMax: nefcMax, nM: nM)
  }
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
    var entries: [MjEntry] = []
    for i in 0..<count {
      let base = 2 + i * 12
      entries.append(
        MjEntry(
          nameHash: words[base], byteOffset: Int(words[base + 1]),
          nq: Int(words[base + 2]), nv: Int(words[base + 3]), nu: Int(words[base + 4]),
          nbody: Int(words[base + 5]), ngeom: Int(words[base + 6]), njnt: Int(words[base + 7]),
          npair: Int(words[base + 8]), nconMax: Int(words[base + 9]),
          nefcMax: Int(words[base + 10]), nM: Int(words[base + 11])))
    }
    return MjBlob(data: data, entries: entries)
  }

  func entry(nameHash: UInt32) -> MjEntry? {
    entries.first { $0.nameHash == nameHash }
  }
}
