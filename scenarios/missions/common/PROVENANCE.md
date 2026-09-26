# PROVENANCE — vendored appearance protos for the indoor scenario catalogue

Every appearance asset used by the worlds under `scenarios/missions/{dev,holdout}/` is
copied here from the local Webots R2025a installation. The source tree is
`work/Webots.app/Contents/projects/appearances/protos/`. Each copied `.proto` keeps its
upstream licence header verbatim:

```
# license: Apache License 2.0
# license url: https://www.apache.org/licenses/LICENSE-2.0
```

Apache-2.0 permits redistribution; that is why these assets are vendored and why the
Cyberbotics *object* protos (`Wall`, `Door`, `Desk`, …, licensed "for use only with
Webots" with an unretrievable licence text) are **not** used anywhere in this catalogue.
The `# documentation url:` header lines inside the copied files are comments left
untouched; no vendored file contains an `EXTERNPROTO` and no vendored file fetches a
texture over the network — the licence-gate command for the whole tree is:

```
grep -RIn 'EXTERNPROTO' scenarios/missions/common/appearances/   # must print nothing
```

## Protos (sha256 of the vendored copy, origin relative to `work/Webots.app/Contents/projects/appearances/protos/`)

| Proto | sha256 | Origin |
|---|---|---|
| CementTiles.proto | `0dee4ab15e9dac95c075cc16a8c8ed5d39f1cf982d38cf674b6d04f7ffb7326d` | `protos/CementTiles.proto` |
| ChequeredParquetry.proto | `4eb545373db7b7eb4a2bde64680fe1c7353e5c17c3fe0bf76046b250412a457f` | `protos/ChequeredParquetry.proto` |
| DarkParquetry.proto | `772fdc1dc50dd7b1a4c4124d6ec44dc379dfa587f133cfa7cec98458e606d31e` | `protos/DarkParquetry.proto` |
| Marble.proto | `0035f3026fcf73b9f60e1eb632daee35ea782ef8f0c95e98d75cae05882d4ef1` | `protos/Marble.proto` |
| PaintedWood.proto | `f17d891125c38c0fb746b560447d05ba7d3a2780405f3882da0055259360a4f3` | `protos/PaintedWood.proto` |
| Parquetry.proto | `0a7d8f9a7e99908bc32c439f40204887d1aecd27a6c897d723df7b1e048c6785` | `protos/Parquetry.proto` |
| Plaster.proto | `570e6d4eed33ff8d9d50916846c9930f41e4959bd66ca0229936aff95a12d9b9` | `protos/Plaster.proto` |
| PorcelainChevronTiles.proto | `4c626fd73443bc84d7e8631063755a117e7bca9f77f35882a42ac428f8f69281` | `protos/PorcelainChevronTiles.proto` |
| RedBricks.proto | `75155d4daee9c2a9d39b63c1ce70ee601b131ddfb4539d5439040945f9a488e7` | `protos/RedBricks.proto` |
| Roughcast.proto | `8fde202185edf87c0c7e85725d2740a61fb6a331b58a2349d0e180a87f49b291` | `protos/Roughcast.proto` |

## Textures (copied `textures/` subtrees, relative URLs preserved)

Copied whole, because the vendored protos address them as `textures/<family>/…` relative
to their own directory. `tree16` below is the first 16 hex characters of the sha256 over
the sorted per-file sha256 list of the subtree — a re-runnable integrity check, not a
security claim.

| Subtree | Files | tree16 | Origin |
|---|---|---|---|
| `textures/cement_tiles/` | 4 | `3347c34900298137` | `protos/textures/cement_tiles/` |
| `textures/marble/` | 4 | `a9c5b460bcfec608` | `protos/textures/marble/` |
| `textures/painted_wood/` | 3 | `cddd525b459bd571` | `protos/textures/painted_wood/` |
| `textures/parquetry/` | 14 | `f1f2b6afcd13fb53` | `protos/textures/parquetry/` |
| `textures/plaster/` | 4 | `f2df95c7ceda7da2` | `protos/textures/plaster/` |
| `textures/porcelain_chevron/` | 4 | `61fdcc7f6501aa92` | `protos/textures/porcelain_chevron/` |
| `textures/red_bricks/` | 4 | `5d232cb5da70a57d` | `protos/textures/red_bricks/` |
| `textures/roughcast/` | 1 | `9a5acda8086b2213` | `protos/textures/roughcast/` |

`textures/parquetry/` is copied in full (all four `type` variants) because
`Parquetry.proto` is a template that selects a variant at load time; only `mosaic`,
`dark_strip` and `chequered` are used by this catalogue.

## Not vendored

* The vehicle: every world references `../../../compat/protos/Iris.proto` unchanged
  (origin: ArduPilot `Webots_Python` example, recorded in
  `scenarios/compat/worlds/compat_stereo.wbt`).
* Every stock Cyberbotics `objects/` proto (licence gap, see
  `work/runs/scenarios/env-research.md` §3.1).
* Any remote URL: no `EXTERNPROTO` in this tree, no texture URL outside this tree.
