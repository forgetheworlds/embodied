# PROVENANCE — appearance assets vendored for the first-indoor scene

Every appearance asset this scene loads is copied into `assets/` so the scene
root is self-contained. The ultimate source is the local Webots R2025a
installation (`work/Webots.app/Contents/projects/appearances/protos/`); the
copies were taken from the already-vendored tree at
`scenarios/missions/common/appearances/` (branch scenarios/indoor, 87a043c),
whose PROVENANCE.md records the original vendoring. Each copied `.proto`
keeps its upstream licence header verbatim:

```
# license: Apache License 2.0
# license url: https://www.apache.org/licenses/LICENSE-2.0
```

Apache-2.0 permits redistribution; that is why these assets are vendored. No
vendored file contains an `EXTERNPROTO` and none fetches anything over the
network:

```
grep -RIn 'EXTERNPROTO' scenarios/first_indoor/assets/   # must print nothing
```

Only the two appearances this world uses are vendored here (Plaster for walls
and ceiling, Parquetry "mosaic" for the floor) plus their texture subtrees;
the sibling catalogue scenes share a wider tree at
`scenarios/missions/common/appearances/` and are not referenced by this scene.

## Copies (sha256 of the file in this directory)

| File | sha256 | Copied from |
|---|---|---|
| protos/Plaster.proto | `570e6d4eed33ff8d9d50916846c9930f41e4959bd66ca0229936aff95a12d9b9` | `scenarios/missions/common/appearances/protos/Plaster.proto` |
| protos/Parquetry.proto | `0a7d8f9a7e99908bc32c439f40204887d1aecd27a6c897d723df7b1e048c6785` | `scenarios/missions/common/appearances/protos/Parquetry.proto` |
| protos/textures/plaster/* | byte-identical subtree | `scenarios/missions/common/appearances/protos/textures/plaster/` |
| protos/textures/parquetry/* | byte-identical subtree | `scenarios/missions/common/appearances/protos/textures/parquetry/` |

The two proto hashes match the table in
`scenarios/missions/common/PROVENANCE.md` line for line, which traces them to
the Webots R2025a installation paths recorded there.
