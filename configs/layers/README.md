# Layer live-sim verifications

One executable per stack layer. Name matches `src/embodied/<layer>/`,
`tests/test_<layer>.py`, and `docs/<layer>.md`.

| Script | Proves |
|---|---|
| `./configs/layers/control` | Vehicle GUIDED primitives through doorways |
| `./configs/layers/control-smooth` | Throwaway continuous cruise (not the gate) |

These are **not** pytest. Each script brings up Webots + SITL, flies the
layer route, and exits 0 only when the receipt gate passes.
