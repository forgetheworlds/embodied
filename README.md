# embodied

A benchmark for the **onboard stack** of an agent flying a simulated drone. The agent is what is tested;
the simulator is not.

The subject is an avatar: a drone body plus the subsystems an agent acts through — estimator, inner loop,
outer loop, perception and agency. One run produces one report card: a task outcome plus a number for each
subsystem, all taken from the same episode log, so the numbers describe one flight rather than five
separate experiments.

**Status: designed, not built.** No implementation yet. The design rests on a verified evidence base.

---

## Why this is not just another drone project

Most agent benchmarks measure whether a task passed. That tells you nothing about *why*. This one measures
the parts, so a failure points at a subsystem instead of at "the agent".

Three decisions shape everything downstream:

- **The agent commands four propeller speeds and three gimbal angles.** There is no flight controller
  underneath it, so every loop from attitude upward is the agent's. The camera is deliberately not
  stabilised — steadying it is a decision the agent can make or fail to make.
- **Disturbance is on by default**, with a still-air run as the control. A loop that only works in still air
  is not a loop yet.
- **Results are report cards against a floor and a ceiling** — the shipped example controller sets the
  floor, two truth-fed oracles set the ceiling. Without a ceiling you cannot tell a good number from a
  mediocre one.

## What is notable about how it was built

The simulator claims are not assertions. A 74-claim ledger records, for each one, its evidence class, the
source, the exact quoted span, the date it was fetched, and what would prove it wrong. A separate
verification pass then tried to falsify the load-bearing claims and re-ran the benchmark independently
rather than trusting the notes: **33 confirmed, 7 corrected, 0 killed outright, 1 unverifiable.**

All four claims the project started from needed correcting. That is the point of checking.

The decisions live in `docs/adr/` — 14 of them, each recording the options **rejected** and what the choice
**costs**, not only what was chosen. `docs/OPEN-QUESTIONS.md` records what is still undecided, with the
constraints attached, so nothing is decided twice or against evidence.

---

## Why Webots

The shipped DJI Mavic 2 Pro model already has most of what is needed, and one of its example controllers
flew it for us.

- The model exposes a camera, GPS, gyro, inertial unit, compass, four propeller motors and three gimbal
  motors, all drivable from Python.
- The shipped patrol controller climbed to 15.7 m and held 14.99 m while following waypoints for 407
  seconds of simulated time.
- A Supervisor process on a second node read the drone's true position for the whole flight while the drone
  itself had no privileged access.
- Camera frames reached an outside Python program which Webots did not launch.

Four costs come with it:

- **No shipped controller reads a camera.** The Python one enables it and never looks, so there is no
  perception example to copy — the perception task starts from nothing.
- **ArduPilot works, but on its own terms.** It is pinned to Webots 2023a in source, and it takes over as
  the flight controller, so the agent would talk MAVLink while images arrive on a different socket.
- **PX4 has no Webots support.** The request was closed as stale, and the documentation never mentions it.
- **A Supervisor cannot read another robot's camera**, which constrains how ground truth is produced. This
  is an open design question, not a detail.

## The four tasks

| Task | What it tests | Baseline |
|---|---|---|
| **Hold and reject** | Hold a point for 60 s while the referee pushes and twists the drone | the shipped controller's altitude hold |
| **Waypoint chain** | Fly a chain of five waypoints | the shipped controller's patrol route |
| **Visual acquisition** | Find a marked object and point the camera at it | **none** — nothing shipped reads a camera |
| **Navigate then inspect** | Fly to a waypoint, then find and report the object near it | navigation from the shipped controller, inspection from nothing |

Not every task exercises every subsystem, so the report card carries a `not exercised` entry rather than
scoring a subsystem that was never called on. A zero there would report a failure that never happened.

---

## What is not in this repository

The design record is maintained privately and is **not** here: the architecture document, the decision
records, the evidence ledger and the raw research output all live outside this repository. This README is
the public face; the reasoning behind the code stays with the author.

What that means for a reader: claims below are traceable in principle but not verifiable from this
repository alone. That is a deliberate choice, not an oversight.

## What is not here yet

- **No implementation.** The build order starts with settling the substrate, then the forked model, then
  measuring the camera cost curve which sets the loop rates, then the referee, logging and scoring before
  any agent code, then the agent, then its ablations.
- **Three questions need a real run** before the design closes: how camera cost changes with frame rate,
  whether the forked model flies like the shipped one, and what the perception task scores with no agent.
- **Bars come after measurement.** There is no pass or fail in the first run; a threshold written before any
  measurement only encodes a guess.

## Caveats on the numbers

Frame rates were measured on an Apple M3 laptop rather than a controlled benchmark machine, and the same
configuration varied between runs — 216 to 798 fps, real-time factor 1.73 to 6.38, with host load reaching
12.88 on 8 cores. Absolute numbers do not transfer; ratios do. How idle the host must be is still an open
question, so treat every rate here as an indicator and every ratio as the finding.
