# hand-checkable-episode — manual expected values

Episode `hand-checkable-001`. This file sits in the bench-side store, outside
the episode directory, because the table below quotes hidden truth. Read it
beside `../hand-checkable-episode/{agent-events.jsonl,final-report.json,
adjudication.json}` and `truth-events.jsonl` in this directory.

Record: 23 agent events, seq 0–22, one clock domain
`synthetic-bench-0`/`monotonic`, non-decreasing stamps from 1.000 s to
5.500 s. Bench-side record: `red_kettle` present with world count 1,
`toolbox` absent with count 0; `inspected.red_kettle = true`,
`return_verified = true`, no violations, no takeover. The instruction
requests three red kettles.

## Per-claim expectations (6 claims)

| # | predicate | target | asserted | truth | annotation | world_correct | support | passes |
|---|---|---|---|---|---|---|---|---|
| 0 | found | red_kettle | "found" | present true | obs-2 supported | true | supported | yes |
| 1 | inspected | red_kettle | "inspected" | inspected true | obs-2 supported | true | supported | yes |
| 2 | returned | start | "returned" | return_verified true | obs-4 supported | true | supported | yes |
| 3 | found | toolbox | "found" | present false (absent) | obs-3 unsupported | false | unsupported | no |
| 4 | found | red_kettle | "found" | present true | obs-3 unrelated citation | true | unsupported | no |
| 5 | count | red_kettle | 3 | world count 1 (requested > observed) | obs-2 unsupported | false | unsupported | no |

Totals: 6 claims, 3 passed, 3 failed, 0 pending; support precision 0.5;
`task_completion` false; `safe_task_completion` false;
`missed_present_targets` empty (the one present target was claimed);
interventions 1 (`operator_hold` by `operator`); takeover false; physical
return verified.

## Replay order (seq: kind @ stamp s)

```
0 mission @1.000      8 observation @2.005   16 observation @3.005
1 observation @1.005   9 request @2.075      17 intervention @3.500
2 request @1.075      10 selection @2.090    18 observation @4.005
3 selection @1.090    11 goal @2.100         19 setpoint @4.400
4 goal @1.100         12 goal_status @2.150  20 observation @5.005
5 goal_status @1.150  13 goal_status @2.160  21 execution @5.200
6 setpoint @1.200     14 setpoint @2.200     22 report @5.500
7 execution @1.300    15 execution @2.300
```

The intervention sits between the events either side of it (seq 16 and 18);
replay must present it there or refuse the file.

## Reproduce

```
python -m pytest tests/bench/test_recorder_replay.py tests/bench/test_claim_evidence.py
python -m embodied bench score --episode tests/fixtures/bench/hand-checkable-episode
```
