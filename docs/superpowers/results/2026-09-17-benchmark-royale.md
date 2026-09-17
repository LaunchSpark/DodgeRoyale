# Benchmark results

Revision `39c665c683f0`, Windows 10.

| Item | Value |
|---|---|
| revision | 39c665c683f02445ff67f2d3986297bba56187a1 |
| dirty | False |
| python | 3.12.2 |
| torch | 2.14.0+cu126 |
| cuda available | True |
| device name | NVIDIA GeForce GTX 1070 |
| cpu | Intel64 Family 6 Model 158 Stepping 9, GenuineIntel |
| logical cores | 8 |
| ram gib | 15.9 |
| os | Windows 10 |
| rustc | rustc 1.95.0 (59807616e 2026-04-14) |
| binary | D:\projects\DodgeRoyale-vfr\target\release\dodge-royale.exe |
| dirty files | none |
| diff sha256 | None |
| diff path | None |

## 8 envs

2 gym workers, 100 enemies, rollout 16 steps, minibatch 128, device `cuda:0`.

| Stage | Per batch | Env steps/s | Note |
|---|---|---|---|
| round trip (simulate + encode + transport) | 6.75 ms | 1,184 | one STEP answered by the gym; the Rust benchmark splits it further |
| inference (one batch forward) | 4.67 ms | 1,713 |  |
| optimization (one PPO update) | 91.79 ms | 1,395 | 16 steps x 8 envs, minibatch 128, 1 epoch |
|   of which: parsing the reply | 0.13 ms | 63,114 | part of the round trip, decoded from memory so no waiting is counted |
| end to end (learn) | 19.43 ms | 412 | 256 env steps over 32 batch steps and 2 updates; 0.62 s total |

One rollout of 16 steps and the update that follows it, 128 env steps of work, 275 ms:

| Stage of one cycle | Time | Share |
|---|---|---|
| collection: round trip | 108 ms | 39% |
| collection: inference | 75 ms | 27% |
| optimization: one update | 92 ms | 33% |

Rollout observations at 16 steps: 14 MiB. At 1,024 steps they would be 0.88 GiB, which is why this benchmark does not allocate one.
Peak host working set 1,589 MiB; peak device allocation 654 MiB.

## 64 envs

2 gym workers, 100 enemies, rollout 16 steps, minibatch 128, device `cuda:0`.

| Stage | Per batch | Env steps/s | Note |
|---|---|---|---|
| round trip (simulate + encode + transport) | 47.96 ms | 1,335 | one STEP answered by the gym; the Rust benchmark splits it further |
| inference (one batch forward) | 12.45 ms | 5,141 |  |
| optimization (one PPO update) | 713.01 ms | 1,436 | 16 steps x 64 envs, minibatch 128, 1 epoch |
|   of which: parsing the reply | 4.85 ms | 13,189 | part of the round trip, decoded from memory so no waiting is counted |
| end to end (learn) | 116.73 ms | 548 | 2048 env steps over 32 batch steps and 2 updates; 3.74 s total |

One rollout of 16 steps and the update that follows it, 1024 env steps of work, 1,679 ms:

| Stage of one cycle | Time | Share |
|---|---|---|
| collection: round trip | 767 ms | 46% |
| collection: inference | 199 ms | 12% |
| optimization: one update | 713 ms | 42% |

Rollout observations at 16 steps: 112 MiB. At 1,024 steps they would be 7.03 GiB, which is why this benchmark does not allocate one.
Peak host working set 1,829 MiB; peak device allocation 669 MiB.

## Baselines

Fixed policies over the same bounded seed suite [11, 12, 13, 14]. A timeout is a censored survival observation, not evidence of skill.

| Policy | Episodes | Mean frames | Median | Deaths | Timeouts | Censored |
|---|---|---|---|---|---|---|
| idle | 16 | 283.8 | 276.0 | 16 | 0 | 0% |
| random | 16 | 292.4 | 299.0 | 16 | 0 | 0% |
