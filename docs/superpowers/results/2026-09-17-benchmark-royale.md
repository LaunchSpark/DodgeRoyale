# Benchmark results

Revision `b7facf78f5c7` (working tree dirty), Windows 10.

| Item | Value |
|---|---|
| revision | b7facf78f5c7dd42f58d0bf7f4368944362b78e4 |
| dirty | True |
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

## 8 envs

2 gym workers, 100 enemies, rollout 16 steps, minibatch 128, device `cuda:0`.

| Stage | Per batch | Env steps/s | Note |
|---|---|---|---|
| round trip (simulate + encode + transport) | 5.32 ms | 1,504 | one STEP answered by the gym; the Rust benchmark splits it further |
| inference (one batch forward) | 4.36 ms | 1,836 |  |
| optimization (one PPO update) | 86.81 ms | 1,474 | 16 steps x 8 envs, minibatch 128, 1 epoch |
| end to end (learn) | 17.71 ms | 452 | 256 env steps over 32 batch steps and 2 updates; 0.57 s total |

Rollout observations at 16 steps: 14 MiB. At 1,024 steps they would be 0.88 GiB, which is why this benchmark does not allocate one.
Peak host working set 1,585 MiB; peak device allocation 654 MiB.

## 64 envs

2 gym workers, 100 enemies, rollout 16 steps, minibatch 128, device `cuda:0`.

| Stage | Per batch | Env steps/s | Note |
|---|---|---|---|
| round trip (simulate + encode + transport) | 44.72 ms | 1,431 | one STEP answered by the gym; the Rust benchmark splits it further |
| inference (one batch forward) | 11.45 ms | 5,590 |  |
| optimization (one PPO update) | 629.17 ms | 1,628 | 16 steps x 64 envs, minibatch 128, 1 epoch |
| end to end (learn) | 110.49 ms | 579 | 2048 env steps over 32 batch steps and 2 updates; 3.54 s total |

Rollout observations at 16 steps: 112 MiB. At 1,024 steps they would be 7.03 GiB, which is why this benchmark does not allocate one.
Peak host working set 1,825 MiB; peak device allocation 669 MiB.

## Baselines

Fixed policies over the same bounded seed suite [11, 12, 13, 14]. A timeout is a censored survival observation, not evidence of skill.

| Policy | Episodes | Mean frames | Median | Deaths | Timeouts | Censored |
|---|---|---|---|---|---|---|
| idle | 16 | 283.8 | 276.0 | 16 | 0 | 0% |
| random | 16 | 292.4 | 299.0 | 16 | 0 | 0% |


## The Rust side, split further

`cargo bench --locked --no-default-features --bench gym_throughput` on the same
revision and machine. The gym answers a STEP by simulating, encoding and
writing, and the protocol cannot report those separately, so the round-trip row
above is split here instead. Eight workers, 100 enemies.

| Stage | Per env step | 8 envs / batch | 64 envs / batch |
|---|---|---|---|
| Simulate | 541 us | | |
| Encode | 68 us | | |
| Batch step (simulate + encode, 8 workers) | 191-283 us | 2.26 ms | 12.24 ms |
| Transfer (write, pipe, parse) | 209-238 us | 1.68 ms | 15.26 ms |
| The same bytes, one `write_all` | 18-24 us | 0.14 ms | 1.53 ms |

## Reading these numbers

**The `dirty` flag above covers documentation edited while the benchmark ran,
not the code under test.** The measured revision is committed.

**Simulation dominates the Rust side.** At 541 us an env step against 68 us to
encode, the arena is eight times the encoder. Nothing here says whether that is
the 4,950 pairwise collision checks at 100 enemies or Bevy's per-`update`
overhead, and that remains unprofiled.

**Transport no longer degrades with batch size.** The round trip holds 1,504
env-steps/s at 8 envs and 1,431 at 64. Before this revision it collapsed to 117
at 64 envs, because the client read the child's pipe unbuffered and a 7 MB
batch became thousands of small reads. That is worth stating because the
failure was silent: the 8-env case looked fine.

**Optimization is the wall at scale, not the arenas.** One PPO update over a
64-env rollout costs 629 ms against 45 ms to collect a batch. End to end that
is 579 env-steps/s at 64 envs and 452 at 8, so sixty-four environments buy
about 28% more throughput than eight, not eight times more. Rollout collection
is no longer the limit; the update is.

**Memory is bounded by choice, not by headroom.** These runs use 16-step
rollouts. At the 1,024-step default a 64-env rollout would hold 7.03 GiB of
observations before any training tensors, against 15.9 GiB of host RAM, which
is why 8 envs is the training default and 64 is a benchmark configuration. Peak
device allocation stayed at 669 MiB of the GTX 1070's 8 GiB, so the GPU is not
the constraint at this scale.

**The baselines are the bar to beat.** An idle player survives 284 frames on
average, a uniformly random one 292 -- 4.7 and 4.9 seconds. Every episode ended
in a death and none hit the 3,600-frame budget, so neither figure is censored
and both are honest survival times. A policy that has learned anything must
clear roughly 290 frames; one that reports 3,600 has hit the budget, which is a
censored observation rather than a result.

## Recommended next experiment

Nothing here justifies changing the observation contract. The two measurements
that would, in order:

1. **Profile the 541 us simulation step.** It is the largest single cost in the
   system and the least understood. Splitting collision checks from Bevy's
   scheduling overhead decides whether spatial indexing is worth building.
2. **Re-measure optimization against minibatch size.** At 64 envs the update is
   fourteen times the collection. The 128 cap was chosen for memory, not for
   speed, and its effect on wall time has never been measured.

Compression stays a follow-up. The bulk-copy codec already landed; packing
channels 0-2 and 4 would save a further 8.4 ms per 64-env batch against a
629 ms update, which is not where the time goes.
