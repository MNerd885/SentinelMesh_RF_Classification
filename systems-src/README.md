# Sentinel Mesh: runtime, energy and hardware-integration sources

This folder holds two groups of files.

1. **Validation models** (`ckpt_sim.c`, `energy_sim.py`): the code behind
   Sec. VII-A, VII-B and Annex A.5 to A.7 and A.11 of the report.
2. **First-draft sources** (the other files): the notebook snippets that the
   report calls "the first firmware draft".

## 1. Validation models

**Status: rewritten on 2026-11-24.** The original files were lost. Both models
were rewritten from the description in the report (Sec. V to VII, Annex A.3,
A.5 to A.7, A.10, A.11) and then checked against the numbers printed there.
They have not been peer reviewed.

| File | What it does | How to run |
|---|---|---|
| `ckpt_sim.c` | Fault injection on the checkpoint protocol | `gcc -O2 -Wall -o ckpt_sim ckpt_sim.c && ./ckpt_sim` |
| `energy_sim.py` | One-second energy simulation of the node | `python3 energy_sim.py` (add `fig` for the figure; needs matplotlib) |
| `fig3_six_day_trace.png` | Output of `python3 energy_sim.py fig` | |

### energy_sim.py against the report

Run on Python 3.13. Every value below equals the report.

- Table VI and the Annex A.7 table: all 14 cases, all six columns (awake share,
  bursts, alerts sent, missed, heartbeats, blackouts).
- Annex A.5 table: airtime, energy, silence and C_min for SF7 to SF12.
- Annex A.6 balance: 5.62 J harvest, 2.99 J sensing, 0.74 J alerts, 0.34 J
  heartbeats, 0.83 J floor, 0.83 J leakage.
- Annex A.7 night: lights off 19:00, 2.7 V at 19:32, 2.4 V at 20:07, stop at
  20:58, restart at 08:25, then 08:44, 09:10 and 10:08.
- Area sweep 44.1 / 53.7 / 58.5 %, six-day scenario 27.4 % awake with four
  checkpoints, mean sensing latency 4.6 min, 62 % awake at 0.1 mJ per frame.

Rules that the report does not spell out and that were fixed by matching its
numbers (all are stated in the code comments):

- Presence changes happen at 07:20 and then every 40 minutes (last at 22:40).
- The next burst is scheduled from the voltage tier *before* the burst, and
  the next heartbeat from the voltage *before* the packet.
- The harvest uses the rounded 130 uW (not 129.6 uW) and the packet the
  rounded 40.9 mJ.
- A checkpoint (15 uJ) is charged after each alert transmission. The report's
  A.6 line "about 11 checkpoints" counts only the blackout checkpoints.
- The restart threshold is the fixed 2.3 V for every capacitor size.

Small differences from the report text: the daily change of the store is
-0.09 J (the report says -0.10, rounded), and the longest sensing latency in
the 10-day run is 50 min (the report says "up to an hour").

### ckpt_sim.c against the report

Built and run on Windows 11 with `zig cc` 0.17.1 (`-O2 -Wall -Wextra`, no
warnings); the run takes about 10 s and exits with code 0. The file also
compiles without warnings with clang for a Linux target (`-std=c99 -Wall
-Wextra -pedantic -Wconversion`). It has not been built with gcc, which is the
command the report gives. A line-for-line Python port, written as an
independent check, gives the same counts.

- Table V: every cell equals the report. Commit-last 2926 / 2926 / 5814 / 307,
  total 11,973, none mixed. Magic-first 2774 (152 OK, 2622 mixed), 2774 (151,
  2623), 5510 (265, 5245); 10,490 of 11,058 mixed, 94.9 %.
- Random campaign, two million cycles for each of the three cut models (six
  million in total), seed 12345: no corrupt restore, no cold start, power fails
  inside the checkpoint in 4.51 % of cycles, progress lost at most 5 steps.

Points where the model and the report text differ or where a choice was made.
The first three describe the report up to draft v5; draft v6 corrects them
(mean 3.0, "for each cut model", and the wrap rule in Annex A.3):

- **Mean progress lost is 2.99 steps, the report says 2.9.** With 1 to 5 steps
  per cycle the expected mean is 3.0.
- **The report (A.11) says "38 restorations of the newest state".** The model
  gives 38 per cut model, so 114 over the three models.
- **Generation wrap.** The report does not say what follows generation 65535.
  Skipping only the invalid value 0 would give 1, which lives in the same slot
  as 65535, so a cut during that single write would roll back two checkpoints.
  The model wraps to 2 instead, which keeps the slots alternating. An extra
  check over the wrap (2456 cut points) passes; with a wrap to 1, 302 of them
  fail. This check is not part of the 11,973.
- **Test pattern.** The magic-first counts 151 and 265 depend on the test
  values: the pattern (`window[i] = gen + 0.5 * i`) was chosen so that they
  equal Table V. The commit-last counts do not depend on it.
- **Random generator.** PCG32 with seed 12345. The original generator is
  unknown, so the random sequence is not the original one.

## 2. First-draft sources

Recovered on 2026-11-24 from the project's Loop notebook page "Junaid". They
are the first draft that the report refers to (Sec. V-B, Sec. VIII, Annex A.4
and A.8) and are **not** the code behind the validation results.

| File | Notebook block | What it is |
|---|---|---|
| `power_budget.py` | "Python Code" | First power budget and supercapacitor sizing script. |
| `checkpoint_restore.c` | "Step 2: Checkpoint/Restore Firmware Routine" | First checkpoint/restore draft: one slot, magic word, no CRC. |
| `pin_map.mmd` | "Mermaid Code" | MCU to A121 and SX1262 wiring diagram (Mermaid source). |
| `radar_a121.c` | "A. Acconeer A121 Radar Abstraction Layer" | Radar driver stub. |
| `lora_sx1262.c` | SX1262 driver block | LoRa transmit stub. |
| `main_loop.c` | "single processing loop" block | State-machine main loop that ties the stubs together. |

Every line was checked against the notebook with a per-line hash and matches,
with two exceptions: `power_budget.py` was re-indented (the notebook copy had
lost all indentation), and one trailing blank line was dropped from
`radar_a121.c`.

Limits:

- The C files are notebook snippets, not a buildable project. The vendor HAL
  calls are placeholders, SPI transfers are commented out, `lora_sx1262.c` and
  `main_loop.c` have no includes, and both `checkpoint_restore.c` and
  `main_loop.c` define `main`. None of them was compiled or run.
- `checkpoint_restore.c` is the superseded design: a single slot with no CRC
  and no commit-last ordering.
- `power_budget.py` runs and prints 2.239 mW, 13.20 mJ and 0.2264 F. It uses
  early assumptions and treats the radar current as a power. Its 0.2264 F is
  the "226 mF coded" value mentioned in the report. It does not reproduce the
  report's 20.3 mF or 220 mF figures.
