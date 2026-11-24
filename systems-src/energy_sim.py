#!/usr/bin/env python3
"""
energy_sim.py - design-level energy simulation of the Sentinel Mesh leaf node.

The node is a supercapacitor fed by indoor photovoltaic cells. The model
advances in one-second steps. Per step it adds the harvest if the room is lit,
subtracts the standing floor and the capacitor leakage, and clamps the store at
3.3 V. While the node is on it sends queued alerts and heartbeats and runs
sensing bursts, each only if the admission rule of Eq. (1) holds. At 2.2 V the
node pays the checkpoint energy and turns off; at V_on it pays the boot energy
and turns on. The model has no noise, no efficiency curves and no temperature,
and it is deterministic.

Usage:
    python3 energy_sim.py          print all tables
    python3 energy_sim.py fig      also write the six-day trace figure (needs matplotlib)

Section, table and R-numbers in the comments refer to the report
"Sentinel Mesh: Batteryless, Camera-Free Presence Sensing with Intermittent
Computing".
"""

import math
import sys

# ----------------------------------------------------------------------------
# Constants (report Sec. V, VI and Annex A.1)
# ----------------------------------------------------------------------------
V_MAX = 3.3            # storage clamp (R1)
V_CKPT = 2.2           # checkpoint threshold, V_min + guard (R2, R3)
V_HYST = 0.1           # detector hysteresis (R4)
V_ON = V_CKPT + V_HYST  # restart threshold; the hysteresis term of Eq. (2)
K_SAFETY = 1.5         # safety factor k of Eq. (1)
R_ESR = 0.1            # supercapacitor ESR in ohm (R10)

AREA_DESIGN = 21.6     # cm2 of PV cells
LUX_DESIGN = 200.0
# 21.6 cm2 x 8 uW/cm2 x 0.75 = 129.6 uW. The report rounds this to 130 uW and
# the model uses the rounded value, scaled linearly with area and illuminance.
P_HARVEST_DESIGN = 130e-6

E_FRAME = 1.1e-3       # one radar frame with host processing (R13)
E_CLASSIFIER = 100e-6  # classifier budget, once per burst (R14)
T_FRAME = 0.1          # 10 Hz frames
E_CKPT = 15e-6         # one checkpoint (R24)
E_BOOT = 20e-6         # boot and restore (R25)
P_FLOOR = 9.6e-6       # stop mode + charger + monitor (R26), applied all day
LEAK_PER_100MF = 2e-6  # supercapacitor leakage in A per 100 mF (R9)

I_TX = 0.1             # LoRa transmit current in A (R15)
V_RADIO = 3.3
SF = 9                 # spreading factor (R17)
PAYLOAD = 4            # bytes (R16)

TIERS = ((3.0, 300), (2.7, 900), (2.4, 1800))  # V >= level -> sensing interval in s (R27)
TIER_LOWEST = 3600
HB_FAST, HB_SLOW, HB_LEVEL = 3600, 6 * 3600, 2.7  # heartbeat 1/h, 1 per 6 h below 2.7 V (R29)

CHANGES_PER_DAY = 24   # presence changes (R28)
CHANGE_FIRST = 7 * 3600 + 20 * 60   # 07:20, then every 40 min, last one at 22:40
CHANGE_STEP = 40 * 60

DAY = 86400


def lora_airtime(sf, payload=PAYLOAD, bw=125e3, cr=1, crc=1, ih=0):
    """Semtech airtime formula (Annex A.5), preamble of 8 symbols."""
    de = 1 if sf >= 11 else 0
    t_sym = 2 ** sf / bw
    n_payload = 8 + max(math.ceil((8 * payload - 4 * sf + 28 + 16 * crc - 20 * ih)
                                  / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (8 + 4.25 + n_payload) * t_sym


T_TX = lora_airtime(SF)                       # 123.9 ms at SF9
E_TX = round(V_RADIO * I_TX * T_TX, 4)        # 40.9 mJ, rounded as in Table IV


def sense_interval(v):
    for level, seconds in TIERS:
        if v >= level:
            return seconds
    return TIER_LOWEST


def simulate(c=0.220, lux=LUX_DESIGN, lit=(7, 19), frames=32,
             leak_per_100mf=LEAK_PER_100MF, area=AREA_DESIGN, days=10,
             v_start=V_MAX, dark_days=(), e_frame=E_FRAME, trace=False):
    """Run the model and return a dict of totals.

    c               storage capacitance in F
    lux             illuminance while lit
    lit             (start hour, end hour) of the lit part of the day
    frames          frames per sensing burst
    leak_per_100mf  leakage current in A per 100 mF
    area            PV area in cm2
    days            simulated days; the run starts at 00:00 with the node on
    v_start         storage voltage at the start
    dark_days       zero-based indices of days without any light
    trace           if True, also record the voltage and on/off state per second
    """
    e_burst = frames * e_frame + E_CLASSIFIER
    t_burst = frames * T_FRAME
    p_harvest = P_HARVEST_DESIGN * (area / AREA_DESIGN) * (lux / LUX_DESIGN)
    i_leak = leak_per_100mf * c / 0.1
    e_max = 0.5 * c * V_MAX ** 2
    lit_start, lit_end = lit[0] * 3600, lit[1] * 3600
    change_times = {CHANGE_FIRST + i * CHANGE_STEP for i in range(CHANGES_PER_DAY)}

    def admit(e_task, i_task, t_task, v):
        """Admission rule, Eq. (1) plus the sag condition."""
        energy_ok = 0.5 * c * (v * v - V_CKPT ** 2) >= K_SAFETY * e_task
        sag_ok = v - (i_task * t_task / c + i_task * R_ESR) >= V_CKPT
        return energy_ok and sag_ok

    e = 0.5 * c * v_start ** 2      # stored energy in J
    on = True
    next_sense = 0                  # a burst is due at once
    # first heartbeat one interval after the start, from the starting voltage
    next_heartbeat = HB_FAST if v_start >= HB_LEVEL else HB_SLOW
    queued = 0                      # alerts sensed and waiting for the radio
    unsensed = []                   # times of presence changes not yet seen by a burst

    r = dict(awake_s=0, bursts=0, alerts=0, missed=0, heartbeats=0, blackouts=0,
             boots=0, e_harvest=0.0, e_floor=0.0, e_leak=0.0, e_burst=e_burst,
             latencies=[], events=[], v=[], on=[])

    for t in range(days * DAY):
        day, tod = divmod(t, DAY)
        is_lit = lit_start <= tod < lit_end and day not in dark_days

        # 1. Energy flows of this second.
        v = math.sqrt(2 * e / c) if e > 0 else 0.0
        p_in = p_harvest if is_lit else 0.0
        p_leak = i_leak * v
        e = min(max(e + p_in - P_FLOOR - p_leak, 0.0), e_max)
        r['e_harvest'] += p_in
        r['e_floor'] += P_FLOOR
        r['e_leak'] += p_leak
        v = math.sqrt(2 * e / c)

        # 2. Presence changes. One that happens while the node is off is missed.
        if tod in change_times:
            if on:
                unsensed.append(t)
            else:
                r['missed'] += 1

        # 3. Node behaviour.
        if on:
            if v <= V_CKPT:
                # Low-voltage warning: checkpoint and stop. Changes that no
                # burst has seen yet are lost.
                e = max(e - E_CKPT, 0.0)
                on = False
                r['blackouts'] += 1
                r['missed'] += len(unsensed)
                unsensed = []
                r['events'].append((t, 'off', v))
            else:
                # a) queued alert: transmit, then checkpoint (clears the flag,
                #    increments the sequence number).
                if queued > 0 and admit(E_TX, I_TX, T_TX, v):
                    e -= E_TX + E_CKPT
                    queued -= 1
                    r['alerts'] += 1
                    r['events'].append((t, 'alert', v))
                    v = math.sqrt(2 * e / c)
                # b) heartbeat; the next one is scheduled from the voltage
                #    before the packet.
                if t >= next_heartbeat and admit(E_TX, I_TX, T_TX, v):
                    e -= E_TX
                    r['heartbeats'] += 1
                    next_heartbeat = t + (HB_FAST if v >= HB_LEVEL else HB_SLOW)
                    r['events'].append((t, 'heartbeat', v))
                    v = math.sqrt(2 * e / c)
                # c) sensing burst; the next one is scheduled from the voltage
                #    tier before the burst.
                if t >= next_sense and admit(e_burst, e_burst / (v * t_burst), t_burst, v):
                    e -= e_burst
                    r['bursts'] += 1
                    next_sense = t + sense_interval(v)
                    r['events'].append((t, 'burst', v))
                    queued += len(unsensed)
                    r['latencies'] += [t - t0 for t0 in unsensed]
                    unsensed = []
        elif v >= V_ON:
            e -= E_BOOT
            on = True
            r['boots'] += 1
            r['events'].append((t, 'on', v))

        if on:
            r['awake_s'] += 1
        if trace:
            r['v'].append(math.sqrt(2 * e / c))
            r['on'].append(on)

    r['days'] = days
    r['awake_pct'] = 100.0 * r['awake_s'] / (days * DAY)
    r['e_end'] = e
    r['e_start'] = 0.5 * c * v_start ** 2
    return r


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
def per_day(r, key):
    return r[key] / r['days']


def row(label, r):
    return '%-24s %6.1f %7.0f %8.1f %8.1f %8.1f %6d' % (
        label, r['awake_pct'], math.floor(per_day(r, 'bursts') + 0.5),
        per_day(r, 'alerts'), per_day(r, 'missed'), per_day(r, 'heartbeats'),
        r['blackouts'])


HEADER = '%-24s %6s %7s %8s %8s %8s %6s' % (
    'Case (10 days, SF9)', 'Awake%', 'Bursts', 'Alerts', 'Missed', 'Heartb.', 'Black')
HEADER2 = '%-24s %6s %7s %8s %8s %8s %6s' % ('', '', '/day', 'sent/day', 'dark/day', '/day', 'outs')


def hhmm(t):
    tod = t % DAY
    return '%02d:%02d' % (tod // 3600, tod % 3600 // 60)


def print_constants():
    print('Constants')
    print('  harvest at the design point  %.1f uW' % (P_HARVEST_DESIGN * 1e6))
    print('  burst of 32 frames           %.1f mJ' % ((32 * E_FRAME + E_CLASSIFIER) * 1e3))
    print('  LoRa packet, SF%d, %d bytes     %.1f ms, %.1f mJ' % (SF, PAYLOAD, T_TX * 1e3, E_TX * 1e3))
    v_e = math.sqrt(V_CKPT ** 2 + 2 * (E_BOOT + 32 * E_FRAME + E_CLASSIFIER + E_CKPT) / 0.220)
    print('  V_E at 220 mF (Eq. 2)        %.2f V, so V_on = %.1f V' % (v_e, max(V_ON, v_e)))
    print()
    print('LoRa airtime, energy and capacitor size (Annex A.5)')
    print('  SF   Ts(ms)  airtime(ms)  energy(mJ)  silence@1%(s)  Cmin(mF)')
    for sf in range(7, 13):
        t_air = lora_airtime(sf)
        e_tx = V_RADIO * I_TX * t_air
        c_min = 2 * K_SAFETY * e_tx / (V_MAX ** 2 - V_CKPT ** 2)
        print('  %2d  %7.3f  %10.1f  %10.1f  %12.1f  %8.1f' % (
            sf, 2 ** sf / 125e3 * 1e3, t_air * 1e3, e_tx * 1e3, 99 * t_air, c_min * 1e3))
    print()


def print_tables():
    print('Storage size (Table VI and Annex A.7)')
    print(HEADER)
    print(HEADER2)
    for c_mf in (10, 22, 47, 100, 220, 470):
        label = '%d mF%s' % (c_mf, ' (design)' if c_mf == 220 else '')
        print(row(label, simulate(c=c_mf * 1e-3)))
    print()
    print('Sensitivity runs at 220 mF (Annex A.7)')
    print(HEADER)
    print(HEADER2)
    cases = (
        ('220 mF, 100 lux', dict(lux=100)),
        ('220 mF, 500 lux', dict(lux=500)),
        ('lit 08:00 to 18:00', dict(lit=(8, 18))),
        ('lit 07:00 to 21:00', dict(lit=(7, 21))),
        ('burst 16 frames', dict(frames=16)),
        ('burst 64 frames', dict(frames=64)),
        ('leakage 1 uA/100 mF', dict(leak_per_100mf=1e-6)),
        ('leakage 10 uA/100 mF', dict(leak_per_100mf=10e-6)),
        ('frame energy 0.1 mJ', dict(e_frame=0.1e-3)),
    )
    for label, kw in cases:
        print(row(label, simulate(**kw)))
    print()
    print('PV area sweep at 200 lux, 220 mF')
    for area in (10.8, 21.6, 43.2):
        print('  %5.1f cm2   awake %.1f %%' % (area, simulate(area=area)['awake_pct']))
    print()


def print_balance():
    r = simulate()
    d = r['days']
    print('Daily energy balance at the design point (Annex A.6), 10-day average')
    items = (
        ('Usable harvest', r['e_harvest'] / d),
        ('Sensing, %.1f bursts x %.1f mJ' % (r['bursts'] / d, r['e_burst'] * 1e3), -r['bursts'] * r['e_burst'] / d),
        ('Alerts, %.1f x %.1f mJ' % (r['alerts'] / d, E_TX * 1e3), -r['alerts'] * E_TX / d),
        ('Heartbeats, %.1f x %.1f mJ' % (r['heartbeats'] / d, E_TX * 1e3), -r['heartbeats'] * E_TX / d),
        ('Standing floor', -r['e_floor'] / d),
        ('Capacitor leakage', -r['e_leak'] / d),
        ('Blackout checkpoints and boots (%d and %d in %d days)' % (r['blackouts'], r['boots'], d),
         -(r['blackouts'] * E_CKPT + r['boots'] * E_BOOT) / d),
        ('Checkpoints after alerts', -r['alerts'] * E_CKPT / d),
    )
    for label, value in items:
        print('  %-52s %+9.5f J' % (label, value))
    print('  %-52s %+9.5f J' % ('Change of the store per day (starts full)', (r['e_end'] - r['e_start']) / d))
    lat = r['latencies']
    print('  presence change to sensing burst: mean %.1f min, maximum %.0f min (%d changes sensed)' % (
        sum(lat) / len(lat) / 60, max(lat) / 60, len(lat)))
    print()


def print_night():
    """Second simulated day at the design point: a regular night (Annex A.7)."""
    r = simulate(days=3, trace=True)
    v = r['v']
    t0 = DAY                      # start of day 2
    print('A simulated night at the design point (day 2 to day 3)')
    print('  %s  lights off, store at %.2f V' % (hhmm(t0 + 19 * 3600), v[t0 + 19 * 3600]))
    for level in (2.7, 2.4):
        for t in range(t0 + 19 * 3600, t0 + DAY):
            if v[t] < level:
                print('  %s  %.1f V reached (falling)' % (hhmm(t), level))
                break
    t_off = next(t for t, kind, _ in r['events'] if kind == 'off' and t > t0 + 12 * 3600)
    t_on = next(t for t, kind, _ in r['events'] if kind == 'on' and t > t_off)
    print('  %s  2.2 V reached, checkpoint, node stops' % hhmm(t_off))
    print('  %s  2.3 V reached (rising), node boots; off for %.1f h' % (hhmm(t_on), (t_on - t_off) / 3600))
    for level in (2.4, 2.7, 3.0):
        for t in range(t_on, 3 * DAY):
            if v[t] >= level:
                print('  %s  %.1f V reached (rising)' % (hhmm(t), level))
                break
    print()


def six_day():
    """Six days, days 3 to 5 dark, store starting at 2.6 V (Fig. 3)."""
    return simulate(days=6, v_start=2.6, dark_days=(2, 3, 4), trace=True)


def print_six_day():
    r = six_day()
    print('Six-day scenario, days 3 to 5 dark, store starts at 2.6 V (Fig. 3)')
    print('  awake %.1f %% of the six days, %d checkpoints (blackouts), %d boots' % (
        r['awake_pct'], r['blackouts'], r['boots']))
    print()


def make_figure(path='fig3_six_day_trace.png'):
    """Storage voltage over the six-day scenario (Fig. 3 of the report)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = six_day()
    days = r['days']
    step = 60
    times = range(0, days * DAY, step)
    hours = [t / 3600 for t in times]
    volts = r['v'][::step]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for d in range(days):
        if d not in (2, 3, 4):
            ax.axvspan(d * 24 + 7, d * 24 + 19, color='#ffe9b0', alpha=0.7, lw=0)
    # gray bar along the bottom while the node is off
    off_start = None
    for h, t in zip(hours, times):
        is_on = r['on'][t]
        if not is_on and off_start is None:
            off_start = h
        if is_on and off_start is not None:
            ax.axvspan(off_start, h, ymin=0.0, ymax=0.06, color='#444444', lw=0)
            off_start = None
    if off_start is not None:
        ax.axvspan(off_start, days * 24, ymin=0.0, ymax=0.06, color='#444444', lw=0)
    ax.plot(hours, volts, color='#1f3a68', lw=1.2)
    ax.axhline(V_ON, color='#1e7d32', lw=1.2, ls=':')
    ax.axhline(V_CKPT, color='#b22222', lw=1.2, ls='--')
    ax.text(86, V_ON + 0.12, r'$V_{on}$ %.1f V' % V_ON, color='#1e7d32', fontsize=9, style='italic')
    ax.text(86, V_CKPT - 0.28, r'$V_{ckpt}$ %.1f V' % V_CKPT, color='#b22222', fontsize=9, style='italic')
    ax.text(66, 3.05, 'three dark days', fontsize=10)
    ax.set_xlim(0, days * 24)
    ax.set_ylim(0, 3.5)
    ax.set_xticks(range(0, days * 24 + 1, 24))
    ax.set_xlabel('time (h)')
    ax.set_ylabel('storage voltage (V)')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print('wrote %s' % path)


def main(argv):
    print_constants()
    print_tables()
    print_balance()
    print_night()
    print_six_day()
    if len(argv) > 1 and argv[1] == 'fig':
        make_figure()


if __name__ == '__main__':
    main(sys.argv)
