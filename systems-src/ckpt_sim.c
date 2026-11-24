/*
 * ckpt_sim.c - host-side fault-injection model of the Sentinel Mesh checkpoint.
 *
 * The FRAM is a byte array with a write counter. A power cut can be injected
 * before any chosen write unit of a checkpoint. After the cut the boot routine
 * runs and the restored state is compared field by field with the newest and
 * the previous checkpointed state.
 *
 * Two protocols are modelled:
 *   - commit-last, two slots, CRC-32 (report Sec. V-B and Annex A.3);
 *   - the first draft: one slot, magic word written first, no CRC.
 *
 * Three cut models:
 *   - between 16-bit words (on-chip FRAM);
 *   - inside a word, with only the low byte landing;
 *   - between single bytes (the external SPI FRAM actually used).
 *
 * Two campaigns:
 *   - exhaustive: every cut position (and one run without a cut) in each of
 *     38 consecutive generations, plus the first checkpoint on blank memory;
 *   - random: two million power cycles per cut model, fixed seed 12345.
 *
 * Build and run:
 *   gcc -O2 -Wall -o ckpt_sim ckpt_sim.c && ./ckpt_sim
 *
 * The model assumes that writes land in order and that a byte is either
 * written or not. It does not model the radio, the radar or a supply that
 * dips and recovers around the checkpoint threshold.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* Checkpointed state and layout (Annex A.3)                                 */
/* ------------------------------------------------------------------------- */

#define WINDOW_LEN   32

typedef struct {
    uint32_t current_step;        /* where to resume                         */
    uint32_t lora_seq_num;        /* packet numbering across blackouts       */
    uint32_t alert_pending;       /* an alert raised but not yet sent        */
    float    window[WINDOW_LEN];  /* feature window, 32 single-precision     */
} state_t;

#define SLOT_SIZE    148u   /* commit(4) + payload(140) + crc(4)             */
#define COMMIT_LEN   4u     /* generation (2) + bitwise complement (2)       */
#define PAYLOAD_OFF  4u
#define PAYLOAD_LEN  140u   /* bytes 4 to 143                                */
#define CRC_OFF      144u
#define NEW_WORDS    76u    /* 2 (zero commit) + 70 + 2 (CRC) + 2 (commit)   */
#define NEW_BYTES    152u

#define OLD_MAGIC    0xDEADBEEFu
#define OLD_SIZE     144u   /* magic(4) + payload(140), one slot, no CRC     */
#define OLD_WORDS    72u
#define OLD_BYTES    144u

#define FRAM_SIZE    (2u * SLOT_SIZE)

/* ------------------------------------------------------------------------- */
/* FRAM model with power-cut injection                                       */
/* ------------------------------------------------------------------------- */

enum cut_model { CUT_BETWEEN_WORDS = 0, CUT_INSIDE_WORD = 1, CUT_BETWEEN_BYTES = 2 };
#define N_MODELS 3

static const char *const model_name[N_MODELS] = {
    "between words", "inside word", "between bytes"
};

static uint8_t fram[FRAM_SIZE];
static int     cut_model;      /* one of enum cut_model                      */
static long    cut_at;         /* write unit before which power fails; -1: none */
static long    units_done;     /* write units completed since power-on       */
static int     powered;        /* 0 after the injected cut                   */

static void power_on(int model, long cut)
{
    cut_model = model;
    cut_at = cut;
    units_done = 0;
    powered = 1;
}

static unsigned units_per_checkpoint(int model, int old_protocol)
{
    if (model == CUT_BETWEEN_BYTES)
        return old_protocol ? OLD_BYTES : NEW_BYTES;
    return old_protocol ? OLD_WORDS : NEW_WORDS;
}

/* Write len bytes (len is even) in ascending address order. */
static void fram_write(unsigned addr, const uint8_t *src, unsigned len)
{
    unsigned i;

    if (cut_model == CUT_BETWEEN_BYTES) {
        for (i = 0; i < len; i++) {
            if (!powered)
                return;
            if (units_done == cut_at) {
                powered = 0;
                return;
            }
            fram[addr + i] = src[i];
            units_done++;
        }
        return;
    }
    for (i = 0; i < len; i += 2) {
        if (!powered)
            return;
        if (units_done == cut_at) {
            if (cut_model == CUT_INSIDE_WORD)
                fram[addr + i] = src[i];      /* only the low byte lands */
            powered = 0;
            return;
        }
        fram[addr + i] = src[i];
        fram[addr + i + 1] = src[i + 1];
        units_done++;
    }
}

/* ------------------------------------------------------------------------- */
/* Serialization, little-endian, independent of the host                     */
/* ------------------------------------------------------------------------- */

static void put_u16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)(v >> 8);
}

static uint16_t get_u16(const uint8_t *p)
{
    return (uint16_t)(p[0] | ((unsigned)p[1] << 8));
}

static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)(v >> 24);
}

static uint32_t get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* 140-byte payload: current_step, lora_seq_num, alert_pending, window[32]. */
static void encode_payload(uint8_t *p, const state_t *s)
{
    unsigned i;
    uint32_t bits;

    put_u32(p + 0, s->current_step);
    put_u32(p + 4, s->lora_seq_num);
    put_u32(p + 8, s->alert_pending);
    for (i = 0; i < WINDOW_LEN; i++) {
        memcpy(&bits, &s->window[i], sizeof bits);
        put_u32(p + 12 + 4 * i, bits);
    }
}

static void decode_payload(state_t *s, const uint8_t *p)
{
    unsigned i;
    uint32_t bits;

    s->current_step = get_u32(p + 0);
    s->lora_seq_num = get_u32(p + 4);
    s->alert_pending = get_u32(p + 8);
    for (i = 0; i < WINDOW_LEN; i++) {
        bits = get_u32(p + 12 + 4 * i);
        memcpy(&s->window[i], &bits, sizeof bits);
    }
}

/* Field-by-field comparison on the encoded form (floats compared by bits). */
static int state_equal(const state_t *a, const state_t *b)
{
    uint8_t pa[PAYLOAD_LEN], pb[PAYLOAD_LEN];

    encode_payload(pa, a);
    encode_payload(pb, b);
    return memcmp(pa, pb, PAYLOAD_LEN) == 0;
}

/* Cold start: step idle, sequence 0, window empty. */
static void cold_state(state_t *s)
{
    memset(s, 0, sizeof *s);
}

/* CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320). */
static uint32_t crc32_calc(const uint8_t *p, unsigned len)
{
    uint32_t crc = 0xFFFFFFFFu;
    unsigned i, b;

    for (i = 0; i < len; i++) {
        crc ^= p[i];
        for (b = 0; b < 8; b++)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

/* ------------------------------------------------------------------------- */
/* Protocol 1: commit-last, two slots (Sec. V-B, Annex A.3)                  */
/* ------------------------------------------------------------------------- */

/* Generation g goes to slot g mod 2. Generation 0 is never valid. */
static void ckpt_write(const state_t *s, uint16_t gen)
{
    static const uint8_t zero[COMMIT_LEN] = { 0, 0, 0, 0 };
    uint8_t img[SLOT_SIZE];
    unsigned base = (gen & 1u) * SLOT_SIZE;

    put_u16(img + 0, gen);
    put_u16(img + 2, (uint16_t)~gen);
    encode_payload(img + PAYLOAD_OFF, s);
    put_u32(img + CRC_OFF, crc32_calc(img + PAYLOAD_OFF, PAYLOAD_LEN));

    fram_write(base, zero, COMMIT_LEN);                           /* (1) invalidate   */
    fram_write(base + PAYLOAD_OFF, img + PAYLOAD_OFF, PAYLOAD_LEN); /* (2) payload   */
    fram_write(base + CRC_OFF, img + CRC_OFF, 4);                 /* (3) CRC          */
    fram_write(base, img, COMMIT_LEN);                            /* (4) commit, last */
}

/* Valid: complement matches, generation non-zero, CRC matches. */
static int slot_valid(unsigned base, uint16_t *gen)
{
    uint16_t g = get_u16(fram + base);
    uint16_t inv = get_u16(fram + base + 2);

    if (g == 0 || (uint16_t)~g != inv)
        return 0;
    if (crc32_calc(fram + base + PAYLOAD_OFF, PAYLOAD_LEN) != get_u32(fram + base + CRC_OFF))
        return 0;
    *gen = g;
    return 1;
}

/* Boot decision table. Returns 1 on a warm boot, 0 on a cold start. */
static int ckpt_boot(state_t *out, uint16_t *gen_out)
{
    uint16_t g0 = 0, g1 = 0;
    int v0 = slot_valid(0, &g0);
    int v1 = slot_valid(SLOT_SIZE, &g1);
    int use;

    if (!v0 && !v1) {
        cold_state(out);
        *gen_out = 0;
        return 0;
    }
    if (v0 && v1)
        use = ((uint16_t)(g0 - g1) < 0x8000u) ? 0 : 1;   /* higher, modulo 2^16 */
    else
        use = v0 ? 0 : 1;
    decode_payload(out, fram + (unsigned)use * SLOT_SIZE + PAYLOAD_OFF);
    *gen_out = use ? g1 : g0;
    return 1;
}

/*
 * Next generation. Zero is never valid, so the counter wraps from 65535 to 2
 * and not to 1: 65535 lives in slot 1 and the next checkpoint must go to
 * slot 0, otherwise it would overwrite the only copy of the newest state.
 */
static uint16_t next_generation(uint16_t gen)
{
    return gen == 0xFFFFu ? 2 : (uint16_t)(gen + 1u);
}

/* ------------------------------------------------------------------------- */
/* Protocol 2: first draft, magic word first, one slot, no CRC               */
/* ------------------------------------------------------------------------- */

static void old_write(const state_t *s)
{
    uint8_t img[OLD_SIZE];

    put_u32(img, OLD_MAGIC);
    encode_payload(img + 4, s);
    fram_write(0, img, OLD_SIZE);
}

static int old_boot(state_t *out)
{
    if (get_u32(fram) != OLD_MAGIC) {
        cold_state(out);
        return 0;
    }
    decode_payload(out, fram + 4);
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Exhaustive campaign                                                       */
/* ------------------------------------------------------------------------- */

#define FIRST_GEN    3u     /* memory then holds generations 1 and 2         */
#define N_GEN        38u

/* Test pattern: every field differs between consecutive generations. */
static void make_state(state_t *s, unsigned gen)
{
    unsigned i;

    s->current_step = gen % 4u;
    s->lora_seq_num = gen;
    s->alert_pending = gen & 1u;
    for (i = 0; i < WINDOW_LEN; i++)
        s->window[i] = (float)gen + 0.5f * (float)i;
}

typedef struct {
    long cuts;      /* cut points, including the run without a cut           */
    long newest;    /* restored the state being written                      */
    long previous;  /* restored the previous checkpoint (or the cold state)  */
    long mixed;     /* anything else: corrupt                                */
} tally_t;

static void classify(tally_t *t, const state_t *got, const state_t *newest,
                     const state_t *previous)
{
    t->cuts++;
    if (state_equal(got, newest))
        t->newest++;
    else if (state_equal(got, previous))
        t->previous++;
    else
        t->mixed++;
}

static void exhaustive_new(int model, tally_t *t)
{
    uint8_t base[FRAM_SIZE];
    state_t s_new, s_prev, s_prev2, got;
    uint16_t gen_got;
    unsigned g;
    long cut, n = (long)units_per_checkpoint(model, 0);

    memset(t, 0, sizeof *t);
    for (g = FIRST_GEN; g < FIRST_GEN + N_GEN; g++) {
        make_state(&s_prev2, g - 2);
        make_state(&s_prev, g - 1);
        make_state(&s_new, g);

        memset(fram, 0, sizeof fram);
        power_on(model, -1);
        ckpt_write(&s_prev2, (uint16_t)(g - 2));
        ckpt_write(&s_prev, (uint16_t)(g - 1));
        memcpy(base, fram, sizeof fram);

        for (cut = 0; cut <= n; cut++) {            /* cut == n: no cut */
            memcpy(fram, base, sizeof fram);
            power_on(model, cut < n ? cut : -1);
            ckpt_write(&s_new, (uint16_t)g);
            power_on(model, -1);
            ckpt_boot(&got, &gen_got);
            classify(t, &got, &s_new, &s_prev);
        }
    }
}

/* The very first checkpoint, on blank memory: a cut must give a cold start. */
static void exhaustive_first(int model, tally_t *t)
{
    state_t s_new, s_cold, got;
    uint16_t gen_got;
    long cut, n = (long)units_per_checkpoint(model, 0);

    memset(t, 0, sizeof *t);
    make_state(&s_new, 1);
    cold_state(&s_cold);
    for (cut = 0; cut <= n; cut++) {
        memset(fram, 0, sizeof fram);
        power_on(model, cut < n ? cut : -1);
        ckpt_write(&s_new, 1);
        power_on(model, -1);
        ckpt_boot(&got, &gen_got);
        classify(t, &got, &s_new, &s_cold);
    }
}

/*
 * Generation wrap: eight consecutive generations that cross 65535. This is
 * reported on its own line and is not part of the 38-generation campaign.
 */
#define N_WRAP_GEN   8u

static void exhaustive_wrap(int model, tally_t *t)
{
    uint8_t base[FRAM_SIZE];
    state_t s_new, s_prev, s_prev2, got;
    uint16_t g2 = 0xFFFBu, g1, g, gen_got;
    unsigned k;
    long cut, n = (long)units_per_checkpoint(model, 0);

    memset(t, 0, sizeof *t);
    for (k = 0; k < N_WRAP_GEN; k++) {
        g1 = next_generation(g2);
        g = next_generation(g1);
        make_state(&s_prev2, 100u + k);
        make_state(&s_prev, 101u + k);
        make_state(&s_new, 102u + k);

        memset(fram, 0, sizeof fram);
        power_on(model, -1);
        ckpt_write(&s_prev2, g2);
        ckpt_write(&s_prev, g1);
        memcpy(base, fram, sizeof fram);

        for (cut = 0; cut <= n; cut++) {
            memcpy(fram, base, sizeof fram);
            power_on(model, cut < n ? cut : -1);
            ckpt_write(&s_new, g);
            power_on(model, -1);
            ckpt_boot(&got, &gen_got);
            classify(t, &got, &s_new, &s_prev);
        }
        g2 = g1;
    }
}

static void exhaustive_old(int model, tally_t *t)
{
    uint8_t base[FRAM_SIZE];
    state_t s_new, s_prev, got;
    unsigned g;
    long cut, n = (long)units_per_checkpoint(model, 1);

    memset(t, 0, sizeof *t);
    for (g = FIRST_GEN; g < FIRST_GEN + N_GEN; g++) {
        make_state(&s_prev, g - 1);
        make_state(&s_new, g);

        memset(fram, 0, sizeof fram);
        power_on(model, -1);
        old_write(&s_prev);
        memcpy(base, fram, sizeof fram);

        for (cut = 0; cut <= n; cut++) {
            memcpy(fram, base, sizeof fram);
            power_on(model, cut < n ? cut : -1);
            old_write(&s_new);
            power_on(model, -1);
            old_boot(&got);
            classify(t, &got, &s_new, &s_prev);
        }
    }
}

static int run_exhaustive(void)
{
    tally_t t, first, wrap, total_new, total_old;
    int m;

    memset(&total_new, 0, sizeof total_new);
    memset(&total_old, 0, sizeof total_old);
    memset(&first, 0, sizeof first);
    memset(&wrap, 0, sizeof wrap);

    printf("Exhaustive fault injection (%u generations, one cut before each write unit\n", N_GEN);
    printf("plus one run without a cut)\n\n");
    printf("%-24s %-26s %10s %8s %8s %9s %7s\n",
           "Protocol", "Cut model", "Cut points", "OK", "newest", "previous", "Mixed");

    for (m = 0; m < N_MODELS; m++) {
        exhaustive_new(m, &t);
        printf("%-24s %-26s %10ld %8ld %8ld %9ld %7ld\n", "Commit-last, two slots",
               model_name[m], t.cuts, t.newest + t.previous, t.newest, t.previous, t.mixed);
        total_new.cuts += t.cuts;
        total_new.newest += t.newest;
        total_new.previous += t.previous;
        total_new.mixed += t.mixed;
    }
    for (m = 0; m < N_MODELS; m++) {
        exhaustive_first(m, &t);
        first.cuts += t.cuts;
        first.newest += t.newest;
        first.previous += t.previous;
        first.mixed += t.mixed;
    }
    printf("%-24s %-26s %10ld %8ld %8ld %9ld %7ld\n", "Commit-last, two slots",
           "first write, blank memory", first.cuts, first.newest + first.previous,
           first.newest, first.previous, first.mixed);
    total_new.cuts += first.cuts;
    total_new.newest += first.newest;
    total_new.previous += first.previous;
    total_new.mixed += first.mixed;
    printf("%-24s %-26s %10ld %8ld %8ld %9ld %7ld\n", "Commit-last, total", "all of the above",
           total_new.cuts, total_new.newest + total_new.previous, total_new.newest,
           total_new.previous, total_new.mixed);

    for (m = 0; m < N_MODELS; m++) {
        exhaustive_old(m, &t);
        printf("%-24s %-26s %10ld %8ld %8ld %9ld %7ld\n", "Magic-first, one slot",
               model_name[m], t.cuts, t.newest + t.previous, t.newest, t.previous, t.mixed);
        total_old.cuts += t.cuts;
        total_old.newest += t.newest;
        total_old.previous += t.previous;
        total_old.mixed += t.mixed;
    }
    printf("%-24s %-26s %10ld %8ld %8ld %9ld %7ld\n", "Magic-first, total", "all of the above",
           total_old.cuts, total_old.newest + total_old.previous, total_old.newest,
           total_old.previous, total_old.mixed);
    printf("\nMagic-first: %ld of %ld cut points restore a mixture (%.1f%%)\n\n",
           total_old.mixed, total_old.cuts, 100.0 * (double)total_old.mixed / (double)total_old.cuts);

    for (m = 0; m < N_MODELS; m++) {
        exhaustive_wrap(m, &t);
        wrap.cuts += t.cuts;
        wrap.newest += t.newest;
        wrap.previous += t.previous;
        wrap.mixed += t.mixed;
    }
    printf("Extra check, generation counter wrapping past 65535 (%u generations, all three\n"
           "cut models): %ld cut points, %ld OK, %ld not OK\n\n", N_WRAP_GEN, wrap.cuts,
           wrap.newest + wrap.previous, wrap.mixed);

    return (total_new.mixed == 0 && wrap.mixed == 0) ? 0 : 1;
}

/* ------------------------------------------------------------------------- */
/* Random campaign                                                           */
/* ------------------------------------------------------------------------- */

#define SEED          12345ULL
#define N_CYCLES      2000000L
#define P_CHECKPOINT  90u     /* percent of cycles that attempt a checkpoint  */
#define P_CUT_INSIDE  5u      /* percent of those in which power fails inside */

static uint64_t rng_state;

/* PCG32 (XSH-RR), so the sequence is the same on every platform. */
static uint32_t rng_next(void)
{
    uint64_t old = rng_state;
    uint32_t xorshifted, rot;

    rng_state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    xorshifted = (uint32_t)(((old >> 18) ^ old) >> 27);
    rot = (uint32_t)(old >> 59);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

static void rng_seed(uint64_t seed)
{
    rng_state = 0;
    rng_next();
    rng_state += seed;
    rng_next();
}

static uint32_t rng_below(uint32_t n)
{
    return rng_next() % n;
}

/* One application step: change the state the way the firmware would. */
static void app_step(state_t *s, uint32_t k)
{
    s->current_step = (s->current_step + 1u) % 4u;
    if (k % 5u == 0)
        s->lora_seq_num++;
    if (k % 7u == 0)
        s->alert_pending ^= 1u;
    s->window[k % WINDOW_LEN] = (float)(k & 0xFFFFu) * 0.25f;
}

static int run_random(int model)
{
    state_t live, committed, got;
    uint16_t gen = 1, gen_got;
    uint32_t step_count = 0;
    long cycle, corrupt = 0, cold = 0;
    long attempts = 0, cut_inside = 0, landed = 0, no_warning = 0;
    long loss_events = 0, loss_sum = 0, loss_max = 0;
    unsigned n_units = units_per_checkpoint(model, 0);

    rng_seed(SEED);
    memset(fram, 0, sizeof fram);
    cold_state(&live);

    /* The node is in service: generation 1 holds its initial state. */
    power_on(model, -1);
    ckpt_write(&live, gen);
    committed = live;

    for (cycle = 0; cycle < N_CYCLES; cycle++) {
        long steps = 1 + (long)rng_below(5);
        long i;
        uint16_t g = 0;
        int attempted = 0, completed = 0, warm;

        for (i = 0; i < steps; i++)
            app_step(&live, ++step_count);

        if (rng_below(100) < P_CHECKPOINT) {
            long cut = -1;

            attempted = 1;
            attempts++;
            g = next_generation(gen);
            if (rng_below(100) < P_CUT_INSIDE) {
                cut = (long)rng_below(n_units);
                cut_inside++;
            }
            power_on(model, cut);
            ckpt_write(&live, g);
            completed = powered;
        } else {
            no_warning++;          /* power lost without any checkpoint */
        }

        /* Power returns: boot and check what was restored. */
        power_on(model, -1);
        warm = ckpt_boot(&got, &gen_got);
        if (!warm)
            cold++;

        if (attempted && !completed && gen_got == g && state_equal(&got, &live)) {
            /* The cut fell in the last write unit, whose missing bytes already
             * held the right value (zero), so the commit is complete. */
            completed = 1;
            landed++;
        }
        if (completed) {
            if (gen_got != g || !state_equal(&got, &live))
                corrupt++;
            gen = g;
            committed = live;
        } else {
            /* Must be exactly the last completed checkpoint. */
            if (gen_got != gen || !state_equal(&got, &committed))
                corrupt++;
            loss_events++;         /* progress lost: the steps of this cycle */
            loss_sum += steps;
            if (steps > loss_max)
                loss_max = steps;
        }
        live = got;
    }

    printf("%-14s %9ld cycles: checkpoint attempted %ld, cut inside %ld (%.2f%% of cycles), "
           "no warning %ld\n", model_name[model], N_CYCLES, attempts, cut_inside,
           100.0 * (double)cut_inside / (double)N_CYCLES, no_warning);
    printf("%-14s corrupt restores %ld, cold starts %ld, cut in the last unit but commit "
           "complete %ld\n", "", corrupt, cold, landed);
    printf("%-14s progress lost in %ld cycles: mean %.2f steps, maximum %ld\n\n", "",
           loss_events, loss_events ? (double)loss_sum / (double)loss_events : 0.0, loss_max);

    return (corrupt == 0 && cold == 0) ? 0 : 1;
}

int main(void)
{
    int m, fail = 0;

    fail |= run_exhaustive();

    printf("Random campaign (seed %llu, %ld power cycles per cut model)\n\n",
           (unsigned long long)SEED, N_CYCLES);
    for (m = 0; m < N_MODELS; m++)
        fail |= run_random(m);

    printf("%s\n", fail ? "RESULT: FAILED" : "RESULT: commit-last protocol passed every case");
    return fail;
}
