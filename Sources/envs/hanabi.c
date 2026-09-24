#include "../env.h"

typedef struct { uint32_t players, handSize, partner, thinkSteps; } hanabiKnobs;

static const hanabiKnobs HANABI_KNOBS[3] = {
    { 2u, 5u, 0u, 3u },
    { 2u, 5u, 1u, 4u },
    { 4u, 4u, 1u, 8u }
};

static const uint32_t HANABI_COPIES[5] = { 3u, 2u, 2u, 2u, 1u };

enum { HANABI_COLOURS = 5, HANABI_CARDS = 50, HANABI_INFORMATION = 8, HANABI_LIVES = 3, HANABI_SLOTS = 16, HANABI_OBS = 18 };
enum { HANABI_NEXT, HANABI_PREVIOUS, HANABI_PLAY, HANABI_DISCARD, HANABI_HINT_COLOUR, HANABI_HINT_RANK };
enum { HANABI_LITERAL, HANABI_FOCUS };
enum { HANABI_FIREWORKS = HANABI_SLOTS, HANABI_INFO, HANABI_LIFE, HANABI_DECK, HANABI_DISCARDS, HANABI_DISCARDS_HIGH, HANABI_GONE, HANABI_LAST_COPY,
       HANABI_TOUCHED, HANABI_ACTOR, HANABI_THINK, HANABI_CURSOR, HANABI_FINAL, HANABI_TURN, HANABI_GAME, HANABI_LAST, HANABI_WORDS };
enum { HANABI_PLAUSIBLE = 0x1FFFFFF, HANABI_NONE = 0x3E000000, HANABI_CLUED = 0x40000000 };
static const uint32_t HANABI_MARKED = 0x80000000u;
enum { HANABI_MOVE_PLAY, HANABI_MOVE_DISCARD, HANABI_MOVE_COLOUR, HANABI_MOVE_RANK, HANABI_IDLE = 0xFFu, HANABI_EMPTY_SLOT = 0xFFu };
enum { HANABI_COMPLETE = 0x5B6Du, HANABI_ONES = 0x108421u, HANABI_FIVES = 0x1084210u, HANABI_BELOW_FIVE = 0xF7BDEFu };

static const uint8_t HANABI_ORDERED[HANABI_CARDS] = { 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 10, 11, 11, 12, 12, 13, 13, 14,
                                                     15, 15, 15, 16, 16, 17, 17, 18, 18, 19, 20, 20, 20, 21, 21, 22, 22, 23, 23, 24 };

typedef struct { uint32_t playable, useless, exhausted, critical, played; } hanabiTable;

typedef struct { uint32_t clued, chop, colours, ranks, playable, bad, critical; } hanabiHand;

typedef struct {
    hanabiHand hand;
    uint32_t move, play, marked, useless, chop, urgent, claims, save, clue, stall;
    int32_t saveValue, clueValue, stallValue;
} hanabiPlan;

typedef struct { uint32_t key, target, action; } hanabiAim;

typedef struct {
    uint8_t deck[HANABI_CARDS];
    uint8_t spare[2];
    hanabiTable table;
    hanabiPlan plan;
    hanabiAim aim;
    float view[HANABI_OBS];
} hanabiLane;

static hanabiLane *hanabi_lane(uint32_t *extra) { return (hanabiLane *)extra; }
static const hanabiLane *hanabi_peek(const uint32_t *extra) { return (const hanabiLane *)extra; }

uint32_t hanabi_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)((sizeof(hanabiLane) + sizeof(uint32_t) - 1u) / sizeof(uint32_t));
}

static uint32_t hanabi_id(uint32_t slot) { return slot >> 25 & 31u; }

static uint32_t hanabi_card(uint32_t id) { return id << 25 | HANABI_PLAUSIBLE; }

static uint32_t hanabi_group(uint32_t type, uint32_t value) { return type == HANABI_MOVE_COLOUR ? 31u << 5u * value : (uint32_t)HANABI_ONES << value; }

static uint32_t hanabi_ranks(uint32_t slot) {
    uint32_t p = slot & HANABI_PLAUSIBLE;
    return (p | p >> 5 | p >> 10 | p >> 15 | p >> 20) & 31u;
}

static uint32_t hanabi_colours(uint32_t slot) {
    uint32_t p = slot & HANABI_PLAUSIBLE;
    return (p | p >> 1 | p >> 2 | p >> 3 | p >> 4) & HANABI_ONES;
}

static uint32_t hanabi_height(uint32_t fireworks, uint32_t colour) { return fireworks >> 3u * colour & 7u; }

static uint32_t hanabi_known(uint32_t slot, const hanabiTable *t) {
    uint32_t possible = slot & HANABI_PLAUSIBLE & ~t->exhausted;
    return possible && !(possible & ~t->playable);
}

static uint32_t hanabi_hint(uint32_t type, uint32_t target, uint32_t value) { return type | target << 2 | value << 5; }

static uint32_t hanabi_score(uint32_t fireworks) {
    uint32_t score = 0u;
    for (uint32_t c = 0u; c < HANABI_COLOURS; c++) score += hanabi_height(fireworks, c);
    return score;
}

static uint32_t hanabi_discarded(const uint32_t *words, uint32_t id) { return words[id < 16u ? HANABI_DISCARDS : HANABI_DISCARDS_HIGH] >> 2u * (id & 15u) & 3u; }

static void hanabi_colour(const uint32_t *words, hanabiTable *t, uint32_t colour) {
    uint32_t shift = 5u * colour, height = hanabi_height(words[HANABI_FIREWORKS], colour);
    uint32_t gone = words[HANABI_GONE] >> shift & 31u, last = words[HANABI_LAST_COPY] >> shift & 31u, lost = gone & 31u << height;
    uint32_t reach = lost ? (uint32_t)__builtin_ctz(lost) : 5u, below = (1u << height) - 1u, useless = below | (31u << reach & 31u), keep = ~(31u << shift);
    t->playable = (t->playable & keep) | (uint32_t)(height < reach) << (shift + height);
    t->useless = (t->useless & keep) | useless << shift;
    t->exhausted = (t->exhausted & keep) | (gone | (below & last)) << shift;
    t->critical = (t->critical & keep) | (last & ~useless) << shift;
    t->played = (t->played & keep) | below << shift;
}

static void hanabi_bury(uint32_t *words, uint32_t id) {
    uint32_t count = hanabi_discarded(words, id) + 1u, copies = HANABI_COPIES[id % 5u], bit = 1u << id;
    words[id < 16u ? HANABI_DISCARDS : HANABI_DISCARDS_HIGH] += 1u << 2u * (id & 15u);
    if (count + 1u == copies) words[HANABI_LAST_COPY] |= bit;
    if (count == copies) {
        words[HANABI_LAST_COPY] &= ~bit;
        words[HANABI_GONE] |= bit;
    }
}

static uint64_t hanabi_pairs(uint32_t mask) {
    uint64_t x = mask;
    x = (x | x << 16) & 0x0000FFFF0000FFFFull;
    x = (x | x << 8) & 0x00FF00FF00FF00FFull;
    x = (x | x << 4) & 0x0F0F0F0F0F0F0F0Full;
    x = (x | x << 2) & 0x3333333333333333ull;
    return (x | x << 1) & 0x5555555555555555ull;
}

static uint32_t hanabi_singles(uint64_t x) {
    x &= 0x5555555555555555ull;
    x = (x | x >> 1) & 0x3333333333333333ull;
    x = (x | x >> 2) & 0x0F0F0F0F0F0F0F0Full;
    x = (x | x >> 4) & 0x00FF00FF00FF00FFull;
    x = (x | x >> 8) & 0x0000FFFF0000FFFFull;
    return (uint32_t)((x | x >> 16) & 0x1FFFFFFu);
}

static uint32_t hanabi_seen(const hanabiKnobs *k, const uint32_t *words, const hanabiTable *t, uint32_t seat, uint32_t *claims) {
    uint32_t clued = 0u;
    uint64_t count = ((uint64_t)words[HANABI_DISCARDS] | (uint64_t)words[HANABI_DISCARDS_HIGH] << 32) + hanabi_pairs(t->played);
    for (uint32_t other = 0u; other < k->players; other++) {
        if (other == seat) continue;
        const uint32_t *hand = words + other * k->handSize;
        for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++) {
            uint32_t id = hanabi_id(hand[s]);
            count += 1ull << 2u * id;
            clued |= (uint32_t)((hand[s] & HANABI_CLUED) != 0u) << id;
        }
    }
    *claims = clued;
    uint64_t x = count ^ 0x1AB6ADAB6ADABull;
    return hanabi_singles(~(x | x >> 1));
}

static uint32_t hanabi_aim(uint32_t chop, uint32_t clued, uint32_t touched) {
    if (chop != HANABI_EMPTY_SLOT && touched >> chop & 1u) return chop;
    uint32_t fresh = touched & ~clued;
    return 31u - (uint32_t)__builtin_clz(fresh ? fresh : touched);
}

static void hanabi_summarize(const hanabiKnobs *k, const uint32_t *hand, const hanabiTable *t, uint32_t claims, hanabiHand *h) {
    *h = (hanabiHand){ 0u, HANABI_EMPTY_SLOT, 0u, 0u, 0u, 0u, 0u };
    for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++) {
        uint32_t slot = hand[s], id = hanabi_id(slot), bit = 1u << s;
        if (slot & HANABI_CLUED) h->clued |= bit;
        else if (h->chop == HANABI_EMPTY_SLOT) h->chop = s;
        h->colours |= bit << 5u * (id / 5u);
        h->ranks |= bit << 5u * (id % 5u);
        h->playable |= (t->playable >> id & 1u) << s;
        h->bad |= ((t->useless | claims) >> id & 1u) << s;
        h->critical |= (t->critical >> id & 1u) << s;
    }
}

static uint32_t hanabi_draw(const hanabiKnobs *k, uint32_t *words, const hanabiLane *l) {
    uint32_t left = words[HANABI_DECK];
    if (!left) return HANABI_NONE;
    words[HANABI_DECK] = left - 1u;
    if (left == 1u) words[HANABI_FINAL] = k->players + 1u;
    return hanabi_card(l->deck[HANABI_CARDS - left]);
}

static uint32_t hanabi_reveal(const hanabiKnobs *k, uint32_t *words, const hanabiTable *t, uint32_t target, uint32_t type, uint32_t value) {
    uint32_t *hand = words + target * k->handSize, group = hanabi_group(type, value), touched = 0u, clued = 0u, chop = HANABI_EMPTY_SLOT;
    for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++) {
        uint32_t slot = hand[s];
        if (slot & HANABI_CLUED) clued |= 1u << s;
        else if (chop == HANABI_EMPTY_SLOT) chop = s;
        if (group >> hanabi_id(slot) & 1u) {
            touched |= 1u << s;
            hand[s] = (slot & ((uint32_t)~HANABI_PLAUSIBLE | group)) | HANABI_CLUED;
        } else hand[s] = slot & ~group;
    }
    words[HANABI_TOUCHED] = touched << target * k->handSize;
    if (k->partner == HANABI_FOCUS) {
        uint32_t focus = hanabi_aim(chop, clued, touched);
        if (hand[focus] & t->playable) hand[focus] |= HANABI_MARKED;
    }
    return touched;
}

static float hanabi_apply(const hanabiKnobs *k, uint32_t *words, hanabiLane *l, uint32_t seat, uint32_t move, uint32_t *over) {
    uint32_t type = move & 3u, at = move >> 2 & 7u, value = move >> 5 & 7u, record = seat;
    float reward = 0.0f;
    if (type <= HANABI_MOVE_DISCARD) {
        uint32_t *hand = words + seat * k->handSize, id = hanabi_id(hand[at]), colour = id / 5u, fireworks = words[HANABI_FIREWORKS];
        uint32_t base = seat * k->handSize, lower = (1u << at) - 1u, touched = words[HANABI_TOUCHED], field = touched >> base;
        record |= at << 5 | id << 8;
        if (type == HANABI_MOVE_PLAY && hanabi_height(fireworks, colour) == id % 5u) {
            words[HANABI_FIREWORKS] = fireworks + (1u << 3u * colour);
            reward = 1.0f;
            if (id % 5u == 4u && words[HANABI_INFO] < HANABI_INFORMATION) words[HANABI_INFO]++;
            if (words[HANABI_FIREWORKS] == HANABI_COMPLETE) *over = 1u;
            record |= 1u << 2;
        } else {
            hanabi_bury(words, id);
            if (type == HANABI_MOVE_PLAY) {
                record |= 2u << 2;
                if (--words[HANABI_LIFE] == 0u) {
                    reward = -(float)hanabi_score(fireworks);
                    *over = 1u;
                }
            } else {
                words[HANABI_INFO]++;
                record |= 3u << 2;
            }
        }
        hanabi_colour(words, &l->table, colour);
        for (uint32_t s = at; s + 1u < k->handSize; s++) hand[s] = hand[s + 1u];
        hand[k->handSize - 1u] = hanabi_draw(k, words, l);
        words[HANABI_TOUCHED] = (touched & ~(((1u << k->handSize) - 1u) << base)) | (((field & lower) | (field >> 1 & ~lower)) & ((1u << (k->handSize - 1u)) - 1u)) << base;
    } else {
        words[HANABI_INFO]--;
        record |= (type + 2u) << 2 | at << 5 | value << 8 | hanabi_reveal(k, words, &l->table, at, type, value) << 13;
    }
    words[HANABI_TURN]++;
    if (words[HANABI_FINAL] && --words[HANABI_FINAL] == 0u) *over = 1u;
    words[HANABI_LAST] = record;
    words[HANABI_ACTOR] = seat + 1u == k->players ? 0u : seat + 1u;
    words[HANABI_THINK] = 0u;
    return reward;
}

static inline __attribute__((always_inline)) void hanabi_look(const hanabiKnobs *k, const uint32_t *words, float *view) {
    uint32_t cursor = words[HANABI_CURSOR], slot = words[cursor], seat = cursor / k->handSize, index = cursor % k->handSize;
    uint32_t id = hanabi_id(slot), colours = hanabi_colours(slot), ranks = hanabi_ranks(slot), colour = 0u, rank = 0u;
    if (slot != HANABI_NONE) {
        colour = seat ? id / 5u + 1u : colours & (colours - 1u) ? 0u : (uint32_t)__builtin_ctz(colours) / 5u + 1u;
        rank = seat ? id % 5u + 1u : ranks & (ranks - 1u) ? 0u : (uint32_t)__builtin_ctz(ranks) + 1u;
    }
    uint32_t known = colour && rank ? 5u * (colour - 1u) + rank - 1u : HANABI_NONE;
    view[0] = (float)seat / (float)(k->players - 1u);
    view[1] = (float)index / (float)(k->handSize - 1u);
    view[2] = 0.2f * (float)colour;
    view[3] = 0.2f * (float)rank;
    view[4] = 0.2f * (float)__builtin_popcount(colours);
    view[5] = 0.2f * (float)__builtin_popcount(ranks);
    view[6] = words[HANABI_TOUCHED] >> cursor & 1u ? 1.0f : 0.0f;
    view[7] = slot & HANABI_CLUED ? 1.0f : 0.0f;
    view[8] = known == HANABI_NONE ? 0.0f : (float)(HANABI_COPIES[known % 5u] - hanabi_discarded(words, known)) / 3.0f;
}

static void hanabi_scene(const hanabiKnobs *k, const uint32_t *words, float *view) {
    uint32_t fireworks = words[HANABI_FIREWORKS];
    for (uint32_t c = 0u; c < HANABI_COLOURS; c++) view[9u + c] = 0.2f * (float)hanabi_height(fireworks, c);
    view[14] = (float)words[HANABI_INFO] / (float)HANABI_INFORMATION;
    view[15] = (float)words[HANABI_LIFE] / (float)HANABI_LIVES;
    view[16] = (float)words[HANABI_DECK] / (float)(HANABI_CARDS - k->players * k->handSize);
    view[17] = (float)words[HANABI_ACTOR] / (float)(k->players - 1u);
}

static void hanabi_begin(const hanabiKnobs *k, uint32_t seed, uint32_t *words, hanabiLane *l) {
    uint32_t game = words[HANABI_GAME], seats = k->players * k->handSize, draws[64];
    for (uint32_t block = 0u; block < 4u; block++) {
        u44 r = philox4(seed, ENV_RNG_STREAM, (lane4)game, 4u * block + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
        uint32x4x4_t v = { { (uint32x4_t)r.x, (uint32x4_t)r.y, (uint32x4_t)r.z, (uint32x4_t)r.w } };
        vst4q_u32(draws + 16u * block, v);
    }
    uint8_t *deck = l->deck;
    __builtin_memcpy(deck, HANABI_ORDERED, HANABI_CARDS);
    for (uint32_t n = HANABI_CARDS; n > 1u; n--) {
        uint32_t j = (uint32_t)(((uint64_t)draws[HANABI_CARDS - n] * n) >> 32);
        uint8_t swap = deck[n - 1u];
        deck[n - 1u] = deck[j];
        deck[j] = swap;
    }
    for (uint32_t q = 0u; q < HANABI_SLOTS; q++) words[q] = q < seats ? hanabi_card(deck[q]) : 0u;
    words[HANABI_FIREWORKS] = 0u;
    words[HANABI_INFO] = HANABI_INFORMATION;
    words[HANABI_LIFE] = HANABI_LIVES;
    words[HANABI_DECK] = HANABI_CARDS - seats;
    words[HANABI_DISCARDS] = 0u;
    words[HANABI_DISCARDS_HIGH] = 0u;
    words[HANABI_GONE] = 0u;
    words[HANABI_LAST_COPY] = HANABI_FIVES;
    words[HANABI_TOUCHED] = 0u;
    words[HANABI_ACTOR] = game % k->players;
    words[HANABI_THINK] = 0u;
    words[HANABI_CURSOR] = 0u;
    words[HANABI_FINAL] = 0u;
    words[HANABI_TURN] = 0u;
    words[HANABI_LAST] = 0u;
    l->table = (hanabiTable){ HANABI_ONES, 0u, 0u, HANABI_FIVES, 0u };
    hanabi_look(k, words, l->view);
    hanabi_scene(k, words, l->view);
}

static void hanabi_literal(const hanabiKnobs *k, const uint32_t *words, hanabiLane *l, uint32_t seat, uint32_t slice) {
    const uint32_t *hand = words + seat * k->handSize;
    const hanabiTable *t = &l->table;
    hanabiPlan *p = &l->plan;
    if (slice == 0u) {
        p->move = HANABI_IDLE;
        for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++)
            if (!(hand[s] & HANABI_PLAUSIBLE & ~t->playable)) {
                p->move = HANABI_MOVE_PLAY | s << 2;
                return;
            }
        return;
    }
    if (slice < k->players) {
        if (p->move != HANABI_IDLE || !words[HANABI_INFO]) return;
        uint32_t target = (seat + slice) % k->players;
        const uint32_t *other = words + target * k->handSize;
        for (uint32_t s = 0u; s < k->handSize && other[s] != HANABI_NONE; s++) {
            uint32_t id = hanabi_id(other[s]);
            if (!(t->playable >> id & 1u) || !(other[s] & HANABI_PLAUSIBLE & ~t->playable)) continue;
            uint32_t ranks = hanabi_ranks(other[s]);
            p->move = ranks & (ranks - 1u) ? hanabi_hint(HANABI_MOVE_RANK, target, id % 5u) : hanabi_hint(HANABI_MOVE_COLOUR, target, id / 5u);
            return;
        }
        return;
    }
    if (p->move != HANABI_IDLE) return;
    if (words[HANABI_INFO] < HANABI_INFORMATION) {
        p->move = HANABI_MOVE_DISCARD;
        for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++)
            if (!(hand[s] & HANABI_PLAUSIBLE & ~t->useless)) {
                p->move = HANABI_MOVE_DISCARD | s << 2;
                return;
            }
        return;
    }
    uint32_t target = seat + 1u == k->players ? 0u : seat + 1u;
    const uint32_t *next = words + target * k->handSize;
    for (uint32_t s = 0u; s < k->handSize && next[s] != HANABI_NONE; s++) {
        uint32_t ranks = hanabi_ranks(next[s]);
        if (ranks & (ranks - 1u)) {
            p->move = hanabi_hint(HANABI_MOVE_RANK, target, hanabi_id(next[s]) % 5u);
            return;
        }
    }
    for (uint32_t s = 0u; s < k->handSize && next[s] != HANABI_NONE; s++) {
        uint32_t colours = hanabi_colours(next[s]);
        if (colours & (colours - 1u)) {
            p->move = hanabi_hint(HANABI_MOVE_COLOUR, target, hanabi_id(next[s]) / 5u);
            return;
        }
    }
    p->move = hanabi_hint(HANABI_MOVE_RANK, target, hanabi_id(next[0]) % 5u);
}

static void hanabi_survey(const hanabiKnobs *k, const uint32_t *words, hanabiLane *l, uint32_t seat) {
    const uint32_t *hand = words + seat * k->handSize;
    const hanabiTable *t = &l->table;
    hanabiPlan *p = &l->plan;
    uint32_t seen = hanabi_seen(k, words, t, seat, &p->claims);
    p->play = p->marked = p->useless = p->chop = HANABI_EMPTY_SLOT;
    for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++) {
        uint32_t possible = hand[s] & HANABI_PLAUSIBLE & ~seen;
        if (p->chop == HANABI_EMPTY_SLOT && !(hand[s] & HANABI_CLUED)) p->chop = s;
        if (!possible) continue;
        if (p->play == HANABI_EMPTY_SLOT && !(possible & ~t->playable)) p->play = s;
        if (p->marked == HANABI_EMPTY_SLOT && (hand[s] & HANABI_MARKED) && (possible & t->playable)) p->marked = s;
        if (p->useless == HANABI_EMPTY_SLOT && !(possible & ~t->useless)) p->useless = s;
    }
    p->save = p->clue = p->stall = HANABI_IDLE;
    p->saveValue = p->clueValue = p->stallValue = -1000;
    p->urgent = 0u;
    if (!words[HANABI_INFO]) return;
    uint32_t next = seat + 1u == k->players ? 0u : seat + 1u, chop = HANABI_EMPTY_SLOT, calm = 0u;
    const uint32_t *after = words + next * k->handSize;
    for (uint32_t s = 0u; s < k->handSize && after[s] != HANABI_NONE; s++) {
        uint32_t possible = after[s] & HANABI_PLAUSIBLE & ~t->exhausted;
        if (chop == HANABI_EMPTY_SLOT && !(after[s] & HANABI_CLUED)) chop = s;
        calm |= ((after[s] & HANABI_MARKED) && (possible & t->playable)) || (possible && !(possible & ~t->playable)) || (possible && !(possible & ~t->useless));
    }
    p->urgent = chop != HANABI_EMPTY_SLOT && (t->critical >> hanabi_id(after[chop]) & 1u) && !calm;
}

static void hanabi_weigh(const hanabiKnobs *k, const uint32_t *words, hanabiLane *l, uint32_t seat, uint32_t distance, uint32_t type, uint32_t all) {
    const hanabiTable *t = &l->table;
    hanabiPlan *p = &l->plan;
    if (!words[HANABI_INFO]) return;
    uint32_t target = (seat + distance) % k->players;
    const uint32_t *hand = words + target * k->handSize;
    if (type == HANABI_MOVE_COLOUR) hanabi_summarize(k, hand, t, p->claims, &p->hand);
    const hanabiHand h = p->hand;
    uint32_t fields = type == HANABI_MOVE_COLOUR ? h.colours : h.ranks;
    uint32_t wanted = all ? 31u : h.playable | (distance == 1u && h.chop != HANABI_EMPTY_SLOT ? 1u << h.chop : 0u);
    for (uint32_t value = 0u; value < 5u; value++) {
        uint32_t touched = fields >> 5u * value & 31u;
        if (!(touched & wanted)) continue;
        uint32_t focus = hanabi_aim(h.chop, h.clued, touched), slot = hand[focus], id = hanabi_id(slot);
        uint32_t reads = (slot & hanabi_group(type, value) & t->playable) != 0u;
        if (reads && !(h.playable >> focus & 1u)) continue;
        uint32_t fresh = touched & ~h.clued, others = fresh & ~(1u << focus), move = hanabi_hint(type, target, value);
        int32_t bad = __builtin_popcount(others & h.bad), extra = __builtin_popcount(others & h.playable), saved = __builtin_popcount(fresh & h.critical);
        if (reads) {
            if (!(slot & HANABI_MARKED) && !hanabi_known(slot, t) && !(p->claims >> id & 1u)) {
                int32_t worth = 64 - 8 * bad + 4 * extra - (int32_t)(id % 5u) - 2 * (int32_t)(distance - 1u);
                if (worth > p->clueValue) { p->clueValue = worth; p->clue = move; }
            }
        } else if (distance == 1u && focus == h.chop && (h.critical >> focus & 1u)) {
            int32_t worth = 32 - 8 * bad;
            if (worth > p->saveValue) { p->saveValue = worth; p->save = move; }
        }
        int32_t stall = 4 * saved - 4 * bad - __builtin_popcount(touched);
        if (stall > p->stallValue) { p->stallValue = stall; p->stall = move; }
    }
}

static uint32_t hanabi_settle(const hanabiKnobs *k, const uint32_t *words, const hanabiPlan *p, uint32_t seat) {
    uint32_t info = words[HANABI_INFO];
    if (p->play != HANABI_EMPTY_SLOT) return HANABI_MOVE_PLAY | p->play << 2;
    if (info && p->urgent && p->save != HANABI_IDLE) return p->save;
    if (p->marked != HANABI_EMPTY_SLOT) return HANABI_MOVE_PLAY | p->marked << 2;
    if (info && p->clue != HANABI_IDLE) return p->clue;
    if (info < HANABI_INFORMATION) {
        if (p->useless != HANABI_EMPTY_SLOT) return HANABI_MOVE_DISCARD | p->useless << 2;
        if (p->chop != HANABI_EMPTY_SLOT) return HANABI_MOVE_DISCARD | p->chop << 2;
        const uint32_t *hand = words + seat * k->handSize;
        for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++)
            if (!(hand[s] & HANABI_MARKED)) return HANABI_MOVE_DISCARD | s << 2;
        return HANABI_MOVE_DISCARD;
    }
    if (p->stall != HANABI_IDLE) return p->stall;
    uint32_t next = seat + 1u == k->players ? 0u : seat + 1u;
    return hanabi_hint(HANABI_MOVE_RANK, next, hanabi_id(words[next * k->handSize]) % 5u);
}

static float hanabi_think(const hanabiKnobs *k, uint32_t *words, hanabiLane *l, uint32_t *over) {
    uint32_t seat = words[HANABI_ACTOR], slice = words[HANABI_THINK];
    if (k->partner == HANABI_LITERAL) hanabi_literal(k, words, l, seat, slice);
    else if (slice == 0u) hanabi_survey(k, words, l, seat);
    else if (slice + 1u < k->thinkSteps) hanabi_weigh(k, words, l, seat, (slice + 1u) >> 1, slice & 1u ? HANABI_MOVE_COLOUR : HANABI_MOVE_RANK, words[HANABI_INFO] == HANABI_INFORMATION);
    else l->plan.move = hanabi_settle(k, words, &l->plan, seat);
    if (slice + 1u == k->thinkSteps) return hanabi_apply(k, words, l, seat, l->plan.move, over);
    words[HANABI_THINK] = slice + 1u;
    return 0.0f;
}

static inline __attribute__((always_inline)) uint32_t hanabi_press(const hanabiKnobs *k, uint32_t *words, uint32_t action) {
    uint32_t cursor = words[HANABI_CURSOR], positions = k->players * k->handSize;
    if (action == HANABI_NEXT) {
        words[HANABI_CURSOR] = cursor + 1u == positions ? 0u : cursor + 1u;
        return HANABI_IDLE;
    }
    if (action == HANABI_PREVIOUS) {
        words[HANABI_CURSOR] = cursor ? cursor - 1u : positions - 1u;
        return HANABI_IDLE;
    }
    uint32_t slot = words[cursor], seat = cursor / k->handSize, index = cursor % k->handSize, id = hanabi_id(slot);
    if (slot == HANABI_NONE) return HANABI_IDLE;
    if (action == HANABI_PLAY) return seat ? HANABI_IDLE : HANABI_MOVE_PLAY | index << 2;
    if (action == HANABI_DISCARD) return seat || words[HANABI_INFO] == HANABI_INFORMATION ? HANABI_IDLE : HANABI_MOVE_DISCARD | index << 2;
    if (!seat || !words[HANABI_INFO]) return HANABI_IDLE;
    return action == HANABI_HINT_COLOUR ? hanabi_hint(HANABI_MOVE_COLOUR, seat, id / 5u) : hanabi_hint(HANABI_MOVE_RANK, seat, id % 5u);
}

static uint32_t hanabi_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 6.0f ? (uint32_t)c : 0u;
}

static __attribute__((noinline)) float hanabi_advance(const hanabiKnobs *k, uint32_t seed, uint32_t *words, hanabiLane *l, float *obs, uint32_t *done, uint32_t move) {
    uint32_t over = 0u, turn = words[HANABI_TURN];
    float reward = move == HANABI_IDLE ? hanabi_think(k, words, l, &over) : hanabi_apply(k, words, l, 0u, move, &over);
    *done = over;
    if (over) {
        words[HANABI_GAME]++;
        hanabi_begin(k, seed, words, l);
    } else if (words[HANABI_TURN] != turn) {
        hanabi_look(k, words, l->view);
        hanabi_scene(k, words, l->view);
    }
    __builtin_memcpy(obs, l->view, sizeof l->view);
    return reward;
}

float hanabi_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    if (instance >= 3u) __builtin_trap();
    const hanabiKnobs *k = &HANABI_KNOBS[instance];
    hanabiLane *l = hanabi_lane(extra);
    uint32_t move = HANABI_IDLE;
    if (!words[HANABI_ACTOR]) {
        uint32_t cursor = words[HANABI_CURSOR];
        move = hanabi_press(k, words, hanabi_action(ctrl));
        if (move == HANABI_IDLE) {
            if (words[HANABI_CURSOR] != cursor) hanabi_look(k, words, l->view);
            __builtin_memcpy(obs, l->view, sizeof l->view);
            *done = 0u;
            return 0.0f;
        }
    }
    return hanabi_advance(k, seed, words, l, obs, done, move);
}

void hanabi_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    hanabiLane *l = hanabi_lane(extra);
    __builtin_memset(l, 0, sizeof *l);
    l->aim.key = 0xFFFFFFFFu;
    for (uint32_t i = 0u; i < HANABI_WORDS; i++) words[i] = 0u;
    hanabi_begin(&HANABI_KNOBS[instance], seed, words, l);
}

void hanabi_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)words; (void)step; (void)shared;
    __builtin_memcpy(obs, hanabi_peek(extra)->view, sizeof hanabi_peek(extra)->view);
}

static uint32_t hanabi_need(uint32_t slot) {
    uint32_t colours = hanabi_colours(slot), ranks = hanabi_ranks(slot);
    return (uint32_t)((colours & (colours - 1u)) != 0u) + (uint32_t)((ranks & (ranks - 1u)) != 0u);
}

static uint32_t hanabi_teach(const hanabiKnobs *k, const uint32_t *words, const hanabiTable *t, uint32_t claims, int32_t *best) {
    uint32_t choice = HANABI_IDLE, soon = (t->playable & HANABI_BELOW_FIVE) << 1 & ~t->useless;
    *best = 0;
    for (uint32_t distance = 1u; distance < k->players; distance++) {
        const uint32_t *hand = words + distance * k->handSize;
        for (uint32_t type = HANABI_MOVE_COLOUR; type <= HANABI_MOVE_RANK; type++) {
            uint32_t present = 0u;
            for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++) {
                uint32_t id = hanabi_id(hand[s]);
                present |= 1u << (type == HANABI_MOVE_COLOUR ? id / 5u : id % 5u);
            }
            for (; present; present &= present - 1u) {
                uint32_t value = (uint32_t)__builtin_ctz(present), group = hanabi_group(type, value);
                int32_t worth = 0;
                for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++) {
                    uint32_t id = hanabi_id(hand[s]), after = group >> id & 1u ? hand[s] & ((uint32_t)~HANABI_PLAUSIBLE | group) : hand[s] & ~group;
                    int32_t gain = (int32_t)hanabi_need(hand[s]) - (int32_t)hanabi_need(after);
                    if ((t->playable >> id & 1u) && !(claims >> id & 1u)) {
                        if (!(hand[s] & HANABI_PLAUSIBLE & ~t->playable)) continue;
                        worth += !(after & HANABI_PLAUSIBLE & ~t->playable) ? 16 : 8 * gain;
                    } else if (soon >> id & 1u) worth += 3 * gain;
                }
                worth -= (int32_t)(distance - 1u);
                if (worth > *best) { *best = worth; choice = hanabi_hint(type, distance, value); }
            }
        }
    }
    return choice;
}

static uint32_t hanabi_worth(const uint32_t *words, const hanabiTable *t, uint32_t id, uint32_t spares) {
    if ((t->useless >> id & 1u) || (spares >> id & 1u)) return 0u;
    if (t->critical >> id & 1u) return 1000u;
    return 100u - 10u * (id % 5u - hanabi_height(words[HANABI_FIREWORKS], id / 5u));
}

static uint32_t hanabi_plan(const hanabiKnobs *k, const uint32_t *words, hanabiLane *l) {
    hanabiPlan *p = &l->plan;
    const hanabiTable *t = &l->table;
    uint32_t info = words[HANABI_INFO], claims = 0u, spares = 0u, play = HANABI_EMPTY_SLOT, playRank = 9u;
    for (uint32_t q = k->handSize; q < k->players * k->handSize; q++) {
        if (words[q] == HANABI_NONE) continue;
        uint32_t id = hanabi_id(words[q]);
        spares |= 1u << id;
        uint32_t ready = k->partner == HANABI_LITERAL ? !(words[q] & HANABI_PLAUSIBLE & ~t->playable) : (words[q] & HANABI_MARKED) || hanabi_known(words[q], t);
        if (ready && (t->playable >> id & 1u)) claims |= 1u << id;
    }
    for (uint32_t s = 0u; s < k->handSize && words[s] != HANABI_NONE; s++) {
        uint32_t id = hanabi_id(words[s]);
        if ((t->playable >> id & 1u) && !(claims >> id & 1u) && id % 5u < playRank) {
            play = s;
            playRank = id % 5u;
        }
    }
    if (play != HANABI_EMPTY_SLOT) return HANABI_MOVE_PLAY | play << 2;
    uint32_t hint = HANABI_IDLE;
    int32_t hintValue = 0;
    if (k->partner == HANABI_LITERAL) hint = hanabi_teach(k, words, t, claims, &hintValue);
    else {
        hanabi_survey(k, words, l, 0u);
        for (uint32_t slice = 1u; slice + 1u < k->thinkSteps; slice++) hanabi_weigh(k, words, l, 0u, (slice + 1u) >> 1, slice & 1u ? HANABI_MOVE_COLOUR : HANABI_MOVE_RANK, 1u);
        if (p->urgent && p->save != HANABI_IDLE) hint = p->save, hintValue = 64;
        else if (p->clue != HANABI_IDLE) hint = p->clue, hintValue = p->clueValue;
    }
    if (info && hint != HANABI_IDLE && hintValue >= 8) return hint;
    uint32_t discard = 0u, cheapest = 0xFFFFFFFFu;
    for (uint32_t s = 0u; s < k->handSize && words[s] != HANABI_NONE; s++) {
        uint32_t worth = hanabi_worth(words, t, hanabi_id(words[s]), spares);
        for (uint32_t r = 0u; r < s; r++) worth = hanabi_id(words[r]) == hanabi_id(words[s]) ? 0u : worth;
        if (worth < cheapest) {
            cheapest = worth;
            discard = s;
        }
    }
    if (info < HANABI_INFORMATION && (cheapest < 1000u || !info)) return HANABI_MOVE_DISCARD | discard << 2;
    if (info && hint != HANABI_IDLE) return hint;
    if (k->partner == HANABI_FOCUS && p->stall != HANABI_IDLE && info) return p->stall;
    if (info < HANABI_INFORMATION) return HANABI_MOVE_DISCARD | discard << 2;
    return hanabi_hint(HANABI_MOVE_RANK, 1u, hanabi_id(words[k->handSize]) % 5u);
}

void hanabi_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    const hanabiKnobs *k = &HANABI_KNOBS[instance];
    hanabiLane *l = hanabi_lane(extra);
    if (words[HANABI_ACTOR]) {
        ctrl[0] = (float)HANABI_NEXT;
        return;
    }
    uint32_t key = words[HANABI_GAME] << 8 | words[HANABI_TURN], cursor = words[HANABI_CURSOR], positions = k->players * k->handSize;
    if (l->aim.key != key) {
        uint32_t move = hanabi_plan(k, words, l), type = move & 3u, at = move >> 2 & 7u, value = move >> 5 & 7u, target = at, reach = 0xFFFFFFFFu;
        if (type >= HANABI_MOVE_COLOUR)
            for (uint32_t s = 0u; s < k->handSize; s++) {
                uint32_t q = at * k->handSize + s, slot = words[q];
                if (slot == HANABI_NONE || !(hanabi_group(type, value) >> hanabi_id(slot) & 1u)) continue;
                uint32_t forward = (q + positions - cursor) % positions, backward = (cursor + positions - q) % positions;
                uint32_t distance = forward < backward ? forward : backward;
                if (distance < reach) {
                    reach = distance;
                    target = q;
                }
            }
        l->aim.key = key;
        l->aim.target = target;
        l->aim.action = type == HANABI_MOVE_PLAY ? HANABI_PLAY : type == HANABI_MOVE_DISCARD ? HANABI_DISCARD : type == HANABI_MOVE_COLOUR ? HANABI_HINT_COLOUR : HANABI_HINT_RANK;
    }
    uint32_t target = l->aim.target, forward = (target + positions - cursor) % positions, backward = (cursor + positions - target) % positions;
    ctrl[0] = (float)(cursor == target ? l->aim.action : forward <= backward ? HANABI_NEXT : HANABI_PREVIOUS);
}
