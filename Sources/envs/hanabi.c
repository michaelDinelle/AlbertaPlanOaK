#include "../env.h"

typedef struct { uint32_t players, handSize, partner; } hanabiKnobs;

static const hanabiKnobs HANABI_KNOBS[3] = {
    { 2u, 5u, 0u },
    { 2u, 5u, 1u },
    { 4u, 4u, 1u }
};

static const uint32_t HANABI_COPIES[5] = { 3u, 2u, 2u, 2u, 1u };

enum { HANABI_COLOURS = 5, HANABI_CARDS = 50, HANABI_INFORMATION = 8, HANABI_LIVES = 3, HANABI_SLOTS = 16, HANABI_PLAYERS = 4 };
enum { HANABI_NEXT, HANABI_PREVIOUS, HANABI_PLAY, HANABI_DISCARD, HANABI_HINT_COLOUR, HANABI_HINT_RANK };
enum { HANABI_LITERAL, HANABI_FOCUS };
enum { HANABI_FIREWORKS = HANABI_SLOTS, HANABI_INFO, HANABI_LIFE, HANABI_DECK, HANABI_DISCARDS, HANABI_DISCARDS_HIGH, HANABI_GONE, HANABI_LAST_COPY,
       HANABI_TOUCHED, HANABI_CLOCK, HANABI_CURSOR, HANABI_FINAL, HANABI_TURN, HANABI_GAME, HANABI_ENDED, HANABI_MOVES, HANABI_WORDS = HANABI_MOVES + HANABI_PLAYERS };
enum { HANABI_SEEN_FIREWORKS, HANABI_SEEN_INFO = HANABI_COLOURS, HANABI_SEEN_LIFE, HANABI_SEEN_DECK, HANABI_SEEN_CURSOR, HANABI_SEEN_CLOCK = HANABI_SEEN_CURSOR + 2,
       HANABI_SEEN_MOVES, HANABI_RECALL = 5, HANABI_VIEW = 2 * HANABI_SLOTS + HANABI_SEEN_MOVES + HANABI_RECALL * HANABI_PLAYERS };
_Static_assert(HANABI_VIEW <= OAK_ENV_OBS_MAX && HANABI_WORDS <= OAK_ENV_WORDS_MAX, "");
enum { HANABI_PLAUSIBLE = 0x1FFFFFF, HANABI_NONE = 0x3E000000, HANABI_CLUED = 0x40000000, HANABI_FORCED = 0x20000000 };
static const uint32_t HANABI_MARKED = 0x80000000u;
static const uint32_t HANABI_PATIENCE = HANABI_SLOTS;
enum { HANABI_MOVE_PLAY, HANABI_MOVE_DISCARD, HANABI_MOVE_COLOUR, HANABI_MOVE_RANK, HANABI_IDLE = 0xFFu, HANABI_EMPTY_SLOT = 0xFFu };
enum { HANABI_PLAYED = 1, HANABI_MISFIRED, HANABI_DISCARDED, HANABI_TOLD_COLOUR, HANABI_TOLD_RANK };
enum { HANABI_COMPLETE = 0x5B6Du, HANABI_ONES = 0x108421u, HANABI_FIVES = 0x1084210u };

static const uint8_t HANABI_ORDERED[HANABI_CARDS] = { 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 10, 11, 11, 12, 12, 13, 13, 14,
                                                     15, 15, 15, 16, 16, 17, 17, 18, 18, 19, 20, 20, 20, 21, 21, 22, 22, 23, 23, 24 };

typedef struct { uint32_t playable, useless, exhausted, critical, played; } hanabiTable;

typedef struct { uint32_t clued, chop, colours, ranks, playable, bad, critical; } hanabiHand;

typedef struct {
    hanabiHand hand;
    uint32_t play, marked, useless, chop, urgent, claims, save, clue, stall;
    int32_t saveValue, clueValue, stallValue;
} hanabiPlan;

typedef struct { uint32_t key, target, action; } hanabiAim;

typedef struct {
    uint8_t deck[HANABI_CARDS];
    uint8_t spare[2];
    hanabiTable table;
    hanabiPlan plan;
    hanabiAim aim;
    float view[HANABI_VIEW];
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
    uint32_t type = move & 3u, at = move >> 2 & 7u, value = move >> 5 & 7u, record = seat | (move & HANABI_FORCED);
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
            record |= HANABI_PLAYED << 2;
        } else {
            hanabi_bury(words, id);
            if (type == HANABI_MOVE_PLAY) {
                record |= HANABI_MISFIRED << 2;
                if (--words[HANABI_LIFE] == 0u) {
                    reward = -(float)hanabi_score(fireworks);
                    *over = 1u;
                }
            } else {
                words[HANABI_INFO]++;
                record |= HANABI_DISCARDED << 2;
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
    words[HANABI_MOVES + seat] = record;
    return reward;
}

static uint32_t hanabi_width(const hanabiKnobs *k) { return 2u * k->players * k->handSize + HANABI_SEEN_MOVES + HANABI_RECALL * k->players; }

static inline __attribute__((always_inline)) void hanabi_point(const hanabiKnobs *k, const uint32_t *words, float *view) {
    uint32_t cursor = words[HANABI_CURSOR];
    float *tail = view + 2u * k->players * k->handSize;
    tail[HANABI_SEEN_CURSOR] = (float)(cursor / k->handSize) / (float)(k->players - 1u);
    tail[HANABI_SEEN_CURSOR + 1] = (float)(cursor % k->handSize) / (float)(k->handSize - 1u);
    tail[HANABI_SEEN_CLOCK] = (float)words[HANABI_CLOCK] / (float)HANABI_PATIENCE;
}

static void hanabi_recall(const hanabiKnobs *k, uint32_t record, float *move) {
    uint32_t kind = record >> 2 & 7u, at = record >> 5 & 7u, value = record >> 8 & 31u, target = 0u, slot = 0u, colour = 0u, rank = 0u;
    if (kind == HANABI_TOLD_COLOUR || kind == HANABI_TOLD_RANK) {
        target = at;
        colour = kind == HANABI_TOLD_COLOUR ? value + 1u : 0u;
        rank = kind == HANABI_TOLD_RANK ? value + 1u : 0u;
    } else if (kind) {
        slot = at + 1u;
        colour = value / 5u + 1u;
        rank = value % 5u + 1u;
    }
    move[0] = 0.2f * (float)kind;
    move[1] = (float)target / (float)(k->players - 1u);
    move[2] = (float)slot / (float)k->handSize;
    move[3] = 0.2f * (float)colour;
    move[4] = 0.2f * (float)rank;
}

static void hanabi_look(const hanabiKnobs *k, const uint32_t *words, float *view) {
    uint32_t seats = k->players * k->handSize, fireworks = words[HANABI_FIREWORKS];
    for (uint32_t q = 0u; q < seats; q++) {
        uint32_t slot = words[q], colour = 0u, rank = 0u;
        if (q < k->handSize) {
            uint32_t colours = hanabi_colours(slot), ranks = hanabi_ranks(slot);
            if (colours && !(colours & (colours - 1u))) colour = (uint32_t)__builtin_ctz(colours) / 5u + 1u;
            if (ranks && !(ranks & (ranks - 1u))) rank = (uint32_t)__builtin_ctz(ranks) + 1u;
        } else if (slot != HANABI_NONE) {
            colour = hanabi_id(slot) / 5u + 1u;
            rank = hanabi_id(slot) % 5u + 1u;
        }
        view[2u * q] = 0.2f * (float)colour;
        view[2u * q + 1u] = 0.2f * (float)rank;
    }
    float *tail = view + 2u * seats;
    for (uint32_t c = 0u; c < HANABI_COLOURS; c++) tail[HANABI_SEEN_FIREWORKS + c] = 0.2f * (float)hanabi_height(fireworks, c);
    tail[HANABI_SEEN_INFO] = (float)words[HANABI_INFO] / (float)HANABI_INFORMATION;
    tail[HANABI_SEEN_LIFE] = (float)words[HANABI_LIFE] / (float)HANABI_LIVES;
    tail[HANABI_SEEN_DECK] = (float)words[HANABI_DECK] / (float)(HANABI_CARDS - seats);
    for (uint32_t seat = 0u; seat < k->players; seat++) hanabi_recall(k, words[HANABI_MOVES + seat], tail + HANABI_SEEN_MOVES + HANABI_RECALL * seat);
    hanabi_point(k, words, view);
}

static uint32_t hanabi_touch(const hanabiKnobs *k, const uint32_t *hand, uint32_t group) {
    uint32_t touched = 0u;
    for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++) touched |= (group >> hanabi_id(hand[s]) & 1u) << s;
    return touched;
}

static uint32_t hanabi_cue(const hanabiKnobs *k, const uint32_t *words, const hanabiTable *t, uint32_t seat) {
    if (!words[HANABI_INFO]) return HANABI_IDLE;
    for (uint32_t distance = 1u; distance < k->players; distance++) {
        uint32_t target = (seat + distance) % k->players, clued = 0u, chop = HANABI_EMPTY_SLOT;
        const uint32_t *other = words + target * k->handSize;
        for (uint32_t s = 0u; s < k->handSize && other[s] != HANABI_NONE; s++) {
            if (other[s] & HANABI_CLUED) clued |= 1u << s;
            else if (chop == HANABI_EMPTY_SLOT) chop = s;
        }
        for (uint32_t s = 0u; s < k->handSize && other[s] != HANABI_NONE; s++) {
            uint32_t id = hanabi_id(other[s]);
            if (!(t->playable >> id & 1u) || (other[s] & HANABI_MARKED) || !(other[s] & HANABI_PLAUSIBLE & ~t->playable)) continue;
            if (k->partner == HANABI_FOCUS) {
                for (uint32_t type = HANABI_MOVE_COLOUR; type <= HANABI_MOVE_RANK; type++) {
                    uint32_t value = type == HANABI_MOVE_COLOUR ? id / 5u : id % 5u;
                    if (hanabi_aim(chop, clued, hanabi_touch(k, other, hanabi_group(type, value))) == s) return hanabi_hint(type, target, value);
                }
                continue;
            }
            uint32_t ranks = hanabi_ranks(other[s]);
            return ranks & (ranks - 1u) ? hanabi_hint(HANABI_MOVE_RANK, target, id % 5u) : hanabi_hint(HANABI_MOVE_COLOUR, target, id / 5u);
        }
    }
    return HANABI_IDLE;
}

static uint32_t hanabi_literal(const hanabiKnobs *k, const uint32_t *words, const hanabiTable *t, uint32_t seat) {
    const uint32_t *hand = words + seat * k->handSize;
    for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++)
        if (!(hand[s] & HANABI_PLAUSIBLE & ~t->playable)) return HANABI_MOVE_PLAY | s << 2;
    uint32_t cue = hanabi_cue(k, words, t, seat);
    if (cue != HANABI_IDLE) return cue;
    if (words[HANABI_INFO] < HANABI_INFORMATION) {
        for (uint32_t s = 0u; s < k->handSize && hand[s] != HANABI_NONE; s++)
            if (!(hand[s] & HANABI_PLAUSIBLE & ~t->useless)) return HANABI_MOVE_DISCARD | s << 2;
        return HANABI_MOVE_DISCARD;
    }
    uint32_t target = seat + 1u == k->players ? 0u : seat + 1u;
    const uint32_t *next = words + target * k->handSize;
    for (uint32_t s = 0u; s < k->handSize && next[s] != HANABI_NONE; s++) {
        uint32_t ranks = hanabi_ranks(next[s]);
        if (ranks & (ranks - 1u)) return hanabi_hint(HANABI_MOVE_RANK, target, hanabi_id(next[s]) % 5u);
    }
    for (uint32_t s = 0u; s < k->handSize && next[s] != HANABI_NONE; s++) {
        uint32_t colours = hanabi_colours(next[s]);
        if (colours & (colours - 1u)) return hanabi_hint(HANABI_MOVE_COLOUR, target, hanabi_id(next[s]) / 5u);
    }
    return hanabi_hint(HANABI_MOVE_RANK, target, hanabi_id(next[0]) % 5u);
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

static uint32_t hanabi_decide(const hanabiKnobs *k, const uint32_t *words, hanabiLane *l, uint32_t seat) {
    if (k->partner == HANABI_LITERAL) return hanabi_literal(k, words, &l->table, seat);
    uint32_t all = words[HANABI_INFO] == HANABI_INFORMATION;
    hanabi_survey(k, words, l, seat);
    for (uint32_t distance = 1u; distance < k->players; distance++) {
        hanabi_weigh(k, words, l, seat, distance, HANABI_MOVE_COLOUR, all);
        hanabi_weigh(k, words, l, seat, distance, HANABI_MOVE_RANK, all);
    }
    return hanabi_settle(k, words, &l->plan, seat);
}

static float hanabi_round(const hanabiKnobs *k, uint32_t *words, hanabiLane *l, uint32_t seat, uint32_t *over) {
    float reward = 0.0f;
    for (; seat < k->players && !*over; seat++) reward += hanabi_apply(k, words, l, seat, hanabi_decide(k, words, l, seat), over);
    return reward;
}

static float hanabi_begin(const hanabiKnobs *k, uint32_t seed, uint32_t *words, hanabiLane *l) {
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
    words[HANABI_CLOCK] = 0u;
    words[HANABI_CURSOR] = 0u;
    words[HANABI_FINAL] = 0u;
    words[HANABI_TURN] = 0u;
    for (uint32_t seat = 0u; seat < HANABI_PLAYERS; seat++) words[HANABI_MOVES + seat] = 0u;
    l->table = (hanabiTable){ HANABI_ONES, 0u, 0u, HANABI_FIVES, 0u };
    uint32_t first = game % k->players, over = 0u;
    float reward = first ? hanabi_round(k, words, l, first, &over) : 0.0f;
    if (over) __builtin_trap();
    hanabi_look(k, words, l->view);
    return reward;
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

static uint32_t hanabi_forced(const hanabiKnobs *k, const uint32_t *words) {
    if (words[HANABI_INFO] == HANABI_INFORMATION) return hanabi_hint(HANABI_MOVE_RANK, 1u, hanabi_id(words[k->handSize]) % 5u) | HANABI_FORCED;
    for (uint32_t s = 0u; s < k->handSize && words[s] != HANABI_NONE; s++)
        if (!(words[s] & HANABI_CLUED)) return HANABI_MOVE_DISCARD | s << 2 | HANABI_FORCED;
    return HANABI_MOVE_DISCARD | HANABI_FORCED;
}

static __attribute__((noinline)) float hanabi_advance(const hanabiKnobs *k, uint32_t seed, uint32_t *words, hanabiLane *l, float *obs, uint32_t *done, uint32_t move) {
    uint32_t over = 0u;
    float reward = hanabi_apply(k, words, l, 0u, move, &over);
    reward += hanabi_round(k, words, l, 1u, &over);
    *done = over;
    if (over) {
        words[HANABI_ENDED] = words[HANABI_FIREWORKS] | words[HANABI_LIFE] << 15;
        words[HANABI_GAME]++;
        reward += hanabi_begin(k, seed, words, l);
    } else {
        words[HANABI_CLOCK] = 0u;
        hanabi_look(k, words, l->view);
    }
    __builtin_memcpy(obs, l->view, hanabi_width(k) * sizeof(float));
    return reward;
}

float hanabi_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    if (instance >= 3u) __builtin_trap();
    const hanabiKnobs *k = &HANABI_KNOBS[instance];
    hanabiLane *l = hanabi_lane(extra);
    uint32_t move = hanabi_press(k, words, hanabi_action(ctrl)), clock = words[HANABI_CLOCK] + 1u;
    if (move == HANABI_IDLE && clock == HANABI_PATIENCE) move = hanabi_forced(k, words);
    if (move != HANABI_IDLE) return hanabi_advance(k, seed, words, l, obs, done, move);
    words[HANABI_CLOCK] = clock;
    hanabi_point(k, words, l->view);
    __builtin_memcpy(obs, l->view, hanabi_width(k) * sizeof(float));
    *done = 0u;
    return 0.0f;
}

void hanabi_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    hanabiLane *l = hanabi_lane(extra);
    __builtin_memset(l, 0, sizeof *l);
    l->aim.key = 0xFFFFFFFFu;
    for (uint32_t i = 0u; i < HANABI_WORDS; i++) words[i] = 0u;
    (void)hanabi_begin(&HANABI_KNOBS[instance], seed, words, l);
}

void hanabi_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)words; (void)step; (void)shared;
    __builtin_memcpy(obs, hanabi_peek(extra)->view, hanabi_width(&HANABI_KNOBS[instance]) * sizeof(float));
}

static uint32_t hanabi_plan(const hanabiKnobs *k, const uint32_t *words, const hanabiTable *t) {
    uint32_t claims, seen = hanabi_seen(k, words, t, 0u, &claims), marked = HANABI_EMPTY_SLOT;
    for (uint32_t s = 0u; s < k->handSize && words[s] != HANABI_NONE; s++) {
        uint32_t possible = words[s] & HANABI_PLAUSIBLE & ~seen;
        if (possible && !(possible & ~t->playable)) return HANABI_MOVE_PLAY | s << 2;
        if (marked == HANABI_EMPTY_SLOT && (words[s] & HANABI_MARKED) && (possible & t->playable)) marked = s;
    }
    if (marked != HANABI_EMPTY_SLOT) return HANABI_MOVE_PLAY | marked << 2;
    uint32_t cue = hanabi_cue(k, words, t, 0u);
    return cue != HANABI_IDLE ? cue : hanabi_forced(k, words);
}

void hanabi_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs; (void)shared;
    const hanabiKnobs *k = &HANABI_KNOBS[instance];
    hanabiLane *l = hanabi_lane(extra);
    uint32_t key = words[HANABI_GAME] << 8 | words[HANABI_TURN], cursor = words[HANABI_CURSOR], positions = k->players * k->handSize;
    if (l->aim.key != key) {
        uint32_t move = hanabi_plan(k, words, &l->table), type = move & 3u, at = move >> 2 & 7u, value = move >> 5 & 7u, target = at, reach = 0xFFFFFFFFu;
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
