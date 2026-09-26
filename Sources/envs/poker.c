#include "../env.h"

typedef struct { uint32_t suits, rounds, ante, firstRaise, secondRaise, raiseCap, actions, foldUnowed, raiseCapped, opponent, iterations, sweepSteps; float decay; } pokerKnobs;

static const pokerKnobs POKER_KNOBS[3] = {
    { 1u, 1u, 1u, 1u, 0u, 1u, 2u, 1u, 1u, 0u, 0u, 0u, 0.0f },
    { 2u, 2u, 1u, 2u, 4u, 2u, 3u, 1u, 1u, 1u, 1000u, 0u, 0.0f },
    { 2u, 2u, 1u, 2u, 4u, 2u, 3u, 1u, 1u, 2u, 0u, 4096u, 0.0009765625f }
};

enum { POKER_FOLD, POKER_CALL, POKER_RAISE };
enum { POKER_UNIFORM, POKER_CFR, POKER_FICTITIOUS };
enum { POKER_HAND, POKER_TABLE, POKER_LAST, POKER_VERSION };
enum { POKER_THEIRS = 3, POKER_BOARD = 6, POKER_AT = 9, POKER_SEEN = 16, POKER_MINE_CHIPS = 18, POKER_THEIR_CHIPS = 22, POKER_MOVE = 26, POKER_ENDED = 28 };
enum { POKER_RANKS = 3, POKER_IDS = 6, POKER_NODES = 96, POKER_INFOSETS = 288, POKER_PLAN = 18, POKER_ITEMS = 192 };
enum { POKER_TAG = 4, POKER_BATCH = 5, POKER_DRAWS = POKER_BATCH + 4 * ENV_BATCH };
enum { POKER_NONE = 7, POKER_OVER = 127, POKER_POT = 13, POKER_BETS = 2, POKER_WHOLE = 1 << 24 };
enum { POKER_ORACLE = 0, POKER_COUNTS = 2 * POKER_INFOSETS, POKER_VALUES = 4 * POKER_INFOSETS, POKER_SHARED = POKER_VALUES + 4 };

typedef struct { float4 chance[4][POKER_RANKS], showdown[4][POKER_RANKS]; float folds[POKER_NODES], shows[POKER_NODES]; } pokerRules;
typedef struct { float4 reach[2][POKER_NODES], value[2][POKER_NODES]; } pokerTree;

typedef struct {
    float4 ema[POKER_INFOSETS], snap[POKER_INFOSETS];
    pokerTree tree;
    pokerRules rules;
    uint32_t plan[2][POKER_PLAN], oracle[POKER_PLAN], start, items, studied;
} pokerLearner;

static pokerLearner *poker_learner(uint32_t *extra) { return (pokerLearner *)(((uintptr_t)(extra + POKER_DRAWS) + 15u) & ~(uintptr_t)15u); }

uint32_t poker_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return POKER_DRAWS + (POKER_KNOBS[instance].opponent == POKER_FICTITIOUS ? (uint32_t)(sizeof(pokerLearner) / sizeof(uint32_t)) + 3u : 0u);
}

static uint32_t poker_nodes(const pokerKnobs *k) { return k->rounds > 1u ? POKER_NODES : POKER_IDS; }

static uint32_t poker_root(uint32_t id, uint32_t board) { return POKER_IDS * (1u + 3u * (id - 1u) + board); }

static uint32_t poker_moves(const pokerKnobs *k, uint32_t at) {
    uint32_t raises = at % POKER_IDS % 3u;
    return 2u | (uint32_t)(raises > 0u) | (uint32_t)(raises < k->raiseCap) << POKER_RAISE;
}

static inline __attribute__((always_inline)) uint32_t poker_rank(const pokerKnobs *k, uint32_t card) { return card / k->suits; }

static inline __attribute__((always_inline)) uint32_t poker_raised(const pokerKnobs *k, uint32_t at, uint32_t other) { return other + (at < POKER_IDS ? k->firstRaise : k->secondRaise); }

static uint32_t poker_planned(const uint32_t *plan, uint32_t infoset) { return plan[infoset >> 4] >> (2u * (infoset & 15u)) & 3u; }

static void poker_plan(uint32_t *plan, uint32_t infoset, uint32_t move) {
    uint32_t shift = 2u * (infoset & 15u);
    plan[infoset >> 4] = (plan[infoset >> 4] & ~(3u << shift)) | move << shift;
}

static inline __attribute__((always_inline)) uint32_t poker_legal(const pokerKnobs *k, uint32_t id, uint32_t move) {
    uint32_t raises = id % 3u;
    if (move == POKER_FOLD && raises == 0u) return k->foldUnowed;
    return move == POKER_RAISE && raises == k->raiseCap ? k->raiseCapped : move;
}

static uint32_t poker_answer(const pokerKnobs *k, uint32_t id, uint32_t move) {
    return k->actions == 2u ? (uint32_t)(move == POKER_RAISE || (move == POKER_CALL && id % 3u > 0u)) : move;
}

static inline __attribute__((always_inline)) uint32_t poker_choice(const pokerKnobs *k, const float *ctrl) {
    float c = ctrl[0];
    uint32_t choice = c >= 0.0f && c < (float)k->actions ? (uint32_t)c : 0u;
    return k->actions == 2u ? POKER_RAISE * choice : choice;
}

static inline __attribute__((always_inline)) uint32_t poker_next(const pokerKnobs *k, uint32_t at, uint32_t move, uint32_t board) {
    uint32_t id = at % POKER_IDS;
    if (move == POKER_RAISE) return at + 1u;
    if (move == POKER_FOLD) return POKER_OVER;
    if (id == 0u) return at + 3u;
    return k->rounds == 1u || at >= POKER_IDS ? POKER_OVER : poker_root(id, board);
}

static inline __attribute__((always_inline)) uint32_t poker_bet(const pokerKnobs *k, uint32_t table, uint32_t move, uint32_t mover, uint32_t next) {
    uint32_t own = mover ? POKER_THEIR_CHIPS : POKER_MINE_CHIPS, other = table >> (mover ? POKER_MINE_CHIPS : POKER_THEIR_CHIPS) & 15u;
    uint32_t chips = move == POKER_RAISE ? poker_raised(k, table >> POKER_AT & 127u, other) : other;
    table = (table & ~(15u << own | 127u << POKER_AT)) | chips << own | next << POKER_AT;
    return mover ? (table & ~(3u << POKER_SEEN)) | move << POKER_SEEN : table;
}

static inline __attribute__((always_inline)) float poker_payoff(const pokerKnobs *k, uint32_t table, uint32_t move, uint32_t mover) {
    uint32_t mine = table >> POKER_MINE_CHIPS & 15u, theirs = table >> POKER_THEIR_CHIPS & 15u;
    if (move == POKER_FOLD) return mover ? (float)theirs : -(float)mine;
    uint32_t own = poker_rank(k, table & 7u), other = poker_rank(k, table >> POKER_THEIRS & 7u), board = poker_rank(k, table >> POKER_BOARD & 7u);
    if (k->rounds > 1u) {
        own += 3u * (own == board);
        other += 3u * (other == board);
    }
    float stake = (float)(mine > theirs ? mine : theirs);
    return own > other ? stake : own < other ? -stake : 0.0f;
}

static __attribute__((noinline)) void poker_refill(uint32_t seed, uint32_t hand, uint32_t *batch) {
    u44 r = philox4(seed, ENV_RNG_STREAM, hand + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u, (lane4)0u);
    __builtin_memcpy(batch, &r, sizeof r);
}

static inline __attribute__((always_inline)) uint32_t poker_deal(const pokerKnobs *k, uint32_t seed, uint32_t hand, uint32_t *extra) {
    uint32_t slot = batchSlot(&extra[POKER_TAG], hand + 1u);
    if (slot == ENV_BATCH) {
        poker_refill(seed, hand, extra + POKER_BATCH);
        slot = 0u;
    }
    const uint32_t *batch = extra + POKER_BATCH + slot;
    uint32_t x = batch[0], y = batch[ENV_BATCH], z = batch[2 * ENV_BATCH], w = batch[3 * ENV_BATCH];
    extra[0] = y;
    extra[1] = z;
    extra[2] = w;
    extra[3] = y << 24 | (z & 255u) << 16 | (w & 255u) << 8;
    uint32_t deck = 3u * k->suits, pick = (uint32_t)(((uint64_t)x * (deck * (deck - 1u) * (k->rounds > 1u ? deck - 2u : 1u))) >> 32);
    uint32_t first = pick % deck, second = pick / deck % (deck - 1u), board = pick / deck / (deck - 1u);
    second += second >= first;
    uint32_t low = first < second ? first : second, high = first ^ second ^ low;
    board += board >= low;
    board += board >= high;
    uint32_t mine = hand & 1u ? second : first;
    return mine | (first ^ second ^ mine) << POKER_THEIRS | (k->rounds > 1u ? board : POKER_NONE) << POKER_BOARD | k->ante << POKER_MINE_CHIPS | k->ante << POKER_THEIR_CHIPS;
}

static inline __attribute__((always_inline)) uint32_t poker_opponent(const pokerKnobs *k, uint32_t table, const uint32_t *draws, const float *counts, const uint32_t *plan) {
    uint32_t at = table >> POKER_AT & 127u, id = at % POKER_IDS, draw = draws[2u * (k->rounds > 1u && at >= POKER_IDS) + (id / 3u + id % 3u) / 2u];
    uint32_t infoset = 3u * at + poker_rank(k, table >> POKER_THEIRS & 7u), move;
    if (k->opponent == POKER_UNIFORM) move = draw < u01Bound(0.5f) ? POKER_FOLD : POKER_RAISE;
    else if (k->opponent == POKER_CFR) {
        float drawn = (float)(draw >> 8);
        move = (uint32_t)(drawn >= counts[2u * infoset]) + (uint32_t)(drawn >= counts[2u * infoset + 1u]);
    } else move = poker_planned(plan, infoset);
    return poker_legal(k, id, move);
}

static inline __attribute__((always_inline)) void poker_obs(const pokerKnobs *k, const uint32_t *words, float *obs) {
    uint32_t table = words[POKER_TABLE], last = words[POKER_LAST], at = table >> POKER_AT & 127u, raises = at % POKER_IDS % 3u;
    uint32_t round = k->rounds > 1u && at >= POKER_IDS, mine = poker_rank(k, table & 7u), board = round ? poker_rank(k, table >> POKER_BOARD & 7u) : 3u;
    uint32_t seen = table >> POKER_SEEN & 3u, first = (words[POKER_HAND] & 1u) == 0u;
    uint32_t shown = (last >> POKER_ENDED & 1u) && (last >> POKER_MOVE & 3u) == POKER_CALL ? poker_rank(k, last >> POKER_THEIRS & 7u) : 3u;
    float4 cards = flag4((lane4)((lane4){ mine, mine, mine, board } == (lane4){ 0u, 1u, 2u, 0u }));
    float4 seats = flag4((lane4)((lane4){ board, board, first, round } == (lane4){ 1u, 2u, 1u, 1u }));
    float4 stakes = (float4){ (float)(table >> POKER_MINE_CHIPS & 15u), (float)(table >> POKER_THEIR_CHIPS & 15u), (float)raises, (float)(seen == POKER_CALL) } /
                    (float4){ (float)POKER_POT, (float)POKER_POT, (float)POKER_BETS, 1.0f };
    float4 memory = flag4((lane4)((lane4){ seen, shown, shown, shown } == (lane4){ POKER_RAISE, 0u, 1u, 2u }));
    __builtin_memcpy(obs, &cards, sizeof cards);
    __builtin_memcpy(obs + 4, &seats, sizeof seats);
    __builtin_memcpy(obs + 8, &stakes, sizeof stakes);
    __builtin_memcpy(obs + 12, &memory, sizeof memory);
}

static void poker_split(const float4 *rows, float4 *moves) {
    float4 low = __builtin_shufflevector(rows[0], rows[1], 0, 4, 1, 5), high = __builtin_shufflevector(rows[0], rows[1], 2, 6, 3, 7);
    moves[POKER_FOLD] = __builtin_shufflevector(low, rows[2], 0, 1, 4, 7);
    moves[POKER_CALL] = __builtin_shufflevector(low, rows[2], 2, 3, 5, 7);
    moves[POKER_RAISE] = __builtin_shufflevector(high, rows[2], 0, 1, 6, 7);
}

static float4 poker_times(const float4 *columns, float4 reach) { return columns[0] * reach.x + columns[1] * reach.y + columns[2] * reach.z; }

static float4 poker_sum(const float4 *child, uint32_t legal) {
    float4 value = child[POKER_CALL];
    if (legal & 1u) value += child[POKER_FOLD];
    if (legal >> POKER_RAISE & 1u) value += child[POKER_RAISE];
    return value;
}

static void poker_walk(const pokerKnobs *k, pokerRules *rules, uint32_t at, uint32_t actor, uint32_t other) {
    uint32_t id = at % POKER_IDS;
    rules->folds[at] = (float)actor;
    rules->shows[at] = (float)other;
    if (id % 3u < k->raiseCap) poker_walk(k, rules, at + 1u, other, poker_raised(k, at, other));
    if (id == 0u) poker_walk(k, rules, at + 3u, other, other);
    else if (at < POKER_IDS && k->rounds > 1u)
        for (uint32_t board = 0u; board < POKER_RANKS; board++) poker_walk(k, rules, poker_root(id, board), other, other);
}

static void poker_rules(const pokerKnobs *k, pokerRules *rules) {
    float suits = (float)k->suits, deck = 3.0f * suits;
    for (uint32_t board = k->rounds > 1u ? 0u : 3u; board < 4u; board++)
        for (uint32_t other = 0u; other < POKER_RANKS; other++) {
            float4 chance = 0.0f, showdown = 0.0f;
            for (uint32_t own = 0u; own < POKER_RANKS; own++) {
                float weight = (suits - (float)(own == other)) / (3.0f * (deck - 1.0f));
                if (board < 3u) weight *= (suits - (float)(own == board) - (float)(other == board)) / (deck - 2.0f);
                uint32_t mine = own + 3u * (own == board), theirs = other + 3u * (other == board);
                chance[own] = weight;
                showdown[own] = mine > theirs ? weight : mine < theirs ? -weight : 0.0f;
            }
            rules->chance[board][other] = chance;
            rules->showdown[board][other] = showdown;
        }
    poker_walk(k, rules, 0u, k->ante, k->ante);
}

static float4 poker_even(const pokerKnobs *k, uint32_t infoset) {
    uint32_t legal = poker_moves(k, infoset / 3u);
    float4 row = { (float)(legal & 1u), 1.0f, (float)(legal >> POKER_RAISE & 1u), 0.0f };
    return row / (row.x + row.y + row.z);
}

static void poker_uniform(const pokerKnobs *k, float4 *policy) {
    for (uint32_t i = 0u; i < 3u * poker_nodes(k); i++) policy[i] = poker_even(k, i);
}

static void poker_down(const pokerKnobs *k, pokerTree *t, const float4 *policy, uint32_t side, uint32_t at) {
    uint32_t id = at % POKER_IDS;
    if (at == 0u) t->reach[side][0] = (float4){ 1.0f, 1.0f, 1.0f, 0.0f };
    float4 call = t->reach[side][at], raise = call;
    if ((id & 1u) != side) {
        float4 moves[3];
        poker_split(policy + 3u * at, moves);
        call *= moves[POKER_CALL];
        raise *= moves[POKER_RAISE];
    }
    if (id == 0u) t->reach[side][at + 3u] = call;
    else if (at < POKER_IDS && k->rounds > 1u)
        for (uint32_t board = 0u; board < POKER_RANKS; board++) t->reach[side][poker_root(id, board)] = call;
    if (id % 3u < k->raiseCap) t->reach[side][at + 1u] = raise;
}

static void poker_children(const pokerKnobs *k, const pokerRules *rules, const pokerTree *t, const float4 *policy, uint32_t side, uint32_t at, float4 *child) {
    uint32_t id = at % POKER_IDS, raises = id % 3u, block = at / POKER_IDS, board = block ? (block - 1u) % 3u : 3u;
    float4 fold = t->reach[side][at], call = fold;
    if ((id & 1u) != side) {
        float4 moves[3];
        poker_split(policy + 3u * at, moves);
        fold *= moves[POKER_FOLD];
        call *= moves[POKER_CALL];
    }
    if (raises > 0u) child[POKER_FOLD] = poker_times(rules->chance[board], fold) * ((id & 1u) == side ? -rules->folds[at] : rules->folds[at]);
    if (id == 0u) child[POKER_CALL] = t->value[side][at + 3u];
    else if (block || k->rounds == 1u) child[POKER_CALL] = poker_times(rules->showdown[board], call) * rules->shows[at];
    else child[POKER_CALL] = t->value[side][poker_root(id, 0u)] + t->value[side][poker_root(id, 1u)] + t->value[side][poker_root(id, 2u)];
    if (raises < k->raiseCap) child[POKER_RAISE] = t->value[side][at + 1u];
}

static void poker_best(const pokerKnobs *k, const pokerRules *rules, pokerTree *t, const float4 *policy, uint32_t side, uint32_t at, uint32_t *plan) {
    float4 child[3];
    uint32_t legal = poker_moves(k, at);
    poker_children(k, rules, t, policy, side, at, child);
    if ((at % POKER_IDS & 1u) != side) {
        t->value[side][at] = poker_sum(child, legal);
        return;
    }
    float4 value = child[POKER_CALL];
    lane4 best = POKER_CALL;
    if (legal >> POKER_RAISE & 1u) {
        lane4 more = (lane4)(child[POKER_RAISE] > value);
        value = select4(more, child[POKER_RAISE], value);
        best = (best & ~more) | (POKER_RAISE & more);
    }
    if (legal & 1u) {
        lane4 more = (lane4)(child[POKER_FOLD] > value);
        value = select4(more, child[POKER_FOLD], value);
        best &= ~more;
    }
    t->value[side][at] = value;
    for (uint32_t rank = 0u; rank < POKER_RANKS; rank++) poker_plan(plan, 3u * at + rank, best[rank]);
}

static void poker_item(const pokerKnobs *k, const pokerRules *rules, pokerTree *t, const float4 *policy, uint32_t *plan, uint32_t item) {
    uint32_t nodes = poker_nodes(k), at = item < nodes ? item : 2u * nodes - 1u - item;
    if (at % POKER_IDS % 3u > k->raiseCap) return;
    for (uint32_t side = 0u; side < 2u; side++) {
        if (item < nodes) poker_down(k, t, policy, side, at);
        else poker_best(k, rules, t, policy, side, at, plan);
    }
}

static void poker_respond(const pokerKnobs *k, const pokerRules *rules, pokerTree *t, const float4 *policy, uint32_t *plan) {
    for (uint32_t item = 0u; item < 2u * poker_nodes(k); item++) poker_item(k, rules, t, policy, plan, item);
}

static void poker_match(const pokerKnobs *k, const float4 *regret, float4 *policy) {
    for (uint32_t i = 0u; i < 3u * poker_nodes(k); i++) {
        float total = regret[i].x + regret[i].y + regret[i].z;
        policy[i] = total > 0.0f ? regret[i] / total : poker_even(k, i);
    }
}

static void poker_regret(const pokerKnobs *k, const pokerRules *rules, pokerTree *t, const float4 *policy, float4 *regret, uint32_t side, uint32_t at) {
    float4 child[3], moves[3], value = 0.0f;
    uint32_t legal = poker_moves(k, at);
    poker_children(k, rules, t, policy, side, at, child);
    if ((at % POKER_IDS & 1u) != side) {
        t->value[side][at] = poker_sum(child, legal);
        return;
    }
    poker_split(policy + 3u * at, moves);
    for (uint32_t move = 0u; move < 3u; move++)
        if (legal >> move & 1u) value += moves[move] * child[move];
    t->value[side][at] = value;
    for (uint32_t rank = 0u; rank < POKER_RANKS; rank++)
        for (uint32_t move = 0u; move < 3u; move++)
            if (legal >> move & 1u) regret[3u * at + rank][move] = fmaxf(regret[3u * at + rank][move] + child[move][rank] - value[rank], 0.0f);
}

static uint32_t poker_count(float p) { return p * 0x1p24f > 0x1p24f - 1.0f ? (uint32_t)POKER_WHOLE : u01Bound(p) >> 8; }

static void poker_solve(const pokerKnobs *k, const pokerRules *rules, pokerTree *t, float4 *policy, float *counts) {
    float4 regret[POKER_INFOSETS], average[POKER_INFOSETS];
    uint32_t nodes = poker_nodes(k);
    __builtin_memset(regret, 0, sizeof regret);
    __builtin_memset(average, 0, sizeof average);
    for (uint32_t iteration = 1u; iteration <= k->iterations; iteration++)
        for (uint32_t side = 0u; side < 2u; side++) {
            poker_match(k, regret, policy);
            for (uint32_t at = 0u; at < nodes; at++) {
                if (at % POKER_IDS % 3u > k->raiseCap) continue;
                poker_down(k, t, policy, side, at);
                if ((at % POKER_IDS & 1u) != side)
                    for (uint32_t rank = 0u; rank < POKER_RANKS; rank++) average[3u * at + rank] += (float)iteration * t->reach[side][at][rank] * policy[3u * at + rank];
            }
            for (uint32_t at = nodes; at-- > 0u;)
                if (at % POKER_IDS % 3u <= k->raiseCap) poker_regret(k, rules, t, policy, regret, side, at);
        }
    for (uint32_t i = 0u; i < 3u * nodes; i++) {
        uint32_t legal = poker_moves(k, i / 3u);
        float4 mean = average[i] / (average[i].x + average[i].y + average[i].z);
        uint32_t fold = legal & 1u ? poker_count(mean.x) : 0u, stay = legal >> POKER_RAISE & 1u ? poker_count(mean.x + mean.y) : (uint32_t)POKER_WHOLE;
        counts[2u * i] = (float)fold;
        counts[2u * i + 1u] = (float)stay;
        policy[i] = (float4){ (float)fold, (float)(stay - fold), (float)((uint32_t)POKER_WHOLE - stay), 0.0f } * 0x1p-24f;
    }
}

static void poker_record(const pokerKnobs *k, uint32_t instance, const pokerTree *t, const uint32_t *plan, float *table) {
    for (uint32_t i = 0u; i < 3u * poker_nodes(k); i++) table[POKER_ORACLE + POKER_INFOSETS * instance + i] = (float)poker_answer(k, i / 3u % POKER_IDS, poker_planned(plan, i));
    for (uint32_t side = 0u; side < 2u; side++) table[POKER_VALUES + 2u * instance + side] = t->value[side][0].x + t->value[side][0].y + t->value[side][0].z;
}

uint64_t poker_shared(float *table) {
    if (table) {
        pokerRules rules;
        pokerTree tree;
        float4 policy[POKER_INFOSETS];
        uint32_t plan[POKER_PLAN] = { 0u };
        const pokerKnobs *kuhn = &POKER_KNOBS[0], *leduc = &POKER_KNOBS[1];
        poker_rules(kuhn, &rules);
        poker_uniform(kuhn, policy);
        poker_respond(kuhn, &rules, &tree, policy, plan);
        poker_record(kuhn, 0u, &tree, plan, table);
        poker_rules(leduc, &rules);
        poker_solve(leduc, &rules, &tree, policy, table + POKER_COUNTS);
        poker_respond(leduc, &rules, &tree, policy, plan);
        poker_record(leduc, 1u, &tree, plan, table);
    }
    return POKER_SHARED;
}

static inline __attribute__((always_inline)) void poker_learn(const pokerKnobs *k, pokerLearner *l, uint32_t table, uint32_t move) {
    float4 *row = &l->ema[3u * (table >> POKER_AT & 127u) + poker_rank(k, table & 7u)];
    *row += k->decay * (flag4((lane4)((lane4)move == (lane4){ 0u, 1u, 2u, 3u })) - *row);
}

static void poker_adopt(pokerLearner *l, uint32_t *words, uint32_t step) {
    words[POKER_VERSION]++;
    __builtin_memcpy(l->snap, l->ema, sizeof l->snap);
    l->start = step;
    l->items = 0u;
}

static inline __attribute__((always_inline)) void poker_think(const pokerKnobs *k, pokerLearner *l, const uint32_t *words, uint32_t step) {
    uint32_t item = l->items;
    if (item >= POKER_ITEMS || item >= (step - l->start + 1u) * POKER_ITEMS / k->sweepSteps) return;
    poker_item(k, &l->rules, &l->tree, l->snap, l->plan[(words[POKER_VERSION] + 1u) & 1u], item);
    l->items = item + 1u;
}

static inline __attribute__((always_inline)) uint32_t poker_begin(const pokerKnobs *k, uint32_t seed, uint32_t *words, uint32_t *extra, const float *counts, pokerLearner *l, uint32_t step) {
    uint32_t hand = words[POKER_HAND], table = poker_deal(k, seed, hand, extra);
    if (l && l->items == POKER_ITEMS) poker_adopt(l, words, step);
    if (hand & 1u) {
        uint32_t move = poker_opponent(k, table, extra, counts, l ? l->plan[words[POKER_VERSION] & 1u] : 0);
        table = poker_bet(k, table, move, 1u, poker_next(k, 0u, move, 0u));
    }
    return table;
}

void poker_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const pokerKnobs *k = &POKER_KNOBS[instance];
    pokerLearner *l = k->opponent == POKER_FICTITIOUS ? poker_learner(extra) : 0;
    words[POKER_HAND] = 0u;
    words[POKER_LAST] = 0u;
    batchReset(&extra[POKER_TAG]);
    if (l) {
        words[POKER_VERSION] = 0u;
        poker_rules(k, &l->rules);
        poker_uniform(k, l->ema);
        __builtin_memcpy(l->snap, l->ema, sizeof l->snap);
        poker_respond(k, &l->rules, &l->tree, l->snap, l->plan[1]);
        l->items = POKER_ITEMS;
        l->studied = 0u;
    }
    words[POKER_TABLE] = poker_begin(k, seed, words, extra, 0, l, 0u);
}

static inline __attribute__((always_inline)) float poker_play(const pokerKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    pokerLearner *l = k->opponent == POKER_FICTITIOUS ? poker_learner(extra) : 0;
    const float *counts = shared + POKER_COUNTS;
    uint32_t table = words[POKER_TABLE], seat = words[POKER_HAND] & 1u, mover = 0u, next;
    uint32_t move = poker_legal(k, (table >> POKER_AT & 127u) % POKER_IDS, poker_choice(k, ctrl));
    if (l) poker_learn(k, l, table, move);
    while ((next = poker_next(k, table >> POKER_AT & 127u, move, poker_rank(k, table >> POKER_BOARD & 7u))) != POKER_OVER) {
        table = poker_bet(k, table, move, mover, next);
        if ((next % POKER_IDS & 1u) == seat) break;
        move = poker_opponent(k, table, extra, counts, l ? l->plan[words[POKER_VERSION] & 1u] : 0);
        mover = 1u;
    }
    float reward = 0.0f;
    *done = next == POKER_OVER;
    if (next == POKER_OVER) {
        reward = poker_payoff(k, table, move, mover);
        words[POKER_LAST] = (table & ((1u << POKER_MOVE) - 1u)) | move << POKER_MOVE | 1u << POKER_ENDED;
        words[POKER_HAND]++;
        table = poker_begin(k, seed, words, extra, counts, l, step);
    }
    words[POKER_TABLE] = table;
    if (l) poker_think(k, l, words, step);
    poker_obs(k, words, obs);
    return reward;
}

float poker_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    switch (instance) {
        case 0u: return poker_play(&POKER_KNOBS[0], seed, step, ctrl, words, extra, obs, done, shared);
        case 1u: return poker_play(&POKER_KNOBS[1], seed, step, ctrl, words, extra, obs, done, shared);
        case 2u: return poker_play(&POKER_KNOBS[2], seed, step, ctrl, words, extra, obs, done, shared);
        default: __builtin_trap();
    }
}

void poker_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step; (void)shared;
    poker_obs(&POKER_KNOBS[instance], words, obs);
}

static __attribute__((noinline)) void poker_study(const pokerKnobs *k, pokerLearner *l, uint32_t version) {
    float4 policy[POKER_INFOSETS];
    pokerTree tree;
    for (uint32_t i = 0u; i < POKER_INFOSETS; i++) policy[i] = flag4((lane4)((lane4)poker_planned(l->plan[version & 1u], i) == (lane4){ 0u, 1u, 2u, 3u }));
    poker_respond(k, &l->rules, &tree, policy, l->oracle);
    l->studied = version;
}

void poker_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)obs;
    const pokerKnobs *k = &POKER_KNOBS[instance];
    uint32_t table = words[POKER_TABLE], at = table >> POKER_AT & 127u, infoset = 3u * at + poker_rank(k, table & 7u);
    if (k->opponent != POKER_FICTITIOUS) {
        ctrl[0] = shared[POKER_ORACLE + POKER_INFOSETS * instance + infoset];
        return;
    }
    pokerLearner *l = poker_learner(extra);
    if (l->studied != words[POKER_VERSION]) poker_study(k, l, words[POKER_VERSION]);
    ctrl[0] = (float)poker_answer(k, at % POKER_IDS, poker_planned(l->oracle, infoset));
}
