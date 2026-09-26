import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNITS = ROOT / '.build/catalog'
SCALARS = {'int8_t', 'uint8_t', 'int16_t', 'uint16_t', 'int32_t', 'uint32_t', 'int', 'float', 'double'}
HOOKS = ('step', 'reset', 'oracle', 'observe', 'extraWords', 'shared', 'view')
NUMBER = re.compile(r'^-?\d+(?:\.\d+)?(?:[eE]-?\d+)?[uUfF]?$')


def literal(text):
    return json.dumps(text)


def emit(text):
    return f'fputs({literal(text)}, out);'


def scalar(expression):
    return f'fprintf(out, "%.9g", (double)({expression}));'


def flat(expression, base):
    return (emit('[') + f' for (uint32_t j = 0u; j < (uint32_t)(sizeof {expression} / sizeof({base})); j++) '
            f'fprintf(out, "%s%.9g", j ? "," : "", (double)((const {base} *)(const void *)&{expression})[j]); ' + emit(']'))


def declarations(body):
    for declaration in body.split(';'):
        declaration = ' '.join(declaration.split())
        if not declaration:
            continue
        kind, names = declaration.split(' ', 1)
        for name in names.split(','):
            name = name.strip()
            yield kind, name.split('[')[0], '[' in name


def value(expression, kind, array, aliases):
    base = aliases.get(kind, kind)
    return flat(expression, base) if array else scalar(expression)


def structs(source, aliases):
    found = {}
    for body, name in re.findall(r'typedef struct \{([^}]*)\}\s*(\w+);', source):
        fields = list(declarations(body))
        if all(aliases.get(kind, kind) in SCALARS for kind, _, _ in fields):
            found[name] = fields
    return found


def exports(source, aliases, shapes, knobs):
    found = []
    for kind, name, dims in re.findall(r'^static const (\w+) (\w+)((?:\[[^\]]*\])*) = ', source, re.M):
        base = aliases.get(kind, kind)
        if base in SCALARS:
            found.append((name, [value(name, kind, bool(dims), aliases)]))
        elif kind in shapes and kind != knobs:
            def fields(target):
                out = []
                for i, (field, member, array) in enumerate(shapes[kind]):
                    out += [emit((',' if i else '') + literal(member) + ':'), value(f'{target}.{member}', field, array, aliases)]
                return out
            if dims:
                found.append((name, [emit('['), f'for (uint32_t i = 0u; i < (uint32_t)(sizeof {name} / sizeof {name}[0]); i++) {{',
                                      'if (i) fputc(\',\', out);', emit('{'), *fields(f'{name}[i]'), emit('}'), '}', emit(']')]))
            else:
                found.append((name, [emit('{'), *fields(name), emit('}')]))
    return found


def layout(family, source):
    if family == 'hall':
        return [emit(',"grid":'), 'catalogWalls(out, &HALL_WALLS);']
    if family == 'maze':
        return ['if (k->layout != 0u) {', 'int side = (int)maze_side(k);', emit(',"grid":'),
                'fprintf(out, "{\\"w\\":%d,\\"h\\":%d,\\"walls\\":[", side + 2, side + 2);',
                'for (int y = 0; y < side + 2; y++) for (int x = 0; x < side + 2; x++) fprintf(out, "%s%d", x || y ? "," : "", maze_wall(k, 0, x, y));',
                emit(']}'), '}']
    if family == 'corridor':
        return ['int width = 4 * (int)(k->doors + 1u) + 1, height = 7;', emit(',"grid":'),
                'fprintf(out, "{\\"w\\":%d,\\"h\\":%d,\\"walls\\":[", width, height);',
                'for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) fprintf(out, "%s%d", x || y ? "," : "", corridor_wall(k, x, y));',
                emit(']}')]
    if family == 'taxi':
        return [emit(',"fences":['), 'int first = 1;',
                'for (int y = 0; y < (int)TAXI_SIDE; y++) for (int x = 0; x < (int)TAXI_SIDE; x++) for (uint32_t a = 0u; a < 4u; a++) {',
                'int dx, dy; gridDelta(a, &dx, &dy); int nx = x + dx, ny = y + dy;',
                'if (nx < 0 || ny < 0 || nx >= (int)TAXI_SIDE || ny >= (int)TAXI_SIDE || !taxi_blocked(x, y, nx, ny)) continue;',
                'fprintf(out, "%s[%d,%d,%d,%d]", first ? "" : ",", x, y, nx, ny); first = 0;', '}', emit(']')]
    if family == 'cliff':
        return ['uint8_t mark[CLIFF_WIDTH * CLIFF_HEIGHT] = { 0 };',
                'for (uint32_t y = 0u; y < CLIFF_HEIGHT; y++) for (uint32_t x = 0u; x < CLIFF_WIDTH; x++) for (uint32_t a = 0u; a < 4u; a++) {',
                'int dx, dy; gridDelta(a, &dx, &dy); int tx = (int)x + dx, ty = (int)y - dy;',
                'tx = tx < 0 ? 0 : tx >= (int)CLIFF_WIDTH ? (int)CLIFF_WIDTH - 1 : tx; ty = ty < 0 ? 0 : ty >= (int)CLIFF_HEIGHT ? (int)CLIFF_HEIGHT - 1 : ty;',
                'uint8_t code = cliff_move(x, y, a); if (code == CLIFF_FELL) mark[(uint32_t)ty * CLIFF_WIDTH + (uint32_t)tx] = 1u; if (code == CLIFF_REACHED) mark[(uint32_t)ty * CLIFF_WIDTH + (uint32_t)tx] = 2u; }',
                emit(',"hazards":['), 'int first = 1;',
                'for (uint32_t c = 0u; c < CLIFF_WIDTH * CLIFF_HEIGHT; c++) if (mark[c] == 1u) { fprintf(out, "%s[%u,%u]", first ? "" : ",", c % CLIFF_WIDTH, c / CLIFF_WIDTH); first = 0; }',
                emit(']'),
                'for (uint32_t c = 0u; c < CLIFF_WIDTH * CLIFF_HEIGHT; c++) if (mark[c] == 2u) fprintf(out, ",\\"goal\\":[%u,%u]", c % CLIFF_WIDTH, c / CLIFF_WIDTH);']
    if family == 'blocking':
        return ['for (int y = 0; y < (int)BLOCKING_HEIGHT; y++) for (int x = 0; x < (int)BLOCKING_WIDTH; x++) for (uint32_t a = 0u; a < 4u; a++) {',
                'int dx, dy; gridDelta(a, &dx, &dy);',
                'if (!gridWall(&k->phase[0], x + dx, y + dy) && blocking_move(&k->phase[0], x, y, a) / BLOCKING_GOAL) { fprintf(out, ",\\"goal\\":[%d,%d]", x + dx, y + dy); goto placed; } }',
                'placed:;']
    if family == 'jbw':
        decay = re.search(r'response\[current \+ i\] = ([\d.]+f)', source)[1]
        diffusion = re.search(r'\+= ([\d.]+f) \* \(', source)[1]
        return [emit(',"scent":{'), f'fprintf(out, "\\"decay\\":%.9g,\\"diffusion\\":%.9g,\\"lifetime\\":%u,\\"values\\":[", (double){decay}, (double){diffusion}, JBW_DESTRUCTION_LIFETIME);',
                'for (uint32_t item = JBW_EMPTY; item <= JBW_TRUFFLE; item++) { float s[3]; jbw_scent(item, s); fprintf(out, "%s[%.9g,%.9g,%.9g]", item ? "," : "", (double)s[0], (double)s[1], (double)s[2]); }', emit(']}')]
    return []


def unit(path, family, source):
    aliases = dict((alias, kind) for kind, alias in re.findall(r'^typedef (\w+) (\w+);', source, re.M) if kind in SCALARS)
    knobs = re.search(r'typedef struct \{([^}]+)\}\s*(\w+Knobs);', source)
    lines = ['#pragma clang diagnostic ignored "-Wunused-function"', '#pragma clang diagnostic ignored "-Wunused-const-variable"', '#include <stdio.h>']
    lines += [f'#define {family}_{hook} catalog_{family}_{hook}' for hook in HOOKS]
    lines += [f'#include "../../Sources/envs/{path.name}"']
    lines += [f'#undef {family}_{hook}' for hook in HOOKS]
    lines.append('''static void catalogWalls(FILE *out, const GridWalls *grid) {
    fprintf(out, "{\\"w\\":%d,\\"h\\":%d,\\"walls\\":[", grid->width, grid->height);
    for (int y = 0; y < grid->height; y++) for (int x = 0; x < grid->width; x++)
        fprintf(out, "%s%d", x || y ? "," : "", gridWall(grid, x, y));
    fputs("]}", out);
}''')
    body = [emit(',"parameters":{')]
    if knobs:
        table = re.search(rf'static const {knobs[2]} (\w+)\[\d+\] = \{{', source)
        assert table, path
        body.insert(0, f'const {knobs[2]} *k = &{table[1]}[instance];')
        for i, (kind, name, array) in enumerate(declarations(knobs[1])):
            body.append(emit((',' if i else '') + literal(name) + ':'))
            if kind == 'GridWalls':
                count = int(re.search(rf'{name}\[(\d+)\]', knobs[1])[1])
                body += [emit('['), f'for (int j = 0; j < {count}; j++) {{ if (j) fputc(\',\', out); catalogWalls(out, &k->{name}[j]); }}', emit(']')]
            else:
                assert kind in ('int', 'uint32_t', 'float') and not array, kind
                body.append(scalar(f'k->{name}'))
    else:
        body.insert(0, '(void)instance;')
    if family == 'jbw':
        body += [emit(',"polarityPeriod":'), f'fprintf(out, "%u", {re.search(r"step / (\d+u)", source)[1]});']
    body.append(emit('}'))
    constants = [(key, [scalar(key)]) for key, text in re.findall(r'^#define (\w+) (.+)$', source, re.M) if NUMBER.match(text.strip())]
    constants += exports(source, aliases, structs(source, aliases), knobs[2] if knobs else None)
    body.append(emit(',"constants":{'))
    for i, (name, emitted) in enumerate(constants):
        body += [emit((',' if i else '') + literal(name) + ':'), *emitted]
    body.append(emit('}'))
    body += layout(family, source)
    lines += [f'void catalog_{family}(FILE *out, uint32_t instance) {{', *body, '}']
    return '\n'.join(lines) + '\n'


def generate():
    families = re.findall(r'^    F\((\w+), \w+, \w+, \w+\)', (ROOT / 'Sources/env.c').read_text(), re.M)
    shutil.rmtree(UNITS, ignore_errors=True)
    UNITS.mkdir(parents=True)
    owners = {}
    for path in sorted((ROOT / 'Sources/envs').glob('*.c')):
        source = path.read_text()
        family = re.search(r'void (\w+)_reset\(', source).group(1)
        assert family in families and family not in owners, family
        owners[family] = path
        (UNITS / f'{family}.c').write_text(unit(path, family, source))
    assert set(owners) == set(families), set(families) ^ set(owners)
    declarations = ''.join(f'void catalog_{family}(FILE *out, uint32_t instance);\n' for family in families)
    cases = ''.join(f'    if (strcmp(env->family, "{family}") == 0) {{ catalog_{family}(out, env->instance); return; }}\n' for family in families)
    return declarations + 'static void catalogDetails(FILE *out, const EnvSpec *env) {\n' + cases + '    abort();\n}\n'


if __name__ == '__main__':
    (ROOT / '.build/catalog.h').write_text(generate())
