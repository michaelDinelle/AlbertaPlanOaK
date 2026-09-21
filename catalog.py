import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def literal(text):
    return json.dumps(text)


def emit(text):
    return f'fputs({literal(text)}, out);'


def generate():
    blocks = []
    families = set(re.findall(r'^    F\((\w+)\)', (ROOT / 'Sources/env.c').read_text(), re.M))
    for path in sorted((ROOT / 'Sources/envs').glob('*.c')):
        source = path.read_text()
        family = re.search(r'void (\w+)_reset\(', source).group(1)
        assert family in families, family
        families.remove(family)
        lines = [f'if (strcmp(env->family, "{family}") == 0) {{']
        defines = re.findall(r'^#define (\w+) (.+)$', source, re.M)
        lines += [f'#define {key} {value}' for key, value in defines]
        knobs = re.search(r'typedef struct \{([^}]+)\}\s*(\w+Knobs);', source)
        lines.append(emit(',"parameters":{'))
        if knobs:
            table = re.search(rf'static const {knobs[2]} (\w+)\[\d+\] = \{{.*?\n\}};', source, re.S)
            assert table, path
            lines += [knobs[0], table[0], f'const {knobs[2]} *k = &{table[1]}[env->instance];']
            fields = []
            for declaration in knobs[1].split(';'):
                if not declaration.strip():
                    continue
                kind, names = declaration.strip().split(' ', 1)
                for field in names.split(','):
                    fields.append((kind, field.strip()))
            for i, (kind, field) in enumerate(fields):
                name = field.split('[')[0]
                lines.append(emit((',' if i else '') + literal(name) + ':'))
                if kind == 'GridWalls':
                    count = int(re.search(r'\[(\d+)\]', field)[1])
                    lines += [emit('['), f'for (int j = 0; j < {count}; j++) {{ if (j) fputc(\',\', out); catalogWalls(out, &k->{name}[j]); }}', emit(']')]
                else:
                    assert kind in ('int', 'uint32_t', 'float'), kind
                    lines.append(f'fprintf(out, "%.9g", (double)k->{name});')
        if family == 'jbw':
            period = re.search(r'step / (\d+u)', source)[1]
            lines += [emit(',\"polarityPeriod\":'), f'fprintf(out, \"%u\", {period});']
        lines.append(emit('}'))
        if family == 'hall':
            walls = re.search(r'static const GridWalls HALL_WALLS = .*?;', source)[0]
            lines += [walls, emit(',"grid":'), 'catalogWalls(out, &HALL_WALLS);']
        if family == 'jbw':
            scent = re.search(r'static void jbw_scent\(.*?\n\}', source, re.S)[0]
            blocks.append('\n'.join(f'#define {key} {value}' for key, value in defines) + '\n' + scent + '\n' + '\n'.join(f'#undef {key}' for key, _ in defines))
            decay = re.search(r'response\[current \+ i\] = ([\d.]+f)', source)[1]
            diffusion = re.search(r'\+= ([\d.]+f) \* \(', source)[1]
            lines += [emit(',"scent":{'), f'fprintf(out, "\\\"decay\\\":%.9g,\\\"diffusion\\\":%.9g,\\\"lifetime\\\":%u,\\\"values\\\":[", (double){decay}, (double){diffusion}, JBW_DESTRUCTION_LIFETIME);',
                      'for (uint32_t item = JBW_EMPTY; item <= JBW_TRUFFLE; item++) { float s[3]; jbw_scent(item, s); fprintf(out, "%s[%.9g,%.9g,%.9g]", item ? "," : "", (double)s[0], (double)s[1], (double)s[2]); }', emit(']}')]
        lines += [f'#undef {key}' for key, _ in defines]
        lines += ['return;', '}']
        blocks.append('\n'.join(lines))
    assert not families, families
    helpers = [block for block in blocks if not block.startswith('if ')]
    cases = [block for block in blocks if block.startswith('if ')]
    return '\n'.join(helpers) + '''
static void catalogWalls(FILE *out, const GridWalls *grid) {
    fprintf(out, "{\\\"w\\\":%d,\\\"h\\\":%d,\\\"walls\\\":[", grid->width, grid->height);
    for (int y = 0; y < grid->height; y++) for (int x = 0; x < grid->width; x++)
        fprintf(out, "%s%d", x || y ? "," : "", gridWall(grid, x, y));
    fputs("]}", out);
}
static void catalogDetails(FILE *out, const EnvSpec *env) {
''' + '\n'.join(cases) + '\nabort();\n}\n'


if __name__ == '__main__':
    (ROOT / '.build/catalog.h').write_text(generate())
