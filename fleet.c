#define _DARWIN_C_SOURCE
#include <CommonCrypto/CommonDigest.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/clonefile.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
extern char **environ;

#define OAK_ANCHOR "4262868f4bce5ea25ac4905e60742f75d11be371"
enum { OAK_STEPS = 1000000u, OAK_TERMINAL = 100000u, OAK_BOOT = 10000u, OAK_PROBE_STEPS = 20000u, OAK_PROBE_BLOCKS = 3u, OAK_TIEBREAK = 4u, OAK_TRACE_SEC = 10u };
static const double OAK_MARGIN = 0.01, OAK_ALPHA = 0.05, OAK_POWER = 0.8, OAK_PI = 3.14159265358979323846;

static char REPO[PATH_MAX], BENCH[PATH_MAX], ACTIVE[PATH_MAX], JMSG[2048];
static int HAS_ACTIVE, IN_JOB;
static jmp_buf JJMP;
static volatile sig_atomic_t GOT_INT;

static void die(const char *fmt, ...) { va_list a; va_start(a, fmt); vsnprintf(JMSG, sizeof JMSG, fmt, a); va_end(a); if (IN_JOB) longjmp(JJMP, 1); fprintf(stderr, "%s\n", JMSG); exit(1); }
static void *xmal(size_t n) { void *p = malloc(n ? n : 1u); if (!p) die("oom"); return p; }
static void *xcal(size_t n, size_t s) { void *p = calloc(n ? n : 1u, s ? s : 1u); if (!p) die("oom"); return p; }
static void *xrea(void *p, size_t n) { void *q = realloc(p, n ? n : 1u); if (!q) die("oom"); return q; }
static char *xndup(const char *s, size_t n) { char *p = xmal(n + 1u); memcpy(p, s, n); p[n] = '\0'; return p; }
static char *xdup(const char *s) { return xndup(s, strlen(s)); }
static char *pf(const char *fmt, ...) {
    static _Thread_local char ring[32][4096]; static _Thread_local unsigned slot;
    char *d = ring[slot++ & 31u]; va_list a; va_start(a, fmt); if ((size_t)vsnprintf(d, 4096u, fmt, a) >= 4096u) die("path"); va_end(a); return d;
}
typedef struct { char **v; size_t n, c; } SV;
static void svAdd(SV *s, char *p) { if (s->n == s->c) s->v = xrea(s->v, (s->c = s->c ? s->c * 2u : 16u) * sizeof *s->v); s->v[s->n++] = p; }
static int cmpS(const void *a, const void *b) { return strcmp(*(const char *const *)a, *(const char *const *)b); }
static void svSort(SV *s) { if (s->n > 1u) qsort(s->v, s->n, sizeof *s->v, cmpS); }
static void svFree(SV *s) { for (size_t i = 0u; i < s->n; i++) free(s->v[i]); free(s->v); *s = (SV){0}; }
static bool isKind(const char *p, mode_t m) { struct stat st; return !stat(p, &st) && (!m || (st.st_mode & S_IFMT) == m); }
static const char *baseOf(const char *p) { const char *s = strrchr(p, '/'); return s ? s + 1 : p; }
static void mkdirP(const char *p) { char t[PATH_MAX]; snprintf(t, sizeof t, "%s", p); for (char *q = t + 1; *q; q++) if (*q == '/') { *q = '\0'; mkdir(t, 0755); *q = '/'; } if (mkdir(t, 0755) && errno != EEXIST) die("mkdir %s", t); }
static void rmRf(const char *pIn) {
    char p[PATH_MAX]; snprintf(p, sizeof p, "%s", pIn); struct stat st; if (lstat(p, &st)) return;
    if (!S_ISDIR(st.st_mode)) { unlink(p); return; } DIR *d = opendir(p); if (!d) return;
    for (struct dirent *e; (e = readdir(d)); ) if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) rmRf(pf("%s/%s", p, e->d_name));
    closedir(d); rmdir(p);
}
static char *readAll(const char *p, size_t *len) {
    FILE *f = fopen(p, "rb"); if (!f) die("read %s: %s", p, strerror(errno));
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f); char *b = xmal((size_t)sz + 1u);
    if (sz > 0 && fread(b, 1u, (size_t)sz, f) != (size_t)sz) die("short read %s", p); b[sz] = '\0'; fclose(f); if (len) *len = (size_t)sz; return b;
}
static char *readLine(const char *p) { char *s = readAll(p, NULL), *nl = strpbrk(s, "\r\n"); if (nl) *nl = '\0'; return s; }
static void writeAll(const char *p, const void *d, size_t n) { FILE *f = fopen(p, "wb"); if (!f) die("write %s: %s", p, strerror(errno)); if ((n && fwrite(d, 1u, n, f) != n) || fclose(f)) die("write %s", p); }
static void copyFast(const char *s, const char *d) { unlink(d); if (!clonefile(s, d, 0)) return; size_t n = 0u; char *b = readAll(s, &n); writeAll(d, b, n); free(b); struct stat st; if (!stat(s, &st)) chmod(d, st.st_mode & 0777); }
static void copyTree(const char *sIn, const char *dIn) {
    char s[PATH_MAX], d[PATH_MAX]; snprintf(s, sizeof s, "%s", sIn); snprintf(d, sizeof d, "%s", dIn); mkdirP(d); DIR *dir = opendir(s); if (!dir) die("opendir %s", s);
    for (struct dirent *e; (e = readdir(dir)); ) if (e->d_name[0] != '.') { char sp[PATH_MAX], dp[PATH_MAX]; snprintf(sp, sizeof sp, "%s/%s", s, e->d_name); snprintf(dp, sizeof dp, "%s/%s", d, e->d_name); if (isKind(sp, S_IFDIR)) copyTree(sp, dp); else copyFast(sp, dp); }
    closedir(dir);
}
static void listExt(const char *dir, const char *ext, const char *pre, SV *out) {
    DIR *d = opendir(dir); if (!d) return; size_t el = ext ? strlen(ext) : 0u;
    for (struct dirent *e; (e = readdir(d)); ) { size_t nl = strlen(e->d_name); if (e->d_name[0] != '.' && (!el || (nl > el && !strcmp(e->d_name + nl - el, ext)))) svAdd(out, xdup(pre ? pf("%s/%s", pre, e->d_name) : e->d_name)); }
    closedir(d); svSort(out);
}
static void hexEnc(const uint8_t *r, size_t n, char *o) { for (size_t i = 0u; i < n; i++) sprintf(o + 2u * i, "%02x", r[i]); o[2u * n] = '\0'; }
static void shaHex(const void *d, size_t n, char o[65]) { uint8_t r[32]; CC_SHA256(d, (CC_LONG)n, r); hexEnc(r, 32u, o); }
static void shaFile(const char *p, char o[65]) { size_t n = 0u; char *d = readAll(p, &n); shaHex(d, n, o); free(d); }
static bool filesEq(const char *a, const char *b) { struct stat sa, sb; if (stat(a, &sa) || stat(b, &sb) || sa.st_size != sb.st_size) return false; size_t la = 0u, lb = 0u; char *da = readAll(a, &la), *db = readAll(b, &lb); bool eq = la == lb && !memcmp(da, db, la); free(da); free(db); return eq; }
static void hashEntry(CC_SHA256_CTX *c, const char *root, const char *rel) { char z = '\0'; size_t n = 0u; char *d = readAll(pf("%s/%s", root, rel), &n); CC_SHA256_Update(c, rel, (CC_LONG)strlen(rel)); CC_SHA256_Update(c, &z, 1u); CC_SHA256_Update(c, d, (CC_LONG)n); CC_SHA256_Update(c, &z, 1u); free(d); }
static void collectRec(const char *root, const char *rel, SV *out) {
    DIR *d = opendir(pf("%s/%s", root, rel)); if (!d) die("opendir %s", rel);
    for (struct dirent *e; (e = readdir(d)); ) if (e->d_name[0] != '.') { char cr[PATH_MAX]; snprintf(cr, sizeof cr, "%s/%s", rel, e->d_name); if (isKind(pf("%s/%s", root, cr), S_IFDIR)) collectRec(root, cr, out); else if (isKind(pf("%s/%s", root, cr), S_IFREG)) svAdd(out, xdup(cr)); }
    closedir(d);
}
static void digestEntries(const char *root, const char *const *ents, size_t cnt, char out[65]) {
    SV s = {0}; for (size_t i = 0u; i < cnt; i++) svAdd(&s, xdup(ents[i])); svSort(&s); CC_SHA256_CTX c; CC_SHA256_Init(&c);
    for (size_t i = 0u; i < s.n; i++) { SV fs = {0}; if (isKind(pf("%s/%s", root, s.v[i]), S_IFDIR)) { collectRec(root, s.v[i], &fs); svSort(&fs); } else svAdd(&fs, xdup(s.v[i])); for (size_t j = 0u; j < fs.n; j++) hashEntry(&c, root, fs.v[j]); svFree(&fs); }
    svFree(&s); uint8_t r[32]; CC_SHA256_Final(r, &c); hexEnc(r, 32u, out);
}
static void stampNow(char o[32]) { time_t t = time(NULL); struct tm tm; localtime_r(&t, &tm); strftime(o, 32u, "%Y-%m-%dT%H:%M:%S", &tm); }
static uint64_t timeNs(void) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }
static double monoSec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }
static char *fmtCommas(double v) { static _Thread_local char b[8][32]; static _Thread_local unsigned k; char raw[64], *o = b[k++ & 7u], *q = o; snprintf(raw, sizeof raw, "%.0f", v); const char *p = raw; if (*p == '-') *q++ = *p++; for (size_t i = 0u, d = strlen(p); i < d; i++) { if (i && (d - i) % 3u == 0u) *q++ = ','; *q++ = p[i]; } *q = '\0'; return o; }
static char *fmtVal(double v) { static _Thread_local char b[8][32]; static _Thread_local unsigned k; char *o = b[k++ & 7u]; double a = fabs(v); snprintf(o, 32u, (a < 1e-3 || a >= 1e4) ? "%.4g" : "%.4f", v); return o; }
static pid_t spawnProc(char *const av[], const char *cwd, int inFd, int outFd, int errFd, bool setsid, bool search) {
    posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa); if (cwd) posix_spawn_file_actions_addchdir(&fa, cwd);
    if (inFd == -2) posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0); else if (inFd >= 0) posix_spawn_file_actions_adddup2(&fa, inFd, 0);
    if (outFd == -2) posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0); else if (outFd >= 0) posix_spawn_file_actions_adddup2(&fa, outFd, 1);
    if (errFd == -2) posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0); else if (errFd >= 0) posix_spawn_file_actions_adddup2(&fa, errFd, 2);
    posix_spawnattr_t at; posix_spawnattr_init(&at); if (setsid) posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSID);
    pid_t pid = 0; int rc = (search ? posix_spawnp : posix_spawn)(&pid, av[0], &fa, setsid ? &at : NULL, av, environ);
    posix_spawnattr_destroy(&at); posix_spawn_file_actions_destroy(&fa); if (rc) die("spawn %s: %s", av[0], strerror(rc)); return pid;
}
static int runProc(char *const av[], const char *cwd, char **cap, int capFd, bool quiet) {
    int p[2] = {-1, -1}; if (cap && pipe(p)) die("pipe");
    pid_t pid = spawnProc(av, cwd, -1, (cap && capFd == 1) ? p[1] : quiet ? -2 : -1, (cap && capFd == 2) ? p[1] : quiet ? -2 : -1, false, true);
    if (p[1] >= 0) close(p[1]);
    if (cap) { size_t l = 0u; FILE *m = open_memstream(cap, &l); char tmp[8192]; ssize_t n; while ((n = read(p[0], tmp, sizeof tmp)) > 0 || (n < 0 && errno == EINTR)) if (n > 0) fwrite(tmp, 1u, (size_t)n, m); fclose(m); close(p[0]); }
    int st = 0; while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {} return WIFEXITED(st) ? WEXITSTATUS(st) : 128;
}

typedef enum { JN, JB, JD, JS, JA, JO } JK;
typedef struct JV JV; typedef struct { char *k; JV *v; } JF;
struct JV { JK k; double d; char *s; JV **a; JF *f; size_t n, c; };
static JV *jNew(JK k) { JV *v = xcal(1u, sizeof *v); v->k = k; return v; }
static void jFree(JV *v) { if (!v) return; if (v->k == JS) free(v->s); else if (v->k == JA) { for (size_t i = 0u; i < v->n; i++) jFree(v->a[i]); free(v->a); } else if (v->k == JO) { for (size_t i = 0u; i < v->n; i++) { free(v->f[i].k); jFree(v->f[i].v); } free(v->f); } free(v); }
static void jWs(const char **p) { while (isspace((unsigned char)**p)) (*p)++; }
static char *jStr(const char **p) {
    if (*(*p)++ != '"') die("json str"); char *o = NULL; size_t l = 0u; FILE *m = open_memstream(&o, &l);
    while (**p && **p != '"') { if (**p == '\\') { char c = *(++(*p)); if (!c) die("json esc"); (*p)++; fputc(c == 'n' ? '\n' : c == 'r' ? '\r' : c == 't' ? '\t' : c, m); } else fputc(*(*p)++, m); }
    if (*(*p)++ != '"') die("json end"); fclose(m); return o;
}
static JV *jVal(const char **p) {
    jWs(p); char c = **p; if (c == '"') { JV *v = jNew(JS); v->s = jStr(p); return v; }
    if (c == '{' || c == '[') {
        (*p)++; JV *v = jNew(c == '{' ? JO : JA); jWs(p); char cl = c == '{' ? '}' : ']'; if (**p == cl) { (*p)++; return v; }
        for (;;) {
            jWs(p);
            if (c == '{') { char *k = jStr(p); jWs(p); if (*(*p)++ != ':') die("json :"); if (v->n == v->c) v->f = xrea(v->f, (v->c = v->c ? v->c * 2u : 8u) * sizeof *v->f); v->f[v->n++] = (JF){ k, jVal(p) }; }
            else { if (v->n == v->c) v->a = xrea(v->a, (v->c = v->c ? v->c * 2u : 8u) * sizeof *v->a); v->a[v->n++] = jVal(p); }
            jWs(p); if (**p == ',') { (*p)++; continue; } if (**p == cl) { (*p)++; break; } die("json delim");
        }
        return v;
    }
    if (!strncmp(*p, "true", 4u) || !strncmp(*p, "null", 4u)) { *p += 4u; return jNew(c == 't' ? JB : JN); }
    if (!strncmp(*p, "false", 5u)) { *p += 5u; return jNew(JB); }
    char *e = NULL; double d = strtod(*p, &e); if (e == *p) die("json tok"); *p = e; JV *v = jNew(JD); v->d = d; return v;
}
static JV *jParse(const char *s) { return jVal(&s); }
static JV *jFile(const char *p) { char *s = readAll(p, NULL); JV *v = jParse(s); free(s); return v; }
static JV *jGet(const JV *o, const char *k) { if (!o || o->k != JO) return NULL; for (size_t i = 0u; i < o->n; i++) if (!strcmp(o->f[i].k, k)) return o->f[i].v; return NULL; }

static bool isId(char c) { return isalnum((unsigned char)c) || c == '_'; }
static const char *scanId(const char *s, const char *e) { while (s < e && isId(*s)) s++; return s; }
static bool isScalar(const char *s) { static const char *const T[] = { "int8_t","uint8_t","int16_t","uint16_t","int32_t","uint32_t","int","float","double" }; for (size_t i = 0u; i < 9u; i++) if (!strcmp(s, T[i])) return true; return false; }
static bool isNumLit(const char *s) {
    if (*s == '-') s++; if (!isdigit((unsigned char)*s)) return false; while (isdigit((unsigned char)*s)) s++;
    if (*s == '.') { s++; if (!isdigit((unsigned char)*s)) return false; while (isdigit((unsigned char)*s)) s++; }
    if (*s == 'e' || *s == 'E') { s++; if (*s == '-') s++; if (!isdigit((unsigned char)*s)) return false; while (isdigit((unsigned char)*s)) s++; }
    return !*s || ((*s == 'u' || *s == 'U' || *s == 'f' || *s == 'F') && !s[1]);
}
typedef struct { char kind[32], name[64]; bool arr; } CField; typedef struct { char name[64]; CField f[128]; size_t n; } CStruct; typedef struct { char a[64], k[32]; } CAlias;
static const char *resAlias(const CAlias *al, size_t n, const char *k) { for (size_t i = 0u; i < n; i++) if (!strcmp(al[i].a, k)) return al[i].k; return k; }
static size_t parseFields(const char *body, CField *out) {
    size_t n = 0u;
    for (const char *p = body, *sc; *p; p = sc ? sc + 1 : p + strlen(p)) {
        sc = strchr(p, ';'); char *dl = xndup(p, sc ? (size_t)(sc - p) : strlen(p)), *s = dl; while (*s && isspace((unsigned char)*s)) s++;
        if (*s) {
            char *ks = s, kd[32] = {0}; while (*s && !isspace((unsigned char)*s)) s++; memcpy(kd, ks, (size_t)(s - ks));
            for (; *s; ) {
                while (*s && (isspace((unsigned char)*s) || *s == ',')) s++; if (!*s) break; char *ns = s; while (*s && *s != ',') s++; char *ne = s; while (ne > ns && isspace((unsigned char)ne[-1])) ne--;
                if (ne > ns) { char *br = memchr(ns, '[', (size_t)(ne - ns)); size_t nl = br ? (size_t)(br - ns) : (size_t)(ne - ns); while (nl && isspace((unsigned char)ns[nl - 1u])) nl--; memcpy(out[n].kind, kd, 32u); memcpy(out[n].name, ns, nl); out[n].name[nl] = '\0'; out[n++].arr = br != NULL; }
            }
        }
        free(dl);
    }
    return n;
}
static void cKey(FILE *f, size_t *idx, const char *k) { fprintf(f, "fputs(\"%s\\\"%s\\\":\",out);\n", (*idx)++ ? "," : "", k); }
static void cVal(FILE *f, const char *e, const char *k, bool arr, const CAlias *al, size_t na) { const char *b = resAlias(al, na, k); if (arr) fprintf(f, "fputc('[',out);for(uint32_t j=0u;j<(uint32_t)(sizeof %s/sizeof(%s));j++)fprintf(out,\"%%s%%.9g\",j?\",\":\"\",(double)((const %s*)(const void*)&%s)[j]);fputc(']',out);\n", e, b, b, e); else fprintf(f, "fprintf(out,\"%%.9g\",(double)(%s));\n", e); }
static void genCatUnit(const char *outIn, const char *fn, const char *fam, const char *src) {
    static const char *const HOOK[][2] = {
        {"hall","fputs(\",\\\"grid\\\":\",out);catalogWalls(out,&HALL_WALLS);\n"},
        {"maze","if(k->layout){int s=(int)maze_side(k);fprintf(out,\",\\\"grid\\\":{\\\"w\\\":%d,\\\"h\\\":%d,\\\"walls\\\":[\",s+2,s+2);for(int y=0;y<s+2;y++)for(int x=0;x<s+2;x++)fprintf(out,\"%s%d\",x||y?\",\":\"\",maze_wall(k,0,x,y));fputs(\"]}\",out);}\n"},
        {"corridor","int w=4*(int)(k->doors+1u)+1,h=7;fprintf(out,\",\\\"grid\\\":{\\\"w\\\":%d,\\\"h\\\":%d,\\\"walls\\\":[\",w,h);for(int y=0;y<h;y++)for(int x=0;x<w;x++)fprintf(out,\"%s%d\",x||y?\",\":\"\",corridor_wall(k,x,y));fputs(\"]}\",out);\n"},
        {"taxi","fputs(\",\\\"fences\\\":[\",out);int fl=1;for(int y=0;y<(int)TAXI_SIDE;y++)for(int x=0;x<(int)TAXI_SIDE;x++)for(uint32_t a=0u;a<4u;a++){int dx,dy;gridDelta(a,&dx,&dy);int nx=x+dx,ny=y+dy;if(nx<0||ny<0||nx>=(int)TAXI_SIDE||ny>=(int)TAXI_SIDE||!taxi_blocked(x,y,nx,ny))continue;fprintf(out,\"%s[%d,%d,%d,%d]\",fl?\"\":\",\",x,y,nx,ny);fl=0;}fputs(\"]\",out);\n"},
        {"cliff","uint8_t m[CLIFF_WIDTH*CLIFF_HEIGHT]={0};for(uint32_t y=0u;y<CLIFF_HEIGHT;y++)for(uint32_t x=0u;x<CLIFF_WIDTH;x++)for(uint32_t a=0u;a<4u;a++){int dx,dy;gridDelta(a,&dx,&dy);int tx=(int)x+dx,ty=(int)y-dy;tx=tx<0?0:tx>=(int)CLIFF_WIDTH?(int)CLIFF_WIDTH-1:tx;ty=ty<0?0:ty>=(int)CLIFF_HEIGHT?(int)CLIFF_HEIGHT-1:ty;uint8_t c=cliff_move(x,y,a);if(c==CLIFF_FELL)m[(uint32_t)ty*CLIFF_WIDTH+(uint32_t)tx]=1u;if(c==CLIFF_REACHED)m[(uint32_t)ty*CLIFF_WIDTH+(uint32_t)tx]=2u;}fputs(\",\\\"hazards\\\":[\",out);int fl=1;for(uint32_t c=0u;c<CLIFF_WIDTH*CLIFF_HEIGHT;c++)if(m[c]==1u){fprintf(out,\"%s[%u,%u]\",fl?\"\":\",\",c%CLIFF_WIDTH,c/CLIFF_WIDTH);fl=0;}fputs(\"]\",out);for(uint32_t c=0u;c<CLIFF_WIDTH*CLIFF_HEIGHT;c++)if(m[c]==2u)fprintf(out,\",\\\"goal\\\":[%u,%u]\",c%CLIFF_WIDTH,c/CLIFF_WIDTH);\n"},
        {"blocking","for(int y=0;y<(int)BLOCKING_HEIGHT;y++)for(int x=0;x<(int)BLOCKING_WIDTH;x++)for(uint32_t a=0u;a<4u;a++){int dx,dy;gridDelta(a,&dx,&dy);if(!gridWall(&k->phase[0],x+dx,y+dy)&&blocking_move(&k->phase[0],x,y,a)/BLOCKING_GOAL){fprintf(out,\",\\\"goal\\\":[%d,%d]\",x+dx,y+dy);goto pl;}}pl:;\n"}
    };
    char outP[PATH_MAX], kType[64] = {0}, kBody[4096] = {0}; snprintf(outP, sizeof outP, "%s", outIn); CAlias al[32]; size_t na = 0u, ns = 0u, pi = 0u, ci = 0u; CStruct sh[64];
    for (const char *p = src, *nl; *p; p = nl ? nl + 1 : p + strlen(p)) {
        nl = strchr(p, '\n'); size_t ll = nl ? (size_t)(nl - p) : strlen(p);
        if (!strncmp(p, "typedef ", 8u) && strncmp(p, "typedef struct ", 15u)) { char k[32], a[64], sc, *ln = xndup(p, ll); if (sscanf(ln, "typedef %31s %63[^;]%c", k, a, &sc) == 3 && sc == ';' && isScalar(k)) { strcpy(al[na].k, k); strcpy(al[na++].a, a); } free(ln); }
    }
    for (const char *p = src; (p = strstr(p, "typedef struct {")); ) {
        const char *bs = p + 16, *be = strchr(bs, '}'); if (!be) break; const char *nm = be + 1; while (*nm && isspace((unsigned char)*nm)) nm++; const char *ne = scanId(nm, nm + 64);
        if (*ne == ';' && ne > nm && (size_t)(ne - nm) < 64u) {
            size_t nl = (size_t)(ne - nm), bl = (size_t)(be - bs); char name[64] = {0}, *body = xndup(bs, bl); memcpy(name, nm, nl);
            if (nl >= 5u && !strcmp(name + nl - 5u, "Knobs") && !kType[0]) { strcpy(kType, name); memcpy(kBody, body, bl + 1u); }
            CField fs[128]; size_t nf = parseFields(body, fs); bool ok = true; for (size_t i = 0u; i < nf; i++) if (!isScalar(resAlias(al, na, fs[i].kind))) ok = false;
            if (ok && ns < 64u) { strcpy(sh[ns].name, name); memcpy(sh[ns].f, fs, nf * sizeof *fs); sh[ns++].n = nf; } free(body);
        }
        p = ne;
    }
    FILE *f = fopen(outP, "wb"); if (!f) die("write %s", outP);
    fprintf(f, "#include <stdio.h>\n#include \"../../Sources/envs/%s\"\n__attribute__((noinline,cold,optnone,unused)) static void catalogWalls(FILE *out,const GridWalls *g){fprintf(out,\"{\\\"w\\\":%%d,\\\"h\\\":%%d,\\\"walls\\\":[\",g->width,g->height);for(int y=0;y<g->height;y++)for(int x=0;x<g->width;x++)fprintf(out,\"%%s%%d\",x||y?\",\":\"\",gridWall(g,x,y));fputs(\"]}\",out);}\n__attribute__((noinline,cold,optnone)) void catalog_%s(FILE *out,uint32_t instance){\n", fn, fam);
    if (kType[0]) { const char *tp = strstr(src, pf("static const %s ", kType)); if (!tp) die("knobs %s", fam); tp += 14 + strlen(kType); const char *te = scanId(tp, tp + 64); fprintf(f, "const %s *k=&%.*s[instance];\n", kType, (int)(te - tp), tp); } else fputs("(void)instance;\n", f);
    fputs("fputs(\",\\\"parameters\\\":{\",out);\n", f);
    if (kType[0]) { CField kf[128]; size_t nkf = parseFields(kBody, kf); for (size_t i = 0u; i < nkf; i++) { cKey(f, &pi, kf[i].name); if (!strcmp(kf[i].kind, "GridWalls")) fprintf(f, "fputc('[',out);for(int j=0;j<%d;j++){if(j)fputc(',',out);catalogWalls(out,&k->%s[j]);}fputc(']',out);\n", atoi(strstr(kBody, pf("%s[", kf[i].name)) + strlen(kf[i].name) + 1), kf[i].name); else fprintf(f, "fprintf(out,\"%%.9g\",(double)(k->%s));\n", kf[i].name); } }
    if (!strcmp(fam, "jbw")) { const char *pp = strstr(src, "step / ") + 7; size_t i = 0u; while (isdigit((unsigned char)pp[i]) || pp[i] == 'u') i++; fprintf(f, "fputs(\",\\\"polarityPeriod\\\":\",out);fprintf(out,\"%%u\",%.*s);\n", (int)i, pp); }
    fputs("fputs(\"},\\\"constants\\\":{\",out);\n", f);
    for (int pass = 0; pass < 2; pass++) for (const char *p = src, *nl; *p; p = nl ? nl + 1 : p + strlen(p)) {
        nl = strchr(p, '\n'); const char *le = nl ? nl : p + strlen(p);
        if (!pass && !strncmp(p, "#define ", 8u)) {
            const char *ks = p + 8, *ke = scanId(ks, le);
            if (ke > ks && ke < le && *ke == ' ') { const char *vs = ke + 1, *ve = le; while (vs < ve && isspace((unsigned char)*vs)) vs++; while (ve > vs && isspace((unsigned char)ve[-1])) ve--; char *val = xndup(vs, (size_t)(ve - vs)), *key = xndup(ks, (size_t)(ke - ks)); if (isNumLit(val)) { cKey(f, &ci, key); fprintf(f, "fprintf(out,\"%%.9g\",(double)(%s));\n", key); } free(val); free(key); }
        } else if (pass && !strncmp(p, "static const ", 13u)) {
            const char *ks = p + 13, *ke = scanId(ks, le);
            if (ke > ks && ke < le && *ke == ' ') {
                const char *ns2 = ke + 1, *ne2 = scanId(ns2, le), *de = ne2; while (de < le && *de == '[') { while (de < le && *de != ']') de++; if (de < le) de++; }
                if (ne2 > ns2 && de + 3 <= le && !strncmp(de, " = ", 3u)) {
                    char *kd = xndup(ks, (size_t)(ke - ks)), *nm = xndup(ns2, (size_t)(ne2 - ns2)); bool hd = de > ne2;
                    if (isScalar(resAlias(al, na, kd))) { cKey(f, &ci, nm); cVal(f, nm, kd, hd, al, na); }
                    else if (strcmp(kd, kType)) for (size_t si = 0u; si < ns; si++) if (!strcmp(sh[si].name, kd)) {
                        cKey(f, &ci, nm); if (hd) fprintf(f, "fputc('[',out);for(uint32_t i=0u;i<(uint32_t)(sizeof %s/sizeof %s[0]);i++){if(i)fputc(',',out);\n", nm, nm); fputs("fputc('{',out);\n", f);
                        for (size_t fi = 0u, fi0 = 0u; fi < sh[si].n; fi++) { cKey(f, &fi0, sh[si].f[fi].name); cVal(f, pf(hd ? "%s[i].%s" : "%s.%s", nm, sh[si].f[fi].name), sh[si].f[fi].kind, sh[si].f[fi].arr, al, na); }
                        fputs(hd ? "fputc('}',out);}fputc(']',out);\n" : "fputc('}',out);\n", f); break;
                    }
                    free(kd); free(nm);
                }
            }
        }
    }
    fputs("fputc('}',out);\n", f); for (size_t i = 0u; i < 6u; i++) if (!strcmp(fam, HOOK[i][0])) fputs(HOOK[i][1], f);
    if (!strcmp(fam, "jbw")) {
        const char *d1 = strstr(src, "response[current + i] = ") + 24, *q = src; char dec[32] = {0}, dif[32] = {0};
        for (size_t i = 0u; d1[i] && (isdigit((unsigned char)d1[i]) || d1[i] == '.' || d1[i] == 'f'); i++) dec[i] = d1[i];
        while ((q = strstr(q, "+= "))) { const char *r = q + 3; size_t j = 0u; while (r[j] && (isdigit((unsigned char)r[j]) || r[j] == '.')) j++; if (j && r[j] == 'f' && !strncmp(r + j + 1, " * (", 4u)) { memcpy(dif, r, j + 1u); break; } q += 3; }
        fprintf(f, "fprintf(out,\",\\\"scent\\\":{\\\"decay\\\":%%.9g,\\\"diffusion\\\":%%.9g,\\\"lifetime\\\":%%u,\\\"values\\\":[\",(double)%s,(double)%s,JBW_DESTRUCTION_LIFETIME);for(uint32_t it=JBW_EMPTY;it<=JBW_TRUFFLE;it++){float s[3];jbw_scent(it,s);fprintf(out,\"%%s[%%.9g,%%.9g,%%.9g]\",it?\",\":\"\",(double)s[0],(double)s[1],(double)s[2]);}fputs(\"]}\",out);\n", dec, dif);
    }
    fputs("}\n", f); if (fclose(f)) die("close %s", outP);
}
static void genCatalog(const char *tree) {
    char *ec = readAll(pf("%s/Sources/env.c", tree), NULL); SV fams = {0}, files = {0};
    for (const char *p = ec, *nl; *p; p = nl ? nl + 1 : p + strlen(p)) { nl = strchr(p, '\n'); const char *le = nl ? nl : p + strlen(p); if (!strncmp(p, "    F(", 6u)) { const char *fe = scanId(p + 6, le); if (fe > p + 6 && *fe == ',') svAdd(&fams, xndup(p + 6, (size_t)(fe - (p + 6)))); } }
    free(ec); rmRf(pf("%s/.build/catalog", tree)); mkdirP(pf("%s/.build/catalog", tree)); listExt(pf("%s/Sources/envs", tree), ".c", NULL, &files);
    for (size_t i = 0u; i < files.n; i++) {
        char *src = readAll(pf("%s/Sources/envs/%s", tree, files.v[i]), NULL), fam[64] = {0};
        for (const char *rp = strstr(src, "_reset("); rp; rp = strstr(rp + 7, "_reset(")) { const char *s = rp; while (s > src && isId(s[-1])) s--; if (s >= src + 5 && !strncmp(s - 5, "void ", 5u)) { memcpy(fam, s, (size_t)(rp - s)); break; } }
        if (!fam[0]) die("family %s", files.v[i]); genCatUnit(pf("%s/.build/catalog/%s.c", tree, fam), files.v[i], fam, src); free(src);
    }
    if (files.n != fams.n) die("catalog mismatch"); svFree(&files);
    FILE *h = fopen(pf("%s/.build/catalog.h", tree), "wb"); if (!h) die("catalog.h");
    for (size_t i = 0u; i < fams.n; i++) fprintf(h, "void catalog_%s(FILE *out,uint32_t instance);\n", fams.v[i]);
    fputs("static void catalogDetails(FILE *out,const EnvSpec *env){\n", h);
    for (size_t i = 0u; i < fams.n; i++) fprintf(h, "if(!strcmp(env->family,\"%s\")){catalog_%s(out,env->instance);return;}\n", fams.v[i], fams.v[i]);
    fputs("abort();\n}\n", h); fclose(h); svFree(&fams);
}

static char *tcFingerprint(void) {
    char *ca[] = { "clang","-###","-x","c","-c","/dev/null","-o","/dev/null",NULL }, *la[] = { "ld","-v",NULL }, *dr = NULL, *lv = NULL, *o = NULL; size_t l = 0u;
    if (runProc(ca, NULL, &dr, 2, true)) die("clang probe"); const char *is = strstr(dr, "\"-isysroot\" \""); if (!is) die("isysroot");
    is += 13; char *rs = xndup(is, (size_t)(strchr(is, '"') - is)), res[PATH_MAX]; if (!realpath(rs, res)) snprintf(res, sizeof res, "%s", rs); free(rs);
    runProc(la, NULL, &lv, 2, true); FILE *m = open_memstream(&o, &l); fputs(dr, m); fputs(res, m); fputs(lv, m); fclose(m); free(dr); free(lv); return o;
}
typedef struct { size_t sz; char *src, *dst; } CJob;
typedef struct { CJob *j; size_t n; _Atomic size_t nxt; char **fl; size_t nf; const char *tree; pid_t pid; _Atomic int fail; } CPool;
static int cmpJob(const void *a, const void *b) { size_t sa = ((const CJob *)a)->sz, sb = ((const CJob *)b)->sz; return sb > sa ? 1 : sb < sa ? -1 : 0; }
static void *ccWork(void *arg) {
    CPool *p = arg; char **av = xmal((p->nf + 6u) * sizeof *av); av[0] = "clang"; for (size_t i = 0u; i < p->nf; i++) av[1u + i] = p->fl[i]; av[1u + p->nf] = "-c"; av[3u + p->nf] = "-o"; av[5u + p->nf] = NULL;
    for (;;) { size_t i = __c11_atomic_fetch_add(&p->nxt, 1u, __ATOMIC_RELAXED); if (i >= p->n || __c11_atomic_load(&p->fail, __ATOMIC_RELAXED)) break; char *tmp = pf("%s.%d.%zu", p->j[i].dst, (int)p->pid, i); av[2u + p->nf] = p->j[i].src; av[4u + p->nf] = tmp; if (runProc(av, p->tree, NULL, 0, false) || rename(tmp, p->j[i].dst)) { unlink(tmp); __c11_atomic_store(&p->fail, 1, __ATOMIC_RELAXED); break; } }
    free(av); return NULL;
}
static void hashGroup(const char *tree, const char *bk, size_t bl, const char *const *ps, size_t n, uint8_t o[32]) { CC_SHA256_CTX c; CC_SHA256_Init(&c); CC_SHA256_Update(&c, bk, (CC_LONG)bl); for (size_t i = 0u; i < n; i++) hashEntry(&c, tree, ps[i]); CC_SHA256_Final(o, &c); }
static void buildTree(const char *tree, bool verb) {
    mkdirP(pf("%s/.build", tree)); mkdirP(pf("%s/.build/cache/obj", REPO)); mkdirP(pf("%s/.build/cache/thinlto", REPO)); genCatalog(tree);
    static const char ORD[] = "_oak_act\n_perceive\n_encode\n_evaluate\n_learn\n_learnTable\n_dynaSweep\n_sampleAction\n_planAction\n_rotate\n_worker\n_env_step\n";
    writeAll(pf("%s/.build/hot.order", tree), ORD, sizeof ORD - 1u); uint64_t cl = 128u; size_t cls = sizeof cl, bkl = 0u, nm = 0u; sysctlbyname("hw.cachelinesize", &cl, &cls, NULL, 0u);
    char af[48], *tc = tcFingerprint(), *bk = NULL; snprintf(af, sizeof af, "-falign-functions=%llu", (unsigned long long)cl);
    char *fl[] = { "-O3","-mcpu=apple-m4","-flto=thin",af,"-fno-fast-math","-ffp-contract=off","-Werror=double-promotion","-Wall","-Wextra","-ISources" }; size_t nf = sizeof fl / sizeof fl[0];
    FILE *bm = open_memstream(&bk, &bkl); fputs(tc, bm); fputc('\0', bm); free(tc); for (size_t i = 0u; i < nf; i++) { fputs(fl[i], bm); fputc('\0', bm); } fclose(bm);
    const char *ehp[] = { "Sources/env.h","Sources/rng.h","Sources/portable.h" }; uint8_t eh[32], ch[32]; hashGroup(tree, bk, bkl, ehp, 3u, eh);
    SV cH = {0}, cS = {0}, tS = {0}, objs = {0}; listExt(pf("%s/Sources", tree), ".h", "Sources", &cH); svAdd(&cH, xdup(".build/catalog.h")); hashGroup(tree, bk, bkl, (const char *const *)cH.v, cH.n, ch); free(bk); svFree(&cH);
    listExt(pf("%s/Sources", tree), ".c", "Sources", &cS); listExt(pf("%s/.build/catalog", tree), ".c", ".build/catalog", &tS); CJob *ms = xcal(cS.n + tS.n, sizeof *ms);
    for (int pass = 0; pass < 2; pass++) for (size_t i = 0u; i < (pass ? tS.n : cS.n); i++) {
        const char *rel = (pass ? tS.v : cS.v)[i]; size_t l = 0u, sz = 0u; char *d = readAll(pf("%s/%s", tree, rel), &l), z = '\0', hex[65]; sz = l;
        CC_SHA256_CTX c; CC_SHA256_Init(&c); CC_SHA256_Update(&c, pass ? eh : ch, 32u); CC_SHA256_Update(&c, rel, (CC_LONG)strlen(rel)); CC_SHA256_Update(&c, &z, 1u); CC_SHA256_Update(&c, d, (CC_LONG)l);
        if (pass) { const char *inc = strstr(d, "#include \"../../Sources/envs/") + 16; char *er = xndup(inc, (size_t)(strchr(inc, '"') - inc)), *ed = readAll(pf("%s/%s", tree, er), &sz); CC_SHA256_Update(&c, &z, 1u); CC_SHA256_Update(&c, ed, (CC_LONG)sz); free(er); free(ed); }
        uint8_t r[32]; CC_SHA256_Final(r, &c); free(d); hexEnc(r, 32u, hex); hex[32] = '\0'; char *dst = xdup(pf("%s/.build/cache/obj/%s.o", REPO, hex)); svAdd(&objs, dst); if (!isKind(dst, 0)) ms[nm++] = (CJob){ sz, xdup(rel), xdup(dst) };
    }
    svFree(&cS); svFree(&tS);
    if (nm) {
        qsort(ms, nm, sizeof *ms, cmpJob); long nc = sysconf(_SC_NPROCESSORS_ONLN); size_t nt = (size_t)(nc < 1 ? 1 : nc); if (nt > nm) nt = nm;
        CPool pool = { ms, nm, 0u, fl, nf, tree, getpid(), 0 }; pthread_t *ts = xcal(nt, sizeof *ts);
        for (size_t t = 0u; t < nt; t++) pthread_create(&ts[t], NULL, ccWork, &pool); for (size_t t = 0u; t < nt; t++) pthread_join(ts[t], NULL); free(ts); if (pool.fail) die("compile failed");
    }
    for (size_t i = 0u; i < nm; i++) { free(ms[i].src); free(ms[i].dst); } free(ms);
    char **la = xmal((nf + objs.n + 10u) * sizeof *la); size_t a = 0u; la[a++] = "clang"; for (size_t i = 0u; i < nf; i++) la[a++] = fl[i]; for (size_t i = 0u; i < objs.n; i++) la[a++] = objs.v[i];
    la[a++] = "-lm"; la[a++] = "-lpthread"; la[a++] = "-Wl,-order_file,.build/hot.order"; la[a++] = pf("-Wl,-cache_path_lto,%s/.build/cache/thinlto", REPO); la[a++] = "-o"; la[a++] = ".build/oak"; la[a++] = NULL;
    if (runProc(la, tree, NULL, 0, false)) die("link failed"); free(la); svFree(&objs);
    char *na[] = { "nm","-n",pf("%s/.build/oak", tree),NULL }, *no = NULL, fs[128] = {0}; if (runProc(na, NULL, &no, 1, false)) die("nm failed");
    for (const char *p = no, *nl; p && *p; p = nl ? nl + 1 : NULL) { nl = strchr(p, '\n'); char ad[64], kd[8], sm[128]; if (sscanf(p, "%63s %7s %127s", ad, kd, sm) == 3 && (kd[0] == 't' || kd[0] == 'T') && !kd[1] && strcmp(sm, "__mh_execute_header")) { strcpy(fs, sm); break; } }
    free(no); if (strcmp(fs, "_oak_act")) die("first sym %s != _oak_act", fs); if (verb) puts("Build complete.");
}

typedef struct { char *id; uint32_t s0, s1, base; } EBlk;
typedef struct { char root[PATH_MAX], pol[32]; uint32_t steps, lanes, nEnd, *ends, *health, *seeds; double sps, *score; uint64_t *hash; EBlk *blks; size_t nBlk, nSeed; JV *srcJ; } Rec;
static bool recDone(const char *r) { const char *p = pf("%s/meta.json", r); if (!isKind(p, S_IFREG)) return false; JV *m = jFile(p); bool ok = jGet(m, "finishedAt") != NULL; jFree(m); return ok; }
static Rec *loadRec(const char *root) {
    Rec *r = xcal(1u, sizeof *r); snprintf(r->root, sizeof r->root, "%s", root); JV *m = jFile(pf("%s/meta.json", r->root)); if (!jGet(m, "finishedAt")) die("Incomplete: %s", r->root);
    r->steps = (uint32_t)jGet(m, "steps")->d; r->lanes = (uint32_t)jGet(m, "batch")->d; r->sps = jGet(m, "stepsPerSecond")->d; snprintf(r->pol, sizeof r->pol, "%s", jGet(m, "policy")->s);
    JV *sj = jGet(m, "seeds"), *sc = jGet(m, "envSchedule"); r->nSeed = sj->n; r->seeds = xmal(sj->n * sizeof *r->seeds); for (size_t i = 0u; i < sj->n; i++) r->seeds[i] = (uint32_t)sj->a[i]->d;
    r->nBlk = sc->n; r->blks = xcal(sc->n, sizeof *r->blks);
    for (size_t i = 0u; i < sc->n; i++) { JV *e = sc->a[i]; uint32_t s0 = (uint32_t)jGet(e, "seedStart")->d, base = UINT32_MAX; for (size_t s = 0u; s < r->nSeed; s++) if (r->seeds[s] == s0) { base = (uint32_t)s; break; } if (base == UINT32_MAX) die("missing seedStart"); r->blks[i] = (EBlk){ xdup(jGet(e, "envId")->s), s0, (uint32_t)jGet(e, "seedEnd")->d, base }; }
    jFree(m); size_t lE, lS, lH, lHe; r->ends = (uint32_t *)readAll(pf("%s/ends.u32", r->root), &lE); r->nEnd = (uint32_t)(lE / 4u); float *sums = (float *)readAll(pf("%s/series/rewardSum.f32", r->root), &lS); r->hash = (uint64_t *)readAll(pf("%s/hash.u64", r->root), &lH); r->health = (uint32_t *)readAll(pf("%s/health.u32", r->root), &lHe);
    if (!r->nEnd || r->ends[r->nEnd - 1u] != r->steps || lS / 4u != (size_t)r->nEnd * r->lanes || lH / 8u != r->lanes || lHe / 4u != r->lanes) die("Incomplete: %s", r->root);
    uint32_t si = 0u, cut = r->steps > OAK_TERMINAL ? r->steps - OAK_TERMINAL : 0u; for (uint32_t i = 0u; i < r->nEnd; i++) if (r->ends[i] <= cut) si = i;
    double span = (double)(r->ends[r->nEnd - 1u] - r->ends[si]); size_t lr = (size_t)(r->nEnd - 1u) * r->lanes, fr = (size_t)si * r->lanes;
    r->score = xmal(r->lanes * sizeof(double)); for (uint32_t l = 0u; l < r->lanes; l++) r->score[l] = ((double)sums[lr + l] - (double)sums[fr + l]) / span; free(sums); return r;
}
static void freeRec(Rec *r) { if (!r) return; free(r->ends); free(r->score); free(r->hash); free(r->health); free(r->seeds); for (size_t i = 0u; i < r->nBlk; i++) free(r->blks[i].id); free(r->blks); jFree(r->srcJ); free(r); }
static const EBlk *rBlk(const Rec *r, const char *id) { for (size_t i = 0u; i < r->nBlk; i++) if (!strcmp(r->blks[i].id, id)) return &r->blks[i]; return NULL; }
static int rLane(const Rec *r, const char *id, uint32_t s) { const EBlk *b = rBlk(r, id); return (!b || s < b->s0 || s > b->s1) ? -1 : (int)(b->base + (s - b->s0)); }
static JV *rSrc(Rec *r) { return r->srcJ ? r->srcJ : (r->srcJ = jFile(pf("%s/sources.json", r->root))); }
typedef struct { struct { char nm[128]; Rec *r; } e[64]; size_t n; } RCache;
static Rec *rcGet(RCache *c, const char *nm) { for (size_t i = 0u; i < c->n; i++) if (!strcmp(c->e[i].nm, nm)) return c->e[i].r; snprintf(c->e[c->n].nm, 128u, "%s", nm); return (c->e[c->n++].r = loadRec(pf("%s/%s", BENCH, nm))); }
static void rcFree(RCache *c) { for (size_t i = 0u; i < c->n; i++) freeRec(c->e[i].r); c->n = 0u; }

typedef struct { char *id, *cat; uint32_t s0, s1; char dig[65]; } EMeta;
static int cmpEM(const void *a, const void *b) { return strcmp(((const EMeta *)a)->id, ((const EMeta *)b)->id); }
static void envDigests(const char *tree, EMeta *es, size_t n) {
    char *ec = readAll(pf("%s/Sources/env.c", tree), NULL); struct { char id[96], fam[64]; char *row; } *rows = xcal(n + 64u, sizeof *rows); size_t nr = 0u, nf = 0u;
    for (const char *p = ec, *nl; *p; p = nl ? nl + 1 : p + strlen(p)) {
        nl = strchr(p, '\n'); const char *s = p, *e = nl ? nl : p + strlen(p); while (s < e && isspace((unsigned char)*s)) s++; while (e > s && isspace((unsigned char)e[-1])) e--;
        if (e - s > 2 && s[0] == 'X' && s[1] == '(') { const char *c1 = memchr(s + 2, ',', (size_t)(e - s - 2)); if (c1) { const char *fs = c1 + 1; while (fs < e && isspace((unsigned char)*fs)) fs++; const char *fe = scanId(fs, e), *q1 = memchr(fe, '"', (size_t)(e - fe)), *q2 = q1 ? memchr(q1 + 1, '"', (size_t)(e - q1 - 1)) : NULL; if (q2 && fe > fs) { memcpy(rows[nr].fam, fs, (size_t)(fe - fs)); memcpy(rows[nr].id, q1 + 1, (size_t)(q2 - q1 - 1)); rows[nr++].row = xndup(s, (size_t)(e - s)); } } }
    }
    free(ec); SV fs = {0}; listExt(pf("%s/Sources/envs", tree), ".c", NULL, &fs); char **tx = xcal(fs.n, sizeof *tx); for (size_t i = 0u; i < fs.n; i++) tx[i] = readAll(pf("%s/Sources/envs/%s", tree, fs.v[i]), NULL);
    struct { char fam[64], dig[65]; } fc[256];
    for (size_t i = 0u; i < n; i++) {
        size_t r = SIZE_MAX; for (size_t j = 0u; j < nr; j++) if (!strcmp(rows[j].id, es[i].id)) { r = j; break; } if (r == SIZE_MAX) die("no env row %s", es[i].id);
        const char *lh = NULL; for (size_t k = 0u; k < nf; k++) if (!strcmp(fc[k].fam, rows[r].fam)) lh = fc[k].dig;
        if (!lh) {
            const char *hk = pf("%s_step(", rows[r].fam); size_t hl = strlen(hk); int ow = -1, cnt = 0;
            for (size_t f = 0u; f < fs.n; f++) for (const char *q = tx[f]; (q = strstr(q, hk)); q += hl) if (q == tx[f] || !isId(q[-1])) { ow = (int)f; cnt++; break; }
            if (cnt != 1) die("%s owners %d", es[i].id, cnt);
            const char *en[] = { "Sources/env.h","Sources/rng.h","Sources/portable.h",pf("Sources/envs/%s", fs.v[ow]) }; strcpy(fc[nf].fam, rows[r].fam); digestEntries(tree, en, 4u, fc[nf].dig); lh = fc[nf++].dig;
        }
        const char *cb = pf("%s%s", lh, rows[r].row); shaHex(cb, strlen(cb), es[i].dig);
    }
    for (size_t i = 0u; i < fs.n; i++) free(tx[i]); free(tx); svFree(&fs); for (size_t i = 0u; i < nr; i++) free(rows[i].row); free(rows);
}

static int cmpD(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y ? 1 : 0; }
static double meanA(const double *x, size_t n) { double s = 0.0, c = 0.0; for (size_t i = 0u; i < n; i++) { double t = s + x[i]; c += fabs(s) >= fabs(x[i]) ? (s - t) + x[i] : (x[i] - t) + s; s = t; } return (s + c) / (double)n; }
static double qSorted(const double *s, size_t n, double q) { double p = (double)(n - 1u) * q; size_t lo = (size_t)floor(p), hi = (size_t)ceil(p); return s[lo] + (s[hi] - s[lo]) * (p - (double)lo); }
static double iqmA(const double *x, size_t n) { double *s = xmal(n * sizeof *s); memcpy(s, x, n * sizeof *s); qsort(s, n, sizeof *s, cmpD); size_t lo = n / 4u, hi = n - n / 4u; double sm = 0.0; for (size_t i = lo; i < hi; i++) sm += s[i]; free(s); return sm / (double)(hi - lo); }
static double scaleA(const double *x, size_t n) {
    double *s = xmal(n * sizeof *s); memcpy(s, x, n * sizeof *s); qsort(s, n, sizeof *s, cmpD); double iqr = qSorted(s, n, 0.75) - qSorted(s, n, 0.25); if (iqr > 0.0) { free(s); return iqr; }
    double m = qSorted(s, n, 0.5); for (size_t i = 0u; i < n; i++) s[i] = fabs(x[i] - m); qsort(s, n, sizeof *s, cmpD); double mad = qSorted(s, n, 0.5) * 1.4826; free(s); return mad > 0.0 ? mad : 1.0;
}
static double normInv(double p) {
    static const double a[] = {-39.69683028665376,220.9460984245205,-275.9285104469687,138.357751867269,-30.66479806614716,2.506628277459239}, b[] = {-54.47609879822406,161.5858368580409,-155.6989798598866,66.80131188771972,-13.28068155288572}, c[] = {-7.784894002430293e-3,-0.3223964580411365,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783}, d[] = {7.784695709041462e-3,0.3224671290700398,2.445134137142996,3.754408661907416};
    if (p < 0.02425 || p > 0.97575) { double q = sqrt(-2.0 * log(p < 0.5 ? p : 1.0 - p)); return (p < 0.5 ? 1.0 : -1.0) * (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0); }
    double q = p - 0.5, r = q * q; return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
}
typedef struct { double v, a; } RItem;
static int cmpRA(const void *a, const void *b) { double x = ((const RItem *)a)->a, y = ((const RItem *)b)->a; return x < y ? -1 : x > y ? 1 : 0; }
static double wilcoxonLess(const double *df, size_t n) {
    RItem *nz = xmal(n * sizeof *nz); size_t m = 0u; for (size_t i = 0u; i < n; i++) if (df[i] != 0.0) nz[m++] = (RItem){ df[i], fabs(df[i]) };
    if (!m) { free(nz); return 1.0; } qsort(nz, m, sizeof *nz, cmpRA); double pos = 0.0, tie = 0.0;
    for (size_t i = 0u, j; i < m; i = j + 1u) { for (j = i; j + 1u < m && nz[j + 1u].a == nz[i].a; j++) {} double rk = (double)(i + j + 2u) / 2.0, t = (double)(j - i + 1u); for (size_t k = i; k <= j; k++) if (nz[k].v > 0.0) pos += rk; tie += t * t * t - t; }
    free(nz); double dm = (double)m, var = dm * (dm + 1.0) * (2.0 * dm + 1.0) / 24.0 - tie / 48.0; return var <= 0.0 ? 1.0 : 0.5 * erfc(-(pos - dm * (dm + 1.0) / 4.0) / sqrt(2.0 * var));
}
typedef struct { double p; size_t i; } PItem;
static int cmpP(const void *a, const void *b) { double x = ((const PItem *)a)->p, y = ((const PItem *)b)->p; return x < y ? -1 : x > y ? 1 : 0; }
static void holm(const double *pv, size_t n, bool *rej) { PItem *o = xmal(n * sizeof *o); for (size_t i = 0u; i < n; i++) { o[i] = (PItem){ pv[i], i }; rej[i] = false; } qsort(o, n, sizeof *o, cmpP); for (size_t r = 0u; r < n && o[r].p <= OAK_ALPHA / (double)(n - r); r++) rej[o[r].i] = true; free(o); }
static double winsA(const double *a, const double *b, size_t n) { double s = 0.0; for (size_t i = 0u; i < n; i++) s += a[i] > b[i] ? 1.0 : a[i] == b[i] ? 0.5 : 0.0; return s / (double)n; }
static double rhoOf(const double *a, const double *b, size_t n) { double ma = meanA(a, n), mb = meanA(b, n), va = 0.0, vb = 0.0, cv = 0.0; for (size_t i = 0u; i < n; i++) { double da = a[i] - ma, db = b[i] - mb; va += da * da; vb += db * db; cv += da * db; } if (va <= 0.0 || vb <= 0.0) return 0.0; double r = cv / sqrt(va * vb); return r < 1.0 ? r : 1.0; }
static double mdrOf(double rho, size_t tests, size_t n) { return (normInv(1.0 - OAK_ALPHA / (double)tests) + normInv(OAK_POWER)) * sqrt(2.0 * (1.0 - rho)) / (2.0 * normInv(0.75) * sqrt((double)n)); }
static double signTail(size_t w, size_t l) { size_t n = w + l; if (!n) return 1.0; double l2 = (double)n * log(2.0), s = 0.0; for (size_t k = w; k <= n; k++) s += exp(lgamma((double)n + 1.0) - lgamma((double)k + 1.0) - lgamma((double)(n - k) + 1.0) - l2); return s > 1.0 ? 1.0 : s; }
typedef struct { const double *const *g; const size_t *sz; size_t ng, tn; bool iqm; double *reps; size_t r0, r1; } BTask;
static void *bootWork(void *arg) {
    BTask *t = arg; double *p = t->iqm ? xmal(t->tn * sizeof *p) : NULL; size_t lo = t->tn / 4u, hi = t->tn - t->tn / 4u;
    for (size_t r = t->r0; r < t->r1; r++) {
        uint64_t rng = 0x243f6a8885a308d3ull ^ (uint64_t)r * 0x9e3779b97f4a7c15ull; size_t pos = 0u; double sm = 0.0;
        for (size_t g = 0u; g < t->ng; g++) for (size_t j = 0u; j < t->sz[g]; j++) { uint64_t z = (rng += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull; z ^= z >> 31u; double v = t->g[g][(size_t)(((__uint128_t)z * t->sz[g]) >> 64u)]; if (t->iqm) p[pos++] = v; else sm += v; }
        if (t->iqm) { qsort(p, t->tn, sizeof *p, cmpD); double s = 0.0; for (size_t i = lo; i < hi; i++) s += p[i]; t->reps[r] = s / (double)(hi - lo); } else t->reps[r] = sm / (double)t->tn;
    }
    free(p); return NULL;
}
static void bootCI(const double *const *g, const size_t *sz, size_t ng, bool iqm, double *lo, double *hi) {
    size_t tn = 0u; for (size_t i = 0u; i < ng; i++) tn += sz[i]; double *reps = xmal(OAK_BOOT * sizeof *reps);
    long nc = sysconf(_SC_NPROCESSORS_ONLN); size_t nt = (size_t)(nc < 1 ? 1 : nc), ch = (OAK_BOOT + nt - 1u) / nt; pthread_t *ts = xcal(nt, sizeof *ts); BTask *tk = xcal(nt, sizeof *tk);
    for (size_t i = 0u; i < nt; i++) { size_t s = i * ch, e = s + ch > OAK_BOOT ? OAK_BOOT : s + ch; tk[i] = (BTask){ g, sz, ng, tn, iqm, reps, s, e }; pthread_create(&ts[i], NULL, bootWork, &tk[i]); }
    for (size_t i = 0u; i < nt; i++) pthread_join(ts[i], NULL); free(ts); free(tk); qsort(reps, OAK_BOOT, sizeof *reps, cmpD); *lo = qSorted(reps, OAK_BOOT, 0.025); *hi = qSorted(reps, OAK_BOOT, 0.975); free(reps);
}
typedef struct { char **c; size_t nr, nc; } Tab;
static void tabInit(Tab *t, size_t nr, size_t nc) { t->nr = nr; t->nc = nc; t->c = xcal(nr * nc, sizeof *t->c); }
static void tabSet(Tab *t, size_t r, size_t c, const char *s) { t->c[r * t->nc + c] = xdup(s); }
static void tabPrint(const char *const *h, Tab *t) {
    size_t *w = xcal(t->nc, sizeof *w); for (size_t c = 0u; c < t->nc; c++) { w[c] = strlen(h[c]); for (size_t r = 0u; r < t->nr; r++) { size_t l = strlen(t->c[r * t->nc + c]); if (l > w[c]) w[c] = l; } }
    for (size_t c = 0u; c < t->nc; c++) printf("%s%-*s", c ? " | " : "", (int)w[c], h[c]); putchar('\n');
    for (size_t r = 0u; r < t->nr; r++) { for (size_t c = 0u; c < t->nc; c++) printf(c ? " | %*s" : "%-*s", (int)w[c], t->c[r * t->nc + c]); putchar('\n'); }
    free(w); for (size_t i = 0u; i < t->nr * t->nc; i++) free(t->c[i]); free(t->c);
}

typedef struct { pid_t pid; char stat[16], name[64]; } PInfo;
static size_t listPs(PInfo **out) {
    char *av[] = { "ps","-ax","-o","pid=,stat=,comm=",NULL }, *b = NULL; if (runProc(av, NULL, &b, 1, true)) die("ps");
    size_t cap = 256u, n = 0u; PInfo *a = xmal(cap * sizeof *a);
    for (const char *p = b, *nl; p && *p; p = nl ? nl + 1 : NULL) { nl = strchr(p, '\n'); char st[16], cm[PATH_MAX]; long pid = 0; if (sscanf(p, "%ld %15s %[^\n]", &pid, st, cm) == 3) { if (n == cap) a = xrea(a, (cap *= 2u) * sizeof *a); a[n].pid = (pid_t)pid; snprintf(a[n].stat, 16u, "%s", st); snprintf(a[n++].name, 64u, "%s", baseOf(cm)); } }
    free(b); *out = a; return n;
}
static size_t quietPause(pid_t pd[64]) { if (HAS_ACTIVE && isKind(pf("%s/cancel", ACTIVE), 0)) die("CANCELLED"); PInfo *ps = NULL; size_t n = listPs(&ps), k = 0u; for (size_t i = 0u; i < n && k < 64u; i++) if (!strcmp(ps[i].name, "oak") && ps[i].stat[0] != 'T' && !kill(ps[i].pid, SIGSTOP)) pd[k++] = ps[i].pid; free(ps); return k; }
static void quietResume(const pid_t *p, size_t n) { for (size_t i = 0u; i < n; i++) kill(p[i], SIGCONT); }
static double probeOnce(const char *eng, const char *tag, uint32_t idx, uint64_t **st, size_t *sc) {
    char root[PATH_MAX], *o = NULL; snprintf(root, sizeof root, "%s/probe-%s-%u", BENCH, tag, idx); rmRf(root); pid_t pd[64]; size_t np = quietPause(pd), hl = 0u;
    char *av[] = { "caffeinate","-i",(char *)eng,"record","oak",pf("%u", OAK_PROBE_STEPS),root,NULL }; int rc = runProc(av, REPO, &o, 1, false); quietResume(pd, np); if (rc) die("probe %s", eng);
    uint64_t *hs = (uint64_t *)readAll(pf("%s/hash.u64", root), &hl); rmRf(root); double sps = strtod(strrchr(o, ':') + 1, NULL); free(o); if (st) { *st = hs; *sc = hl / 8u; } else free(hs); return sps;
}
static void refEngine(const Rec *ref, char out[PATH_MAX]) {
    char sd[PATH_MAX], ss[PATH_MAX], dg[65], kh[65], tr[PATH_MAX]; snprintf(sd, sizeof sd, "%s/source", ref->root); snprintf(ss, sizeof ss, "%s/Sources", sd); if (!isKind(ss, S_IFDIR)) die("Reference lacks source snapshot: %s", baseOf(ref->root));
    const char *en[] = { "Sources" }; digestEntries(sd, en, 1u, dg); char *tc = tcFingerprint(); CC_SHA256_CTX hc; uint8_t hr[32];
    CC_SHA256_Init(&hc); CC_SHA256_Update(&hc, dg, (CC_LONG)strlen(dg)); CC_SHA256_Update(&hc, tc, (CC_LONG)strlen(tc)); CC_SHA256_Update(&hc, "thin90", 6u); CC_SHA256_Final(hr, &hc); free(tc); hexEnc(hr, 32u, kh); kh[16] = '\0';
    snprintf(tr, sizeof tr, "%s/.build/reference-%s", REPO, kh); if (!isKind(tr, S_IFDIR)) { char st[PATH_MAX]; snprintf(st, sizeof st, "%s.partial", tr); rmRf(st); copyTree(ss, pf("%s/Sources", st)); buildTree(st, false); if (rename(st, tr)) die("rename ref"); }
    snprintf(out, PATH_MAX, "%s/.build/oak", tr);
}
static double speedStage(const char *eng, const char *oth, const Rec *ref, bool cmp) {
    char d1[65], d2[65], kh[65], p[PATH_MAX]; const char *rn = baseOf(ref->root); shaFile(eng, d1); shaFile(oth, d2); const char *cb = pf("%s%s", d1, d2); shaHex(cb, strlen(cb), kh); kh[40] = '\0';
    snprintf(p, sizeof p, "%s/speed/%s.json", BENCH, kh); double ratio = 1.0; size_t div = 0u;
    if (isKind(p, S_IFREG)) { JV *j = jFile(p); ratio = jGet(j, "ratio")->d; div = (size_t)jGet(j, "diverged")->d; printf("speed stage already measured at %s for this binary pair\n", jGet(j, "at")->s); if (cmp && div) printf("reward streams diverge from recorded %.9s on %zu lanes within %u steps\n", rn, div, OAK_PROBE_STEPS); jFree(j); }
    else if (filesEq(eng, oth)) printf("binary identical to recorded %.9s; no probe\n", rn);
    else {
        printf("timing against recorded %.9s\n", rn); fflush(stdout); double rs[OAK_PROBE_BLOCKS];
        for (uint32_t b = 0u; b < OAK_PROBE_BLOCKS; b++) {
            uint64_t *sa = NULL, *sb = NULL; size_t na = 0u, nb = 0u; double a = probeOnce(eng, "candidate", 2u * b, cmp && !b ? &sa : NULL, &na), x = probeOnce(oth, "reference", 2u * b, cmp && !b ? &sb : NULL, &nb);
            if (cmp && !b) { for (size_t i = 0u; i < (na < nb ? na : nb); i++) if (sa[i] != sb[i]) div++; if (div) printf("reward streams diverge from recorded %.9s on %zu of %zu lanes within %u steps\n", rn, div, na, OAK_PROBE_STEPS); free(sa); free(sb); }
            double c = probeOnce(oth, "reference", 2u * b + 1u, NULL, NULL), d = probeOnce(eng, "candidate", 2u * b + 1u, NULL, NULL);
            rs[b] = (a + d) / (x + c); printf("speed block %u: candidate %s  reference %s  ratio %.4f\n", b + 1u, fmtCommas((a + d) / 2.0), fmtCommas((x + c) / 2.0), rs[b]); fflush(stdout);
        }
        qsort(rs, OAK_PROBE_BLOCKS, sizeof(double), cmpD); ratio = qSorted(rs, OAK_PROBE_BLOCKS, 0.5); mkdirP(pf("%s/speed", BENCH)); char ts[32]; stampNow(ts);
        const char *js = pf("{\n \"at\": \"%s\",\n \"ratio\": %.17g,\n \"diverged\": %zu\n}\n", ts, ratio, div); writeAll(p, js, strlen(js));
    }
    printf("paired speed against recorded %.9s: %.4fx\n", rn, ratio); fflush(stdout); return ratio;
}
static void snapTree(const char *dstIn, const char *srcIn) { char dst[PATH_MAX], src[PATH_MAX]; snprintf(dst, sizeof dst, "%s", dstIn); snprintf(src, sizeof src, "%s", srcIn); rmRf(dst); mkdirP(dst); copyTree(pf("%s/Sources", src), pf("%s/Sources", dst)); if (isKind(pf("%s/fleet.c", src), S_IFREG)) copyFast(pf("%s/fleet.c", src), pf("%s/fleet.c", dst)); }
static void keepEng(const char *eng, const char *root) { char st[PATH_MAX]; snprintf(st, sizeof st, "%s/.oak.partial", root); copyFast(eng, st); chmod(st, 0755); if (rename(st, pf("%s/oak", root))) die("keepEng"); }
static void runJob(const char *dir); static bool pendingJob(bool timing, char out[PATH_MAX]);
static void supervise(char *const av[]) {
    pid_t pid = spawnProc(av, REPO, -1, -1, -1, true, true), cpid = 0; char *ca[] = { "caffeinate","-i","-w",pf("%d", (int)pid),NULL }; posix_spawnp(&cpid, "caffeinate", NULL, NULL, ca, environ);
    if (HAS_ACTIVE) { const char *l = pf("%d\n", (int)pid); writeAll(pf("%s/pid", ACTIVE), l, strlen(l)); } int st = 0;
    for (char nxt[PATH_MAX]; waitpid(pid, &st, WNOHANG) != pid; ) {
        if (!pendingJob(true, nxt)) { sleep(2); continue; }
        for (bool stopped = false; !waitpid(pid, &st, WNOHANG) && !stopped; ) { PInfo *pi = NULL; size_t np = listPs(&pi); for (size_t i = 0u; i < np; i++) if (pi[i].pid == pid && pi[i].stat[0] == 'T') stopped = true; free(pi); if (!stopped && kill(pid, SIGTSTP)) break; if (!stopped) usleep(50000); }
        char ts[32]; stampNow(ts); printf("%s paused for probe %s\n", ts, baseOf(nxt)); fflush(stdout); runJob(nxt); stampNow(ts); printf("%s resumed\n", ts); fflush(stdout); kill(pid, SIGCONT);
    }
    if (cpid > 0) waitpid(cpid, NULL, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) die("engine exit %d", st);
}
static void writeSrcJ(const char *p, const char *pol, const EMeta *es, size_t n, const char *pd) { EMeta *s = xmal(n * sizeof *s); memcpy(s, es, n * sizeof *s); qsort(s, n, sizeof *s, cmpEM); FILE *f = fopen(p, "wb"); if (!f) die("write %s", p); fputs("{\n \"envs\": {\n", f); for (size_t i = 0u; i < n; i++) fprintf(f, "  \"%s\": \"%s\"%s\n", s[i].id, s[i].dig, i + 1u < n ? "," : ""); fprintf(f, " },\n \"policy\": \"%s\",\n \"policyDigest\": \"%s\"\n}\n", pol, pd); fclose(f); free(s); }
static Rec *recordRun(const char *eng, const char *pol, const char *rootIn, const EMeta *es, size_t n, const char *pd, const char *tree, const char *const *only, size_t no) {
    char root[PATH_MAX], st[PATH_MAX]; snprintf(root, sizeof root, "%s", rootIn); printf("Recording %s into %s\n", pol, baseOf(root)); fflush(stdout); rmRf(root); snprintf(st, sizeof st, "%s.source", root); snapTree(st, tree);
    char **av = xmal((no + 6u) * sizeof *av); size_t a = 0u; av[a++] = (char *)eng; av[a++] = "record"; av[a++] = (char *)pol; av[a++] = pf("%u", OAK_STEPS); av[a++] = root; for (size_t i = 0u; i < no; i++) av[a++] = (char *)only[i]; av[a] = NULL;
    supervise(av); free(av); writeSrcJ(pf("%s/sources.json", root), pol, es, n, pd); if (rename(st, pf("%s/source", root))) die("rename src"); keepEng(eng, root); return loadRec(root);
}
typedef struct { RCache c; size_t *idx; } BSet;
static BSet ensureBase(const char *pol, const EMeta *es, size_t n, const char *eng, const char *tree) {
    char pd[65], h[65]; const char *pfile = !strcmp(pol, "random") ? "Sources/random.c" : !strcmp(pol, "ppo") ? "Sources/ppo.c" : NULL; digestEntries(tree, pfile ? &pfile : NULL, pfile ? 1u : 0u, pd);
    char **fn = xcal(n, sizeof *fn), **mid = xcal(n, sizeof *mid); size_t *mi = xmal(n * sizeof *mi), nm = 0u;
    for (size_t i = 0u; i < n; i++) { const char *bj = pf("{\"digest\": \"%s\", \"env\": \"%s\", \"policy\": \"%s\", \"policyDigest\": \"%s\", \"seeds\": [%u, %u], \"steps\": %u}", es[i].dig, es[i].id, pol, pd, es[i].s0, es[i].s1, OAK_STEPS); shaHex(bj, strlen(bj), h); h[40] = '\0'; const char *lp = pf("%s/baselines/%s/%s", BENCH, pol, h); if (isKind(lp, S_IFREG)) fn[i] = readLine(lp); else { mi[nm] = i; mid[nm++] = xdup(h); } }
    if (nm) {
        SV s = {0}; for (size_t i = 0u; i < nm; i++) svAdd(&s, xdup(mid[i])); svSort(&s); char *jb = NULL, bn[128]; size_t jl = 0u; FILE *m = open_memstream(&jb, &jl); fputc('[', m); for (size_t i = 0u; i < s.n; i++) fprintf(m, "%s\"%s\"", i ? ", " : "", s.v[i]); fputc(']', m); fclose(m);
        shaHex(jb, jl, h); h[40] = '\0'; free(jb); svFree(&s); snprintf(bn, sizeof bn, "%s-%s", pol, h); EMeta *se = xmal(nm * sizeof *se); const char **sn = xmal(nm * sizeof *sn); for (size_t i = 0u; i < nm; i++) { se[i] = es[mi[i]]; sn[i] = se[i].id; }
        freeRec(recordRun(eng, pol, pf("%s/%s", BENCH, bn), se, nm, pd, tree, sn, nm)); free(se); free(sn); mkdirP(pf("%s/baselines/%s", BENCH, pol));
        for (size_t i = 0u; i < nm; i++) { const char *l = pf("%s\n", bn); writeAll(pf("%s/baselines/%s/%s", BENCH, pol, mid[i]), l, strlen(l)); fn[mi[i]] = xdup(bn); }
    } else { printf("Reusing %s for all %zu environments\n", pol, n); fflush(stdout); }
    for (size_t i = 0u; i < nm; i++) free(mid[i]); free(mid); free(mi); BSet bs = { .idx = xcal(n, sizeof(size_t)) };
    for (size_t i = 0u; i < n; i++) { Rec *rec = rcGet(&bs.c, fn[i]); size_t sl = 0u; while (strcmp(bs.c.e[sl].nm, fn[i])) sl++; JV *sr = rSrc(rec), *se = jGet(jGet(sr, "envs"), es[i].id); const EBlk *b = rBlk(rec, es[i].id); if (!b || strcmp(rec->pol, pol) || rec->steps != OAK_STEPS || strcmp(jGet(sr, "policyDigest")->s, pd) || !se || strcmp(se->s, es[i].dig) || b->s0 != es[i].s0 || b->s1 != es[i].s1) die("Baseline mismatch %s %s: %s", pol, es[i].id, fn[i]); bs.idx[i] = sl; free(fn[i]); }
    free(fn); return bs;
}

typedef struct { double lo, hi; bool guard, ident; } Verdict;
static Verdict reportGate(const Rec *c, const BSet *rb, const BSet *pb, const BSet *ob, const Rec *best) {
    size_t ne = c->nBlk, totD = 0u; const char *hB[] = {"env","oak","random","oracle","score","ppo","win>ppo","best","d","win>best","guard p","rho","mdr"}, *hN[] = {"env","oak","random","oracle","score","ppo","win>ppo","mdr"};
    Tab tab; tabInit(&tab, ne, best ? 13u : 8u); double **dG = xcal(ne, sizeof *dG), **wG = xcal(ne, sizeof *wG), *pv = xcal(ne, sizeof *pv), *wb = xcal(ne, sizeof *wb); size_t *gS = xcal(ne, sizeof *gS);
    for (size_t e = 0u; e < ne; e++) {
        const EBlk *b = &c->blks[e]; size_t n = b->s1 - b->s0 + 1u; const double *oak = c->score + b->base;
        const Rec *rr = rb->c.e[rb->idx[e]].r, *pr = pb->c.e[pb->idx[e]].r, *orR = ob->c.e[ob->idx[e]].r;
        const double *rnd = rr->score + rBlk(rr, b->id)->base, *ppo = pr->score + rBlk(pr, b->id)->base, *ora = orR->score + rBlk(orR, b->id)->base;
        double mO = meanA(oak, n), mR = meanA(rnd, n), mOr = meanA(ora, n);
        tabSet(&tab, e, 0u, b->id); tabSet(&tab, e, 1u, fmtVal(mO)); tabSet(&tab, e, 2u, fmtVal(mR)); tabSet(&tab, e, 3u, fmtVal(mOr)); tabSet(&tab, e, 4u, fmtVal((mO - mR) / (mOr - mR))); tabSet(&tab, e, 5u, fmtVal(meanA(ppo, n))); tabSet(&tab, e, 6u, fmtVal(winsA(oak, ppo, n)));
        if (best) {
            const double *ref = best->score + rBlk(best, b->id)->base; double sc = scaleA(ref, n), *df = xmal(n * sizeof(double)); dG[e] = xmal(n * sizeof(double)); wG[e] = xmal(n * sizeof(double));
            for (size_t i = 0u; i < n; i++) { df[i] = oak[i] - ref[i]; dG[e][i] = df[i] / sc; wG[e][i] = df[i] > 0.0 ? 1.0 : df[i] == 0.0 ? 0.5 : 0.0; }
            gS[e] = n; totD += n; double p = wilcoxonLess(df, n), rh = rhoOf(oak, ref, n), w = winsA(oak, ref, n); pv[e] = p; wb[e] = w; free(df);
            tabSet(&tab, e, 7u, fmtVal(meanA(ref, n))); tabSet(&tab, e, 8u, fmtVal(iqmA(dG[e], n))); tabSet(&tab, e, 9u, fmtVal(w)); tabSet(&tab, e, 10u, fmtVal(p)); tabSet(&tab, e, 11u, fmtVal(rh)); tabSet(&tab, e, 12u, fmtVal(mdrOf(rh, ne, n)));
        } else tabSet(&tab, e, 7u, fmtVal(mdrOf(0.0, ne, n)));
    }
    tabPrint(best ? hB : hN, &tab); uint32_t dead = 0u; for (uint32_t l = 0u; l < c->lanes; l++) if (c->health[l] != 0xffffffffu) dead++; printf("dead lanes: %u of %u\n", dead, c->lanes);
    Verdict v = {0};
    if (best) {
        double *allD = xmal(totD * sizeof *allD), plo = 0.0, phi = 0.0; for (size_t e = 0u, pos = 0u; e < ne; e++) { memcpy(allD + pos, dG[e], gS[e] * sizeof(double)); pos += gS[e]; }
        double suite = iqmA(allD, totD); free(allD); bootCI((const double *const *)dG, gS, ne, true, &v.lo, &v.hi); bootCI((const double *const *)wG, gS, ne, false, &plo, &phi);
        bool *rej = xcal(ne, 1u); holm(pv, ne, rej); char *fd = NULL; size_t fl = 0u; FILE *fm = open_memstream(&fd, &fl);
        for (size_t e = 0u; e < ne; e++) if (rej[e]) { if (v.guard) fputs(", ", fm); fputs(c->blks[e].id, fm); v.guard = true; } fclose(fm);
        v.ident = true; for (size_t e = 0u; e < ne && v.ident; e++) for (uint32_t s = c->blks[e].s0; s <= c->blks[e].s1; s++) { int cl = rLane(c, c->blks[e].id, s), bl = rLane(best, c->blks[e].id, s); if (cl < 0 || bl < 0 || c->hash[cl] != best->hash[bl]) { v.ident = false; break; } }
        printf("suite IQM d: %+.4f  95%% CI [%+.4f, %+.4f]   P(improve): %.3f [%.3f, %.3f]\nregression guard (Holm %g): %s%s\nreward streams: %s\n", suite, v.lo, v.hi, meanA(wb, ne), plo, phi, OAK_ALPHA, v.guard ? "fired on " : "silent", v.guard ? fd : "", v.ident ? "identical" : "different");
        free(fd); free(rej);
    }
    fflush(stdout); for (size_t e = 0u; e < ne; e++) { free(dG[e]); free(wG[e]); } free(dG); free(wG); free(gS); free(pv); free(wb); return v;
}

static Rec *tryRun(const char *engIn, const char *rootIn, uint32_t f, uint32_t cnt, const char *tree) {
    char eng[PATH_MAX], root[PATH_MAX]; snprintf(eng, sizeof eng, "%s", engIn); snprintf(root, sizeof root, "%s", rootIn);
    if (recDone(root)) { printf("already tried: %.9s (seeds %u..%u)\n", baseOf(root), f, f + cnt - 1u); fflush(stdout); }
    else { rmRf(root); char st[PATH_MAX]; if (tree) { snprintf(st, sizeof st, "%s.source", root); snapTree(st, tree); } pid_t pd[64]; size_t np = quietPause(pd); char *av[] = { "caffeinate","-i",eng,"try","oak",pf("%u", OAK_STEPS),pf("%u", f),pf("%u", cnt),root,NULL }; int rc = runProc(av, REPO, NULL, 0, false); quietResume(pd, np); if (rc) die("try failed"); if (tree) { if (rename(st, pf("%s/source", root))) die("rename"); keepEng(eng, root); } }
    return loadRec(root);
}
typedef struct { char *env, *cat; double c, r, b, p, d; bool same; } TRow;
static int cmpTR(const void *a, const void *b) { const TRow *x = a, *y = b; int c = strcmp(x->cat, y->cat); return c ? c : strcmp(x->env, y->env); }
static bool lookupM(Rec *const *rs, size_t n, const char *e, uint32_t s, double *sc, uint64_t *h) { for (size_t i = 0u; i < n; i++) { int l = rLane(rs[i], e, s); if (l >= 0) { *sc = rs[i]->score[l]; *h = rs[i]->hash[l]; return true; } } return false; }
typedef struct { char ver[32]; bool closer; size_t w, l; } TSum;
static TSum cmpTries(const EMeta *mt, size_t ne, const Rec *best, Rec *const *cs, size_t nc, Rec *const *rs, size_t nr) {
    JV *lk = jFile(pf("%s/baselines.json", best->root)); RCache rc = {0}; TRow *tr = xcal(ne, sizeof *tr);
    for (size_t e = 0u; e < ne; e++) {
        const char *env = cs[0]->blks[e].id, *cat = "unknown"; for (size_t m = 0u; m < ne; m++) if (!strcmp(mt[m].id, env)) { cat = mt[m].cat; break; }
        Rec *rr = rcGet(&rc, jGet(jGet(lk, "random"), env)->s), *orR = rcGet(&rc, jGet(jGet(lk, "oracle"), env)->s), *pr = rcGet(&rc, jGet(jGet(lk, "ppo"), env)->s);
        const EBlk *rb = rBlk(rr, env), *ob = rBlk(orR, env); double lo = meanA(rr->score + rb->base, rb->s1 - rb->s0 + 1u), den = meanA(orR->score + ob->base, ob->s1 - ob->s0 + 1u) - lo, sc = 0, sr = 0, sb = 0, sp = 0; size_t nk = 0u; bool same = true;
        for (size_t ci = 0u; ci < nc; ci++) for (uint32_t s = rBlk(cs[ci], env)->s0; s <= rBlk(cs[ci], env)->s1; s++) { double cs2 = 0.0, rs2 = 0.0; uint64_t ch = 0u, rh = 0u; if (!lookupM(cs, nc, env, s, &cs2, &ch) || !lookupM(rs, nr, env, s, &rs2, &rh)) die("missing lane"); if (ch != rh) same = false; sc += (cs2 - lo) / den; sr += (rs2 - lo) / den; sb += (best->score[rLane(best, env, s)] - lo) / den; sp += (pr->score[rLane(pr, env, s)] - lo) / den; nk++; }
        tr[e] = (TRow){ xdup(env), xdup(cat), sc / (double)nk, sr / (double)nk, sb / (double)nk, sp / (double)nk, same ? 0.0 : (sc - sr) / (double)nk, same };
    }
    jFree(lk); rcFree(&rc); qsort(tr, ne, sizeof *tr, cmpTR);
    const char *h1[] = {"env","category","oak","ref","best","ppo","d"}, *h2[] = {"group","n","oak","ref","best","ppo","wins","losses","p better","p worse"};
    Tab t1; tabInit(&t1, ne, 7u); for (size_t e = 0u; e < ne; e++) { tabSet(&t1, e, 0u, tr[e].env); tabSet(&t1, e, 1u, tr[e].cat); tabSet(&t1, e, 2u, fmtVal(tr[e].c)); tabSet(&t1, e, 3u, fmtVal(tr[e].r)); tabSet(&t1, e, 4u, fmtVal(tr[e].b)); tabSet(&t1, e, 5u, fmtVal(tr[e].p)); tabSet(&t1, e, 6u, tr[e].same ? "=" : fmtVal(tr[e].d)); } tabPrint(h1, &t1);
    SV cats = {0}; for (size_t e = 0u; e < ne; e++) { bool seen = false; for (size_t i = 0u; i < cats.n; i++) if (!strcmp(cats.v[i], tr[e].cat)) seen = true; if (!seen) svAdd(&cats, xdup(tr[e].cat)); }
    svSort(&cats); size_t ng = 1u + cats.n, sw = 0u, sl = 0u; double *pB = xcal(ng, sizeof *pB), *pW = xcal(ng, sizeof *pW); Tab t2; tabInit(&t2, ng, 10u);
    for (size_t g = 0u; g < ng; g++) {
        const char *gn = g ? cats.v[g - 1u] : "suite"; size_t cnt = 0u, w = 0u, l = 0u; double sc = 0, sr = 0, sb = 0, sp = 0;
        for (size_t e = 0u; e < ne; e++) if (!g || !strcmp(tr[e].cat, gn)) { cnt++; sc += tr[e].c; sr += tr[e].r; sb += tr[e].b; sp += tr[e].p; if (tr[e].d > 0.0) w++; else if (tr[e].d < 0.0) l++; }
        pB[g] = signTail(w, l); pW[g] = signTail(l, w); if (!g) { sw = w; sl = l; }
        tabSet(&t2, g, 0u, gn); tabSet(&t2, g, 1u, pf("%zu", cnt)); tabSet(&t2, g, 2u, fmtVal(sc / (double)cnt)); tabSet(&t2, g, 3u, fmtVal(sr / (double)cnt)); tabSet(&t2, g, 4u, fmtVal(sb / (double)cnt)); tabSet(&t2, g, 5u, fmtVal(sp / (double)cnt)); tabSet(&t2, g, 6u, pf("%zu", w)); tabSet(&t2, g, 7u, pf("%zu", l)); tabSet(&t2, g, 8u, fmtVal(pB[g])); tabSet(&t2, g, 9u, fmtVal(pW[g]));
    }
    putchar('\n'); tabPrint(h2, &t2); svFree(&cats);
    double *cv = xmal(ne * sizeof(double)), *rv = xmal(ne * sizeof(double)), cS[3], rS[3]; size_t cB = 0u, cA = 0u, rB = 0u, rA = 0u; bool allS = true;
    for (size_t e = 0u; e < ne; e++) { cv[e] = tr[e].c; rv[e] = tr[e].r; if (tr[e].c < 0.0) cB++; if (tr[e].c > tr[e].p) cA++; if (tr[e].r < 0.0) rB++; if (tr[e].r > tr[e].p) rA++; if (!tr[e].same) allS = false; }
    cS[0] = meanA(cv, ne); cS[1] = iqmA(cv, ne); qsort(cv, ne, sizeof(double), cmpD); cS[2] = qSorted(cv, ne, 0.5); rS[0] = meanA(rv, ne); rS[1] = iqmA(rv, ne); qsort(rv, ne, sizeof(double), cmpD); rS[2] = qSorted(rv, ne, 0.5); free(cv); free(rv);
    printf("oak: mean %+.4f  IQM %+.4f  median %+.4f  below Random %zu  above PPO %zu\nref: mean %+.4f  IQM %+.4f  median %+.4f  below Random %zu  above PPO %zu\n", cS[0], cS[1], cS[2], cB, cA, rS[0], rS[1], rS[2], rB, rA);
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN); if (ncpu < 1) ncpu = 1;
    for (size_t i = 0u; i < nc; i++) { const Rec *r = cs[i]; uint32_t dead = 0u; for (uint32_t l = 0u; l < r->lanes; l++) if (r->health[l] != 0xffffffffu) dead++; double cor = (double)ncpu < (double)r->lanes ? (double)ncpu : (double)r->lanes; printf("%.12s: dead lanes %u of %u   %s steps/s   %.3f us per step per core   %.1f s\n", baseOf(r->root), dead, r->lanes, fmtCommas(r->sps), cor / r->sps * 1e6, (double)r->lanes * (double)r->steps / r->sps); }
    fflush(stdout); bool *rW = xcal(ng, 1u), *rB2 = xcal(ng, 1u), aW = false, aB = false; holm(pW, ng, rW); holm(pB, ng, rB2); for (size_t g = 0u; g < ng; g++) { if (rW[g]) aW = true; if (rB2[g]) aB = true; }
    free(rW); free(rB2); free(pB); free(pW); for (size_t e = 0u; e < ne; e++) { free(tr[e].env); free(tr[e].cat); } free(tr);
    TSum ts = { .closer = cS[0] > rS[0] && cS[1] > rS[1] && cS[2] > rS[2], .w = sw, .l = sl }; strcpy(ts.ver, allS ? "IDENTICAL" : aW ? "WORSE" : aB ? "BETTER" : "NO SIGNAL"); return ts;
}
static void attemptTry(const char *eng, const char *tree, const EMeta *mt, size_t ne, const Rec *best, char out[256]) {
    mkdirP(pf("%s/tries", BENCH)); char key[65], ch[64] = {0}, lbl[64], cp[PATH_MAX], ver[64]; shaFile(eng, key); key[40] = '\0'; snprintf(cp, sizeof cp, "%s/tries/champion", BENCH);
    Rec *cand = tryRun(eng, pf("%s/tries/%s", BENCH, key), 0u, 1u, tree), *refO = NULL; const Rec *ref = best;
    if (isKind(cp, S_IFREG)) { char *t = readLine(cp); snprintf(ch, sizeof ch, "%s", t); free(t); ref = refO = loadRec(pf("%s/tries/%s", BENCH, ch)); snprintf(lbl, sizeof lbl, "champion %.9s", ch); } else snprintf(lbl, sizeof lbl, "best %.9s", baseOf(best->root));
    Rec *cA[2] = { cand, NULL }, *rA[2] = { (Rec *)ref, NULL }; TSum s = cmpTries(mt, ne, best, cA, 1u, rA, 1u); snprintf(ver, sizeof ver, "%s", s.ver);
    if (!strcmp(s.ver, "NO SIGNAL") && s.closer) {
        printf("\ntie-break: mean, IQM and median all improved; adding seeds 1..%u of every block\n", OAK_TIEBREAK - 1u); fflush(stdout);
        Rec *w = tryRun(eng, pf("%s/tries/%s+1+%u", BENCH, key, OAK_TIEBREAK - 1u), 1u, OAK_TIEBREAK - 1u, NULL), *rw = NULL; size_t nr = 1u;
        if (ch[0]) { rA[1] = rw = tryRun(pf("%s/tries/%s/oak", BENCH, ch), pf("%s/tries/%s+1+%u", BENCH, ch, OAK_TIEBREAK - 1u), 1u, OAK_TIEBREAK - 1u, NULL); nr = 2u; }
        cA[1] = w; s = cmpTries(mt, ne, best, cA, 2u, rA, nr); snprintf(ver, sizeof ver, "%s on %u seeds", s.ver, OAK_TIEBREAK); freeRec(w); freeRec(rw);
    }
    if (!strncmp(ver, "IDENTICAL", 9u) || !strncmp(ver, "BETTER", 6u)) { const char *l = pf("%s\n", key); writeAll(cp, l, strlen(l)); }
    freeRec(cand); freeRec(refO); snprintf(out, 256u, "%s: try %.9s against %s; suite %zu wins, %zu losses", ver, key, lbl, s.w, s.l);
}

typedef struct { char nm[128]; size_t n; } TFunc;
static int cmpTF(const void *a, const void *b) { size_t x = ((const TFunc *)a)->n, y = ((const TFunc *)b)->n; return y > x ? 1 : y < x ? -1 : 0; }
static void addCnt(TFunc *a, size_t *n, size_t cap, const char *nm, size_t v) { size_t i = 0u; while (i < *n && strcmp(a[i].nm, nm)) i++; if (i < *n) a[i].n += v; else if (*n < cap) { snprintf(a[*n].nm, sizeof a[*n].nm, "%s", nm); a[(*n)++].n = v; } }
static const char *xmlAttr(const char *s, const char *e, const char *k, size_t *len) { size_t kl = strlen(k); for (const char *p = s; p < e; p++) if ((size_t)(e - p) > kl + 2u && !memcmp(p, k, kl) && p[kl] == '=' && (p[kl + 1u] == '"' || p[kl + 1u] == '\'') && (p == s || p[-1] == ' ' || p[-1] == '\t')) { const char *v = p + kl + 2u, *q = memchr(v, p[kl + 1u], (size_t)(e - v)); if (q) { *len = (size_t)(q - v); return v; } } return NULL; }
static uint32_t xmlId(const char *s, const char *e, bool *hi, bool *hr, uint32_t *rf) { size_t l = 0u; const char *a = xmlAttr(s, e, "id", &l), *b = xmlAttr(s, e, "ref", &l); *hi = a != NULL; *hr = b != NULL; *rf = b ? (uint32_t)strtoul(b, NULL, 10) : 0u; return a ? (uint32_t)strtoul(a, NULL, 10) : 0u; }
static void traceEng(const char *eng) {
    char rt[PATH_MAX], rp[PATH_MAX], *xml = NULL, *nmo = NULL; snprintf(rt, sizeof rt, "%s/probe-trace", BENCH); snprintf(rp, sizeof rp, "%s/trace.trace", BENCH); rmRf(rt); rmRf(rp);
    pid_t pd[64]; size_t np = quietPause(pd), ns = 0u, cs = 0u, rows = 0u, nf = 0u, np2 = 0u;
    char *ra[] = { (char *)eng,"record","oak",pf("%u", OAK_PROBE_STEPS),rt,NULL }; pid_t pid = spawnProc(ra, REPO, -1, -2, -1, false, true); while (!isKind(pf("%s/meta.json", rt), 0)) usleep(100000);
    char *xa[] = { "xcrun","xctrace","record","--template","Time Profiler","--attach",pf("%d", (int)pid),"--time-limit",pf("%us", OAK_TRACE_SEC),"--output",rp,NULL };
    int rc = runProc(xa, REPO, NULL, 0, true); waitpid(pid, NULL, 0); quietResume(pd, np); rmRf(rt); if (rc) { rmRf(rp); die("xctrace failed"); }
    char *ea[] = { "xcrun","xctrace","export","--input",rp,"--xpath","/trace-toc/run[@number=\"1\"]/data/table[@schema=\"time-profile\"]",NULL }, *na[] = { "nm","-n",(char *)eng,NULL };
    rc = runProc(ea, REPO, &xml, 1, true); rmRf(rp); if (rc) die("xctrace export");
    typedef struct { uint64_t a; char nm[96]; } Sym; Sym *syms = NULL;
    if (!runProc(na, REPO, &nmo, 1, false)) { for (char *p = nmo, *nl; *p; p = nl ? nl + 1 : p + strlen(p)) { nl = strchr(p, '\n'); char kd = 0, nm[96] = {0}; uint64_t a = 0u; if (nl) *nl = '\0'; if (sscanf(p, "%llx %c %95s", &a, &kd, nm) == 3 && (kd == 't' || kd == 'T')) { if (ns == cs) syms = xrea(syms, (cs = cs ? cs * 2u : 512u) * sizeof *syms); syms[ns].a = a; snprintf(syms[ns++].nm, 96u, "%s", nm); } } free(nmo); }
    enum { IDMAX = 1u << 21 }; typedef struct { char nm[96]; uint64_t addr; uint32_t bin, cnt; } Fr;
    Fr *fr = xcal(IDMAX, sizeof *fr); uint32_t *bt = xcal(IDMAX, sizeof *bt), curFr = 0u, curBt = 0u, rowBt = 0u; uint64_t *bLd = xcal(IDMAX, sizeof *bLd); uint8_t *bOak = xcal(IDMAX, 1u); bool inBt = false, inRow = false;
    for (const char *p = xml; (p = strchr(p, '<')) != NULL; p++) {
        const char *gt = strchr(p, '>'); if (!gt) break; bool hi = false, hr = false; uint32_t rf = 0u; size_t nl = 0u, al = 0u;
        if (!strncmp(p, "<row>", 5) || !strncmp(p, "<row ", 5)) { inRow = true; rowBt = 0u; }
        else if (!strncmp(p, "</row>", 6)) { if (inRow && rowBt && rowBt < IDMAX && bt[rowBt] && bt[rowBt] < IDMAX) { fr[bt[rowBt]].cnt++; rows++; } inRow = false; rowBt = 0u; }
        else if (!strncmp(p, "</backtrace>", 12) || !strncmp(p, "</tagged-backtrace>", 19)) { inBt = false; curBt = 0u; } else if (!strncmp(p, "</frame>", 8)) curFr = 0u;
        else if (!strncmp(p, "<binary ", 8)) { uint32_t id = xmlId(p, gt, &hi, &hr, &rf); if (hi && id < IDMAX) { const char *nm = xmlAttr(p, gt, "name", &nl), *la = xmlAttr(p, gt, "load-addr", &al); if (nm && nl == 3u && !memcmp(nm, "oak", 3)) bOak[id] = 1u; if (la) bLd[id] = strtoull(la, NULL, 16); } if (curFr && (hi || hr)) fr[curFr].bin = hi ? id : rf; }
        else if (!strncmp(p, "<frame ", 7)) { uint32_t id = xmlId(p, gt, &hi, &hr, &rf), fid = hi ? id : rf; if (hi && id < IDMAX) { const char *nm = xmlAttr(p, gt, "name", &nl), *ad = xmlAttr(p, gt, "addr", &al); if (nm) { if (nl >= 95u) nl = 95u; memcpy(fr[id].nm, nm, nl); fr[id].nm[nl] = '\0'; } else strcpy(fr[id].nm, "?"); if (ad) fr[id].addr = strtoull(ad, NULL, 16); if (gt[-1] != '/') curFr = id; } if (inBt && curBt < IDMAX && !bt[curBt] && fid < IDMAX) bt[curBt] = fid; }
        else if (!strncmp(p, "<backtrace ", 11) || !strncmp(p, "<tagged-backtrace ", 18)) { uint32_t id = xmlId(p, gt, &hi, &hr, &rf), bid = hi ? id : hr ? rf : 0u; if (hi && id < IDMAX) { curBt = id; inBt = (gt[-1] != '/'); } if (inRow && !rowBt && bid < IDMAX) rowBt = bid; }
        p = gt;
    }
    TFunc *fs = xcal(2048u, sizeof *fs), *pl = xcal(4096u, sizeof *pl);
    for (size_t i = 1u; i < IDMAX; i++) if (fr[i].cnt) { addCnt(fs, &nf, 2048u, fr[i].nm, fr[i].cnt); uint32_t b = fr[i].bin; if (b < IDMAX && bOak[b] && bLd[b] && ns) { uint64_t st = (0x100000000ULL + fr[i].addr - bLd[b]) & ~3ULL; size_t lo = 0u, hi2 = ns; while (lo + 1u < hi2) { size_t md = (lo + hi2) / 2u; if (syms[md].a <= st) lo = md; else hi2 = md; } addCnt(pl, &np2, 4096u, pf("%s+0x%llx", syms[lo].nm, (unsigned long long)(st - syms[lo].a)), fr[i].cnt); } }
    qsort(fs, nf, sizeof *fs, cmpTF); qsort(pl, np2, sizeof *pl, cmpTF); printf("%zu samples\n", rows);
    for (size_t i = 0u; i < nf && i < 20u; i++) printf("  %6.2f%%  %6zu  %s\n", rows ? 100.0 * (double)fs[i].n / (double)rows : 0.0, fs[i].n, fs[i].nm);
    for (size_t i = 0u; i < np2 && i < 35u; i++) printf("  %6.2f%%  %6zu  %s\n", rows ? 100.0 * (double)pl[i].n / (double)rows : 0.0, pl[i].n, pl[i].nm);
    fflush(stdout); free(fs); free(pl); free(fr); free(bt); free(bLd); free(bOak); free(syms); free(xml);
}

static void execJob(const char *treeIn, const char *verb, char out[256]) {
    char tree[PATH_MAX], bp[PATH_MAX], bestId[64], eng[PATH_MAX], td[65], tid[65]; snprintf(tree, sizeof tree, "%s", treeIn); snprintf(bp, sizeof bp, "%s/best", REPO);
    if (!isKind(bp, S_IFREG)) die("No recorded best"); char *bt = readLine(bp); if (!bt[0]) { free(bt); die("No recorded best"); } snprintf(bestId, sizeof bestId, "%s", bt); free(bt);
    Rec *ref = loadRec(pf("%s/%s", BENCH, bestId)); buildTree(tree, false); snprintf(eng, sizeof eng, "%s/.build/oak", tree);
    const char *te[] = { "Sources","fleet.c" }; digestEntries(tree, te, isKind(pf("%s/fleet.c", tree), S_IFREG) ? 2u : 1u, td); shaHex(td, strlen(td), tid); tid[40] = '\0';
    char *ea[] = { eng,"envs",NULL }, *eo = NULL; if (runProc(ea, REPO, &eo, 1, false)) die("oak envs"); JV *tb = jParse(eo); free(eo); size_t ne = tb->n; EMeta *ms = xcal(ne, sizeof *ms);
    for (size_t i = 0u; i < ne; i++) { JV *r = tb->a[i]; ms[i] = (EMeta){ xdup(jGet(r, "envId")->s), xdup(jGet(r, "category")->s), (uint32_t)jGet(r, "seedStart")->d, (uint32_t)jGet(r, "seedEnd")->d, {0} }; }
    jFree(tb); envDigests(tree, ms, ne); bool judged = strcmp(verb, "adopt") != 0;
    if (judged) {
        bool ok = ref->steps == OAK_STEPS && ref->nBlk == ne; for (size_t i = 0u; i < ne && ok; i++) { const EBlk *b = rBlk(ref, ms[i].id); if (!b || b->s0 != ms[i].s0 || b->s1 != ms[i].s1) ok = false; } if (!ok) die("Schedule differs from best; use ./fleet adopt");
        JV *re = jGet(rSrc(ref), "envs"); char *ch = NULL; size_t cl = 0u, nc = 0u; FILE *cm = open_memstream(&ch, &cl);
        for (size_t i = 0u; i < ne; i++) { JV *rv = jGet(re, ms[i].id); if (!rv || strcmp(rv->s, ms[i].dig)) { if (nc++) fputs(", ", cm); fputs(ms[i].id, cm); } } fclose(cm);
        if (nc && !strcmp(verb, "try")) { char msg[1024]; snprintf(msg, sizeof msg, "REFUSE: env sources changed (%s)", ch); free(ch); die("%s", msg); }
        if (nc) { printf("Environment sources changed (%zu): %s\n", nc, ch); fflush(stdout); } free(ch);
    }
    if (!strcmp(verb, "trace")) { traceEng(eng); snprintf(out, 256u, "trace: sampled %u s", OAK_TRACE_SEC); goto done; }
    if (!strcmp(verb, "try")) { attemptTry(eng, tree, ms, ne, ref, out); goto done; }
    char oth[PATH_MAX], tOth[PATH_MAX]; refEngine(ref, oth);
    if (judged && filesEq(eng, oth)) { if (!strcmp(verb, "probe")) { snprintf(out, 256u, "probe: binary identical to recorded %.9s", bestId); goto done; } die("REJECT: binary identical to recorded %.9s; nothing to judge", bestId); }
    Rec *anc = NULL; const Rec *tRef = ref; strcpy(tOth, oth); if (!judged && strcmp(bestId, OAK_ANCHOR)) { tRef = anc = loadRec(pf("%s/%s", BENCH, OAK_ANCHOR)); refEngine(anc, tOth); }
    double ratio = speedStage(eng, tOth, tRef, judged); const char *trn = baseOf(tRef->root); freeRec(anc);
    if (!strcmp(verb, "probe")) { snprintf(out, 256u, "probe: %.4fx against recorded %.9s", ratio, bestId); goto done; }
    if (ratio < 1.0) die("%s: paired speed %.4fx against recorded %.9s; best stays %s", judged ? "REJECT" : "REFUSE", ratio, trn, bestId);
    BSet rb = ensureBase("random", ms, ne, eng, tree), pb = ensureBase("ppo", ms, ne, eng, tree), ob = ensureBase("oracle", ms, ne, eng, tree);
    char cr[PATH_MAX], pd[65], bk[65]; snprintf(cr, sizeof cr, "%s/%s", BENCH, tid); const char *pe[] = { "Sources" }; digestEntries(tree, pe, 1u, pd);
    Rec *cand = recDone(cr) ? (writeSrcJ(pf("%s/sources.json", cr), "oak", ms, ne, pd), snapTree(pf("%s/source", cr), tree), loadRec(cr)) : recordRun(eng, "oak", cr, ms, ne, pd, tree, NULL, 0u);
    shaFile(eng, bk); bk[40] = '\0'; const char *trd = pf("%s/tries/%s", BENCH, bk);
    if (recDone(trd)) { Rec *att = loadRec(trd); size_t nd = 0u; for (size_t i = 0u; i < att->nBlk; i++) { int al = rLane(att, att->blks[i].id, att->blks[i].s0), cl = rLane(cand, att->blks[i].id, att->blks[i].s0); if (al < 0 || cl < 0 || att->hash[al] != cand->hash[cl]) nd++; } freeRec(att); if (nd) die("ERROR: try differs from fleet on %zu envs", nd); }
    EMeta *ord = xmal(ne * sizeof *ord); memcpy(ord, ms, ne * sizeof *ord); qsort(ord, ne, sizeof *ord, cmpEM); const char *pN[3] = { "oracle","ppo","random" }; const BSet *pS[3] = { &ob,&pb,&rb };
    FILE *bf = fopen(pf("%s/baselines.json", cr), "wb"); fputs("{\n", bf);
    for (int pi = 0; pi < 3; pi++) { fprintf(bf, " \"%s\": {\n", pN[pi]); for (size_t k = 0u; k < ne; k++) { size_t ix = 0u; while (strcmp(ms[ix].id, ord[k].id)) ix++; fprintf(bf, "  \"%s\": \"%s\"%s\n", ord[k].id, pS[pi]->c.e[pS[pi]->idx[ix]].nm, k + 1u < ne ? "," : ""); } fprintf(bf, " }%s\n", pi < 2 ? "," : ""); }
    fputs("}\n", bf); fclose(bf); free(ord);
    Verdict v = reportGate(cand, &rb, &pb, &ob, judged ? ref : NULL);
    bool promote = !judged || ((v.ident || (v.lo >= -OAK_MARGIN && !v.guard)) && (((v.lo > 0.0 && !v.guard) && ratio >= 1.0) || ratio >= 1.0 + OAK_MARGIN));
    if (!judged) printf("Steps/s: %s\n", fmtCommas(cand->sps)); else printf("Steps/s: %s recorded; paired speed against best: %.4fx\n", fmtCommas(cand->sps), ratio);
    fflush(stdout); rcFree(&rb.c); free(rb.idx); rcFree(&pb.c); free(pb.idx); rcFree(&ob.c); free(ob.idx); freeRec(cand);
    if (promote) { const char *l = pf("%s\n", tid); writeAll(bp, l, strlen(l)); unlink(pf("%s/tries/champion", BENCH)); snprintf(out, 256u, "%s: %s", judged ? "PROMOTE" : "ADOPT: best is", tid); }
    else die("REJECT: best stays %s", bestId);
done:
    for (size_t i = 0u; i < ne; i++) { free(ms[i].id); free(ms[i].cat); } free(ms); freeRec(ref);
}

static bool isTiming(const char *v) { return !strcmp(v, "probe") || !strcmp(v, "trace") || !strcmp(v, "try"); }
static void listJobs(SV *o) { SV raw = {0}; listExt(pf("%s/.build/queue", REPO), NULL, NULL, &raw); for (size_t i = 0u; i < raw.n; i++) { const char *jp = pf("%s/.build/queue/%s", REPO, raw.v[i]); if (isKind(pf("%s/job.json", jp), S_IFREG)) svAdd(o, xdup(jp)); } svFree(&raw); }
static bool pendingJob(bool timing, char out[PATH_MAX]) { SV js = {0}; listJobs(&js); bool ok = false; for (size_t i = 0u; i < js.n && !ok; i++) if (!isKind(pf("%s/started", js.v[i]), 0)) { JV *j = jFile(pf("%s/job.json", js.v[i])); if (isTiming(jGet(j, "verb")->s) == timing) { snprintf(out, PATH_MAX, "%s", js.v[i]); ok = true; } jFree(j); } svFree(&js); return ok; }
static void runJob(const char *dirIn) {
    char dir[PATH_MAX], verb[32], ts[32], pa[PATH_MAX], ver[256] = {0}; snprintf(dir, sizeof dir, "%s", dirIn);
    JV *j = jFile(pf("%s/job.json", dir)); snprintf(verb, sizeof verb, "%s", jGet(j, "verb")->s); jFree(j); stampNow(ts); writeAll(pf("%s/started", dir), pf("%s\n", ts), strlen(ts) + 1u);
    int ph = HAS_ACTIVE, pij = IN_JOB; if (ph) strcpy(pa, ACTIVE); strcpy(ACTIVE, dir); HAS_ACTIVE = 1; fflush(stdout); fflush(stderr);
    int so = dup(1), se = dup(2), fd = open(pf("%s/log", dir), O_WRONLY | O_CREAT | O_APPEND, 0644); dup2(fd, 1); dup2(fd, 2); close(fd); printf("%s %s %s\n", ts, verb, baseOf(dir)); fflush(stdout);
    jmp_buf pj; memcpy(pj, JJMP, sizeof pj); IN_JOB = 1; if (!setjmp(JJMP)) { if (isKind(pf("%s/cancel", dir), 0)) die("CANCELLED"); execJob(pf("%s/tree", dir), verb, ver); } else snprintf(ver, sizeof ver, "%s", JMSG);
    IN_JOB = pij; memcpy(JJMP, pj, sizeof JJMP); stampNow(ts); printf("%s %s\n", ts, ver); fflush(stdout); fflush(stderr);
    dup2(so, 1); dup2(se, 2); close(so); close(se); HAS_ACTIVE = ph; if (ph) strcpy(ACTIVE, pa); unlink(pf("%s/pid", dir)); writeAll(pf("%s/verdict", dir), pf("%s\n", ver), strlen(ver) + 1u);
    FILE *vf = fopen(pf("%s/verdicts.log", BENCH), "a"); if (vf) { fprintf(vf, "%s %s %s: %s\n", ts, verb, baseOf(dir), ver); fclose(vf); }
    char *sc = NULL; size_t sl = 0u; FILE *sm = open_memstream(&sc, &sl); fputs("display notification \"", sm); for (const char *s = pf("%.240s", ver); *s; s++) { if (*s == '"' || *s == '\\') fputc('\\', sm); fputc(*s == '\n' ? ' ' : *s, sm); } fprintf(sm, "\" with title \"OaK %s\"", verb); fclose(sm);
    char *oa[] = { "osascript","-e",sc,NULL }; runProc(oa, NULL, NULL, 0, true); free(sc); rmRf(pf("%s/tree", dir));
}
static void cmdServe(void) {
    if (getpriority(PRIO_PROCESS, 0) > 0) { puts("refusing reduced priority"); return; }
    mkdirP(BENCH); int fd = open(pf("%s/runner.lock", BENCH), O_WRONLY | O_CREAT, 0644); if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB)) return;
    setvbuf(stdout, NULL, _IOLBF, 0u); SV js = {0}; listJobs(&js);
    for (size_t i = 0u; i < js.n; i++) {
        char sp[PATH_MAX], pp[PATH_MAX]; snprintf(sp, sizeof sp, "%s/started", js.v[i]); snprintf(pp, sizeof pp, "%s/pid", js.v[i]);
        if (isKind(sp, 0) && !isKind(pf("%s/verdict", js.v[i]), 0)) { if (isKind(pp, S_IFREG)) { char *t = readLine(pp); pid_t orp = (pid_t)atol(t); free(t); PInfo *pi = NULL; size_t np = listPs(&pi); for (size_t k = 0u; k < np; k++) if (pi[k].pid == orp && !strcmp(pi[k].name, "oak")) kill(orp, SIGKILL); free(pi); unlink(pp); } unlink(sp); }
    }
    svFree(&js); for (char job[PATH_MAX]; ; ) { if (pendingJob(true, job) || pendingJob(false, job)) { runJob(job); continue; } flock(fd, LOCK_UN); if ((!pendingJob(true, job) && !pendingJob(false, job)) || flock(fd, LOCK_EX | LOCK_NB)) { close(fd); return; } }
}
static bool wakeRunner(const char *self) { if (getpriority(PRIO_PROCESS, 0) > 0) return false; mkdirP(BENCH); int lfd = open(pf("%s/runner.log", BENCH), O_WRONLY | O_CREAT | O_APPEND, 0644); char *av[] = { (char *)self,"serve",NULL }; spawnProc(av, REPO, -2, lfd, lfd, true, false); close(lfd); return true; }
static void onSig(int s) { (void)s; GOT_INT = 1; }
static void followJob(const char *dir, const char *self) {
    signal(SIGINT, onSig); off_t off = 0; double chk = monoSec();
    for (;;) {
        if (GOT_INT) { writeAll(pf("%s/cancel", dir), "", 0u); printf("\ncancelling %s\n", baseOf(dir)); return; }
        bool done = isKind(pf("%s/verdict", dir), 0); FILE *f = fopen(pf("%s/log", dir), "rb");
        if (f) { fseeko(f, off, SEEK_SET); char b[4096]; size_t n; while ((n = fread(b, 1u, sizeof b, f))) { fwrite(b, 1u, n, stdout); off += (off_t)n; } fflush(stdout); fclose(f); }
        if (done) return;
        if (monoSec() - chk > 10.0) { chk = monoSec(); int fd = open(pf("%s/runner.lock", BENCH), O_WRONLY | O_CREAT, 0644); if (fd >= 0) { if (!flock(fd, LOCK_EX | LOCK_NB)) { flock(fd, LOCK_UN); close(fd); wakeRunner(self); } else close(fd); } }
        usleep(500000);
    }
}

static void svgTxt(FILE *f, double x, double y, int sz, const char *col, const char *anc, int wt, const char *fmt, ...) { fprintf(f, "<text x=\"%.1f\" y=\"%.1f\" font-size=\"%d\" fill=\"%s\" text-anchor=\"%s\" font-weight=\"%d\">", x, y, sz, col, anc, wt); va_list a; va_start(a, fmt); vfprintf(f, fmt, a); va_end(a); fputs("</text>\n", f); }
static void cmdRadar(void) {
    static const char *const AG[4] = {"OaK","Oracle","PPO","Random"}, *const CL[4] = {"#5548b4","#52a96a","#68a8cc","#e88152"}, *const CT[8] = {"control","navigation","memory","decision","prediction","world","games","puzzles"}, *const PN[3] = {"oracle","ppo","random"};
    char *best = readLine(pf("%s/best", REPO)); Rec *oak = loadRec(pf("%s/%s", BENCH, best)); JV *lk = jFile(pf("%s/%s/baselines.json", BENCH, best)); RCache rc = {0};
    char *ea[] = { pf("%s/%s/oak", BENCH, best),"envs",NULL }, *eo = NULL; if (runProc(ea, REPO, &eo, 1, false)) die("envs"); JV *cj = jParse(eo); free(eo);
    size_t ne = oak->nBlk; struct { char env[96], cat[32]; double m[4]; } *rs = xcal(ne, sizeof *rs); double sps[4] = { oak->sps, 0, 0, 0 };
    for (int pi = 0; pi < 3; pi++) {
        struct { char nm[128]; double ln; Rec *r; } pl[64]; size_t np = 0u;
        for (size_t e = 0u; e < ne; e++) { const EBlk *b = &oak->blks[e]; const char *rn = jGet(jGet(lk, PN[pi]), b->id)->s; Rec *f = rcGet(&rc, rn); const EBlk *pb = rBlk(f, b->id); rs[e].m[pi + 1] = meanA(f->score + pb->base, pb->s1 - pb->s0 + 1u); size_t k = 0u; while (k < np && strcmp(pl[k].nm, rn)) k++; if (k == np) { snprintf(pl[np].nm, 128u, "%s", rn); pl[np].ln = 0.0; pl[np++].r = f; } pl[k].ln += (double)(b->s1 - b->s0 + 1u); }
        double tl = 0.0, tt = 0.0; for (size_t k = 0u; k < np; k++) { tl += pl[k].ln; tt += pl[k].ln / pl[k].r->sps; } sps[pi + 1] = tl / tt;
    }
    for (size_t e = 0u; e < ne; e++) { const EBlk *b = &oak->blks[e]; snprintf(rs[e].env, 96u, "%s", b->id); rs[e].m[0] = meanA(oak->score + b->base, b->s1 - b->s0 + 1u); for (size_t i = 0u; i < cj->n; i++) if (!strcmp(jGet(cj->a[i], "envId")->s, b->id)) { snprintf(rs[e].cat, 32u, "%s", jGet(cj->a[i], "category")->s); break; } }
    jFree(cj); jFree(lk); rcFree(&rc); uint32_t si = 0u, cut = oak->steps > OAK_TERMINAL ? oak->steps - OAK_TERMINAL : 0u; for (uint32_t i = 0u; i < oak->nEnd; i++) if (oak->ends[i] <= cut) si = i;
    char wb[32], sb[32]; strcpy(wb, fmtCommas((double)(oak->ends[oak->nEnd - 1u] - oak->ends[si]))); strcpy(sb, fmtCommas((double)oak->steps));
    double ax[9][4] = {{0}}, cx = 430.0, cy = 390.0, R = 240.0, lo = -0.25, sp = 1.25, ang[8], cols[8]; size_t ac[9] = {0}, ah[8] = {0};
    for (size_t e = 0u; e < ne; e++) { double den = rs[e].m[1] - rs[e].m[3], nm[4]; for (int a = 0; a < 4; a++) nm[a] = (rs[e].m[a] - rs[e].m[3]) / den; int ci = 0; while (ci < 8 && strcmp(rs[e].cat, CT[ci])) ci++; ac[ci]++; ac[8]++; for (int a = 0; a < 4; a++) { ax[ci][a] += nm[a]; ax[8][a] += nm[a]; } if (nm[0] > nm[2]) ah[ci]++; }
    for (int g = 0; g < 9; g++) for (int a = 0; a < 4; a++) ax[g][a] /= (double)ac[g]; for (int i = 0; i < 8; i++) { ang[i] = -OAK_PI / 2.0 + 2.0 * OAK_PI * i / 8.0; cols[i] = 330.0 + i * 750.0 / 7.0; }
    FILE *svg = fopen(pf("%s/oak-performance-radar.svg", REPO), "wb"); if (!svg) die("svg");
    fprintf(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1250\" height=\"930\" viewBox=\"0 0 1250 930\" role=\"img\" aria-labelledby=\"title desc\">\n<title id=\"title\">OaK and PPO across the 8 environment categories</title>\n<desc id=\"desc\">Mean normalized reward per step over the final %s steps of the %s-step best recording, one axis per declared category with its environment count, Random at 0 and Oracle at 1. Outlines only: the polygon area is not the score. The overall ranking is the equal-weight mean over all %zu environments.</desc>\n<rect width=\"1250\" height=\"930\" fill=\"#ffffff\"/>\n<g font-family=\"Arial, Helvetica, sans-serif\" fill=\"#243044\">\n", wb, sb, ne);
    svgTxt(svg, 54, 55, 30, "#243044", "start", 600, "Policy comparison"); svgTxt(svg, 54, 86, 17, "#637084", "start", 400, "Mean normalized reward per step \xc2\xb7 %zu environments \xc2\xb7 best recording %.9s", ne, best);
    for (int i = 1; i <= 3; i++) fprintf(svg, "<circle cx=\"430\" cy=\"390\" r=\"%.1f\" fill=\"none\" stroke=\"#cbd2dc\" stroke-width=\"1\" stroke-dasharray=\"2 4\"/>\n", R * (0.25 * i - lo) / sp);
    fprintf(svg, "<circle cx=\"430\" cy=\"390\" r=\"%.1f\" fill=\"none\" stroke=\"%s\" stroke-width=\"2.5\" stroke-dasharray=\"3 5\"/>\n<circle cx=\"430\" cy=\"390\" r=\"%.1f\" fill=\"none\" stroke=\"%s\" stroke-width=\"2.5\" stroke-dasharray=\"9 6\"/>\n", R * (-lo) / sp, CL[3], R * (1.0 - lo) / sp, CL[1]);
    for (int i = 0; i < 8; i++) fprintf(svg, "<path d=\"M 430 390 L %.2f %.2f\" stroke=\"#d0d6df\" stroke-width=\"1\"/>\n", cx + R * cos(ang[i]), cy + R * sin(ang[i]));
    for (int p = 0; p < 2; p++) {
        int a = p ? 0 : 2; fputs("<polygon points=\"", svg); for (int i = 0; i < 8; i++) { double rd = R * (ax[i][a] - lo) / sp; fprintf(svg, "%s%.2f,%.2f", i ? " " : "", cx + rd * cos(ang[i]), cy + rd * sin(ang[i])); } fprintf(svg, "\" fill=\"none\" stroke=\"%s\" stroke-width=\"3.5\" stroke-linejoin=\"round\"/>\n", CL[a]);
        for (int i = 0; i < 8; i++) { double rd = R * (ax[i][a] - lo) / sp, x = cx + rd * cos(ang[i]), y = cy + rd * sin(ang[i]); if (a == 2) fprintf(svg, "<rect x=\"%.2f\" y=\"%.2f\" width=\"8\" height=\"8\" fill=\"%s\"/>\n", x - 4.0, y - 4.0, CL[a]); else fprintf(svg, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"5\" fill=\"%s\"/>\n", x, y, CL[a]); }
    }
    for (int i = 0; i < 5; i++) svgTxt(svg, cx + R * (0.25 * i - lo) / sp - 2.0, cy + 21.0, 13, "#647084", "middle", 400, "%g", 0.25 * i);
    svgTxt(svg, 422, 412, 12, "#647084", "end", 400, "\xe2\x88\x92""0.25"); svgTxt(svg, cx + R * (-lo) / sp + 6.0, 382, 12, CL[3], "start", 600, "Random"); svgTxt(svg, cx + R * (1.0 - lo) / sp + 6.0, 382, 12, CL[1], "start", 600, "Oracle");
    for (int i = 0; i < 8; i++) { double x = cx + (R + 40.0) * cos(ang[i]), y = cy + (R + 40.0) * sin(ang[i]); const char *an = cos(ang[i]) > 0.35 ? "start" : cos(ang[i]) < -0.35 ? "end" : "middle"; svgTxt(svg, x, y, 20, "#243044", an, 400, "%s", CT[i]); svgTxt(svg, x, y + 21.0, 14, "#687489", an, 400, "n = %zu", ac[i]); }
    int rk[4] = {0,1,2,3}; for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) if (ax[8][rk[j]] > ax[8][rk[i]]) { int t = rk[i]; rk[i] = rk[j]; rk[j] = t; }
    svgTxt(svg, 900, 222, 24, "#243044", "start", 600, "Overall ranking"); svgTxt(svg, 900, 248, 15, "#637084", "start", 400, "equal weight over all %zu environments", ne);
    for (int j = 0; j < 4; j++) { int a = rk[j], y = 300 + j * 76; svgTxt(svg, 900, y, 20, CL[a], "start", 600, "%d. %s", j + 1, AG[a]); svgTxt(svg, 1180, y, 20, "#243044", "end", 400, "%.3f", ax[8][a]); if (ax[8][a] > 0.0) fprintf(svg, "<rect x=\"900\" y=\"%d\" width=\"%.2f\" height=\"13\" fill=\"%s\"/>\n", y + 12, 280.0 * ax[8][a], CL[a]); else fprintf(svg, "<path d=\"M 900 %d v 17\" stroke=\"%s\" stroke-width=\"2\"/>\n", y + 10, CL[a]); svgTxt(svg, 900, y + 46, 16, "#637084", "start", 400, "%s steps/s", fmtCommas(sps[a])); }
    svgTxt(svg, 900, 621, 15, "#637084", "start", 400, "steps per second; baselines lane-weighted over their recordings");
    svgTxt(svg, 54, 745, 17, "#243044", "start", 600, "Axis scores"); for (int i = 0; i < 8; i++) svgTxt(svg, cols[i], 745, 15, "#243044", "middle", 400, "%s", CT[i]);
    for (int j = 0; j < 2; j++) { int a = j ? 2 : 0, y = 775 + j * 26; svgTxt(svg, 54, y, 17, CL[a], "start", 600, "%s", AG[a]); for (int i = 0; i < 8; i++) svgTxt(svg, cols[i], y, 16, "#243044", "middle", 400, "%.3f", ax[i][a]); }
    svgTxt(svg, 54, 827, 15, "#637084", "start", 400, "OaK ahead of PPO on"); for (int i = 0; i < 8; i++) svgTxt(svg, cols[i], 827, 15, "#637084", "middle", 400, "%zu of %zu", ah[i], ac[i]);
    svgTxt(svg, 54, 904, 15, "#637084", "start", 400, "Equal weight per environment \xc2\xb7 outlines, not areas \xc2\xb7 Random = 0 \xc2\xb7 Oracle = 1 \xc2\xb7 final %s of %s steps", wb, sb); fputs("</g></svg>\n", svg); fclose(svg);
    char rmP[PATH_MAX]; snprintf(rmP, sizeof rmP, "%s/README.md", REPO); char *rm = readAll(rmP, NULL);
    const char *st = strstr(rm, " SPS on the"), *ss = st, *fl = strstr(rm, "Reward per step over the final"), *ll = strstr(rm, "Baseline throughput"), *al = strchr(ll, '\n');
    while (ss > rm && (isdigit((unsigned char)ss[-1]) || ss[-1] == ',')) ss--; FILE *rf = fopen(rmP, "wb"); if (!rf) die("README");
    fwrite(rm, 1u, (size_t)(ss - rm), rf); fputs(fmtCommas(sps[0]), rf); fwrite(st, 1u, (size_t)(fl - st), rf);
    fprintf(rf, "Reward per step over the final %s steps of the %s-step best recording (`%s`), all %zu environments with %u seeds each (%s lanes), with the Oracle, Random and PPO baselines on the same seeds. Mean over seeds; Score = (OaK \xe2\x88\x92 Random) / (Oracle \xe2\x88\x92 Random), unclipped.\n\n| Environment | Category | OaK | Oracle | Random | PPO | Score |\n|---|---|---:|---:|---:|---:|---:|\n", wb, sb, best, ne, oak->lanes / (uint32_t)ne, fmtCommas((double)oak->lanes));
    for (size_t e = 0u; e < ne; e++) fprintf(rf, "| %s | %s | %s | %s | %s | %s | %s |\n", rs[e].env, rs[e].cat, fmtVal(rs[e].m[0]), fmtVal(rs[e].m[1]), fmtVal(rs[e].m[3]), fmtVal(rs[e].m[2]), fmtVal((rs[e].m[0] - rs[e].m[3]) / (rs[e].m[1] - rs[e].m[3])));
    fprintf(rf, "\nBaseline throughput, lane-weighted over the recordings that hold each environment's baseline: PPO %s steps/s, Random %s steps/s, Oracle %s steps/s.", fmtCommas(sps[2]), fmtCommas(sps[3]), fmtCommas(sps[1]));
    if (al) fputs(al, rf); fclose(rf); free(rm); for (int g = 0; g < 9; g++) printf("%-11s n=%3zu  OaK %+.3f  Oracle %+.3f  PPO %+.3f  Random %+.3f\n", g < 8 ? CT[g] : "all", ac[g], ax[g][0], ax[g][1], ax[g][2], ax[g][3]);
    free(best); free(rs); freeRec(oak);
}

int main(int argc, char **argv) {
    char raw[PATH_MAX], self[PATH_MAX], cp[PATH_MAX], src[PATH_MAX]; uint32_t sz = sizeof raw;
    if (_NSGetExecutablePath(raw, &sz) || !realpath(raw, self)) die("self"); snprintf(cp, sizeof cp, "%s", self); if (!realpath(dirname(cp), REPO)) die("repo");
    snprintf(BENCH, sizeof BENCH, "%s/.oak-benchmarks", REPO); snprintf(src, sizeof src, "%s/fleet.c", REPO); struct stat sS, sE;
    if (!stat(src, &sS) && !stat(self, &sE) && (sS.st_mtimespec.tv_sec > sE.st_mtimespec.tv_sec || (sS.st_mtimespec.tv_sec == sE.st_mtimespec.tv_sec && sS.st_mtimespec.tv_nsec > sE.st_mtimespec.tv_nsec))) {
        char tmp[PATH_MAX]; snprintf(tmp, sizeof tmp, "%s.rebuild.%d", self, (int)getpid()); char *cc[] = { "clang","-O3","-mcpu=apple-m4","-std=c11","-fno-fast-math","-ffp-contract=off","-Wall","-Wextra",src,"-lm","-lpthread","-o",tmp,NULL };
        if (runProc(cc, REPO, NULL, 0, false) || rename(tmp, self)) { unlink(tmp); die("self-rebuild"); } execv(self, argv); die("execv");
    }
    const char *verb = argc > 1 ? argv[1] : "judge";
    if (!strcmp(verb, "build")) { buildTree(REPO, true); return 0; } if (!strcmp(verb, "radar")) { cmdRadar(); return 0; } if (!strcmp(verb, "serve")) { cmdServe(); return 0; }
    if (!isTiming(verb) && strcmp(verb, "judge") && strcmp(verb, "adopt")) die("usage: ./fleet [judge|try|probe|trace|adopt|build|radar|serve]");
    mkdirP(pf("%s/.build/queue", REPO)); char nm[64], ts[32], st[PATH_MAX], job[PATH_MAX];
    snprintf(nm, sizeof nm, "%llu-%s", (unsigned long long)timeNs(), verb); snprintf(st, sizeof st, "%s/.build/queue/.%s", REPO, nm); snprintf(job, sizeof job, "%s/.build/queue/%s", REPO, nm);
    snapTree(pf("%s/tree", st), REPO); stampNow(ts); const char *jb = pf("{\"verb\": \"%s\", \"submitted\": \"%s\"}\n", verb, ts); writeAll(pf("%s/job.json", st), jb, strlen(jb));
    if (rename(st, job)) die("rename job");
    if (!wakeRunner(self)) printf("queued %s as %s; run from un-niced shell\n", verb, nm); else if (isTiming(verb)) followJob(job, self);
    else printf("queued %s as %s; its log is .build/queue/%s/log and the verdict lands in .oak-benchmarks/verdicts.log\n", verb, nm, nm);
    return 0;
}
