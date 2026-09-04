// Quantum Code Generator - LLVM IR generation
use crate::mir::*;
use crate::ast::Type;
use std::path::Path;

fn trait_object_typename(trait_name: &str) -> String {
    format!("{}_VTable", trait_name)
}

pub struct CodeGenerator {
    opt_level: u8,
    debug: bool,
    emit_c: bool,
    void_fns: std::collections::HashSet<String>,
    /// Names declared via `extern "C" { ... }` — these must never be
    /// C-keyword-mangled since they have to match the linked library's
    /// actual exported symbol name exactly.
    extern_fns: std::collections::HashSet<String>,
    /// Local variable IDs (within the function currently being generated)
    /// known to hold a String value. Populated fresh for each function
    /// before generating its body. Used so `a + b` between two string
    /// *variables* (not just string literals) correctly emits q_strcat()
    /// instead of raw pointer arithmetic.
    string_locals: std::collections::HashSet<usize>,
    /// Additional libraries to pass to the linker as `-l<name>`, beyond
    /// the always-linked math library. Populated from `#link "name"`
    /// source directives and the `--link <name>` CLI flag.
    pub link_libs: Vec<String>,
    pub link_dirs: Vec<String>,
}

/// Mangle a Quantum function/variable name to avoid conflicts with C reserved
/// words and common C library names. The strategy: prefix with `q_usr_` for
/// any name that would be a C keyword or conflict.
fn c_safe_name(name: &str) -> String {
    const C_RESERVED: &[&str] = &[
        // C keywords
        "auto","break","case","char","const","continue","default","do",
        "double","else","enum","extern","float","for","goto","if","inline",
        "int","long","register","restrict","return","short","signed","sizeof",
        "static","struct","switch","typedef","union","unsigned","void",
        "volatile","while",
        // C11 / common
        "_Bool","_Complex","_Imaginary","_Alignas","_Alignof","_Atomic",
        "_Static_assert","_Thread_local","_Noreturn","_Generic",
        // Common C stdlib names that conflict
        "index","time","string","near","far","interrupt","signal","abort",
        "exit","free","malloc","realloc","calloc","printf","scanf","puts",
        "gets","fopen","fclose","read","write","open","close","main",
    ];
    // `main` must stay as-is — it's the C entry point
    if name == "main" { return name.to_string(); }
    if C_RESERVED.contains(&name) {
        format!("q_{}", name)
    } else {
        name.to_string()
    }
}

/// Returns true if `op` is a Copy/Move of a local known to hold a String
/// (per the `string_locals` set built for the function currently being
/// generated).
fn operand_is_string_local(op: &MirOperand, string_locals: &std::collections::HashSet<usize>) -> bool {
    match op {
        MirOperand::Copy(MirPlace::Local(id)) | MirOperand::Move(MirPlace::Local(id)) => {
            string_locals.contains(id)
        }
        _ => false,
    }
}

impl CodeGenerator {
    pub fn new(opt_level: u8, debug: bool, emit_c: bool) -> Self {
        Self { opt_level, debug, emit_c, void_fns: std::collections::HashSet::new(),
               extern_fns: std::collections::HashSet::new(), link_libs: Vec::new(), link_dirs: Vec::new(),
               string_locals: std::collections::HashSet::new() }
    }

    pub fn generate(&mut self, mir: MirProgram, output_path: &Path) -> Result<(), String> {
        // Collect void-returning functions (builtins + user-defined)
        self.void_fns.clear();
        for b in &["print","println","print_int","println_int","print_float","println_float",
                   "print_bool","println_bool","print_newline","eprint","eprintln",
                   "printf","puts","abort","exit","__quantum_panic",
                   "q_free","q_memcpy","q_memmove","q_write_i64","q_write_f64","q_strmap_insert","q_strmap_free"] {
            self.void_fns.insert(b.to_string());
        }
        for func in &mir.functions {
            if matches!(func.return_type, crate::ast::Type::Void) {
                self.void_fns.insert(func.name.clone());
            }
        }

        let mut c_code = String::new();

        // Includes
        c_code.push_str("#include <stdio.h>\n");
        c_code.push_str("#include <stdlib.h>\n");
        c_code.push_str("#include <stdint.h>\n");
        c_code.push_str("#include <stdbool.h>\n");
        c_code.push_str("#include <string.h>\n#include <signal.h>\n#include <ctype.h>\n#include <math.h>\n\n");

        // Runtime types for Option and Result — a single universal
        // representation using int64_t (Quantum's universal value slot)
        // rather than parameterized C generics, which don't exist.
        c_code.push_str("typedef struct { bool has_value; int64_t value; } QuantumOption;\n");
        c_code.push_str("typedef struct { bool is_ok; int64_t value; const char* error; } QuantumResult;\n\n");

                // Emit C struct type definitions for every Quantum struct, before
        // any forward declarations or function bodies that might reference
        // them as local variable types.
        for item in &mir.structs {
            // Skip generic struct templates (those with TypeParam fields) —
            // only the concrete monomorphized versions get emitted as C structs.
            let is_generic = item.fields.iter()
                .any(|(_, t)| matches!(t, crate::ast::Type::TypeParam(_)));
            if is_generic { continue; }
            c_code.push_str(&format!("typedef struct {} {{\n", item.name));
            for (field_name, field_ty) in &item.fields {
                c_code.push_str(&format!("    {} {};\n", self.type_to_c(field_ty), field_name));
            }
            c_code.push_str(&format!("}} {};\n", item.name));
        }
        if !mir.structs.is_empty() {
            c_code.push_str("\n");
        }

        // Emit vtable struct types for each trait before forward decls.
        for td in &mir.traits {
            c_code.push_str("typedef struct {\n    void* data;\n");
            for m in &td.methods {
                let ret = self.type_to_c(&m.return_type);
                let params = if m.param_types.is_empty() {
                    "void*".to_string()
                } else {
                    format!("void*, {}", m.param_types.iter().map(|t| self.type_to_c(t)).collect::<Vec<_>>().join(", "))
                };
                c_code.push_str(&format!("    {} (*{})({});\n", ret, m.name, params));
            }
            c_code.push_str(&format!("}} {};\n", trait_object_typename(&td.name)));
        }
        if !mir.traits.is_empty() { c_code.push_str("\n"); }

        // Collect thunk code — emitted AFTER forward declarations so
        // mangled method names are already declared when thunks reference them.
        let mut thunk_code = String::new();
        for (struct_name, trait_name) in &mir.trait_impls {
            if let Some(td) = mir.traits.iter().find(|t| &t.name == trait_name) {
                for m in &td.methods {
                    let ret = self.type_to_c(&m.return_type);
                    let mangled = format!("{}_{}", struct_name, m.name);
                    let thunk_name = format!("__thunk_{}_{}_{}", trait_name, struct_name, m.name);
                    let extra: String = m.param_types.iter().enumerate()
                        .map(|(i, t)| format!("{} a{}", self.type_to_c(t), i))
                        .collect::<Vec<_>>().join(", ");
                    let decl = if extra.is_empty() { "void* self_ptr".to_string() }
                               else { format!("void* self_ptr, {}", extra) };
                    let fwd_args = if extra.is_empty() {
                        format!("(*({}*)self_ptr)", struct_name)
                    } else {
                        let rest = m.param_types.iter().enumerate()
                            .map(|(i, _)| format!("a{}", i)).collect::<Vec<_>>().join(", ");
                        format!("(*({}*)self_ptr), {}", struct_name, rest)
                    };
                    thunk_code.push_str(&format!(
                        "static {} {}({}) {{ return {}({}); }}\n",
                        ret, thunk_name, decl, mangled, fwd_args
                    ));
                }
            }
        }

        // Forward declarations
        for func in &mir.functions {
            let ret = if func.name == "main" {
                "int".to_string()
            } else {
                self.type_to_c(&func.return_type)
            };
            let params = if func.params.is_empty() {
                "void".to_string()
            } else {
                func.params.iter()
                    .map(|p| self.param_to_c(p))
                    .collect::<Vec<_>>().join(", ")
            };
            c_code.push_str(&format!("{} {}({});\n", ret, c_safe_name(&func.name), params));
        }
        c_code.push_str("\n");

        // Extern function declarations — these have NO body in the
        // generated C; the linker resolves them from a linked library
        // (see `--link <name>` or `#link "name"` directive). We emit a
        // plain C forward declaration identical in form to a normal one,
        // but never generate a definition for these names later.
        if !mir.externs.is_empty() {
            c_code.push_str("// extern declarations (resolved by linker)\n");
            for ext_fn in &mir.externs {
                self.extern_fns.insert(ext_fn.name.clone());
                let ret = ext_fn.return_type.as_ref()
                    .map(|t| self.type_to_c(t))
                    .unwrap_or_else(|| "void".to_string());
                let params = if ext_fn.params.is_empty() {
                    "void".to_string()
                } else {
                    ext_fn.params.iter()
                        .map(|p| self.type_to_c(&p.ty))
                        .collect::<Vec<_>>().join(", ")
                };
                // Extern functions keep their exact name — no C-keyword
                // mangling — since they must match the external library's
                // actual symbol name exactly.
                c_code.push_str(&format!("extern {} {}({});\n", ret, ext_fn.name, params));
                // Register as non-void / void appropriately for call codegen
                if matches!(ext_fn.return_type, None) {
                    self.void_fns.insert(ext_fn.name.clone());
                }
            }
            c_code.push_str("\n");
        }

        // Emit thunks now — after forward decls so mangled names are declared.
        if !thunk_code.is_empty() {
            c_code.push_str(&thunk_code);
            c_code.push_str("\n");
        }

        // Built-in print function
        c_code.push_str(r#"
void print(const char* s) {
    if (s) printf("%s", s);
}

void println(const char* s) {
    if (s) printf("%s\n", s);
    else printf("\n");
}

// Overloads for when locals are stored as int64_t holding char* pointer
void print_str(int64_t s) {
    print((const char*)(intptr_t)s);
}

void println_str(int64_t s) {
    println((const char*)(intptr_t)s);
}

void print_int(int64_t n) {
    printf("%ld", (long)n);
}

void println_int(int64_t n) {
    printf("%ld\n", (long)n);
}

void print_float(double f) {
    printf("%g", f);
}

void println_float(double f) {
    printf("%g\n", f);
}

void print_bool(bool b) {
    printf("%s", b ? "true" : "false");
}

void println_bool(bool b) {
    printf("%s\n", b ? "true" : "false");
}

void print_newline() {
    printf("\n");
}

// Quantum memory wrapper functions.
// malloc/realloc names conflict with stdlib.h declarations so we wrap them.
int64_t q_alloc(int64_t size) { return (int64_t)(intptr_t)malloc((size_t)size); }
int64_t q_realloc(int64_t ptr, int64_t size) { return (int64_t)(intptr_t)realloc((void*)(intptr_t)ptr, (size_t)size); }
void    q_free(int64_t ptr) { free((void*)(intptr_t)ptr); }
void    q_memcpy(int64_t dst, int64_t src, int64_t n) { memcpy((void*)(intptr_t)dst,(void*)(intptr_t)src,(size_t)n); }
void    q_memmove(int64_t dst, int64_t src, int64_t n) { memmove((void*)(intptr_t)dst,(void*)(intptr_t)src,(size_t)n); }
int64_t q_read_i64(int64_t ptr, int64_t idx) { return ((int64_t*)(intptr_t)ptr)[idx]; }
void    q_write_i64(int64_t ptr, int64_t idx, int64_t val) { ((int64_t*)(intptr_t)ptr)[idx] = val; }
double  q_read_f64(int64_t ptr, int64_t idx) { return ((double*)(intptr_t)ptr)[idx]; }
void    q_write_f64(int64_t ptr, int64_t idx, double val) { ((double*)(intptr_t)ptr)[idx] = val; }

// String interpolation helpers
// __q_to_string: converts int64 to string (used for int/i64 variables)
// __q_to_string_f: converts double to string (used for float variables)
// __q_to_string_s: identity for strings (used for string variables)
// __q_strcat: concatenates two strings
static const char* __q_to_string(int64_t v) {
    char* buf = (char*)malloc(32); snprintf(buf, 32, "%lld", (long long)v); return buf; }
/* String variable stored as int64 pointer — cast back to char* */
/* Universal string pointer wrapper: works for const char* locals */
static const char* __q_to_string_p(const char* s) { return s ? s : ""; }
/* For int64 that stores a string pointer */
static const char* __q_to_string_i64p(int64_t p) {
    return p ? (const char*)(intptr_t)p : ""; }
/* For double → string */
static const char* __q_double_to_str(double v) {
    char* buf=(char*)malloc(32); snprintf(buf,32,"%g",v); return buf; }
static const char* __q_to_string_s2(const char* s) { return s ? s : ""; }
static const char* __q_to_string_f(double v) {
    char* buf = (char*)malloc(32); snprintf(buf, 32, "%g", v); return buf; }
static const char* __q_to_string_s(int64_t s_ptr) { const char* s=(const char*)(intptr_t)s_ptr; return s ? s : ""; }
static const char* __q_to_string_b(int32_t v) { return v ? "true" : "false"; }
/* Universal interpolation: tries to format any value */
#define __q_interp(v) _Generic((v),     double:      __q_to_string_f,     float:       __q_to_string_f,     int64_t:     __q_to_string,       int32_t:     (const char*(*)(int32_t))__q_to_string_b,     const char*: __q_to_string_s,     char*:       __q_to_string_s  )(v)
static const char* __q_str_safe(int64_t p) {
    /* handles both const char* and int64 pointer */
    return p ? (const char*)(intptr_t)p : ""; }
static const char* __q_strcat(const char* a, const char* b) {
    if(!a)a=""; if(!b)b="";
    size_t la=strlen(a),lb=strlen(b);
    char* buf=(char*)malloc(la+lb+1);
    memcpy(buf,a,la); memcpy(buf+la,b,lb+1); return buf; }
/* Variant that accepts int64 second arg (string stored as pointer) */
#define __q_strcat_i(a,b) __q_strcat((a),(const char*)(intptr_t)(b))
static const char* __q_strcat_p(const char* a, int64_t b_int) {
    const char* b=__q_str_safe(b_int);
    return __q_strcat(a,b); }


// String dot-methods: .upper() .lower() .trim() etc.
static const char* q_str_to_upper(const char* s) {
    if(!s) return ""; char* r=strdup(s);
    for(char* p=r;*p;p++) *p=toupper((unsigned char)*p); return r; }
static const char* q_str_to_lower(const char* s) {
    if(!s) return ""; char* r=strdup(s);
    for(char* p=r;*p;p++) *p=tolower((unsigned char)*p); return r; }
static const char* q_str_trim(const char* s) {
    if(!s) return "";
    while(*s==' '||*s=='\t'||*s=='\n'||*s=='\\r') s++;
    size_t l=strlen(s); char* r=strdup(s);
    while(l>0&&(r[l-1]==' '||r[l-1]=='\t'||r[l-1]=='\n'||r[l-1]=='\\r')) r[--l]='\0';
    return r; }
static int32_t q_str_len(const char* s) { return s?(int32_t)strlen(s):0; }
static int32_t q_str_contains(const char* s,const char* sub) {
    return(s&&sub&&strstr(s,sub))?1:0; }
static int32_t q_str_starts_with(const char* s,const char* p) {
    return(s&&p&&strncmp(s,p,strlen(p))==0)?1:0; }
static int32_t q_str_ends_with(const char* s,const char* p) {
    if(!s||!p)return 0; size_t ls=strlen(s),lp=strlen(p);
    return ls>=lp&&strcmp(s+ls-lp,p)==0?1:0; }
static const char* q_str_replace(const char* s,const char* f,const char* r) {
    if(!s||!f||!r)return s?s:""; size_t fs=strlen(f),rs=strlen(r),ls=strlen(s);
    if(fs==0)return s;
    char* buf=(char*)malloc(ls*(rs>fs?rs:fs)/(fs>0?fs:1)+rs+ls+1);
    char* out=buf; const char* p=s;
    while(*p){if(strncmp(p,f,fs)==0){memcpy(out,r,rs);out+=rs;p+=fs;}else{*out++=*p++;}}
    *out='\0'; return buf; }
static const char* q_str_split_first(const char* s,const char* d) {
    if(!s||!d)return s?s:""; const char* p=strstr(s,d); if(!p)return s;
    size_t l=p-s; char* r=(char*)malloc(l+1); memcpy(r,s,l); r[l]='\0'; return r; }
static const char* q_str_split_last(const char* s,const char* d) {
    if(!s||!d)return s?s:""; const char* p=NULL,*q=s; size_t dl=strlen(d);
    while((q=strstr(q,d))){p=q;q+=dl;} return p?p+dl:s; }
static int32_t q_str_index(const char* s,int32_t i) {
    if(!s||i<0||(size_t)i>=strlen(s))return-1; return(unsigned char)s[i]; }
static const char* q_str_substr(const char* s,int32_t st,int32_t len) {
    if(!s)return ""; int32_t sl=(int32_t)strlen(s);
    if(st<0)st=0; if(st>=sl)return ""; if(len<0||st+len>sl)len=sl-st;
    char* r=(char*)malloc(len+1); memcpy(r,s+st,len); r[len]='\0'; return r; }
static const char* q_str_repeat(const char* s,int32_t n) {
    if(!s||n<=0)return ""; size_t l=strlen(s);
    char* r=(char*)malloc(l*n+1);
    for(int32_t i=0;i<n;i++) memcpy(r+i*l,s,l); r[l*n]='\0'; return r; }
static int64_t q_str_parse_int(const char* s) { return s?strtoll(s,NULL,10):0; }
static double  q_str_parse_float(const char* s) { return s?strtod(s,NULL):0.0; }
static const char* q_str_from_int(int64_t v) {
    char* r=(char*)malloc(32); snprintf(r,32,"%lld",(long long)v); return r; }
static const char* q_str_from_float(double v) {
    char* r=(char*)malloc(32); snprintf(r,32,"%g",v); return r; }

// Number dot-methods: .abs() .sqrt() .pow() .floor() .ceil() .round() .min() .max() .clamp()
static double q_num_abs(double x)   { return x < 0.0 ? -x : x; }
static double q_num_sqrt(double x)  { return sqrt(x); }
static double q_num_pow(double x, double e) { return pow(x, e); }
static double q_num_floor(double x) { return floor(x); }
static double q_num_ceil(double x)  { return ceil(x); }
static double q_num_round(double x) { return round(x); }
static double q_num_min(double a, double b) { return a < b ? a : b; }
static double q_num_max(double a, double b) { return a > b ? a : b; }
static double q_num_clamp(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x); }
static int32_t q_print_help(const char* h, int64_t unused) {
    (void)unused; printf("%s", h); return 0; }


// ── Int methods ──────────────────────────────────────────────────
static int64_t  q_int_abs(int64_t x)    { return x < 0 ? -x : x; }
static double   q_int_to_float(int64_t x) { return (double)x; }
static const char* q_int_to_str(int64_t x) {
    char* b=(char*)malloc(32); snprintf(b,32,"%lld",(long long)x); return b; }
static int32_t  q_int_is_even(int64_t x)  { return x % 2 == 0 ? 1 : 0; }
static int32_t  q_int_is_odd(int64_t x)   { return x % 2 != 0 ? 1 : 0; }
static int64_t  q_int_pow(int64_t x, int64_t e) {
    int64_t r=1; for(int64_t i=0;i<e;i++) r*=x; return r; }
static int64_t  q_int_min(int64_t a, int64_t b) { return a<b?a:b; }
static int64_t  q_int_max(int64_t a, int64_t b) { return a>b?a:b; }
static int64_t  q_int_clamp(int64_t x, int64_t lo, int64_t hi) {
    return x<lo?lo:(x>hi?hi:x); }
static int32_t  q_int_bit_count(int64_t x) { return __builtin_popcountll((unsigned long long)x); }

// ── Float methods ─────────────────────────────────────────────────
static int64_t  q_float_to_int(double x)  { return (int64_t)x; }
static const char* q_float_to_str(double x) {
    char* b=(char*)malloc(32); snprintf(b,32,"%g",x); return b; }
static int32_t  q_float_is_nan(double x)     { return isnan(x)?1:0; }
static int32_t  q_float_is_infinite(double x){ return isinf(x)?1:0; }
static int32_t  q_float_is_finite(double x)  { return isfinite(x)?1:0; }
static int32_t  q_float_is_positive(double x){ return x>0.0?1:0; }
static int32_t  q_float_is_negative(double x){ return x<0.0?1:0; }

// ── Bool methods ──────────────────────────────────────────────────
static const char* q_bool_to_str(int32_t x)  { return x?"true":"false"; }
static int64_t  q_bool_to_int(int32_t x)     { return (int64_t)x; }
static int32_t  q_bool_and(int32_t a, int32_t b) { return a&&b?1:0; }
static int32_t  q_bool_or(int32_t a, int32_t b)  { return a||b?1:0; }
static int32_t  q_bool_not(int32_t x)        { return x?0:1; }

// ── Char methods ──────────────────────────────────────────────────
static int32_t  q_char_to_int(int32_t c)   { return c; }
static const char* q_char_to_str(int32_t c) {
    char* b=(char*)malloc(2); b[0]=(char)c; b[1]=0; return b; }
static int32_t  q_char_is_alpha(int32_t c)  { return isalpha(c)?1:0; }
static int32_t  q_char_is_digit(int32_t c)  { return isdigit(c)?1:0; }
static int32_t  q_char_is_upper(int32_t c)  { return isupper(c)?1:0; }
static int32_t  q_char_is_lower(int32_t c)  { return islower(c)?1:0; }
static int32_t  q_char_is_space(int32_t c)  { return isspace(c)?1:0; }
static int32_t  q_char_upper(int32_t c)     { return toupper(c); }
static int32_t  q_char_lower(int32_t c)     { return tolower(c); }

// ── Extra string methods ───────────────────────────────────────────
static const char* q_str_capitalize(const char* s) {
    if(!s||!*s) return s?s:"";
    char* r=strdup(s); r[0]=toupper((unsigned char)r[0]);
    for(int i=1;r[i];i++) r[i]=tolower((unsigned char)r[i]); return r; }
static const char* q_str_title(const char* s) {
    if(!s) return ""; char* r=strdup(s); int new_word=1;
    for(int i=0;r[i];i++){
        if(isspace((unsigned char)r[i])){new_word=1;}
        else if(new_word){r[i]=toupper((unsigned char)r[i]);new_word=0;}
        else{r[i]=tolower((unsigned char)r[i]);}
    } return r; }
static const char* q_str_lstrip(const char* s) {
    if(!s) return ""; const char* p=s;
    while(*p==' '||*p=='\\t'||*p=='\\n'||*p=='\\r') p++;
    return strdup(p); }
static const char* q_str_rstrip(const char* s) {
    if(!s) return ""; char* r=strdup(s); int l=(int)strlen(r);
    while(l>0&&(r[l-1]==' '||r[l-1]=='\\t'||r[l-1]=='\\n'||r[l-1]=='\\r')) r[--l]='\0';
    return r; }
static int32_t q_str_count(const char* s, const char* sub) {
    if(!s||!sub||!*sub) return 0;
    int32_t n=0; size_t sl=strlen(sub); const char* p=s;
    while((p=strstr(p,sub))){n++;p+=sl;} return n; }
static const char* q_str_center(const char* s, int32_t w, int32_t pad_char) {
    if(!s) return ""; int32_t l=(int32_t)strlen(s);
    if(l>=w) return strdup(s);
    int32_t total=w-l, left=total/2, right=total-left;
    char* r=(char*)malloc(w+1);
    memset(r,pad_char,left); memcpy(r+left,s,l);
    memset(r+left+l,pad_char,right); r[w]=' '; return r; }
static const char* q_str_zfill(const char* s, int32_t w) {
    return q_str_center(s,w,'0'); }
static int32_t q_str_find(const char* s, const char* sub) {
    if(!s||!sub) return -1;
    const char* p=strstr(s,sub); return p?(int32_t)(p-s):-1; }
static int32_t q_str_rfind(const char* s, const char* sub) {
    if(!s||!sub) return -1;
    const char* last=NULL,*p=s; size_t sl=strlen(sub);
    while((p=strstr(p,sub))){last=p;p+=sl;}
    return last?(int32_t)(last-s):-1; }
static int32_t q_str_eq(const char* a, const char* b) {
    if(!a||!b) return a==b?1:0; return strcmp(a,b)==0?1:0; }

// String hash map (open-addressing, string keys, int64 values)
// q_strmap_new(cap)         -> i64 handle
// q_strmap_insert(h,k,v)   -> void (grows if needed)
// q_strmap_get(h,k,out)    -> i32 (1=found, 0=missing; out filled)
// q_strmap_contains(h,k)   -> i32
// q_strmap_len(h)           -> i32
// q_strmap_free(h)          -> void
typedef struct QStrMapEntry { const char* key; int64_t value; } QStrMapEntry;
typedef struct QStrMap { QStrMapEntry* entries; int32_t cap; int32_t len; } QStrMap;
static uint32_t q_fnv1a(const char* s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
    return h;
}
static int64_t q_strmap_new(int32_t cap) {
    if (cap < 8) cap = 8;
    QStrMap* m = (QStrMap*)calloc(1, sizeof(QStrMap));
    m->entries = (QStrMapEntry*)calloc(cap, sizeof(QStrMapEntry));
    m->cap = cap; m->len = 0;
    return (int64_t)(intptr_t)m;
}
static void q_strmap_rehash(QStrMap* m, int32_t new_cap) {
    QStrMapEntry* old = m->entries; int32_t old_cap = m->cap;
    m->entries = (QStrMapEntry*)calloc(new_cap, sizeof(QStrMapEntry));
    m->cap = new_cap; m->len = 0;
    for (int32_t i = 0; i < old_cap; i++) {
        if (!old[i].key) continue;
        uint32_t h = q_fnv1a(old[i].key) % new_cap;
        while (m->entries[h].key && strcmp(m->entries[h].key, old[i].key) != 0)
            h = (h + 1) % new_cap;
        m->entries[h] = old[i]; m->len++;
    }
    free(old);
}
static void q_strmap_insert(int64_t handle, const char* key, int64_t value) {
    QStrMap* m = (QStrMap*)(intptr_t)handle;
    if (!key) return;
    if (m->len * 4 >= m->cap * 3) q_strmap_rehash(m, m->cap * 2);
    uint32_t h = q_fnv1a(key) % m->cap;
    while (m->entries[h].key && strcmp(m->entries[h].key, key) != 0)
        h = (h + 1) % m->cap;
    if (!m->entries[h].key) {
        m->entries[h].key = strdup(key); m->len++;
    }
    m->entries[h].value = value;
}
static int32_t q_strmap_get(int64_t handle, const char* key, int64_t* out) {
    QStrMap* m = (QStrMap*)(intptr_t)handle;
    if (!key || m->cap == 0) return 0;
    uint32_t h = q_fnv1a(key) % m->cap;
    for (int32_t i = 0; i < m->cap; i++) {
        if (!m->entries[h].key) return 0;
        if (strcmp(m->entries[h].key, key) == 0) { if(out)*out=m->entries[h].value; return 1; }
        h = (h + 1) % m->cap;
    }
    return 0;
}
static int32_t q_strmap_contains(int64_t handle, const char* key) {
    return q_strmap_get(handle, key, NULL);
}
static int32_t q_strmap_len(int64_t handle) {
    return ((QStrMap*)(intptr_t)handle)->len;
}
static void q_strmap_free(int64_t handle) {
    QStrMap* m = (QStrMap*)(intptr_t)handle;
    for (int32_t i = 0; i < m->cap; i++)
        if (m->entries[i].key) free((void*)m->entries[i].key);
    free(m->entries); free(m);
}

// File I/O primitives
// q_file_read_text(path) -> int64 ptr to heap string (caller must free with q_free)
// q_file_write_text(path, str) -> int (1=ok, 0=err)
// q_file_exists(path) -> int (1=yes, 0=no)
// q_file_lines(path) -> int64 ptr to array of line ptrs (null-terminated)
// q_csv_read(path, out_ptr, max_rows, max_cols) -> int (number of rows read)
static int64_t q_file_read_text(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return 0; }
    size_t n = fread(buf, 1, sz, f);
    buf[n] = '\0';
    fclose(f);
    return (int64_t)(intptr_t)buf;
}
static int32_t q_file_write_text(const char* path, const char* text) {
    FILE* f = fopen(path, "w");
    if (!f) return 0;
    fputs(text, f);
    fclose(f);
    return 1;
}
static int32_t q_file_append_text(const char* path, const char* text) {
    FILE* f = fopen(path, "a");
    if (!f) return 0;
    fputs(text, f);
    fclose(f);
    return 1;
}
static int32_t q_file_exists(const char* path) {
    FILE* f = fopen(path, "r");
    if (f) { fclose(f); return 1; }
    return 0;
}
static int32_t q_file_line_count(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    int n = 0; int c;
    while ((c = fgetc(f)) != EOF) if (c == '\n') n++;
    fclose(f); return n;
}
// CSV reader: fills a flat float64 array, returns number of rows parsed
static int32_t q_csv_read(const char* path, int64_t out_ptr, int32_t max_rows, int32_t max_cols) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    double* out = (double*)(intptr_t)out_ptr;
    char line[65536];
    int row = 0;
    // Skip header line
    if (fgets(line, sizeof(line), f) == NULL) { fclose(f); return 0; }
    while (row < max_rows && fgets(line, sizeof(line), f)) {
        char* p = line;
        for (int col = 0; col < max_cols; col++) {
            char* end;
            double v = strtod(p, &end);
            if (end == p) break;
            out[row * max_cols + col] = v;
            p = end;
            if (*p == ',' || *p == '\t') p++;
        }
        row++;
    }
    fclose(f);
    return row;
}
// Read a single column as floats
static int32_t q_csv_read_col(const char* path, int64_t out_ptr, int32_t col_idx, int32_t max_rows) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    double* out = (double*)(intptr_t)out_ptr;
    char line[65536];
    // Skip header
    if (fgets(line, sizeof(line), f) == NULL) { fclose(f); return 0; }
    int row = 0;
    while (row < max_rows && fgets(line, sizeof(line), f)) {
        char* p = line;
        for (int col = 0; ; col++) {
            char* end;
            double v = strtod(p, &end);
            if (end == p && col > col_idx) break;
            if (col == col_idx) { out[row] = v; break; }
            p = end;
            if (*p == ',' || *p == '\t') p++; else break;
        }
        row++;
    }
    fclose(f);
    return row;
}

// Beginner-friendly runtime panic — shows a clear explanation of what
// went wrong, where, and how to fix it, rather than just aborting silently.
void __quantum_panic(const char* msg) {
    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║         Quantum Runtime Error            ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════╝\n\n");
    fprintf(stderr, "  ✗ %s\n\n", msg);
    fprintf(stderr, "  Your program stopped here because a value\n");
    fprintf(stderr, "  was used that didn't exist or wasn't valid.\n\n");
    fprintf(stderr, "  How to fix this:\n");
    fprintf(stderr, "    • Check the value before using it (use .is_ok or .has_value)\n");
    fprintf(stderr, "    • Use .unwrap_or(default) to provide a fallback\n");
    fprintf(stderr, "    • Use the ? operator to propagate errors up to the caller\n\n");
    signal(SIGABRT, SIG_DFL);  // reset before abort to avoid double-message
    abort();
}

// String operations
static char _q_strbuf[65536];
static int  _q_strbuf_pos = 0;

const char* q_strcat(const char* a, const char* b) {
    int remaining = (int)(sizeof(_q_strbuf) - _q_strbuf_pos - 1);
    int na = a ? (int)strlen(a) : 0;
    int nb = b ? (int)strlen(b) : 0;
    if (na + nb >= remaining) { _q_strbuf_pos = 0; }
    char* dst = _q_strbuf + _q_strbuf_pos;
    if (a) memcpy(dst, a, na);
    if (b) memcpy(dst + na, b, nb);
    dst[na + nb] = '\0';
    _q_strbuf_pos += na + nb + 1;
    return dst;
}

char _q_intbuf[64];
/* const char* q_int_to_str defined above */

char _q_floatbuf[64];
/* const char* q_float_to_str defined above */

/* const char* q_bool_to_str(bool... — defined above */

int64_t q_strlen(const char* s) { return s ? (int64_t)strlen(s) : 0; }

"#);

        // Global variables
        for global in &mir.globals {
            c_code.push_str(&format!(
                "{} {} = {};\n",
                self.type_to_c(&global.ty),
                global.name,
                self.constant_to_c(&global.value)
            ));
        }
        c_code.push_str("\n");

        // Function definitions
        for func in &mir.functions {
            let mut func_code = self.generate_function(func)?;
            // Safe mode: inject crash handler setup at the very start of main()
            if mir.safe_mode && func.name == "main" {
                func_code = func_code.replacen(
                    "int main(void) {\n",
                    r#"void __quantum_signal_handler(int sig) {
    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║         Quantum Runtime Error            ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════╝\n\n");
    if (sig == 11) {
        fprintf(stderr, "  ✗ Segmentation fault — your program accessed\n");
        fprintf(stderr, "    memory it shouldn't have. This can happen when:\n");
        fprintf(stderr, "    • Accessing an array index out of bounds\n");
        fprintf(stderr, "    • Using a null pointer\n");
        fprintf(stderr, "    • Stack overflow from infinite recursion\n");
    } else if (sig == 6) {
        fprintf(stderr, "  ✗ Program aborted — this usually means unwrap()\n");
        fprintf(stderr, "    was called on a None or Err value.\n");
    } else {
        fprintf(stderr, "  ✗ Program crashed with signal %d\n", sig);
    }
    fprintf(stderr, "\n  Tip: Add error checking or use .unwrap_or() for safer code.\n\n");
    exit(1);
}

int main(void) {
    signal(SIGSEGV, __quantum_signal_handler);
    signal(SIGABRT, __quantum_signal_handler);
"#,
                    1,
                );
            }
            c_code.push_str(&func_code);
        }

        // Add #include <signal.h> for safe_mode signal handlers
        if mir.safe_mode {
            c_code = c_code.replacen(
                "#include <string.h>\n",
                "#include <string.h>\n#include <signal.h>\n",
                1,
            );
        }

        // Write C code to temp file
        let c_file = output_path.with_extension("c");
        std::fs::write(&c_file, c_code)
            .map_err(|e| format!("Failed to write C code: {}", e))?;

        // Compile with GCC/Clang
        let compiler = if cfg!(target_os = "windows") { "gcc" } else { "cc" };

        let mut cmd = std::process::Command::new(compiler);
        cmd.arg(&c_file)
           .arg("-o")
           .arg(output_path);

        // Optimization flags
        match self.opt_level {
            0 => { cmd.arg("-O0"); }
            1 => { cmd.arg("-O1"); }
            2 => { cmd.arg("-O2"); }
            _ => { cmd.arg("-O3"); }
        }

        if self.debug {
            cmd.arg("-g");
        }

        // Math library — always linked since it's extremely common and
        // has no downside (every libc ships it).
        cmd.arg("-lm");

        // Additional libraries requested via `#link "name"` source
        // directives or the `--link <name>` CLI flag, for calling into
        // other external C libraries via `extern "C" { ... }` blocks.
        for dir in &self.link_dirs {
            cmd.arg(format!("-L{}", dir));
            cmd.arg(format!("-Wl,-rpath,{}", dir));
        }
        for lib in &self.link_libs {
            cmd.arg(format!("-l{}", lib));
        }

        let output = cmd.output()
            .map_err(|e| format!("Failed to run compiler: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Compilation failed:\n{}", stderr));
        }

        // Clean up C file


        Ok(())
    }

    fn generate_function(&mut self, func: &MirFunction) -> Result<String, String> {
        // Track which locals in THIS function are String-typed, so binary
        // `+` between string variables (not just literals) emits q_strcat.
        self.string_locals.clear();
        for local in func.params.iter().chain(func.locals.iter()) {
            if matches!(local.ty, Type::String) {
                self.string_locals.insert(local.id);
            }
        }
        let mut code = String::new();

        // Function signature — main must be int main(void)
        let ret_type = if func.name == "main" {
            "int".to_string()
        } else {
            self.type_to_c(&func.return_type)
        };
        let is_void_fn = ret_type == "void";

        code.push_str(&format!("{} {}(", ret_type, c_safe_name(&func.name)));

        if func.params.is_empty() {
            code.push_str("void");
        } else {
            for (i, param) in func.params.iter().enumerate() {
                if i > 0 { code.push_str(", "); }
                code.push_str(&self.param_to_c(param));
            }
        }

        code.push_str(") {\n");

        // Local variables
        for local in &func.locals {
            if !func.params.iter().any(|p| p.id == local.id) {
                // C array declarations put the size after the variable
                // name (`int32_t arr[5];`), not after the type the way
                // type_to_c renders it standalone (`int32_t[5]`) — handle
                // this layout difference explicitly.
                if let Type::Array(inner, Some(size)) = &local.ty {
                    code.push_str(&format!(
                        "    {} {}[{}];\n",
                        self.type_to_c(inner),
                        self.local_name(local.id),
                        size
                    ));
                } else {
                    code.push_str(&format!(
                        "    {} {};\n",
                        self.type_to_c(&local.ty),
                        self.local_name(local.id)
                    ));
                }
            }
        }

        if !func.locals.is_empty() { code.push_str("\n"); }

        // Pre-pass: determine which locals hold string values (string literals,
        // q_strcat results, q_*_to_str results, or string-typed locals combined
        // via `+`). Computed via fixed-point iteration since string-ness can
        // propagate through chains of assignments and concatenations — e.g.
        // `let s = greeting + n.to_string()` depends on both `greeting` (string
        // literal) and the temp holding `q_int_to_str(n)` (a Call destination).
        let is_string_producing_call = |f: &str| matches!(f, "q_strcat" | "q_int_to_str" | "q_float_to_str" | "q_bool_to_str");

        let mut string_locals: std::collections::HashSet<usize> = std::collections::HashSet::new();
        // Seed with parameters/locals that are explicitly declared as
        // Type::String — the fixed-point propagation below only tracks
        // strings created via assignment/concatenation and previously
        // missed plain string-typed parameters entirely (e.g. `fn f(s: string)`
        // followed by `result + s` would treat `s` as non-string and wrap
        // it in q_int_to_str, corrupting the concatenation).
        for local in func.params.iter().chain(func.locals.iter()) {
            if matches!(local.ty, Type::String) {
                string_locals.insert(local.id);
            }
        }
        loop {
            let mut changed = false;
            for bb in &func.basic_blocks {
                for stmt in &bb.statements {
                    if let MirStatement::Assign { place: MirPlace::Local(id), rvalue } = stmt {
                        if string_locals.contains(id) { continue; }
                        let is_str = match rvalue {
                            MirRvalue::Use(MirOperand::Constant(MirConstant::String(_))) => true,
                            MirRvalue::Use(MirOperand::Copy(MirPlace::Local(src)))
                            | MirRvalue::Use(MirOperand::Move(MirPlace::Local(src))) => string_locals.contains(src),
                            MirRvalue::BinaryOp(crate::ast::BinOp::Add, l, r) => {
                                let l_str = matches!(l, MirOperand::Constant(MirConstant::String(_)))
                                    || matches!(l, MirOperand::Copy(MirPlace::Local(id2)) | MirOperand::Move(MirPlace::Local(id2)) if string_locals.contains(id2));
                                let r_str = matches!(r, MirOperand::Constant(MirConstant::String(_)))
                                    || matches!(r, MirOperand::Copy(MirPlace::Local(id2)) | MirOperand::Move(MirPlace::Local(id2)) if string_locals.contains(id2));
                                l_str || r_str
                            }
                            _ => false,
                        };
                        if is_str { string_locals.insert(*id); changed = true; }
                    }
                }
                // Call destinations: q_strcat / q_*_to_str results are strings
                if let MirTerminator::Call { func: fname, destination: Some(MirPlace::Local(id)), .. } = &bb.terminator {
                    if is_string_producing_call(fname) && !string_locals.contains(id) {
                        string_locals.insert(*id);
                        changed = true;
                    }
                }
            }
            if !changed { break; }
        }

        // Generate basic blocks
        for (i, bb) in func.basic_blocks.iter().enumerate() {
            if i > 0 { code.push_str(&format!("label_{}:\n", bb.id)); }
            for stmt in &bb.statements {
                code.push_str(&self.generate_statement(stmt, &string_locals));
            }
            // For main, replace bare `return;` with `return 0;`
            let term = self.generate_terminator(&bb.terminator, &func.locals, &string_locals);
            if func.name == "main" {
                code.push_str(&term.replace("    return;\n", "    return 0;\n"));
            } else if is_void_fn {
                // Suppress `return _lN;` in void functions — these arise
                // when a void wrapper calls a void extern and the call
                // result temp leaks as a trailing expression. Just drop
                // the return value; the function's declared return type
                // is void so returning anything is a C error.
                let cleaned = term.lines()
                    .filter(|line| {
                        let t = line.trim();
                        !(t.starts_with("return ") && t != "return 0;")
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let cleaned = if cleaned.is_empty() { cleaned } else { cleaned + "\n" };
                code.push_str(&cleaned);
            } else {
                code.push_str(&term);
            }
        }

        // Ensure main always has return 0
        if func.name == "main" && !code.contains("return 0;") {
            code.push_str("    return 0;\n");
        }

        code.push_str("}\n\n");
        Ok(code)
    }

    fn generate_statement(&self, stmt: &MirStatement, string_locals: &std::collections::HashSet<usize>) -> String {
        match stmt {
            MirStatement::Assign { place, rvalue } => {
                let place_str = self.place_to_c(place);

                // Special-case: Aggregate (array literal) assigned to an
                // already-declared local. C doesn't allow brace-init syntax
                // (`arr = {1,2,3};`) as a standalone statement — only at
                // the point of declaration — so instead emit one assignment
                // per element (`arr[0] = 1; arr[1] = 2; ...`), which is
                // valid C and produces the same result.
                if let MirRvalue::Aggregate(elements) = rvalue {
                    let mut out = String::new();
                    for (i, el) in elements.iter().enumerate() {
                        out.push_str(&format!(
                            "    {}[{}] = {};\n",
                            place_str, i, self.operand_to_c(el)
                        ));
                    }
                    return out;
                }

                // Special-case: BinaryOp::Add where either operand is a known
                // string (literal or string_locals member) -> q_strcat, with
                // both operands cast to const char* as needed.
                if let MirRvalue::BinaryOp(crate::ast::BinOp::Add, left, right) = rvalue {
                    let left_is_str = self.operand_is_string(left, string_locals);
                    let right_is_str = self.operand_is_string(right, string_locals);
                    if left_is_str || right_is_str {
                        let l = self.operand_to_c_as_string(left, string_locals);
                        let r = self.operand_to_c_as_string(right, string_locals);
                        let rvalue_str = format!("q_strcat({}, {})", l, r);
                        return format!("    {} = (int64_t)({});\n", place_str, rvalue_str);
                    }
                }

                let mut rvalue_str = self.rvalue_to_c(rvalue);
                if rvalue_str.starts_with("q_strcat") {
                    rvalue_str = self.fixup_strcat_args(rvalue, string_locals);
                }
                if rvalue_str.starts_with("q_strcat") || rvalue_str.starts_with("\"") {
                    // This local should be const char* — emit as direct assignment
                    // We cast to suppress the int64_t → const char* mismatch
                    format!("    {} = (int64_t)({});\n", place_str, rvalue_str)
                } else {
                    format!("    {} = {};\n", place_str, rvalue_str)
                }
            }
            MirStatement::StorageLive(_) | MirStatement::StorageDead(_) => {
                // No-op in C
                String::new()
            }
        }
    }

    /// True if an operand is statically known to hold a string value (a
    /// literal, or a local tracked in string_locals).
    fn operand_is_string(&self, op: &MirOperand, string_locals: &std::collections::HashSet<usize>) -> bool {
        match op {
            MirOperand::Constant(MirConstant::String(_)) => true,
            MirOperand::Copy(MirPlace::Local(id)) | MirOperand::Move(MirPlace::Local(id)) => string_locals.contains(id),
            _ => false,
        }
    }

    /// Render an operand for use as a `const char*` argument: string literals
    /// render as-is, string_locals get cast back from int64_t, and non-string
    /// operands (the "other side" of a `string + non_string` concatenation)
    /// get converted via q_int_to_str/q_float_to_str/q_bool_to_str based on
    /// the most likely numeric representation (int64_t default).
    fn operand_to_c_as_string(&self, op: &MirOperand, string_locals: &std::collections::HashSet<usize>) -> String {
        let s = self.operand_to_c(op);
        match op {
            MirOperand::Constant(MirConstant::String(_)) => s,
            MirOperand::Copy(MirPlace::Local(id)) | MirOperand::Move(MirPlace::Local(id)) => {
                if string_locals.contains(id) {
                    format!("(const char*)(intptr_t)({})", s)
                } else {
                    // Non-string operand being concatenated — stringify it.
                    format!("q_int_to_str({})", s)
                }
            }
            MirOperand::Constant(MirConstant::Int(_)) => format!("q_int_to_str({})", s),
            MirOperand::Constant(MirConstant::Float(_)) => format!("q_float_to_str({})", s),
            MirOperand::Constant(MirConstant::Bool(_)) => format!("q_bool_to_str({})", s),
            _ => s,
        }
    }

    /// For a q_strcat MirRvalue, re-render its arguments casting any operand
    /// that is a "string local" (int64_t holding a char* via prior cast) back
    /// to const char*, so q_strcat receives correctly-typed pointers.
    fn fixup_strcat_args(&self, rvalue: &MirRvalue, string_locals: &std::collections::HashSet<usize>) -> String {
        // q_strcat is only ever produced from a Call captured as a Use/Aggregate
        // in this codegen; but rvalue_to_c renders calls textually. Since we
        // can't easily re-walk into the call's operands here without a richer
        // MIR shape, fall back to a textual fix: locals appearing as bare
        // identifiers `_lN` that are in string_locals get wrapped in a cast.
        let rendered = self.rvalue_to_c(rvalue);
        let mut out = String::new();
        let mut chars = rendered.chars().peekable();
        let mut ident = String::new();
        while let Some(c) = chars.next() {
            if c == '_' || c.is_alphanumeric() {
                ident.push(c);
                continue;
            }
            if !ident.is_empty() {
                out.push_str(&self.maybe_cast_ident(&ident, string_locals));
                ident.clear();
            }
            out.push(c);
        }
        if !ident.is_empty() {
            out.push_str(&self.maybe_cast_ident(&ident, string_locals));
        }
        out
    }

    fn maybe_cast_ident(&self, ident: &str, string_locals: &std::collections::HashSet<usize>) -> String {
        if let Some(rest) = ident.strip_prefix("_l") {
            if let Ok(id) = rest.parse::<usize>() {
                if string_locals.contains(&id) {
                    return format!("(const char*)(intptr_t)({})", ident);
                }
            }
        }
        ident.to_string()
    }

    fn generate_terminator(&self, term: &MirTerminator, locals: &[MirLocal], string_locals: &std::collections::HashSet<usize>) -> String {
        match term {
            MirTerminator::Return(None) => {
                "    return;\n".to_string()
            }
            MirTerminator::Return(Some(op)) => {
                format!("    return {};\n", self.operand_to_c(op))
            }
            MirTerminator::Goto(target) => {
                format!("    goto label_{};\n", target)
            }
            MirTerminator::If { condition, then_block, else_block } => {
                format!(
                    "    if ({}) goto label_{}; else goto label_{};\n",
                    self.operand_to_c(condition),
                    then_block,
                    else_block
                )
            }
            MirTerminator::Call { func, args, destination, target } => {
                // Determine the type of an operand by looking up local declarations
                // (used to pick the correct print/println variant for ints/floats/
                // bools vs strings, instead of blindly reinterpreting everything as
                // a char* — which crashes for numeric values like loop counters).
                let operand_type = |op: &MirOperand| -> Option<Type> {
                    match op {
                        MirOperand::Copy(MirPlace::Local(id)) | MirOperand::Move(MirPlace::Local(id)) => {
                            locals.iter().find(|l| l.id == *id).map(|l| l.ty.clone())
                        }
                        MirOperand::Constant(MirConstant::Int(_)) => Some(Type::Int),
                        MirOperand::Constant(MirConstant::Float(_)) => Some(Type::Float),
                        MirOperand::Constant(MirConstant::Bool(_)) => Some(Type::Bool),
                        MirOperand::Constant(MirConstant::String(_)) => Some(Type::String),
                        _ => None,
                    }
                };

                let is_print_fn = matches!(func.as_str(), "print" | "println" | "eprint" | "eprintln");

                // For single-argument print/println calls, pick a type-correct
                // variant (print_int, print_float, print_bool, or the default
                // string-based print) based on the argument's MIR type.
                if is_print_fn && args.len() == 1 {
                    let s = self.operand_to_c(&args[0]);
                    let ty = operand_type(&args[0]);
                    let is_str_literal = s.starts_with('"') || s.starts_with("q_strcat") || s.starts_with("q_int_to_str") || s.starts_with("q_float_to_str") || s.starts_with("q_bool_to_str");

                    let (callee, arg) = match ty {
                        Some(Type::Int) | Some(Type::I8) | Some(Type::I16) | Some(Type::I32) | Some(Type::I64)
                        | Some(Type::I128) | Some(Type::UInt) | Some(Type::U8) | Some(Type::U16)
                        | Some(Type::U32) | Some(Type::U64) | Some(Type::U128) => {
                            let base = if func.starts_with("eprint") { "print_int" } else if func == "println" || func == "eprintln" { "println_int" } else { "print_int" };
                            (base.to_string(), s)
                        }
                        Some(Type::Float) | Some(Type::F32) | Some(Type::F64) => {
                            let base = if func == "println" || func == "eprintln" { "println_float" } else { "print_float" };
                            (base.to_string(), s)
                        }
                        Some(Type::Bool) => {
                            let base = if func == "println" || func == "eprintln" { "println_bool" } else { "print_bool" };
                            (base.to_string(), s)
                        }
                        Some(Type::String) | None if is_str_literal || ty == Some(Type::String) => {
                            (func.clone(), s)
                        }
                        Some(Type::Inferred) | None => {
                            // Check if this operand is a local known to hold a
                            // string (assigned from a string literal or q_strcat).
                            let is_string_local = match &args[0] {
                                MirOperand::Copy(MirPlace::Local(id)) | MirOperand::Move(MirPlace::Local(id)) => string_locals.contains(id),
                                _ => false,
                            };
                            if is_string_local {
                                // Cast the int64_t-stored pointer back to const char*
                                let base = func.clone();
                                (base, format!("(const char*)(intptr_t)({})", s))
                            } else if is_str_literal {
                                (func.clone(), s)
                            } else {
                                // Inferred/unknown locals default to int64_t in C and
                                // are overwhelmingly numeric (loop counters, accumulators).
                                let base = if func == "println" || func == "eprintln" { "println_int" } else { "print_int" };
                                (base.to_string(), s)
                            }
                        }
                        _ => {
                            // Unknown type and not a recognisable string expression —
                            // fall back to the (legacy) pointer-reinterpret cast only
                            // when we have no better information.
                            (func.clone(), format!("(const char*)(intptr_t)({})", s))
                        }
                    };

                    let call = format!("{}({})", callee, arg);
                    return format!("    {};\n    goto label_{};\n", call, target);
                }

                let args_str: Vec<String> = args.iter()
                    .map(|a| {
                        let s = self.operand_to_c(a);
                        // For multi-arg print/println, cast int64_t locals to const char*
                        // in case they hold string pointers
                        if is_print_fn && !s.starts_with('"') && !s.starts_with("q_strcat")
                        {
                            format!("(const char*)(intptr_t)({})", s)
                        } else {
                            s
                        }
                    })
                    .collect();
                let args_joined = args_str.join(", ");

                // Dynamic dispatch via vtable function pointer —
                // encoded as __dyn_dispatch__methodname with the vtable
                // struct as args[0].
                if let Some(method_name) = func.strip_prefix("__dyn_dispatch__") {
                    let receiver_c = args_str.first().cloned().unwrap_or_default();
                    let rest: String = args_str.iter().skip(1).cloned()
                        .collect::<Vec<_>>().join(", ");
                    let call_args = if rest.is_empty() {
                        format!("{}.data", receiver_c)
                    } else {
                        format!("{}.data, {}", receiver_c, rest)
                    };
                    let call = format!("({}.{})({})", receiver_c, method_name, call_args);
                    let stmt = if let Some(dest) = destination {
                        format!("    {} = {};\n    goto label_{};\n", self.place_to_c(dest), call, target)
                    } else {
                        format!("    {};\n    goto label_{};\n", call, target)
                    };
                    return stmt;
                }

                // Sanitize function name to avoid C keyword conflicts.
                // Builtins and extern-declared functions are never mangled —
                // extern functions must match the linked library's exact
                // exported symbol name.
                let safe_func = if matches!(func.as_str(),
                    "print"|"println"|"eprint"|"eprintln"|
                    "q_strcat"|"q_int_to_str"|"q_float_to_str"|"q_bool_to_str"|"q_strlen"
                ) || func == "__quantum_panic"
                  || matches!(func.as_str(), "q_str_len"|"q_str_contains"|"q_str_starts_with"|"q_str_ends_with"|"q_str_to_upper"|"q_str_to_lower"|"q_str_trim"|"q_str_replace"|"q_str_split_first"|"q_str_split_last"|"q_str_index"|"q_str_substr"|"q_str_parse_int"|"q_str_parse_float"|"q_str_from_int"|"q_str_from_float"|"q_str_repeat"|"__q_to_string"|"__q_to_string_p"|"__q_to_string_i64p"|"__q_to_string_f"|"__q_strcat"|"q_alloc"|"q_realloc"|"q_free"|"q_memcpy"|"q_memmove"|"q_read_i64"|"q_write_i64"|"q_read_f64"|"q_write_f64"|"q_file_read_text"|"q_file_write_text"|"q_file_append_text"|"q_file_exists"|"q_file_line_count"|"q_csv_read"|"q_csv_read_col"|"q_strmap_new"|"q_strmap_insert"|"q_strmap_get"|"q_strmap_contains"|"q_strmap_len"|"q_strmap_free")
                  || self.extern_fns.contains(func.as_str()) {
                    func.clone()
                } else {
                    c_safe_name(func)
                };
                let call = format!("{}({})", safe_func, args_joined);

                // Known void functions — never assign return value
                const VOID_FNS: &[&str] = &[
                    "print", "println", "print_int", "print_float", "print_bool",
                    "println_int", "println_float", "println_bool", "print_newline",
                    "eprint", "eprintln", "printf", "puts", "abort", "exit",
                ];
                // Treat all unknown functions as potentially void to avoid C errors.
                // If the result is needed it will be used; if not, the assignment is harmless
                // in C as long as we don't assign void. So: only assign if func is NOT void.
                let is_void = self.void_fns.contains(func.as_str()) || VOID_FNS.contains(&func.as_str());
                // Additional heuristic: if destination local is _lN and the call is to
                // a user function, still try to assign but wrap safely.

                if let Some(dest) = destination {
                    if is_void {
                        // Void function — just call, no assignment
                        format!("    {};\n    goto label_{};\n", call, target)
                    } else {
                        // Non-void (or unknown) — assign result; safe for most cases
                        format!(
                            "    {} = {};\n    goto label_{};\n",
                            self.place_to_c(dest),
                            call,
                            target
                        )
                    }
                } else {
                    format!("    {};\n    goto label_{};\n", call, target)
                }
            }
            MirTerminator::Unreachable => {
                "    abort();\n".to_string()
            }
        }
    }

    fn place_to_c(&self, place: &MirPlace) -> String {
        match place {
            MirPlace::Local(id) => self.local_name(*id),
            MirPlace::Deref(p) => format!("(*{})", self.place_to_c(p)),
            MirPlace::Field(p, field_name) => format!("{}.{}", self.place_to_c(p), field_name),
            MirPlace::Index(p, idx) => format!("{}[{}]", self.place_to_c(p), self.operand_to_c(idx)),
        }
    }

    fn rvalue_to_c(&self, rvalue: &MirRvalue) -> String {
        match rvalue {
            MirRvalue::Use(op) => self.operand_to_c(op),
            MirRvalue::BinaryOp(op, left, right) => {
                let l = self.operand_to_c(left);
                let r = self.operand_to_c(right);
                // String concatenation: use q_strcat when either side is a
                // string literal OR a variable known to hold a String
                // (tracked in self.string_locals for the function currently
                // being generated) — not just literals, so `a + b` between
                // two string parameters/locals also concatenates correctly
                // instead of doing raw pointer arithmetic.
                let left_is_string = l.starts_with('"') || operand_is_string_local(left, &self.string_locals);
                let right_is_string = r.starts_with('"') || operand_is_string_local(right, &self.string_locals);
                if matches!(op, crate::ast::BinOp::Add) && (left_is_string || right_is_string) {
                    format!("q_strcat({}, {})", l, r)
                } else {
                    format!("({} {} {})", l, self.binop_to_c(op), r)
                }
            }
            MirRvalue::UnaryOp(op, operand) => {
                format!("({}{})", self.unop_to_c(op), self.operand_to_c(operand))
            }
            MirRvalue::Ref(_mutable, place) => {
                format!("(&{})", self.place_to_c(place))
            }
            MirRvalue::Len(place) => {
                format!("(sizeof({}) / sizeof(*{}))", self.place_to_c(place), self.place_to_c(place))
            }
            MirRvalue::Cast(op, ty) => {
                format!("(({})({}))", self.type_to_c(ty), self.operand_to_c(op))
            }
            MirRvalue::Aggregate(ops) => {
                let elements = ops.iter()
                    .map(|o| self.operand_to_c(o))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{ {} }}", elements)
            }
            MirRvalue::StructInit(struct_name, fields) => {
                let field_inits = fields.iter()
                    .map(|(name, op)| format!(".{}={}", name, self.operand_to_c(op)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({}){{{}}}", struct_name, field_inits)
            }
            MirRvalue::VTableConstruct { trait_name, struct_name, data_operand, method_names } => {
                let data_c = self.operand_to_c(data_operand);
                let mut field_inits = format!(".data=(void*)&{}", data_c);
                for method in method_names {
                    field_inits.push_str(&format!(
                        ", .{}=__thunk_{}_{}_{}", method, trait_name, struct_name, method
                    ));
                }
                format!("({}){{{}}}", trait_object_typename(trait_name), field_inits)
            }

            MirRvalue::MakeSome(op) => {
                format!("(QuantumOption){{.has_value=true, .value=(int64_t)({})}}", self.operand_to_c(op))
            }
            MirRvalue::MakeNone => {
                "(QuantumOption){.has_value=false, .value=0}".to_string()
            }
            MirRvalue::MakeOk(op) => {
                format!("(QuantumResult){{.is_ok=true, .value=(int64_t)({}), .error=0}}", self.operand_to_c(op))
            }
            MirRvalue::MakeErr(op) => {
                format!("(QuantumResult){{.is_ok=false, .value=0, .error=(const char*)(intptr_t)({})}}", self.operand_to_c(op))
            }
        }
    }
    fn operand_to_c(&self, op: &MirOperand) -> String {
        match op {
            MirOperand::Copy(place) | MirOperand::Move(place) => {
                self.place_to_c(place)
            }
            MirOperand::Constant(c) => self.constant_to_c(c),
        }
    }

    fn constant_to_c(&self, c: &MirConstant) -> String {
        match c {
            MirConstant::Int(n) => n.to_string(),
            MirConstant::Float(f) => f.to_string(),
            MirConstant::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
            MirConstant::String(s) => format!("\"{}\"", s.replace("\"", "\\\"")),
            MirConstant::Null => "NULL".to_string(),
        }
    }

    fn binop_to_c(&self, op: &crate::ast::BinOp) -> &str {
        use crate::ast::BinOp::*;
        match op {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Mod => "%",
            Power => "**", // Note: Need to use pow() function
            Eq => "==",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            And => "&&",
            Or => "||",
            BitAnd => "&",
            BitOr => "|",
            BitXor => "^",
            Lshift => "<<",
            Rshift => ">>",
            Range | RangeInclusive => "..", // Not directly supported in C
        }
    }

    fn unop_to_c(&self, op: &crate::ast::UnOp) -> &str {
        use crate::ast::UnOp::*;
        match op {
            Neg => "-",
            Not => "!",
            BitNot => "~",
        }
    }

    /// Render a single function parameter as C, e.g. `int32_t x` or, for a
    /// fixed-size array parameter, `int32_t arr[5]` — C puts the array size
    /// after the variable name, not after the type, the same layout
    /// difference handled for local variable declarations elsewhere.
    fn param_to_c(&self, param: &MirLocal) -> String {
        if let Type::Array(inner, Some(size)) = &param.ty {
            format!("{} {}[{}]", self.type_to_c(inner), self.local_name(param.id), size)
        } else {
            format!("{} {}", self.type_to_c(&param.ty), self.local_name(param.id))
        }
    }

    fn type_to_c(&self, ty: &Type) -> String {
        match ty {
            Type::Void => "void".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Char => "char".to_string(),
            Type::Int | Type::I32 => "int32_t".to_string(),
            Type::I8 => "int8_t".to_string(),
            Type::I16 => "int16_t".to_string(),
            Type::I64 => "int64_t".to_string(),
            Type::I128 => "__int128".to_string(),
            Type::UInt | Type::U32 => "uint32_t".to_string(),
            Type::U8 => "uint8_t".to_string(),
            Type::U16 => "uint16_t".to_string(),
            Type::U64 => "uint64_t".to_string(),
            Type::U128 => "unsigned __int128".to_string(),
            Type::Float | Type::F64 => "double".to_string(),
            Type::F32 => "float".to_string(),
            Type::String => "const char*".to_string(),
            Type::Array(inner, Some(size)) => {
                format!("{}[{}]", self.type_to_c(inner), size)
            }
            Type::Array(inner, None) | Type::Slice(inner) => {
                format!("{}*", self.type_to_c(inner))
            }
            Type::Reference(inner, _) => {
                format!("{}*", self.type_to_c(inner))
            }
            Type::Named(name) => name.clone(),
            Type::TraitObject(name) => trait_object_typename(name),
            Type::Option(_) => "QuantumOption".to_string(),
            Type::Result(_, _) => "QuantumResult".to_string(),
            Type::Inferred => "int64_t".to_string(),
            Type::Function(_, _) => "int64_t".to_string(),
            _ => "int64_t".to_string(),
        }
    }

    fn local_name(&self, id: usize) -> String {
        format!("_l{}", id)
    }

    fn sanitize_name(&self, name: &str) -> String {
        // Replace invalid C identifier characters
        name.replace("-", "_")
            .replace(".", "_")
            .replace(":", "_")
    }
}
