"""
Loop Static Analyzer — analyzes while/for loops in C/C++ source files.
Outputs JSON for consumption by the dashboard.
"""

import re
import sys
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class LoopInfo:
    kind: str                          # "while" | "for" | "do-while"
    line: int
    col: int
    depth: int                         # 0 = top-level, 1 = nested, …
    is_nested: bool
    condition: str
    body_snippet: str
    computation_types: List[str]       # pointer, arithmetic, bitwise, …
    is_canonicalizable: bool
    canonicalize_reason: str
    file: str


@dataclass
class FileReport:
    file: str
    total_while: int = 0
    total_for: int = 0
    total_do_while: int = 0
    nested_while: int = 0
    nested_for: int = 0
    canonicalizable_while: int = 0
    canonicalizable_for: int = 0
    loops: List[LoopInfo] = field(default_factory=list)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def strip_comments(src: str) -> str:
    """Remove // and /* */ comments while keeping line counts intact."""
    result = []
    i = 0
    while i < len(src):
        if src[i:i+2] == '//':
            while i < len(src) and src[i] != '\n':
                result.append(' ')
                i += 1
        elif src[i:i+2] == '/*':
            while i < len(src) and src[i:i+2] != '*/':
                result.append('\n' if src[i] == '\n' else ' ')
                i += 1
            result.append('  ')
            i += 2
        else:
            result.append(src[i])
            i += 1
    return ''.join(result)


def strip_strings(src: str) -> str:
    """Replace string/char literals with blanks (preserve newlines)."""
    result = []
    i = 0
    while i < len(src):
        if src[i] in ('"', "'"):
            delim = src[i]
            result.append(delim)
            i += 1
            while i < len(src):
                ch = src[i]
                if ch == '\\':
                    result.append(' ')
                    i += 1
                    result.append('\n' if i < len(src) and src[i] == '\n' else ' ')
                    i += 1
                    continue
                result.append('\n' if ch == '\n' else ' ')
                i += 1
                if ch == delim:
                    break
        else:
            result.append(src[i])
            i += 1
    return ''.join(result)


def find_matching_brace(src: str, start: int) -> int:
    """Given index of '{', return index of matching '}'."""
    depth = 0
    i = start
    while i < len(src):
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def find_closing_paren(src: str, start: int) -> int:
    """Given index of '(', return index of matching ')'."""
    depth = 0
    i = start
    while i < len(src):
        if src[i] == '(':
            depth += 1
        elif src[i] == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def classify_computation(body: str) -> List[str]:
    types = []
    # Pointer ops
    if re.search(r'\*\s*\w|\w\s*->\s*\w|\w\s*\[\s*\w', body):
        types.append("pointer")
    # Arithmetic
    if re.search(r'[\+\-\*\/\%]=|[^<>!=]=\s*[^=]|\+\+|--', body):
        types.append("arithmetic")
    # Bitwise
    if re.search(r'[&|^~]|<<|>>', body):
        types.append("bitwise")
    # Function calls
    if re.search(r'\w+\s*\(', body):
        types.append("function_call")
    # Memory allocation
    if re.search(r'\b(malloc|calloc|realloc|free|new|delete)\b', body):
        types.append("memory_alloc")
    # I/O
    if re.search(r'\b(printf|scanf|cout|cin|fwrite|fread|fprintf)\b', body):
        types.append("io")
    # Array access
    if re.search(r'\w\s*\[\s*\w', body):
        types.append("array_access")
    if not types:
        types.append("none_detected")
    return types


# ──────────────────────────────────────────────
# Canonicalization check
# ──────────────────────────────────────────────

def _extract_top_level_statements(body: str) -> List[str]:
    """
    Split a loop body into top-level statements, respecting brace nesting.
    Blocks like if/for/while/{ } are returned as single opaque tokens so
    we can distinguish between an increment that appears at the top level
    versus one that is buried inside a conditional branch.
    """
    stmts = []
    depth = 0
    current = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == '{':
            depth += 1
            current.append(ch)
        elif ch == '}':
            depth -= 1
            current.append(ch)
            if depth == 0:
                stmts.append(''.join(current).strip())
                current = []
        elif ch == ';' and depth == 0:
            current.append(ch)
            stmts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    tail = ''.join(current).strip()
    if tail:
        stmts.append(tail)
    return [s for s in stmts if s]





def _has_early_exit(body: str) -> bool:
    """
    Return True if the loop body contains a top-level early-exit statement:
    return, break, goto, or longjmp/exit calls at depth 0.
    'Top-level' means not buried inside a nested loop or function body —
    but we conservatively flag even those inside conditionals, because any
    reachable early exit makes the trip count unpredictable.
    We ignore 'continue' since it does not shorten the loop, only skips.
    """
    # Strip inner brace-blocks so we only see depth-0 tokens, then scan.
    # We keep the keyword that precedes each block so we can still detect
    # return/break inside if-blocks (which DO affect canonicalizability).
    early_pat = re.compile(
        r'\breturn\b'           # return (with or without value)
        r'|\bbreak\b'           # break
        r'|\bgoto\b'            # goto label
        r'|\b(?:exit|longjmp|abort)\s*\('   # abrupt process/stack exits
    )
    return bool(early_pat.search(body))


def _increment_is_constant_stride(var: str, stmt: str) -> bool:
    """
    Check whether `stmt` is a valid canonical incr-expr for `var` per the
    OpenMP / C canonical loop form definition:

        ++var  |  var++  |  --var  |  var--
        var += incr      |  var -= incr
        var = var + incr |  var = incr + var  |  var = var - incr

    where `incr` must be a loop-invariant INTEGER expression — here we
    conservatively require a compile-time constant (integer literal or
    ALL-CAPS macro).

    EXPLICITLY REJECTED (not part of the canonical incr-expr grammar):
        var *= k    — multiplication: not a linear sequence
        var /= k    — division:       not a linear sequence (k <= j, k /= 2)
        var %= k    — modulo:         not a linear sequence
        var += runtimeVar  — data-dependent stride
    """
    v = re.escape(var)

    # ── ++var / var++ / --var / var-- ─────────────────────────────────────
    if re.search(rf'(?<!\w){v}\s*(\+\+|--)' rf'|(\+\+|--)\s*{v}(?!\w)', stmt):
        return True

    def _is_loop_invariant_incr(rhs: str) -> bool:
        """True if rhs looks like a compile-time constant or ALL-CAPS macro."""
        rhs = rhs.strip().rstrip(';').strip()
        if re.fullmatch(r'[0-9]+[uUlL]*', rhs):            # integer literal
            return True
        if re.fullmatch(r'0[xX][0-9a-fA-F]+[uUlL]*', rhs): # hex literal
            return True
        if re.fullmatch(r'[A-Z_][A-Z0-9_]*', rhs):          # MACRO constant
            return True
        return False

    # ── var += incr  /  var -= incr  (ONLY + and -, NOT * / %) ──────────
    # Explicitly exclude *=, /=, %= — they are not in the canonical grammar.
    m = re.search(rf'(?<!\w){v}\s*([+\-])=(.*)', stmt)
    if m:
        return _is_loop_invariant_incr(m.group(2))

    # ── var = var + incr  /  var = incr + var  /  var = var - incr ───────
    # Only + and - are valid; * and / produce non-linear sequences.
    m = re.search(rf'(?<!\w){v}\s*=\s*{v}\s*([+\-])\s*(.*)', stmt)
    if m:
        return _is_loop_invariant_incr(m.group(2))

    m = re.search(rf'(?<!\w){v}\s*=\s*(.*?)\s*\+\s*{v}(?!\w)', stmt)
    if m:
        return _is_loop_invariant_incr(m.group(1))  # incr + var form

    return False


def _body_has_continue_before_toplevel_increment(var: str, body: str) -> bool:
    """
    Detect the pattern where a `continue` statement inside a conditional
    block makes the bottom-of-body increment NOT always reachable:

        if (condition) {
            var++;
            continue;       ← skips the rest of the body for this iteration
        }
        ...
        var++;              ← looks top-level but only reached if !condition

    If ANY conditional block that contains a `continue` also contains an
    increment of `var`, then a top-level increment below it is only reached
    on iterations that did NOT take that branch.  The overall increment is
    therefore still conditional.

    Returns True if such a pattern is detected (→ loop is NOT canonical).
    """
    inc_pat = re.compile(
        rf'(?<!\w){re.escape(var)}\s*(\+\+|--)'
        rf'|(\+\+|--)\s*{re.escape(var)}(?!\w)'
        rf'|(?<!\w){re.escape(var)}\s*[+\-]='
    )
    continue_pat = re.compile(r'\bcontinue\b')

    # Walk depth-0 brace blocks and flag those that contain BOTH
    # an increment of var AND a continue statement.
    depth = 0
    i = 0
    buf = []
    last_kw = ''
    conditional_kw = re.compile(r'\b(if|else|switch|case)\b')

    while i < len(body):
        ch = body[i]
        if ch == '{':
            if depth == 0:
                last_kw = ''.join(buf).strip()
            depth += 1
            buf.append(ch)
        elif ch == '}':
            depth -= 1
            buf.append(ch)
            if depth == 0:
                block = ''.join(buf)
                is_cond = bool(conditional_kw.search(last_kw))
                if is_cond and inc_pat.search(block) and continue_pat.search(block):
                    return True
                buf = []
                last_kw = ''
        else:
            buf.append(ch)
        i += 1
    return False


def _increment_is_unconditional(var: str, body: str) -> bool:
    """
    Return True only if the loop-variable is incremented:
      (a) at the top level of the body (not inside if/else/switch/case), AND
      (b) not shadowed by a continue in a preceding conditional branch, AND
      (c) by a valid canonical stride (++/-- or += / -= with a constant).

    Rule (b) catches:
        if (cond) { var++; continue; }
        var++;    ← looks top-level but is only reached when !cond
    """
    # First check for the continue-shadowing pattern
    if _body_has_continue_before_toplevel_increment(var, body):
        return False

    conditional_kw = re.compile(r'\b(if|else|switch|case)\b')
    ptr_advance    = re.compile(
        rf'(?<!\w){re.escape(var)}\s*=\s*{re.escape(var)}\s*(->\s*\w+|\.\w+)'
    )

    depth = 0
    i = 0
    buf = []
    last_kw_before_block = ''

    while i < len(body):
        ch = body[i]

        if ch == '{':
            if depth == 0:
                last_kw_before_block = ''.join(buf).strip()
            depth += 1
            buf.append(ch)

        elif ch == '}':
            depth -= 1
            buf.append(ch)
            if depth == 0:
                block_text = ''.join(buf)
                is_conditional = bool(conditional_kw.search(last_kw_before_block))
                if not is_conditional:
                    for sub_stmt in block_text.strip('{}').split(';'):
                        s = sub_stmt.strip()
                        if (not ptr_advance.search(s)
                                and _increment_is_constant_stride(var, s)):
                            return True
                buf = []
                last_kw_before_block = ''

        elif ch == ';' and depth == 0:
            stmt = ''.join(buf).strip()
            buf = []
            if (not ptr_advance.search(stmt)
                    and _increment_is_constant_stride(var, stmt)):
                return True
        else:
            buf.append(ch)

        i += 1

    remaining = ''.join(buf).strip()
    if remaining and depth == 0:
        if (not ptr_advance.search(remaining)
                and _increment_is_constant_stride(var, remaining)):
            return True

    return False


def _extract_bound_variables(rhs: str) -> List[str]:
    """
    Extract all plain identifiers from the RHS of a loop condition that
    could be modified in the body, making the bound non-invariant.

    e.g.  "ptr"           → ["ptr"]
          "r1->len"       → ["r1"]          (struct member, base var)
          "n - 1"         → ["n"]
          "MAX_SIZE"      → ["MAX_SIZE"]
          "r->len + off"  → ["r", "off"]
    We extract every word-token that is not a pure integer literal.
    """
    # Remove integer/hex literals so they don't appear as "variables"
    cleaned = re.sub(r'\b0[xX][0-9a-fA-F]+[uUlL]*\b', '', rhs)
    cleaned = re.sub(r'\b[0-9]+[uUlL]*\b', '', cleaned)
    # All remaining word-tokens are potential variables
    return list(set(re.findall(r'\b([a-zA-Z_]\w*)\b', cleaned)))


def _var_is_modified_in_body(var: str, body: str) -> bool:
    """
    Return True if `var` is modified anywhere in the loop body —
    including inside conditional blocks.  This is used to check that
    the bound expression is loop-invariant.

    Modifications detected:
        var++   var--   ++var   --var
        var OP= expr
        var = expr          (any assignment, including  *ptr-- = ...)
        *var--  *var++      (combined deref+modify via postfix on var)
    """
    v = re.escape(var)
    patterns = [
        rf'(?<!\w){v}\s*(\+\+|--)',           # var++  var--
        rf'(\+\+|--)\s*{v}(?!\w)',             # ++var  --var
        rf'(?<!\w){v}\s*[+\-*/%&|^]?=(?!=)',  # var =  var +=  var -= …
        rf'\*{v}(\+\+|--)',                    # *var++  *var--  (ptr deref + modify)
        rf'(\+\+|--)\*{v}',                   # ++*var  --*var
    ]
    combined = '|'.join(f'(?:{p})' for p in patterns)
    return bool(re.search(combined, body))


def check_canonicalizable_while(cond: str, body: str):
    """
    A while loop is canonicalizable (convertible to a for loop) if ALL of:
      1. Condition is a simple comparison:  var  relop  expr
      2. No early exits (return/break/goto) anywhere in the body.
      3. The loop variable is incremented/decremented UNCONDITIONALLY
         at the top level of the body (not inside if/else/switch).
      4. The increment stride is a compile-time constant (not a runtime
         variable such as the return value of read() or recv()).

    Violations:
      - Conditional increment  → if(c=='\\n'){ i++; }
      - Data-dependent stride  → r += n   (n = read(...))
      - Early exit             → if (n<=0) return r;
      - Pointer walk           → head = head->next
    """
    cond = cond.strip()

    # ── Rule 1: simple condition ──────────────────────────────────────────
    # Reject immediately if condition is compound (&&, ||) — two-variable
    # conditions like  idx_piv < r_piv->len && idx < r->len  mean the loop
    # has multiple simultaneously-advancing counters and is not canonical.
    if re.search(r'&&|\|\|', cond):
        return False, (
            "Condition is compound (&&/||) — loop has multiple termination "
            "variables and cannot be expressed as a single canonical counter"
        )

    # Condition must be exactly:  simple_identifier  relop  rhs
    # where the LHS is a plain variable name (no ->, ., *, array subscript).
    # Struct-field or pointer conditions like  p->len < n  are not canonical.
    m = re.match(r'^(\w+)\s*([<>]=?|[!=]=)\s*(.+)$', cond)
    if not m:
        return False, "Condition is not a simple comparison (var relop expr)"
    var = m.group(1)
    rhs = m.group(3).strip()

    # RHS must not itself be a compound expression or contain logical ops
    if re.search(r'&&|\|\|', rhs):
        return False, "Condition RHS contains logical operators — not a simple bound"

    # ── Rule 2: no early exits ────────────────────────────────────────────
    if _has_early_exit(body):
        return False, (
            "Body contains an early exit (return/break/goto) — "
            "loop may terminate before the condition is false"
        )

    # ── Rule 5: bound must be loop-invariant ─────────────────────────────
    # The RHS of the condition (the bound) must not be modified in the body.
    # e.g.  ptr1 < ptr  — if ptr is modified via *ptr-- in the body,
    # the bound changes every iteration → not canonical.
    # Also catches:  i < n  where n is reassigned inside the loop.
    bound_vars = _extract_bound_variables(rhs)
    for bv in bound_vars:
        if _var_is_modified_in_body(bv, body):
            return False, (
                f"Bound variable '{bv}' in condition is modified inside the "
                f"loop body — the upper bound is not loop-invariant"
            )

    # ── Rules 3 & 4: unconditional, constant-stride increment ────────────
    if _increment_is_unconditional(var, body):
        return True, f"Counter variable '{var}' is modified unconditionally with a constant stride"

    # Distinguish the two failure modes for a better error message
    any_inc = re.compile(
        rf'(?<!\w){re.escape(var)}\s*(\+\+|--)'
        rf'|(\+\+|--)\s*{re.escape(var)}(?!\w)'
        rf'|(?<!\w){re.escape(var)}\s*[\+\-\*\/]='
        rf'|(?<!\w){re.escape(var)}\s*=\s*{re.escape(var)}\s*[\+\-\*\/]'
    )
    if any_inc.search(body):
        return False, (
            f"Counter variable '{var}' is modified conditionally or with a "
            f"data-dependent (runtime) stride — trip count is not statically predictable"
        )
    return False, f"No increment/decrement of '{var}' detected in body"


def check_canonicalizable_for(init: str, cond: str, incr: str):
    """
    A for loop is canonical if:
    - Init declares/assigns a single loop variable
    - Condition is a simple comparison
    - Increment is simple
    """
    if not init.strip():
        return False, "No initializer"
    if not cond.strip():
        return False, "No condition (infinite loop)"
    if not incr.strip():
        return False, "No increment expression"
    m_init = re.search(r'(\w+)\s*=\s*[^,;]+$', init.strip())
    if not m_init:
        return False, "Initializer not a simple assignment"
    var = m_init.group(1)
    m_cond = re.match(rf'^\s*{re.escape(var)}\s*[<>!=]=?\s*.+$', cond.strip())
    if not m_cond:
        return False, f"Condition does not reference init variable '{var}'"
    return True, "Standard canonical form (init; cond; incr)"


# ──────────────────────────────────────────────
# Core parser
# ──────────────────────────────────────────────

def parse_loops(src: str, filename: str) -> FileReport:
    report = FileReport(file=filename)
    clean = strip_strings(strip_comments(src))

    lines = clean.split('\n')
    # Build index: char_pos → line number
    char_to_line = []
    for ln, line in enumerate(lines, 1):
        for _ in line:
            char_to_line.append(ln)
        char_to_line.append(ln)  # for \n

    def pos_to_line(pos):
        if pos < len(char_to_line):
            return char_to_line[pos]
        return len(lines)

    def pos_to_col(pos):
        # walk back to find start of line
        p = pos
        while p > 0 and clean[p-1] != '\n':
            p -= 1
        return pos - p + 1

    def depth_at(pos, loop_ranges):
        """Count how many loop bodies contain this position."""
        d = 0
        for (s, e) in loop_ranges:
            if s < pos < e:
                d += 1
        return d

    loop_ranges = []  # list of (body_start, body_end) for nesting calc

    # ── WHILE loops ──────────────────────────────────────────
    while_pat = re.compile(r'\bwhile\s*\(', re.DOTALL)
    for m in while_pat.finditer(clean):
        kw_start = m.start()
        paren_start = clean.index('(', kw_start)
        paren_end = find_closing_paren(clean, paren_start)
        if paren_end == -1:
            continue
        cond_raw = clean[paren_start+1:paren_end]
        # Find body
        brace_start = clean.find('{', paren_end)
        if brace_start == -1 or (brace_start > paren_end + 50):
            continue
        brace_end = find_matching_brace(clean, brace_start)
        if brace_end == -1:
            continue

        # Check do-while (look back)
        pre = clean[max(0, kw_start-10):kw_start].strip()
        is_do_while = pre.endswith('}') or re.search(r'\bdo\b', clean[max(0, kw_start-20):kw_start])

        body = src[brace_start+1:brace_end]
        line_no = pos_to_line(kw_start)
        col_no = pos_to_col(kw_start)
        depth = depth_at(kw_start, loop_ranges)
        loop_ranges.append((brace_start, brace_end))

        can, reason = check_canonicalizable_while(cond_raw, body)
        comp = classify_computation(body)
        snippet = body.strip()[:120].replace('\n', ' ')

        li = LoopInfo(
            kind="do-while" if is_do_while else "while",
            line=line_no, col=col_no,
            depth=depth, is_nested=(depth > 0),
            condition=cond_raw.strip()[:80],
            body_snippet=snippet,
            computation_types=comp,
            is_canonicalizable=can,
            canonicalize_reason=reason,
            file=filename
        )
        report.loops.append(li)
        if is_do_while:
            report.total_do_while += 1
        else:
            report.total_while += 1
            if depth > 0:
                report.nested_while += 1
            if can:
                report.canonicalizable_while += 1

    # ── FOR loops ────────────────────────────────────────────
    for_pat = re.compile(r'\bfor\s*\(', re.DOTALL)
    for m in for_pat.finditer(clean):
        kw_start = m.start()
        paren_start = clean.index('(', kw_start)
        paren_end = find_closing_paren(clean, paren_start)
        if paren_end == -1:
            continue
        header = clean[paren_start+1:paren_end]
        # Split into init ; cond ; incr
        parts = header.split(';', 2)
        init = parts[0].strip() if len(parts) > 0 else ''
        cond = parts[1].strip() if len(parts) > 1 else ''
        incr = parts[2].strip() if len(parts) > 2 else ''

        brace_start = clean.find('{', paren_end)
        if brace_start == -1 or (brace_start > paren_end + 50):
            continue
        brace_end = find_matching_brace(clean, brace_start)
        if brace_end == -1:
            continue

        body = src[brace_start+1:brace_end]
        line_no = pos_to_line(kw_start)
        col_no = pos_to_col(kw_start)
        depth = depth_at(kw_start, loop_ranges)
        loop_ranges.append((brace_start, brace_end))

        can, reason = check_canonicalizable_for(init, cond, incr)
        comp = classify_computation(body)
        snippet = body.strip()[:120].replace('\n', ' ')

        li = LoopInfo(
            kind="for",
            line=line_no, col=col_no,
            depth=depth, is_nested=(depth > 0),
            condition=cond[:80],
            body_snippet=snippet,
            computation_types=comp,
            is_canonicalizable=can,
            canonicalize_reason=reason,
            file=filename
        )
        report.loops.append(li)
        report.total_for += 1
        if depth > 0:
            report.nested_for += 1
        if can:
            report.canonicalizable_for += 1

    report.loops.sort(key=lambda l: l.line)
    return report


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def analyze(paths):
    reports = []
    for p in paths:
        try:
            src = Path(p).read_text(errors='replace')
            r = parse_loops(src, str(p))
            reports.append(asdict(r))
        except Exception as e:
            reports.append({"file": str(p), "error": str(e)})
    return reports


def format_txt_report(results: list) -> str:
    """
    Format analysis results as a human-readable plain-text report.
    """
    SEP  = "=" * 72
    SEP2 = "-" * 72
    lines = []

    lines.append(SEP)
    lines.append("  LOOP STATIC ANALYSIS REPORT")
    lines.append(f"  Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(SEP)

    # ── Global summary across all files ──────────────────────────────────
    total_files   = len(results)
    grand_while   = sum(r.get("total_while",   0) for r in results)
    grand_for     = sum(r.get("total_for",     0) for r in results)
    grand_do      = sum(r.get("total_do_while",0) for r in results)
    grand_nw      = sum(r.get("nested_while",  0) for r in results)
    grand_nf      = sum(r.get("nested_for",    0) for r in results)
    grand_cw      = sum(r.get("canonicalizable_while", 0) for r in results)
    grand_cf      = sum(r.get("canonicalizable_for",   0) for r in results)
    grand_total   = grand_while + grand_for + grand_do
    grand_nested  = grand_nw + grand_nf
    grand_canon   = grand_cw + grand_cf

    lines.append("")
    lines.append("GLOBAL SUMMARY")
    lines.append(SEP2)
    lines.append(f"  Files analysed          : {total_files}")
    lines.append(f"  Total loops             : {grand_total}")
    lines.append(f"    While loops           : {grand_while}  (nested: {grand_nw})")
    lines.append(f"    For loops             : {grand_for}  (nested: {grand_nf})")
    lines.append(f"    Do-while loops        : {grand_do}")
    lines.append(f"  Nested loops (total)    : {grand_nested}")
    lines.append(f"  Canonicalizable while   : {grand_cw}")
    lines.append(f"  Canonicalizable for     : {grand_cf}")
    if grand_total:
        pct = grand_canon / grand_total * 100
        lines.append(f"  Canonicalizability rate : {grand_canon}/{grand_total}  ({pct:.1f}%)")
    lines.append("")

    # ── Per-file sections ─────────────────────────────────────────────────
    for r in results:
        if "error" in r:
            lines.append(SEP)
            lines.append(f"FILE: {r['file']}")
            lines.append(f"  ERROR: {r['error']}")
            lines.append("")
            continue

        total = r["total_while"] + r["total_for"] + r["total_do_while"]
        nested = r["nested_while"] + r["nested_for"]
        canon  = r["canonicalizable_while"] + r["canonicalizable_for"]

        lines.append(SEP)
        lines.append(f"FILE: {r['file']}")
        lines.append(SEP2)
        lines.append(f"  Total loops             : {total}")
        lines.append(f"    While loops           : {r['total_while']}  (nested: {r['nested_while']},  canonicalizable: {r['canonicalizable_while']})")
        lines.append(f"    For loops             : {r['total_for']}  (nested: {r['nested_for']},  canonicalizable: {r['canonicalizable_for']})")
        lines.append(f"    Do-while loops        : {r['total_do_while']}")
        lines.append(f"  Nested loops (total)    : {nested}")
        if total:
            lines.append(f"  Canonicalizability      : {canon}/{total}  ({canon/total*100:.1f}%)")
        lines.append("")

        # Per-loop detail table
        loops = r.get("loops", [])
        if not loops:
            lines.append("  (no loops found)")
            lines.append("")
            continue

        lines.append(f"  {'#':<4} {'KIND':<10} {'LINE':>5} {'DEPTH':>5}  {'CANONICAL':<10}  CONDITION")
        lines.append(f"  {'-'*4} {'-'*10} {'-'*5} {'-'*5}  {'-'*10}  {'-'*30}")

        for idx, loop in enumerate(loops, 1):
            can_str  = "Yes" if loop["is_canonicalizable"] else "No"
            nest_str = f"d={loop['depth']}"
            cond     = loop["condition"][:45] + ("…" if len(loop["condition"]) > 45 else "")
            lines.append(f"  {idx:<4} {loop['kind']:<10} {loop['line']:>5} {nest_str:>5}  {can_str:<10}  {cond}")

        lines.append("")
        lines.append("  LOOP DETAILS")
        lines.append("  " + SEP2)

        for idx, loop in enumerate(loops, 1):
            lines.append(f"  [{idx}] {loop['kind'].upper()}  at line {loop['line']}, col {loop['col']}")
            lines.append(f"      Depth        : {loop['depth']} {'(nested)' if loop['is_nested'] else '(top-level)'}")
            lines.append(f"      Condition    : {loop['condition'] or '—'}")
            comp = ", ".join(loop.get("computation_types", []))
            lines.append(f"      Computation  : {comp or '—'}")
            canon_str = "YES" if loop["is_canonicalizable"] else "NO"
            lines.append(f"      Canonical    : {canon_str}  —  {loop['canonicalize_reason']}")
            snippet = loop.get("body_snippet", "").strip()
            if snippet:
                lines.append(f"      Body snippet : {snippet[:80]}{'…' if len(snippet)>80 else ''}")
            lines.append("")

    lines.append(SEP)
    lines.append("  END OF REPORT")
    lines.append(SEP)
    return "\n".join(lines)


if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else ['.']
    files = []
    for t in targets:
        p = Path(t)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(p.rglob('*.c'))
            files.extend(p.rglob('*.cpp'))
            files.extend(p.rglob('*.cc'))
            files.extend(p.rglob('*.cxx'))

    results = analyze(files)

    # ── JSON output (stdout) ──────────────────────────────────────────────
    print(json.dumps(results, indent=2))

    # ── Plain-text report (file) ──────────────────────────────────────────
    report_path = Path("loop_analysis_report.txt")
    report_text = format_txt_report(results)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[Report written to: {report_path.resolve()}]", file=sys.stderr)
