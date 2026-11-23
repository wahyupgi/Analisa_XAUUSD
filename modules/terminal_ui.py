# -*- coding: utf-8 -*-
"""
modules/terminal_ui.py
Formatter warna terminal terpusat berbasis Rich (UI-only, aman untuk TTY interaktif & fallback polos).
"""
import builtins
import os
import re
import shutil
import sys
import textwrap

try:
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    _RICH_AVAILABLE = True
except Exception:
    Console = None
    Live = None
    Text = None
    _RICH_AVAILABLE = False

_CONSOLE_STREAM = None
_CONSOLE_INSTANCE = None

_RE_RISK_HEADER = re.compile(r"^E\.\s+RISK WARNING\b")
_RE_BANNER_DIVIDER_HEADER = re.compile(r"^={3,}\s+.+\s+={3,}$")
_RE_BRACKET_NUM_HEADER = re.compile(r"^\[\d{1,2}\.\s+[A-Z0-9 /&_()\-—+]+\]$")
_RE_SUB_HEADER_PARENS_LOWER = re.compile(
    r"^(?:\d{1,2}|[A-F])\.\s+[A-Z][A-Z0-9 .,&/+—-]+\([^)]*[a-z][^)]*\):?$"
)
_RE_UPPER_SECTION_HEADER = re.compile(
    r"^(?:\d{1,2}|[A-F])\.\s+[A-Z][A-Z0-9 .,&/+—-]+(?:\s*\([A-Z0-9 /&_.,\-—+]+\))?:?$"
)
_RE_FUNDAMENTAL_HEADER = re.compile(r"^\[RINGKASAN KONTEKS FUNDAMENTAL\b.*\]:?$")

_SUB_SECTION_TITLES = frozenset({
    "ANALYSIS",
    "MONITORING",
    "OTHER",
    "UTILITY TAMBAHAN",
    "ACTIVE SETUP",
    "RADAR CANDIDATE",
    "ENTRY STATUS",
    "SMART MONITOR",
    "FINAL",
    "LIFECYCLE DETAIL",
    "VALIDATION",
    "BLOCKERS",
    "ALERT PRIORITY BREAKDOWN",
    "US NEWS RADAR & GOLD MARKET NEWS",
    "SMC SCANNER DETAIL",
    "SNR SCANNER DETAIL",
    "SUPPORT ZONES",
    "RESISTANCE ZONES",
    "BREAKOUT / BREAKDOWN & RECLAIM WATCH",
    "FINAL DECISION",
    "BREAKDOWN PER KELAS RISIKO",
    "BREAKDOWN PER MODEL AI",
})

_INLINE_RULES = (
    # Priority 1: INVALID / CONFLICT / RISK ALERT (bold bright_red)
    (re.compile(r"\[(?:RISK WARNING|ACCOUNT RISK|CRITICAL WARNING[^\]]*|SNR CRITICAL[^\]]*|INVALID|UNAVAILABLE|CONFLICT[^\]]*)\]"), "bold bright_red", 0),
    (re.compile(r"\b(?:INVALID\s+—\s+BROKEN|INVALID\s+ZONE|SL/TP\s+MISMATCH|RISK\s+ALERT|CRITICAL\s+WARNING|HIGH\s+RISK|TANPA\s+STOP\s+LOSS|TANPA\s+SL|WITHOUT\s+SL|SL\s+MISSING!?|UNBOUNDED|DATA\s+UNAVAILABLE|TIDAK\s+VALID|BLOCKED\s+BY\s+SNR|SL\s+INVALID|RISIKO\s+TINGGI)\b"), "bold bright_red", 0),
    (re.compile(r"\b(?:INVALIDATED|INVALID|CONFLICT|KONFLIK|DISCONNECTED|UNAVAILABLE|PLACED_MISMATCH|EXPIRED|BROKEN|DITOLAK|GAGAL|ERROR|FAILED|FAIL|JEBOL|BATAL)\b"), "bold bright_red", 0),
    (re.compile(r"(?i:NEWS\s+RISK\s*:\s*|IMPACT\s*:\s*|NEWS\s+)(HIGH)\b"), "bold bright_red", 1),

    # Priority 2: WAIT / APPROACHING (yellow)
    (re.compile(r"\b(?:NO\s+MATCH|OTHER\s+ORDERS?(?::\s*\d+)?|FAR\s+ZONE\s*/\s*REFERENCE\s+ONLY|FAR\s+ZONE|ACTIVE\s+FOCUS:\s*NONE|NOT\s+PLACED|NOT\s+CONFIRMED|NO\s+ENTRY|NO\s+SETUP|NO\s+ACTIVE\s+(?:SETUP|MONITORING\s+ZONE)|POTENTIAL\s+BREAK(?:OUT|DOWN)|STALE(?:_CACHE|\s+DATA)?|PROVISIONAL(?:\s+ONLY)?|ZONE_TOUCHED|TOUCHED\s+PREVIOUSLY|IN\s+ZONE|PARTIALLY_MITIGATED)\b"), "yellow", 0),
    (re.compile(r"\b(?:APPROACHING|WAITING|WAIT|UNCONFIRMED|UNVERIFIED|RUNNING|PARTIAL|SEPARATED|WEAK|COUNTERTREND|TERLEWAT|RECLAIMED|HISTORICAL_TOUCH|HISTORY_UNCERTAIN)\b"), "yellow", 0),
    (re.compile(r"(?i:NEWS\s+RISK\s*:\s*|IMPACT\s*:\s*|NEWS\s+)(MEDIUM|UNKNOWN)\b"), "yellow", 1),

    # Priority 3: BUY / BULLISH & READY / CONFIRMED (bold green)
    (re.compile(r"\b(?:MT5\s+PLACED|PENDING\s+READY|POTENTIAL\s+BULLISH|BUY\s+(?:LIMIT|STOP)|HIT\s+TP[12]?|CONFIRMED\s+ENTRY)\b"), "bold green", 0),
    (re.compile(r"\b(?:BUY|BULLISH|READY|CONFIRMED|VALID|CONNECTED|PASSED|PASS|VERIFIED|ALIGNED|PLACED_VERIFIED|DIAMBIL|BERHASIL|LOLOS|AMAN)\b"), "bold green", 0),
    (re.compile(r"(?i:NEWS\s+RISK\s*:\s*|IMPACT\s*:\s*)(LOW)\b"), "bold green", 1),
    (re.compile(r"(?<![\w.])\+\d+(?:\.\d+)?R\b"), "bold green", 0),

    # Priority 4: SELL / BEARISH (bold red)
    (re.compile(r"\b(?:POTENTIAL\s+BEARISH|SELL\s+(?:LIMIT|STOP)|HIT\s+SL)\b"), "bold red", 0),
    (re.compile(r"\b(?:SELL|BEARISH)\b"), "bold red", 0),
    (re.compile(r"(?<![\w.])-\d+(?:\.\d+)?R\b"), "bold red", 0),

    # Priority 5: Sub-header inline prefix (bold yellow)
    (re.compile(r"^PENDING\s+PLAN\b"), "bold yellow", 0),

    # Priority 6: Harga / Informasi MT5 (cyan / bold cyan)
    (re.compile(r"\[(?:SMART MONITOR|UNIFIED MONITOR|SNR MONITOR|SCANNER)\]"), "bold cyan", 0),
    (re.compile(r"\[(?:MT5[^\]]*|LIVE MT5|IDENTITAS AKUN MT5)\]"), "cyan", 0),
    (re.compile(r"\b(?i:Bid|Ask|Mid)\s*[:=]?\s*\d+(?:,\d{3})*(?:\.\d+)?\b"), "cyan", 0),
    (re.compile(r"\b(?i:Spread|Spr)\s*[:=]?\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:USD|pts?))?(?:\s*\(\d+(?:\.\d+)?\s*MT5\s*pts\))?"), "cyan", 0),
    (re.compile(r"\b(?i:Ticket)\s*#\d+\b"), "cyan", 0),
    (re.compile(r"\bMT5\b|\bXAUUSD[A-Za-z0-9._-]*\b"), "cyan", 0),
    (re.compile(r"[+-]?\d+(?:,\d{3})*\.\d{2}\s*(?:USD|USC|EUR|GBP|IDR)\b"), "cyan", 0),
    (re.compile(r"\b\d{4,5}\.\d{2}\b(?!%)"), "cyan", 0),
)


def should_use_color(stream=None) -> bool:
    """Aktifkan warna hanya jika Rich tersedia, NO_COLOR/TERM=dumb non-aktif, dan output ke TTY interaktif."""
    if not _RICH_AVAILABLE:
        return False
    if os.environ.get("NO_COLOR", "") != "":
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    target = sys.stdout if stream is None else stream
    try:
        return bool(target is not None and callable(getattr(target, "isatty", None)) and target.isatty())
    except Exception:
        return False


def _get_console(target):
    global _CONSOLE_STREAM, _CONSOLE_INSTANCE
    if _CONSOLE_INSTANCE is None or _CONSOLE_STREAM is not target:
        _CONSOLE_STREAM = target
        _CONSOLE_INSTANCE = Console(
            file=target,
            highlight=False,
            markup=False,
            soft_wrap=True,
            no_color=False,
        )
    return _CONSOLE_INSTANCE


def _classify_header_style(stripped: str):
    """Kembalikan style Rich jika baris adalah judul bagian murni, selain itu None."""
    if not stripped:
        return None
    no_trail = stripped[:-1].rstrip() if stripped.endswith(":") else stripped
    if ": " in no_trail or " :" in no_trail:
        return None

    if _RE_RISK_HEADER.match(stripped):
        return "bold bright_red"

    if stripped.startswith("### ") or _RE_BANNER_DIVIDER_HEADER.match(stripped) or _RE_BRACKET_NUM_HEADER.match(stripped):
        return "bold cyan"

    if _RE_SUB_HEADER_PARENS_LOWER.match(stripped):
        return "bold yellow"

    if _RE_UPPER_SECTION_HEADER.match(stripped):
        return "bold cyan"

    if (
        ("XAUUSD" in stripped and any(k in stripped for k in ("SCANNER", "MONITOR", "RADAR", "AUDIT")))
        or stripped.startswith((
            "AUDIT FAKTA AKUN MT5",
            "📖 COMMAND MENU",
            "📖 JURNAL SINYAL",
            "📡 MEMULAI SNR ADVANCED MONITOR",
            "📊 [MONITORING STATUS SINYAL]",
        ))
    ):
        return "bold cyan"

    if (
        no_trail in _SUB_SECTION_TITLES
        or stripped.startswith(("📊 RINGKASAN PERFORMA", "🎯 RINGKASAN KHUSUS"))
        or _RE_FUNDAMENTAL_HEADER.match(stripped)
        or (stripped.startswith("#") and "| Waktu Lokal" in stripped)
    ):
        return "bold yellow"

    return None


def _apply_pattern(text, line: str, occupied: bytearray, pattern, style: str, group: int = 0) -> None:
    for m in pattern.finditer(line):
        start, end = m.span(group)
        if start < 0 or start >= end:
            continue
        if any(occupied[start:end]):
            continue
        occupied[start:end] = b"\x01" * (end - start)
        text.stylize(style, start, end)


def style_line(line: str):
    """Memformat satu baris output menjadi rich.text.Text tanpa mengubah karakter asli."""
    if not _RICH_AVAILABLE:
        return line
    text = Text(line)
    stripped = line.strip()
    if not stripped:
        return text

    header_style = _classify_header_style(stripped)
    if header_style is not None:
        text.stylize(header_style)
        return text

    occupied = bytearray(len(line))
    for pattern, style, group in _INLINE_RULES:
        _apply_pattern(text, line, occupied, pattern, style, group)
    return text


def format_colored_text(text: str):
    """Memformat teks multi-baris menjadi rich.text.Text (atau string polos saat fallback)."""
    if not _RICH_AVAILABLE:
        return text
    lines = text.split("\n")
    return Text("\n").join(style_line(ln) for ln in lines)


def print_terminal(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
    """Cetak ke terminal dengan pewarnaan Rich otomatis pada TTY interaktif, dan fallback polos jika non-TTY."""
    raw = sep.join(str(a) for a in args)
    target = sys.stdout if file is None else file
    if not should_use_color(target):
        builtins.print(raw, end=end, file=target, flush=flush)
        return

    cr_written = False
    try:
        to_render = raw
        if to_render.startswith("\r"):
            target.write("\r")
            cr_written = True
            to_render = to_render[1:]
        if to_render:
            _get_console(target).print(
                format_colored_text(to_render),
                end="",
                soft_wrap=True,
                crop=False,
            )
        if end:
            target.write(end)
        if flush:
            target.flush()
    except Exception:
        fallback_text = raw[1:] if cr_written and raw.startswith("\r") else raw
        builtins.print(fallback_text, end=end, file=target, flush=flush)


def _enable_windows_ansi() -> bool:
    """Aktifkan mode ANSI VT100 pada Windows PowerShell/CMD jika tersedia."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & 0x0004:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def get_terminal_width(default: int = 80, min_width: int = 44, max_width: int = 120) -> int:
    """Ambil lebar terminal aman via shutil.get_terminal_size() agar baris tidak auto-wrap di Windows."""
    try:
        cols = int(shutil.get_terminal_size(fallback=(default, 24)).columns)
    except Exception:
        cols = default
    if cols <= 0:
        cols = default
    safe_cols = max(32, cols - 2)
    if safe_cols < min_width:
        return safe_cols
    return min(safe_cols, max_width)


def get_terminal_height(default: int = 24) -> int:
    """Ambil tinggi terminal via shutil.get_terminal_size()."""
    try:
        rows = int(shutil.get_terminal_size(fallback=(80, default)).lines)
        return rows if rows > 0 else default
    except Exception:
        return default


def truncate_clean(text: str, max_len: int) -> str:
    """Potong teks secara terkontrol pada batas pemisah/spasi agar angka atau kata penting tidak terpotong di tengah."""
    clean = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
    if max_len <= 0:
        return ""
    if len(clean) <= max_len:
        return clean
    if max_len <= 3:
        return clean[:max_len]
    limit = max_len - 3
    head = clean[:limit]
    for sep in (" | ", "; ", ", ", " "):
        pos = head.rfind(sep)
        if pos >= max(8, limit // 2):
            return head[:pos].rstrip(" |;,:-") + "..."
    return head.rstrip(" |;,:-") + "..."


def wrap_dashboard_field(prefix: str, value: str, width: int, max_lines=None):
    """
    Format pasangan `prefix + value` agar setiap baris <= width dengan wrapping rapi
    pada batas segmen ` | ` atau spasi TANPA truncation `...` (kecuali max_lines dibatasi eksplisit).
    """
    w = max(32, int(width))
    clean_val = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    avail = max(12, w - len(prefix))
    if len(clean_val) <= avail:
        return [prefix + clean_val]

    segments = [seg.strip() for seg in clean_val.split(" | ") if seg.strip()]
    wrapped = []
    cur = ""
    for seg in segments:
        candidate = f"{cur} | {seg}" if cur else seg
        if len(candidate) <= avail:
            cur = candidate
            continue
        if cur:
            wrapped.append(cur)
            cur = ""
        if len(seg) <= avail:
            cur = seg
        else:
            chunks = textwrap.wrap(
                seg,
                width=avail,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [seg]
            for ch in chunks[:-1]:
                wrapped.append(ch)
            cur = chunks[-1]
    if cur:
        wrapped.append(cur)

    if not wrapped:
        wrapped = [clean_val]

    if max_lines is not None and max_lines > 0 and len(wrapped) > max_lines:
        kept = wrapped[: max_lines - 1]
        remainder = " | ".join(wrapped[max_lines - 1 :])
        kept.append(truncate_clean(remainder, avail))
        wrapped = kept

    indent = " " * len(prefix)
    out = [prefix + wrapped[0]]
    for cont in wrapped[1:]:
        out.append(indent + cont)
    return out


def fit_dashboard_text(text: str, width: int = None) -> str:
    """Pastikan seluruh baris dashboard (termasuk divider) tidak melebihi lebar terminal dengan wrap rapi (tanpa `...`)."""
    w = get_terminal_width() if width is None else max(32, int(width))
    fitted = []
    for raw_ln in str(text or "").splitlines():
        s_ln = raw_ln.strip()
        if len(s_ln) >= 10 and len(set(s_ln)) == 1 and s_ln[0] in ("=", "-", "─"):
            fitted.append(s_ln[0] * w)
        elif len(raw_ln) > w:
            colon_idx = raw_ln.find(": ")
            if 0 < colon_idx <= min(24, w // 2):
                pref = raw_ln[: colon_idx + 2]
                val = raw_ln[colon_idx + 2 :]
                fitted.extend(wrap_dashboard_field(pref, val, w, max_lines=None))
            else:
                lead_len = len(raw_ln) - len(raw_ln.lstrip(" "))
                lead = raw_ln[:lead_len]
                body = raw_ln[lead_len:]
                fitted.extend(wrap_dashboard_field(lead, body, w, max_lines=None))
        else:
            fitted.append(raw_ln)
    return "\n".join(fitted)


def supports_unicode_glyphs(stream=None) -> bool:
    """True jika encoding stream output mendukung box-drawing dan simbol panah TradingView."""
    target = sys.stdout if stream is None else stream
    enc = getattr(target, "encoding", None) or "utf-8"
    try:
        "\u250c\u2500\u2510\u2502\u2514\u2518\u251c\u2524\u2534\u25b2\u25bc\u25cf".encode(enc)
        return True
    except Exception:
        return False


_BIG_DIGIT_GLYPHS = {
    "0": ("┌─┐", "│ │", "└─┘"),
    "1": (" ┐ ", " │ ", "─┴─"),
    "2": ("──┐", "┌─┘", "└──"),
    "3": ("──┐", " ─┤", "──┘"),
    "4": ("│ │", "└─┤", "  │"),
    "5": ("┌──", "└─┐", "──┘"),
    "6": ("┌──", "├─┐", "└─┘"),
    "7": ("──┐", "  │", "  │"),
    "8": ("┌─┐", "├─┤", "└─┘"),
    "9": ("┌─┐", "└─┤", "──┘"),
    ".": (" ", " ", "."),
    ",": (" ", " ", ","),
    "-": ("   ", "───", "   "),
    " ": (" ", " ", " "),
}


def render_big_price_digits(price_str: str) -> list:
    """Render angka harga ke 3 baris box-drawing presisi 1-kolom yang tajam di Windows PowerShell/Terminal."""
    s = str(price_str or "---").strip()
    if not s:
        s = "---"
    rows = [[], [], []]
    for ch in s:
        g = _BIG_DIGIT_GLYPHS.get(ch, _BIG_DIGIT_GLYPHS["-"])
        rows[0].append(g[0])
        rows[1].append(g[1])
        rows[2].append(g[2])
    return [" ".join(rows[0]), " ".join(rows[1]), " ".join(rows[2])]


_ANSI_SGR = {
    "bold bright_green": "\x1b[1;92m",
    "bold bright_red": "\x1b[1;91m",
    "bold yellow": "\x1b[1;93m",
    "bold bright_white": "\x1b[1;97m",
    "bold cyan": "\x1b[1;96m",
    "cyan": "\x1b[36m",
    "dim": "\x1b[2m",
    "dim cyan": "\x1b[2;36m",
    "dim green": "\x1b[2;32m",
}


def _ansi_wrap(segment: str, style: str) -> str:
    code = _ANSI_SGR.get(style)
    return f"{code}{segment}\x1b[0m" if (code and segment) else segment


def build_tradingview_dashboard_renderable(plain_text: str, widget_meta=None, stream=None, ansi_mode: bool = False):
    """
    Mewarnai dashboard gabungan TradingView Price Widget + Smart Monitor dengan Rich Text
    (atau ANSI SGR kompatibel Windows bila Rich tidak tersedia).
    """
    meta = widget_meta if isinstance(widget_meta, dict) else {}
    lines = str(plain_text or "").splitlines()
    if not meta:
        if ansi_mode:
            return lines
        return format_colored_text(plain_text) if should_use_color(stream) else (Text(plain_text) if _RICH_AVAILABLE else plain_text)

    feed_st = str(meta.get("feed_status") or "OPEN").upper()
    price_dir = str(meta.get("price_direction") or "FLAT").upper()
    chg_dir = str(meta.get("change_direction") or "FLAT").upper()
    spread_wide = bool(meta.get("spread_wide", False))

    if feed_st == "DISCONNECTED":
        price_style = "bold bright_red"
        status_style = "bold bright_red"
    elif feed_st == "STALE":
        price_style = "bold yellow"
        status_style = "bold yellow"
    elif feed_st == "CLOSED":
        price_style = "bold yellow"
        status_style = "bold yellow"
    else:
        status_style = "bold bright_green"
        if price_dir == "UP":
            price_style = "bold bright_green"
        elif price_dir == "DOWN":
            price_style = "bold bright_red"
        elif chg_dir == "UP":
            price_style = "bold bright_green"
        elif chg_dir == "DOWN":
            price_style = "bold bright_red"
        else:
            price_style = "bold bright_white"

    change_style = (
        "bold bright_green" if chg_dir == "UP"
        else ("bold bright_red" if chg_dir == "DOWN" else "dim")
    )

    header_idx = meta.get("header_idx")
    sym_span = meta.get("sym_span")
    status_span = meta.get("status_span")
    time_span = meta.get("time_span")
    price_rows = {int(r[0]): (int(r[1]), str(r[2])) for r in (meta.get("price_rows") or [])}
    compact_idx = meta.get("compact_price_idx")
    compact_price_span = meta.get("compact_price_span")
    compact_chg_span = meta.get("compact_change_span")

    if ansi_mode:
        ansi_lines = []
        for idx, ln in enumerate(lines):
            if idx == header_idx and sym_span and status_span:
                s0, s1 = sym_span
                b0, b1 = status_span
                out_ln = (
                    ln[:s0]
                    + _ansi_wrap(ln[s0:s1], "bold bright_white")
                    + ln[s1:b0]
                    + _ansi_wrap(ln[b0:b1], status_style)
                    + ln[b1:]
                )
                ansi_lines.append(out_ln)
            elif idx in price_rows:
                split_col, side_role = price_rows[idx]
                left_part = ln[:split_col]
                right_part = ln[split_col:]
                r_style = change_style if side_role == "change" else ("bold bright_red" if (side_role == "meta" and feed_st == "DISCONNECTED") else "cyan")
                ansi_lines.append(_ansi_wrap(left_part, price_style) + _ansi_wrap(right_part, r_style))
            elif idx == compact_idx and compact_price_span:
                p0, p1 = compact_price_span
                if compact_chg_span and compact_chg_span[0] >= p1:
                    c0, c1 = compact_chg_span
                    ansi_lines.append(
                        ln[:p0] + _ansi_wrap(ln[p0:p1], price_style)
                        + ln[p1:c0] + _ansi_wrap(ln[c0:c1], change_style) + ln[c1:]
                    )
                else:
                    ansi_lines.append(ln[:p0] + _ansi_wrap(ln[p0:p1], price_style) + ln[p1:])
            else:
                ansi_lines.append(ln)
        return ansi_lines

    if not should_use_color(stream):
        return Text(plain_text) if _RICH_AVAILABLE else plain_text

    rendered_lines = []
    for idx, ln in enumerate(lines):
        if idx == header_idx:
            t = Text(ln)
            if sym_span:
                t.stylize("bold bright_white", sym_span[0], min(len(ln), sym_span[1]))
            if status_span:
                t.stylize(status_style, status_span[0], min(len(ln), status_span[1]))
            if time_span:
                t.stylize("dim", time_span[0], min(len(ln), time_span[1]))
            rendered_lines.append(t)
        elif idx in price_rows:
            split_col, side_role = price_rows[idx]
            split_c = min(len(ln), max(0, split_col))
            t = Text(ln)
            if split_c > 0:
                t.stylize(price_style, 0, split_c)
            if split_c < len(ln):
                right = ln[split_c:]
                if side_role == "change":
                    d1_pos = right.find("D1:")
                    if d1_pos > 0:
                        t.stylize(change_style, split_c, split_c + d1_pos)
                        t.stylize("dim cyan", split_c + d1_pos, len(ln))
                    elif feed_st in ("DISCONNECTED", "STALE"):
                        t.stylize(status_style, split_c, len(ln))
                    else:
                        t.stylize(change_style, split_c, len(ln))
                elif side_role == "quote":
                    t.stylize("cyan", split_c, len(ln))
                    if spread_wide:
                        spr_pos = right.find("Spr:")
                        if spr_pos >= 0:
                            t.stylize("bold bright_red", split_c + spr_pos, len(ln))
                elif side_role == "meta":
                    if feed_st in ("DISCONNECTED", "STALE", "CLOSED"):
                        t.stylize(status_style, split_c, len(ln))
                    else:
                        tick_pos = right.find("Tick ")
                        if tick_pos > 0:
                            t.stylize("dim green", split_c, split_c + tick_pos)
                            t.stylize(price_style, split_c + tick_pos, len(ln))
                        else:
                            t.stylize("dim green", split_c, len(ln))
            rendered_lines.append(t)
        elif idx == compact_idx:
            t = Text(ln)
            if compact_price_span:
                t.stylize(price_style, compact_price_span[0], min(len(ln), compact_price_span[1]))
            if compact_chg_span:
                t.stylize(change_style, compact_chg_span[0], min(len(ln), compact_chg_span[1]))
            rendered_lines.append(t)
        else:
            rendered_lines.append(style_line(ln))
    return Text("\n").join(rendered_lines)


def can_use_live_dashboard(stream=None) -> bool:
    """True jika terminal mendukung in-place refresh (Windows PowerShell / TTY interaktif)."""
    if os.environ.get("NO_LIVE_DASHBOARD", "").strip() not in ("", "0", "false", "False"):
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    target = sys.stdout if stream is None else stream
    try:
        if target is None or not callable(getattr(target, "isatty", None)) or not target.isatty():
            return False
    except Exception:
        return False
    if _RICH_AVAILABLE and Live is not None:
        return True
    return _enable_windows_ansi()


class LiveTerminalDashboard:
    """
    Dashboard terminal in-place refresh yang kompatibel dengan Windows/PowerShell
    (memakai Rich Live pada TTY interaktif, VT100 aman sebagai cadangan, serta
    fallback output ringkas anti-spam jika terminal tidak mendukung refresh).
    """

    def __init__(self, stream=None, force_fallback: bool = False):
        self.stream = sys.stdout if stream is None else stream
        self.supports_live = (not force_fallback) and can_use_live_dashboard(self.stream)
        self._live = None
        self._use_rich_live = bool(self.supports_live and _RICH_AVAILABLE and Live is not None)
        self._use_ansi_live = bool(self.supports_live and not self._use_rich_live)
        self._ansi_lines = 0
        self._ansi_width = 0
        self._rendered_once = False
        self._last_text = ""
        self._last_render_sig = None

    def start(self) -> None:
        if self.supports_live:
            _enable_windows_ansi()
        if self._use_rich_live and self._live is None:
            try:
                console = _get_console(self.stream)
                self._live = Live(
                    Text(""),
                    console=console,
                    auto_refresh=False,
                    transient=False,
                    vertical_overflow="ellipsis",
                )
                self._live.start()
            except Exception:
                self._live = None
                self._use_rich_live = False
                self._use_ansi_live = _enable_windows_ansi() and self.supports_live
                self.supports_live = self._use_ansi_live

    def update(self, dashboard_text: str, force_print_fallback: bool = False, widget_meta=None) -> None:
        w = get_terminal_width()
        text = fit_dashboard_text(str(dashboard_text or ""), width=w)
        meta = widget_meta if isinstance(widget_meta, dict) else None
        render_sig = (
            text,
            meta.get("feed_status") if meta else None,
            meta.get("price_direction") if meta else None,
            meta.get("change_direction") if meta else None,
            bool(meta.get("spread_wide")) if meta else None,
        )
        if self._rendered_once and render_sig == self._last_render_sig and not force_print_fallback:
            return

        if self._use_rich_live and self._live is not None:
            try:
                renderable = build_tradingview_dashboard_renderable(text, widget_meta=meta, stream=self.stream, ansi_mode=False)
                self._live.update(renderable, refresh=True)
                self._rendered_once = True
                self._last_text = text
                self._last_render_sig = render_sig
                return
            except Exception:
                self.stop()
                self._use_rich_live = False
                self.supports_live = False

        if self._use_ansi_live:
            try:
                plain_lines = text.splitlines()
                styled_lines = (
                    build_tradingview_dashboard_renderable(text, widget_meta=meta, stream=self.stream, ansi_mode=True)
                    if meta else plain_lines
                )
                pad_w = min(w, max(self._ansi_width, max((len(ln) for ln in plain_lines), default=0)))
                if self._rendered_once and self._ansi_lines > 0:
                    # Gunakan CUU (\x1b[nA) + CR (\r) dan padding spasi agar kompatibel penuh di terminal Windows
                    self.stream.write(f"\x1b[{self._ansi_lines}A\r")
                total_rows = max(len(plain_lines), self._ansi_lines)
                buf = []
                for idx in range(total_rows):
                    p_ln = plain_lines[idx] if idx < len(plain_lines) else ""
                    s_ln = styled_lines[idx] if idx < len(styled_lines) else ""
                    extra_sp = " " * max(0, pad_w - len(p_ln))
                    buf.append(s_ln + extra_sp)
                self.stream.write("\n".join(buf) + "\n")
                self.stream.flush()
                self._ansi_lines = total_rows
                self._ansi_width = pad_w
                self._rendered_once = True
                self._last_text = text
                self._last_render_sig = render_sig
                return
            except Exception:
                self._use_ansi_live = False
                self.supports_live = False

        # Fallback non-refresh: cetak dashboard ringkas hanya saat pertama kali atau diminta eksplisit
        if not self._rendered_once or force_print_fallback:
            if should_use_color(self.stream) and meta:
                try:
                    renderable = build_tradingview_dashboard_renderable(text, widget_meta=meta, stream=self.stream, ansi_mode=False)
                    if self._rendered_once:
                        self.stream.write("\n")
                    _get_console(self.stream).print(renderable, soft_wrap=True, crop=False)
                    self.stream.flush()
                    self._rendered_once = True
                    self._last_text = text
                    self._last_render_sig = render_sig
                    return
                except Exception:
                    pass
            print_terminal("\n" + text if self._rendered_once else text, file=self.stream, flush=True)
            self._rendered_once = True
            self._last_text = text
            self._last_render_sig = render_sig

    def emit_event(self, event_line: str) -> None:
        raw_ev = " ".join(str(event_line or "").replace("\r", " ").replace("\n", " ").split())
        if not raw_ev:
            return
        w = get_terminal_width()
        wrapped_ev = wrap_dashboard_field("", raw_ev, w, max_lines=None)
        msg = "\n".join(wrapped_ev)
        if self._use_rich_live and self._live is not None:
            try:
                renderable = format_colored_text(msg) if should_use_color(self.stream) else msg
                self._live.console.print(renderable, soft_wrap=True, crop=False)
                return
            except Exception:
                pass
        if self._use_ansi_live and self._rendered_once and self._ansi_lines > 0:
            try:
                w = get_terminal_width()
                self.stream.write(f"\x1b[{self._ansi_lines}A\r")
                self.stream.write(msg.ljust(min(w, max(self._ansi_width, len(msg)))) + "\n")
                lines = self._last_text.splitlines()
                pad_w = min(w, max(self._ansi_width, max((len(ln) for ln in lines), default=0)))
                self.stream.write("\n".join(ln.ljust(pad_w) if len(ln) < pad_w else ln for ln in lines) + "\n")
                self.stream.flush()
                self._ansi_lines = len(lines)
                self._ansi_width = pad_w
                return
            except Exception:
                pass
        print_terminal(msg, file=self.stream, flush=True)

    def stop(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
