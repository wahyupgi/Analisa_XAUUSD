# -*- coding: utf-8 -*-
"""
modules/terminal_ui.py
Formatter warna terminal terpusat berbasis Rich (UI-only, aman untuk TTY interaktif & fallback polos).
"""
import builtins
import os
import re
import sys

try:
    from rich.console import Console
    from rich.text import Text
    _RICH_AVAILABLE = True
except Exception:
    Console = None
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
    (re.compile(r"\b(?:RISK\s+ALERT|CRITICAL\s+WARNING|HIGH\s+RISK|TANPA\s+STOP\s+LOSS|TANPA\s+SL|UNBOUNDED|DATA\s+UNAVAILABLE|TIDAK\s+VALID|BLOCKED\s+BY\s+SNR|SL\s+INVALID|RISIKO\s+TINGGI)\b"), "bold bright_red", 0),
    (re.compile(r"\b(?:INVALIDATED|INVALID|CONFLICT|KONFLIK|DISCONNECTED|UNAVAILABLE|PLACED_MISMATCH|EXPIRED|BROKEN|DITOLAK|GAGAL|ERROR|FAILED|FAIL|JEBOL|BATAL)\b"), "bold bright_red", 0),
    (re.compile(r"(?i:NEWS\s+RISK\s*:\s*|IMPACT\s*:\s*|NEWS\s+)(HIGH)\b"), "bold bright_red", 1),

    # Priority 2: WAIT / APPROACHING (yellow)
    (re.compile(r"\b(?:NOT\s+CONFIRMED|NO\s+ENTRY|NO\s+SETUP|NO\s+ACTIVE\s+(?:SETUP|MONITORING\s+ZONE)|POTENTIAL\s+BREAK(?:OUT|DOWN)|STALE(?:_CACHE|\s+DATA)?|PROVISIONAL(?:\s+ONLY)?|ZONE_TOUCHED|TOUCHED\s+PREVIOUSLY|IN\s+ZONE|PARTIALLY_MITIGATED)\b"), "yellow", 0),
    (re.compile(r"\b(?:APPROACHING|WAITING|WAIT|UNCONFIRMED|UNVERIFIED|RUNNING|PARTIAL|WEAK|COUNTERTREND|TERLEWAT|RECLAIMED|HISTORICAL_TOUCH|HISTORY_UNCERTAIN)\b"), "yellow", 0),
    (re.compile(r"(?i:NEWS\s+RISK\s*:\s*|IMPACT\s*:\s*|NEWS\s+)(MEDIUM|UNKNOWN)\b"), "yellow", 1),

    # Priority 3: BUY / BULLISH & READY / CONFIRMED (bold green)
    (re.compile(r"\b(?:POTENTIAL\s+BULLISH|BUY\s+(?:LIMIT|STOP)|HIT\s+TP[12]?|CONFIRMED\s+ENTRY)\b"), "bold green", 0),
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
    (re.compile(r"\b(?i:Spread)\s*[:=]?\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*pts?)?\b"), "cyan", 0),
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
