"""Remove personal information from a Discord message export before training/publishing.

    python scripts/scrub_data.py raw_messages.txt data/messages.txt \
        --redact-file data/redact.local.txt --report

What gets replaced (placeholders are plain ASCII so they don't grow the vocabulary):

    email addresses               -> [email]
    UK / international phone nos  -> [phone]
    UK postcodes                  -> [postcode]
    "12 Something Way/Road/..."   -> [address]
    Discord user/role/channel IDs -> @user / @role / #channel
    custom emoji <:name:1234>     -> :name:
    any line or substring listed in --redact-file -> [redacted]

--redact-file is a private, git-ignored list of exact strings (one per line,
case-insensitive) for things regexes can't catch: passwords, real names, street
names, etc. A line that is *exactly* a listed string is dropped entirely.

Regexes are a safety net, not a guarantee: always skim the --report output.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    ("phone", re.compile(r"(?<!\d)(?:\+44\s?7\d{3}|07\d{3})\s?\d{3}\s?\d{3}(?!\d)"), "[phone]"),
    ("phone", re.compile(r"(?<!\d)\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"), "[phone]"),
    ("postcode", re.compile(r"\b[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}\b"), "[postcode]"),
    ("address", re.compile(
        r"\b\d{1,4}[a-zA-Z]?\s+(?:[A-Za-z]+\s){1,3}"
        r"(?:way|road|rd|street|st|lane|ln|avenue|ave|close|drive|dr|gardens|crescent|place|court|grove|terrace|hill)\b",
        re.IGNORECASE), "[address]"),
    ("discord_user", re.compile(r"<@!?\d{15,22}>"), "@user"),
    ("discord_role", re.compile(r"<@&\d{15,22}>"), "@role"),
    ("discord_channel", re.compile(r"<#\d{15,22}>"), "#channel"),
    ("custom_emoji", re.compile(r"<a?(:\w+:)\d{15,22}>"), r"\1"),
]

# Things that look like postcodes but aren't (model names etc.). Extend as needed.
POSTCODE_FALSE_POSITIVES = re.compile(
    r"^(?:sd\d|rtx|gtx|gt\d|ff[0-9a-f]|[ir]\d\s?\d(?:st|nd|rd|th)|s\d{3}hd|t00bs|p45bs)",
    re.IGNORECASE,
)


def load_redactions(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def scrub(text: str, redactions: list[str]) -> tuple[str, Counter, list[str]]:
    counts: Counter = Counter()
    examples: list[str] = []
    redact_lower = {r.lower() for r in redactions}
    redact_res = [re.compile(re.escape(r), re.IGNORECASE) for r in sorted(redactions, key=len, reverse=True)]

    out_lines = []
    for line in text.split("\n"):
        if line.strip().lower() in redact_lower:
            counts["dropped_line"] += 1
            continue
        original = line
        for rx in redact_res:
            line, n = rx.subn("[redacted]", line)
            counts["redact_file"] += n
        for name, rx, repl in PATTERNS:
            if name == "postcode":
                def _pc(m):
                    if POSTCODE_FALSE_POSITIVES.match(m.group(0)):
                        return m.group(0)
                    counts["postcode"] += 1
                    return repl
                line = rx.sub(_pc, line)
            else:
                line, n = rx.subn(repl, line)
                counts[name] += n
        if line != original and len(examples) < 40:
            examples.append(line)
        out_lines.append(line)
    return "\n".join(out_lines), counts, examples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--redact-file")
    p.add_argument("--report", action="store_true", help="print counts and example scrubbed lines")
    args = p.parse_args()

    with open(args.src, encoding="utf-8") as f:
        text = f.read()
    clean, counts, examples = scrub(text, load_redactions(args.redact_file))
    with open(args.dst, "w", encoding="utf-8") as f:
        f.write(clean)

    print(f"wrote {args.dst}: {len(text):,} -> {len(clean):,} chars")
    for k, v in sorted(counts.items()):
        if v:
            print(f"  {k:16s} {v}")
    if args.report:
        print("\nexample scrubbed lines:")
        for e in examples:
            print("  " + e[:160])


if __name__ == "__main__":
    main()
