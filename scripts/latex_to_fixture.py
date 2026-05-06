"""Extract a clean prose fixture from a LaTeX file.

Strips LaTeX commands while preserving content, so we get
mining-suitable plain text from a paper source. Used to prep a
research-paper fixture for the mining harness.

Defaults: read the mirror_preprint.tex, take abstract + introduction +
first subsection, write to tests/fixtures/mirror_paper_excerpt.txt.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def latex_to_plain(text: str) -> str:
    """Strip LaTeX into clean prose. Best-effort, not pandoc-level."""
    s = text

    # Drop comment-line dividers (% ═══...) and trailing % comments
    s = re.sub(r"^\s*%[^\n]*$", "", s, flags=re.MULTILINE)

    # Drop \label{...}
    s = re.sub(r"\\label\{[^}]*\}", "", s)

    # Section headings → plain heading text on its own line
    s = re.sub(r"\\section\{([^}]+)\}", r"\n\n\1\n", s)
    s = re.sub(r"\\subsection\{([^}]+)\}", r"\n\n\1\n", s)
    s = re.sub(r"\\subsubsection\{([^}]+)\}", r"\n\n\1\n", s)

    # Inline formatting → just keep the content
    for cmd in ("textbf", "textit", "emph", "textsc", "texttt", "underline"):
        s = re.sub(rf"\\{cmd}\{{([^{{}}]*)\}}", r"\1", s)

    # Citations and refs → strip entirely (the prose still reads OK)
    s = re.sub(r"\\citep\{[^}]*\}", "", s)
    s = re.sub(r"\\citet\{[^}]*\}", "", s)
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\ref\{[^}]*\}", "", s)
    s = re.sub(r"\\autoref\{[^}]*\}", "", s)
    s = re.sub(r"~?\(?Section[~ ]\\ref\{[^}]*\}\)?", "", s)

    # Math: keep simple inline content, strip delimiters
    s = re.sub(r"\$([^$]*)\$", r"\1", s)

    # Common environments: keep the content, drop the wrappers
    s = re.sub(r"\\begin\{abstract\}", "", s)
    s = re.sub(r"\\end\{abstract\}", "", s)
    s = re.sub(r"\\begin\{enumerate\}\[[^\]]*\]", "", s)
    s = re.sub(r"\\begin\{enumerate\}", "", s)
    s = re.sub(r"\\end\{enumerate\}", "", s)
    s = re.sub(r"\\begin\{itemize\}\[[^\]]*\]", "", s)
    s = re.sub(r"\\begin\{itemize\}", "", s)
    s = re.sub(r"\\end\{itemize\}", "", s)
    s = re.sub(r"\\begin\{center\}", "", s)
    s = re.sub(r"\\end\{center\}", "", s)

    # \item → bullet for readability
    s = re.sub(r"\\item\s*", "  - ", s)

    # Drop remaining bare commands (kept simple — better to underclean than mangle)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", "", s)

    # LaTeX special chars
    s = s.replace("---", "—")
    s = s.replace("--", "–")
    s = s.replace("``", '"')
    s = s.replace("''", '"')
    s = s.replace("\\&", "&")
    s = s.replace("\\%", "%")
    s = s.replace("\\$", "$")
    s = s.replace("~", " ")  # non-breaking space → regular space
    s = re.sub(r"\\([{}_#&%])", r"\1", s)  # escaped chars

    # Collapse multiple blank lines + trim
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("arxiv_submission/mirror_preprint.tex")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tests/fixtures/mirror_paper_excerpt.txt")
    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        sys.exit(2)

    raw = src.read_text(encoding="utf-8")
    # Lines 51..150 of the preprint = abstract + intro + first background subsection.
    # We do this on the raw text first (line-based), then strip LaTeX.
    lines = raw.splitlines()
    excerpt = "\n".join(lines[50:150])  # 0-based, so [50:150] = lines 51-150

    plain = latex_to_plain(excerpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(plain, encoding="utf-8")

    chars = len(plain)
    paras = plain.count("\n\n") + 1
    print(f"Wrote {out} ({chars} chars, {paras} paragraphs)")


if __name__ == "__main__":
    main()
