"""Strip YAML frontmatter and normalize Obsidian wikilinks for Notion ingestion."""
import re
import sys
from pathlib import Path

SRC = Path("capitulos")
DST = Path("output/notion")
DST.mkdir(parents=True, exist_ok=True)

FRONTMATTER = re.compile(r"^---\n.*?\n---\n\n?", re.DOTALL)
WIKILINK_ALIAS = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
WIKILINK_PLAIN = re.compile(r"\[\[([^\]]+)\]\]")
HASHTAG = re.compile(r"(?m)^#(\w[\w/\-áéíóúñÁÉÍÓÚÑ]*(?:\s+#\w[\w/\-áéíóúñÁÉÍÓÚÑ]*)*)\s*$")

for md in sorted(SRC.glob("*.md")):
    text = md.read_text(encoding="utf-8")
    text = FRONTMATTER.sub("", text, count=1)
    text = WIKILINK_ALIAS.sub(r"**\2**", text)
    text = WIKILINK_PLAIN.sub(r"**\1**", text)
    out = DST / md.name
    out.write_text(text, encoding="utf-8")
    print(f"{md.name}: {len(text)} bytes")
