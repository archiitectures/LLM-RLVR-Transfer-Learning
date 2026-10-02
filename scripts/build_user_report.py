"""Render the tracked user-guide source into a readable PDF and preview pages.

Usage: uv run --group docs python scripts/build_user_report.py --evidence PATH ...
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def inline(text):
    value = html.escape(text)
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<link href="\2" color="#176B75">\1</link>', value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    return value


def render(source: str, output: Path):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "GuideBody",
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            spaceAfter=7,
            textColor=colors.HexColor("#243343"),
        )
    )
    styles.add(
        ParagraphStyle(
            "GuideTitle",
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=29,
            textColor=colors.HexColor("#102B46"),
            spaceAfter=13,
        )
    )
    styles.add(
        ParagraphStyle(
            "GuideSub",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#176B75"),
            spaceBefore=10,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            "GuideTable", parent=styles["GuideBody"], fontSize=8.3, leading=11, spaceAfter=0
        )
    )
    styles.add(
        ParagraphStyle(
            "GuideCode",
            fontName="Courier",
            fontSize=8.1,
            leading=11,
            textColor=colors.HexColor("#163B46"),
            leftIndent=8,
            rightIndent=8,
            borderPadding=8,
            backColor=colors.HexColor("#F0F5F7"),
            spaceAfter=10,
            alignment=TA_LEFT,
        )
    )
    story = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- page -->":
            story.append(PageBreak())
        elif line.startswith("# "):
            story.extend(
                [
                    Paragraph(inline(line[2:]), styles["GuideTitle"]),
                    HRFlowable(width="100%", thickness=1, color=colors.HexColor("#176B75")),
                    Spacer(1, 9),
                ]
            )
        elif line.startswith(("## ", "### ")):
            story.append(Paragraph(inline(line.lstrip("# ")), styles["GuideSub"]))
        elif line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(code), styles["GuideCode"]))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+", c) for c in cells):
                    rows.append([Paragraph(inline(c), styles["GuideTable"]) for c in cells])
                i += 1
            i -= 1
            width = A4[0] - 36 * mm
            cols = len(rows[0])
            widths = (
                [width * 0.25, width * 0.25, width * 0.5] if cols == 3 else [width / cols] * cols
            )
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEBF0")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#F6F8FA")],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#A2BBC5")),
                    ]
                )
            )
            story.extend([table, Spacer(1, 10)])
        else:
            para = [line]
            while (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].startswith(("#", "|", "```", "<!--"))
            ):
                i += 1
                para.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(para)), styles["GuideBody"]))
        i += 1

    def furniture(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#647687"))
        canvas.drawString(18 * mm, 12 * mm, "TRANSFERLAB  /  CAPABILITIES & FIRST EXPERIMENTS")
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, str(document.page))
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=22 * mm,
        title="TransferLab: capabilities and first-experiment guide",
        author="TransferLab project",
        pageCompression=1,
    )
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("docs/USER_GUIDE.md"))
    parser.add_argument("--output", type=Path, default=Path("docs/TransferLab_User_Guide.pdf"))
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--preview", type=Path, default=Path("reports/guide-preview"))
    args = parser.parse_args()
    evidence = {r["stage"]: r for p in args.evidence for r in [json.loads(p.read_text())]}
    table = ["| Stage | Recorded status | Evidence and meaning |", "|---|---|---|"]
    meanings = {
        "local": "Regression and fixture workflow checks",
        "cpu": "Real tiny-model execution and resume",
        "sandbox": "Docker isolation and official EvalPlus checks",
        "gpu": "Intended-model, five-arm CUDA pilot",
    }
    for stage, meaning in meanings.items():
        record = evidence.get(stage)
        status = record["status"].upper() if record else "NOT EXECUTED"
        detail = meaning
        if record:
            detail += "; " + (
                f"{record['seconds']:.1f} s elapsed"
                if record["status"] == "passed"
                else record.get("error", "")
            )
            if "test_counts" in record:
                detail += f"; {record['test_counts']['tests']} tests, {record['test_counts']['skipped']} skipped"
        table.append(f"| {stage} | {status} | {detail} |")
    revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    from transferlab.io import source_fingerprint

    metadata = (
        f"PDF built {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        f"Repository base commit: {revision}. Source fingerprint: {source_fingerprint(Path.cwd())[:16]}. "
        "Qualification evidence is saved in the corresponding runs/checks output folders."
    )
    source = (
        args.source.read_text()
        .replace("{{readiness_table}}", "\n".join(table))
        .replace("{{build_metadata}}", metadata)
    )
    render(source, args.output)
    import pymupdf

    pdf = pymupdf.open(args.output)
    args.preview.mkdir(parents=True, exist_ok=True)
    pages = []
    for index, page in enumerate(pdf):
        image_path = args.preview / f"page-{index + 1:02}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(image_path)
        pages.append(
            {"page": index + 1, "characters": len(page.get_text()), "preview": str(image_path)}
        )
    (args.preview / "qa.json").write_text(
        json.dumps({"pdf": str(args.output), "pages": pages}, indent=2)
    )
    print(
        json.dumps(
            {"pdf": str(args.output), "pages": len(pdf), "preview": str(args.preview)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
