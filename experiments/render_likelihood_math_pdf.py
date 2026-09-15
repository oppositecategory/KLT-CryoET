"""Render the likelihood-scoring mathematical note as a paginated PDF."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/statistical_likelihood_scoring.md"
OUTPUT = ROOT / "docs/statistical_likelihood_scoring.pdf"

PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT = 0.09
RIGHT = 0.93
TOP = 0.94
BOTTOM = 0.075


def clean_inline_markdown(text: str) -> str:
    """Remove the small Markdown subset used for inline code and emphasis."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    return text


class Renderer:
    """Simple text-and-math paginator backed by Matplotlib's PDF backend."""

    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.page_number = 0
        self.figure = None
        self.y = TOP
        self.new_page()

    def new_page(self) -> None:
        if self.figure is not None:
            self.finish_page()
        self.page_number += 1
        self.figure = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        self.figure.patch.set_facecolor("white")
        self.y = TOP
        self.figure.text(
            LEFT,
            0.035,
            "3-D KLT Statistical Likelihood Scoring",
            fontsize=7.5,
            color="#555555",
        )
        self.figure.text(
            RIGHT,
            0.035,
            str(self.page_number),
            fontsize=7.5,
            color="#555555",
            ha="right",
        )

    def finish_page(self) -> None:
        self.pdf.savefig(self.figure)
        plt.close(self.figure)
        self.figure = None

    def ensure(self, height: float) -> None:
        if self.y - height < BOTTOM:
            self.new_page()

    def heading(self, text: str, level: int) -> None:
        size = {1: 20, 2: 14, 3: 11}.get(level, 10)
        height = {1: 0.075, 2: 0.052, 3: 0.042}.get(level, 0.04)
        self.ensure(height + 0.02)
        if level > 1:
            self.y -= 0.012
        self.figure.text(
            LEFT,
            self.y,
            clean_inline_markdown(text),
            fontsize=size,
            fontweight="bold",
            color="#18324a",
            va="top",
        )
        self.y -= height

    def paragraph(self, text: str, bullet: bool = False) -> None:
        text = clean_inline_markdown(text)
        width = 91 if bullet else 96
        prefix = "•  " if bullet else ""
        subsequent = "   " if bullet else ""
        lines = textwrap.wrap(
            text,
            width=width,
            initial_indent=prefix,
            subsequent_indent=subsequent,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        line_height = 0.0185
        height = line_height * len(lines) + 0.010
        self.ensure(height)
        self.figure.text(
            LEFT,
            self.y,
            "\n".join(lines),
            fontsize=9.2,
            color="#202020",
            va="top",
            linespacing=1.35,
        )
        self.y -= height

    def equation(self, expression: str) -> None:
        # MathText supports the notation used in the source after these small
        # compatibility substitutions.
        expression = expression.replace(r"\hbox", r"\mathrm")
        expression = expression.replace(r"\operatorname", r"\mathrm")
        expression = expression.replace(r"\lVert", r"\Vert")
        expression = expression.replace(r"\rVert", r"\Vert")
        height = 0.050
        self.ensure(height)
        try:
            self.figure.text(
                0.5,
                self.y,
                f"${expression}$",
                fontsize=11,
                ha="center",
                va="top",
                color="#111111",
            )
        except ValueError:
            self.figure.text(
                LEFT + 0.03,
                self.y,
                expression,
                fontsize=9,
                family="monospace",
                va="top",
            )
        self.y -= height


def render() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    with PdfPages(OUTPUT) as pdf:
        renderer = Renderer(pdf)
        paragraph_parts: list[str] = []
        in_equation = False
        equation_parts: list[str] = []

        def flush_paragraph() -> None:
            if paragraph_parts:
                renderer.paragraph(" ".join(paragraph_parts))
                paragraph_parts.clear()

        for raw_line in lines:
            line = raw_line.strip()
            if in_equation:
                if line.endswith("$$"):
                    equation_parts.append(line[:-2])
                    renderer.equation(" ".join(equation_parts).strip())
                    equation_parts.clear()
                    in_equation = False
                else:
                    equation_parts.append(line)
                continue
            if line.startswith("$$"):
                flush_paragraph()
                remainder = line[2:]
                if remainder.endswith("$$"):
                    renderer.equation(remainder[:-2].strip())
                else:
                    equation_parts.append(remainder)
                    in_equation = True
                continue
            if not line:
                flush_paragraph()
                continue
            heading = re.match(r"^(#{1,3})\s+(.*)$", line)
            if heading:
                flush_paragraph()
                renderer.heading(heading.group(2), len(heading.group(1)))
                continue
            if line.startswith("- "):
                flush_paragraph()
                renderer.paragraph(line[2:], bullet=True)
                continue
            paragraph_parts.append(line)

        flush_paragraph()
        renderer.finish_page()


if __name__ == "__main__":
    render()
    print(OUTPUT)
