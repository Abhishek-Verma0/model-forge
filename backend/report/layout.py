"""A small page layout on top of matplotlib's PDF backend (already installed):
A4 pages, headings, wrapped paragraphs, tables that continue across pages,
images and charts. Uses matplotlib.figure.Figure, never pyplot -- pyplot's global
state is not thread-safe and reports are built inside web requests."""

import io
import textwrap

import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

PAGE_W, PAGE_H = 8.27, 11.69   # A4 in inches (ISO 216)
MARGIN = 0.7
CHAR_W = 0.52                  # average glyph width of DejaVu Sans as a share of the font size


class Document:
    def __init__(self, title, subtitle=""):
        self.buf = io.BytesIO()
        self.pdf = PdfPages(self.buf)
        self.title, self.subtitle = title, subtitle
        self.fig, self.y, self.page = None, 0.0, 0
        md = self.pdf.infodict()
        md["Title"], md["Creator"] = title, "Model Forge (ResearchAI Studio)"

    # -- page mechanics -------------------------------------------------------
    def _finish_page(self):
        if self.fig is not None:
            self.fig.text(0.5, 0.35 / PAGE_H, f"{self.title}  ·  page {self.page}", ha="center", fontsize=7,
                          color="#666666")
            self.pdf.savefig(self.fig)
        self.fig = None

    def _new_page(self):
        self._finish_page()
        self.fig = Figure(figsize=(PAGE_W, PAGE_H))
        self.page += 1
        self.y = PAGE_H - MARGIN

    def _need(self, height):
        if self.fig is None or self.y - height < MARGIN:
            self._new_page()

    def _text(self, x, text, size, **kw):
        self.fig.text(x / PAGE_W, self.y / PAGE_H, text, fontsize=size, va="top", **kw)

    # -- content --------------------------------------------------------------
    def cover(self, lines):
        self._new_page()
        self.y = PAGE_H - 2.2
        self._text(MARGIN, self.title, 22, weight="bold", color="#1f3a5f")
        self.y -= 0.5
        if self.subtitle:
            self._text(MARGIN, self.subtitle, 12, color="#2f6eb5")
            self.y -= 0.5
        for line in lines:
            self.para(line, 10)

    def heading(self, text, level=1):
        size = {1: 15, 2: 12, 3: 10.5}[level]
        self._need(0.9 if level == 1 else 0.6)
        self.y -= 0.12 if level == 1 else 0.05
        self._text(MARGIN, text, size, weight="bold", color="#1f3a5f" if level == 1 else "#222222")
        self.y -= size / 72 * 1.7

    def para(self, text, size=9, color="#222222", indent=0.0):
        width = int((PAGE_W - 2 * MARGIN - indent) * 72 / (size * CHAR_W))
        line_h = size / 72 * 1.45
        for raw in str(text).split("\n"):
            for line in textwrap.wrap(raw, width) or [""]:
                self._need(line_h)
                self._text(MARGIN + indent, line, size, color=color)
                self.y -= line_h
        self.y -= line_h * 0.3

    def bullets(self, items, size=9):
        for it in items:
            self.para(f"•  {it}", size, indent=0.15)

    def table(self, header, rows, widths=None, size=7.5, max_rows=None):
        """Rows of strings; columns sized by `widths` (shares of the text width).
        Cells are cut to fit with an ellipsis; the header repeats on new pages."""
        rows = rows[:max_rows] if max_rows else rows
        total = PAGE_W - 2 * MARGIN
        widths = widths or [1 / len(header)] * len(header)
        xs = [MARGIN + total * sum(widths[:i]) for i in range(len(header))]
        caps = [max(3, int(total * w * 72 / (size * CHAR_W)) - 1) for w in widths]
        line_h = size / 72 * 1.55

        def cell(v, cap):
            s = "" if v is None else str(v)
            return s if len(s) <= cap else s[:cap - 1] + "…"

        def head():
            self._need(line_h * 2)
            for x, h, cap in zip(xs, header, caps):
                self._text(x, cell(h, cap), size, weight="bold")
            self.y -= line_h
            self.fig.add_artist(_rule(self.y + line_h * 0.25))

        head()
        for r in rows:
            if self.fig is None or self.y - line_h < MARGIN:
                self._new_page()
                head()
            for x, v, cap in zip(xs, r, caps):
                self._text(x, cell(v, cap), size)
            self.y -= line_h
        self.y -= line_h * 0.6

    def image(self, data, height=3.2, caption=None):
        """A PNG (bytes or a data: URI) scaled to `height` inches, keeping its aspect."""
        if isinstance(data, str) and data.startswith("data:"):
            import base64
            data = base64.b64decode(data.split(",", 1)[1])
        img = mpimg.imread(io.BytesIO(data), format="png")
        h_px, w_px = img.shape[:2]
        width = min(PAGE_W - 2 * MARGIN, height * w_px / h_px)
        height = width * h_px / w_px
        self._need(height + 0.3)
        ax = self.fig.add_axes([MARGIN / PAGE_W, (self.y - height) / PAGE_H, width / PAGE_W, height / PAGE_H])
        ax.imshow(img)
        ax.axis("off")
        self.y -= height + 0.1
        if caption:
            self.para(caption, 7.5, color="#555555")

    def chart(self, draw, height=3.0, width=None, left=0.45):
        """draw(ax) on a fresh axes `height` inches tall (full text width unless given);
        `left` = room for the y labels (long category names need ~1.7 in)."""
        width = width or PAGE_W - 2 * MARGIN
        self._need(height + 0.45)
        ax = self.fig.add_axes([(MARGIN + left) / PAGE_W, (self.y - height) / PAGE_H,
                                (width - left - 0.15) / PAGE_W, (height - 0.25) / PAGE_H])
        draw(ax)
        self.y -= height + 0.35

    def charts(self, draws, height=2.6, left=0.45):
        """Two charts side by side."""
        self._need(height + 0.45)
        half = (PAGE_W - 2 * MARGIN) / 2
        for i, draw in enumerate(draws[:2]):
            ax = self.fig.add_axes([(MARGIN + i * half + left) / PAGE_W, (self.y - height) / PAGE_H,
                                    (half - left - 0.25) / PAGE_W, (height - 0.3) / PAGE_H])
            draw(ax)
        self.y -= height + 0.35

    def bytes(self):
        self._finish_page()
        self.pdf.close()
        return self.buf.getvalue()


def _rule(y):
    from matplotlib.lines import Line2D
    return Line2D([MARGIN / PAGE_W, 1 - MARGIN / PAGE_W], [y / PAGE_H] * 2, lw=0.5, color="#999999")


if __name__ == "__main__":
    d = Document("Self-check report", "layout")
    d.cover(["A cover line."])
    d.heading("Section")
    d.para("word " * 400)
    d.table(["a", "b"], [[i, "x" * 200] for i in range(120)], [0.2, 0.8])
    d.chart(lambda ax: ax.plot([1, 2, 3]))
    d.charts([lambda ax: ax.bar([1, 2], [3, 4]), lambda ax: ax.scatter([1], [1])])
    out = d.bytes()
    assert out.startswith(b"%PDF") and d.page >= 4, d.page
    print(f"layout self-check passed ({d.page} pages, {len(out) // 1024} KB)")
