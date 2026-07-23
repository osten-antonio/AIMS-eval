import json
import os
import subprocess
import random
import unicodedata
from pathlib import Path

# pip install fonttools
from fontTools.ttLib import TTFont

random.seed(42)

HANDWRITING_FONTS = [
    "Caveat-Medium.ttf",
    "Caveat-VariableFont_wght.ttf",
    "Excalifont-Regular.ttf",
    "GochiHand-Regular.ttf",
]

FONT_DIR = Path("/home/osten/Projects/WADS-nougat-evaluation/Fonts/")

class FontCoverage:
    """Reads the cmap table of a TTF/OTF to discover exactly which
    Unicode codepoints the font contains.  Anything absent is wrapped
    in a XeLaTeX fallback font family instead of being left to turn
    into a box.
    """
    def __init__(self, font_dir: Path):
        self.font_dir = font_dir
        self._cache: dict[str, set[int]] = {}

    def _load(self, font_name: str) -> set[int]:
        path = self.font_dir / font_name
        if not path.exists():
            return set()
        font = TTFont(str(path))
        chars = set()
        for table in font["cmap"].tables:
            if table.isUnicode():
                chars.update(table.cmap.keys())
        return chars

    def supports(self, font_name: str, char: str) -> bool:
        if font_name not in self._cache:
            self._cache[font_name] = self._load(font_name)
        return ord(char) in self._cache[font_name]

    def filter_text(self, text: str, font_name: str,
                    fallback_cmd: str = r"\fallbackfont") -> str:
        """Return LaTeX text where unsupported chars are wrapped in
        *fallback_cmd*.  ASCII printable chars and LaTeX specials are
        handled without consulting the font.
        """
        # Normalize so combining marks become precomposed chars when possible
        text = unicodedata.normalize("NFC", text)

        latex_specials = {
            '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
            '\n': r'\\ ', '\r': '',
        }

        out = []
        for ch in text:
            if ch in latex_specials:
                out.append(latex_specials[ch])
                continue

            code = ord(ch)
            # ASCII printable is safe for every font we use
            if 32 <= code <= 126:
                out.append(ch)
                continue

            # Ask the real font: do you have this glyph?
            if self.supports(font_name, ch):
                out.append(ch)
            else:
                out.append(f'{{{fallback_cmd} {ch}}}')
        return ''.join(out)


# Global instance, parses each font once and caches the result
_coverage = FontCoverage(FONT_DIR)

_LATEX_SPECIALS = {
    '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
    '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
    '\n': r'\\ ', '\r': '',
}

def escape_latex_text(text):
    """Escape LaTeX specials in a single pass over the input.

    A sequential chain of str.replace() calls corrupts itself here: replacing
    '$' -> '\\$' first, then later replacing '\\' -> '\\textbackslash{}',
    re-matches the backslash it JUST inserted and mangles '\\$' into
    '\\textbackslash{}$' - an unescaped '$' that reopens math mode and
    crashes pdflatex. Walking the original string character-by-character
    (as FontCoverage.filter_text already does for the handwritten path)
    avoids ever re-scanning our own output.
    """
    return ''.join(_LATEX_SPECIALS.get(ch, ch) for ch in text)


def gen_printed_latex(latex_str):
    return (
        r"\documentclass[12pt]{article}"
        r"\usepackage[margin=2in]{geometry}"
        r"\usepackage[utf8]{inputenc}"
        r"\usepackage{amsmath, amssymb}"
        r"\begin{document}"
        r"\begin{center}"
        r"\Huge \[__LATEX__\]"
        r"\end{center}"
        r"\end{document}"
    ).replace("__LATEX__", latex_str)

def gen_handwriting_latex(latex_str, font_name):
    # \fallbackfont is defined here; DejaVu Sans has massive Unicode coverage
    return (
        r"\documentclass[12pt]{article}"
        r"\usepackage[margin=2in]{geometry}"
        r"\usepackage{fontspec}"
        r"\setmainfont{__FONT__}[Path=__FONTDIR__]"
        r"\newfontfamily\fallbackfont{DejaVu Sans}[Scale=MatchLowercase]"
        r"\usepackage{amsmath, amssymb}"
        r"\begin{document}"
        r"\begin{center}"
        r"\huge\[__LATEX__\]"
        r"\end{center}"
        r"\end{document}"
    ).replace("__FONT__", font_name)\
     .replace("__FONTDIR__", str(FONT_DIR) + "/")\
     .replace("__LATEX__", latex_str)

def gen_word_problem_latex(latex_str, text, font_name=None):
    if font_name:
        tex = (
            r"\documentclass[12pt]{article}"
            r"\usepackage[margin=2in, top=3in, bottom=3in, left=2in, right=2in]{geometry}"
            r"\usepackage{fontspec}"
            r"\setmainfont{__FONT__}[Path=__FONTDIR__]"
            r"\newfontfamily\fallbackfont{DejaVu Sans}[Scale=MatchLowercase]"
            r"\usepackage{amsmath, amssymb}"
            r"\begin{document}"
            r"\noindent\large "
            r"\parbox{\textwidth}{__TEXT__}"
            r"\vspace{1cm}"
            r"\begin{center}"
            r"\huge\[__LATEX__\]"
            r"\end{center}"
            r"\end{document}"
        ).replace("__FONT__", font_name).replace("__FONTDIR__", str(FONT_DIR) + "/")
    else:
        tex = (
            r"\documentclass[12pt]{article}"
            r"\usepackage[margin=2in, top=3in, bottom=3in, left=2in, right=2in]{geometry}"
            r"\usepackage[utf8]{inputenc}"
            r"\usepackage{amsmath, amssymb}"
            r"\begin{document}"
            r"\noindent\large "
            r"\parbox{\textwidth}{__TEXT__}"
            r"\vspace{1cm}"
            r"\begin{center}"
            r"\huge\[__LATEX__\]"
            r"\end{center}"
            r"\end{document}"
        )
    return tex.replace("__TEXT__", text).replace("__LATEX__", latex_str)

def render_to_png(tex_path, output_png, is_handwriting=False):
    tex_dir = os.path.dirname(tex_path)
    tex_filename = os.path.basename(tex_path)
    abs_tex_dir = os.path.abspath(tex_dir)

    if is_handwriting:
        cmd = ["xelatex", "-interaction=nonstopmode", "-output-directory", abs_tex_dir, tex_filename]
    else:
        cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", abs_tex_dir, tex_filename]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=abs_tex_dir, errors='ignore')
    if result.returncode != 0:
        print(f"  LaTeX fail for {tex_filename}: rc={result.returncode}")

    pdf_path = os.path.join(abs_tex_dir, os.path.basename(tex_path).replace(".tex", ".pdf"))
    if os.path.exists(pdf_path):
        png_path = output_png.replace(".png", "")
        cmd = ["pdftoppm", "-r", "300", "-png", pdf_path, png_path]
        subprocess.run(cmd, capture_output=True, text=True, errors='ignore')
        actual_png = png_path + "-1.png"
        if os.path.exists(actual_png):
            return actual_png
        elif os.path.exists(output_png):
            return output_png
    return None


if __name__ == "__main__":
    with open("data/raw_items.json") as f:
        equations = json.load(f)
    with open("data/raw_word_problems.json") as f:
        word_problems = json.load(f)

    all_items = equations + word_problems
    print(f"Total items to render: {len(all_items)}")

    os.makedirs("data/images", exist_ok=True)
    manifest = []
    skipped = []

    batch_size = 10
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i:i + batch_size]

        for item in batch:
          try:
            idx = len(manifest)
            if "latex" not in item:
                raise ValueError("item has no 'latex' field")
            latex_str = item["latex"]
            # Infer category instead of crashing if it's missing: word
            # problems carry a 'text' field, plain equations don't.
            category = item.get("category") or ("word_problem" if item.get("text") else "equation")

            tex_dir = f"data/tex/item_{idx}"
            os.makedirs(tex_dir, exist_ok=True)

            if category == "equation":
                tex_content = gen_printed_latex(latex_str)
            else:
                # Minimal escaping for pdflatex safety (single-pass, see
                # escape_latex_text - a sequential-replace chain here used to
                # double-escape its own output whenever the text had a '$' or
                # '%', e.g. any word problem mentioning a dollar amount)
                t = escape_latex_text(item.get("text", ""))
                tex_content = gen_word_problem_latex(latex_str=latex_str, text=t)

            tex_path = os.path.join(tex_dir, "render.tex")
            with open(tex_path, "w") as f:
                f.write(tex_content)

            png_path = f"data/images/item_{idx}_printed.png"
            result = render_to_png(tex_path, png_path, is_handwriting=False)

            if result:
                entry = {
                    "id": idx, "image_path": result, "latex": latex_str,
                    "category": category, "render_style": "printed", "font": "none"
                }
                if category == "word_problem":
                    entry["text"] = item.get("text", "")
                manifest.append(entry)

            font = HANDWRITING_FONTS[idx % len(HANDWRITING_FONTS)]

            if category == "equation":
                tex_content = gen_handwriting_latex(latex_str, font)
            else:
                safe_text = _coverage.filter_text(item.get("text", ""), font)
                tex_content = gen_word_problem_latex(latex_str=latex_str,
                                                     text=safe_text,
                                                     font_name=font)

            tex_path2 = os.path.join(tex_dir, "render_hw.tex")
            with open(tex_path2, "w") as f:
                f.write(tex_content)

            png_path2 = f"data/images/item_{idx}_handwritten.png"
            result2 = render_to_png(tex_path2, png_path2, is_handwriting=True)

            if result2:
                entry2 = {
                    "id": idx, "image_path": result2, "latex": latex_str,
                    "category": category, "render_style": "handwritten", "font": font
                }
                if category == "word_problem":
                    entry2["text"] = item.get("text", "")
                manifest.append(entry2)
          except Exception as e:
            print(f"  Skipping item {i}: {e!r} -- {item!r}")
            skipped.append({"item": item, "error": repr(e)})

        print(f"Batch {i // batch_size + 1}: Rendered {min(batch_size, len(all_items) - i)} items")
        # Save after every batch so a crash later in the run never loses
        # everything already rendered - previously manifest.json was only
        # written once at the very end, so any single bad item meant NO
        # manifest at all, even though the images themselves were on disk.
        with open("data/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    if skipped:
        with open("data/skipped_items.json", "w") as f:
            json.dump(skipped, f, indent=2)
        print(f"\n{len(skipped)} item(s) skipped - see data/skipped_items.json")

    print(f"\nTotal rendered: {len(manifest)} images")
    print(f"Manifest saved to data/manifest.json")