from __future__ import annotations

from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "test-assets" / "road_svgs"
WIDTH = 1280
HEIGHT = 720


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lane_points(
    bottom_x: float,
    top_x: float,
    top_y: float,
    curve: float = 0.0,
) -> list[tuple[float, float]]:
    bottom_y = HEIGHT - 20
    mid_y = (bottom_y + top_y) / 2.0
    mid_x = (bottom_x + top_x) / 2.0 + curve
    return [(bottom_x, bottom_y), (mid_x, mid_y), (top_x, top_y)]


def svg_polyline(points: Iterable[tuple[float, float]], color: str, width: int, dash: str = "") -> str:
    point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{point_str}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash_attr} />'
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str, width: int, opacity: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" opacity="{opacity:.2f}" />'
    )


def background() -> str:
    return f"""
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#8ec5ff" />
        <stop offset="55%" stop-color="#d8efff" />
        <stop offset="56%" stop-color="#597551" />
        <stop offset="100%" stop-color="#41573c" />
      </linearGradient>
      <linearGradient id="road" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#505050" />
        <stop offset="100%" stop-color="#222222" />
      </linearGradient>
    </defs>
    <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#sky)" />
    <polygon points="280,720 520,280 760,280 1000,720" fill="url(#road)" />
    """


def render_case(
    name: str,
    title: str,
    left_bottom: float,
    left_top: float,
    right_bottom: float,
    right_top: float,
    *,
    top_y: float = 290.0,
    left_curve: float = 0.0,
    right_curve: float = 0.0,
    left_color: str = "#f8f8f0",
    right_color: str = "#f8f8f0",
    left_width: int = 14,
    right_width: int = 14,
    left_dash: str = "",
    right_dash: str = "",
    extra: str = "",
) -> tuple[str, str]:
    left = svg_polyline(lane_points(left_bottom, left_top, top_y, left_curve), left_color, left_width, left_dash)
    right = svg_polyline(lane_points(right_bottom, right_top, top_y, right_curve), right_color, right_width, right_dash)
    label = (
        f'<rect x="20" y="20" width="380" height="60" rx="12" fill="#000000" opacity="0.55" />'
        f'<text x="40" y="58" font-size="28" fill="#ffffff" font-family="monospace">{title}</text>'
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
{background()}
{left}
{right}
{extra}
{label}
</svg>
"""
    return name, svg


def noisy_marks() -> str:
    return "\n".join(
        [
            svg_line(140, 620, 280, 500, "#d0d0d0", 4, 0.35),
            svg_line(1060, 610, 1160, 520, "#d0d0d0", 4, 0.35),
            svg_line(350, 690, 350, 620, "#707070", 3, 0.5),
            svg_line(930, 690, 930, 620, "#707070", 3, 0.5),
        ]
    )


def shadow_band() -> str:
    return '<rect x="250" y="430" width="780" height="90" fill="#000000" opacity="0.18" />'


def false_left() -> str:
    return svg_line(420, 700, 560, 320, "#f4d35e", 8, 0.75)


def false_right() -> str:
    return svg_line(870, 700, 735, 320, "#f4d35e", 8, 0.75)


def broken_left_segments() -> str:
    return "\n".join(
        [
            svg_line(430, 700, 485, 560, "#f8f8f0", 14),
            svg_line(505, 500, 555, 360, "#f8f8f0", 14),
        ]
    )


def broken_right_segments() -> str:
    return "\n".join(
        [
            svg_line(850, 700, 800, 550, "#f8f8f0", 14),
            svg_line(780, 500, 730, 360, "#f8f8f0", 14),
        ]
    )


CASES = [
    render_case("01_center_straight", "Centre droite", 420, 555, 860, 725),
    render_case("02_center_wide", "Centre large", 360, 520, 920, 760),
    render_case("03_center_narrow", "Centre etroite", 470, 575, 810, 705),
    render_case("04_shift_left", "Route decalee a gauche", 350, 510, 790, 675),
    render_case("05_shift_right", "Route decalee a droite", 490, 600, 930, 790),
    render_case("06_perspective_steep", "Perspective forte", 410, 595, 870, 690, top_y=250),
    render_case("07_perspective_shallow", "Perspective faible", 420, 510, 860, 770, top_y=360),
    render_case("08_curve_left", "Courbe gauche legere", 430, 560, 860, 710, left_curve=-45, right_curve=-55),
    render_case("09_curve_right", "Courbe droite legere", 420, 550, 850, 700, left_curve=45, right_curve=55),
    render_case("10_lane_opens_left", "Ouverture vers gauche", 440, 520, 840, 750, left_curve=-30, right_curve=20),
    render_case("11_lane_opens_right", "Ouverture vers droite", 430, 590, 850, 690, left_curve=30, right_curve=-20),
    render_case("12_dashed_both", "Lignes pointillees", 420, 555, 860, 725, left_dash="24 18", right_dash="24 18"),
    render_case("13_broken_left", "Ligne gauche coupee", 420, 555, 860, 725, left_color="transparent", extra=broken_left_segments()),
    render_case("14_broken_right", "Ligne droite coupee", 420, 555, 860, 725, right_color="transparent", extra=broken_right_segments()),
    render_case("15_faded_left", "Ligne gauche fade", 420, 555, 860, 725, left_color="#a7a7a0"),
    render_case("16_faded_right", "Ligne droite fade", 420, 555, 860, 725, right_color="#a7a7a0"),
    render_case("17_shadow_band", "Ombre sur route", 420, 555, 860, 725, extra=shadow_band()),
    render_case("18_roadside_noise", "Bruit lateral", 420, 555, 860, 725, extra=noisy_marks()),
    render_case("19_false_line_left", "Fausse ligne gauche", 420, 555, 860, 725, extra=false_left()),
    render_case("20_false_line_right", "Fausse ligne droite", 420, 555, 860, 725, extra=false_right()),
    render_case("21_offset_and_noise", "Decalee et bruit", 360, 520, 800, 690, extra=noisy_marks() + "\n" + shadow_band()),
    render_case("22_narrow_curve_left", "Etroite courbe gauche", 470, 590, 810, 710, left_curve=-35, right_curve=-45),
    render_case("23_wide_curve_right", "Large courbe droite", 340, 520, 960, 770, left_curve=40, right_curve=70),
    render_case("24_high_contrast", "Contraste fort", 420, 555, 860, 725, left_color="#ffffff", right_color="#ffffcc", left_width=18, right_width=18),
]


def write_readme(case_names: list[str]) -> None:
    readme = OUT_DIR / "README.md"
    lines = [
        "# SVG de test pour la detection de route",
        "",
        "Ces fichiers simulent une route vue de face avec deux lignes detectables par l'application OpenCV.",
        "",
        "## Contenu",
        "",
    ]
    lines.extend(f"- `{name}.svg`" for name in case_names)
    lines.extend(
        [
            "",
            "## Astuce",
            "",
            "Pour tester rapidement, ouvre `index.html` dans un navigateur ou affiche les SVG sur un autre ecran face a la camera du telephone.",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")


def write_index(case_names: list[str]) -> None:
    cards = "\n".join(
        f"""
        <div class="card">
          <h2>{name}</h2>
          <img src="{name}.svg" alt="{name}" />
        </div>
        """
        for name in case_names
    )
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Routes SVG de test</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #101820;
      color: #f6f7f8;
    }}
    header {{
      padding: 24px 28px 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      padding: 20px 28px 32px;
    }}
    .card {{
      background: #1c2730;
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.24);
    }}
    h1, h2 {{
      margin: 0 0 10px;
    }}
    img {{
      width: 100%;
      height: auto;
      border-radius: 10px;
      background: #0b1014;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Jeu de test SVG pour l'app OpenCV</h1>
    <p>{len(case_names)} scenes generees automatiquement.</p>
  </header>
  <section class="grid">
    {cards}
  </section>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case_names: list[str] = []

    for name, content in CASES:
        (OUT_DIR / f"{name}.svg").write_text(content, encoding="utf-8")
        case_names.append(name)

    write_readme(case_names)
    write_index(case_names)

    print(f"{len(case_names)} SVG generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
