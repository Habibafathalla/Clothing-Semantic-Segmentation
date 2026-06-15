import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CLASS_NAMES, PALETTE


def _rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def legend_html(detected_classes: list[int]) -> str:
    if not detected_classes:
        return "<p>No classes detected.</p>"

    cards = ""
    for cid in detected_classes:
        name      = CLASS_NAMES[cid]
        rgb       = PALETTE[cid]
        rgb_tuple = tuple(rgb.tolist())
        hex_color = _rgb_to_hex(rgb)

        cards += f"""
        <div style="display:flex;align-items:center;gap:8px;padding:6px 8px;
                    background:white;border-radius:7px;border:1px solid #e0e0e0;font-size:13px;">
            <div style="width:22px;height:22px;border-radius:5px;flex-shrink:0;
                        background:rgb{rgb_tuple};border:1px solid #333;"></div>
            <div>
                <strong>{name}</strong><br>
                <span style="font-size:12px;">RGB: {rgb_tuple} | {hex_color}</span>
            </div>
        </div>"""

    return f"""
    <div style="margin-top:8px;padding:10px;border-radius:10px;background:#f7f7f7;
                border:1px solid #ddd;max-height:180px;overflow-y:auto;">
        <h4 style="margin-top:0;margin-bottom:8px;">Detected Classes and Corresponding Colors</h4>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:6px;">
            {cards}
        </div>
    </div>"""


def error_html(err: Exception) -> str:
    return f"""
    <div style="margin-top:8px;padding:12px;border-radius:10px;background:#fff3f3;
                border:1px solid #ffb3b3;color:#8a0000;font-size:14px;white-space:pre-wrap;">
        <strong>Error:</strong><br>{err}
    </div>"""