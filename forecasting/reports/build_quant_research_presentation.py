from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FORECASTING = ROOT / "forecasting"
REPORTS = FORECASTING / "reports"
ARTIFACTS = FORECASTING / "artifacts"

OUT_PPTX = REPORTS / "stock_forecasting_quant_report_ru.pptx"
OUT_MD = REPORTS / "stock_forecasting_quant_report_ru.md"

SLIDE_W = 12_192_000
SLIDE_H = 6_858_000
EMU_PER_IN = 914_400

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"

COLORS = {
    "bg": "F7F8FA",
    "white": "FFFFFF",
    "ink": "111827",
    "muted": "4B5563",
    "line": "D1D5DB",
    "panel": "EEF2F7",
    "teal": "0F766E",
    "teal_dark": "134E4A",
    "blue": "1D4ED8",
    "amber": "B45309",
    "red": "B91C1C",
    "green": "15803D",
    "slate": "243447",
}

MODEL_LABELS = {
    "hist_gradient_boosting": "HistGradientBoosting",
    "lstm": "LSTM per-ticker",
    "momentum": "Momentum",
    "naive_persistence": "Naive persistence",
    "ridge": "Ridge",
    "global_lstm_all": "Global LSTM all",
    "global_lstm_stationary": "Global LSTM stationary",
    "global_mamba_all": "Global Mamba all",
    "global_mamba_stationary": "Global Mamba stationary",
}


def inch(v: float) -> int:
    return int(round(v * EMU_PER_IN))


def pct(v: Any, digits: int = 1, signed: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    sign = "+" if signed else ""
    return f"{float(v) * 100:{sign}.{digits}f}%"


def num(v: Any, digits: int = 2, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    sign = "+" if signed else ""
    return f"{float(v):{sign}.{digits}f}"


def clean_date(value: str) -> str:
    return value.split("T", 1)[0]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs() -> dict[str, Any]:
    return {
        "eda": read_json(ARTIFACTS / "reports" / "eda_summary.json"),
        "lstm": read_json(ARTIFACTS / "reports" / "lstm_eda_summary.json"),
        "global_lstm": read_json(ARTIFACTS / "reports" / "global_lstm_eda_summary.json"),
        "mamba": read_json(ARTIFACTS / "reports" / "mamba_eda_summary.json"),
        "comparison": read_json(ARTIFACTS / "reports" / "final_strict_comparison.json"),
    }


def panel_rows(comparison: dict[str, Any], horizon: str) -> list[dict[str, Any]]:
    rows = [
        r
        for r in comparison["horizons"][horizon]["test_panel_signal_metrics"]
        if r.get("signal_mode") == "overlapping_tranches"
    ]
    return sorted(rows, key=lambda r: (r.get("cumulative_return") or -10), reverse=True)


def selected_summary(comparison: dict[str, Any], horizon: str) -> dict[str, Any]:
    rows = comparison["horizons"][horizon]["selected_model_buy_hold_benchmark_metrics"]
    positives = sum(1 for r in rows if (r.get("selected_model_cumulative_return") or 0) > 0)
    excess_positive = sum(1 for r in rows if (r.get("excess_cumulative_return") or 0) > 0)
    best = max(rows, key=lambda r: r.get("selected_model_cumulative_return") or -10)
    worst = min(rows, key=lambda r: r.get("selected_model_cumulative_return") or 10)
    return {
        "rows": rows,
        "positives": positives,
        "excess_positive": excess_positive,
        "best": best,
        "worst": worst,
    }


def leakage_summary(comparison: dict[str, Any], horizon: str) -> str:
    rows = comparison["horizons"][horizon]["leakage_audit"]
    passed = sum(1 for r in rows if r.get("passed"))
    return f"{passed}/{len(rows)}"


def mamba_summary(comparison: dict[str, Any], horizon: str) -> list[dict[str, Any]]:
    rows = [
        r
        for r in comparison["horizons"][horizon]["test_signal_metrics"]
        if r.get("comparison_source") == "global_mamba"
        and r.get("signal_mode") == "overlapping_tranches"
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["model_name"], []).append(row)
    out = []
    for model, model_rows in sorted(grouped.items()):
        returns = [r["cumulative_return"] for r in model_rows if r.get("cumulative_return") is not None]
        sharpes = [r["sharpe"] for r in model_rows if r.get("sharpe") is not None]
        out.append(
            {
                "model": model,
                "positive": sum(1 for v in returns if v > 0),
                "n": len(model_rows),
                "mean_return": sum(returns) / len(returns) if returns else None,
                "median_sharpe": sorted(sharpes)[len(sharpes) // 2] if sharpes else None,
                "best": max(model_rows, key=lambda r: r.get("cumulative_return") or -10),
                "worst": min(model_rows, key=lambda r: r.get("cumulative_return") or 10),
            }
        )
    return out


def xml_header() -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def solid_fill(color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def line(color: str = "FFFFFF", width: int = 0) -> str:
    if width <= 0:
        return "<a:ln><a:noFill/></a:ln>"
    return f'<a:ln w="{width}">{solid_fill(color)}</a:ln>'


def run_xml(text: str, size: int, color: str, bold: bool = False) -> str:
    b = ' b="1"' if bold else ""
    return (
        f'<a:r><a:rPr lang="ru-RU" sz="{size * 100}"{b}>'
        f'{solid_fill(color)}'
        '<a:latin typeface="Aptos"/><a:cs typeface="Aptos"/>'
        f'</a:rPr><a:t>{escape(text)}</a:t></a:r>'
    )


def paragraph_xml(text: str, size: int, color: str, bold: bool = False, align: str = "l") -> str:
    return f'<a:p><a:pPr algn="{align}"/>{run_xml(text, size, color, bold)}</a:p>'


def rich_paragraph_xml(parts: list[tuple[str, bool, str]], size: int, align: str = "l") -> str:
    runs = "".join(run_xml(text, size, color, bold) for text, bold, color in parts)
    return f'<a:p><a:pPr algn="{align}"/>{runs}</a:p>'


@dataclass
class Slide:
    title: str | None = None
    bg: str = COLORS["bg"]
    shapes: list[str] = field(default_factory=list)
    rels: dict[str, Path] = field(default_factory=dict)
    next_id: int = 10

    def sid(self) -> int:
        self.next_id += 1
        return self.next_id

    def add_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str | None = None,
        radius: str = "rect",
        opacity: int | None = None,
    ) -> None:
        sid = self.sid()
        fill_xml = solid_fill(fill)
        if opacity is not None:
            fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"><a:alpha val="{opacity}"/></a:srgbClr></a:solidFill>'
        self.shapes.append(
            f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{inch(x)}" y="{inch(y)}"/><a:ext cx="{inch(w)}" cy="{inch(h)}"/></a:xfrm>
    <a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>
    {fill_xml}
    {line(stroke or fill, 9500) if stroke else line()}
  </p:spPr>
  <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
</p:sp>"""
        )

    def add_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        lines: list[str] | str,
        size: int = 18,
        color: str = COLORS["ink"],
        bold: bool = False,
        align: str = "l",
        fill: str | None = None,
        stroke: str | None = None,
        margin: float = 0.08,
        radius: str = "rect",
    ) -> None:
        sid = self.sid()
        if isinstance(lines, str):
            parts = lines.split("\n")
        else:
            parts = lines
        body = "".join(paragraph_xml(p, size, color, bold, align) for p in parts)
        fill_xml = solid_fill(fill) if fill else "<a:noFill/>"
        stroke_xml = line(stroke, 9500) if stroke else line()
        self.shapes.append(
            f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{inch(x)}" y="{inch(y)}"/><a:ext cx="{inch(w)}" cy="{inch(h)}"/></a:xfrm>
    <a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>
    {fill_xml}
    {stroke_xml}
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" lIns="{inch(margin)}" tIns="{inch(margin)}" rIns="{inch(margin)}" bIns="{inch(margin)}"/>
    <a:lstStyle/>
    {body}
  </p:txBody>
</p:sp>"""
        )

    def add_rich_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        paragraphs: list[list[tuple[str, bool, str]]],
        size: int = 18,
        fill: str | None = None,
        stroke: str | None = None,
        margin: float = 0.08,
    ) -> None:
        sid = self.sid()
        body = "".join(rich_paragraph_xml(p, size) for p in paragraphs)
        fill_xml = solid_fill(fill) if fill else "<a:noFill/>"
        stroke_xml = line(stroke, 9500) if stroke else line()
        self.shapes.append(
            f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="Rich Text {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{inch(x)}" y="{inch(y)}"/><a:ext cx="{inch(w)}" cy="{inch(h)}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    {fill_xml}
    {stroke_xml}
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" lIns="{inch(margin)}" tIns="{inch(margin)}" rIns="{inch(margin)}" bIns="{inch(margin)}"/>
    <a:lstStyle/>
    {body}
  </p:txBody>
</p:sp>"""
        )

    def add_image(self, path: Path, x: float, y: float, w: float, h: float) -> None:
        sid = self.sid()
        rel_id = f"rId{len(self.rels) + 2}"
        self.rels[rel_id] = path
        with Image.open(path) as img:
            iw, ih = img.size
        ratio = iw / ih
        box_ratio = w / h
        fw, fh = w, h
        if ratio > box_ratio:
            fh = w / ratio
        else:
            fw = h * ratio
        fx = x + (w - fw) / 2
        fy = y + (h - fh) / 2
        self.shapes.append(
            f"""
<p:pic>
  <p:nvPicPr><p:cNvPr id="{sid}" name="{escape(path.name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
  <p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
  <p:spPr>
    <a:xfrm><a:off x="{inch(fx)}" y="{inch(fy)}"/><a:ext cx="{inch(fw)}" cy="{inch(fh)}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
  </p:spPr>
</p:pic>"""
        )


def add_slide_title(slide: Slide, title: str, subtitle: str | None = None) -> None:
    slide.add_rect(0, 0, 12.0, 0.16, COLORS["teal"])
    slide.add_text(0.45, 0.32, 8.8, 0.48, title, size=25, color=COLORS["ink"], bold=True, margin=0.0)
    if subtitle:
        slide.add_text(0.47, 0.78, 8.8, 0.3, subtitle, size=10, color=COLORS["muted"], margin=0.0)


def add_footer(slide: Slide, idx: int) -> None:
    slide.add_text(
        0.45,
        7.08,
        6.8,
        0.18,
        "Источник: forecasting/artifacts/reports/final_strict_comparison.json и локальные notebook artifacts",
        size=7,
        color="6B7280",
        margin=0.0,
    )
    slide.add_text(11.45, 7.08, 0.32, 0.18, str(idx), size=7, color="6B7280", align="r", margin=0.0)


def add_metric_card(slide: Slide, x: float, y: float, w: float, h: float, label: str, value: str, accent: str) -> None:
    slide.add_rect(x, y, w, h, COLORS["white"], stroke=COLORS["line"])
    slide.add_rect(x, y, 0.08, h, accent)
    slide.add_text(x + 0.18, y + 0.12, w - 0.25, 0.22, label.upper(), size=7, color=COLORS["muted"], bold=True, margin=0.0)
    slide.add_text(x + 0.18, y + 0.38, w - 0.25, 0.42, value, size=18, color=COLORS["ink"], bold=True, margin=0.0)


def add_bullets(slide: Slide, x: float, y: float, w: float, h: float, bullets: list[str], size: int = 15) -> None:
    slide.add_text(
        x,
        y,
        w,
        h,
        [f"• {b}" for b in bullets],
        size=size,
        color=COLORS["ink"],
        margin=0.04,
    )


def add_table(
    slide: Slide,
    x: float,
    y: float,
    col_widths: list[float],
    row_h: float,
    headers: list[str],
    rows: list[list[str]],
    font_size: int = 9,
    header_fill: str = COLORS["slate"],
) -> None:
    total_w = sum(col_widths)
    slide.add_rect(x, y, total_w, row_h, header_fill, stroke=header_fill)
    cx = x
    for i, head in enumerate(headers):
        slide.add_text(cx + 0.03, y + 0.03, col_widths[i] - 0.06, row_h - 0.04, head, size=font_size, color=COLORS["white"], bold=True, margin=0.01)
        cx += col_widths[i]
    for r_idx, row in enumerate(rows):
        ry = y + row_h * (r_idx + 1)
        fill = COLORS["white"] if r_idx % 2 == 0 else "F3F4F6"
        slide.add_rect(x, ry, total_w, row_h, fill, stroke=COLORS["line"])
        cx = x
        for c_idx, cell in enumerate(row):
            color = COLORS["ink"]
            if c_idx in {1, 2} and cell.startswith("+"):
                color = COLORS["green"]
            if c_idx in {1, 2} and cell.startswith("-"):
                color = COLORS["red"]
            slide.add_text(cx + 0.03, ry + 0.03, col_widths[c_idx] - 0.06, row_h - 0.04, cell, size=font_size, color=color, margin=0.01)
            cx += col_widths[c_idx]


def model_table_rows(rows: list[dict[str, Any]], limit: int = 8) -> list[list[str]]:
    out = []
    for row in rows[:limit]:
        out.append(
            [
                MODEL_LABELS.get(row["model_name"], row["model_name"]),
                pct(row.get("cumulative_return")),
                num(row.get("sharpe")),
                pct(row.get("max_drawdown")),
                pct(row.get("turnover"), digits=2, signed=False),
                str(int(row.get("number_of_trades") or 0)),
            ]
        )
    return out


def create_markdown(data: dict[str, Any]) -> str:
    eda = data["eda"]
    lstm = data["lstm"]
    global_lstm = data["global_lstm"]
    mamba = data["mamba"]
    comparison = data["comparison"]
    week_rows = panel_rows(comparison, "week")
    month_rows = panel_rows(comparison, "month")
    week_selected = selected_summary(comparison, "week")
    month_selected = selected_summary(comparison, "month")
    week_mamba = mamba_summary(comparison, "week")
    month_mamba = mamba_summary(comparison, "month")

    def panel_md(rows: list[dict[str, Any]]) -> str:
        lines = ["| Model | Cum. return | Sharpe | Max DD | Turnover | Trades |", "|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        MODEL_LABELS.get(row["model_name"], row["model_name"]),
                        pct(row.get("cumulative_return")),
                        num(row.get("sharpe")),
                        pct(row.get("max_drawdown")),
                        pct(row.get("turnover"), digits=2, signed=False),
                        str(int(row.get("number_of_trades") or 0)),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    def mamba_md(rows: list[dict[str, Any]]) -> str:
        lines = ["| Model | Positive tickers | Mean ticker return | Median Sharpe | Best ticker |", "|---|---:|---:|---:|---|"]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        MODEL_LABELS.get(row["model"], row["model"]),
                        f"{row['positive']}/{row['n']}",
                        pct(row["mean_return"]),
                        num(row["median_sharpe"]),
                        f"{row['best']['ticker']} {pct(row['best']['cumulative_return'])}",
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    return f"""# Прогнозирование доходности акций: количественный отчет

Дата отчета: 2026-06-14
PowerPoint: `{OUT_PPTX.name}`

## Ключевые выводы

- Данные: {eda["rows_clean"]:,} OHLCV-наблюдений, {eda["tickers_count"]} тикеров, период {clean_date(eda["start_date"])} - {clean_date(eda["end_date"])}.
- Горизонты: неделя = 5 торговых дней, месяц = 21 торговый день; target = future log return от next open.
- Anti-leakage audit: week {leakage_summary(comparison, "week")}, month {leakage_summary(comparison, "month")}.
- Лучший full-panel результат за неделю: {MODEL_LABELS[week_rows[0]["model_name"]]}, cumulative return {pct(week_rows[0]["cumulative_return"])}, Sharpe {num(week_rows[0]["sharpe"])}.
- Лучший full-panel результат за месяц: {MODEL_LABELS[month_rows[0]["model_name"]]}, cumulative return {pct(month_rows[0]["cumulative_return"])}, Sharpe {num(month_rows[0]["sharpe"])}.
- Выбранные по directional accuracy модели положительны в тесте: week {week_selected["positives"]}/7, month {month_selected["positives"]}/7.

## Слайды

### 1. Титульный слайд

Проект: `forecasting/`. Горизонты: week = 5 торговых дней, month = 21 торговый день. Target: next-open-to-future-open log return.

### 2. Executive Summary

- Research-loop завершен: EDA, leakage-safe features, strict validation/test, signal backtest, графики и модельные артефакты.
- Anti-leakage checks: week {leakage_summary(comparison, "week")}, month {leakage_summary(comparison, "month")}.
- Главный вывод: production trading требует selector по trading utility, а не только directional accuracy.

### 3. Объем работ

Выполнены notebook stages: `01_eda`, LSTM/global LSTM/Mamba EDA, table/LSTM/global LSTM/Mamba forecasting, финальное сравнение моделей.

### 4. Данные и признаки

| Dataset | Rows / sequences | Features | Комментарий |
|---|---:|---:|---|
| Base table | {eda['rows_model_dataset_by_horizon']['week']:,} | {eda['feature_count']} | rolling/cross-sectional OHLCV |
| Per-ticker LSTM | {lstm['artifacts'][0]['rows_model_dataset']:,} | {lstm['artifacts'][0]['feature_count']} | +33 sequence features |
| Global LSTM | {global_lstm['artifacts'][0]['sequences_lookback_60']:,} | {global_lstm['artifacts'][0]['global_all_features']} | ticker one-hot + pooled panel |
| Global Mamba | {mamba['artifacts'][0]['sequences_lookback_126']:,} | {mamba['artifacts'][0]['mamba_all_features']} | +108 Mamba-specific features |

### 5. Валидационный протокол

- Chronological train / validation / test; random split не используется.
- Purging по target dates защищает от leakage.
- Headline signal mode: `overlapping_tranches`.
- Costs: 10 bps transaction cost + 5 bps slippage.

### 6. Модельный периметр

Baselines: naive persistence, momentum. Tabular: ridge, HistGradientBoosting. Deep: per-ticker LSTM, global LSTM all/stationary, global Mamba all/stationary.

### 7. Week: full-panel results

{panel_md(week_rows)}

### 8. Month: full-panel results

{panel_md(month_rows)}

### 9. Selected models vs buy-and-hold

| Horizon | Positive selected models | Excess over buy-and-hold | Best selected | Worst selected |
|---|---:|---:|---|---|
| Week | {week_selected['positives']}/7 | {week_selected['excess_positive']}/7 | {week_selected['best']['ticker']} {pct(week_selected['best']['selected_model_cumulative_return'])} | {week_selected['worst']['ticker']} {pct(week_selected['worst']['selected_model_cumulative_return'])} |
| Month | {month_selected['positives']}/7 | {month_selected['excess_positive']}/7 | {month_selected['best']['ticker']} {pct(month_selected['best']['selected_model_cumulative_return'])} | {month_selected['worst']['ticker']} {pct(month_selected['worst']['selected_model_cumulative_return'])} |

### 10. Диагностика отбора

Directional accuracy слабо связан с торговой метрикой. В аудите корреляция validation directional accuracy vs test Sharpe равна -0.006 для week и 0.049 для month; validation Sharpe vs test Sharpe равна -0.043 и -0.543 соответственно.

### 11. Mamba extension

Week:

{mamba_md(week_mamba)}

Month:

{mamba_md(month_mamba)}

Примечание: mean return здесь является простым средним ticker-level metrics, не capital-weighted panel.

### 12. Production и риски

- Objective mismatch.
- Single out-of-sample regime.
- Multiple testing.
- Limited-history тикеры MBNK и SVCB.
- Serving/API mismatch между researched daily ML horizons и fallback AutoARIMA timeframes.

### 13. Рекомендации

1. Заменить основной selector на validation trading utility с штрафами за drawdown, turnover и нестабильность.
2. Добавить multi-fold purged walk-forward across regimes.
3. Оптимизировать signal rule и position sizing, а не только модель прогноза.
4. Синхронизировать serving API с реально исследованными горизонтами и daily timeframe.
"""


def build_slides(data: dict[str, Any]) -> list[Slide]:
    eda = data["eda"]
    lstm = data["lstm"]
    global_lstm = data["global_lstm"]
    mamba = data["mamba"]
    comparison = data["comparison"]
    week_rows = panel_rows(comparison, "week")
    month_rows = panel_rows(comparison, "month")
    week_selected = selected_summary(comparison, "week")
    month_selected = selected_summary(comparison, "month")
    week_mamba = mamba_summary(comparison, "week")
    month_mamba = mamba_summary(comparison, "month")

    slides: list[Slide] = []

    s = Slide(bg="F3F7F6")
    s.add_rect(0, 0, 12.0, 7.5, "F3F7F6")
    s.add_rect(0, 0, 0.36, 7.5, COLORS["teal_dark"])
    s.add_text(0.75, 0.75, 9.5, 0.65, "Прогнозирование доходности акций", size=32, color=COLORS["ink"], bold=True, margin=0.0)
    s.add_text(0.78, 1.5, 8.8, 0.45, "Количественный исследовательский отчет по pipeline forecasting/", size=17, color=COLORS["muted"], margin=0.0)
    s.add_text(0.78, 2.02, 8.0, 0.32, "Дата подготовки: 14 июня 2026", size=12, color=COLORS["muted"], margin=0.0)
    add_metric_card(s, 0.78, 3.05, 2.45, 1.05, "Вселенная", f'{eda["tickers_count"]} тикеров', COLORS["teal"])
    add_metric_card(s, 3.45, 3.05, 2.45, 1.05, "Данные", f'{eda["rows_clean"]:,}'.replace(",", " "), COLORS["blue"])
    add_metric_card(s, 6.12, 3.05, 2.45, 1.05, "Период", f'{clean_date(eda["start_date"])}', COLORS["amber"])
    add_metric_card(s, 8.79, 3.05, 2.45, 1.05, "Последняя дата", f'{clean_date(eda["end_date"])}', COLORS["green"])
    s.add_text(0.78, 5.2, 8.6, 0.52, "Неделя: 5 торговых дней | Месяц: 21 торговый день | target: next-open-to-future-open log return", size=14, color=COLORS["ink"], margin=0.0)
    s.add_text(0.78, 6.65, 7.2, 0.22, "Research code. Не является инвестиционной рекомендацией.", size=8, color="6B7280", margin=0.0)
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Executive Summary", "Что можно заключить по готовому исследовательскому контуру")
    add_bullets(
        s,
        0.65,
        1.25,
        6.0,
        3.2,
        [
            "Пайплайн закрывает полный research-loop: EDA, leakage-safe features, strict validation/test, backtest, графики и модельные артефакты.",
            f"Anti-leakage проверки прошли полностью: week {leakage_summary(comparison, 'week')}, month {leakage_summary(comparison, 'month')}.",
            f"Лучший full-panel week: {MODEL_LABELS[week_rows[0]['model_name']]}, {pct(week_rows[0]['cumulative_return'])}, Sharpe {num(week_rows[0]['sharpe'])}.",
            f"Лучший full-panel month: {MODEL_LABELS[month_rows[0]['model_name']]}, {pct(month_rows[0]['cumulative_return'])}, Sharpe {num(month_rows[0]['sharpe'])}, но всего {int(month_rows[0]['number_of_trades'])} сделки.",
            "Основной риск не в утечке данных, а в objective mismatch: directional accuracy плохо выбирает торгово устойчивые модели.",
        ],
        size=15,
    )
    add_metric_card(s, 7.15, 1.25, 1.55, 0.95, "Week winner", pct(week_rows[0]["cumulative_return"]), COLORS["green"])
    add_metric_card(s, 8.95, 1.25, 1.55, 0.95, "Sharpe", num(week_rows[0]["sharpe"]), COLORS["green"])
    add_metric_card(s, 7.15, 2.45, 1.55, 0.95, "Month winner", pct(month_rows[0]["cumulative_return"]), COLORS["blue"])
    add_metric_card(s, 8.95, 2.45, 1.55, 0.95, "Sharpe", num(month_rows[0]["sharpe"]), COLORS["blue"])
    s.add_text(7.15, 4.0, 3.65, 0.95, "Статус: сильный research prototype; для production trading нужен selector по utility, multi-fold проверка и sizing.", size=13, color=COLORS["ink"], fill=COLORS["white"], stroke=COLORS["line"])
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Объем Работ", "Сквозной процесс от данных до тестовых signal metrics")
    steps = [
        ("01", "EDA и контроль качества", "OHLCV schema, missing checks, target construction."),
        ("02", "Leakage-safe features", f"{eda['feature_count']} базовых признаков; rolling, volatility, volume, cross-sectional."),
        ("03", "Sequence datasets", "LSTM/global LSTM/Mamba artifacts без fitting scaler до split."),
        ("04", "Strict training", "Optuna, purged train/validation/test, final refit."),
        ("05", "Backtest", "Long/cash signal, overlapping tranches, costs 10 bps + slippage 5 bps."),
        ("06", "Artifacts/MLOps", "Reports, plots, final.pkl, MLflow registration hooks."),
    ]
    y = 1.15
    for i, title, desc in steps:
        s.add_rect(0.7, y, 0.55, 0.55, COLORS["teal"])
        s.add_text(0.78, y + 0.12, 0.4, 0.2, i, size=12, color=COLORS["white"], bold=True, margin=0.0)
        s.add_text(1.45, y, 3.4, 0.22, title, size=14, color=COLORS["ink"], bold=True, margin=0.0)
        s.add_text(1.45, y + 0.28, 8.8, 0.22, desc, size=11, color=COLORS["muted"], margin=0.0)
        y += 0.82
    s.add_text(0.7, 6.35, 10.5, 0.35, "Ноутбуки: 01_eda, 01b/01c/01d sequence EDA, 02 table/LSTM/global LSTM/Mamba, 03 model comparison.", size=11, color=COLORS["ink"], fill=COLORS["white"], stroke=COLORS["line"])
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Данные И Признаки", "Единая база для tabular, LSTM, global LSTM и Mamba")
    feature_rows = [
        ["Base table", f"{eda['rows_model_dataset_by_horizon']['week']:,}".replace(",", " "), f"{eda['feature_count']}", "rolling/cross-sectional OHLCV"],
        ["Per-ticker LSTM", f"{lstm['artifacts'][0]['rows_model_dataset']:,}".replace(",", " "), f"{lstm['artifacts'][0]['feature_count']}", "+33 sequence features"],
        ["Global LSTM", f"{global_lstm['artifacts'][0]['sequences_lookback_60']:,}".replace(",", " "), f"{global_lstm['artifacts'][0]['global_all_features']}", "ticker one-hot + global pooling"],
        ["Global Mamba", f"{mamba['artifacts'][0]['sequences_lookback_126']:,}".replace(",", " "), f"{mamba['artifacts'][0]['mamba_all_features']}", "+108 regime/gap/vol/volume features"],
    ]
    add_table(s, 0.75, 1.25, [2.2, 1.55, 1.25, 4.6], 0.45, ["Dataset", "Rows/seq.", "Features", "Комментарий"], feature_rows, font_size=10)
    add_bullets(
        s,
        0.82,
        4.0,
        9.8,
        1.45,
        [
            f"Clean OHLCV: {eda['rows_clean']:,} строк, {eda['tickers_count']} тикеров, {clean_date(eda['start_date'])} - {clean_date(eda['end_date'])}.",
            "Целевая переменная строится через future log return; признаки используют только текущие и прошлые строки.",
            "Горизонт month теряет часть последних строк из-за target shift: 19 295 против 19 407 для week.",
        ],
        size=13,
    )
    s.add_image(ARTIFACTS / "plots" / "eda_close_volume.png", 7.25, 4.95, 4.0, 1.45)
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Валидационный Протокол", "Почему результаты можно считать out-of-sample диагностикой")
    protocol_rows = [
        ["Split", "Chronological train / validation / test; random split не используется"],
        ["Purging", "Training rows с target_date после начала validation/test удаляются"],
        ["History cap", "Mature names: максимум 1260 train rows; limited history flagged"],
        ["Signal", "Long/cash, expanding median anchor, overlapping tranches headline mode"],
        ["Costs", "10 bps transaction cost + 5 bps slippage"],
        ["Audit", f"Leakage checks: week {leakage_summary(comparison, 'week')}, month {leakage_summary(comparison, 'month')}"],
    ]
    add_table(s, 0.75, 1.1, [2.0, 8.7], 0.48, ["Компонент", "Реализация"], protocol_rows, font_size=10)
    week_split = comparison["horizons"]["week"]["outer_splits"][0]
    month_split = comparison["horizons"]["month"]["outer_splits"][0]
    s.add_text(0.85, 4.8, 4.9, 0.85, f"Week test window example: {clean_date(week_split['test_start'])} - {clean_date(week_split['test_end'])}; target dates до {clean_date(week_split['test_target_end'])}.", size=13, color=COLORS["ink"], fill=COLORS["white"], stroke=COLORS["line"])
    s.add_text(6.15, 4.8, 4.9, 0.85, f"Month test window example: {clean_date(month_split['test_start'])} - {clean_date(month_split['test_end'])}; target dates до {clean_date(month_split['test_target_end'])}.", size=13, color=COLORS["ink"], fill=COLORS["white"], stroke=COLORS["line"])
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Модельный Периметр", "Сравнение простых, табличных и sequence-моделей")
    model_rows = [
        ["Baselines", "naive_persistence, momentum", "Контрольная точка и low-complexity alpha"],
        ["Linear", "ridge", "Стабильный табличный benchmark"],
        ["Tree boosting", "hist_gradient_boosting", "Нелинейные interactions без deep stack"],
        ["Per-ticker deep", "LSTM", "Отдельная модель на тикер и горизонт"],
        ["Pooled deep", "global_lstm_all/stationary", "Одна модель на panel с calendar-global purging"],
        ["State-space", "global_mamba_all/stationary", "Official mamba-ssm, расширенный feature set"],
    ]
    add_table(s, 0.7, 1.15, [1.9, 3.35, 5.1], 0.5, ["Класс", "Модели", "Роль в исследовании"], model_rows, font_size=10)
    s.add_text(0.78, 5.25, 9.9, 0.9, "Метрика отбора в текущем final report: directional_accuracy. Это удобно для диагностики направлений, но не оптимизирует net PnL, drawdown, turnover и trade count.", size=14, color=COLORS["ink"], fill="FFF7ED", stroke="FDBA74")
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Тестовые Результаты: Week", "Full-panel signal metrics, overlapping tranches")
    add_table(
        s,
        0.55,
        1.08,
        [2.75, 1.15, 0.9, 1.05, 1.0, 0.7],
        0.42,
        ["Model", "Cum. ret", "Sharpe", "Max DD", "Turnover", "Trades"],
        model_table_rows(week_rows),
        font_size=8,
    )
    s.add_text(8.35, 1.1, 2.65, 0.95, f"Победитель: {MODEL_LABELS[week_rows[0]['model_name']]}\n{pct(week_rows[0]['cumulative_return'])}, Sharpe {num(week_rows[0]['sharpe'])}", size=15, color=COLORS["white"], bold=True, fill=COLORS["green"], margin=0.1)
    add_bullets(
        s,
        8.35,
        2.35,
        2.8,
        2.3,
        [
            "Ridge дал лучший week panel и умеренный max drawdown.",
            "Global LSTM stationary положительный, но с 8 trades.",
            "Naive/momentum не покрывают costs в full-panel тесте.",
        ],
        size=12,
    )
    s.add_image(ARTIFACTS / "horizons" / "week" / "strict_protocol" / "plots" / "summary" / "selected_models_test_economic_metrics.png", 0.8, 4.85, 10.1, 1.55)
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Тестовые Результаты: Month", "Full-panel signal metrics, overlapping tranches")
    add_table(
        s,
        0.55,
        1.08,
        [2.75, 1.15, 0.9, 1.05, 1.0, 0.7],
        0.42,
        ["Model", "Cum. ret", "Sharpe", "Max DD", "Turnover", "Trades"],
        model_table_rows(month_rows),
        font_size=8,
    )
    s.add_text(8.35, 1.1, 2.65, 0.95, f"Победитель: {MODEL_LABELS[month_rows[0]['model_name']]}\n{pct(month_rows[0]['cumulative_return'])}, Sharpe {num(month_rows[0]['sharpe'])}", size=15, color=COLORS["white"], bold=True, fill=COLORS["blue"], margin=0.1)
    add_bullets(
        s,
        8.35,
        2.35,
        2.8,
        2.3,
        [
            "Global LSTM all выглядит перспективно, но sample короткий.",
            "Sharpe завышается при низкой волатильности и 33 trades.",
            "Naive persistence тоже положительный, но с высоким turnover.",
        ],
        size=12,
    )
    s.add_image(ARTIFACTS / "horizons" / "month" / "strict_protocol" / "plots" / "summary" / "selected_models_test_economic_metrics.png", 0.8, 4.85, 10.1, 1.55)
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Выбранные Модели И Buy-And-Hold", "Directional-accuracy selection не всегда дает экономическое преимущество")
    summary_rows = [
        [
            "Week",
            f"{week_selected['positives']}/7",
            f"{week_selected['excess_positive']}/7",
            f"{week_selected['best']['ticker']} {pct(week_selected['best']['selected_model_cumulative_return'])}",
            f"{week_selected['worst']['ticker']} {pct(week_selected['worst']['selected_model_cumulative_return'])}",
        ],
        [
            "Month",
            f"{month_selected['positives']}/7",
            f"{month_selected['excess_positive']}/7",
            f"{month_selected['best']['ticker']} {pct(month_selected['best']['selected_model_cumulative_return'])}",
            f"{month_selected['worst']['ticker']} {pct(month_selected['worst']['selected_model_cumulative_return'])}",
        ],
    ]
    add_table(s, 0.75, 1.1, [1.05, 1.5, 1.55, 2.7, 2.7], 0.5, ["Horizon", "Positive", "Excess > BH", "Best selected", "Worst selected"], summary_rows, font_size=10)
    s.add_image(ARTIFACTS / "horizons" / "week" / "strict_protocol" / "plots" / "summary" / "selected_models_vs_buy_hold_benchmarks.png", 0.85, 2.65, 10.2, 1.45)
    s.add_image(ARTIFACTS / "horizons" / "month" / "strict_protocol" / "plots" / "summary" / "selected_models_vs_buy_hold_benchmarks.png", 0.85, 4.55, 10.2, 1.45)
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Диагностика Отбора", "Validation directional accuracy почти не предсказывает test PnL")
    corr_rows = [
        ["DA vs test Sharpe", "-0.006", "0.049"],
        ["DA vs test cumulative return", "-0.042", "0.049"],
        ["Validation Sharpe vs test Sharpe", "-0.043", "-0.543"],
        ["Validation cum. return vs test cum. return", "-0.257", "-0.314"],
        ["DA vs test DA", "0.326", "-0.369"],
    ]
    add_table(s, 0.65, 1.1, [3.8, 1.2, 1.2], 0.44, ["Relationship", "Week corr.", "Month corr."], corr_rows, font_size=9)
    s.add_image(ARTIFACTS / "horizons" / "week" / "strict_protocol" / "plots" / "summary" / "validation_sharpe_vs_test_sharpe.png", 6.9, 1.0, 4.1, 2.5)
    s.add_image(ARTIFACTS / "horizons" / "month" / "strict_protocol" / "plots" / "summary" / "validation_sharpe_vs_test_sharpe.png", 6.9, 3.85, 4.1, 2.5)
    s.add_text(0.7, 4.45, 5.55, 1.0, "Вывод: модель нужно выбирать по validation trading utility после costs и risk constraints, а directional accuracy оставить диагностической метрикой.", size=14, color=COLORS["ink"], fill="FEF2F2", stroke="FCA5A5")
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Mamba Extension", "Новый state-space контур уже встроен в notebook chain")
    mamba_rows = []
    for horizon, rows in [("Week", week_mamba), ("Month", month_mamba)]:
        for row in rows:
            mamba_rows.append(
                [
                    horizon,
                    MODEL_LABELS[row["model"]],
                    f"{row['positive']}/{row['n']}",
                    pct(row["mean_return"]),
                    num(row["median_sharpe"]),
                    f"{row['best']['ticker']} {pct(row['best']['cumulative_return'])}",
                ]
            )
    add_table(s, 0.65, 1.1, [0.85, 2.25, 1.25, 1.25, 1.25, 2.0], 0.42, ["Hz", "Model", "Pos.", "Mean ret", "Med Sharpe", "Best ticker"], mamba_rows, font_size=8)
    add_bullets(
        s,
        0.75,
        4.4,
        10.1,
        1.3,
        [
            f"Mamba feature set: {mamba['artifacts'][0]['mamba_all_features']} all-features и {mamba['artifacts'][0]['mamba_stationary_features']} stationary-features; рекомендованные lookbacks 60/90/126/252.",
            "Leakage audit для global_mamba входит в общий strict comparison и проходит проверки.",
            "Следующий шаг: добавить Mamba в агрегированную panel-selection таблицу и сравнивать по trading utility.",
        ],
        size=12,
    )
    s.add_text(0.75, 6.05, 9.7, 0.35, "Mean ret на этом слайде - простое среднее ticker-level signal metrics, не capital-weighted panel.", size=9, color=COLORS["muted"], margin=0.0)
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Production И Риски", "Что важно перед переводом из research в торговый продукт")
    risk_rows = [
        ["Objective mismatch", "Текущий selector optimizing directional_accuracy не учитывает costs, drawdown и turnover."],
        ["Single regime", "Test sample около полугода; для Sharpe и low-trade моделей этого мало."],
        ["Multiple testing", "Много моделей, тикеров, горизонтов и thresholds увеличивают риск selection luck."],
        ["Limited history", "MBNK и SVCB явно flagged; оценки нестабильнее mature names."],
        ["Serving mismatch", "ML daily horizons не равны произвольным forecast_period/time_frame в bot/API."],
    ]
    add_table(s, 0.7, 1.1, [2.3, 8.25], 0.52, ["Риск", "Практический смысл"], risk_rows, font_size=10)
    s.add_text(0.78, 5.35, 9.8, 0.55, "MLflow контур предусмотрен: stock-forecast-train-register-mlflow --artifact-root artifacts --skip-training", size=12, color=COLORS["ink"], fill=COLORS["white"], stroke=COLORS["line"])
    slides.append(s)

    s = Slide()
    add_slide_title(s, "Рекомендации", "Приоритеты следующей итерации")
    recs = [
        ("1", "Заменить selector", "utility = Sharpe - lambda_drawdown*DD - lambda_turnover*turnover - instability penalty; hard constraints до ранжирования."),
        ("2", "Multi-fold purged walk-forward", "Несколько режимов рынка, worst-fold Sharpe, bootstrap CI, deflated/probabilistic Sharpe."),
        ("3", "Оптимизировать торговое правило", "No-trade band, threshold by expected net edge, hysteresis, turnover cap, volatility-adjusted thresholds."),
        ("4", "Добавить position sizing", "Vol targeting, forecast edge / forecast volatility, exposure cap, liquidity cap, drawdown de-risking."),
        ("5", "Синхронизировать serving", "API должен различать researched daily ML horizons и fallback AutoARIMA timeframes."),
    ]
    y = 1.15
    for num_i, title, desc in recs:
        s.add_rect(0.75, y, 0.5, 0.5, COLORS["teal"])
        s.add_text(0.91, y + 0.12, 0.18, 0.2, num_i, size=12, color=COLORS["white"], bold=True, margin=0.0)
        s.add_text(1.45, y - 0.02, 3.0, 0.25, title, size=14, color=COLORS["ink"], bold=True, margin=0.0)
        s.add_text(1.45, y + 0.3, 8.9, 0.28, desc, size=11, color=COLORS["muted"], margin=0.0)
        y += 0.95
    s.add_text(0.75, 6.35, 10.3, 0.35, "Bottom line: продолжать как research platform; production trading approval только после utility-based selection и multi-regime evidence.", size=12, color=COLORS["white"], bold=True, fill=COLORS["slate"], margin=0.08)
    slides.append(s)

    for i, slide in enumerate(slides, 1):
        if i > 1:
            add_footer(slide, i)
    return slides


def slide_xml(slide: Slide) -> str:
    shapes = "\n".join(slide.shapes)
    return f"""{xml_header()}
<p:sld xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:cSld>
    <p:bg><p:bgPr>{solid_fill(slide.bg)}<a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {shapes}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def slide_rels_xml(slide: Slide, media_map: dict[Path, str]) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    for rel_id, path in slide.rels.items():
        rels.append(
            f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_map[path]}"/>'
        )
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>"""


def content_types_xml(slide_count: int) -> str:
    overrides = [
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
    ]
    for idx in range(1, slide_count + 1):
        overrides.append(
            f'<Override PartName="/ppt/slides/slide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
    return f"""{xml_header()}
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  {''.join(overrides)}
</Types>"""


def package_rels_xml() -> str:
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def presentation_xml(slide_count: int) -> str:
    sld_ids = "".join(
        f'<p:sldId id="{255 + idx}" r:id="rId{idx + 1}"/>' for idx in range(1, slide_count + 1)
    )
    return f"""{xml_header()}
<p:presentation xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}" saveSubsetFonts="1">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>"""


def presentation_rels_xml(slide_count: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    for idx in range(1, slide_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{idx}.xml"/>'
        )
    rels.extend(
        [
            f'<Relationship Id="rId{slide_count + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>',
            f'<Relationship Id="rId{slide_count + 3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>',
            f'<Relationship Id="rId{slide_count + 4}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>',
        ]
    )
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>"""


def slide_master_xml() -> str:
    return f"""{xml_header()}
<p:sldMaster xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  </p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def slide_master_rels_xml() -> str:
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def slide_layout_xml() -> str:
    return f"""{xml_header()}
<p:sldLayout xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def slide_layout_rels_xml() -> str:
    return f"""{xml_header()}
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


def theme_xml() -> str:
    return f"""{xml_header()}
<a:theme xmlns:a="{NS_A}" name="QuantReport">
  <a:themeElements>
    <a:clrScheme name="QuantReport">
      <a:dk1><a:srgbClr val="{COLORS['ink']}"/></a:dk1>
      <a:lt1><a:srgbClr val="{COLORS['white']}"/></a:lt1>
      <a:dk2><a:srgbClr val="{COLORS['slate']}"/></a:dk2>
      <a:lt2><a:srgbClr val="{COLORS['bg']}"/></a:lt2>
      <a:accent1><a:srgbClr val="{COLORS['teal']}"/></a:accent1>
      <a:accent2><a:srgbClr val="{COLORS['blue']}"/></a:accent2>
      <a:accent3><a:srgbClr val="{COLORS['amber']}"/></a:accent3>
      <a:accent4><a:srgbClr val="{COLORS['green']}"/></a:accent4>
      <a:accent5><a:srgbClr val="{COLORS['red']}"/></a:accent5>
      <a:accent6><a:srgbClr val="6B7280"/></a:accent6>
      <a:hlink><a:srgbClr val="{COLORS['blue']}"/></a:hlink>
      <a:folHlink><a:srgbClr val="{COLORS['teal']}"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="QuantReport"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>"""


def pres_props_xml() -> str:
    return f"""{xml_header()}
<p:presentationPr xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:showPr showNarration="1"><p:present/></p:showPr>
</p:presentationPr>"""


def view_props_xml() -> str:
    return f"""{xml_header()}
<p:viewPr xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:normalViewPr><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/></p:normalViewPr>
  <p:slideViewPr><p:cSldViewPr><p:cViewPr varScale="1"><p:scale><a:sx n="100" d="100"/><a:sy n="100" d="100"/></p:scale><p:origin x="0" y="0"/></p:cViewPr><p:guideLst/></p:cSldViewPr></p:slideViewPr>
  <p:notesTextViewPr><p:cViewPr varScale="1"><p:scale><a:sx n="100" d="100"/><a:sy n="100" d="100"/></p:scale><p:origin x="0" y="0"/></p:cViewPr></p:notesTextViewPr>
  <p:gridSpacing cx="76200" cy="76200"/>
</p:viewPr>"""


def table_styles_xml() -> str:
    return f"""{xml_header()}
<a:tblStyleLst xmlns:a="{NS_A}" def="{{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}}"/>"""


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""{xml_header()}
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Прогнозирование доходности акций</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def app_xml(slide_count: int) -> str:
    return f"""{xml_header()}
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex OOXML generator</Application>
  <PresentationFormat>Widescreen</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Company/>
</Properties>"""


def write_pptx(slides: list[Slide]) -> None:
    media_paths: list[Path] = []
    for slide in slides:
        for path in slide.rels.values():
            if path not in media_paths:
                media_paths.append(path)
    media_map = {path: f"image{idx + 1}.png" for idx, path in enumerate(media_paths)}

    with zipfile.ZipFile(OUT_PPTX, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(slides)))
        zf.writestr("_rels/.rels", package_rels_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("docProps/app.xml", app_xml(len(slides)))
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels_xml(len(slides)))
        zf.writestr("ppt/presProps.xml", pres_props_xml())
        zf.writestr("ppt/viewProps.xml", view_props_xml())
        zf.writestr("ppt/tableStyles.xml", table_styles_xml())
        zf.writestr("ppt/theme/theme1.xml", theme_xml())
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels_xml())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels_xml())
        for idx, slide in enumerate(slides, 1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml(slide))
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels_xml(slide, media_map))
        for path, name in media_map.items():
            zf.write(path, f"ppt/media/{name}")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    slides = build_slides(data)
    write_pptx(slides)
    OUT_MD.write_text(create_markdown(data), encoding="utf-8")
    print(f"Wrote {OUT_PPTX.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
