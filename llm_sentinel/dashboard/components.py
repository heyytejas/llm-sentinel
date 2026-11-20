"""
Reusable UI Components and Plotly Chart Builders for LLM Sentinel Dashboard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Color Palette for Security Dashboard
COLOR_CRITICAL = "#EF4444"
COLOR_HIGH = "#F97316"
COLOR_MEDIUM = "#F59E0B"
COLOR_LOW = "#10B981"
COLOR_SAFE = "#06B6D4"
COLOR_PRIMARY = "#3B82F6"
COLOR_BACKGROUND = "#0F172A"
COLOR_CARD = "#1E293B"
COLOR_TEXT = "#F8FAFC"
COLOR_MUTED = "#94A3B8"


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent slate/dark security styling to Plotly figures."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT, family="sans-serif", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.15)",
        zerolinecolor="rgba(148, 163, 184, 0.3)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.15)",
        zerolinecolor="rgba(148, 163, 184, 0.3)",
    )
    return fig


def render_severity_badge(severity: str) -> str:
    """Return styled HTML badge for attack severity."""
    sev = str(severity).lower()
    if sev == "critical":
        bg, border, text = "#7F1D1D", "#DC2626", "#FCA5A5"
    elif sev == "high":
        bg, border, text = "#7C2D12", "#EA580C", "#FDBA74"
    elif sev == "medium":
        bg, border, text = "#78350F", "#D97706", "#FDE68A"
    else:  # low
        bg, border, text = "#064E3B", "#059669", "#A7F3D0"

    return f"""<span style="background-color: {bg}; border: 1px solid {border}; color: {text};
        padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">
        {sev.upper()}</span>"""


def render_verdict_badge(verdict: str, is_attack: bool = True) -> str:
    """Return styled HTML badge for attack or control verdict."""
    v = str(verdict).upper()
    if v in ("SUCCESS", "FAIL"):
        # If attack succeeded, that's a security vulnerability (Red)
        bg = "#7F1D1D" if is_attack else "#064E3B"
        border = "#DC2626" if is_attack else "#059669"
        text = "#FCA5A5" if is_attack else "#A7F3D0"
        label = "BREACHED" if is_attack else "FALSE POSITIVE"
    elif v in ("FAILURE", "PASS"):
        # If attack failed, that's a secure defense (Green)
        bg = "#064E3B" if is_attack else "#1E293B"
        border = "#059669" if is_attack else "#475569"
        text = "#A7F3D0" if is_attack else "#CBD5E1"
        label = "BLOCKED (SAFE)" if is_attack else "NORMAL (SAFE)"
    elif v == "UNCERTAIN":
        bg, border, text, label = "#312E81", "#4F46E5", "#C7D2FE", "UNCERTAIN"
    else:
        bg, border, text, label = "#334155", "#64748B", "#E2E8F0", v

    return f"""<span style="background-color: {bg}; border: 1px solid {border}; color: {text};
        padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
        {label}</span>"""


def render_ethics_banner() -> None:
    """Display standard security ethics notice for academic evaluation."""
    st.markdown(
        """
        <div style="background-color: rgba(30, 41, 59, 0.6); border-left: 4px solid #3B82F6;
            padding: 12px 16px; border-radius: 6px; margin-bottom: 20px;">
            <div style="font-weight: 600; color: #93C5FD; font-size: 0.95rem;">
                🛡️ Controlled Security Evaluation Environment
            </div>
            <div style="color: #CBD5E1; font-size: 0.85rem; margin-top: 4px;">
                All adversarial probes are restricted to local test targets and synthetic corpus fixtures
                (reserved <code>.example</code> domains). No unauthorized external attacks are conducted.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Chart Builders ────────────────────────────────────────────────────────────

def build_category_asr_chart(category_asr: Dict[str, float]) -> go.Figure:
    """Build bar chart of Attack Success Rate by OWASP Category."""
    if not category_asr:
        fig = go.Figure()
        fig.add_annotation(text="No category data available", showarrow=False)
        return apply_chart_theme(fig)

    categories = list(category_asr.keys())
    rates = [val * 100.0 for val in category_asr.values()]

    colors = []
    for r in rates:
        if r >= 60:
            colors.append(COLOR_CRITICAL)
        elif r >= 35:
            colors.append(COLOR_HIGH)
        elif r > 0:
            colors.append(COLOR_MEDIUM)
        else:
            colors.append(COLOR_LOW)

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=rates,
                marker_color=colors,
                text=[f"{r:.1f}%" for r in rates],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="<b>Attack Success Rate (ASR %) by OWASP Category</b>",
        xaxis_title="OWASP Category",
        yaxis_title="ASR (%)",
        yaxis_range=[0, max(100, (max(rates) + 10) if rates else 100)],
    )
    return apply_chart_theme(fig)


def build_group_asr_chart(group_asr: Dict[str, float]) -> go.Figure:
    """Build horizontal bar chart of ASR by Attack Group."""
    if not group_asr:
        fig = go.Figure()
        fig.add_annotation(text="No group data available", showarrow=False)
        return apply_chart_theme(fig)

    sorted_groups = sorted(group_asr.items(), key=lambda x: x[1])
    groups = [k for k, v in sorted_groups]
    rates = [v * 100.0 for k, v in sorted_groups]

    fig = go.Figure(
        data=[
            go.Bar(
                y=groups,
                x=rates,
                orientation="h",
                marker_color=COLOR_PRIMARY,
                text=[f"{r:.1f}%" for r in rates],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="<b>Attack Success Rate by Attack Group</b>",
        xaxis_title="ASR (%)",
        yaxis_title="Attack Group",
        xaxis_range=[0, 105],
    )
    return apply_chart_theme(fig)


def build_type_breakdown_chart(df: pd.DataFrame, verdict_mode: str = "both") -> go.Figure:
    """Build stacked or grouped bar chart of exploit outcome by attack type."""
    if df.empty or "type" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No type breakdown available", showarrow=False)
        return apply_chart_theme(fig)

    attack_df = df[~df["is_control"]].copy()
    if attack_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No attack cases found", showarrow=False)
        return apply_chart_theme(fig)

    if verdict_mode == "rule":
        attack_df["is_success"] = attack_df["rule_based_verdict"].isin(["SUCCESS", "FAIL"])
    elif verdict_mode == "judge":
        attack_df["is_success"] = attack_df["judge_verdict"] == "SUCCESS"
    else:
        attack_df["is_success"] = (attack_df["rule_based_verdict"].isin(["SUCCESS", "FAIL"])) & (attack_df["judge_verdict"] == "SUCCESS")

    types = sorted(attack_df["type"].unique())
    succeeded = []
    blocked = []

    for t in types:
        sub = attack_df[attack_df["type"] == t]
        succ_cnt = int(sub["is_success"].sum())
        block_cnt = len(sub) - succ_cnt
        succeeded.append(succ_cnt)
        blocked.append(block_cnt)

    fig = go.Figure(
        data=[
            go.Bar(name="Breached (Exploited)", x=types, y=succeeded, marker_color=COLOR_CRITICAL),
            go.Bar(name="Blocked (Defended)", x=types, y=blocked, marker_color=COLOR_LOW),
        ]
    )
    fig.update_layout(
        barmode="stack",
        title="<b>Exploit Outcome by Attack Type</b>",
        xaxis_title="Attack Type",
        yaxis_title="Probe Count",
    )
    return apply_chart_theme(fig)


def build_position_analysis_chart(df: pd.DataFrame, verdict_mode: str = "both") -> go.Figure:
    """
    Build chart analyzing prompt injection success by position:
    start vs middle vs end, highlighting II-001, PP-001, PP-002.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No position data available", showarrow=False)
        return apply_chart_theme(fig)

    # Focus on indirect and position test cases
    target_ids = {"II-001", "PP-001", "PP-002"}
    sub = df[df["case_id"].isin(target_ids)].copy()

    pos_map = {
        "PP-001": "start (PP-001)",
        "II-001": "middle (II-001)",
        "PP-002": "end (PP-002)",
    }

    if sub.empty:
        # Fall back to general indirect cases if position cases are not present
        sub = df[df["type"] == "indirect"].copy()
        if sub.empty:
            fig = go.Figure()
            fig.add_annotation(text="No indirect injection cases found", showarrow=False)
            return apply_chart_theme(fig)
        sub["pos_label"] = sub["case_id"]
    else:
        sub["pos_label"] = sub["case_id"].map(pos_map).fillna(sub["case_id"])

    if verdict_mode == "rule":
        sub["is_success"] = sub["rule_based_verdict"].isin(["SUCCESS", "FAIL"])
    elif verdict_mode == "judge":
        sub["is_success"] = sub["judge_verdict"] == "SUCCESS"
    else:
        sub["is_success"] = (sub["rule_based_verdict"].isin(["SUCCESS", "FAIL"])) & (sub["judge_verdict"] == "SUCCESS")

    grouped = sub.groupby("pos_label")["is_success"].mean() * 100.0
    labels = list(grouped.index)
    rates = list(grouped.values)

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=rates,
                marker_color=[COLOR_HIGH, COLOR_CRITICAL, COLOR_MEDIUM][: len(labels)],
                text=[f"{r:.1f}%" for r in rates],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="<b>Indirect Injection: Position Sensitivity Analysis</b>",
        xaxis_title="Injection Placement in Document",
        yaxis_title="Success Rate (%)",
        yaxis_range=[0, 105],
    )
    return apply_chart_theme(fig)


def build_model_comparison_chart(
    cat_a: Dict[str, float],
    cat_b: Dict[str, float],
    name_a: str = "Model A",
    name_b: str = "Model B",
) -> go.Figure:
    """Build side-by-side grouped bar chart comparing Category ASR between two models/runs."""
    all_cats = sorted(list(set(cat_a.keys()) | set(cat_b.keys())))
    if not all_cats:
        fig = go.Figure()
        fig.add_annotation(text="No comparison categories available", showarrow=False)
        return apply_chart_theme(fig)

    rates_a = [cat_a.get(c, 0.0) * 100.0 for c in all_cats]
    rates_b = [cat_b.get(c, 0.0) * 100.0 for c in all_cats]

    fig = go.Figure(
        data=[
            go.Bar(name=name_a, x=all_cats, y=rates_a, marker_color="#EF4444"),
            go.Bar(name=name_b, x=all_cats, y=rates_b, marker_color="#10B981"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=f"<b>Category ASR Comparison: {name_a} vs. {name_b}</b>",
        xaxis_title="OWASP Category",
        yaxis_title="ASR (%)",
        yaxis_range=[0, 105],
    )
    return apply_chart_theme(fig)


def build_detector_metrics_chart(
    rule_metrics: Dict[str, Any],
    judge_metrics: Dict[str, Any],
) -> go.Figure:
    """Compare Detector Evaluation Metrics (Rule-Based vs. LLM-as-a-Judge)."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    display_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    rule_vals = [rule_metrics.get(m, 0.0) * 100.0 for m in metrics]
    judge_vals = [judge_metrics.get(m, 0.0) * 100.0 for m in metrics]

    fig = go.Figure(
        data=[
            go.Bar(name="Rule-Based Detector", x=display_names, y=rule_vals, marker_color="#3B82F6"),
            go.Bar(name="LLM-as-a-Judge", x=display_names, y=judge_vals, marker_color="#8B5CF6"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="<b>Detector Benchmark: Rule-Based vs. LLM-as-a-Judge</b>",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 105],
    )
    return apply_chart_theme(fig)
