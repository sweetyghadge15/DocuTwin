import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any

def build_document_dna_graph(
    pairs_df: pd.DataFrame,
    all_doc_names: List[str],
    threshold: float = 0.55
) -> go.Figure:
    """
    Creates an interactive Network-style Plotly relationship graph ('Document DNA').

    Args:
        pairs_df: DataFrame of pairwise document similarities with 'document_a', 'document_b', 'hybrid_similarity'.
        all_doc_names: List of all uploaded document names.
        threshold: Minimum hybrid similarity threshold for drawing connecting edges.

    Returns:
        Plotly Figure graph object.
    """
    if not all_doc_names:
        fig = go.Figure()
        fig.update_layout(title="No document nodes available")
        return fig

    N = len(all_doc_names)
    
    # Position nodes deterministically in a circular layout
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    node_positions = {
        name: (float(radius * np.cos(a)), float(radius * np.sin(a)))
        for name, a in zip(all_doc_names, angles)
    }

    # Filter pairs matching threshold
    valid_edges = []
    if not pairs_df.empty:
        for _, row in pairs_df.iterrows():
            sim = float(row.get("hybrid_similarity", row.get("similarity", 0.0)))
            if sim >= threshold:
                valid_edges.append({
                    "u": row["document_a"],
                    "v": row["document_b"],
                    "weight": sim
                })

    # Build Edge Scatter Traces
    edge_x = []
    edge_y = []
    edge_hover = []
    middle_node_x = []
    middle_node_y = []
    middle_node_text = []

    for edge in valid_edges:
        u, v, w = edge["u"], edge["v"], edge["weight"]
        if u in node_positions and v in node_positions:
            x0, y0 = node_positions[u]
            x1, y1 = node_positions[v]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Label on midpoint of edge
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            middle_node_x.append(mx)
            middle_node_y.append(my)
            middle_node_text.append(f"{w:.1%}")

    fig = go.Figure()

    # Draw edge lines
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2.5, color="#6366F1"),
        hoverinfo="none",
        showlegend=False
    ))

    # Draw edge percentage text markers
    if middle_node_x:
        fig.add_trace(go.Scatter(
            x=middle_node_x,
            y=middle_node_y,
            mode="markers+text",
            text=middle_node_text,
            textposition="middle center",
            hoverinfo="none",
            marker=dict(size=28, color="#EEF2FF", line=dict(width=1.5, color="#4F46E5")),
            textfont=dict(size=10, color="#3730A3", family="sans-serif"),
            showlegend=False
        ))

    # Compute node sizes based on degree / average similarity
    node_x = [node_positions[n][0] for n in all_doc_names]
    node_y = [node_positions[n][1] for n in all_doc_names]
    
    node_hover = []
    for name in all_doc_names:
        related_count = sum(1 for e in valid_edges if e["u"] == name or e["v"] == name)
        node_hover.append(f"<b>{name}</b><br>Active Connections: {related_count}")

    # Draw node markers
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[name if len(name) < 25 else name[:22] + "..." for name in all_doc_names],
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=36,
            color="#4F46E5",
            line=dict(width=3, color="#818CF8")
        ),
        textfont=dict(size=12, color="#1E293B", family="sans-serif"),
        showlegend=False
    ))

    fig.update_layout(
        title=dict(
            text=f"Document DNA — Relationship Graph (Threshold ≥ {threshold:.0%})",
            font=dict(size=18, family="sans-serif")
        ),
        showlegend=False,
        height=580,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(visible=False, zeroline=False),
        yaxis=dict(visible=False, zeroline=False),
        template="plotly_white",
        plot_bgcolor="rgba(248,250,252,1)"
    )

    return fig
