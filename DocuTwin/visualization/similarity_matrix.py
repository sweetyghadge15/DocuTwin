import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_similarity_heatmap(
    similarity_df: pd.DataFrame,
    title: str = "Document Similarity Matrix",
    colorscale: str = "Viridis"
) -> go.Figure:
    """
    Generates an interactive Plotly heatmap for a similarity matrix DataFrame.

    Args:
        similarity_df: Square DataFrame indexed and columned by document names.
        title: Chart title string.
        colorscale: Plotly colorscale string name.

    Returns:
        Plotly Figure object.
    """
    if similarity_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No matrix data available")
        return fig

    labels = list(similarity_df.columns)
    z_values = similarity_df.values * 100.0  # Convert to percentages for display

    # Text annotations inside matrix cells
    text_values = []
    for row in z_values:
        text_row = [f"{val:.1f}%" for val in row]
        text_values.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=labels,
        y=labels,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        hoverongaps=False,
        colorscale=colorscale,
        zmin=0,
        zmax=100,
        colorbar=dict(title="Similarity %", ticksuffix="%")
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="sans-serif")),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        height=550,
        margin=dict(l=50, r=50, t=60, b=80),
        template="plotly_white"
    )

    return fig
