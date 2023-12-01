from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid", palette="Set3")

from dash import Dash, dash_table
from dash.dash_table.Format import Format, Scheme

from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_loss(loss_dict: dict[str, list[float]], split: str, filename: str) -> plt.Axes:
    """
    Generates a plot of the loss values over epochs and saves it as an image.

    Parameters:
        loss_dict (dict[str, list[float]]):
            A dictionary mapping the names of the loss functions to their respective loss values.
        split (str):
            The name of the split (e.g., 'training', 'validation') for which the loss is being plotted.
        filename (str):
            The name of the file to save the plot image as.

    Returns:
        plt.Axes: The axes object representing the plot.

    """
    # Plot the loss
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(data=loss_dict, legend="auto", ax=ax)

    ax.set_title(f"Loss / Epoch in {split}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")

    # Save the plot image
    plt.savefig(f"{Path.cwd().parent.parent}/images/{filename}.png")

    return ax


# Plot the metrics results by dataset with bar groups
def plot_metrics(metrics_dict: dict[str, dict[str, float]], filename: str) -> plt.Axes:
    data = pd.DataFrame(metrics_dict).transpose()
    ax = data.plot(kind="bar", figsize=(16, 7), rot=0)
    ax.set_title(f"Metrics by Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    plt.xticks(rotation=10)
    plt.savefig(f"{Path.cwd().parent.parent}/images/{filename}.png")
    return ax


def plot_proportion(labels: Sequence[str], dataset_name: str, true_values: Sequence[float],
                    pred_values: Sequence[float] = None,
                    split: bool = True):
    if split:
        # Create subplots: use 'domain' type for Pie subplot
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=true_values, name="True"),
                      1, 1)
        fig.add_trace(go.Pie(labels=labels, values=pred_values, name="Pred"),
                      1, 2)

        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.4, hoverinfo="label+percent+name")

        fig.update_layout(
            title_text=f"Predicted label proportions vs true label proportions in {dataset_name}",
            # Add annotations in the center of the donut pies.
            annotations=[dict(text='True', x=0.18, y=0.5, font_size=20, showarrow=False),
                         dict(text='Pred', x=0.82, y=0.5, font_size=20, showarrow=False)])
        fig.show()
    else:
        fig = go.Figure(data=[go.Pie(labels=labels, values=true_values)])
        fig.update_layout(title_text=f"Label proportions in {dataset_name}")
        fig.show()
    pass


def plot_dash_table(result: pd.DataFrame, method_names: list[str], dataset_names: list[str], summary_names: list[str]):
    app = Dash(__name__)

    method_columns = [
        {
            "name": ["Method", method_name],
            "id": method_name,
            "type": "text",
        }
        for method_name in method_names
    ]

    dataset_columns = [
        {
            "name": ["Dataset", dataset_name],
            "id": dataset_name,
            "type": "numeric",
            "format": Format(
                scheme=Scheme.fixed,
                precision=4,
                nully="NA",
            ),
        }
        for dataset_name in dataset_names
    ]

    summary_columns = [
        {
            "name": ["Summary", summary_name],
            "id": summary_name,
            "type": "numeric",
            "format": Format(
                scheme=Scheme.fixed,
                precision=4,
                nully="NA",
            ),
        }
        for summary_name in summary_names
    ]

    # A list of schemas for columns of the table
    columns = [
        *method_columns,
        *dataset_columns,
        *summary_columns,
    ]

    highlight_top_three_cells_in_column = [
        {
            "if": {
                "filter_query": "{{{}}}={}".format(dataset_name, _),
                "column_id": dataset_name,
            },
            "backgroundColor": "lightblue",
        }
        for dataset_name in dataset_names
        for _ in result[dataset_name].nlargest(4)
    ]

    app.layout = dash_table.DataTable(  # Customize the layout
        columns=columns,
        data=result.to_dict("records"),
        merge_duplicate_headers=True,  # Merge duplicate headers
        # style_as_list_view=True,  # Use list view
        style_cell={
            "textAlign": "center",  # Align text to center
        },
        style_data_conditional=(
            [
                {
                    "if": {"row_index": "odd"},  # Even rows are in a different color
                    "backgroundColor": "rgb(248, 248, 248)",
                },
            ]
            + highlight_top_three_cells_in_column
            + [
                {
                    "if": {
                        "filter_query": "{{Average Rank}}={}".format(
                            _
                        ),  # Highlight the top three cells in "Average_rank" column
                        "column_id": ["Average Rank", "Sampler"],
                    },
                    "backgroundColor": "lightblue",
                }
                for _ in result["Average Rank"].nsmallest(4)
            ]
        ),
        style_header={
            "fontWight": "bold",
            "backgroundColor": "#F5F5F5",
        },
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        row_selectable="multi",
        column_selectable="multi",
    )

    # Run the Dash App
    app.run()
