import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid", palette="Set3")

from dash import Dash, dash_table
from dash.dash_table.Format import Format, Scheme

from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyshiftlearn.config import MEDIA_PATH


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
    plt.savefig(MEDIA_PATH / f"{filename}.png")

    return ax


def plot_metrics(metrics_dict: dict[str, dict[str, float]], filename: str = None) -> plt.Axes:
    """
    Generates a plot of the metrics values for each dataset after training and saves it as an image.

    Parameters:
        metrics_dict (dict[str, dict[str, float]]):
            A dictionary mapping the names of the metrics to their respective values.
        filename (str):
            The name of the file to save the plot image as.

    Returns:
        plt.Axes: The axes object representing the plot.
    """

    data = pd.DataFrame(metrics_dict).transpose()
    ax = data.plot(kind="bar", figsize=(16, 7), rot=0)
    ax.set_title(f"Metrics by Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    plt.xticks(rotation=10)
    if filename is not None:
        plt.savefig(MEDIA_PATH / f"{filename}.png")
    return ax


def plot_proportion(labels: Sequence[str], dataset_name: str, true_values: Sequence[float], filename: str = None,
                    pred_values: Sequence[float] = None,
                    split: bool = True, **kwargs) -> go.Figure:
    """
    Generates a pie chart of the proportion of each label in the dataset.

    Parameters:
        labels (Sequence[str]):
            A list of the labels in the dataset.
        dataset_name (str):
            The name of the dataset.
        true_values (Sequence[float]):
            A list of the true label proportions in the dataset.
        filename (str):
            The name of the file to save the plot image as.
        pred_values (Sequence[float]):
            A list of the predicted label proportions in the dataset.
        split (bool):
            Whether the plot is for the training or validation set.
        **kwargs:
            Additional keyword arguments to pass to the `make_subplots` function.

    Returns:
        go.Figure: The figure object representing the plot.
    """
    if split:
        # Create subplots: use 'domain' type for Pie subplot
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]], **kwargs)
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
    else:
        fig = go.Figure(data=[go.Pie(labels=labels, values=true_values)])
        fig.update_layout(title_text=f"Label proportions in {dataset_name}")

    if filename is not None:
        fig.write_image(MEDIA_PATH / f"{filename}.png")

    return fig


def plot_dash_table(
    result: pd.DataFrame, method_names: Sequence[str], dataset_names: Sequence[str], summary_names: Sequence[str]
) -> Dash:
    """
    Plot a dash table of the results

    Parameters:
        result (pd.DataFrame):
            The result dataframe
        method_names (Sequence[str]):
            The names of the methods
        dataset_names (Sequence[str]):
            The names of the datasets
        summary_names (Sequence[str]):
            The names of the summaries

    Returns:
        Dash: The dash app
    """
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

    return app
