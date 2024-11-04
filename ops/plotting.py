"""
Data Visualization and Plotting Utilities
This module provides a comprehensive set of functions for data visualization and plotting
(relating to step 5 -- clustering). It includes functions for:

1. Volcano Plots: Visualizing significance and effect size in differential expression analyses.
2. Scatter Plots: Creating customizable scatter plots for comparing two features.
3. Dimensionality Reduction Plots: Visualizing high-dimensional data in 2D space.
4. Heatmaps: Generating clustered heatmaps with various annotation options.

"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt  # Warning: need matplotlib >= 3.3.3 for symlog to work properly
from matplotlib.ticker import SymmetricalLogLocator, FixedLocator
import pandas as pd
from ops.io import GLASBEY
import random
from itertools import combinations
from functools import partial
from adjustText import adjust_text

# Set default plotting parameters
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
# plt.rcParams['mathtext.rm'] = 'Helvetica'
# plt.rcParams['mathtext.it'] = 'Helvetica:italic'
# plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
# plt.rcParams['mathtext.sf'] = 'Helvetica'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['font.size'] = 8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def plot_cell_histogram(
        df, 
        cutoff, 
        bins=50, 
        figsize=(12, 6),
        save_plot_path=None
    ):
    """
    Plot a histogram of cell numbers with a vertical cutoff line and return genes below the cutoff.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'cell_number' and 'gene_symbol_0' columns
    cutoff : float
        Vertical line position and threshold for identifying genes
    bins : int, optional
        Number of bins for histogram (default: 50)
    figsize : tuple, optional
        Figure size as (width, height) (default: (12, 6))
    save_plot_path : str, optional
        Path to save the plot as an image (default: None)
        
    Returns:
    --------
    None
    """
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Plot histogram using seaborn for better styling
    sns.histplot(data=df, x='cell_number', bins=bins, color='skyblue', alpha=0.6)
    
    # Add vertical line at cutoff
    plt.axvline(x=cutoff, color='red', linestyle='--', label=f'Cutoff: {cutoff}')
    
    # Customize the plot
    plt.title('Distribution of Cell Numbers', fontsize=12, pad=15)
    plt.xlabel('Cell Number', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.legend()
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Get genes below cutoff
    genes_below_cutoff = df[df['cell_number'] <= cutoff]['gene_symbol_0'].tolist()
    
    # Show the plot
    plt.show()

    # Save the plot if path is provided
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    
    # rint genes below cutoff
    print(f'Number of genes below cutoff: {len(genes_below_cutoff)}')
    print(genes_below_cutoff)

def volcano(
    df,
    feature=None,
    x="score",
    y="pval",
    alpha_level=0.05,
    change=0,
    prelog=True,
    xscale=None,
    control_query=None,
    annotate_query=None,
    annotate_labels=False,
    high_color='green',
    low_color='magenta',
    default_color='gray',
    control_color=sns.color_palette()[1],
    threshold_kwargs=dict(),
    annotate_kwargs=dict(),
    ax=None,
    rasterized=True,
    adjust_labels=True,
    **kwargs
):
    """
    Create a volcano plot to visualize significance and effect size.

    Parameters:
        df (pd.DataFrame): DataFrame with the data to plot.
        feature (str, optional): Prefix to use for x and y columns.
        x (str): Column name for x-axis data.
        y (str): Column name for y-axis data (p-values).
        alpha_level (float): Threshold for significance.
        change (float or list): Effect size thresholds for highlighting.
        prelog (bool): If True, transform p-values using -log10.
        xscale (str, optional): Scale for x-axis ("symlog" or None).
        control_query (str, optional): Query to subset control data.
        annotate_query (str, optional): Query to subset data for annotation.
        annotate_labels (bool or str): Column name for annotation labels.
        high_color (str): Color for high significance points.
        low_color (str): Color for low significance points.
        default_color (str): Color for non-significant points.
        control_color (str): Color for control points.
        threshold_kwargs (dict): Additional arguments for threshold lines.
        annotate_kwargs (dict): Additional arguments for annotated points.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        rasterized (bool): If True, use rasterized rendering.
        adjust_labels (bool): If True, adjust label positions to avoid overlap.

    Returns:
        matplotlib.axes.Axes: The Axes object with the plot.
    """
    df_ = df.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Update x and y if a feature prefix is provided
    if feature is not None:
        x = f"{feature}_{x}".strip("_")
        y = f"{feature}_{y}".strip("_")

    # Set alpha level to minimum y-value if not specified
    if alpha_level is None:
        alpha_level = df_[y].min() - 0.1

    # Convert change to a list of two values if it's a single value
    if not isinstance(change, list):
        try:
            change = [-change, change]
        except:
            change = [change, change]

    # Set default threshold line properties
    _ = threshold_kwargs.setdefault('color','gray')
    _ = threshold_kwargs.setdefault('linestyle','--')

    # Handle change values and draw threshold lines
    if change == [None, None]:
        change = [df_[x].min() - 1] * 2
    else:
        if change[0] is not None:
            ax.axvline(change[0], **threshold_kwargs)
        else:
            change[0] = df_[x].min() - 1

        if change[1] is not None:
            ax.axvline(change[1], **threshold_kwargs)
        else:
            change[1] = df_[x].max() + 1

    # Mark significant points
    df_["significant"] = False
    df_.loc[(df_[x] < change[0]) & (df_[y] < alpha_level), "significant"] = "low"
    df_.loc[(df_[x] > change[1]) & (df_[y] < alpha_level), "significant"] = "high"

    # Apply -log10 transformation if prelog is True
    if prelog:
        df_[y] = -np.log10(df_[y])
        alpha_level = -np.log10(alpha_level)

    # Apply control query if provided
    if control_query is not None:
        df_.loc[df_.index.isin(df_.query(control_query).index), "significant"] = "control"

    # Apply annotation query if provided
    if annotate_query is not None:
        df_annotate = df_.query(annotate_query)

    # Create scatter plot with significant points highlighted
    sns.scatterplot(
        data=df_,
        x=x,
        y=y,
        hue="significant",
        hue_order=["high", False, "low", "control"],
        palette=[high_color, default_color, low_color, control_color],
        legend=None,
        ax=ax,
        rasterized=rasterized,
        **kwargs
    )

    # Annotate specific points if annotate_query is provided
    if annotate_query is not None:
        kwargs_ = kwargs.copy()
        kwargs_.update(annotate_kwargs)
        hue = kwargs_.pop('hue', 'significant')
        if hue == "significant":
            hue_order = kwargs_.pop('hue_order', ['high', False, 'low', 'control'])
            palette = kwargs_.pop('palette', [high_color, default_color, low_color, control_color])
        else:
            hue_order = kwargs_.pop('hue_order', None)
            palette = kwargs_.pop('palette', None)
        sns.scatterplot(
            data=df_annotate,
            x=x,
            y=y,
            ax=ax,
            hue=hue,
            hue_order=hue_order,
            palette=palette,
            legend=None,
            rasterized=rasterized,
            **kwargs_,
        )
        if annotate_labels:
            labels = []
            for _, entry in df_annotate.iterrows():
                labels.append(ax.annotate(entry[annotate_labels], (entry[x], entry[y]),
                    arrowprops=dict(arrowstyle='-', relpos=(0, 0), shrinkA=0, shrinkB=0)))

    # Set y-axis scale and labels if not using prelog
    if not prelog:
        ax.set_yscale("log", basey=10)
        y_max = np.floor(np.log10(df_[y].min()))
        ax.set_ylim([1.15, 10 ** y_max])
        ax.set_yticks(np.logspace(y_max, 0, -(int(y_max) - 1)))
        ax.set_yticklabels(
            labels=[
                f"$\\mathdefault{{10^{{{int(n)}}}}}$"
                for n in np.linspace(y_max, 0, -(int(y_max) - 1))
            ]
        )

    # Apply symlog scale if specified
    if xscale == "symlog":
        ax = symlog_axis(df_[x], ax, 'x')

    # Draw significance line
    ax.axhline(alpha_level, **threshold_kwargs)

    # Set axis labels
    ax.set_xlabel(" ".join(x.split("_")))
    if prelog:
        ax.set_ylabel("-log10(p-value)")
    else:
        ax.set_ylabel("p-value")

    # Adjust label positions if specified
    if adjust_labels:
        try:
            adjust_text(labels, df_[x].values, df_[y].values, ax=ax, force_text=(0.1, 0.05), force_points=(0.01, 0.025))
        except:
            pass

    return ax

def two_feature(
    df,
    x,
    y,
    annotate_query=None,
    annotate_labels=False,
    annotate_kwargs=dict(edgecolor='black'),
    xscale=None,
    yscale=None,
    control_query=None,
    control_kwargs=dict(),
    ax=None,
    rasterized=True,
    adjust_labels=True,
    save_plot_path=None, 
    **kwargs
):
    """
    Create a scatter plot comparing two features.

    Parameters:
        df (pd.DataFrame): DataFrame with the data to plot.
        x (str): Column name for x-axis data.
        y (str): Column name for y-axis data.
        annotate_query (str, optional): Query to subset data for annotation.
        annotate_labels (bool or str): Column name for annotation labels.
        annotate_kwargs (dict): Additional arguments for annotated points.
        xscale (str, optional): Scale for x-axis ("symlog" or None).
        yscale (str, optional): Scale for y-axis ("symlog" or None).
        control_query (str, optional): Query to subset control data.
        control_kwargs (dict): Additional arguments for control points.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        rasterized (bool): If True, use rasterized rendering.
        adjust_labels (bool): If True, adjust label positions to avoid overlap.
        save_plot_path (str, optional): Path to save the plot as an image.

    Returns:
        matplotlib.axes.Axes: The Axes object with the plot.
    """
    df_ = df.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if annotate_query is not None:
        df_annotate = df_.query(annotate_query)

    if control_query is not None:
        df_control = df_.query(control_query)
        df_ = df_[~(df_.index.isin(df_control.index))]

    sns.scatterplot(
        data=df_,
        x=x,
        y=y,
        ax=ax,
        rasterized=rasterized,
        **kwargs
    )

    if control_query is not None:
        _ = control_kwargs.setdefault('color', sns.color_palette()[1])
        kwargs_ = kwargs.copy()
        kwargs_.update(control_kwargs)
        sns.scatterplot(
            data=df_control,
            x=x,
            y=y,
            ax=ax,
            rasterized=rasterized,
            **kwargs_,
        )

    if annotate_query is not None:
        kwargs_ = kwargs.copy()
        _ = annotate_kwargs.setdefault('edgecolor', 'black')
        _ = annotate_kwargs.setdefault('alpha', 1)
        kwargs_.update(annotate_kwargs)
        sns.scatterplot(
            data=df_annotate,
            x=x,
            y=y,
            ax=ax,
            rasterized=rasterized,
            **kwargs_,
        )
        if annotate_labels:
            labels = []
            for _, entry in df_annotate.iterrows():
                labels.append(ax.annotate(entry[annotate_labels], (entry[x], entry[y]), 
                    arrowprops=dict(arrowstyle='-', relpos=(0, 0), shrinkA=0, shrinkB=0)))

    # Apply symlog scale if specified
    if xscale == "symlog":
        ax = symlog_axis(df_[x], ax, 'x')

    if yscale == "symlog":
        ax = symlog_axis(df_[y], ax, 'y')

    ax.set_xlabel(" ".join(x.split("_")))
    ax.set_ylabel(" ".join(y.split("_")))

    if adjust_labels:
        try:
            adjust_text(labels, df_[x].values, df_[y].values, ax=ax, force_text=(0.1, 0.05), force_points=(0.01, 0.025))
        except:
            pass
    
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')

    return ax

def dimensionality_reduction(
    df,
    x="X",
    y="Y",
    default_kwargs={'color': 'lightgray', 'alpha': 0.5},
    control_query='gene_id=="-1"',
    control_color="black",
    control_legend=True,
    control_kwargs=dict(),
    label_query=None,
    label_hue="cluster",
    label_as_cmap=False,
    label_palette="glasbey",
    label_kwargs=dict(),
    randomize_palette=False,
    label_legend=False,
    legend_kwargs=dict(),
    hide_axes=False,
    ax=None,
    rasterized=True,
    save_plot_path=None,
    **kwargs,
):
    """
    Create a scatter plot for dimensionality reduction results.

    Parameters:
        df (pd.DataFrame): DataFrame with the data to plot.
        x (str): Column name for x-axis data.
        y (str): Column name for y-axis data.
        default_kwargs (dict): Default arguments for the scatter plot.
        control_query (str): Query to subset control data.
        control_color (str): Color for control points.
        control_legend (bool or str): If True, include legend for control data.
        control_kwargs (dict): Additional arguments for control points.
        label_query (str, optional): Query to subset data for labels.
        label_hue (str): Column name for label color grouping.
        label_as_cmap (bool): If True, use a color map for labels.
        label_palette (str or list): Palette for label colors.
        label_kwargs (dict): Additional arguments for labeled points.
        randomize_palette (bool or int): Randomize color palette if True.
        label_legend (bool): If True, include legend for labels.
        legend_kwargs (dict): Additional arguments for the legend.
        hide_axes (bool): If True, hide the axes.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        rasterized (bool): If True, use rasterized rendering.
        save_plot_path (str, optional): Path to save the plot as an image.

    Returns:
        matplotlib.axes.Axes: The Axes object with the plot.
    """
    df_ = df.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if control_query is not None:
        df_control = df_.query(control_query)
        df_ = df_[~(df_.index.isin(df_control.index))]

    if label_query is not None:
        df_label = df_.query(label_query)
        df_ = df_[~(df_.index.isin(df_label.index))]

    sns.scatterplot(data=df_, x=x, y=y, **default_kwargs, **kwargs, ax=ax, rasterized=rasterized)

    if control_query is not None:
        if "legend" not in control_kwargs:
            if isinstance(control_legend, str):
                control_kwargs["label"] = control_legend
            elif control_legend:
                control_kwargs["label"] = "control"
            else:
                control_kwargs["legend"] = False

        kwargs_ = kwargs.copy()
        kwargs_.update(control_kwargs)

        sns.scatterplot(
            data=df_control,
            x=x,
            y=y,
            color=control_color,
            alpha=0.75,
            **kwargs_,
            ax=ax,
            rasterized=rasterized
        )

    if label_query is not None:
        n_colors = 1
        if label_hue is not None:
            n_colors = df_label[label_hue].nunique()

            if label_palette == "glasbey":
                palette = sns.color_palette(
                    ((GLASBEY.reshape(3, 256).T) / 256)[1:], n_colors=n_colors
                )
            else:
                palette = sns.color_palette(
                    label_palette, n_colors=n_colors, as_cmap=label_as_cmap
                )

            if randomize_palette:
                random.seed(int(randomize_palette))
                random.shuffle(palette)
        else:
            palette = None

        kwargs_ = kwargs.copy()
        kwargs_.update(label_kwargs)

        sns.scatterplot(
            data=df_label,
            x=x,
            y=y,
            hue=label_hue,
            palette=palette,
            legend=label_legend,
            **kwargs_,
            ax=ax,
            rasterized=rasterized
        )

        if label_legend:
            loc = legend_kwargs.pop('loc', (1.05, 0.33))
            if label_as_cmap:
                hue_norm = kwargs.get(
                    'hue_norm',
                    (df_label[label_hue].astype(float).min(), df_label[label_hue].astype(float).max())
                )
                s = kwargs.get('s', 10)
                hdl, _ = ax.get_legend_handles_labels()
                legend_colors = sns.color_palette(label_palette, as_cmap=True)(np.linspace(0, 255, 5, dtype=int))
                legend_color_vals = np.linspace(*hue_norm, 5)
                legend_color_header = hdl[0]
                legend_elements = [
                    plt.scatter([], [], marker='o', s=s, color=c,
                        linewidth=0.5, edgecolor='k', label=str(cl))
                    for c, cl in zip(legend_colors, legend_color_vals)
                ]
                ax.legend(handles=legend_elements, loc=loc, ncol=1, **legend_kwargs)
            else:
                n_cols = max(1, (n_colors // 20))
                ax.legend(loc=loc, ncol=n_cols, **legend_kwargs)

    if hide_axes:
        ax.axis("off")

    if save_plot_path:
        ax.figure.savefig(save_plot_path, dpi=300, bbox_inches='tight')

    return ax


def heatmap(
    df,
    figsize=None,
    row_colors=None,
    col_colors=None,
    row_palette='Set2',
    col_palette='Set2',
    label_fontsize=5,
    rasterized=True,
    colors_ratio=0.1,
    spinewidth=0.25,
    alternate_ticks=(True, True),
    alternate_tick_length=(30, 30),
    label_every=(1, 1),
    xticklabel_kwargs=dict(),
    yticklabel_kwargs=dict(),
    xticks_emphasis=[],
    yticks_emphasis=[],
    save_plot_path=None,
    **kwargs
):
    """Generates a heatmap with optional clustering and color annotations.
    
    Note:
    Weird things happen if you make the heatmap aspect ratio too big/small
    (e.g., figsize=(1.3,6) looks about the same as (2.7,6)).
    
    Parameters:
    - df (pd.DataFrame): Data for the heatmap.
    - figsize (tuple): Figure size (width, height). If None, size is auto-calculated.
    - row_colors (str or pd.Series): Column name or Series for row color annotations. If str, it's a column name; if pd.Series, it should contain color values.
    - col_colors (str or pd.Series): Index name or Series for column color annotations. If str, it's an index name; if pd.Series, it should contain color values.
    - row_palette (str): Palette name for row colors. Defaults to 'Set2'.
    - col_palette (str): Palette name for column colors. Defaults to 'Set2'.
    - label_fontsize (int): Font size for axis labels. Defaults to 5.
    - rasterized (bool): Whether to rasterize the heatmap. Defaults to True.
    - colors_ratio (float): Ratio for the color bar size. Defaults to 0.1.
    - spinewidth (float): Width of the spines around the heatmap. Defaults to 0.25.
    - alternate_ticks (tuple): Whether to alternate major and minor ticks for x and y axes. Defaults to (True, True).
    - alternate_tick_length (tuple): Length of major and minor ticks. Defaults to (30, 30).
    - label_every (tuple): Frequency of label display for x and y axes. Defaults to (1, 1).
    - xticklabel_kwargs (dict): Additional keyword arguments for x-axis tick labels.
    - yticklabel_kwargs (dict): Additional keyword arguments for y-axis tick labels.
    - xticks_emphasis (list): List of x-axis tick labels to emphasize.
    - yticks_emphasis (list): List of y-axis tick labels to emphasize.
    - save_plot_path (str): Path to save the plot as an image.
    - **kwargs: Additional arguments for seaborn’s `clustermap`.

    Returns:
    - sns.matrix.ClusterGrid: The clustermap object with the heatmap.
    """
    
    # Extract parameters from kwargs
    vmin = kwargs.pop('vmin', -5)
    vmax = kwargs.pop('vmax', 5)
    col_cluster = kwargs.pop('col_cluster', False)
    row_cluster = kwargs.pop('row_cluster', False)
    cmap = kwargs.pop('cmap', 'vlag')
    cbar_pos = kwargs.pop('cbar_pos', None)
    
    # Check and set dendrogram ratio
    if col_cluster or row_cluster:
        dendrogram_ratio = kwargs.pop('dendrogram_ratio', 0.1)
        if dendrogram_ratio == 0:
            raise ValueError('dendrogram_ratio must be greater than zero if clustering rows or columns.')
    else:
        dendrogram_ratio = kwargs.pop('dendrogram_ratio', 0)
    
    # Process row colors if provided
    if isinstance(row_colors, str):
        # Map row color column to colors using specified palette
        row_color_map = {group: color 
                         for group, color 
                         in zip(df[row_colors].unique(), sns.color_palette(row_palette, n_colors=df[row_colors].nunique()))}
        row_colors = df[row_colors].map(row_color_map)
        row_divisions = row_colors.value_counts()[row_colors.unique()].cumsum().values[:-1]
    
    # Process column colors if provided
    if isinstance(col_colors, str):
        # Map column color index to colors using specified palette
        col_color_map = {group: color 
                         for group, color 
                         in zip(df.loc[col_colors].unique(), sns.color_palette(col_palette, n_colors=df.loc[col_colors].nunique()))}
        col_colors = df.loc[col_colors].map(col_color_map)
        col_divisions = col_colors.value_counts()[col_colors.unique()].cumsum().values[:-1]

    # Remove color columns from dataframe
    if row_colors is not None:
        df = df[[col for col in df.columns if col != row_colors.name]]
    if col_colors is not None:
        df = df.loc[[index != col_colors.name for index in df.index]]

    # Calculate figure size if not provided
    y_len, x_len = df.shape
    if figsize is None:
        figsize = np.array([x_len, y_len]) * 0.08
    
    # Create clustermap
    cg = sns.clustermap(
        df,
        figsize=figsize,
        row_colors=row_colors,
        col_colors=col_colors,
        vmin=vmin,
        vmax=vmax,
        col_cluster=col_cluster,
        row_cluster=row_cluster,
        cmap=cmap,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=colors_ratio,
        rasterized=rasterized,
        cbar_pos=cbar_pos,
        **kwargs
    )

    # Update layout to remove space between subplots
    cg.gs.update(hspace=0, wspace=0)
    
    # Remove color axis ticks and add dividers if clustering is not applied
    if row_colors is not None:
        cg.ax_row_colors.set_xticks([])
        if not row_cluster:
            [cg.ax_heatmap.axhline(d, color='black', linestyle='--', linewidth=0.25) for d in row_divisions]
    if col_colors is not None:
        cg.ax_col_colors.set_yticks([])
        if not col_cluster:
            [cg.ax_heatmap.axvline(d, color='black', linestyle='--', linewidth=0.25) for d in col_divisions]

    # Remove axis labels
    cg.ax_heatmap.set_ylabel(None)
    cg.ax_heatmap.set_xlabel(None)

    # Add spines with specified width
    for _, spine in cg.ax_heatmap.spines.items():
        spine.set_visible(True)
        spine.set_lw(spinewidth)
    
    # Set up tick labels and ticks with alternating offsets
    x_le, y_le = label_every
    x_at, y_at = alternate_ticks
    x_atl, y_atl = alternate_tick_length
    ytickrotation = yticklabel_kwargs.pop('rotation', 'horizontal')
    xtickrotation = xticklabel_kwargs.pop('rotation', 'vertical')

    if col_cluster:
        x_labels = df.columns.get_level_values(0)[cg.dendrogram_col.reordered_ind]
    else:
        x_labels = df.columns.get_level_values(0)
    if row_cluster:
        y_labels = df.index.get_level_values(0)[cg.dendrogram_row.reordered_ind]
    else:
        y_labels = df.index.get_level_values(0)

    # Set major and minor ticks with alternating labels
    cg.ax_heatmap.xaxis.set_major_locator(FixedLocator(np.linspace(0.5, x_len - (x_len - 0.5) % (x_le * 2), int(np.ceil((x_len - 0.5) / (2 * x_le))))))
    cg.ax_heatmap.xaxis.set_minor_locator(FixedLocator(np.linspace(0.5 + x_le, x_len - (x_len - 0.5 - x_le) % (x_le * 2), int(np.ceil((x_len - 0.5 - x_le) / (2 * x_le))))))
    cg.ax_heatmap.yaxis.set_major_locator(FixedLocator(np.linspace(0.5, y_len - (y_len - 0.5) % (y_le * 2), int(np.ceil((y_len - 0.5) / (2 * y_le))))))
    cg.ax_heatmap.yaxis.set_minor_locator(FixedLocator(np.linspace(0.5 + y_le, y_len - (y_len - 0.5 - y_le) % (y_le * 2), int(np.ceil((y_len - 0.5 - y_le) / (2 * y_le))))))
    _ = cg.ax_heatmap.set_yticklabels(y_labels[::2 * y_le], fontsize=label_fontsize, rotation=ytickrotation, rotation_mode='anchor', **yticklabel_kwargs)
    _ = cg.ax_heatmap.set_yticklabels(y_labels[y_le::2 * y_le], minor=True, fontsize=label_fontsize, rotation=ytickrotation, rotation_mode='anchor', **yticklabel_kwargs)
    _ = cg.ax_heatmap.set_xticklabels(x_labels[::2 * x_le], fontsize=label_fontsize, rotation=xtickrotation, rotation_mode='anchor', **xticklabel_kwargs)
    _ = cg.ax_heatmap.set_xticklabels(x_labels[x_le::2 * x_le], minor=True, fontsize=label_fontsize, rotation=xtickrotation, rotation_mode='anchor', **xticklabel_kwargs)

    # Set tick parameters for both major and minor ticks
    if y_at:
        cg.ax_heatmap.tick_params(axis='y', which='major', pad=2, length=2)
        cg.ax_heatmap.tick_params(axis='y', which='minor', pad=2, length=y_atl)
    else:
        cg.ax_heatmap.tick_params(axis='y', which='both', pad=2, length=2)
    if x_at:
        cg.ax_heatmap.tick_params(axis='x', which='major', pad=-2, length=2)
        cg.ax_heatmap.tick_params(axis='x', which='minor', pad=-2, length=x_atl)
    else:
        cg.ax_heatmap.tick_params(axis='x', which='both', pad=-2, length=2)

    # Emphasize specific ticks if provided
    if len(xticks_emphasis) > 0:
        [tick.set_visible(False) for tick in cg.ax_heatmap.get_xticklabels(which='both') if tick.get_text() in xticks_emphasis]
    if len(yticks_emphasis) > 0:
        [tick.set_color('red') for tick in cg.ax_heatmap.get_yticklabels(which='both') if tick.get_text() in yticks_emphasis]

    if save_plot_path:
        cg.savefig(save_plot_path, dpi=150, bbox_inches='tight')

    return cg

def boxplot_jitter(jitter_ax="x", jitter_range=(-0.25, 0.25), *args, **kwargs):
    """Creates a boxplot with jittered data points.

    Parameters:
    - jitter_ax (str): Axis to apply jitter ("x" or "y"). Defaults to "x".
    - jitter_range (tuple): Range of jitter for data points. Defaults to (-0.25, 0.25).
    - *args, **kwargs: Additional arguments for seaborn’s `boxplot`.

    Returns:
    - ax (matplotlib.axes.Axes): The Axes object with the boxplot.
    """
    kwargs.setdefault('flierprops',
        dict(marker=".", markeredgecolor="none", alpha=0.1, markersize=5)
    )
    ax = sns.boxplot(*args, **kwargs)

    # Validate jitter_ax parameter
    if jitter_ax not in ["x", "y"]:
        raise ValueError(f'`jitter_ax` must be one of {"x", "y"}')
    else:
        ax_ = int(jitter_ax == "y")

    # Apply jitter to data points
    for line in ax.get_lines()[5::6]:
        data = np.array(line.get_data())
        data[ax_] = data[ax_] + np.random.uniform(*jitter_range, data[ax_].size)
        line.set_data(data)

    return ax

def symlog_axis(vals, ax, which):
    """Sets a symmetrical log scale for the specified axis.

    Parameters:
    - vals (array-like): Data values for setting the axis limits.
    - ax (matplotlib.axes.Axes): The Axes object to modify.
    - which (str): Axis to modify ("x" or "y").

    Returns:
    - ax (matplotlib.axes.Axes): The Axes object with the modified axis scale.
    """
    if which == 'x':
        ax.set_xscale("symlog", linthresh=1, base=10, subs=np.arange(1, 11))
        op_ax = ax.xaxis
    elif which == 'y':
        ax.set_yscale("symlog", linthresh=1, base=10, subs=np.arange(1, 11))
        op_ax = ax.yaxis
    else:
        raise ValueError(f'which must be one of "x" or "y".')

    op_ax.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=0.1, subs=np.arange(1, 10))
    )
    ax_min_sign, ax_max_sign = np.sign(vals.min()), np.sign(vals.max())
    ax_min, ax_max = np.ceil(np.log10(abs(vals.min()))), np.ceil(
        np.log10(abs(vals.max()))
    )
    op_ax.set_view_interval(ax_min_sign * 10 ** ax_min, ax_max_sign * 10 ** ax_max)

    ticklabels = []
    ticks = []
    if ax_min_sign == -1:
        if ax_max_sign == -1:
            ticklabels.extend(
                [
                    f"$\\mathdefault{{-10^{{{int(n)}}}}}$"
                    for n in np.linspace(ax_min, ax_max, int(ax_min - ax_max) + 1)
                ]
            )
            ticks.append(-np.logspace(ax_min, ax_max, int(ax_min - ax_max) + 1))
        else:
            ticklabels.extend(
                [
                    f"$\\mathdefault{{-10^{{{int(n)}}}}}$"
                    for n in np.linspace(ax_min, 0, int(ax_min) + 1)
                ]
            )
            ticks.append(-np.logspace(ax_min, 0, int(ax_min) + 1))

    if ax_max_sign == 1:
        if ax_min_sign == 1:
            ticklabels.extend(
                [
                    f"$\\mathdefault{{10^{{{int(n)}}}}}$"
                    for n in np.linspace(ax_min, ax_max, int(ax_max - ax_min) + 1)
                ]
            )
            ticks.append(np.linspace(ax_min, ax_max, int(ax_max - ax_min) + 1))
        else:
            ticklabels.append("0")
            ticklabels.extend(
                [
                    f"$\\mathdefault{{10^{{{int(n)}}}}}$"
                    for n in np.linspace(0, ax_max, int(ax_max) + 1)
                ]
            )
            ticks.append(np.array([0]))
            ticks.append(np.logspace(0, ax_max, int(ax_max) + 1))

    op_ax.set_ticks(np.concatenate(ticks))
    op_ax.set_ticklabels(ticklabels)
    return ax


def get_cp_feature_table(
    compartments=["cell", "nucleus"],
    channels=["dapi", "tubulin", "gh2ax", "phalloidin"],
    foci_channel="gh2ax",
    correlation=True,
    neighbor_distances=[1],
):
    """
    Generates a feature table for cellular compartments, channels, and additional features.

    Parameters:
    - compartments (list of str): List of compartments (e.g., ["cell", "nucleus"]).
    - channels (list of str): List of channels (e.g., ["dapi", "tubulin", "gh2ax", "phalloidin"]).
    - foci_channel (str or None): Specific channel for foci features, or None if not applicable.
    - correlation (bool): Whether to include correlation features.
    - neighbor_distances (list of int): List of distances for neighbor features.

    Returns:
    - pd.DataFrame: DataFrame containing feature types and their associated metadata.
    """

    from ops.cp_emulator import (
        intensity_columns_multichannel,
        intensity_distribution_columns_multichannel,
        texture_columns_multichannel,
        correlation_columns_multichannel,
        shape_columns,
        shape_features as shape_features_,
    )

    # Collect feature lists
    intensity_features = [v for sublist in intensity_columns_multichannel.values() for v in sublist]
    distribution_features = [v for sublist in intensity_distribution_columns_multichannel.values() for v in sublist]
    texture_features = [v for sublist in texture_columns_multichannel.values() for v in sublist]
    correlation_features = [v for sublist in correlation_columns_multichannel.values() for v in sublist]
    shape_features = list(shape_features_.keys())

    # Remove specific shape features
    for f in ["zernike", "centroid", "feret_diameter", "radius", "hu_moments"]:
        if f in shape_features:
            shape_features.remove(f)

    # Add additional shape features
    shape_features.extend(shape_columns.values())
    shape_features.extend([f"hu_moments_{n}" for n in range(7)])

    # Create DataFrame for foci features, if applicable
    if foci_channel is not None:
        from ops.features import foci
        df_foci = pd.DataFrame(foci.keys(), columns=["feature_type"])\
            .assign(level_0="cell", level_1=foci_channel, level_2="intensity")
    else:
        df_foci = None

    # Create DataFrame for grayscale features
    df_grayscale = pd.concat(
        [df_foci] +
        [pd.DataFrame(intensity_features, columns=["feature_type"]).assign(
            level_0=compartment, level_1=channel, level_2="intensity"
        ) for compartment in compartments for channel in channels] +
        [pd.DataFrame(distribution_features, columns=["feature_type"]).assign(
            level_0=compartment, level_1=channel, level_2="distribution"
        ) for compartment in compartments for channel in channels] +
        [pd.DataFrame(texture_features, columns=["feature_type"]).assign(
            level_0=compartment, level_1=channel, level_2="texture"
        ) for compartment in compartments for channel in channels]
    ).assign(
        feature=lambda x: x.apply(lambda x: x[["level_0", "level_1", "feature_type"]].str.cat(sep="_"), axis=1)
    )

    # Create DataFrame for correlation features, if applicable
    if correlation:
        df_correlation = pd.DataFrame(
            [
                {
                    "feature_type": feature.rsplit("_", 2)[0],
                    "level_0": compartment,
                    "level_1": "correlation",
                    "level_2": f"{first}_{second}"
                }
                for feature in correlation_features
                for compartment in compartments
                for first, second in combinations(channels, r=2)
            ]
        ).drop_duplicates().assign(
            feature=lambda x: x.apply(lambda x: x[["level_0", "feature_type", "level_2"]].str.cat(sep="_"), axis=1)
        )

        # Reverse certain correlation feature names
        df_correlation = pd.concat(
            [
                df_correlation,
                df_correlation.query('feature_type in ["K","manders","rwc","lstsq_slope"]').assign(
                    feature=lambda x: x["feature"].apply(lambda x: "_".join(x.rsplit("_", 2)[:1] + x.rsplit("_", 2)[-1:0:-1]))
                )
            ]
        )
    else:
        df_correlation = pd.DataFrame()

    # Create DataFrame for shape features
    df_shape = pd.concat(
        [pd.DataFrame(shape_features, columns=["feature_type"]).assign(
            level_0=compartment, level_1="shape", level_2="shape"
        ) for compartment in compartments]
    ).assign(
        feature=lambda x: x.apply(lambda x: x[["level_0", "feature_type"]].str.cat(sep="_"), axis=1)
    )

    # Create DataFrame for neighbor features
    df_neighbors = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "feature_type": f"{feature}_{neighbor_distance}",
                        "level_0": compartment,
                        "level_1": "neighbors",
                        "level_2": "neighbors"
                    }
                    for feature in ["number_neighbors", "percent_touching"]
                    for compartment in compartments
                    for neighbor_distance in neighbor_distances
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "feature_type": feature,
                        "level_0": compartment,
                        "level_1": "neighbors",
                        "level_2": "neighbors"
                    }
                    for feature in [
                        "first_neighbor_distance",
                        "second_neighbor_distance",
                        "angle_between_neighbors"
                    ]
                    for compartment in compartments
                ]
            )
        ]
    ).assign(
        feature=lambda x: x.apply(lambda x: x[["level_0", "feature_type"]].str.cat(sep="_"), axis=1)
    )

    # Combine all feature DataFrames into a single DataFrame
    return pd.concat([df_grayscale, df_correlation, df_shape, df_neighbors]).set_index("feature")
