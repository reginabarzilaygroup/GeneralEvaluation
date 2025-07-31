import io
import warnings

warnings.filterwarnings(action="ignore", category=RuntimeWarning)

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import sklearn
import seaborn as sns

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, \
    roc_auc_score
from sklearn.calibration import CalibrationDisplay

def load_input_df(name, content, **kwargs):
    if name.endswith(".csv"):
        input_df = pd.read_csv(content, sep=",", **kwargs)
    elif name.endswith(".tsv"):
        input_df = pd.read_csv(content, sep="\t", **kwargs)
    elif name.endswith("xlsx") or name.endswith("xls"):
        input_df = pd.read_excel(content, **kwargs)
    else:
        raise ValueError(f"Unknown extension in file {name}")

    return input_df

def get_split_names(input_df, split_col, true_col, do_print=True):
    all_split_names = input_df[split_col].unique()
    all_split_names = list(sorted(all_split_names))[::-1]
    if do_print:
        print(f"All splits: {all_split_names}")
        print(f"Number of positives: ")
        
    for split_name, split_df in input_df.groupby(split_col):
        N = split_df.shape[0]
        N_pos = split_df[true_col].sum()
        f_pos = N_pos / N
        
        if do_print:
            print(f"{split_name}: {N_pos} / {N} ({f_pos:0.4%})")

    return all_split_names

def metrics_by_category(input_df, categories, category_name, n_bootstraps=0, progress_bar=None):
    """Calculate ROC curves and binary metrics, separated by category (ie year)"""
    curves_by_cat = {}
    stats_by_cat = {}
    metrics_list = []
    num_cats = len(categories)
    bi_iters = n_bootstraps + 1
    total_bis = (n_bootstraps + 1) * num_cats
    for ii, cat in enumerate(categories):
        name = cat["name"]
        true_col = cat["true_col"]
        pred_col = cat["pred_col"]
        cur_df = input_df.query(f"{true_col} >= 0")

        cur_stats, cur_curves = calculate_auc_stats_curves(cur_df[true_col].values, cur_df[pred_col].values)

        bootstrapped_aucs = []
        np.random.seed(42)
        for bi in range(bi_iters):
            if progress_bar:
                progbar_text = f"Processing {name}"
                progbar_text += f" Bootstrap {bi}" if bi >= 1 else ""
                progress_bar.progress((ii*bi_iters + bi) / total_bis, progbar_text)

            # Note that we use the same thresholds for each bootstrap.
            # We're simulating generalization error, not model uncertainty.
            prc_thresholds = cur_curves["prc_thresholds"]
            y_true, y_pred = cur_df[true_col].values, cur_df[pred_col].values
            if bi >= 1:
                N = len(y_true)
                cur_bi_inds = np.random.choice(np.arange(N), N, replace=True)
                y_true, y_pred = y_true[cur_bi_inds], y_pred[cur_bi_inds]

            metrics_df = binary_metrics_by_threshold(y_true, y_pred, prc_thresholds)
            metrics_df["bootstrap_index"] = bi
            metrics_df[category_name] = name

            metrics_list.append(metrics_df)

            bi_cur_stats, bi_cur_curves = calculate_auc_stats_curves(y_true, y_pred, calculate_curves=False)
            bootstrapped_aucs.append(bi_cur_stats["auc"])

        if n_bootstraps > 0:
            cur_stats["auc_ci"] = np.percentile(bootstrapped_aucs, [5, 95]).round(4)

        curves_by_cat[name] = cur_curves
        stats_by_cat[name] = cur_stats

    all_metrics_df = pd.concat(metrics_list)
    return curves_by_cat, stats_by_cat, all_metrics_df

def calculate_auc_stats_curves(y_true, y_pred, calculate_curves=True):
    stat_dict = {"auc": None, "pr_auc": None}
    try:
        stat_dict["auc"] = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        # This can happen if all y_true are the same
        stat_dict["auc"] = np.nan

    try:
        stat_dict["pr_auc"] = float(average_precision_score(y_true, y_pred))
    except ValueError:
        stat_dict["pr_auc"] = np.nan

    curve_dict = {}
    if calculate_curves:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
        # Compute Precision-Recall curve and area
        precision, recall, prc_thresholds = precision_recall_curve(y_true, y_pred)
        curve_dict = {"fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thresholds,
                      "y_true": y_true, "y_pred": y_pred}
        curve_dict.update({"precision": precision, "recall": recall, "prc_thresholds": prc_thresholds})

    stat_dict["N"] = len(y_true)
    return stat_dict, curve_dict

def plot_roc_prc(curves_by_split, stats_by_split, title_prefix="", all_split_names=None):
    if all_split_names is None:
        all_split_names = list(curves_by_split.keys())
        
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    
    # Plot ROC Curves
    # _d = 0.05
    # xoffs = 0.77
    # yoffs = 0.65

    # ax[0].text(xoffs, yoffs+_d, s="AUROC: ", transform=ax[0].transAxes)
    curve_labels = []
    for si, split_name in enumerate(all_split_names):
        fpr, tpr = curves_by_split[split_name]["fpr"], curves_by_split[split_name]["tpr"]
        ax[0].plot(fpr, tpr, lw=2)

        roc_auc = stats_by_split[split_name]["auc"]
        auroc_str = f'{split_name}: {roc_auc:.1%}'
        curve_labels.append(auroc_str)
        # ax[0].text(xoffs, yoffs-_d*si, s=auroc_str, transform=ax[0].transAxes)
    
    ax[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
    ax[0].set_xlim([0.0, 1.01])
    ax[0].set_ylim([0.0, 1.01])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title(f'{title_prefix} Receiver Operating Characteristic')
    ax[0].legend(curve_labels, loc="lower right")

    # Plot Precision-Recall Curve
    # xoffs = 0.77
    # yoffs = 0.90
    # ax[1].text(xoffs, yoffs+_d, s="AUPRC: ", transform=ax[1].transAxes)
    curve_labels = []
    for si, split_name in enumerate(all_split_names):
        recall, prec = curves_by_split[split_name]["recall"], curves_by_split[split_name]["precision"]
        ax[1].plot(recall, prec, lw=2)

        pr_auc = stats_by_split[split_name]["pr_auc"]
        auprc_str = f'{split_name}: {pr_auc:.1%}'
        curve_labels.append(auprc_str)
        # ax[1].text(xoffs, yoffs-_d*si, s=auprc_str, transform=ax[1].transAxes)
        
    ax[1].set_xlim([0.0, 1.01])
    ax[1].set_ylim([0.0, 1.01])
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision (PPV)')
    ax[1].set_title(f'{title_prefix} Precision-Recall Curve')
    ax[1].legend(curve_labels, loc="upper right")

    return fig, ax

def confusion_matrix_3d(y_true, y_pred, thresholds):
    """
    Confusion matrix whose i-th row and j-th column entry indicates the number of samples with
    true label being i-th class,
    and predicted label being j-th class.
    :param y_true:
    :param y_pred:
    :param thresholds:
    :return:
    """
    n_thresholds = len(thresholds)
    output = np.zeros([2, 2, n_thresholds], dtype=int)

    for ti, t_val in enumerate(thresholds):
        cur_pred = y_pred >= t_val
        output[:, :, ti] = confusion_matrix(y_true, cur_pred)

    return output

def binary_metrics_by_threshold(y_true, y_pred, thresholds):
    full_conf_matr = confusion_matrix_3d(y_true, y_pred, thresholds)
    N = len(y_true)
    simple_data = {"threshold": thresholds,
                   "TN": full_conf_matr[0, 0, :],
                   "FN": full_conf_matr[1, 0, :],
                   "TP": full_conf_matr[1, 1, :],
                   "FP": full_conf_matr[0, 1, :],
                   "N": N}
    metrics_df = pd.DataFrame(data=simple_data)

    metrics_df["pred_pos_rate"] = (metrics_df["TP"] + metrics_df["FP"]) / N

    # https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Confusion_matrix
    metrics_df["sensitivity"] = full_conf_matr[1, 1, :] / np.sum(full_conf_matr[1, :, :], axis=0)
    metrics_df["recall"] = metrics_df["sensitivity"]
    metrics_df["TPR"] = metrics_df["recall"]
    metrics_df["FNR"] = 1.0 - metrics_df["TPR"]
    
    metrics_df["specificity"] = full_conf_matr[0, 0, :] / np.sum(full_conf_matr[0, :, :], axis=0)
    metrics_df["TNR"] = metrics_df["specificity"]
    metrics_df["NPV"] = full_conf_matr[0, 0, :] / np.sum(full_conf_matr[:, 0, :], axis=0)
    
    metrics_df["PPV"] = full_conf_matr[1, 1, :] / np.sum(full_conf_matr[:, 1, :], axis=0)
    metrics_df["FDR"] = 1.0 - metrics_df["PPV"]
    metrics_df["precision"] = metrics_df["PPV"]
    
    # Look at the risk of groups above threshold vs below, for each threshold
    metrics_df["relative_risk"] =  (metrics_df["PPV"] / (1.0 - metrics_df["NPV"])).fillna(value=0.0).replace(np.inf, 0.0)
    metrics_df["LR+"] = metrics_df["relative_risk"]
    metrics_df["LR-"] = ((1.0 - metrics_df["PPV"]) / metrics_df["NPV"]).fillna(value=0.0).replace(np.inf, 0.0)

    _tmp = 2*metrics_df["TP"]
    metrics_df["f1_score"] = _tmp / (_tmp + metrics_df[["FP", "FN"]].sum(axis=1))

    metrics_df["balanced_accuracy"] = metrics_df[["recall", "specificity"]].mean(axis=1)

    metrics_df["N"] = N

    if False:
        # https://pmc.ncbi.nlm.nih.gov/articles/PMC10454914/
        # Net benefit = (TP / N) - (FP / N)*(P_t)/(1-P_t)
        _tmp1 = metrics_df["TP"] / metrics_df["N"]
        _tmp2 = metrics_df["FP"] / metrics_df["N"]
        _tmp3 = metrics_df["threshold"] / (1.0 - metrics_df["threshold"])
        metrics_df["net_benefit"] = _tmp1 - (_tmp2*_tmp3)
        # The null operation of treating all patients, aka consider all patients positive
        # The "false positives" would therefore be the total number of negatives, which is the sum of TN and FP at each threshold
        _act_neg_rate = (metrics_df["TN"] + metrics_df["FP"])/metrics_df["N"]
        metrics_df["_net_benefit_treat_all"] = _tmp1 - (_act_neg_rate*_tmp3)

    return metrics_df

def plot_binary_metrics(metrics_df, title_prefix=""):
    x_column = "threshold"
    joint_columns_to_plot = ["PPV", "sensitivity", "specificity"]
    xrange = [0.0, 0.20]
    axis_formatter = ticker.FuncFormatter(lambda x, _: f'{x:.1%}')
    
    def pretty_str(instr):
        tmp = instr.replace("_", " ")
        if tmp.upper() != tmp:
            tmp = tmp.capitalize()
        return tmp
    
    fig = plt.figure()
    for cp in joint_columns_to_plot:
        plt.plot(metrics_df[x_column], metrics_df[cp], label=pretty_str(cp))

    _ = plt.xlabel(pretty_str(x_column))
    # _ = plt.legend(loc='upper right', bbox_to_anchor=(0.99, 0.7), borderaxespad=0.)
    _ = plt.legend(loc='lower right')
    _ = plt.xlim(*xrange)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(axis_formatter)
    ax.yaxis.set_major_formatter(axis_formatter)

    fig.suptitle(f"{title_prefix} Binary Metrics")
    figures = [fig]

    # "net_benefit"
    sep_columns = ["relative_risk"]
    for cp in sep_columns:
        fig = plt.figure()
        ax = plt.gca()
        plt.plot(metrics_df[x_column], metrics_df[cp], label=pretty_str(cp))

        if cp == "net_benefit":
            cy = "_net_benefit_treat_all"
            treat_all = metrics_df[cy]
            keep_locs = np.argwhere(treat_all >= 0).flatten()
            plt.plot(metrics_df[x_column].iloc[keep_locs], metrics_df[cy].iloc[keep_locs], "k--", label="Treat All")

        _ = plt.xlabel(pretty_str(x_column))
        _ = plt.xlim(*xrange)
        figures.append(fig)

        cur_title = cp.replace("_", " ").capitalize()
        fig.suptitle(f"{title_prefix} {cur_title}")

        ax.xaxis.set_major_formatter(axis_formatter)
        # ax.yaxis.set_major_formatter(axis_formatter)

    return figures

def plot_histograms(input_df, true_col, pred_col, title_prefix=""):
    plot_df = input_df.copy()
    true_value_counts_ = plot_df[true_col].value_counts()
    nneg = true_value_counts_[0]
    npos = true_value_counts_[1]
    cat_map = {0: f"Negative ({nneg})", 1: f"Positive ({npos})"}
    plot_df[true_col] = plot_df[true_col].map(cat_map)
    cat_order = [cat_map[0], cat_map[1]]

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), 
                                  gridspec_kw={'wspace': 0.3})
    
    ax = sns.kdeplot(plot_df, ax=ax0, x=pred_col, hue=true_col, hue_order=cat_order,
                linewidth=0.5, bw_adjust=1.0,
                fill=True, common_norm=False, cumulative=False)
    _ = ax.set_xlim([0.0, 1.0])
    _ = ax.set_ylim([0.0, 5.0])
    leg = ax.get_legend()
    _ = leg.set_title(f"{title_prefix} Outcome")
    _ = ax.set_xlabel("Predicted Probability")
    
    ax = ax1
    _ = sns.boxplot(plot_df, ax=ax1, y=pred_col, x=true_col, hue=true_col,
                    order=cat_order, hue_order=cat_order)
    # _ = sns.move_legend(ax, loc="upper right")
    _ = ax.set_xlabel("Outcome")
    _ = ax.set_ylabel("Predicted Probability")
    leg = ax.get_legend()
    if leg:
        _ = leg.set_title("Outcome")
    
    return fig

def get_aucs(y_true, y_pred, target_recall=None):
    # Compute ROC curve and ROC area
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    auroc = auc(fpr, tpr)
    
    # Compute Precision-Recall curve and area
    precision, recall, prc_thresholds = precision_recall_curve(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    stats = {"auroc": auroc, "auprc": auprc}

    if target_recall is not None:
        closest_ind = np.argmin(np.abs(recall - target_recall))
        stats["threshold"] = prc_thresholds[closest_ind]
        stats["recall"] = recall[closest_ind]

    return stats

def print_threshold_metrics(all_metrics_df, split_col, split_name, check_key="recall", check_values=None):
    metrics_df = all_metrics_df[all_metrics_df[split_col] == split_name]
    # display(f"  Split name: {split_name}  ")

    if check_values is None:
        check_values = [0.50, 0.60, 0.70, 0.85, 0.9, 0.95, 0.99, 1.0]
    display_cols = ["threshold", "recall", "PPV", "relative_risk", "TP", "FP", "N"]
    fmt_dict = {"recall": "{:.2%}",
                "PPV": "{:.2%}",
                "relative_risk": "{:0.2f}"}
    
    check_rows = []
    for cv in check_values:
        min_row = np.argmin(np.abs(metrics_df[check_key] - cv))
        check_rows.append(min_row)
    
    # This looks nice but doesn't export to PDF properly
    # display(metrics_df.iloc[check_rows][display_cols].style.hide(axis="index").format(fmt_dict))

    # display(metrics_df.iloc[check_rows][display_cols])
    # display("------")

def plot_waffle(summary_metrics_by_cat_standard, standard_name="Sensitivity",
                category_name="Year", category_val="Year 1"):
    from pywaffle import Waffle

    df = summary_metrics_by_cat_standard.query(f"standard == '{standard_name}'")
    data_dict = df[df[category_name] == category_val].to_dict(orient="records")[0]

    # icons = ["user-check", "user-slash", "user-slash", "user-check"]
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    colors = ["forestgreen", "maroon", "darkorange", "dimgray"]
    icons = ["circle"]*len(colors)
    values = {
        "TP": data_dict["TP"],
        "FN": data_dict["FN"],
        "FP": data_dict["FP"],
        "TN": data_dict["TN"],
    }
    fig = plt.figure(
        FigureClass=Waffle,
        rows=32,
        columns=32,
        values=values,
        colors=colors,
        icons=icons,
        font_size=8,
        starting_location="NW",
        vertical=True,
        legend={
            "loc": "upper left",
            "bbox_to_anchor": (1, 1),
            "fontsize": 12,
        },
    )
    plt.title(f"{category_val}, {standard_name} Target Result")
    plt.tight_layout()

    return fig


def plot_calibration_curve(input_df, true_col, pred_col, fig_name):
    plot_df = input_df.copy()
    y_true = plot_df[true_col].values
    y_pred = plot_df[pred_col].values

    xrange = [0.0, 0.20]
    axis_formatter = ticker.FuncFormatter(lambda x, _: f'{x:.1%}')
    n_bins = 10
    strategy = "quantile"

    fig = plt.figure()
    ax = plt.gca()
    disp = CalibrationDisplay.from_predictions(y_true, y_pred, name=fig_name, ax=ax,
                                               n_bins=n_bins, strategy=strategy)
    _ = ax.set_xlim(*xrange)
    _ = ax.set_ylim(*xrange)
    _ = ax.xaxis.set_major_formatter(axis_formatter)
    _ = ax.yaxis.set_major_formatter(axis_formatter)

    _ = ax.set_xlabel("Mean predicted probability")
    _ = ax.set_ylabel("Fraction of positives")
    _ = ax.set_title(f"{fig_name} Calibration Curve")
    plt.tight_layout()

    return fig


def create_table_percentiles(split_df, pred_col, true_col, fig_name) -> plt.Figure:
    """
    Create a table of percentiles for the predicted probabilities.
    So basically a histogram but in table form.
    :param split_df:
    :param pred_col:
    :param true_col:
    :param fig_name:
    :return:
    """

    percentiles = np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]).astype(int)
    pred_values = split_df[pred_col].values
    perc_values = np.percentile(pred_values, percentiles)
    perc_values = np.round(perc_values, 4)
    # Ensure percentiles are displayed as integers (no trailing .0)
    perc_df = pd.DataFrame({
        "Percentile": percentiles.astype(str),
        "Score": perc_values,
    })

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 7))  # Adjust the size as needed
    # Create the table
    table = ax.table(cellText=perc_df.values,
                     colLabels=perc_df.columns,
                     cellLoc='center', loc='center')

    df = perc_df.copy()
    col_width = 1.0 / df.shape[1]
    col_height = 0.09
    # table.auto_set_font_size(False)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_width(col_width)
        cell.set_height(col_height)

    # Remove axis and set tight layout
    ax.axis('off')
    plt.subplots_adjust(top=0.8, bottom=0.1)
    # Set the title at the top of the figure
    fig.suptitle(f"{fig_name} Prediction Score Percentiles", y=0.98, fontweight='bold')

    return fig

