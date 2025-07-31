#!/usr/bin/env python

__doc__ = """
Run set of evaluations on collected data validation.
We expect a datasheet of patient and exam information.
"""

import argparse
import os
import traceback

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import general_eval.general_eval_lib as gel
import general_eval.metrics as gel_metrics
import general_eval.utils as utils
from general_eval.general_eval_lib import plot_roc_prc

DAYS_PER_YEAR = 365

DIAGNOSIS_DAYS_COL = "Days_to_Cancer"
FOLLOWUP_DAYS_COL = "Days_Followup"


def _get_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--ds_name", default="My Dataset", required=False,
                        type=str, help="Name of dataset. Used in figure titles.")

    parser.add_argument("--input_path", default=None, required=True,
                        type=str, help="Path to the file containing information on a per-exam basis"
                                       "(one exam per row). CSV format.")

    parser.add_argument("--output_path", default=None,
                        type=str, help="Path to save the output report file.")

    parser.add_argument("--category_name", default="Year",
                        type=str, help="Column to use for subdividing data into subsets. "
                                       "Analysis will be repeated across each unique value of this column.")

    parser.add_argument("--recall_target", default=0.85,
                        type=str, help="Target recall value for the model.")

    return parser

def _is_positive_diag(days):
    if days is None:
        return False
    elif str(days).lower() in {"-1", "-1.0", "nan", "nat", "", "none"}:
        return False
    else:
        try:
            return float(days) >= 0
        except ValueError:
            return False

def _is_negative_diag(days):
    return not _is_positive_diag(days)

def _days_to_months(days):
    if _is_negative_diag(days):
        return -1
    return int(days) / 30

def _days_to_years(days):
    if _is_negative_diag(days):
        return -1
    return int(days) / DAYS_PER_YEAR

def _gen_year_vec(row, diagnosis_days_col, followup_days_col, max_followup=5):
    """
    Generate a binary vector indicating whether the patient has been diagnosed with cancer within the given year.
    :param row:
    :param diagnosis_days_col:
    :param followup_days_col:
    :param max_followup:
    :return:
        y_seq - Indicator vector of whether the patient has been diagnosed with cancer within the given year.
            -1 indicates that the patient is not available for followup at that time.
            0 indicates that the patient has not been diagnosed with cancer at that time.
            1 indicates that the patient has been diagnosed with cancer at that time.
        y_mask - Indicator vector of whether the patient is available for followup at that time.
    """
    days_to_last_followup = int(row[followup_days_col])
    years_to_last_followup = int(days_to_last_followup // DAYS_PER_YEAR)

    days_to_cancer = int(row[diagnosis_days_col])
    years_to_cancer = int(days_to_cancer // DAYS_PER_YEAR) if days_to_cancer > -1 else 100

    y = years_to_cancer <= max_followup
    y_seq = np.zeros(max_followup)
    if y:
        # time_at_event = years_to_cancer
        y_seq[years_to_cancer:] = 1
        y_mask = np.ones(max_followup)
    else:
        time_at_event = min(years_to_last_followup, max_followup - 1)
        y_mask = np.array([1] * (time_at_event + 1) + [0] * (max_followup - (time_at_event + 1)))

    for ii, yv in enumerate(y_seq):
        y_seq[ii] = y_seq[ii] if y_mask[ii] else -1

    return y_seq, y_mask

def _fmt_values(values, decimals=1, force_no_perc=False):
    output = []
    perc_fmt_str = f"{{:.{decimals}%}}"
    for val in values:
        try:
            if type(val) == str or type(val) == int:
                output.append(str(val))
            elif float(val) <= 1. and not force_no_perc:
                output.append(perc_fmt_str.format(val))
            else:
                output.append(f"{val:.{decimals}f}")
        except ValueError:
            output.append(str(val))
    return output

def _fmt_ci(pair, decimals=1):
    fmt_str = f"{{:.{decimals}%}}"
    return "[" + fmt_str.format(pair[0]) + ", " + fmt_str.format(pair[1]) + "]"

def _coerce_val_to_int(val):
    if val is None:
        return -1
    elif str(val).lower() in {"-1", "-1.0", "nan", "nat", "", "none"}:
        return -1
    elif pd.isna(val):
        return -1
    else:
        return int(val)

def _coerce_series_to_int(series):
    return series.apply(_coerce_val_to_int)

def generate_true_columns(input_df, diagnosis_days_col, followup_days_col, true_prefix="true", num_years=5):

    input_df["__interval_months"] = input_df[diagnosis_days_col].apply(_days_to_months)

    year_cols = [f"{true_prefix}_year{y}" for y in range(1, num_years+1)]
    for rn, row in input_df.iterrows():
        y_seq, y_mask = _gen_year_vec(row, diagnosis_days_col, followup_days_col, max_followup=num_years)
        input_df.loc[rn, year_cols] = y_seq

    return input_df

def create_basic_stats_table(input_df, diagnosis_days_col):
    basic_stats = {"Total": len(input_df),
                   "Positive": int(input_df[diagnosis_days_col].apply(_is_positive_diag).sum())}
    month_intervals = [(0, 3), (3, 12), (12, 24), (24, 36), (36, 48), (48, 60), (60, 72), (72, np.inf)]
    for start, end in month_intervals:
        if start is None or start == -np.inf:
            start_str = "-Inf"
        else:
            start_str = f"{start} months"
        if end is None or end == np.inf:
            end_str = "Inf"
        else:
            end_str = f"{end} months"

        key = f"{start_str} to {end_str}"

        keep_rows = input_df["__interval_months"].between(start, end, inclusive="left")
        basic_stats[key] = int(keep_rows.sum())

    # pprint.pprint(basic_stats)
    basic_stats_df = pd.DataFrame([basic_stats]).T
    basic_stats_df.columns = ["Count"]
    # fraction = basic_stats_df["Count"] / basic_stats_df.loc["Total", "Count"]
    basic_stats_df["Count"] = basic_stats_df["Count"].astype(int)
    basic_stats_df = basic_stats_df.reset_index()
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 3))  # Adjust the size as needed
    # Create the table
    table = plt.table(cellText=basic_stats_df.values,
                      colLabels=basic_stats_df.columns,
                      cellLoc='center', loc='center')
    df = basic_stats_df
    col_width = 1.0 / df.shape[1]
    col_height = 0.1
    # table.auto_set_font_size(False)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_width(col_width)
        cell.set_height(col_height)

    ax.add_table(table)
    plt.title("Patient Counts by Time Interval", pad=16, fontweight='bold')
    plt.subplots_adjust(top=0.80)
    ax.axis('tight')
    ax.axis('off')

    return fig, basic_stats_df

def create_perf_stats_table(stats_by_cat, subgroup_name=None, subgroup_value=None, bottom_text=None):
    perf_df = pd.DataFrame(stats_by_cat).T
    has_ci = "auc_ci" in perf_df.columns
    keep_cols = ["auc", "pr_auc", "N"]
    if has_ci:
        keep_cols.insert(1, "auc_ci")
        perf_df["auc_ci"] = perf_df["auc_ci"].apply(_fmt_ci)

    for cc in ["auc", "pr_auc"]:
        perf_df[cc] = _fmt_values(perf_df[cc].values)

    perf_df["N"] = perf_df["N"].astype(int)
    perf_df = perf_df[keep_cols]

    perf_df = perf_df.reset_index()
    column_labels = ["Year", "AUROC", "AUPRC", "N"]
    if has_ci:
        column_labels.insert(2, "AUC CI (5%-95%)")
    perf_df.columns = column_labels

    title_suffix = None
    if subgroup_name:
        assert subgroup_value is not None
        title_suffix = f"{subgroup_name}: {subgroup_value}"
        perf_df.insert(1, subgroup_name, subgroup_value)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 3))  # Adjust the size as needed
    # Create the table
    df = perf_df
    table = plt.table(cellText=df.values,
                      colLabels=df.columns,
                      cellLoc='center', loc='center')
    col_width = 1.0 / df.shape[1]
    col_height = 0.1
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_width(col_width)
        cell.set_height(col_height)

    ax.add_table(table)
    label = "Overall Performance"
    if title_suffix:
        label += f" ({title_suffix})"
    plt.title(label, pad=16, fontweight='bold')
    plt.subplots_adjust(top=0.80)
    ax.axis('tight')
    ax.axis('off')

    if bottom_text:
        table_bbox = table.get_window_extent(fig.canvas.get_renderer())
        inv = ax.transAxes.inverted()
        table_bbox_axes = inv.transform_bbox(table_bbox)
        myx = table_bbox_axes.x0
        # print(f"my x: {myx}")
        ax.text(myx, +0.1, bottom_text, ha='left', va='top', transform=ax.transAxes, fontsize=12)


    return fig, perf_df

def generate_standards_df(all_metrics_df, standards, categories, category_name):
    # Print out table of thresholds and stats

    summary_metrics_by_cat_standard = []
    all_standard_metrics = set([ss["metric"] for ss in standards])
    for standard in standards:
        for cat in categories:
            split_name = cat["name"]
            split_df = all_metrics_df.query(f"{category_name} == '{split_name}'").copy()
            if "bootstrap_index" in all_metrics_df.columns:
                split_df = split_df.query("bootstrap_index == 0").copy()

            if standard["direction"] == "min_diff":
                split_df["diff"] = np.abs(split_df[standard["metric"]] - standard["target_value"])
                split_df = split_df.sort_values("diff", ascending=True)
                best_row = split_df.iloc[0].copy()
            else:
                split_df = split_df.sort_values(standard["metric"], ascending=standard["direction"] == "min")
                best_row = split_df.iloc[0].copy()

            if "bootstrap_index" in all_metrics_df.columns:
                num_bootstraps = len(all_metrics_df["bootstrap_index"].unique())

                if num_bootstraps >= 2:
                    bootstrapped_df = all_metrics_df.query(f"{category_name} == '{split_name}'") \
                                                    .query("threshold == @best_row['threshold']")

                    for metric in all_standard_metrics:
                        bootstrapped_vals = bootstrapped_df[metric].values
                        bootstrapped_ci = np.percentile(bootstrapped_vals, [5, 95])
                        col_label = f"{metric}_ci"
                        best_row[col_label] = _fmt_ci(bootstrapped_ci)

            best_row["standard"] = standard["name"]
            summary_metrics_by_cat_standard.append(best_row)

    summary_metrics_by_cat_standard = pd.DataFrame(summary_metrics_by_cat_standard).reset_index(drop=True)
    return summary_metrics_by_cat_standard

def plot_summary_tables_on_pdf(pdf_pages, summary_metrics_by_cat_standard, category_name):

    standard_names = summary_metrics_by_cat_standard["standard"].unique()
    for standard_name in standard_names:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 3))
        # ax.axis('tight')
        ax.axis('off')
        side_margin = 0.01
        top_margin = 0.1
        ax.set_position([side_margin, 0.01, 1.-2*side_margin, 1.-top_margin])

        df = summary_metrics_by_cat_standard.query(f"standard == '{standard_name}'")
        df = df.round(4)

        # Re-arrange column order
        custom_column_order = [
            "standard", category_name, "threshold", "sensitivity", "sensitivity_ci", "PPV", "PPV_ci", "specificity",
            "LR+", "pred_pos_rate", "N",
        ]
        custom_column_order = [cc for cc in custom_column_order if cc in df.columns]
        fmt_kwargs = {"LR+": {"decimals": 2, "force_no_perc": True},
                      "N": {"decimals": 0, "force_no_perc": True},}

        df = df[custom_column_order]
        numeric_columns = custom_column_order[2:]
        for cc in numeric_columns:
            kwargs_ = fmt_kwargs.get(cc, {})
            df[cc] = _fmt_values(df[cc].values, **kwargs_)
        custom_column_labels = list(df.columns)
        cust_mapping = [("f1_score", "F1"), ("balanced_accuracy", "Balanced Acc."),
                        ("pred_pos_rate", "Pred. Pos. Rate"),
                        ("sensitivity", "Sensitivity"), ("sensitivity_ci", "Sensitivity CI"), ("PPV_ci", "PPV CI"),]
        new_cols = []
        for old, new in cust_mapping:
            if old in custom_column_labels:
                custom_column_labels[custom_column_labels.index(old)] = new
                new_cols.append(new)

        def _fmt_col_name(colname):
            if colname in new_cols:
                return colname
            elif len(colname) >= 4:
                return colname.capitalize()
            else:
                return colname.upper()

        custom_column_labels = list(map(_fmt_col_name, custom_column_labels))

        table = plt.table(
            cellText=df.values,
            colLabels=custom_column_labels,
            cellLoc='center',
            loc='center',)

        # Format table for readability
        main_fontsize = 14
        # Leave some margins
        row_height = 1.0/(1.0 + df.shape[0])
        col_width = 1.0/df.shape[1]
        table.auto_set_font_size(False)
        table.set_fontsize(main_fontsize)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(fontweight='bold')

            # cell.set_text_props(ha='right')
            cell.set_height(row_height)
            cell.set_width(col_width)
            num_chars = len(cell.get_text().get_text())
            new_fontsize = main_fontsize
            if num_chars >= 12:
                new_fontsize = main_fontsize - 4
            elif num_chars >= 8:
                new_fontsize = main_fontsize - 2
            cell.get_text().set_fontsize(new_fontsize)

        ax.add_table(table)

        fig.suptitle(f"Best Thresholds by {standard_name}", fontsize=main_fontsize, fontweight='bold')
        pdf_pages.savefig(fig)
        # plt.show()

def glossary_of_terms():
    # Print definitions of terms used in the report

    text1 = """
Sensitivity: The proportion of true positive cases that are correctly 
identified by the model. Also known as "Recall".

Specificity: The proportion of true negative cases that are correctly 
identified by the model. Also known as "True Negative Rate".

PPV: Positive Predicted Value. 
The proportion of positive cases identified by the model 
that are actually positive. Also known as "Precision".
"""

    text2 = """
AUROC: Area Under the Receiver Operating Characteristic curve.
A random classifier has an AUROC of 0.5, 
while a perfect classifier has an AUROC of 1.0.

AUPRC: Area Under the Precision-Recall curve.
A random classifier has an AUPRC equal to the fraction of 
positive cases in the dataset, 
while a perfect classifier has an AUPRC of 1.0.

Uno's C-index: Concordance (C) index. A measure of the model's ability to 
correctly rank the predicted risk of patients, with a value of 1.0 indicating
perfect concordance and 0.5 indicating no concordance.

LR+/Relative Risk: The positive likelihood ratio, which is the ratio of the
true positive rate to the false positive rate.
"""

    all_texts = [text1, text2]
    all_figs = []

    for text in all_texts:
        text = text.strip()
        fig, ax = plt.subplots()
        ax.axis('off')

        ax.text(0.01, 0.95, "Glossary of Terms", fontsize=16, fontweight='bold')
        ax.text(0.01, 0.9, text, fontsize=14, ha="left", va="top")
        plt.tight_layout()
        all_figs.append(fig)

    return all_figs


def get_pdf_metadata(ds_name):
    commit_hash = utils.get_git_commit_hash()
    metadata = {
        "Title": f"{ds_name} Evaluation Report",
        "Author": "https://github.com/reginabarzilaygroup/GeneralEvaluation",
        "Authors": "https://github.com/reginabarzilaygroup/GeneralEvaluation",
        "Creator": "https://github.com/reginabarzilaygroup/GeneralEvaluation",
        "Subject": f"Git commit: {commit_hash}"
    }
    return metadata


def run_full_eval(ds_name, input_path, category_name="Year", subgroups=(), sensitivity_target=0.85, ppv_target=0.20,
                  output_path=None, n_bootstraps=0, progress_bar=None):
    # Column names
    diagnosis_days_col = DIAGNOSIS_DAYS_COL
    followup_days_col = FOLLOWUP_DAYS_COL
    # num_years = None
    outcome_cols = [diagnosis_days_col, followup_days_col]
    required_columns = outcome_cols

    # Set a minimum number of months between exam and diagnosis?
    min_months = 0
    min_days = int(min_months*30)
    categories = [
        {"name": "Year 1", "pred_col": "Year1", "true_col": "true_year1"},
        {"name": "Year 2", "pred_col": "Year2", "true_col": "true_year2"},
        {"name": "Year 3", "pred_col": "Year3", "true_col": "true_year3"},
        {"name": "Year 4", "pred_col": "Year4", "true_col": "true_year4"},
        {"name": "Year 5", "pred_col": "Year5", "true_col": "true_year5"},
        {"name": "Year 6", "pred_col": "Year6", "true_col": "true_year6"},
    ]
    min_n_per_class = 10

    warnings = []

    standards = [
        {"name": f"Sensitivity", "metric": "sensitivity", "direction": "min_diff", "target_value": sensitivity_target},
        {"name": f"PPV", "metric": "PPV", "direction": "min_diff", "target_value": ppv_target},
        # {"name": "F1", "metric": "f1_score", "direction": "max"},
        # {"name": "Balanced Accuracy", "metric": "balanced_accuracy", "direction": "max"},
    ]

    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_path), f"{ds_name} evaluation report.pdf")

    input_name = os.path.basename(input_path)
    input_df = gel.load_input_df(input_name, input_path, comment="#")

    # Remove spaces from column names
    input_df.columns = input_df.columns.str.replace(" ", "")

    keep_categories = []
    for ic, cat in enumerate(categories):
        if cat["pred_col"] in input_df.columns:
            keep_categories.append(cat)
        elif ic <= 5:
            warnings.append(f"Input file does not contain expected prediction column: {cat['pred_col']}")

    num_years = len(keep_categories)
    categories = keep_categories
    category_names = [cat["pred_col"] for cat in categories]
    if len(keep_categories) == 0:
        raise ValueError(f"Input file does not contain any of the expected categories: {category_names}")

    keep_subgroups = []
    for sg in subgroups:
        if sg not in input_df.columns:
            warnings.append(f"Input file does not contain subgroup column: {sg}")
        else:
            keep_subgroups.append(sg)
    subgroups = keep_subgroups
    if warnings:
        warnings.append(f"Columns found in uploaded file: {', '.join(input_df.columns)}")


    # If the followup days value is not present, for a positive diagnosis, set it to the diagnosis days value
    input_df[followup_days_col] = input_df.apply(
        lambda row: row[diagnosis_days_col] if _is_positive_diag(row[diagnosis_days_col]) and
                     pd.isna(row[followup_days_col]) else row[followup_days_col], axis=1)

    input_df = input_df.dropna(subset=[followup_days_col])

    numeric_cols = outcome_cols + [cat["pred_col"] for cat in categories]
    input_df[outcome_cols] = input_df[outcome_cols].apply(_coerce_series_to_int)
    input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    for rc in required_columns:
        if rc not in input_df.columns:
            raise ValueError(f"Input file does not contain required column: {rc}")

    # Keep rows where the diagnosis is at least min_days after the exam
    def _keep_row(row, min_days=min_days):
        # Keep the negatives
        if _is_negative_diag(row[diagnosis_days_col]):
            return True

        # Only include cases where the diagnosis is at least min_days after the exam
        return int(row[diagnosis_days_col]) > min_days

    # Read days until diagnosis, generate binary columns for each year
    generate_true_columns(input_df, diagnosis_days_col, followup_days_col, num_years=num_years)

    if min_months is not None:
        keep_rows = input_df.apply(_keep_row, axis=1)
        input_df = input_df.loc[keep_rows, :]


    # TODO Bootstrapping for c_index too
    c_index = gel_metrics.get_concordance_index_from_df(input_df, diagnosis_days_col, followup_days_col,
                                                        score_columns=category_names)

    ### Calculate ROC curves and binary metrics, separated by category (ie year)
    ###
    curves_by_cat, stats_by_cat, all_metrics_df = gel.metrics_by_category(input_df, categories,
                                                                          category_name=category_name,
                                                                          n_bootstraps=n_bootstraps,
                                                                          progress_bar=progress_bar)

    # Calculate ROC curves and binary metrics for each category (ie year) and subgroup
    subgroup_stats_by_cat = {}
    subgroup_c_indexes = {}
    for subgroup_name in subgroups:
        subgroup_values = input_df[subgroup_name].unique()
        for subgroup_value in subgroup_values:
            if progress_bar is not None:
                progress_bar.progress(0.98, f"Calculating metrics for {subgroup_name}: {subgroup_value}")

            subgroup_df = input_df[input_df[subgroup_name] == subgroup_value]
            cur_num_samples = subgroup_df.shape[0]
            if cur_num_samples == 0:
                continue

            _, cur_stats_by_cat, _ = gel.metrics_by_category(subgroup_df, categories,
                                                         category_name=category_name,
                                                         n_bootstraps=n_bootstraps)
            subgroup_stats_by_cat[(subgroup_name, subgroup_value)] = cur_stats_by_cat

            subgroup_c_indexes[(subgroup_name, subgroup_value)] = gel_metrics.get_concordance_index_from_df(subgroup_df,
                                                                  diagnosis_days_col, followup_days_col,
                                                                  score_columns=category_names)


        del cur_stats_by_cat
    ###
    ###

    # ------------ Plotting ------------ #
    if progress_bar is not None:
        progress_bar.progress(0.99, "Generating figures")

    sns.set_theme(style="darkgrid")
    # Preconfigured values: {paper, notebook, talk, poster}
    sns.set_context("notebook", font_scale=1.0)
    pdf_pages = PdfPages(output_path, metadata=get_pdf_metadata(ds_name))

    def _save_and_close_fig(_pdf_pages, _fig):
        _pdf_pages.savefig(_fig)
        plt.close(_fig)

    ### FIGURE Plot ROC Curves
    fig, ax = plot_roc_prc(curves_by_cat, stats_by_cat, ds_name)
    _save_and_close_fig(pdf_pages, fig)

    ### TABLE Performance statistics (AUC, ROC) by category
    bottom_text = f"Uno's C-index: {c_index:.1%}"
    fig, overall_perf_df = create_perf_stats_table(stats_by_cat, bottom_text=bottom_text)
    _save_and_close_fig(pdf_pages, fig)

    for subgroup_key, cur_stats_by_cat in subgroup_stats_by_cat.items():
        ### TABLE Performance statistics (AUC, ROC) by subgroup
        subgroup_name, subgroup_value = subgroup_key
        subgroup_c_index = subgroup_c_indexes[subgroup_key]
        bottom_text = f"Uno's C-index: {subgroup_c_index:.1%}"
        fig, overall_perf_df = create_perf_stats_table(cur_stats_by_cat, subgroup_name, subgroup_value, bottom_text=bottom_text)
        _save_and_close_fig(pdf_pages, fig)

    ### TABLE Basic stats table (counts of patients within specific time windows)
    fig, basic_stats_df = create_basic_stats_table(input_df, diagnosis_days_col)
    _save_and_close_fig(pdf_pages, fig)

    ### TABLEs of thresholds for particular targets
    summary_metrics_by_cat_standard = generate_standards_df(all_metrics_df, standards, categories, category_name)
    plot_summary_tables_on_pdf(pdf_pages, summary_metrics_by_cat_standard, category_name)

    figs = glossary_of_terms()
    for fig in figs:
        _save_and_close_fig(pdf_pages, fig)

    ### FIGURE

    ### MULTIPLE FIGURES
    # Loop through categories (ie years) and plot binary metrics and distributions
    for cat in categories:
        split_name = cat["name"]
        true_col = cat["true_col"]
        pred_col = cat["pred_col"]
        fig_name = f"{split_name}"
        split_df = input_df.query(f"{true_col} >= 0")

        metrics_df = all_metrics_df
        if category_name is not None:
            metrics_df = all_metrics_df[all_metrics_df[category_name] == split_name]
        if "bootstrap_index" in all_metrics_df.columns:
            metrics_df = metrics_df.query("bootstrap_index == 0")

        class_counts = split_df[true_col].value_counts()
        skip_cat = False
        if len(class_counts) < 2:
            warnings.append(f"Split {split_name} only has {len(class_counts)} classes. Skipping plots.")
            skip_cat = True
        for cc, val in class_counts.items():
            if val <= min_n_per_class:
                warnings.append(f"Split {split_name}, class {cc} only has N={val} samples. Skipping plots.")
                skip_cat = True
        if skip_cat:
            continue

        ### FIGURE
        figures = gel.plot_binary_metrics(metrics_df, fig_name)

        for fig in figures:
            _save_and_close_fig(pdf_pages, fig)

        # Waffle chart showing distribution of positive/negative cases
        for standard in standards:
            fig = gel.plot_waffle(summary_metrics_by_cat_standard, category_name=category_name, category_val=split_name,
                                  standard_name=standard["name"])
            _save_and_close_fig(pdf_pages, fig)

        ### FIGURE
        # Histogram and boxplot of predictions by actual class
        try:
            fig = gel.plot_histograms(split_df, true_col, pred_col, fig_name)
            _save_and_close_fig(pdf_pages, fig)
        except Exception as exc:
            warnings.append(f"Error plotting histogram {split_name}: {exc}")
            print(traceback.format_exc())
            continue

        ### TABLE
        # Table of percentiles of predictions
        fig = gel.create_table_percentiles(split_df, pred_col, true_col, fig_name)
        _save_and_close_fig(pdf_pages, fig)

        ### FIGURE
        # Calibration plot
        fig = gel.plot_calibration_curve(split_df, true_col, pred_col, fig_name)
        _save_and_close_fig(pdf_pages, fig)


    pdf_pages.close()

    return output_path, all_metrics_df, warnings


if True and __name__ == "__main__":
    args= _get_parser().parse_args()
    run_full_eval(args.ds_name, args.input_path, category_name=args.category_name, sensitivity_target=args.recall_target,
                  output_path=args.output_path)
