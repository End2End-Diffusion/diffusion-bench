"""Logging utilities for training metrics."""

import csv
import os


def save_eval_to_csv(exp_name: str, mod_name: str, global_step: int, eval_stats: dict,
                     eval_dir: str | None = None):
    """Append evaluation results to a CSV file.

    Different eval datasets contribute different metric columns.
    To keep every metric under a correctly labeled
    column, we rewrite the whole file each call with a header that is the union
    of all rows' columns; missing cells are left blank. Eval CSVs are tiny, so
    the full rewrite is negligible.
    """
    if eval_dir is None:
        eval_dir = os.path.join("experiments", os.environ.get("RAE_USER", "jas"), "evals", "stage1")
    os.makedirs(eval_dir, exist_ok=True)
    csv_path = os.path.join(eval_dir, f"{exp_name}_{mod_name}.csv")

    # Read existing rows so we can rewrite with the union of all columns.
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            for row in csv.DictReader(f):
                row.pop(None, None)  # drop overflow values from older malformed files
                rows.append(row)
    rows.append({'step': global_step, **eval_stats})

    # Union of columns, preserving first-seen order, with 'step' first.
    fieldnames = ['step']
    for row in rows:
        for key in row:
            if key is not None and key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["save_eval_to_csv"]
