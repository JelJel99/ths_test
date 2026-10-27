"""
Pattern-Based Robustness Analysis for IndoBERT Clickbait Detection
==================================================================

This module analyses how an *already-trained* IndoBERT specialist behaves when
the same headline is perturbed by SYNONYM replacement at three intensity
levels - Low, Medium, High. It does NOT train or modify the model.

What it does
------------
1.  Loads the per-domain Low / Medium / High *semantic* (synonym) perturbation
    CSVs produced by ``EvaluationEngine.generate_all_perturbations`` and matches
    the three levels row-by-row on the original headline ``id``.
2.  Validates data quality (missing / duplicate ids, missing text, inconsistent
    labels across levels) and reports those issues SEPARATELY - they are never
    silently treated as "failed perturbations".
3.  Classifies every valid, eligible sample into exactly one of five
    exact-string-equality patterns, or into a ``Failed`` category when any
    level did not reach its target perturbation range.
4.  Produces dynamic distribution statistics, per-level success/failure stats,
    pattern-specific model metrics (accuracy, macro-F1), determinism checks,
    publication-quality figures, and a findings/limitations report.

Design notes
------------
*   The only part that needs the model is the prediction step, which accepts a
    ``predict_fn`` callable so the analysis logic is fully testable without a
    GPU or checkpoint.
*   Exact string equality is used for pattern assignment. Different strings do
    NOT necessarily carry different meaning or monotonically-increasing
    semantic change - this is stated explicitly in the report.
*   Nothing here overwrites the source datasets, checkpoints, or existing
    results; all outputs go to a dedicated analysis directory.

Label mapping: 0 = non-clickbait, 1 = clickbait.

Author: thesis robustness-analysis pipeline
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# The three perturbation intensity levels, always processed in this order.
LEVELS: Tuple[str, str, str] = ("low", "medium", "high")

# The five exact-string-equality patterns.
PATTERN_LMH_EQUAL = "L=M=H"          # all three identical
PATTERN_MH_EQ_L_DIFF = "M=H!=L"      # medium == high, low differs
PATTERN_LM_EQ_H_DIFF = "L=M!=H"      # low == medium, high differs
PATTERN_LH_EQ_M_DIFF = "L=H!=M"      # low == high, medium differs
PATTERN_ALL_DIFF = "L!=M!=H"         # all three pairwise different

PATTERNS: Tuple[str, ...] = (
    PATTERN_LMH_EQUAL,
    PATTERN_MH_EQ_L_DIFF,
    PATTERN_LM_EQ_H_DIFF,
    PATTERN_LH_EQ_M_DIFF,
    PATTERN_ALL_DIFF,
)

# Samples that did not satisfy the per-level target range at every level.
CATEGORY_FAILED = "Failed"

# The three patterns prioritised for the main performance analysis.
MAIN_PATTERNS: Tuple[str, str, str] = (
    PATTERN_LMH_EQUAL,
    PATTERN_MH_EQ_L_DIFF,
    PATTERN_ALL_DIFF,
)

# Data-quality issue codes (reported separately; excluded from valid denominators).
DQ_MISSING_LEVEL = "missing_level_row"        # id absent from one or more levels
DQ_DUPLICATE_ID = "duplicate_id"              # id appears more than once in a level
DQ_MISSING_TEXT = "missing_text"              # perturbed text missing/empty at some level
DQ_INCONSISTENT_LABEL = "inconsistent_label"  # ground-truth label differs across levels

# Canonical domain -> raw-file mapping (mirrors config.DataConfig.DOMAIN_FILES).
DOMAIN_FILES: Dict[str, str] = {
    "Technology": "technology.csv",
    "Politics": "politic.csv",
    "Health": "health.csv",
    "Sport": "sport.csv",
    "Education": "education.csv",
}

# Column-name candidates. The perturbation CSVs keep the original ``id`` and a
# lowercase ``label``; perturbed text lives in both ``perturbed_text`` and the
# overwritten ``text`` column. We resolve defensively.
ID_COLUMN_CANDIDATES = ("id", "Id", "ID")
LABEL_COLUMN_CANDIDATES = ("label", "Label")
PERTURBED_TEXT_COLUMN_CANDIDATES = ("perturbed_text", "text")
IN_RANGE_COLUMN_CANDIDATES = ("perturbation_in_range",)


# ---------------------------------------------------------------------------
# RESULT CONTAINERS
# ---------------------------------------------------------------------------
PERTURBED_TEXT_COLUMN_CANDIDATES = ("perturbed_text", "text")
IN_RANGE_COLUMN_CANDIDATES = ("perturbation_in_range",)
class DomainFrame:
    """Aligned per-domain data after id-matching the three levels."""

    domain: str
    # Wide frame indexed by id with per-level text, in-range flag and label.
    aligned: pd.DataFrame
    # Data-quality records: list of dicts {id, issue, detail}.
    data_quality: List[dict] = field(default_factory=list)
    # Per-level raw row counts (before alignment), for reconciliation.
    raw_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# COLUMN RESOLUTION HELPERS
# ---------------------------------------------------------------------------

def resolve_column(
    dataframe: pd.DataFrame,
    candidate_names: Sequence[str],
    role: str,
    source: str,
) -> str:
    """Return the first present column from ``candidate_names`` or raise.

    We look through the candidate names one at a time and return the first one
    that actually exists in the dataframe. If none exist we raise a clear
    error that lists what we tried and what columns were available.
    """
    for candidate_name in candidate_names:
        if candidate_name in dataframe.columns:
            return candidate_name

    available_columns = list(dataframe.columns)
    raise KeyError(
        f"[{source}] Could not find a '{role}' column. "
        f"Tried {list(candidate_names)}; available: {available_columns}"
    )


def read_perturbed_csv(path: str) -> pd.DataFrame:
    """
    Read one perturbed-level CSV.

    The engine writes these with ``to_csv(index=False)`` (comma-separated).
    We still sniff for a semicolon header as a defensive fallback, because the
    *raw* dataset files are semicolon-delimited and a user may point this at a
    hand-made file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Perturbed CSV not found: {path}")

    dataframe = pd.read_csv(path)

    # Fallback: a single-column frame whose only header still contains ';'
    # means the file was really semicolon-delimited. Re-read it that way.
    only_one_column = dataframe.shape[1] == 1
    header_looks_semicolon = ";" in str(dataframe.columns[0])
    if only_one_column and header_looks_semicolon:
        dataframe = pd.read_csv(path, sep=";")

    return dataframe


def text_value_is_blank(text_value) -> bool:
    """Return True when a perturbed-text value is missing or only whitespace."""
    if pd.isna(text_value):
        return True
    if str(text_value).strip() == "":
        return True
    return False


# ---------------------------------------------------------------------------
# STAGE 1 - LOAD & ALIGN BY ID  (+ DATA-QUALITY VALIDATION)
# ---------------------------------------------------------------------------

def load_and_align_domain(
    domain: str,
    level_paths: Dict[str, str],
) -> DomainFrame:
    """
    Load the Low/Medium/High perturbed CSVs for one domain and align them by
    the original headline ``id``.

    Parameters
    ----------
    domain:
        Canonical domain name (e.g. ``"Technology"``).
    level_paths:
        Mapping ``{"low": path, "medium": path, "high": path}``.

    Returns
    -------
    DomainFrame
        ``aligned`` is a frame indexed by ``id`` with, per level ``X`` in
        {low, medium, high}:
            ``text_X``   perturbed headline string
            ``in_range_X`` bool - did level X reach its target range
            ``label_X``  ground-truth label observed in that level's file
        plus a single reconciled ``label`` column (only where consistent).
        Rows with data-quality problems are EXCLUDED from ``aligned`` and
        recorded in ``data_quality`` instead.

    Notes
    -----
    Data-quality problems are reported, never reclassified as failed
    perturbations:
        * duplicate id within a level
        * id missing from one or more levels
        * missing / empty perturbed text at any level
        * ground-truth label inconsistent across the three levels
    """
    # Make sure the caller gave us a path for every level we expect.
    missing_levels = []
    for level in LEVELS:
        if level not in level_paths:
            missing_levels.append(level)
    if missing_levels:
        raise ValueError(
            f"[{domain}] level_paths missing entries for: {missing_levels}"
        )

    normalised_frame_by_level: Dict[str, pd.DataFrame] = {}
    raw_counts: Dict[str, int] = {}
    data_quality_records: List[dict] = []

    # --- read each level and normalise its column names -------------------
    for level in LEVELS:
        path = level_paths[level]
        raw_frame = read_perturbed_csv(path)
        raw_counts[level] = int(len(raw_frame))

        source_label = f"{domain}/{level}"
        id_column = resolve_column(
            raw_frame, ID_COLUMN_CANDIDATES, "id", source_label
        )
        label_column = resolve_column(
            raw_frame, LABEL_COLUMN_CANDIDATES, "label", source_label
        )
        text_column = resolve_column(
            raw_frame, PERTURBED_TEXT_COLUMN_CANDIDATES,
            "perturbed_text", source_label
        )
        in_range_column = resolve_column(
            raw_frame, IN_RANGE_COLUMN_CANDIDATES,
            "perturbation_in_range", source_label
        )

        # Build a small frame with uniform, level-suffixed column names.
        normalised_frame = pd.DataFrame({
            "id": raw_frame[id_column],
            f"label_{level}": raw_frame[label_column],
            f"text_{level}": raw_frame[text_column],
            f"in_range_{level}": raw_frame[in_range_column].astype("boolean"),
        })

        # Duplicate ids are a data-quality problem. We record every duplicated
        # id, then drop ALL copies because we cannot know which row is correct.
        duplicate_id_mask = normalised_frame["id"].duplicated(keep=False)
        if duplicate_id_mask.any():
            duplicated_ids = normalised_frame.loc[duplicate_id_mask, "id"]
            unique_duplicated_ids = duplicated_ids.unique().tolist()
            unique_duplicated_ids = sorted(unique_duplicated_ids, key=str)
            for duplicated_id in unique_duplicated_ids:
                data_quality_records.append({
                    "id": duplicated_id,
                    "issue": DQ_DUPLICATE_ID,
                    "detail": f"appears multiple times in level '{level}'",
                })
            normalised_frame = normalised_frame.loc[~duplicate_id_mask].copy()

        normalised_frame_by_level[level] = normalised_frame.set_index("id")

    # --- reconcile the id sets across the three levels --------------------
    id_set_by_level: Dict[str, set] = {}
    for level in LEVELS:
        id_set_by_level[level] = set(normalised_frame_by_level[level].index)

    ids_in_all_levels = (
        id_set_by_level["low"]
        & id_set_by_level["medium"]
        & id_set_by_level["high"]
    )
    ids_in_any_level = (
        id_set_by_level["low"]
        | id_set_by_level["medium"]
        | id_set_by_level["high"]
    )

    ids_missing_from_some_level = ids_in_any_level - ids_in_all_levels
    for incomplete_id in sorted(ids_missing_from_some_level, key=str):
        levels_present = []
        for level in LEVELS:
            if incomplete_id in id_set_by_level[level]:
                levels_present.append(level)
        data_quality_records.append({
            "id": incomplete_id,
            "issue": DQ_MISSING_LEVEL,
            "detail": f"present only in levels {levels_present}",
        })

    # --- build the aligned (wide) frame over the common ids ---------------
    if len(ids_in_all_levels) > 0:
        ordered_common_ids = sorted(ids_in_all_levels, key=str)
        frames_to_join = []
        for level in LEVELS:
            frames_to_join.append(
                normalised_frame_by_level[level].loc[ordered_common_ids]
            )
        aligned = pd.concat(frames_to_join, axis=1)
    else:
        aligned = pd.DataFrame()

    # --- missing / empty text check ---------------------------------------
    if not aligned.empty:
        blank_text_mask = pd.Series(False, index=aligned.index)
        for level in LEVELS:
            text_column_name = f"text_{level}"
            level_is_blank = aligned[text_column_name].apply(text_value_is_blank)
            blank_text_mask = blank_text_mask | level_is_blank

        ids_with_blank_text = aligned.index[blank_text_mask].tolist()
        for blank_text_id in ids_with_blank_text:
            levels_that_are_blank = []
            for level in LEVELS:
                text_value = aligned.loc[blank_text_id, f"text_{level}"]
                if text_value_is_blank(text_value):
                    levels_that_are_blank.append(level)
            data_quality_records.append({
                "id": blank_text_id,
                "issue": DQ_MISSING_TEXT,
                "detail": f"empty perturbed text at levels {levels_that_are_blank}",
            })

        aligned = aligned.loc[~blank_text_mask].copy()

    # --- label-consistency check ------------------------------------------
    if not aligned.empty:
        label_low = aligned["label_low"]
        label_medium = aligned["label_medium"]
        label_high = aligned["label_high"]

        # A sample is consistent when all three level labels are equal.
        low_equals_medium = (label_low == label_medium)
        medium_equals_high = (label_medium == label_high)
        label_is_consistent = low_equals_medium & medium_equals_high

        ids_with_inconsistent_label = aligned.index[~label_is_consistent].tolist()
        for inconsistent_id in ids_with_inconsistent_label:
            label_values_by_level = {}
            for level in LEVELS:
                label_values_by_level[level] = aligned.loc[
                    inconsistent_id, f"label_{level}"
                ]
            data_quality_records.append({
                "id": inconsistent_id,
                "issue": DQ_INCONSISTENT_LABEL,
                "detail": f"labels differ across levels: {label_values_by_level}",
            })

        aligned = aligned.loc[label_is_consistent].copy()
        if not aligned.empty:
            # All three labels are equal here, so taking the low label and
            # casting to int gives us a single reconciled ground-truth label.
            aligned["label"] = aligned["label_low"].astype(int)

    frame = DomainFrame(
        domain=domain,
        aligned=aligned,
        data_quality=data_quality_records,
        raw_counts=raw_counts,
    )
    logger.info(
        "[%s] aligned=%d valid ids | data-quality issues=%d "
        "(raw low/med/high = %d/%d/%d)",
        domain, len(aligned), len(data_quality_records),
        raw_counts["low"], raw_counts["medium"], raw_counts["high"],
    )
    return frame


# ---------------------------------------------------------------------------
# STAGE 2 - PATTERN CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_pattern(low_text: str, medium_text: str, high_text: str) -> str:
    """
    Assign exactly one of the five exact-string-equality patterns.

    Pairwise equality is computed on the exact strings. For the all-different
    pattern we explicitly verify low != medium, low != high and medium != high.
    """
    low_equals_medium = (low_text == medium_text)
    low_equals_high = (low_text == high_text)
    medium_equals_high = (medium_text == high_text)

    # All three identical.
    if low_equals_medium and medium_equals_high:
        return PATTERN_LMH_EQUAL

    # Medium and High identical, Low differs.
    if medium_equals_high and not low_equals_medium:
        return PATTERN_MH_EQ_L_DIFF

    # Low and Medium identical, High differs.
    if low_equals_medium and not medium_equals_high:
        return PATTERN_LM_EQ_H_DIFF

    # Low and High identical, Medium differs.
    if low_equals_high and not low_equals_medium:
        return PATTERN_LH_EQ_M_DIFF

    # None of the pairs are equal: all three strings are pairwise different.
    assert not low_equals_medium
    assert not low_equals_high
    assert not medium_equals_high
    return PATTERN_ALL_DIFF


def assign_patterns(frame: DomainFrame) -> pd.DataFrame:
    """
    Produce a per-sample assignment table for one domain.

    Every valid (data-quality-clean) sample is accounted for exactly once:
    either in one of the five patterns (if eligible - all three levels in
    range) or in the ``Failed`` category.

    Returns
    -------
    DataFrame with one row per valid id and columns:
        domain, id, label,
        text_low, text_medium, text_high,
        in_range_low, in_range_medium, in_range_high,
        eligible (bool), n_levels_failed (0..3),
        category  -> one of PATTERNS or CATEGORY_FAILED
    """
    assignment_columns = [
        "domain", "id", "label",
        "text_low", "text_medium", "text_high",
        "in_range_low", "in_range_medium", "in_range_high",
        "eligible", "n_levels_failed", "category",
    ]

    aligned = frame.aligned
    if aligned.empty:
        return pd.DataFrame(columns=assignment_columns)

    assignment_rows: List[dict] = []
    for sample_id, aligned_row in aligned.iterrows():
        low_in_range = bool(aligned_row["in_range_low"])
        medium_in_range = bool(aligned_row["in_range_medium"])
        high_in_range = bool(aligned_row["in_range_high"])

        # Count how many of the three levels failed their target range.
        number_of_levels_failed = 0
        if not low_in_range:
            number_of_levels_failed = number_of_levels_failed + 1
        if not medium_in_range:
            number_of_levels_failed = number_of_levels_failed + 1
        if not high_in_range:
            number_of_levels_failed = number_of_levels_failed + 1

        sample_is_eligible = (number_of_levels_failed == 0)

        if sample_is_eligible:
            category = classify_pattern(
                str(aligned_row["text_low"]),
                str(aligned_row["text_medium"]),
                str(aligned_row["text_high"]),
            )
        else:
            category = CATEGORY_FAILED

        assignment_rows.append({
            "domain": frame.domain,
            "id": sample_id,
            "label": int(aligned_row["label"]),
            "text_low": aligned_row["text_low"],
            "text_medium": aligned_row["text_medium"],
            "text_high": aligned_row["text_high"],
            "in_range_low": low_in_range,
            "in_range_medium": medium_in_range,
            "in_range_high": high_in_range,
            "eligible": sample_is_eligible,
            "n_levels_failed": number_of_levels_failed,
            "category": category,
        })

    return pd.DataFrame(assignment_rows)


# ---------------------------------------------------------------------------
# STAGE 3 - DISTRIBUTION & PER-LEVEL STATISTICS  (dynamic, nothing hardcoded)
# ---------------------------------------------------------------------------

def pattern_distribution(assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-(domain, category) distribution table.

    Every pattern is included for every domain even when its count is zero, so
    the table shape is stable and sparse/unobserved patterns are explicit.
    The ``Failed`` category is listed alongside the five patterns but is
    clearly flagged via the ``is_failed`` column.

    Denominators (made explicit as named columns):
        * ``pct_of_valid``    = n / (all valid samples in the domain)
                                 where "valid" = eligible + failed (i.e. every
                                 data-quality-clean sample).
        * ``pct_of_eligible`` = n / (eligible samples in the domain)
                                 undefined (NaN) for the Failed category.

    Columns
    -------
    domain, category, is_failed, n,
    n_label0, n_label1,
    n_failing_ge1_level,
    pct_of_valid, pct_of_eligible,
    denom_valid, denom_eligible
    """
    all_categories = list(PATTERNS) + [CATEGORY_FAILED]

    if assignments.empty:
        domains = []
    else:
        domains = sorted(assignments["domain"].unique().tolist())

    distribution_rows: List[dict] = []
    for domain in domains:
        domain_assignments = assignments[assignments["domain"] == domain]

        # "Valid" means every data-quality-clean sample (eligible + failed).
        denominator_valid = int(len(domain_assignments))

        eligible_mask = domain_assignments["category"] != CATEGORY_FAILED
        denominator_eligible = int(eligible_mask.sum())

        for category in all_categories:
            category_rows = domain_assignments[
                domain_assignments["category"] == category
            ]
            count = int(len(category_rows))
            count_label0 = int((category_rows["label"] == 0).sum())
            count_label1 = int((category_rows["label"] == 1).sum())

            # Number of samples in this category that failed >=1 level.
            # For the five patterns this is 0 by construction; for Failed it
            # is the count of samples failing at least one level.
            failed_at_least_one_mask = category_rows["n_levels_failed"] >= 1
            count_failing_at_least_one_level = int(failed_at_least_one_mask.sum())

            category_is_failed = (category == CATEGORY_FAILED)

            # Percentage of all valid samples in this domain.
            if denominator_valid > 0:
                percentage_of_valid = 100.0 * count / denominator_valid
            else:
                percentage_of_valid = np.nan

            # Percentage among eligible samples. Undefined for the Failed
            # category, because Failed samples are not eligible.
            if category_is_failed:
                percentage_of_eligible = np.nan
            elif denominator_eligible > 0:
                percentage_of_eligible = 100.0 * count / denominator_eligible
            else:
                percentage_of_eligible = np.nan

            distribution_rows.append({
                "domain": domain,
                "category": category,
                "is_failed": category_is_failed,
                "n": count,
                "n_label0": count_label0,
                "n_label1": count_label1,
                "n_failing_ge1_level": count_failing_at_least_one_level,
                "pct_of_valid": percentage_of_valid,
                "pct_of_eligible": percentage_of_eligible,
                "denom_valid": denominator_valid,
                "denom_eligible": denominator_eligible,
            })

    return pd.DataFrame(distribution_rows)


def per_level_success_stats(
    assignments: pd.DataFrame,
    frames: Optional[Dict[str, DomainFrame]] = None,
) -> pd.DataFrame:
    """
    Per-level (independent) perturbation success / failure counts and rates.

    A single sample may be in-range at one level and out-of-range at another;
    each level is counted independently over ALL valid samples of the domain
    (eligible + failed), because ``in_range_X`` is defined for every valid
    sample regardless of overall eligibility.

    Columns: domain, level, n_valid, n_success, n_fail, success_rate,
             fail_rate
    """
    if assignments.empty:
        domains = []
    else:
        domains = sorted(assignments["domain"].unique().tolist())

    per_level_rows: List[dict] = []
    for domain in domains:
        domain_assignments = assignments[assignments["domain"] == domain]
        number_of_valid_samples = int(len(domain_assignments))

        for level in LEVELS:
            in_range_column_name = f"in_range_{level}"
            number_of_successes = int(domain_assignments[in_range_column_name].sum())
            number_of_failures = number_of_valid_samples - number_of_successes

            if number_of_valid_samples > 0:
                success_rate = number_of_successes / number_of_valid_samples
                fail_rate = number_of_failures / number_of_valid_samples
            else:
                success_rate = np.nan
                fail_rate = np.nan

            per_level_rows.append({
                "domain": domain,
                "level": level,
                "n_valid": number_of_valid_samples,
                "n_success": number_of_successes,
                "n_fail": number_of_failures,
                "success_rate": success_rate,
                "fail_rate": fail_rate,
            })

    return pd.DataFrame(per_level_rows)


def data_quality_summary(frames: Dict[str, DomainFrame]) -> pd.DataFrame:
    """
    Flatten per-domain data-quality records into one long-form table.

    Returns a table with columns: domain, id, issue, detail.
    """
    data_quality_rows: List[dict] = []
    for domain, frame in frames.items():
        for record in frame.data_quality:
            row = {"domain": domain}
            row.update(record)
            data_quality_rows.append(row)

    if len(data_quality_rows) == 0:
        return pd.DataFrame(columns=["domain", "id", "issue", "detail"])

    return pd.DataFrame(data_quality_rows)


def reconcile_counts(
    frames: Dict[str, DomainFrame],
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validation: confirm every valid sample is accounted for exactly once and
    that valid + unique-data-quality ids reconcile with the aligned total.

    For each domain we check:
        n_valid (assignments)  == n_aligned (frame.aligned)
        sum of pattern+Failed counts == n_valid
        n_eligible + n_failed == n_valid

    Returns a per-domain reconciliation table with a boolean ``ok`` column and
    logs a warning for any mismatch.
    """
    reconciliation_rows: List[dict] = []
    for domain, frame in frames.items():
        domain_assignments = assignments[assignments["domain"] == domain]
        number_of_valid_samples = int(len(domain_assignments))
        number_of_aligned_rows = int(len(frame.aligned))

        # Count the distinct ids that were excluded for data-quality reasons.
        data_quality_frame = pd.DataFrame(frame.data_quality)
        if data_quality_frame.empty:
            number_of_data_quality_ids = 0
        else:
            number_of_data_quality_ids = int(data_quality_frame["id"].nunique())

        # The sum across the five patterns + Failed must equal n_valid.
        category_counts = domain_assignments["category"].value_counts()
        sum_of_category_counts = int(category_counts.sum())

        eligible_mask = domain_assignments["category"] != CATEGORY_FAILED
        failed_mask = domain_assignments["category"] == CATEGORY_FAILED
        number_of_eligible = int(eligible_mask.sum())
        number_of_failed = int(failed_mask.sum())

        valid_matches_aligned = (number_of_valid_samples == number_of_aligned_rows)
        categories_sum_to_valid = (sum_of_category_counts == number_of_valid_samples)
        parts_sum_to_valid = (
            (number_of_eligible + number_of_failed) == number_of_valid_samples
        )
        reconciliation_ok = (
            valid_matches_aligned
            and categories_sum_to_valid
            and parts_sum_to_valid
        )

        if not reconciliation_ok:
            logger.warning(
                "[%s] RECONCILIATION MISMATCH: valid=%d aligned=%d "
                "sum_categories=%d eligible=%d failed=%d",
                domain, number_of_valid_samples, number_of_aligned_rows,
                sum_of_category_counts, number_of_eligible, number_of_failed,
            )

        reconciliation_rows.append({
            "domain": domain,
            "n_valid": number_of_valid_samples,
            "n_aligned": number_of_aligned_rows,
            "n_eligible": number_of_eligible,
            "n_failed": number_of_failed,
            "sum_categories": sum_of_category_counts,
            "n_data_quality_ids_excluded": number_of_data_quality_ids,
            "raw_low": frame.raw_counts.get("low"),
            "raw_medium": frame.raw_counts.get("medium"),
            "raw_high": frame.raw_counts.get("high"),
            "ok": reconciliation_ok,
        })

    return pd.DataFrame(reconciliation_rows)


# ---------------------------------------------------------------------------
# STAGE 4 - PATTERN-SPECIFIC MODEL EVALUATION
# ---------------------------------------------------------------------------

# A predict function maps a list of texts -> list of int labels (0/1), in the
# SAME order as the input. This indirection keeps the analysis logic
# independent of torch so it can be tested with a simple stub.
PredictFn = Callable[[List[str]], Sequence[int]]


def compute_macro_f1(true_labels: Sequence[int],
                     predicted_labels: Sequence[int]) -> float:
    """Macro-F1 with labels=[0, 1] and zero_division=0 (sklearn)."""
    from sklearn.metrics import f1_score
    macro_f1_value = f1_score(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    return float(macro_f1_value)


def compute_accuracy(true_labels: Sequence[int],
                     predicted_labels: Sequence[int]) -> float:
    """Plain accuracy (sklearn). Returns NaN for an empty group."""
    from sklearn.metrics import accuracy_score
    if len(true_labels) == 0:
        return float("nan")
    accuracy_value = accuracy_score(true_labels, predicted_labels)
    return float(accuracy_value)


def predict_assignments(
    assignments: pd.DataFrame,
    predict_fn: PredictFn,
) -> pd.DataFrame:
    """
    Run model inference on the Low / Medium / High text of every ELIGIBLE
    sample and attach predictions.

    Only eligible samples (category != Failed) are scored, because the pattern
    performance analysis is defined over eligible samples. Each level is
    predicted on the exact perturbed text stored for that level, preserving
    row order so predictions align with ids and labels.

    All eligible texts of a level are predicted in ONE ``predict_fn`` call, so
    preprocessing/inference is identical across levels.

    Returns a copy of the eligible rows with added integer columns:
        pred_low, pred_medium, pred_high
    """
    eligible_mask = assignments["category"] != CATEGORY_FAILED
    eligible = assignments[eligible_mask].copy()

    if eligible.empty:
        for level in LEVELS:
            eligible[f"pred_{level}"] = pd.Series(dtype="int64")
        return eligible

    eligible = eligible.reset_index(drop=True)
    for level in LEVELS:
        text_column_name = f"text_{level}"
        texts_for_level = eligible[text_column_name].astype(str).tolist()

        predictions_for_level = list(predict_fn(texts_for_level))
        if len(predictions_for_level) != len(texts_for_level):
            raise RuntimeError(
                f"predict_fn returned {len(predictions_for_level)} predictions "
                f"for {len(texts_for_level)} texts at level '{level}'"
            )

        eligible[f"pred_{level}"] = np.asarray(predictions_for_level, dtype=int)

    return eligible


def pattern_metrics(predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(domain, pattern, level) metrics on the predicted eligible set.

    For each group the same sample ids and ground-truth labels are used across
    Low / Medium / High (the group is defined once, then each level's
    predictions are scored against the identical label vector).

    Columns: domain, pattern, level, n, n_label0, n_label1, accuracy, macro_f1
    """
    metric_columns = [
        "domain", "pattern", "level", "n",
        "n_label0", "n_label1", "accuracy", "macro_f1",
    ]
    if predicted.empty:
        return pd.DataFrame(columns=metric_columns)

    metric_rows: List[dict] = []
    domains = sorted(predicted["domain"].unique().tolist())
    for domain in domains:
        domain_predictions = predicted[predicted["domain"] == domain]

        for pattern in PATTERNS:
            pattern_group = domain_predictions[
                domain_predictions["category"] == pattern
            ]
            group_size = int(len(pattern_group))
            true_labels = pattern_group["label"].astype(int).tolist()
            count_label0 = int((pattern_group["label"] == 0).sum())
            count_label1 = int((pattern_group["label"] == 1).sum())

            for level in LEVELS:
                if group_size == 0:
                    accuracy_value = float("nan")
                    macro_f1_value = float("nan")
                else:
                    predicted_labels = (
                        pattern_group[f"pred_{level}"].astype(int).tolist()
                    )
                    accuracy_value = compute_accuracy(true_labels, predicted_labels)
                    macro_f1_value = compute_macro_f1(true_labels, predicted_labels)

                metric_rows.append({
                    "domain": domain,
                    "pattern": pattern,
                    "level": level,
                    "n": group_size,
                    "n_label0": count_label0,
                    "n_label1": count_label1,
                    "accuracy": accuracy_value,
                    "macro_f1": macro_f1_value,
                })

    return pd.DataFrame(metric_rows)


def determinism_checks(predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Verify prediction-level consistency implied by each pattern.

    Under identical preprocessing and deterministic inference, equal input
    text MUST yield equal predictions. For each eligible sample we check the
    equalities implied by its pattern:

        L=M=H   -> pred_low == pred_medium == pred_high
        M=H!=L  -> pred_medium == pred_high
        L=M!=H  -> pred_low == pred_medium
        L=H!=M  -> pred_low == pred_high
        L!=M!=H -> (no equality constraint)

    Returns a per-(domain, pattern) table with the number of samples checked
    and the number of violations. Any violation > 0 indicates a determinism /
    alignment / normalisation problem that must be investigated, NOT silently
    suppressed.
    """
    determinism_columns = [
        "domain", "pattern", "n_checked", "n_violations", "ok",
    ]
    if predicted.empty:
        return pd.DataFrame(columns=determinism_columns)

    # For each pattern, list the level-pairs whose predictions MUST be equal.
    implied_equal_pairs_by_pattern: Dict[str, List[Tuple[str, str]]] = {
        PATTERN_LMH_EQUAL: [("low", "medium"), ("medium", "high")],
        PATTERN_MH_EQ_L_DIFF: [("medium", "high")],
        PATTERN_LM_EQ_H_DIFF: [("low", "medium")],
        PATTERN_LH_EQ_M_DIFF: [("low", "high")],
        PATTERN_ALL_DIFF: [],
    }

    determinism_rows: List[dict] = []
    domains = sorted(predicted["domain"].unique().tolist())
    for domain in domains:
        domain_predictions = predicted[predicted["domain"] == domain]

        for pattern in implied_equal_pairs_by_pattern:
            required_equal_pairs = implied_equal_pairs_by_pattern[pattern]
            pattern_group = domain_predictions[
                domain_predictions["category"] == pattern
            ]
            group_size = int(len(pattern_group))

            # Nothing to check when the pattern implies no equality or the
            # group is empty.
            if len(required_equal_pairs) == 0 or group_size == 0:
                determinism_rows.append({
                    "domain": domain,
                    "pattern": pattern,
                    "n_checked": group_size,
                    "n_violations": 0,
                    "ok": True,
                })
                continue

            number_of_violations = 0
            for _, sample_row in pattern_group.iterrows():
                sample_has_violation = False
                for first_level, second_level in required_equal_pairs:
                    first_prediction = int(sample_row[f"pred_{first_level}"])
                    second_prediction = int(sample_row[f"pred_{second_level}"])
                    if first_prediction != second_prediction:
                        sample_has_violation = True
                        break  # one violation is enough to flag this sample
                if sample_has_violation:
                    number_of_violations = number_of_violations + 1

            determinism_rows.append({
                "domain": domain,
                "pattern": pattern,
                "n_checked": group_size,
                "n_violations": int(number_of_violations),
                "ok": (number_of_violations == 0),
            })

    return pd.DataFrame(determinism_rows)


# ---------------------------------------------------------------------------
# STAGE 5 - SPARSE PATTERNS & POOLED EXPLORATORY METRICS
# ---------------------------------------------------------------------------

def identify_sparse_patterns(
    distribution: pd.DataFrame,
    min_samples: int = 20,
) -> pd.DataFrame:
    """
    Flag, dynamically, which (domain, pattern) cells are sparse.

    A cell is "sparse" when its eligible-sample count is below ``min_samples``;
    "unobserved" when it is exactly zero. ``min_samples`` is a readability
    threshold for *domain-level* plotting, not a statistical claim.

    Returns: domain, pattern, n, sparse (bool), unobserved (bool)
    """
    # Only the five patterns are considered here, not the Failed category.
    pattern_only_distribution = distribution[~distribution["is_failed"]]

    sparse_rows: List[dict] = []
    for _, distribution_row in pattern_only_distribution.iterrows():
        count = int(distribution_row["n"])
        sparse_rows.append({
            "domain": distribution_row["domain"],
            "pattern": distribution_row["category"],
            "n": count,
            "sparse": count < min_samples,
            "unobserved": count == 0,
        })

    return pd.DataFrame(sparse_rows)


def pooled_sparse_metrics(
    predicted: pd.DataFrame,
    patterns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    OPTIONAL exploratory analysis: pool a pattern's samples ACROSS domains and
    compute one metric set per level.

    This is clearly exploratory. Pooling across domains can conceal
    domain-specific differences, so the pooled sample count and class
    distribution are reported alongside every metric and must be interpreted
    with caution. Patterns are never merged with each other to inflate n.

    Returns: pattern, level, n_pooled, n_label0, n_label1, accuracy, macro_f1,
             n_domains_pooled
    """
    pooled_columns = [
        "pattern", "level", "n_pooled", "n_label0", "n_label1",
        "accuracy", "macro_f1", "n_domains_pooled",
    ]
    if predicted.empty:
        return pd.DataFrame(columns=pooled_columns)

    if patterns is None:
        patterns_to_pool = PATTERNS
    else:
        patterns_to_pool = tuple(patterns)

    pooled_rows: List[dict] = []
    for pattern in patterns_to_pool:
        pooled_group = predicted[predicted["category"] == pattern]
        pooled_size = int(len(pooled_group))
        true_labels = pooled_group["label"].astype(int).tolist()
        count_label0 = int((pooled_group["label"] == 0).sum())
        count_label1 = int((pooled_group["label"] == 1).sum())
        number_of_domains_pooled = int(pooled_group["domain"].nunique())

        for level in LEVELS:
            if pooled_size == 0:
                accuracy_value = float("nan")
                macro_f1_value = float("nan")
            else:
                predicted_labels = (
                    pooled_group[f"pred_{level}"].astype(int).tolist()
                )
                accuracy_value = compute_accuracy(true_labels, predicted_labels)
                macro_f1_value = compute_macro_f1(true_labels, predicted_labels)

            pooled_rows.append({
                "pattern": pattern,
                "level": level,
                "n_pooled": pooled_size,
                "n_label0": count_label0,
                "n_label1": count_label1,
                "accuracy": accuracy_value,
                "macro_f1": macro_f1_value,
                "n_domains_pooled": number_of_domains_pooled,
            })

    return pd.DataFrame(pooled_rows)


# ---------------------------------------------------------------------------
# STAGE 6 - PUBLICATION-QUALITY FIGURES
# ---------------------------------------------------------------------------

# Consistent, colour-blind-friendly palette for the five patterns + Failed.
PATTERN_COLORS: Dict[str, str] = {
    PATTERN_LMH_EQUAL:    "#4C72B0",  # blue
    PATTERN_MH_EQ_L_DIFF: "#55A868",  # green
    PATTERN_LM_EQ_H_DIFF: "#C44E52",  # red
    PATTERN_LH_EQ_M_DIFF: "#8172B3",  # purple
    PATTERN_ALL_DIFF:     "#CCB974",  # gold
    CATEGORY_FAILED:      "#999999",  # grey
}
# Consistent colours for the three intensity levels.
LEVEL_COLORS: Dict[str, str] = {
    "low": "#f39c12",
    "medium": "#e67e22",
    "high": "#e74c3c",
}


def apply_thesis_style() -> None:
    """Set readable, thesis-appropriate matplotlib defaults."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.autolayout": False,
    })


def figure_pattern_distribution(
    distribution: pd.DataFrame,
    save_path: str,
    use_percentage: bool = False,
) -> str:
    """
    Figure 1 - stacked bar chart of the five equality patterns by domain, with
    Failed shown as a separate (grey) stacked segment.

    The denominator is stated in the axis label and title. When
    ``use_percentage`` is True, bars are percentages of all VALID domain
    samples (eligible + failed); otherwise raw counts.
    """
    import matplotlib.pyplot as plt
    apply_thesis_style()

    domains = sorted(distribution["domain"].unique().tolist())
    all_categories = list(PATTERNS) + [CATEGORY_FAILED]

    # Build a value matrix with one row per domain and one column per category.
    number_of_domains = len(domains)
    number_of_categories = len(all_categories)
    value_matrix = np.zeros((number_of_domains, number_of_categories), dtype=float)

    for domain_index in range(number_of_domains):
        domain = domains[domain_index]
        domain_distribution = distribution[distribution["domain"] == domain]

        if len(domain_distribution) > 0:
            denominator_valid = float(domain_distribution["denom_valid"].iloc[0])
        else:
            denominator_valid = 0.0

        for category_index in range(number_of_categories):
            category = all_categories[category_index]
            category_cell = domain_distribution[
                domain_distribution["category"] == category
            ]
            if len(category_cell) > 0:
                count = float(category_cell["n"].iloc[0])
            else:
                count = 0.0

            if use_percentage and denominator_valid > 0:
                value_matrix[domain_index, category_index] = (
                    100.0 * count / denominator_valid
                )
            else:
                value_matrix[domain_index, category_index] = count

    figure, axis = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(number_of_domains)
    stacked_bottoms = np.zeros(number_of_domains)

    for category_index in range(number_of_categories):
        category = all_categories[category_index]
        category_values = value_matrix[:, category_index]
        axis.bar(
            x_positions,
            category_values,
            bottom=stacked_bottoms,
            width=0.68,
            label=category,
            color=PATTERN_COLORS[category],
            edgecolor="white",
            linewidth=0.6,
        )
        stacked_bottoms = stacked_bottoms + category_values

    axis.set_xticks(x_positions)
    axis.set_xticklabels(domains, rotation=0)

    if use_percentage:
        y_axis_label = "Percentage of valid samples (%)"
    else:
        y_axis_label = "Number of valid samples (count)"
    axis.set_ylabel(y_axis_label)
    axis.set_xlabel("Domain")

    denominator_note = (
        "denominator = all valid samples per domain "
        "(eligible + failed; data-quality exclusions removed)"
    )
    axis.set_title(
        "Figure 1. Text-equality pattern distribution by domain\n"
        f"({denominator_note})",
        fontsize=12,
    )
    axis.legend(title="Pattern", bbox_to_anchor=(1.01, 1.0), loc="upper left")
    figure.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figure: %s", save_path)
    return save_path


def figure_main_pattern_macro_f1(
    metrics: pd.DataFrame,
    save_path: str,
    main_patterns: Sequence[str] = MAIN_PATTERNS,
) -> str:
    """
    Figure 2 - pattern-specific Macro-F1 comparison for the main patterns.

    One panel per main pattern; x-axis = domain; grouped bars for Low / Medium
    / High. Sample counts (n) are annotated above each domain group so sparse
    cells cannot masquerade as reliable comparisons.
    """
    import matplotlib.pyplot as plt
    apply_thesis_style()

    main_patterns = list(main_patterns)
    domains = sorted(metrics["domain"].unique().tolist())
    number_of_panels = len(main_patterns)

    figure, axes = plt.subplots(
        1, number_of_panels, figsize=(6.2 * number_of_panels, 5.2), sharey=True,
    )
    # When there is a single panel, matplotlib returns one axis, not a list.
    if number_of_panels == 1:
        axes = [axes]

    bar_width = 0.26
    x_positions = np.arange(len(domains))

    for panel_index in range(number_of_panels):
        pattern = main_patterns[panel_index]
        axis = axes[panel_index]
        pattern_metrics_frame = metrics[metrics["pattern"] == pattern]

        for level_index in range(len(LEVELS)):
            level = LEVELS[level_index]

            macro_f1_values = []
            sample_counts = []
            for domain in domains:
                cell = pattern_metrics_frame[
                    (pattern_metrics_frame["domain"] == domain)
                    & (pattern_metrics_frame["level"] == level)
                ]
                if len(cell) > 0:
                    macro_f1_values.append(float(cell["macro_f1"].iloc[0]))
                    sample_counts.append(int(cell["n"].iloc[0]))
                else:
                    macro_f1_values.append(np.nan)
                    sample_counts.append(0)

            bar_offset = (level_index - 1) * bar_width
            bar_heights = np.nan_to_num(macro_f1_values, nan=0.0)
            axis.bar(
                x_positions + bar_offset,
                bar_heights,
                width=bar_width,
                label=level.capitalize(),
                color=LEVEL_COLORS[level],
                edgecolor="white",
                linewidth=0.5,
            )

            # Annotate the sample count once per domain group (on the High bars).
            if level == "high":
                for domain_index in range(len(domains)):
                    sample_count = sample_counts[domain_index]
                    axis.text(
                        x_positions[domain_index], 1.02,
                        f"n={sample_count}",
                        ha="center", va="bottom",
                        fontsize=8, color="#333333",
                    )

        axis.set_title(pattern)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(domains, rotation=30, ha="right")
        axis.set_ylim(0, 1.12)
        axis.set_xlabel("Domain")

    axes[0].set_ylabel("Macro-F1 (labels=[0,1], macro, zero_division=0)")
    axes[-1].legend(title="Level", bbox_to_anchor=(1.01, 1.0), loc="upper left")
    figure.suptitle(
        "Figure 2. Pattern-specific Macro-F1 across Low / Medium / "
        "High (main patterns)",
        fontsize=13, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figure: %s", save_path)
    return save_path


def figure_per_level_success(
    level_stats: pd.DataFrame,
    save_path: str,
) -> str:
    """
    Additional figure - per-level perturbation success rate by domain.

    Grouped bars: x-axis = domain, one bar per level, height = success rate.
    Failure rate is the complement and is noted in the caption to avoid a
    redundant second chart.
    """
    import matplotlib.pyplot as plt
    apply_thesis_style()

    domains = sorted(level_stats["domain"].unique().tolist())
    bar_width = 0.26
    x_positions = np.arange(len(domains))

    figure, axis = plt.subplots(figsize=(10, 5.5))

    for level_index in range(len(LEVELS)):
        level = LEVELS[level_index]

        success_rate_values = []
        for domain in domains:
            cell = level_stats[
                (level_stats["domain"] == domain)
                & (level_stats["level"] == level)
            ]
            if len(cell) > 0:
                success_rate_values.append(float(cell["success_rate"].iloc[0]))
            else:
                success_rate_values.append(np.nan)

        bar_offset = (level_index - 1) * bar_width
        axis.bar(
            x_positions + bar_offset,
            success_rate_values,
            width=bar_width,
            label=level.capitalize(),
            color=LEVEL_COLORS[level],
            edgecolor="white",
            linewidth=0.5,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(domains)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("In-range (success) rate")
    axis.set_xlabel("Domain")
    axis.set_title(
        "Per-level synonym-perturbation success rate by domain\n"
        "(failure rate = 1 - success rate; "
        "denominator = valid samples per domain)",
        fontsize=12,
    )
    axis.legend(title="Level")
    figure.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figure: %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# PATH RESOLUTION  (maps the generate_all_perturbations layout to level paths)
# ---------------------------------------------------------------------------

def resolve_semantic_level_paths(
    results_root: str,
    domain: str,
    ptype: str = "semantic",
) -> Dict[str, str]:
    """
    Build the {level: path} mapping for the SYNONYM (semantic) perturbation
    CSVs produced by ``EvaluationEngine.generate_all_perturbations``.

    Layout: ``<results_root>/<domain-lowercased>/<ptype>/<level>.csv``.

    ``results_root`` is typically ``/content/drive/MyDrive/thesis/results/<model>``
    on Colab, but any directory following the same layout works.
    """
    domain_folder_name = str(domain).strip().lower()
    domain_directory = os.path.join(results_root, domain_folder_name, ptype)

    level_paths: Dict[str, str] = {}
    for level in LEVELS:
        level_paths[level] = os.path.join(domain_directory, f"{level}.csv")
    return level_paths


def make_checkpoint_predict_fn(
    checkpoint_dir: str,
    model_name: str,
    domain: str,
    full_model_name: str = "indobenchmark/indobert-base-p1",
    max_length: int = 128,
    device: Optional[str] = None,
    random_seed: int = 42,
    batch_size: int = 32,
) -> PredictFn:
    """
    Build a ``predict_fn`` backed by the EXISTING trained specialist checkpoint.

    This reuses the project's ``ModelTrainer`` unchanged - it loads
    ``<checkpoint_dir>/<model_name>/<Domain.capitalize()>/best_model.pt`` and
    returns a callable ``texts -> predicted int labels`` using the exact
    tokenizer / preprocessing / inference path already in the codebase.

    The model is loaded ONCE and captured in the closure, so repeated calls
    (one per level) do not reload weights. No training occurs.

    Import of ``model_trainer`` is deferred so that the pure analysis functions
    in this module can be used without torch installed.
    """
    from model_trainer import ModelTrainer  # deferred import (needs torch)

    capitalised_domain = str(domain).capitalize()
    checkpoint_path = os.path.join(
        checkpoint_dir, model_name, capitalised_domain, "best_model.pt"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found for {model_name}/{domain}: {checkpoint_path}"
        )

    trainer = ModelTrainer(
        model_name=full_model_name,
        max_length=max_length,
        device=device,
        random_seed=random_seed,
    )
    trainer.load_checkpoint(checkpoint_path)
    logger.info("Loaded checkpoint for predict_fn: %s", checkpoint_path)

    def predict_texts(texts: List[str]) -> List[int]:
        if len(texts) == 0:
            return []

        predicted_labels, _probabilities = trainer.predict(
            texts, batch_size=batch_size
        )

        predicted_labels_flat = np.asarray(predicted_labels).ravel().tolist()
        predicted_labels_as_int = []
        for predicted_label in predicted_labels_flat:
            predicted_labels_as_int.append(int(predicted_label))
        return predicted_labels_as_int

    return predict_texts


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def build_report(
    distribution: pd.DataFrame,
    level_stats: pd.DataFrame,
    metrics: pd.DataFrame,
    determinism: pd.DataFrame,
    reconciliation: pd.DataFrame,
    dq_summary: pd.DataFrame,
    sparse: pd.DataFrame,
    model_name: str,
) -> str:
    """Assemble a concise plain-text findings + limitations report.

    We build the report as a list of text lines and join them at the end.
    Each ``report_lines.append(...)`` adds one printed line.
    """
    report_lines: List[str] = []

    report_lines.append("=" * 78)
    report_lines.append("PATTERN-BASED ROBUSTNESS ANALYSIS — FINDINGS & LIMITATIONS")
    report_lines.append(
        f"Model: {model_name}   "
        f"Perturbation: SYNONYM (semantic), Low/Medium/High"
    )
    report_lines.append("=" * 78)
    report_lines.append("")

    # ---- 1. data quality -------------------------------------------------
    report_lines.append(
        "1. DATA QUALITY (reported separately; excluded from valid denominators)"
    )
    if dq_summary.empty:
        report_lines.append(
            "   No data-quality issues detected (no missing/duplicate ids, "
            "missing text, or inconsistent labels)."
        )
    else:
        distinct_ids_per_issue = dq_summary.groupby("issue")["id"].nunique()
        for issue_name, distinct_id_count in distinct_ids_per_issue.items():
            report_lines.append(
                f"   - {issue_name}: {int(distinct_id_count)} distinct id(s)"
            )
    report_lines.append("")

    # ---- 2. reconciliation ----------------------------------------------
    if reconciliation.empty:
        all_domains_reconcile = True
    else:
        all_domains_reconcile = bool(reconciliation["ok"].all())

    report_lines.append("2. COUNT RECONCILIATION")
    report_lines.append(
        "   All domains reconcile (valid == aligned == sum of categories): "
        f"{all_domains_reconcile}"
    )
    for _, reconciliation_row in reconciliation.iterrows():
        report_lines.append(
            f"   - {reconciliation_row['domain']}: "
            f"valid={reconciliation_row['n_valid']} "
            f"(eligible={reconciliation_row['n_eligible']}, "
            f"failed={reconciliation_row['n_failed']}), "
            f"data-quality ids excluded="
            f"{reconciliation_row['n_data_quality_ids_excluded']}, "
            f"ok={reconciliation_row['ok']}"
        )
    report_lines.append("")

    # ---- 3. distribution -------------------------------------------------
    report_lines.append(
        "3. PATTERN DISTRIBUTION (per domain; denominator = valid samples)"
    )
    distribution_domains = sorted(distribution["domain"].unique().tolist())
    for domain in distribution_domains:
        domain_distribution = distribution[distribution["domain"] == domain]
        if len(domain_distribution) > 0:
            denominator_valid = int(domain_distribution["denom_valid"].iloc[0])
        else:
            denominator_valid = 0
        report_lines.append(f"   [{domain}] valid={denominator_valid}")

        for _, distribution_row in domain_distribution.iterrows():
            percentage_of_valid = distribution_row["pct_of_valid"]
            if pd.notna(percentage_of_valid):
                percentage_text = f"{percentage_of_valid:5.1f}%"
            else:
                percentage_text = "   NA"
            report_lines.append(
                f"       {distribution_row['category']:<9} "
                f"n={int(distribution_row['n']):>4}  "
                f"({percentage_text} of valid; "
                f"label0={int(distribution_row['n_label0'])}, "
                f"label1={int(distribution_row['n_label1'])})"
            )
    report_lines.append("")

    # ---- 4. per-level success -------------------------------------------
    report_lines.append("4. PER-LEVEL PERTURBATION SUCCESS (independent per level)")
    level_stats_domains = sorted(level_stats["domain"].unique().tolist())
    for domain in level_stats_domains:
        domain_level_stats = level_stats[level_stats["domain"] == domain]
        level_summary_parts = []
        for _, level_row in domain_level_stats.iterrows():
            success_rate = level_row["success_rate"]
            if pd.notna(success_rate):
                success_rate_text = f"{success_rate * 100:4.1f}%"
            else:
                success_rate_text = "NA"
            level_summary_parts.append(
                f"{level_row['level']}="
                f"{level_row['n_success']}/{level_row['n_valid']} "
                f"({success_rate_text})"
            )
        report_lines.append(f"   [{domain}] " + "  ".join(level_summary_parts))
    report_lines.append("")

    # ---- 5. sparse patterns ---------------------------------------------
    report_lines.append(
        "5. SPARSE / UNOBSERVED PATTERNS (dynamic; readability threshold)"
    )
    if sparse.empty:
        report_lines.append("   (no distribution available)")
    else:
        sparse_cells = sparse[sparse["sparse"]]
        unobserved_cells = sparse[sparse["unobserved"]]
        report_lines.append(
            f"   Unobserved (n=0) cells: {len(unobserved_cells)}; "
            f"sparse (below threshold) cells: {len(sparse_cells)}."
        )
        report_lines.append(
            "   Domain-level performance graphs are suppressed for sparse "
            "cells; optional pooled metrics are exploratory only."
        )
    report_lines.append("")

    # ---- 6. main-pattern metrics ----------------------------------------
    report_lines.append(
        "6. PATTERN-SPECIFIC PERFORMANCE (main patterns; report n + class split)"
    )

    metrics_are_available = True
    if metrics is None:
        metrics_are_available = False
    elif metrics.empty:
        metrics_are_available = False
    elif "pattern" not in metrics.columns:
        metrics_are_available = False

    if not metrics_are_available:
        report_lines.append(
            "   (model evaluation was skipped -> no pattern-specific metrics; "
            "re-run with a checkpoint to populate this section.)"
        )
        report_lines.append("")
        patterns_to_report = []
        main_pattern_metrics = metrics
    else:
        patterns_to_report = list(MAIN_PATTERNS)
        main_pattern_metrics = metrics[metrics["pattern"].isin(MAIN_PATTERNS)]

    for pattern in patterns_to_report:
        report_lines.append(f"   Pattern {pattern}:")
        this_pattern_metrics = main_pattern_metrics[
            main_pattern_metrics["pattern"] == pattern
        ]
        pattern_domains = sorted(this_pattern_metrics["domain"].unique().tolist())
        for domain in pattern_domains:
            domain_cell = this_pattern_metrics[
                this_pattern_metrics["domain"] == domain
            ]
            if domain_cell.empty:
                continue

            group_size = int(domain_cell["n"].iloc[0])
            count_label0 = int(domain_cell["n_label0"].iloc[0])
            count_label1 = int(domain_cell["n_label1"].iloc[0])

            macro_f1_parts = []
            for level in LEVELS:
                level_cell = domain_cell[domain_cell["level"] == level]
                if len(level_cell) > 0:
                    macro_f1_value = float(level_cell["macro_f1"].iloc[0])
                else:
                    macro_f1_value = float("nan")

                level_initial = level[:1].upper()
                if pd.notna(macro_f1_value):
                    macro_f1_parts.append(f"{level_initial}={macro_f1_value:.3f}")
                else:
                    macro_f1_parts.append(f"{level_initial}=NA")

            report_lines.append(
                f"       {domain:<11} n={group_size:>4} "
                f"(0:{count_label0}/1:{count_label1})  "
                f"macroF1 " + " ".join(macro_f1_parts)
            )
    report_lines.append("")

    # ---- 7. determinism --------------------------------------------------
    report_lines.append("7. DETERMINISM / CONSISTENCY CHECKS")

    determinism_available = True
    if determinism is None:
        determinism_available = False
    elif determinism.empty:
        determinism_available = False
    elif "n_violations" not in determinism.columns:
        determinism_available = False

    if not determinism_available:
        report_lines.append(
            "   (model evaluation was skipped -> no determinism checks run.)"
        )
        report_lines.append("")
        violating_rows = pd.DataFrame(
            columns=["domain", "pattern", "n_violations", "n_checked"]
        )
    else:
        violating_rows = determinism[determinism["n_violations"] > 0]

    checks_ran = determinism_available
    no_violations_found = violating_rows.empty
    if checks_ran and no_violations_found:
        report_lines.append(
            "   PASS: for every pattern, equal-text levels produced equal "
            "predictions (0 violations). This confirms deterministic, aligned "
            "inference."
        )
    else:
        report_lines.append(
            "   WARNING: prediction-equality violations detected — investigate "
            "normalisation / row alignment / checkpoint / inference determinism:"
        )
        for _, violation_row in violating_rows.iterrows():
            report_lines.append(
                f"     - {violation_row['domain']}/{violation_row['pattern']}: "
                f"{int(violation_row['n_violations'])}/"
                f"{int(violation_row['n_checked'])} samples"
            )
    report_lines.append("")

    # ---- 8. limitations --------------------------------------------------
    report_lines.append("8. LIMITATIONS & INTERPRETATION")
    report_lines.append(
        "   - Patterns use EXACT STRING EQUALITY. Different strings do not "
        "necessarily differ in meaning, and 'different' does not imply a larger "
        "or monotonically increasing semantic perturbation."
    )
    report_lines.append(
        "   - L=M=H is primarily a CONSISTENCY CHECK (identical inputs must give "
        "identical outputs); matching metrics there are expected and are NOT "
        "evidence of robustness to changed text."
    )
    report_lines.append(
        "   - Sample sizes differ across domains and patterns; this is accepted. "
        "Small groups are interpreted cautiously and n + class distribution are "
        "always reported. Small numeric gaps are not treated as meaningful "
        "without regard to n, class balance, and uncertainty."
    )
    report_lines.append(
        "   - Pooled sparse-pattern metrics (if used) can hide domain "
        "differences and are exploratory only."
    )
    report_lines.append(
        "   - These are descriptive comparisons; no causal claim is made."
    )
    report_lines.append("=" * 78)

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

@dataclass
class AnalysisPaths:
    """Resolved output file locations for one analysis run."""
    base_dir: str
    tables_dir: str
    figures_dir: str
    report_path: str


def prepare_output_dirs(output_dir: str) -> AnalysisPaths:
    """Create the output sub-directories and return their resolved paths."""
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "pattern_analysis_report.txt")
    return AnalysisPaths(
        base_dir=output_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        report_path=report_path,
    )


def run_pattern_analysis(
    results_root: str,
    output_dir: str,
    predict_fn_factory: Optional[Callable[[str], PredictFn]] = None,
    domains: Optional[Sequence[str]] = None,
    model_name: str = "base",
    ptype: str = "semantic",
    sparse_min_samples: int = 20,
    make_figures: bool = True,
    distribution_as_percentage: bool = False,
) -> Dict[str, object]:
    """
    End-to-end pattern-based robustness analysis.

    Parameters
    ----------
    results_root:
        Directory containing ``<domain>/<ptype>/<level>.csv`` (the output of
        ``EvaluationEngine.generate_all_perturbations`` for one model, e.g.
        ``/content/drive/MyDrive/thesis/results/base``).
    output_dir:
        Directory to write all tables, figures, and the report. Created if
        missing. Existing source data / checkpoints / results are never
        overwritten.
    predict_fn_factory:
        Callable ``domain -> predict_fn``. If ``None``, model evaluation is
        skipped (distribution + data-quality still run). Use
        :func:`make_checkpoint_predict_fn` bound per domain on Colab.
    domains:
        Which domains to process. Defaults to the five canonical domains that
        have a directory under ``results_root``.
    model_name:
        Label recorded in the report (e.g. ``"base"``, ``"large"``).

    Returns
    -------
    dict of all computed DataFrames + the written file paths.
    """
    paths = prepare_output_dirs(output_dir)

    # Decide which domains to process. By default we take every canonical
    # domain that actually has a <domain>/<ptype> directory on disk.
    if domains is None:
        discovered_domains = []
        for canonical_domain in DOMAIN_FILES:
            domain_folder_name = canonical_domain.lower()
            candidate_directory = os.path.join(
                results_root, domain_folder_name, ptype
            )
            if os.path.isdir(candidate_directory):
                discovered_domains.append(canonical_domain)
        domains = discovered_domains

        if len(domains) == 0:
            raise FileNotFoundError(
                f"No <domain>/{ptype} directories found under {results_root}. "
                f"Expected e.g. {results_root}/technology/{ptype}/low.csv"
            )
    logger.info("Domains to analyse: %s", list(domains))

    # --- STAGE 1-2: load, align, classify, one domain at a time -----------
    frames: Dict[str, DomainFrame] = {}
    assignment_frames: List[pd.DataFrame] = []
    for domain in domains:
        level_paths = resolve_semantic_level_paths(results_root, domain, ptype)
        frame = load_and_align_domain(domain, level_paths)
        frames[domain] = frame
        assignment_frames.append(assign_patterns(frame))

    if len(assignment_frames) > 0:
        assignments = pd.concat(assignment_frames, ignore_index=True)
    else:
        assignments = pd.DataFrame()

    # --- STAGE 3: distribution + per-level + data-quality + reconcile -----
    distribution = pattern_distribution(assignments)
    level_stats = per_level_success_stats(assignments, frames)
    dq_summary = data_quality_summary(frames)
    reconciliation = reconcile_counts(frames, assignments)
    sparse = identify_sparse_patterns(distribution, min_samples=sparse_min_samples)

    # --- STAGE 4: model evaluation (only when a predictor is provided) ----
    predicted = pd.DataFrame()
    metrics = pd.DataFrame()
    determinism = pd.DataFrame()
    pooled = pd.DataFrame()

    if predict_fn_factory is not None:
        predicted_parts: List[pd.DataFrame] = []
        for domain in domains:
            domain_assignments = assignments[assignments["domain"] == domain]
            if domain_assignments.empty:
                continue
            predict_fn = predict_fn_factory(domain)
            domain_predictions = predict_assignments(domain_assignments, predict_fn)
            predicted_parts.append(domain_predictions)

        if len(predicted_parts) > 0:
            predicted = pd.concat(predicted_parts, ignore_index=True)
            metrics = pattern_metrics(predicted)
            determinism = determinism_checks(predicted)
            pooled = pooled_sparse_metrics(predicted)
    else:
        logger.warning(
            "predict_fn_factory is None -> skipping model evaluation; "
            "only distribution/data-quality outputs will be produced."
        )

    # --- STAGE 5: write tables -------------------------------------------
    written: Dict[str, str] = {}

    def save_table(dataframe_to_save: pd.DataFrame, file_name: str) -> None:
        if dataframe_to_save is None:
            return
        table_path = os.path.join(paths.tables_dir, file_name)
        dataframe_to_save.to_csv(table_path, index=False)
        written[file_name] = table_path
        logger.info("Saved table: %s (%d rows)", table_path, len(dataframe_to_save))

    save_table(assignments, "per_sample_pattern_assignments.csv")
    save_table(distribution, "pattern_distribution_by_domain.csv")
    save_table(level_stats, "per_level_success_failure.csv")
    save_table(dq_summary, "data_quality_issues.csv")
    save_table(reconciliation, "reconciliation_checks.csv")
    save_table(sparse, "sparse_pattern_flags.csv")

    if not metrics.empty:
        save_table(metrics, "pattern_specific_metrics.csv")
    if not determinism.empty:
        save_table(determinism, "determinism_checks.csv")
    if not pooled.empty:
        save_table(pooled, "pooled_sparse_exploratory_metrics.csv")

    if not predicted.empty:
        # Detailed per-sample predictions (retain ids, labels, pattern, texts,
        # and L/M/H predictions so metrics are fully reproducible).
        prediction_columns_to_keep = [
            "domain", "id", "label", "category",
            "text_low", "text_medium", "text_high",
            "pred_low", "pred_medium", "pred_high",
        ]
        save_table(
            predicted[prediction_columns_to_keep],
            "per_sample_predictions.csv",
        )

    # --- STAGE 6: figures -------------------------------------------------
    figure_paths: Dict[str, str] = {}
    if make_figures and not distribution.empty:
        try:
            figure1_path = os.path.join(
                paths.figures_dir, "fig1_pattern_distribution.png"
            )
            figure_paths["figure1_pattern_distribution"] = (
                figure_pattern_distribution(
                    distribution,
                    figure1_path,
                    use_percentage=distribution_as_percentage,
                )
            )

            per_level_figure_path = os.path.join(
                paths.figures_dir, "fig_per_level_success.png"
            )
            figure_paths["figure_per_level_success"] = figure_per_level_success(
                level_stats,
                per_level_figure_path,
            )

            if not metrics.empty:
                figure2_path = os.path.join(
                    paths.figures_dir, "fig2_main_pattern_macro_f1.png"
                )
                figure_paths["figure2_main_pattern_macro_f1"] = (
                    figure_main_pattern_macro_f1(metrics, figure2_path)
                )
        except Exception as figure_error:  # figures are non-fatal
            logger.warning(
                "Figure generation failed (non-fatal): %s", figure_error
            )

    # --- STAGE 7: report --------------------------------------------------
    report = build_report(
        distribution=distribution,
        level_stats=level_stats,
        metrics=metrics,
        determinism=determinism,
        reconciliation=reconciliation,
        dq_summary=dq_summary,
        sparse=sparse,
        model_name=model_name,
    )
    with open(paths.report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    written["pattern_analysis_report.txt"] = paths.report_path
    logger.info("Saved report: %s", paths.report_path)

    # Machine-readable manifest of everything written.
    model_was_evaluated = predict_fn_factory is not None
    manifest = {
        "model_name": model_name,
        "results_root": results_root,
        "output_dir": output_dir,
        "domains": list(domains),
        "tables": written,
        "figures": figure_paths,
        "model_evaluated": model_was_evaluated,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    return {
        "assignments": assignments,
        "distribution": distribution,
        "level_stats": level_stats,
        "data_quality": dq_summary,
        "reconciliation": reconciliation,
        "sparse": sparse,
        "predicted": predicted,
        "metrics": metrics,
        "determinism": determinism,
        "pooled": pooled,
        "report": report,
        "written": written,
        "figures": figure_paths,
        "manifest": manifest,
    }


# ===========================================================================
# ===========================================================================
# SUCCESS-BASED PATTERN ANALYSIS  (Report A: Validity  +  Report B: Patterns)
# ===========================================================================
# ===========================================================================
#
# This section implements the thesis requirement of STRICTLY SEPARATING two
# questions that must never be mixed:
#
#   Report A  "Perturbation Validity"
#       For each severity level (Low, Medium, High) INDEPENDENTLY, how many
#       samples met the perturbation validity criteria, how many failed, and
#       (when available) why they failed. Failures are REPORTED, never hidden.
#
#   Report B  "Pattern Distribution & Evaluation"
#       The exact text-equality patterns (L=M=H, M=H!=L, ...) and the model
#       metrics (Accuracy, Macro-F1) computed STRICTLY on the subset of ids
#       that were successfully perturbed at ALL THREE levels and therefore can
#       be aligned by the original headline id.
#
# "Successfully perturbed" uses the engine's own definition, the
# ``perturbation_success`` column, which is True only when a row is changed AND
# its word-cosine is in [0.80, 0.95] AND it hit its token target
# (words_changed == target_count). See perturbation_engine.py.
#
# The whole section is written with plain ``for`` loops and descriptive names
# so every line can be explained during a thesis defence.


# ---------------------------------------------------------------------------
# SIMPLE ENGINE -> ANALYSIS HANDOFF  (one plain DataFrame per level)
# ---------------------------------------------------------------------------
#
# The perturbation engine already saves everything we need to a plain CSV at
#     <results_root>/<domain>/semantic/<level>.csv
# with explicit column names. For this analysis we only read four of them:
#
#     id                    - original headline id (stable sample identifier)
#     label                 - ground-truth label (0 or 1)
#     perturbed_text        - the perturbed headline for this level
#     perturbation_success  - True only when the row was changed AND its
#                             word-cosine is in [0.80, 0.95] AND it hit its
#                             token target (words_changed == target_count)
#
# That is the entire handoff: pandas reads the CSV, we keep those four columns,
# and every later step works on a standard DataFrame. No wrapper classes, no
# column-guessing, no intermediate "tidy" objects.

SUCCESS_ANALYSIS_COLUMNS = ["id", "label", "perturbed_text", "perturbation_success"]


def read_level_csv(path: str) -> pd.DataFrame:
    """
    Read ONE perturbed-level CSV and return a plain DataFrame holding just the
    four columns the success-based analysis needs.

    Failed perturbations are KEPT here (Report A needs them); they are only
    dropped later, for Report B, by the id-alignment step.
    """
    frame = pd.read_csv(path)

    missing_columns = []
    for column_name in SUCCESS_ANALYSIS_COLUMNS:
        if column_name not in frame.columns:
            missing_columns.append(column_name)
    if missing_columns:
        raise KeyError(
            f"{path}: missing expected column(s) {missing_columns}. "
            f"Available columns: {list(frame.columns)}"
        )

    slim_frame = frame[SUCCESS_ANALYSIS_COLUMNS].copy()
    # Make the success flag a plain True/False column (treat missing as False).
    slim_frame["perturbation_success"] = (
        slim_frame["perturbation_success"].fillna(False).astype(bool)
    )
    return slim_frame


# ---------------------------------------------------------------------------
# STANDARDIZED "COMMON ID" ALIGNMENT  (one method for the whole project)
# ---------------------------------------------------------------------------
#
# This is the single, defensible way the project aligns samples across the
# Low / Medium / High perturbation levels. Every module (this analysis AND the
# evaluation engine) calls the SAME function so the methodology is identical
# everywhere and can be explained in one place during the defence.
#
# The rule, stated plainly:
#   1. At each level, a sample "counts" only if it was SUCCESSFULLY perturbed
#      (perturbation_success == True).
#   2. We then keep only the ids that succeeded at ALL THREE levels
#      (an intersection of the per-level successful-id sets).
#   3. Matching is always by the original headline ``id`` - never by row
#      position - so the same sample is compared across levels.
#
# Two behaviours are deliberately OPTIONAL and OFF by default, so this function
# reproduces the evaluation engine's existing results exactly:
#   * drop_duplicate_ids  - if True, an id that appears more than once inside a
#                           single level is removed entirely (we cannot know
#                           which duplicate row is correct). The engine does
#                           NOT do this today, so it defaults to False.
#   * sort_result         - if True, return a deterministically ordered list
#                           (sorted by str). The engine uses the ids as a set
#                           (membership only), so ordering does not affect its
#                           numbers; it defaults to False here and the engine
#                           ignores order anyway.


def successful_ids_for_level(level_frame: pd.DataFrame) -> set:
    """
    Return the SET of ids that were successfully perturbed in ONE level frame.

    This mirrors the evaluation engine's original ``_successful_ids`` exactly:
    take the rows whose ``perturbation_success`` is True and collect their
    ``id`` values into a set. No duplicate handling, no ordering - identical to
    the engine's current behaviour.
    """
    if "id" not in level_frame.columns:
        raise ValueError("Cannot align by id: 'id' column is missing.")
    if "perturbation_success" not in level_frame.columns:
        raise ValueError(
            "Cannot align by success: 'perturbation_success' column is missing."
        )

    success_flag = level_frame["perturbation_success"].astype(bool)
    successful_rows = level_frame.loc[success_flag]
    return set(successful_rows["id"].tolist())


def count_duplicate_ids_per_level(
    level_frame_by_level: Dict[str, pd.DataFrame],
) -> Dict[str, int]:
    """
    Report (do NOT drop) how many ids are duplicated within each level.

    This is purely diagnostic: it lets us SEE whether duplicate ids exist and
    how many, so their impact can be assessed separately before deciding to
    change any behaviour. It never alters the data.

    Returns a mapping ``{level: number_of_ids_that_appear_more_than_once}``.
    """
    duplicate_count_by_level: Dict[str, int] = {}
    for level in LEVELS:
        level_frame = level_frame_by_level[level]
        id_value_counts = level_frame["id"].value_counts()

        number_of_duplicated_ids = 0
        for id_value in id_value_counts.index:
            how_many_times = int(id_value_counts[id_value])
            if how_many_times > 1:
                number_of_duplicated_ids = number_of_duplicated_ids + 1

        duplicate_count_by_level[level] = number_of_duplicated_ids
    return duplicate_count_by_level


def align_common_successful_ids(
    level_frame_by_level: Dict[str, pd.DataFrame],
    drop_duplicate_ids: bool = False,
    sort_result: bool = False,
):
    """
    The standardized "common id" alignment used across the whole project.

    Steps (explicit, so each can be explained):
      1. For every level, collect the set of successfully-perturbed ids.
      2. Optionally drop ids that are duplicated within a level (OFF by default
         to preserve the evaluation engine's existing results).
      3. Intersect the per-level sets so only ids successful at ALL THREE
         levels remain.
      4. Return either a set (default, engine-compatible) or, if requested, a
         deterministically sorted list.

    Parameters
    ----------
    level_frame_by_level:
        Mapping ``{"low": df, "medium": df, "high": df}``. Each df must have an
        ``id`` column and a ``perturbation_success`` column.
    drop_duplicate_ids:
        If True, remove any id that appears more than once inside a single
        level before intersecting. Default False (engine-preserving).
    sort_result:
        If True, return a list ordered by ``sorted(..., key=str)``. Default
        False, which returns a plain set (what the engine uses).
    """
    successful_ids_by_level: Dict[str, set] = {}
    for level in LEVELS:
        level_frame = level_frame_by_level[level]

        if drop_duplicate_ids:
            # Remove every copy of any id that is duplicated within this level.
            deduplicated_frame = level_frame.drop_duplicates(
                subset="id", keep=False
            )
            successful_ids_by_level[level] = successful_ids_for_level(
                deduplicated_frame
            )
        else:
            successful_ids_by_level[level] = successful_ids_for_level(level_frame)

    # Intersect across the three levels with an explicit loop.
    common_successful_ids = successful_ids_by_level[LEVELS[0]]
    for level in LEVELS[1:]:
        common_successful_ids = common_successful_ids & successful_ids_by_level[level]

    if sort_result:
        return sorted(common_successful_ids, key=str)
    return common_successful_ids


# ---------------------------------------------------------------------------
# REPORT A - PERTURBATION VALIDITY  (independent per level, failures kept)
# ---------------------------------------------------------------------------

def print_report_a_validity(validity_table: pd.DataFrame) -> None:
    """
    Print REPORT A directly in the notebook: per-level perturbation validity,
    with success rates AND failure counts. Failures are shown, never hidden.
    """
    print("=" * 78)
    print("REPORT A - PERTURBATION VALIDITY (independent per severity level)")
    print("=" * 78)
    print(
        "A sample is VALID at a level when it was changed, its word-cosine is "
        "in [0.80, 0.95], and it hit its token target (10/20/30%)."
    )
    print(
        "Failures are reported here, NOT discarded. They are only excluded from "
        "Report B's matched subset."
    )
    print("")

    if validity_table.empty:
        print("   (no validity data available)")
        print("")
        return

    domains = sorted(validity_table["domain"].unique().tolist())
    for domain in domains:
        print(f"[{domain}]")
        domain_rows = validity_table[validity_table["domain"] == domain]
        for level in LEVELS:
            level_rows = domain_rows[domain_rows["level"] == level]
            if level_rows.empty:
                continue
            level_row = level_rows.iloc[0]

            number_of_successful_rows = int(level_row["n_success"])
            number_of_total_rows = int(level_row["n_total"])
            number_of_failed_rows = int(level_row["n_fail"])
            success_rate_percentage = float(level_row["success_rate"]) * 100.0

            print(
                f"   {level.capitalize():<7} "
                f"success = {number_of_successful_rows:>4}/{number_of_total_rows:<4} "
                f"({success_rate_percentage:5.1f}%)   "
                f"failed = {number_of_failed_rows:>4}"
            )
        print("")


# ---------------------------------------------------------------------------
# REPORT B - PATTERN DISTRIBUTION & EVALUATION  (matched successful subset)
# ---------------------------------------------------------------------------

def build_success_pattern_distribution(classified_by_domain: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Per-(domain, pattern) distribution over the MATCHED SUCCESSFUL subset.

    Every one of the five patterns is listed for every domain, even at zero, so
    the table shape is stable. There is no Failed category here on purpose.

    Columns: domain, pattern, n, n_label0, n_label1, pct_of_matched,
             denom_matched
    """
    distribution_rows: List[dict] = []
    domains = sorted(classified_by_domain.keys())
    for domain in domains:
        classified_table = classified_by_domain[domain]
        number_of_matched_samples = int(len(classified_table))

        for pattern in PATTERNS:
            if classified_table.empty:
                pattern_rows = classified_table
            else:
                pattern_rows = classified_table[classified_table["pattern"] == pattern]

            count = int(len(pattern_rows))
            if count > 0:
                count_label0 = int((pattern_rows["label"] == 0).sum())
                count_label1 = int((pattern_rows["label"] == 1).sum())
            else:
                count_label0 = 0
                count_label1 = 0

            if number_of_matched_samples > 0:
                percentage_of_matched = 100.0 * count / number_of_matched_samples
            else:
                percentage_of_matched = np.nan

            distribution_rows.append({
                "domain": domain,
                "pattern": pattern,
                "n": count,
                "n_label0": count_label0,
                "n_label1": count_label1,
                "pct_of_matched": percentage_of_matched,
                "denom_matched": number_of_matched_samples,
            })

    return pd.DataFrame(distribution_rows)


def evaluate_success_patterns(
    classified_by_domain: Dict[str, pd.DataFrame],
    predict_fn_factory: Optional[Callable[[str], PredictFn]],
) -> pd.DataFrame:
    """
    Compute Accuracy and Macro-F1 per (domain, pattern, level) on the MATCHED
    SUCCESSFUL subset only.

    For each domain we build one predict_fn (bound to that domain's specialist)
    and score the Low / Medium / High perturbed text of every matched sample.
    The ground-truth label is identical across levels for a given id.

    Returns columns:
        domain, pattern, level, n, n_label0, n_label1, accuracy, macro_f1
    If ``predict_fn_factory`` is None, an empty frame is returned (Report B then
    shows distribution only).
    """
    metric_columns = [
        "domain", "pattern", "level", "n",
        "n_label0", "n_label1", "accuracy", "macro_f1",
    ]
    if predict_fn_factory is None:
        return pd.DataFrame(columns=metric_columns)

    metric_rows: List[dict] = []
    domains = sorted(classified_by_domain.keys())
    for domain in domains:
        classified_table = classified_by_domain[domain]
        if classified_table.empty:
            continue

        # Build this domain's predictor once, then reuse it for all three levels.
        predict_fn = predict_fn_factory(domain)

        # Predict every level's text for the whole matched subset of this domain.
        predictions_by_level: Dict[str, List[int]] = {}
        for level in LEVELS:
            text_column_name = f"text_{level}"
            texts_for_level = classified_table[text_column_name].astype(str).tolist()
            predictions_for_level = list(predict_fn(texts_for_level))
            if len(predictions_for_level) != len(texts_for_level):
                raise RuntimeError(
                    f"predict_fn returned {len(predictions_for_level)} predictions "
                    f"for {len(texts_for_level)} texts at level '{level}' "
                    f"(domain={domain})"
                )
            predictions_by_level[level] = predictions_for_level

        # Attach predictions back to the table so we can slice per pattern.
        scored_table = classified_table.copy().reset_index(drop=True)
        for level in LEVELS:
            scored_table[f"pred_{level}"] = np.asarray(
                predictions_by_level[level], dtype=int
            )

        # Now compute metrics per pattern and per level.
        for pattern in PATTERNS:
            pattern_group = scored_table[scored_table["pattern"] == pattern]
            group_size = int(len(pattern_group))
            true_labels = pattern_group["label"].astype(int).tolist()
            count_label0 = int((pattern_group["label"] == 0).sum())
            count_label1 = int((pattern_group["label"] == 1).sum())

            for level in LEVELS:
                if group_size == 0:
                    accuracy_value = float("nan")
                    macro_f1_value = float("nan")
                else:
                    predicted_labels = pattern_group[f"pred_{level}"].astype(int).tolist()
                    accuracy_value = compute_accuracy(true_labels, predicted_labels)
                    macro_f1_value = compute_macro_f1(true_labels, predicted_labels)

                metric_rows.append({
                    "domain": domain,
                    "pattern": pattern,
                    "level": level,
                    "n": group_size,
                    "n_label0": count_label0,
                    "n_label1": count_label1,
                    "accuracy": accuracy_value,
                    "macro_f1": macro_f1_value,
                })

    return pd.DataFrame(metric_rows)


def print_report_b_patterns(
    success_distribution: pd.DataFrame,
    success_metrics: pd.DataFrame,
) -> None:
    """
    Print REPORT B directly in the notebook: pattern distribution and model
    metrics, computed STRICTLY on the matched successful subset of ids.
    """
    print("=" * 78)
    print("REPORT B - PATTERN DISTRIBUTION & EVALUATION (matched successful subset)")
    print("=" * 78)
    print(
        "Only ids successfully perturbed at ALL THREE levels are used here, "
        "aligned by the original headline id."
    )
    print("")

    if success_distribution.empty:
        print("   (no matched successful samples -> nothing to report)")
        print("")
        return

    # --- B.1 distribution -------------------------------------------------
    print("B.1  Text-equality pattern distribution (per domain)")
    domains = sorted(success_distribution["domain"].unique().tolist())
    for domain in domains:
        domain_distribution = success_distribution[
            success_distribution["domain"] == domain
        ]
        if len(domain_distribution) > 0:
            denominator_matched = int(domain_distribution["denom_matched"].iloc[0])
        else:
            denominator_matched = 0
        print(f"   [{domain}] matched successful samples = {denominator_matched}")

        for _, distribution_row in domain_distribution.iterrows():
            percentage_of_matched = distribution_row["pct_of_matched"]
            if pd.notna(percentage_of_matched):
                percentage_text = f"{percentage_of_matched:5.1f}%"
            else:
                percentage_text = "   NA"
            print(
                f"       {distribution_row['pattern']:<9} "
                f"n={int(distribution_row['n']):>4}  ({percentage_text})  "
                f"label0={int(distribution_row['n_label0'])}, "
                f"label1={int(distribution_row['n_label1'])}"
            )
        print("")

    # --- B.2 metrics ------------------------------------------------------
    print("B.2  Model metrics on the matched subset (Accuracy / Macro-F1)")
    if success_metrics.empty:
        print(
            "   (model evaluation was skipped -> distribution only. "
            "Provide a predict_fn_factory to populate metrics.)"
        )
        print("")
        return

    metric_domains = sorted(success_metrics["domain"].unique().tolist())
    for domain in metric_domains:
        print(f"   [{domain}]")
        domain_metrics = success_metrics[success_metrics["domain"] == domain]
        for pattern in PATTERNS:
            pattern_metrics_rows = domain_metrics[domain_metrics["pattern"] == pattern]
            if pattern_metrics_rows.empty:
                continue

            group_size = int(pattern_metrics_rows["n"].iloc[0])
            if group_size == 0:
                # Skip patterns that were not observed in this domain.
                continue

            accuracy_parts = []
            macro_f1_parts = []
            for level in LEVELS:
                level_rows = pattern_metrics_rows[pattern_metrics_rows["level"] == level]
                level_initial = level[:1].upper()
                if level_rows.empty:
                    accuracy_parts.append(f"{level_initial}=NA")
                    macro_f1_parts.append(f"{level_initial}=NA")
                    continue

                accuracy_value = float(level_rows["accuracy"].iloc[0])
                macro_f1_value = float(level_rows["macro_f1"].iloc[0])
                accuracy_parts.append(f"{level_initial}={accuracy_value:.3f}")
                macro_f1_parts.append(f"{level_initial}={macro_f1_value:.3f}")

            print(
                f"       {pattern:<9} n={group_size:>4}  "
                f"Acc[{' '.join(accuracy_parts)}]  "
                f"MacroF1[{' '.join(macro_f1_parts)}]"
            )
        print("")


# ---------------------------------------------------------------------------
# ORCHESTRATOR - ONE LINEAR FLOW FOR THE NOTEBOOK
# ---------------------------------------------------------------------------

def run_success_based_pattern_analysis(
    results_root: str,
    predict_fn_factory: Optional[Callable[[str], PredictFn]] = None,
    domains: Optional[Sequence[str]] = None,
    ptype: str = "semantic",
    print_reports: bool = True,
) -> Dict[str, object]:
    """
    End-to-end, success-based pattern analysis designed to be called from a
    notebook cell. It flows linearly and prints the two required reports:

        Load data  ->  Report A (validity)  ->  success-align ids
                    ->  assign patterns  ->  evaluate model  ->  Report B

    Parameters
    ----------
    results_root:
        Directory containing ``<domain>/<ptype>/<level>.csv`` produced by
        ``EvaluationEngine.generate_all_perturbations``.
    predict_fn_factory:
        Callable ``domain -> predict_fn``. If None, Report B shows distribution
        only (no model metrics). Use :func:`make_checkpoint_predict_fn` bound
        per domain, or reuse already-loaded specialist trainers.
    domains:
        Which domains to analyse. Defaults to every canonical domain that has a
        ``<domain>/<ptype>`` directory on disk.
    ptype:
        Perturbation family folder name. Always ``"semantic"`` (synonym). No
        typo logic exists in this pipeline.

    Returns
    -------
    dict with every computed table so the notebook can inspect or save them:
        validity_table, success_distribution, success_metrics,
        classified_by_domain, common_ids_by_domain
    """
    # --- decide which domains to process ---------------------------------
    if domains is None:
        discovered_domains = []
        for canonical_domain in DOMAIN_FILES:
            domain_folder_name = canonical_domain.lower()
            candidate_directory = os.path.join(results_root, domain_folder_name, ptype)
            if os.path.isdir(candidate_directory):
                discovered_domains.append(canonical_domain)
        domains = discovered_domains
        if len(domains) == 0:
            raise FileNotFoundError(
                f"No <domain>/{ptype} directories found under {results_root}."
            )
    logger.info("Success-based analysis domains: %s", list(domains))

    validity_rows: List[dict] = []
    classified_by_domain: Dict[str, pd.DataFrame] = {}
    common_ids_by_domain: Dict[str, List] = {}

    # Process one domain at a time. Everything below works on plain DataFrames
    # read straight from the engine's CSVs - no wrapper objects in between.
    for domain in domains:
        level_paths = resolve_semantic_level_paths(results_root, domain, ptype)

        # Read the three level CSVs into plain DataFrames.
        frame_by_level: Dict[str, pd.DataFrame] = {}
        for level in LEVELS:
            frame_by_level[level] = read_level_csv(level_paths[level])

        # -- Report A input: count validity per level (failures are kept) --
        # For each level we simply count how many rows succeeded vs failed,
        # using the engine's own perturbation_success flag.
        for level in LEVELS:
            level_frame = frame_by_level[level]
            success_flag = level_frame["perturbation_success"]

            number_of_total_rows = int(len(level_frame))
            number_of_successful_rows = int(success_flag.sum())
            number_of_failed_rows = number_of_total_rows - number_of_successful_rows
            if number_of_total_rows > 0:
                success_rate = number_of_successful_rows / number_of_total_rows
            else:
                success_rate = 0.0

            validity_rows.append({
                "domain": domain,
                "level": level,
                "n_total": number_of_total_rows,
                "n_success": number_of_successful_rows,
                "n_fail": number_of_failed_rows,
                "success_rate": success_rate,
            })

        # -- Report B input: success-based id alignment --------------------
        # Use the SINGLE standardized alignment helper. For this analysis we
        # turn ON duplicate dropping and deterministic ordering, so one id
        # cannot corrupt the aligned subset and runs are reproducible.
        ordered_common_ids = align_common_successful_ids(
            frame_by_level,
            drop_duplicate_ids=True,
            sort_result=True,
        )
        common_ids_by_domain[domain] = ordered_common_ids

        # Index each level by id so we can line up the same id across levels.
        indexed_by_level: Dict[str, pd.DataFrame] = {}
        for level in LEVELS:
            deduplicated = frame_by_level[level].drop_duplicates(
                subset="id", keep=False
            )
            indexed_by_level[level] = deduplicated.set_index("id")

        # Build one aligned+classified row per common id.
        classified_rows: List[dict] = []
        for sample_id in ordered_common_ids:
            low_row = indexed_by_level["low"].loc[sample_id]
            medium_row = indexed_by_level["medium"].loc[sample_id]
            high_row = indexed_by_level["high"].loc[sample_id]

            low_text = str(low_row["perturbed_text"])
            medium_text = str(medium_row["perturbed_text"])
            high_text = str(high_row["perturbed_text"])

            # Exact-string pattern assignment (unchanged methodology).
            pattern = classify_pattern(low_text, medium_text, high_text)

            classified_rows.append({
                "domain": domain,
                "id": sample_id,
                "label": int(low_row["label"]),
                "text_low": low_text,
                "text_medium": medium_text,
                "text_high": high_text,
                "pattern": pattern,
            })

        classified_columns = [
            "domain", "id", "label",
            "text_low", "text_medium", "text_high", "pattern",
        ]
        classified_by_domain[domain] = pd.DataFrame(
            classified_rows, columns=classified_columns
        )

        logger.info(
            "[%s] matched successful ids across low/medium/high: %d",
            domain, len(ordered_common_ids),
        )

    validity_table = pd.DataFrame(
        validity_rows,
        columns=["domain", "level", "n_total", "n_success", "n_fail", "success_rate"],
    )

    # --- distribution + model metrics (Report B input) --------------------
    success_distribution = build_success_pattern_distribution(classified_by_domain)
    success_metrics = evaluate_success_patterns(
        classified_by_domain, predict_fn_factory
    )

    # --- print the two reports directly in the notebook -------------------
    if print_reports:
        print_report_a_validity(validity_table)
        print_report_b_patterns(success_distribution, success_metrics)

    return {
        "validity_table": validity_table,
        "success_distribution": success_distribution,
        "success_metrics": success_metrics,
        "classified_by_domain": classified_by_domain,
        "common_ids_by_domain": common_ids_by_domain,
    }


# ---------------------------------------------------------------------------
# CONSOLIDATED ENTRY POINT  (checkpoint wiring + one linear run)
# ---------------------------------------------------------------------------
#
# This folds in what used to live in the separate ``run_pattern_analysis.py``
# runner: it maps a short model key to its HuggingFace id, builds a per-domain
# predictor from the EXISTING trained checkpoints, and runs the success-based
# analysis end to end. No training happens; no argparse/CLI is needed - call
# ``main(...)`` directly from a notebook cell.

# Map the short model key to the HuggingFace model id the checkpoint was
# fine-tuned from (mirrors evaluate_pipeline.py model_mapping).
SHORT_NAME_TO_HUGGINGFACE_MODEL = {
    "base": "indobenchmark/indobert-base-p1",
    "large": "indobenchmark/indobert-large-p1",
    "lite": "indobenchmark/indobert-lite-base-p1",
}


def build_checkpoint_predict_fn_factory(
    checkpoint_dir: str,
    model_name: str,
    full_model_name: str,
    max_length: int = 128,
    device: Optional[str] = None,
    random_seed: int = 42,
    batch_size: int = 32,
):
    """
    Return a ``domain -> predict_fn`` factory bound to the trained checkpoints.

    The analysis calls this factory once per domain. For each domain it builds
    a predictor that loads that domain's specialist checkpoint and returns
    predicted labels. No training happens here.
    """
    def predict_fn_factory_for_domain(domain: str):
        return make_checkpoint_predict_fn(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            domain=domain,
            full_model_name=full_model_name,
            max_length=max_length,
            device=device,
            random_seed=random_seed,
            batch_size=batch_size,
        )

    return predict_fn_factory_for_domain


def main(
    results_root: str,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "base",
    domains: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 128,
    random_seed: int = 42,
    skip_model: bool = False,
    print_reports: bool = True,
) -> Dict[str, object]:
    """
    Consolidated, notebook-friendly entry point for the success-based pattern
    analysis. One linear call:

        Load data -> Report A (validity) -> align common ids
                  -> assign patterns -> evaluate model -> Report B

    Parameters
    ----------
    results_root:
        Directory containing ``<domain>/semantic/<level>.csv``.
    checkpoint_dir:
        Root checkpoint directory. Required unless ``skip_model=True``. Each
        specialist is expected at
        ``<checkpoint_dir>/<model_name>/<Domain.capitalize()>/best_model.pt``.
    model_name:
        Short model key ("base" / "large" / "lite"); anything else is treated
        as a full HuggingFace id.
    skip_model:
        If True, run distribution-only (Report B shows no model metrics).
    """
    # Resolve the full HuggingFace model id (fall back to the string as given).
    if model_name in SHORT_NAME_TO_HUGGINGFACE_MODEL:
        full_model_name = SHORT_NAME_TO_HUGGINGFACE_MODEL[model_name]
    else:
        full_model_name = model_name

    # Decide whether we will score the model or only compute the distribution.
    if skip_model:
        predict_fn_factory = None
    else:
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir is required unless skip_model=True")
        predict_fn_factory = build_checkpoint_predict_fn_factory(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            full_model_name=full_model_name,
            max_length=max_length,
            device=device,
            random_seed=random_seed,
            batch_size=batch_size,
        )

    results = run_success_based_pattern_analysis(
        results_root=results_root,
        predict_fn_factory=predict_fn_factory,
        domains=domains,
        ptype="semantic",
        print_reports=print_reports,
    )
    return results
