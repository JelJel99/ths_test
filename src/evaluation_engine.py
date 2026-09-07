# 2 SEP 0016 - Evaluation Engine
"""
Evaluation Engine Module for IndoBERT Clickbait Detection System

This module handles metric calculation, in-domain evaluation, cross-domain evaluation,
perturbation testing orchestration, and results aggregation for robustness analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score, matthews_corrcoef
)
from datetime import datetime
import logging
import ast

from debug_logger import (
    dbg_eval_in_domain,
    dbg_eval_cross_domain,
    dbg_eval_confusion_matrix,
    dbg_eval_domain_shift,
    dbg_perturbation_samples,
    dbg_perturbation_stats,
    dbg_perturbation_metrics,
    dbg_perturbation_robustness,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates performance metrics for clickbait detection.
    """

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary containing all metrics
        """

        precision = precision_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0
        )

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),

            # Macro metrics
            'macro_precision': precision_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average='macro',
                zero_division=0
            ),

            'macro_recall': recall_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average='macro',
                zero_division=0
            ),

            'macro_f1': f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average='macro',
                zero_division=0
            ),

            # Class 0
            'precision_0': precision[0],
            'recall_0': recall[0],
            'f1_0': f1[0],

            # Class 1
            'precision_1': precision[1],
            'recall_1': recall[1],

            'f1_1': f1[1],

            'mcc': matthews_corrcoef(y_true, y_pred),
            'roc_auc': None
        }

        # Use y_proba for ROC-AUC
        if y_proba is not None:
            try:
                probs = (
                    y_proba[:, 1]
                    if len(np.array(y_proba).shape) == 2
                    else y_proba
                )
                metrics['roc_auc'] = roc_auc_score(y_true, probs)
            except Exception:
                metrics['roc_auc'] = None

        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        return metrics

    @staticmethod
    def calculate_robustness_metrics(
        clean_metrics: Dict[str, float],
        perturbed_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate robustness degradation metrics.

        Args:
            clean_metrics: Metrics on clean data
            perturbed_metrics: Metrics on perturbed data

        Returns:
            Dictionary with degradation metrics
        """
        robustness = {}

        for metric_name in ['accuracy','macro_precision','macro_recall','macro_f1']:
            if metric_name in clean_metrics and metric_name in perturbed_metrics:
                clean_val = clean_metrics[metric_name]
                perturbed_val = perturbed_metrics[metric_name]

                # Absolute drop
                robustness[f'{metric_name}_drop'] = clean_val - perturbed_val

                # Relative drop (percentage)
                if clean_val > 0:
                    robustness[f'{metric_name}_drop_pct'] = (
                        (clean_val - perturbed_val) / clean_val * 100
                    )
                else:
                    robustness[f'{metric_name}_drop_pct'] = 0.0

        return robustness

    @staticmethod
    def calculate_domain_shift_metrics(
        source_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
        in_domain_target_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate Source Drop (SD) and Target Drop (TD) metrics.
        """

        shift_metrics = {}

        # Source Drop (SD)
        for metric_name in ['accuracy','macro_precision','macro_recall','macro_f1']:
            if metric_name in source_metrics and metric_name in target_metrics:
                source_val = source_metrics[metric_name]
                cross_domain_val = target_metrics[metric_name]

                shift_metrics[f'sd_{metric_name}'] = (
                    source_val - cross_domain_val
                )

                if source_val > 0:
                    shift_metrics[f'sd_{metric_name}_pct'] = (
                        (source_val - cross_domain_val)
                        / source_val * 100
                    )
                else:
                    shift_metrics[f'sd_{metric_name}_pct'] = 0.0

        # Target Drop (TD)
        for metric_name in ['accuracy','macro_precision','macro_recall','macro_f1']:
            if (
                metric_name in in_domain_target_metrics
                and metric_name in target_metrics
            ):
                in_domain_val = in_domain_target_metrics[metric_name]
                cross_domain_val = target_metrics[metric_name]

                shift_metrics[f'td_{metric_name}'] = (
                    in_domain_val - cross_domain_val
                )

                if in_domain_val > 0:
                    shift_metrics[f'td_{metric_name}_pct'] = (
                        (in_domain_val - cross_domain_val)
                        / in_domain_val * 100
                    )
                else:
                    shift_metrics[f'td_{metric_name}_pct'] = 0.0

        return shift_metrics

    @staticmethod
    def calculate_prediction_flip(
        clean_predictions,
        perturbed_predictions,
        true_labels,
        successful_perturbations=None
    ) -> Dict[str, Any]:
        """
        Calculate prediction flips between clean and perturbed predictions.

        A prediction flip occurs when:
            clean prediction != perturbed prediction

        Also reports:
            - flip direction: 0 -> 1 and 1 -> 0
            - correctness transitions
            - flip rate over all samples
            - flip rate among successfully perturbed samples
        """

        clean_predictions = np.asarray(clean_predictions)
        perturbed_predictions = np.asarray(perturbed_predictions)
        true_labels = np.asarray(true_labels)

        if not (
            len(clean_predictions)
            == len(perturbed_predictions)
            == len(true_labels)
        ):
            raise ValueError(
                "Prediction and label arrays must have the same length."
            )

        total_samples = len(clean_predictions)

        # ------------------------------------------------------------
        # Identify prediction flips
        # ------------------------------------------------------------

        flip_mask = clean_predictions != perturbed_predictions

        flip_count = int(flip_mask.sum())

        flip_rate = (
            flip_count / total_samples
            if total_samples > 0
            else 0.0
        )

        # ------------------------------------------------------------
        # Flip direction
        # ------------------------------------------------------------

        flip_0_to_1_mask = (
            (clean_predictions == 0)
            & (perturbed_predictions == 1)
        )

        flip_1_to_0_mask = (
            (clean_predictions == 1)
            & (perturbed_predictions == 0)
        )

        flip_0_to_1_count = int(flip_0_to_1_mask.sum())
        flip_1_to_0_count = int(flip_1_to_0_mask.sum())

        # ------------------------------------------------------------
        # Correctness transitions
        # ------------------------------------------------------------

        clean_correct = clean_predictions == true_labels
        perturbed_correct = perturbed_predictions == true_labels

        correct_to_incorrect = int(
            (clean_correct & ~perturbed_correct).sum()
        )

        incorrect_to_correct = int(
            (~clean_correct & perturbed_correct).sum()
        )

        clean_correct_count = int(clean_correct.sum())
        clean_incorrect_count = int((~clean_correct).sum())

        # Rate among ALL samples
        correct_to_incorrect_rate = (
            correct_to_incorrect / total_samples
            if total_samples > 0
            else 0.0
        )

        incorrect_to_correct_rate = (
            incorrect_to_correct / total_samples
            if total_samples > 0
            else 0.0
        )

        # Rate conditional on original prediction correctness
        correct_to_incorrect_rate_among_correct = (
            correct_to_incorrect / clean_correct_count
            if clean_correct_count > 0
            else 0.0
        )

        incorrect_to_correct_rate_among_incorrect = (
            incorrect_to_correct / clean_incorrect_count
            if clean_incorrect_count > 0
            else 0.0
        )

        # ------------------------------------------------------------
        # Successful perturbations only
        # ------------------------------------------------------------

        successful_count = None
        flip_count_successful_only = None
        flip_rate_successful_only = None

        if successful_perturbations is not None:
            successful_mask = np.asarray(
                successful_perturbations,
                dtype=bool
            )

            if len(successful_mask) != total_samples:
                raise ValueError(
                    "successful_perturbations must have "
                    "the same length as predictions."
                )

            successful_count = int(successful_mask.sum())
            successful_flip_mask = flip_mask & successful_mask

            flip_count_successful_only = int(
                successful_flip_mask.sum()
            )

            flip_rate_successful_only = (
                flip_count_successful_only / successful_count
                if successful_count > 0
                else 0.0
            )

        successful_rate = None
        if successful_perturbations is not None:
            successful_rate = (
                successful_count / total_samples
                if total_samples > 0
                else 0.0
            )

        return {
            "total_samples": total_samples,

            "successful_perturbations": successful_count,
            "successful_perturbation_rate": successful_rate,
            "successful_perturbation_rate_pct": (
                successful_rate * 100
                if successful_rate is not None
                else None
            ),

            "flip_count": flip_count,
            "flip_rate": flip_rate,
            "flip_rate_pct": flip_rate * 100,

            "flip_0_to_1_count": flip_0_to_1_count,
            "flip_1_to_0_count": flip_1_to_0_count,

            "flip_0_to_1_rate": (
                flip_0_to_1_count / total_samples
                if total_samples > 0
                else 0.0
            ),

            "flip_1_to_0_rate": (
                flip_1_to_0_count / total_samples
                if total_samples > 0
                else 0.0
            ),

            "flip_count_successful_only": flip_count_successful_only,

            "flip_rate_successful_only": (
                flip_rate_successful_only
            ),

            "flip_rate_successful_only_pct": (
                flip_rate_successful_only * 100
                if flip_rate_successful_only is not None
                else None
            ),

            "flips": {
                "correct_to_incorrect": correct_to_incorrect,
                "incorrect_to_correct": incorrect_to_correct,

                # Among all samples
                "correct_to_incorrect_rate": correct_to_incorrect_rate,
                "correct_to_incorrect_rate_pct":
                    correct_to_incorrect_rate * 100,

                "incorrect_to_correct_rate": incorrect_to_correct_rate,
                "incorrect_to_correct_rate_pct":
                    incorrect_to_correct_rate * 100,

                # Conditional rates
                "clean_correct_count": clean_correct_count,
                "clean_incorrect_count": clean_incorrect_count,

                "correct_to_incorrect_rate_among_correct":
                    correct_to_incorrect_rate_among_correct,

                "correct_to_incorrect_rate_among_correct_pct":
                    correct_to_incorrect_rate_among_correct * 100,

                "incorrect_to_correct_rate_among_incorrect":
                    incorrect_to_correct_rate_among_incorrect,

                "incorrect_to_correct_rate_among_incorrect_pct":
                    incorrect_to_correct_rate_among_incorrect * 100
            }
        }


class InDomainEvaluator:
    """
    Handles in-domain evaluation for trained models.
    """

    def __init__(self, model_trainer):
        """
        Initialize in-domain evaluator.

        Args:
            model_trainer: ModelTrainer instance with trained model
        """
        self.model_trainer = model_trainer
        self.metrics_calculator = MetricsCalculator()
        logger.info("InDomainEvaluator initialized")

    def evaluate(
        self,
        test_df: pd.DataFrame,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on in-domain test data.

        Args:
            test_df: Test DataFrame with 'text' and 'label' columns
            domain: Domain name (for logging)

        Returns:
            Dictionary with evaluation results
        """
        domain_str = f" ({domain})" if domain else ""
        logger.info(f"Evaluating in-domain{domain_str}: {len(test_df)} samples")

        # Get predictions
        y_true = test_df['label'].values
        y_pred, y_proba = self.model_trainer.predict(test_df['text'].tolist())

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)

        # Add metadata
        results = {
            'domain': domain,
            'num_samples': len(test_df),
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'true_labels': y_true.tolist()
        }

        logger.info(
           f"In-domain Macro-F1={metrics['macro_f1']:.4f} | "
           f"F1_0={metrics['f1_0']:.4f} | "
           f"F1_1={metrics['f1_1']:.4f}"
          )

        # ── debug: in-domain metrics + confusion matrix ────────────────────
        dbg_eval_in_domain(
            domain=domain or "?",
            n_samples=len(test_df),
            metrics=metrics,
        )
        if metrics.get('confusion_matrix'):
            dbg_eval_confusion_matrix(
                context=f"in-domain/{domain}",
                cm=metrics['confusion_matrix'],
            )

        return results

    def evaluate_by_domain(
        self,
        test_df: pd.DataFrame,
        domains: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model separately for each domain.

        Args:
            test_df: Test DataFrame with 'domain' column
            domains: List of domain names

        Returns:
            Dictionary mapping domain names to evaluation results
        """
        logger.info(f"Evaluating across {len(domains)} domains")

        results = {}

        for domain in domains:
            domain_df = test_df[test_df['domain'] == domain].copy()
            if len(domain_df) > 0:
                results[domain] = self.evaluate(domain_df, domain)
            else:
                logger.warning(f"No samples found for domain: {domain}")

        return results


class CrossDomainEvaluator:
    """
    Handles cross-domain evaluation (5x5 matrix).
    """

    def __init__(self, model_trainers: Dict[str, Any]):
        """
        Initialize cross-domain evaluator.

        Args:
            model_trainers: Dictionary mapping domain names to ModelTrainer instances
        """
        self.model_trainers = model_trainers
        self.metrics_calculator = MetricsCalculator()
        logger.info("CrossDomainEvaluator initialized")

    def evaluate_cross_domain(
        self,
        source_domain: str,
        target_domain: str,
        target_test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate source domain model on target domain data.
        """
        logger.info(f"Cross-domain evaluation: {source_domain} -> {target_domain}")

        # Ensure we are selecting the trainer that corresponds to the source_domain
        if source_domain not in self.model_trainers:
            raise ValueError(f"No model found for source domain: {source_domain}")

        # This is the vital step: selecting the specialist model for the source domain
        model_trainer = self.model_trainers[source_domain]

        # Get predictions using the specialist model
        y_true = target_test_df['label'].values
        y_pred, y_proba = model_trainer.predict(target_test_df['text'].tolist())

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)

        # ── debug: cross-domain metrics + confusion matrix ─────────────────
        dbg_eval_cross_domain(
            source=source_domain,
            target=target_domain,
            n_samples=len(target_test_df),
            metrics=metrics,
        )
        if metrics.get('confusion_matrix'):
            dbg_eval_confusion_matrix(
                context=f"cross-domain/{source_domain}->{target_domain}",
                cm=metrics['confusion_matrix'],
            )

        return {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'num_samples': len(target_test_df),
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'true_labels': y_true.tolist()
        }

    def evaluate_all_combinations(self, test_data: Dict[str, pd.DataFrame]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Evaluate all source-target domain combinations (5x5 matrix).

        Source domains for which no specialist was loaded are skipped with a
        warning instead of raising, so partial evaluation can complete for the
        domains that did train successfully.

        Args:
            test_data: Dictionary mapping domain names to test DataFrames

        Returns:
            Dictionary mapping (source, target) tuples to evaluation results
        """
        domains = list(test_data.keys())
        results = {}
        for source_domain in domains:
            if source_domain not in self.model_trainers:
                logger.warning(
                    f"Skipping cross-domain evaluation for source '{source_domain}': "
                    "no specialist model was loaded for this domain."
                )
                continue
            for target_domain in domains:
                if target_domain in test_data:
                    # Keep as Tuple key for internal logic
                    key = (source_domain, target_domain)
                    results[key] = self.evaluate_cross_domain(
                        source_domain, target_domain, test_data[target_domain]
                    )
        return results

    def create_performance_matrix(
        self,
        cross_domain_results: Dict[Tuple[str, str], Dict[str, Any]],
        metric: str = 'macro-f1'
    ) -> pd.DataFrame:
        """
        Create a performance matrix for visualization.

        Args:
            cross_domain_results: Results from evaluate_all_combinations
            metric: Metric to use for matrix values

        Returns:
            DataFrame with source domains as rows, target domains as columns
        """
        domains = sorted(
            set(
                [k[0] for k in cross_domain_results.keys()] +
                [k[1] for k in cross_domain_results.keys()]
            )
        )

        matrix = pd.DataFrame(
            index=domains,
            columns=domains,
            dtype=float
        )

        for (source, target), result in cross_domain_results.items():
            if metric in result['metrics']:
                matrix.loc[source, target] = result['metrics'][metric]

        return matrix


class PerturbationEvaluator:
    """
    Orchestrates perturbation testing across multiple levels.
    """

    def __init__(
        self,
        model_trainer,
        perturbation_engine,
        perturbation_output_dir: Optional[str] = None
    ):
        """
        Initialize perturbation evaluator.

        Args:
            model_trainer: ModelTrainer instance
            perturbation_engine: PerturbationEngine instance
            perturbation_output_dir: Directory where perturbation data
                                     will be saved.
        """
        self.model_trainer = model_trainer
        self.perturbation_engine = perturbation_engine
        self.metrics_calculator = MetricsCalculator()
        self.perturbation_output_dir = perturbation_output_dir

        logger.info("PerturbationEvaluator initialized")

    #def _calculate_perturbation_statistics(
    #    self,
    #    original_df: pd.DataFrame,
    #    perturbed_df: pd.DataFrame
    #) -> Dict[str, Any]:
    #    """
    #    Calculate actual perturbation change statistics.
    #    """
    #
    #    original_texts = (
    #        original_df['text'].astype(str).tolist()
    #    )
    #
    #    perturbed_texts = (
    #        perturbed_df['text'].astype(str).tolist()
    #    )
    #
    #    if len(original_texts) != len(perturbed_texts):
    #        raise ValueError(
    #            "Original and perturbed datasets must have "
    #            "the same number of samples."
    #        )
    #
    #    changed_samples = 0
    #    word_change_rates = []
    #    char_change_rates = []
    #
    #    for original, perturbed in zip(
    #        original_texts,
    #        perturbed_texts
    #    ):
    #
    #        if original != perturbed:
    #            changed_samples += 1
    #
    #        original_words = original.split()
    #        perturbed_words = perturbed.split()
    #
    #        # Position-based word changes
    #        max_len = max(
    #            len(original_words),
    #            len(perturbed_words)
    #        )
    #
    #        changed_words = sum(
    #            1
    #            for i in range(max_len)
    #            if (
    #                i >= len(original_words)
    #                or i >= len(perturbed_words)
    #                or original_words[i] != perturbed_words[i]
    #            )
    #        )
    #
    #        word_change_rate = (
    #            changed_words / len(original_words)
    #            if len(original_words) > 0
    #            else 0.0
    #        )
    #
    #        word_change_rates.append(word_change_rate)
    #
    #        # Character change rate
    #        max_char_len = max(
    #            len(original),
    #            len(perturbed)
    #        )
    #
    #        changed_chars = sum(
    #            1
    #            for i in range(max_char_len)
    #            if (
    #                i >= len(original)
    #                or i >= len(perturbed)
    #                or original[i] != perturbed[i]
    #            )
    #        )
    #
    #        char_change_rate = (
    #            changed_chars / len(original)
    #            if len(original) > 0
    #            else 0.0
    #        )
    #
    #        char_change_rates.append(char_change_rate)
    #
    #    total = len(original_texts)
    #
    #    return {
    #        'total_samples': total,
    #
    #        'changed_samples': changed_samples,
    #
    #        'unchanged_samples': (
    #            total - changed_samples
    #        ),
    #
    #        'sample_change_rate': (
    #            changed_samples / total
    #            if total > 0
    #            else 0.0
    #        ),
    #
    #        'mean_word_change_rate': (
    #            float(np.mean(word_change_rates))
    #            if word_change_rates
    #            else 0.0
    #        ),
    #
    #        'median_word_change_rate': (
    #            float(np.median(word_change_rates))
    #            if word_change_rates
    #            else 0.0
    #        ),
    #
    #        'mean_char_change_rate': (
    #            float(np.mean(char_change_rates))
    #            if char_change_rates
    #            else 0.0
    #        )
    #    }

    def evaluate_with_perturbation(
        self,
        test_df: pd.DataFrame,
        perturbation_level: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on perturbed test data.

        Failed perturbations are retained and counted separately.
        """

        domain_str = f" ({domain})" if domain else ""

        logger.info(
            f"Evaluating with {perturbation_level}-level "
            f"perturbation{domain_str}"
        )

        # ---------------------------------------------------------------
        # Apply perturbations
        # ---------------------------------------------------------------
        perturbed_df = self.perturbation_engine.apply_to_dataframe(
            test_df.copy(),
            text_column='text',
            level=perturbation_level
        )

        ## ── debug: perturbation sample pairs ──────────────────────────────
        #dbg_perturbation_samples(
        #    level=perturbation_level,
        #    domain=domain or "?",
        #    originals=original_texts,
        #    perturbed=perturbed_texts,
        #)
        #
        ## ── debug: perturbation character/word change stats ──────────────
        #import numpy as _np
        #
        #char_changes = [
        #    sum(1 for a, b in zip(o, p) if a != b) / max(len(o), 1)
        #    for o, p in zip(original_texts, perturbed_texts)
        #]
        #
        #word_changes = [
        #    len(
        #        set(o.split()).symmetric_difference(
        #            set(p.split())
        #        )
        #    ) / max(len(o.split()), 1)
        #    for o, p in zip(original_texts, perturbed_texts)
        #]
        #
        #dbg_perturbation_stats(
        #    level=perturbation_level,
        #    domain=domain or "?",
        #    n_texts=len(perturbed_df),
        #    char_change_mean=(
        #        float(_np.mean(char_changes))
        #        if char_changes else 0.0
        #    ),
        #    word_change_mean=(
        #        float(_np.mean(word_changes))
        #        if word_changes else 0.0
        #    ),
        #)

        # ---------------------------------------------------------------
        # Get predictions
        # ---------------------------------------------------------------
        y_true = perturbed_df['label'].values

        y_pred, y_proba = self.model_trainer.predict(
            perturbed_df['text'].tolist()
        )

        # ---------------------------------------------------------------
        # Calculate metrics
        # ---------------------------------------------------------------
        metrics = self.metrics_calculator.calculate_metrics(
            y_true,
            y_pred,
            y_proba
        )

        # ── debug: metrics after perturbation ─────────────────────────────
        dbg_perturbation_metrics(
            level=perturbation_level,
            domain=domain or "?",
            metrics=metrics,
        )

        # ---------------------------------------------------------------
        # Store results
        # ---------------------------------------------------------------
        results = {
            'domain': domain,
            'perturbation_level': perturbation_level,

            # Dataset size
            'num_samples': len(perturbed_df),

            # Evaluation metrics
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'true_labels': y_true.tolist()
        }

        logger.info(
            f"{perturbation_level.capitalize()}-level perturbation"
            f"{domain_str} "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
        )

        return results

    def evaluate_existing_perturbation(
        self,
        perturbed_df: pd.DataFrame,
        clean_results: Dict[str, Any],
        perturbation_level: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an already-generated perturbation.

        IMPORTANT:
        This method does NOT generate a new perturbation.
        """

        logger.info(
            f"Evaluating existing {perturbation_level}-level "
            f"perturbation ({domain})"
        )

        y_true = perturbed_df['label'].values

        y_pred, y_proba = self.model_trainer.predict(
            perturbed_df['text'].tolist()
        )

        metrics = self.metrics_calculator.calculate_metrics(
            y_true,
            y_pred,
            y_proba
        )

        robustness = self.metrics_calculator.calculate_robustness_metrics(
            clean_results['metrics'],
            metrics
        )

        # ------------------------------------------------------------
        # Prediction flip analysis
        # ------------------------------------------------------------
        if 'perturbation_success' not in perturbed_df.columns:
            raise ValueError(
                f"evaluate_existing_perturbation function Missing 'perturbation_success'"
            )
        successful_perturbations = (
            perturbed_df["perturbation_success"].tolist()
        )

        prediction_flip = self.metrics_calculator.calculate_prediction_flip(
            clean_predictions=clean_results['predictions'],
            perturbed_predictions=y_pred.tolist(),
            true_labels=y_true.tolist(),
            successful_perturbations=successful_perturbations
        )

        # ------------------------------------------------------------
        # Perturbation statistics
        # ------------------------------------------------------------
        perturbation_stats = {
            'total_samples': len(perturbed_df),

            'changed_samples': int(
                (
                    perturbed_df['is_same_as_original'] == False
                ).sum()
            ) if 'is_same_as_original' in perturbed_df.columns
            else None,

            'sample_change_rate': float(
                (
                    perturbed_df['is_same_as_original'] == False
                ).mean()
            ) if 'is_same_as_original' in perturbed_df.columns
            else None,

            'mean_word_change_rate': float(
                perturbed_df[
                    'actual_ratio_all_words'
                ].mean()
            ) if 'actual_ratio_all_words' in perturbed_df.columns
            else None,

            'median_word_change_rate': float(
                perturbed_df[
                    'actual_ratio_all_words'
                ].median()
            ) if 'actual_ratio_all_words' in perturbed_df.columns
            else None,

            'mean_eligible_word_change_rate': float(
                perturbed_df[
                    'actual_ratio_eligible_words'
                ].mean()
            ) if 'actual_ratio_eligible_words' in perturbed_df.columns
            else None,

            'mean_word_cosine_similarity': float(
                perturbed_df[
                    'word_cosine_similarity_mean'
                ].dropna().mean()
            ) if (
                'word_cosine_similarity_mean'
                in perturbed_df.columns
                and perturbed_df[
                    'word_cosine_similarity_mean'
                ].notna().any()
            ) else None,

            'min_word_cosine_similarity': float(
                perturbed_df[
                    'word_cosine_similarity_min'
                ].dropna().min()
            ) if (
                'word_cosine_similarity_min'
                in perturbed_df.columns
                and perturbed_df[
                    'word_cosine_similarity_min'
                ].notna().any()
            ) else None,

            'max_word_cosine_similarity': float(
                perturbed_df[
                    'word_cosine_similarity_max'
                ].dropna().max()
            ) if (
                'word_cosine_similarity_max'
                in perturbed_df.columns
                and perturbed_df[
                    'word_cosine_similarity_max'
                ].notna().any()
            ) else None,
        }

        # ------------------------------------------------------------
        # Recover individual replacement similarities
        #
        # The perturbation dataset was saved to CSV, so dataframe
        # attrs are no longer available.
        #
        # Instead, read the persisted "replacements" column.
        # ------------------------------------------------------------

        similarities = []

        if 'replacements' in perturbed_df.columns:

            for replacements in perturbed_df[
                'replacements'
            ]:

                # CSV may store the list as a string
                if isinstance(replacements, str):

                    try:
                        replacements = ast.literal_eval(
                            replacements
                        )

                    except (ValueError, SyntaxError):
                        continue

                if not isinstance(
                    replacements,
                    list
                ):
                    continue

                for replacement in replacements:

                    if not isinstance(
                        replacement,
                        dict
                    ):
                        continue

                    similarity = replacement.get(
                        'word_cosine_similarity'
                    )

                    if similarity is None:
                        continue

                    try:
                        similarities.append(
                            float(similarity)
                        )
                    except (TypeError, ValueError):
                        continue

        # ------------------------------------------------------------
        # Semantic similarity statistics
        # ------------------------------------------------------------

        if similarities:

            similarity_statistics = {
                'count': len(similarities),

                'mean': float(
                    np.mean(similarities)
                ),

                'median': float(
                    np.median(similarities)
                ),

                'std': float(
                    np.std(similarities)
                ),

                'min': float(
                    np.min(similarities)
                ),

                'max': float(
                    np.max(similarities)
                ),
            }

        else:

            similarity_statistics = {
                'count': 0,
                'mean': None,
                'median': None,
                'std': None,
                'min': None,
                'max': None,
            }

        # ------------------------------------------------------------
        # Combine perturbation statistics
        # ------------------------------------------------------------

        combined_perturbation_stats = {
            **perturbation_stats,
            'semantic_similarity':
                similarity_statistics,
        }

        return {
            'domain': domain,
            'perturbation_level': perturbation_level,
            'num_samples': len(perturbed_df),

            'perturbation_statistics':
                combined_perturbation_stats,

            'metrics': metrics,
            'robustness_metrics': robustness,

            'prediction_flip': prediction_flip,

            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'true_labels': y_true.tolist()
        }


    def evaluate_all_levels(
        self,
        test_df: pd.DataFrame,
        clean_results: Dict[str, Any],
        domain: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model across all perturbation levels.
        """

        levels = ['low', 'medium', 'high']

        results = {
            'clean': clean_results
        }

        for level in levels:

            results[level] = self.evaluate_with_perturbation(
                test_df,
                level,
                domain
            )

            # Calculate robustness metrics
            robustness = self.metrics_calculator.calculate_robustness_metrics(
                clean_results['metrics'],
                results[level]['metrics']
            )

            results[level]['robustness_metrics'] = robustness

            # ── debug: robustness drop metrics ───────────────────────────
            dbg_perturbation_robustness(
                level=level,
                domain=domain or "?",
                rob=robustness,
            )

        return results


class EvaluationEngine:
    """
    Main evaluation engine orchestrating all evaluation types.
    """

    def __init__(
        self,
        model_trainers: Dict[str, Any],
        perturbation_engine,
        output_dir: str = "evaluation_results",
        perturbation_output_dir: Optional[str] = None
    ):
        """
        Initialize evaluation engine.

        Args:
            model_trainers: Dictionary mapping domain names to ModelTrainer instances
            perturbation_engine: PerturbationEngine instance
            output_dir: Directory to save evaluation results
            perturbation_output_dir: Directory to save generated perturbation data
        """
        self.model_trainers = model_trainers
        self.perturbation_engine = perturbation_engine
        self.output_dir = output_dir
        self.perturbation_output_dir = perturbation_output_dir

        # Initialize evaluators
        self.metrics_calculator = MetricsCalculator()
        self.cross_domain_evaluator = CrossDomainEvaluator(model_trainers)

        os.makedirs(output_dir, exist_ok=True)

        if perturbation_output_dir is not None:
            os.makedirs(perturbation_output_dir, exist_ok=True)

        logger.info(
            f"EvaluationEngine initialized. "
            f"Results will be saved to {output_dir}"
        )

        if perturbation_output_dir:
            logger.info(
                f"Perturbation data will be saved to "
                f"{perturbation_output_dir}"
            )

    def generate_all_perturbations(
        self,
        test_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:

        logger.info("Generating all perturbations ONCE")

        perturbed_test_data = {}

        for domain, test_df in test_data.items():
            logger.info(f"Generating perturbations for {domain}")
            perturbed_test_data[domain] = {}

            domain_output_dir = os.path.join(
                self.perturbation_output_dir,
                str(domain).strip().lower()
            )
            os.makedirs(domain_output_dir, exist_ok=True)

            for level in ['low', 'medium', 'high']:
                logger.info(
                    f"  Generating {domain} - {level}"
                )

                perturbed_df = self.perturbation_engine.apply_to_dataframe(
                    test_df.copy(),
                    text_column='text',
                    level=level
                )

                # ============================================================
                # DEFINE FINAL PERTURBATION SUCCESS
                # ============================================================

                if (
                    'is_same_as_original' not in perturbed_df.columns
                    or 'perturbation_in_range' not in perturbed_df.columns
                ):
                    raise ValueError(
                        "Required perturbation validation columns are missing: "
                        "'is_same_as_original' and/or 'perturbation_in_range'."
                    )

                perturbed_df['perturbation_success'] = (
                    (~perturbed_df['is_same_as_original'].astype(bool))
                    & perturbed_df['perturbation_in_range'].astype(bool)
                )

                # ============================================================
                # DEBUG
                # ============================================================

                print("\nperturbation_success:")
                print(
                    perturbed_df['perturbation_success']
                    .value_counts(dropna=False)
                )

                # ============================================================
                # STORE
                # ============================================================

                perturbed_test_data[domain][level] = perturbed_df

                output_path = os.path.join(
                    domain_output_dir,
                    f"{level}.csv"
                )

                perturbed_df.to_csv(
                    output_path,
                    index=False
                )

                logger.info(
                    f"  Saved: {output_path}"
                )

        logger.info("All perturbations generated and saved successfully")

        return perturbed_test_data

    def run_complete_evaluation(
        self,
        test_data: Dict[str, pd.DataFrame],
        include_perturbations: bool = True
    ):
        """
        Run complete evaluation pipeline including in-domain, cross-domain,
        and perturbation testing.

        Args:
            test_data: Dictionary mapping domain names to test DataFrames
            include_perturbations: Whether to include perturbation testing

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Starting complete evaluation pipeline")

        results = {
            'timestamp': datetime.now().isoformat(),
            'domains': list(test_data.keys()),
            'in_domain': {},
            'cross_domain': {},
            'perturbation': {},
            'cross_domain_perturbation': {}
        }

        # 1. In-domain evaluation
        logger.info("Phase 1: In-domain evaluation")
        for domain, test_df in test_data.items():
            if domain in self.model_trainers:
                evaluator = InDomainEvaluator(
                    self.model_trainers[domain]
                )
                results['in_domain'][domain] = (
                    evaluator.evaluate(test_df, domain)
                )

        # 2. Cross-domain evaluation
        logger.info("Phase 2: Cross-domain evaluation")
        results['cross_domain'] = (
            self.cross_domain_evaluator
            .evaluate_all_combinations(test_data)
        )
        # 2.1 Calculate Domain Shift (SD and TD)
        logger.info("Phase 2.1: Calculating Domain Shift metrics...")
        in_domain_data = results['in_domain']

        # Iterate over the flat dictionary directly
        for (source, target), cross_results in results['cross_domain'].items():

            # Only calculate shift for cross-domain (source != target)
            if source != target:
                if source in in_domain_data and target in in_domain_data:
                    source_in_domain = in_domain_data[source]['metrics']
                    target_in_domain = in_domain_data[target]['metrics']

                    shift = self.metrics_calculator.calculate_domain_shift_metrics(
                        source_in_domain,
                        cross_results['metrics'],
                        target_in_domain
                    )
                    cross_results['domain_shift'] = shift

                    # ── debug: domain shift values ─────────────────────────
                    dbg_eval_domain_shift(
                        source=source,
                        target=target,
                        shift=shift,
                    )
                else:
                    logger.warning(f"Missing in-domain data for shift calculation: {source} -> {target}")

        # 3. Generate perturbations ONCE
        perturbed_test_data = None
        if include_perturbations:
            perturbed_test_data = (
                self.generate_all_perturbations(test_data)
            )

        # 4. In-domain perturbation evaluation
        if include_perturbations:

            logger.info("Phase 4: In-domain perturbation evaluation")

            for domain in test_data:
                if domain not in self.model_trainers:
                    continue
                evaluator = PerturbationEvaluator(
                    model_trainer=self.model_trainers[domain],
                    perturbation_engine=self.perturbation_engine,
                    perturbation_output_dir=self.perturbation_output_dir
                )
                clean_results = results['in_domain'][domain]
                results['perturbation'][domain] = {
                    'clean': clean_results
                }
                for level in ['low', 'medium', 'high']:
                    perturbed_df = (
                        perturbed_test_data[domain][level]
                    )
                    results['perturbation'][domain][level] = (
                        evaluator.evaluate_existing_perturbation(
                            perturbed_df=perturbed_df,
                            clean_results=clean_results,
                            perturbation_level=level,
                            domain=domain
                        )
                    )
        # ============================================================
        # PHASE 5 — CROSS-DOMAIN PERTURBATION
        # ============================================================

        if include_perturbations:

            cross_results = (
                self.run_cross_domain_with_perturbations(
                    test_data=test_data,
                    perturbed_test_data=perturbed_test_data
                )
            )

            results['cross_domain_perturbation'] = (
                cross_results['cross_domain_perturbation']
            )

        logger.info("Complete evaluation pipeline finished")

        # Save results
        self.save_results(results)

        return results

    def run_cross_domain_with_perturbations(
        self,
        test_data: Dict[str, pd.DataFrame],
        perturbed_test_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Any]:
        """
        Run cross-domain evaluation with perturbations (interaction effects).

        Args:
            test_data: Dictionary mapping domain names to test DataFrames
            in_domain_results: Results from in-domain evaluation
        Returns:
            Dictionary with cross-domain perturbation results
        """
        logger.info("Starting cross-domain evaluation with perturbations")

        results = {
            'timestamp': datetime.now().isoformat(),
            'cross_domain_perturbation': {}
        }

        domains = list(test_data.keys())
        perturbation_levels = ['clean', 'low', 'medium', 'high']

        for source_domain in domains:
            if source_domain not in self.model_trainers:
                continue

            model_trainer = self.model_trainers[source_domain]

            for target_domain in domains:
                if target_domain not in test_data:
                    continue

                key = f"{source_domain}_to_{target_domain}"
                results['cross_domain_perturbation'][key] = {}

                target_df = test_data[target_domain]

                # Evaluate on clean data
                y_true = target_df['label'].values
                y_pred, y_proba = model_trainer.predict(target_df['text'].tolist())
                clean_metrics = self.metrics_calculator.calculate_metrics(
                    y_true, y_pred, y_proba
                )
                results['cross_domain_perturbation'][key]['clean'] = {
                    'metrics': clean_metrics,
                    'predictions': y_pred.tolist(),
                    'probabilities': y_proba.tolist(),
                    'true_labels': y_true.tolist()
                }

                # Evaluate with perturbations
                for level in ['low', 'medium', 'high']:

                    # Reuse the already-generated perturbation
                    perturbed_df = perturbed_test_data[target_domain][level]
                    perturbation_stats = perturbed_df.attrs.get(
                        'perturbation_stats',
                        {}
                    )

                    similarities = perturbed_df.attrs.get(
                        'replacement_similarities',
                        []
                    )

                    if similarities:
                        similarity_statistics = {
                            'count': len(similarities),
                            'mean': float(np.mean(similarities)),
                            'median': float(np.median(similarities)),
                            'std': float(np.std(similarities)),
                            'min': float(np.min(similarities)),
                            'max': float(np.max(similarities))
                        }
                    else:
                        similarity_statistics = {
                            'count': 0,
                            'mean': None,
                            'median': None,
                            'std': None,
                            'min': None,
                            'max': None
                        }

                    perturbation_statistics = {
                        **perturbation_stats,
                        'semantic_similarity': similarity_statistics
                    }

                    y_true = perturbed_df['label'].values
                    y_pred, y_proba = model_trainer.predict(
                        perturbed_df['text'].tolist()
                    )
                    perturbed_metrics = self.metrics_calculator.calculate_metrics(
                        y_true, y_pred, y_proba
                    )

                    # Calculate robustness metrics
                    robustness = self.metrics_calculator.calculate_robustness_metrics(
                        clean_metrics,
                        perturbed_metrics
                    )

                    if 'perturbation_success' not in perturbed_df.columns:
                        raise ValueError(
                            f"Missing 'perturbation_success' for "
                            f"{target_domain} - {level}"
                        )

                    prediction_flip = self.metrics_calculator.calculate_prediction_flip(
                        clean_predictions=(
                            results['cross_domain_perturbation'][key]['clean']['predictions']
                        ),
                        perturbed_predictions=y_pred.tolist(),
                        true_labels=y_true.tolist(),
                        successful_perturbations=(
                            perturbed_df['perturbation_success'].tolist()
                            if 'perturbation_success' in perturbed_df.columns
                            else None
                        )
                    )

                    results['cross_domain_perturbation'][key][level] = {
                        'metrics': perturbed_metrics,
                        'robustness_metrics': robustness,
                        'prediction_flip': prediction_flip,
                        'perturbation_statistics':
                            perturbation_statistics,
                        'predictions': y_pred.tolist(),
                        'probabilities': y_proba.tolist(),
                        'true_labels': y_true.tolist()
                    }

                    logger.info(
                        f"{source_domain} -> {target_domain} ({level}): "
                        f"Macro-F1={perturbed_metrics['macro_f1']:.4f}"
                    )

        logger.info("Cross-domain perturbation evaluation complete")

        # Save results
        self.save_results(results, filename='cross_domain_perturbation_results.json')

        return results

    def aggregate_results(
        self,
        results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Aggregate evaluation results into a summary DataFrame.

        Args:
            results: Results dictionary from run_complete_evaluation

        Returns:
            DataFrame with aggregated results
        """
        logger.info("Aggregating evaluation results")

        rows = []

        # In-domain results
        if 'in_domain' in results:
            for domain, domain_results in results['in_domain'].items():
                row = {
                    'evaluation_type': 'in_domain',
                    'source_domain': domain,
                    'target_domain': domain,
                    'perturbation_level': 'clean',
                    **domain_results['metrics']
                }
                rows.append(row)

        # Cross-domain results
        if 'cross_domain' in results:
            for (source, target), domain_results in results['cross_domain'].items():
                row = {
                    'evaluation_type': 'cross_domain',
                    'source_domain': source,
                    'target_domain': target,
                    'perturbation_level': 'clean',
                    **domain_results['metrics']
                }
                # Also include domain_shift if it exists
                if 'domain_shift' in domain_results:
                    row.update(domain_results['domain_shift'])
                rows.append(row)

        # Perturbation results
        if 'perturbation' in results:
            for domain, pert_results in results['perturbation'].items():
                for level, level_results in pert_results.items():
                    if level != 'clean':
                        row = {
                            'evaluation_type': 'perturbation',
                            'source_domain': domain,
                            'target_domain': domain,
                            'perturbation_level': level,
                            **level_results['metrics']
                        }
                        if 'robustness_metrics' in level_results:
                            row.update(level_results['robustness_metrics'])
                        if 'prediction_flip' in level_results:
                            flip = level_results['prediction_flip']
                            row.update({
                                'flip_count':
                                    flip['flip_count'],

                                'flip_rate':
                                    flip['flip_rate'],

                                'flip_rate_pct':
                                    flip['flip_rate_pct'],

                                'flip_0_to_1_count':
                                    flip['flip_0_to_1_count'],

                                'flip_1_to_0_count':
                                    flip['flip_1_to_0_count'],

                                'correct_to_incorrect':
                                    flip['flips']['correct_to_incorrect'],

                                'incorrect_to_correct':
                                    flip['flips']['incorrect_to_correct'],

                                'correct_to_incorrect_rate':
                                    flip['flips']['correct_to_incorrect_rate'],

                                'incorrect_to_correct_rate':
                                    flip['flips']['incorrect_to_correct_rate'],

                                # Successful perturbation analysis
                                'successful_perturbations':
                                    flip.get(
                                        'successful_perturbations'
                                    ),

                                'successful_perturbation_rate':
                                    flip.get(
                                        'successful_perturbation_rate'
                                    ),

                                'successful_perturbation_rate_pct':
                                    flip.get(
                                        'successful_perturbation_rate_pct'
                                    ),

                                'flip_count_successful_only':
                                    flip.get(
                                        'flip_count_successful_only'
                                    ),

                                'flip_rate_successful_only':
                                    flip.get(
                                        'flip_rate_successful_only'
                                    ),

                                'flip_rate_successful_only_pct':
                                    flip.get(
                                        'flip_rate_successful_only_pct'
                                    ),
                            })

                        if 'perturbation_statistics' in level_results:
                            stats = level_results[
                                'perturbation_statistics'
                            ]
                            row.update({
                                'changed_samples':
                                    stats.get(
                                        'changed_samples'
                                    ),
                                'sample_change_rate':
                                    stats.get(
                                        'sample_change_rate'
                                    ),
                                'mean_word_change_rate':
                                    stats.get(
                                        'mean_word_change_rate'
                                    ),
                                'median_word_change_rate':
                                    stats.get(
                                        'median_word_change_rate'
                                    ),
                                'mean_eligible_word_change_rate':
                                    stats.get(
                                        'mean_eligible_word_change_rate'
                                    ),
                                'mean_word_cosine_similarity':
                                    stats.get(
                                        'mean_word_cosine_similarity'
                                    ),
                                'min_word_cosine_similarity':
                                    stats.get(
                                        'min_word_cosine_similarity'
                                    ),
                                'max_word_cosine_similarity':
                                    stats.get(
                                        'max_word_cosine_similarity'
                                    ),
                            })

                            semantic_similarity = stats.get(
                                'semantic_similarity',
                                {}
                            )
                            row.update({
                                'similarity_count':
                                    semantic_similarity.get(
                                        'count'
                                    ),
                                'similarity_mean':
                                    semantic_similarity.get(
                                        'mean'
                                    ),
                                'similarity_median':
                                    semantic_similarity.get(
                                        'median'
                                    ),
                                'similarity_std':
                                    semantic_similarity.get(
                                        'std'
                                    ),
                                'similarity_min':
                                    semantic_similarity.get(
                                        'min'
                                    ),
                                'similarity_max':
                                    semantic_similarity.get(
                                        'max'
                                    ),
                            })

                        rows.append(row)

        # Cross-domain perturbation results
        if 'cross_domain_perturbation' in results:
            for key, domain_results in (
                results['cross_domain_perturbation'].items()
            ):
                source, target = key.split('_to_', 1)

                for level, level_results in domain_results.items():
                    metrics = level_results['metrics']
                    row = {
                        'evaluation_type':
                            'cross_domain_perturbation',
                        'source_domain': source,
                        'target_domain': target,
                        'perturbation_level': level,
                        **metrics
                    }
                    if level != 'clean':
                        if 'robustness_metrics' in level_results:
                            row.update(
                                level_results['robustness_metrics']
                            )
                        if 'prediction_flip' in level_results:
                            flip = level_results[
                                'prediction_flip'
                            ]
                            row.update({
                                'flip_count':
                                    flip['flip_count'],
                                'flip_rate':
                                    flip['flip_rate'],
                                'flip_rate_pct':
                                    flip['flip_rate_pct'],
                                'flip_0_to_1_count':
                                    flip['flip_0_to_1_count'],
                                'flip_1_to_0_count':
                                    flip['flip_1_to_0_count'],
                                'correct_to_incorrect':
                                    flip['flips'][
                                        'correct_to_incorrect'
                                    ],
                                'incorrect_to_correct':
                                    flip['flips'][
                                        'incorrect_to_correct'
                                    ],
                                'correct_to_incorrect_rate':
                                    flip['flips'][
                                        'correct_to_incorrect_rate'
                                    ],
                                'incorrect_to_correct_rate':
                                    flip['flips'][
                                        'incorrect_to_correct_rate'
                                    ],
                                'successful_perturbations':
                                    flip[
                                        'successful_perturbations'
                                    ],
                                'successful_perturbation_rate':
                                    flip.get(
                                        'successful_perturbation_rate'
                                    ),

                                'successful_perturbation_rate_pct':
                                    flip.get(
                                        'successful_perturbation_rate_pct'
                                    ),

                                'flip_count_successful_only':
                                    flip.get(
                                        'flip_count_successful_only'
                                    ),

                                'flip_rate_successful_only':
                                    flip.get(
                                        'flip_rate_successful_only'
                                    ),

                                'flip_rate_successful_only_pct':
                                    flip.get(
                                        'flip_rate_successful_only_pct'
                                    ),
                            })

                        if 'perturbation_statistics' in level_results:
                            stats = level_results[
                                'perturbation_statistics'
                            ]

                            row.update({
                                'changed_samples':
                                    stats.get(
                                        'changed_samples'
                                    ),
                                'sample_change_rate':
                                    stats.get(
                                        'sample_change_rate'
                                    ),
                                'mean_word_change_rate':
                                    stats.get(
                                        'mean_word_change_rate'
                                    ),
                                'median_word_change_rate':
                                    stats.get(
                                        'median_word_change_rate'
                                    ),
                                'mean_char_change_rate':
                                    stats.get(
                                        'mean_char_change_rate'
                                    ),
                                'mean_word_cosine_similarity':
                                    stats.get(
                                        'mean_word_cosine_similarity'
                                    ),
                                'min_word_cosine_similarity':
                                    stats.get(
                                        'min_word_cosine_similarity'
                                    ),
                                'max_word_cosine_similarity':
                                    stats.get(
                                        'max_word_cosine_similarity'
                                    ),
                            })

                            # ----------------------------------------------------
                            # Semantic similarity distribution
                            # ----------------------------------------------------

                            semantic_similarity = stats.get(
                                'semantic_similarity',
                                {}
                            )

                            row.update({
                                'similarity_count':
                                    semantic_similarity.get(
                                        'count'
                                    ),

                                'similarity_mean':
                                    semantic_similarity.get(
                                        'mean'
                                    ),

                                'similarity_median':
                                    semantic_similarity.get(
                                        'median'
                                    ),

                                'similarity_std':
                                    semantic_similarity.get(
                                        'std'
                                    ),

                                'similarity_min':
                                    semantic_similarity.get(
                                        'min'
                                    ),

                                'similarity_max':
                                    semantic_similarity.get(
                                        'max'
                                    ),
                            })

                    rows.append(row)

        df = pd.DataFrame(rows)

        # Save aggregated results
        output_path = os.path.join(self.output_dir, 'aggregated_results.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Aggregated results saved to {output_path}")

        return df

    def save_results(self, results: Dict[str, Any], filename: str = 'evaluation_results.json') -> str:
        """
        Serialise evaluation results to JSON, converting tuple keys and numpy
        types to JSON-safe equivalents.

        Args:
            results:  Results dict from run_complete_evaluation()
            filename: Output filename (default: 'evaluation_results.json')

        Returns:
            Full path of the saved file
        """
        from utils import make_json_serializable
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(make_json_serializable(results), f, indent=2)
        logger.info(f"Results saved to {output_path}")
        return output_path

    def generate_summary_report(
        self,
        results: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable summary report.

        Args:
            results: Results dictionary from evaluation

        Returns:
            Summary report as string
        """
        report_lines = [
            "=" * 80,
            "EVALUATION SUMMARY REPORT",
            "=" * 80,
            f"Timestamp: {results.get('timestamp', 'N/A')}",
            f"Domains: {', '.join(results.get('domains', []))}",
            ""
        ]

        # In-domain summary
        if 'in_domain' in results:
            report_lines.append("IN-DOMAIN EVALUATION:")
            report_lines.append("-" * 80)
            for domain, domain_results in results['in_domain'].items():
                metrics = domain_results['metrics']
                report_lines.append(
                    f"  {domain}: "
                    f"Macro-F1={metrics['macro_f1']:.4f}, "
                    f"Acc={metrics['accuracy']:.4f}, "
                    f"Macro-Prec={metrics['macro_precision']:.4f}, "
                    f"Macro-Rec={metrics['macro_recall']:.4f}"
                )
            report_lines.append("")

        # Cross-domain summary
        if 'cross_domain' in results:
            report_lines.append("CROSS-DOMAIN EVALUATION (Macro-F1):")
            report_lines.append("-" * 80)

            # Create performance matrix
            matrix = self.cross_domain_evaluator.create_performance_matrix(
                results['cross_domain'],
                metric='macro_f1'
            )
            report_lines.append(matrix.to_string())
            report_lines.append("")

        # Perturbation summary
        if 'perturbation' in results:
            report_lines.append("PERTURBATION ROBUSTNESS:")
            report_lines.append("-" * 80)
            for domain, pert_results in results['perturbation'].items():
                report_lines.append(f"  {domain}:")
                for level in ['clean', 'low', 'medium', 'high']:
                    if level in pert_results:
                        metrics = pert_results[level]['metrics']
                        macro_f1 = metrics['macro_f1']
                        report_lines.append(f"    {level.capitalize()}: Macro-F1={macro_f1:.4f}")

                        if level != 'clean' and 'robustness_metrics' in pert_results[level]:
                            rob = pert_results[level]['robustness_metrics']
                            drop = rob.get('macro_f1_drop', 0)
                            drop_pct = rob.get('macro_f1_drop_pct', 0)
                            report_lines.append(
                                f"      Drop: {drop:.4f} ({drop_pct:.2f}%)"
                            )
                report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save report
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Summary report saved to {report_path}")

        return report