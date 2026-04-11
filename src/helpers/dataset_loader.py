"""
Dataset loader module for text summarization evaluation.

Provides centralized dataset management with statistics, sampling, and reproducibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class DatasetInfo:
    """
    Container for dataset statistics and metadata.
    """

    def __init__(
        self,
        name: str,
        num_documents: int,
        text_column: str,
        summary_column: Optional[str] = None,
    ):
        self.name = name
        self.num_documents = num_documents
        self.text_column = text_column
        self.summary_column = summary_column
        self.avg_text_length = None
        self.avg_summary_length = None
        self.vocab_size = None
        self.compression_ratio = None
        self.texts = None
        self.summaries = None

    def compute_statistics(self, df: pd.DataFrame) -> None:
        """
        Compute dataset statistics from DataFrame.

        Args:
            df: DataFrame with text and optional summary columns
        """
        texts = df[self.text_column].astype(str).tolist()
        self.texts = texts

        # Average text length (words)
        text_lengths = [len(text.split()) for text in texts]
        self.avg_text_length = np.mean(text_lengths)

        # Vocabulary size (unique words)
        all_words = set()
        for text in texts:
            all_words.update(text.lower().split())
        self.vocab_size = len(all_words)

        # Summary statistics if available
        if self.summary_column and self.summary_column in df.columns:
            summaries = df[self.summary_column].astype(str).tolist()
            self.summaries = summaries
            summary_lengths = [len(s.split()) for s in summaries]
            self.avg_summary_length = np.mean(summary_lengths)

            # Compression ratio (avg summary length / avg text length)
            if self.avg_text_length > 0:
                self.compression_ratio = self.avg_summary_length / self.avg_text_length

    def __repr__(self) -> str:
        """String representation of dataset info."""
        lines = [
            f"Dataset: {self.name}",
            f"Documents: {self.num_documents}",
            f"Avg text length: {self.avg_text_length:.1f} words" if self.avg_text_length else "Avg text length: N/A",
            f"Vocab size: {self.vocab_size:,}" if self.vocab_size else "Vocab size: N/A",
        ]

        if self.avg_summary_length:
            lines.append(f"Avg summary length: {self.avg_summary_length:.1f} words")

        if self.compression_ratio:
            lines.append(f"Compression ratio: {self.compression_ratio:.2%}")

        return "\n".join(lines)


def load_cnn_dailymail(
    split: str = 'test',
    limit: Optional[int] = None,
    base_path: str = 'resources/cnn_dailymail',
    seed: int = RANDOM_SEED
) -> Tuple[DatasetInfo, pd.DataFrame]:
    """
    Load CNN/DailyMail dataset.

    Args:
        split: Dataset split ('train', 'validation', 'test')
        limit: Maximum number of documents to load (None for all)
        base_path: Base directory path for dataset
        seed: Random seed for sampling

    Returns:
        Tuple of (DatasetInfo, DataFrame)

    Raises:
        FileNotFoundError: If dataset file doesn't exist

    Note:
        CNN/DailyMail dataset needs to be downloaded separately.
        See: https://huggingface.co/datasets/cnn_dailymail
    """
    file_path = Path(base_path) / f"{split}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"CNN/DailyMail dataset not found at {file_path}.\n"
            f"Please download the dataset from:\n"
            f"https://huggingface.co/datasets/cnn_dailymail\n"
            f"Or use load_wiki_movies() or load_mtsamples() for available datasets."
        )

    # Load dataset
    df = pd.read_csv(file_path)

    # Check required columns
    if 'article' not in df.columns or 'highlights' not in df.columns:
        raise ValueError(
            f"Expected 'article' and 'highlights' columns in {file_path}, "
            f"found: {list(df.columns)}"
        )

    # Sample if limit specified
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    # Create dataset info
    info = DatasetInfo(
        name=f"CNN/DailyMail ({split})",
        num_documents=len(df),
        text_column='article',
        summary_column='highlights'
    )
    info.compute_statistics(df)

    return info, df


def load_wiki_movies(
    limit: Optional[int] = None,
    base_path: str = 'resources',
    seed: int = RANDOM_SEED,
    min_plot_length: int = 50
) -> Tuple[DatasetInfo, pd.DataFrame]:
    """
    Load Wikipedia movie plots dataset.

    This dataset contains 134K movie plots suitable for summarization tasks.

    Args:
        limit: Maximum number of documents to load (None for all)
        base_path: Base directory path for dataset
        seed: Random seed for sampling
        min_plot_length: Minimum plot length in words (filters out very short plots)

    Returns:
        Tuple of (DatasetInfo, DataFrame)

    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    file_path = Path(base_path) / "wiki_movie_plots_deduped.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Wiki movie plots dataset not found at {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Check required columns
    if 'Plot' not in df.columns:
        raise ValueError(
            f"Expected 'Plot' column in {file_path}, found: {list(df.columns)}"
        )

    # Filter out very short plots
    df = df[df['Plot'].astype(str).str.split().str.len() >= min_plot_length].copy()
    df = df.reset_index(drop=True)

    # Sample if limit specified
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    # Create dataset info (no reference summaries available)
    info = DatasetInfo(
        name="Wikipedia Movie Plots",
        num_documents=len(df),
        text_column='Plot',
        summary_column=None
    )
    info.compute_statistics(df)

    return info, df


def load_mtsamples(
    limit: Optional[int] = None,
    base_path: str = 'resources',
    seed: int = RANDOM_SEED,
    min_text_length: int = 100
) -> Tuple[DatasetInfo, pd.DataFrame]:
    """
    Load MTSamples medical transcriptions dataset.

    This dataset contains 5K medical transcriptions with descriptions.
    Can be used for domain-specific summarization evaluation.

    Args:
        limit: Maximum number of documents to load (None for all)
        base_path: Base directory path for dataset
        seed: Random seed for sampling
        min_text_length: Minimum transcription length in words

    Returns:
        Tuple of (DatasetInfo, DataFrame)

    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    file_path = Path(base_path) / "mtsamples.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"MTSamples dataset not found at {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Check required columns
    if 'transcription' not in df.columns or 'description' not in df.columns:
        raise ValueError(
            f"Expected 'transcription' and 'description' columns in {file_path}, "
            f"found: {list(df.columns)}"
        )

    # Filter out very short transcriptions
    df = df[df['transcription'].astype(str).str.split().str.len() >= min_text_length].copy()
    df = df.reset_index(drop=True)

    # Sample if limit specified
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    # Create dataset info (description can serve as reference summary)
    info = DatasetInfo(
        name="MTSamples Medical Transcriptions",
        num_documents=len(df),
        text_column='transcription',
        summary_column='description'
    )
    info.compute_statistics(df)

    return info, df


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    stratify_column: Optional[str] = None,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Create a stratified sample from a DataFrame.

    Args:
        df: Input DataFrame
        n: Number of samples to draw
        stratify_column: Column to stratify by (e.g., 'medical_specialty')
        seed: Random seed for reproducibility

    Returns:
        Stratified sample DataFrame
    """
    if stratify_column and stratify_column in df.columns:
        # Stratified sampling
        samples_per_category = n // df[stratify_column].nunique()

        sampled_dfs = []
        for category in df[stratify_column].unique():
            category_df = df[df[stratify_column] == category]
            sample_size = min(samples_per_category, len(category_df))
            sampled_dfs.append(
                category_df.sample(n=sample_size, random_state=seed)
            )

        result = pd.concat(sampled_dfs, ignore_index=True)

        # If we don't have enough samples, randomly sample more
        if len(result) < n:
            remaining = n - len(result)
            remaining_df = df[~df.index.isin(result.index)]
            additional = remaining_df.sample(
                n=min(remaining, len(remaining_df)),
                random_state=seed
            )
            result = pd.concat([result, additional], ignore_index=True)

        return result.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        # Simple random sampling
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets.

    Args:
        df: Input DataFrame
        test_size: Proportion of data for test set (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Calculate split point
    split_idx = int(len(df_shuffled) * (1 - test_size))

    train_df = df_shuffled.iloc[:split_idx].reset_index(drop=True)
    test_df = df_shuffled.iloc[split_idx:].reset_index(drop=True)

    return train_df, test_df


def get_available_datasets() -> List[str]:
    """
    List available datasets in the resources folder.

    Returns:
        List of available dataset names
    """
    available = []
    resources_path = Path('resources')

    if (resources_path / 'wiki_movie_plots_deduped.csv').exists():
        available.append('wiki_movies')

    if (resources_path / 'mtsamples.csv').exists():
        available.append('mtsamples')

    if (resources_path / 'cnn_dailymail' / 'test.csv').exists():
        available.append('cnn_dailymail')

    return available


if __name__ == '__main__':
    """
    Demo script showing dataset loading capabilities.
    """
    print("=" * 60)
    print("Dataset Loader Demo")
    print("=" * 60)

    # Check available datasets
    available = get_available_datasets()
    print(f"\nAvailable datasets: {', '.join(available)}")
    print()

    # Load and display info for each available dataset
    datasets_to_try = [
        ('wiki_movies', lambda: load_wiki_movies(limit=100)),
        ('mtsamples', lambda: load_mtsamples(limit=100)),
    ]

    for name, loader_func in datasets_to_try:
        if name in available:
            try:
                print(f"\n{'=' * 60}")
                print(f"Loading {name} (sample of 100 documents)...")
                print(f"{'=' * 60}")
                info, df = loader_func()
                print(info)
                print(f"\nDataFrame shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
            except Exception as e:
                print(f"Error loading {name}: {e}")

    print(f"\n{'=' * 60}")
    print("Demo completed!")
    print(f"{'=' * 60}")
