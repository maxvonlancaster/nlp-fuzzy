"""
Configuration management for text summarization methods.

Provides hyperparameter configuration classes with defaults and serialization support.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from pathlib import Path

# Global random seed for reproducibility
RANDOM_SEED = 42


@dataclass
class BaseConfig:
    """
    Base configuration class for all summarization methods.
    """
    n_sentences: int = 3  # Number of sentences to extract
    random_seed: int = RANDOM_SEED

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __str__(self) -> str:
        """String representation of configuration."""
        lines = [f"{self.__class__.__name__}:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class KMeansConfig(BaseConfig):
    """
    Configuration for K-Means clustering summarization.

    Justification for defaults:
    - n_clusters=3: Matches typical summary length, balances coverage vs redundancy
    - embedding_model: SBERT provides strong sentence-level semantics
    - use_pca: False by default (ablation studies will test this)
    - pca_components: 50 preserves most variance while reducing dimensionality
    """
    n_clusters: Optional[int] = None  # If None, defaults to n_sentences
    embedding_model: str = "all-MiniLM-L6-v2"  # SBERT model
    use_pca: bool = False  # Dimensionality reduction
    pca_components: int = 50  # PCA target dimensions
    kmeans_init: str = "k-means++"  # Initialization method
    kmeans_n_init: int = 10  # Number of initializations
    kmeans_max_iter: int = 300  # Maximum iterations

    def __post_init__(self):
        """Set n_clusters to n_sentences if not specified."""
        if self.n_clusters is None:
            self.n_clusters = self.n_sentences


@dataclass
class KohonenConfig(BaseConfig):
    """
    Configuration for Kohonen Self-Organizing Map (SOM) summarization.

    Justification for defaults:
    - som_size: Auto-computed as sqrt(n_sentences) for adaptive grid
    - vector_size: 100 balances expressiveness and computational cost
    - sigma: 1.0 standard neighborhood size for SOM
    - learning_rate: 0.5 typical for SOM training
    - iterations: 1000 sufficient for convergence on small datasets
    """
    som_x: Optional[int] = None  # Grid width (auto-computed if None)
    som_y: Optional[int] = None  # Grid height (auto-computed if None)
    vector_size: int = 100  # Word2Vec embedding dimension
    word2vec_window: int = 5  # Word2Vec context window
    word2vec_min_count: int = 1  # Minimum word frequency
    word2vec_epochs: int = 100  # Word2Vec training epochs
    sigma: float = 1.0  # SOM neighborhood radius
    learning_rate: float = 0.5  # SOM learning rate
    iterations: int = 1000  # SOM training iterations
    use_pca: bool = False  # Dimensionality reduction before SOM

    def __post_init__(self):
        """Auto-compute SOM grid size if not specified."""
        if self.som_x is None or self.som_y is None:
            import math
            size = max(3, int(math.ceil(math.sqrt(self.n_sentences * 2))))
            self.som_x = size
            self.som_y = size


@dataclass
class TextRankConfig(BaseConfig):
    """
    Configuration for TextRank graph-based summarization.

    Justification for defaults:
    - similarity_metric: 'tfidf' is classic TextRank approach
    - damping_factor: 0.85 standard for PageRank algorithms
    - max_iter: 100 sufficient for convergence
    - tol: 1e-4 standard convergence tolerance
    """
    similarity_metric: str = "tfidf"  # 'tfidf' or 'embeddings'
    damping_factor: float = 0.85  # PageRank damping factor
    max_iter: int = 100  # Maximum PageRank iterations
    tol: float = 1e-4  # Convergence tolerance
    ngram_range: tuple = (1, 5)  # TF-IDF n-gram range
    stop_words: str = "english"  # Stopwords for TF-IDF
    use_idf: bool = True  # Use IDF weighting
    smooth_idf: bool = True  # Smooth IDF weights


@dataclass
class LexRankConfig(BaseConfig):
    """
    Configuration for LexRank graph-based summarization.

    Justification for defaults:
    - similarity_threshold: 0.1 standard LexRank threshold
    - embedding_model: SBERT for semantic similarity
    - use_continuous: True for continuous similarity (vs binary edges)
    """
    similarity_threshold: float = 0.1  # Minimum similarity for edge
    embedding_model: str = "all-MiniLM-L6-v2"  # SBERT model
    use_continuous: bool = True  # Use continuous similarity weights
    damping_factor: float = 0.85  # PageRank damping factor
    max_iter: int = 100  # Maximum PageRank iterations
    tol: float = 1e-4  # Convergence tolerance


@dataclass
class BaselineConfig(BaseConfig):
    """
    Configuration for baseline methods (Lead-N, Random).

    Justification for defaults:
    - lead_n: Simple first-N sentences baseline
    - random: Random selection baseline for lower bound
    """
    method: str = "lead"  # 'lead' or 'random'

    def __post_init__(self):
        """Validate method type."""
        if self.method not in ['lead', 'random']:
            raise ValueError(f"Baseline method must be 'lead' or 'random', got: {self.method}")


@dataclass
class EvaluationConfig(BaseConfig):
    """
    Configuration for evaluation settings.
    """
    rouge_metrics: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    use_stemmer: bool = True  # Use stemming in ROUGE
    bootstrap_samples: int = 1000  # For confidence intervals
    confidence_level: float = 0.95  # Confidence interval level
    significance_test: str = "paired_t"  # 'paired_t' or 'wilcoxon'


class ConfigManager:
    """
    Manager for loading and saving multiple configurations.
    """

    def __init__(self):
        self.configs: Dict[str, BaseConfig] = {}

    def add_config(self, name: str, config: BaseConfig) -> None:
        """Add a configuration with a given name."""
        self.configs[name] = config

    def get_config(self, name: str) -> BaseConfig:
        """Get configuration by name."""
        if name not in self.configs:
            raise KeyError(f"Configuration '{name}' not found")
        return self.configs[name]

    def save_all(self, directory: str) -> None:
        """Save all configurations to a directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        for name, config in self.configs.items():
            filepath = dir_path / f"{name}.json"
            config.to_json(str(filepath))

    def load_all(self, directory: str) -> None:
        """Load all configurations from a directory."""
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Configuration directory not found: {directory}")

        # Mapping of config names to classes
        config_classes = {
            'kmeans': KMeansConfig,
            'kohonen': KohonenConfig,
            'textrank': TextRankConfig,
            'lexrank': LexRankConfig,
            'baseline_lead': BaselineConfig,
            'baseline_random': BaselineConfig,
            'evaluation': EvaluationConfig,
        }

        for config_file in dir_path.glob("*.json"):
            name = config_file.stem
            if name in config_classes:
                config_class = config_classes[name]
                self.configs[name] = config_class.from_json(str(config_file))

    def __str__(self) -> str:
        """String representation of all configurations."""
        lines = ["ConfigManager:"]
        for name, config in self.configs.items():
            lines.append(f"\n{name}:")
            lines.append(str(config))
        return "\n".join(lines)


def get_default_configs() -> ConfigManager:
    """
    Create a ConfigManager with default configurations for all methods.

    Returns:
        ConfigManager with default configurations
    """
    manager = ConfigManager()

    # Add default configurations for each method
    manager.add_config('kmeans', KMeansConfig())
    manager.add_config('kohonen', KohonenConfig())
    manager.add_config('textrank', TextRankConfig())
    manager.add_config('lexrank', LexRankConfig())
    manager.add_config('baseline_lead', BaselineConfig(method='lead'))
    manager.add_config('baseline_random', BaselineConfig(method='random'))
    manager.add_config('evaluation', EvaluationConfig())

    return manager


def create_hyperparameter_table() -> str:
    """
    Generate a markdown table with all hyperparameters and justifications.

    Returns:
        Markdown-formatted hyperparameter table
    """
    lines = ["# Hyperparameter Configuration\n"]
    lines.append("## K-Means Configuration\n")
    lines.append("| Parameter | Default | Justification |")
    lines.append("|-----------|---------|---------------|")
    lines.append("| n_clusters | 3 | Matches typical summary length, balances coverage vs redundancy |")
    lines.append("| embedding_model | all-MiniLM-L6-v2 | SBERT provides strong sentence-level semantics |")
    lines.append("| use_pca | False | Tested in ablation studies |")
    lines.append("| pca_components | 50 | Preserves most variance while reducing dimensionality |")
    lines.append("| kmeans_init | k-means++ | Better initialization than random |")
    lines.append("| kmeans_n_init | 10 | Multiple initializations for stability |")
    lines.append("| kmeans_max_iter | 300 | Sufficient for convergence |")

    lines.append("\n## Kohonen (SOM) Configuration\n")
    lines.append("| Parameter | Default | Justification |")
    lines.append("|-----------|---------|---------------|")
    lines.append("| som_size | auto (√n) | Adaptive grid based on expected clusters |")
    lines.append("| vector_size | 100 | Balances expressiveness and computational cost |")
    lines.append("| word2vec_window | 5 | Standard context window for Word2Vec |")
    lines.append("| sigma | 1.0 | Standard neighborhood size for SOM |")
    lines.append("| learning_rate | 0.5 | Typical for SOM training |")
    lines.append("| iterations | 1000 | Sufficient for convergence |")

    lines.append("\n## TextRank Configuration\n")
    lines.append("| Parameter | Default | Justification |")
    lines.append("|-----------|---------|---------------|")
    lines.append("| similarity_metric | tfidf | Classic TextRank approach |")
    lines.append("| damping_factor | 0.85 | Standard for PageRank algorithms |")
    lines.append("| max_iter | 100 | Sufficient for convergence |")
    lines.append("| tol | 1e-4 | Standard convergence tolerance |")
    lines.append("| ngram_range | (1, 5) | Captures phrases up to 5 words |")

    lines.append("\n## LexRank Configuration\n")
    lines.append("| Parameter | Default | Justification |")
    lines.append("|-----------|---------|---------------|")
    lines.append("| similarity_threshold | 0.1 | Standard LexRank threshold |")
    lines.append("| embedding_model | all-MiniLM-L6-v2 | SBERT for semantic similarity |")
    lines.append("| use_continuous | True | Continuous similarity weights |")
    lines.append("| damping_factor | 0.85 | Standard for PageRank |")

    lines.append("\n## Global Settings\n")
    lines.append("| Parameter | Default | Justification |")
    lines.append("|-----------|---------|---------------|")
    lines.append("| random_seed | 42 | Ensures reproducibility across all methods |")
    lines.append("| n_sentences | 3 | Standard summary length for evaluation |")

    return "\n".join(lines)


if __name__ == '__main__':
    """
    Demo script showing configuration management capabilities.
    """
    print("=" * 60)
    print("Configuration Management Demo")
    print("=" * 60)

    # Create default configurations
    manager = get_default_configs()
    print("\nDefault configurations loaded:")
    print(manager)

    # Save configurations to file
    print("\n" + "=" * 60)
    print("Saving configurations to 'configs/' directory...")
    manager.save_all('configs')
    print("Configurations saved!")

    # Generate hyperparameter table
    print("\n" + "=" * 60)
    print("Hyperparameter Table (first 500 chars):")
    print("=" * 60)
    table = create_hyperparameter_table()
    print(table[:500] + "...\n")

    # Show how to modify a configuration
    print("=" * 60)
    print("Example: Modifying K-Means configuration")
    print("=" * 60)
    kmeans_config = manager.get_config('kmeans')
    print("\nOriginal:")
    print(kmeans_config)

    # Modify and show
    kmeans_config.n_clusters = 5
    kmeans_config.use_pca = True
    print("\nModified:")
    print(kmeans_config)

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
