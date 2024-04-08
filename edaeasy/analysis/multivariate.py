from IPython.display import display, HTML, Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List
from sklearn.feature_selection import mutual_info_regression
import re

from edaeasy.analysis.analysis import Analysis

class MultivariateAnalysis(Analysis):
    """Perform multivariate analysis on a dataset."""

    def __init__(self, df: pd.DataFrame, problem_type: str, target: str, path: str,
                 numerical: Optional[List[str]] = None, categorical: Optional[List[str]] = None) -> None:
        """
        Initialize MultivariateAnalysis instance.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            problem_type (str): Type of the problem, 'classification' or 'regression'.
            target (str): Name of the target variable.
            path (str): Path to save analysis results.
            numerical (Optional[List[str]], optional): List of numerical feature names. Defaults to None.
            categorical (Optional[List[str]], optional): List of categorical feature names. Defaults to None.
        """
        super().__init__(df=df, problem_type=problem_type, target=target, path=path,
                         numerical=numerical, categorical=categorical)
        self._divide()

    def generate_report(self) -> None:
        """Generate analysis report."""
        if self.problem_type == 'classification':
            sections = [
                ('Correlation Matrix', self.correlation),
                ('Distribution of numericals between target', self.dist_numerical_target),
                ('Distribution of categoricals between target', self.dist_categorical_target)
            ]
        else:
            sections = [
                ('Correlation Matrix', self.correlation),
                ('MI Score', self.mi_score),
                ('Relations between numericals', self.numerical_relations),
                ('Distribution of categoricals between target', self.dist_categorical_target)
            ]
        self._display_sections(sections)

    def _divide(self) -> None:
        """Divide features into numerical and categorical."""
        if self.numerical is None:
            numerical_columns = self.df.drop(columns=[self.target]).select_dtypes(include=['int64', 'float64']).columns
            filtered_numerical_columns = [col for col in numerical_columns if self.df[col].nunique() > 15]
            self.numerical = filtered_numerical_columns
        if self.categorical is None:
            categorical_columns = self.df.drop(columns=[self.target]).select_dtypes(exclude=['int64', 'float64']).columns
            additional_categorical_columns = [col for col in self.df.drop(columns=[self.target]).columns
                                               if (self.df[col].nunique() <= 15) and (self.df[col].dtype in ['int64', 'float64'])]
            self.categorical = list(set(categorical_columns).union(additional_categorical_columns))

    def correlation(self) -> None:
        """Calculate and display correlation matrix."""
        if self.problem_type == 'classification':
            correlation_matrix = self.df[self.numerical].corr()
        else:
            correlation_matrix = self.df[self.numerical + [self.target]].corr()
        self._plot_heatmap(correlation_matrix)
        display(Image(filename=f'{self.path}/correlation-matrix.png'))
        threshold = 0.6
        high_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) >= threshold:
                    high_correlation_pairs.append((correlation_matrix.index[i], correlation_matrix.columns[j],
                                                   correlation_matrix.iloc[i, j]))
        if len(high_correlation_pairs) > 0:
            self._high_correlation_dependency(high_correlation_pairs)

    def mi_score(self) -> None:
        """Calculate and display mutual information scores."""
        X = self.df[self.numerical].values
        y = self.df[self.target].values
        mi_scores = mutual_info_regression(X, y)
        self._plot_bar(mi_scores)
        display(Image(f'{self.path}/mi-bar.png'))

    def dist_numerical_target(self) -> None:
        """Display distribution of numerical features between target."""
        self._plot_bar_violin()
        display(Image(f'{self.path}/distribution-numerical-target.png'))

    def dist_categorical_target(self) -> None:
        """Display distribution of categorical features between target."""
        if self.problem_type == 'classification':
            self._plot_countplots()
            display(Image(f'{self.path}/distribution-catclass-target.png'))
        else:
            self._plot_boxplot()
            display(Image(f'{self.path}/distribution-catregr-target.png'))

    def numerical_relations(self) -> None:
        """Display relations between numerical features."""
        self._plot_pairplot()
        display(Image(f'{self.path}/pairplot.png'))

    def _high_correlation_dependency(self, high_correlation_pairs: List[Tuple[str, str, float]]) -> None:
        """Display dependencies in high correlation pairs."""
        display(HTML(f'<h2>Dependencies in high correlation pairs: </h2>'))
        for feat_1, feat_2, cor in high_correlation_pairs:
            self._plot_jointplot(feat_1, feat_2, cor)

            feat_1 = re.sub(r'[^\w\s]', '', feat_1).lower().replace(' ', '').replace('(', '').replace(')', '')
            feat_2 = re.sub(r'[^\w\s]', '', feat_2).lower().replace(' ', '').replace('(', '').replace(')', '')

            path_to = f'{self.path}/jointplot-{feat_1}-{feat_2}.png'

            display(Image(path_to))

    def _plot_heatmap(self, correlation_matrix: pd.DataFrame) -> None:
        """Plot heatmap of correlation matrix."""
        fig, axes = plt.subplots(figsize=(6, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='bone_r', fmt=".2f", ax=axes)
        axes.set_title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.path}/correlation-matrix.png')
        plt.close()

    def _plot_jointplot(self, feat_1: str, feat_2: str, cor: float) -> None:
        """Plot jointplot for two features."""
        sns.jointplot(data=self.df, x=feat_1, y=feat_2, kind='scatter')
        plt.title(f"Jointplot for '{feat_1}' and '{feat_2}': {round(cor, 2)} ", y=1.02)
        plt.tight_layout()

        feat_1 = re.sub(r'[^\w\s]', '', feat_1).lower().replace(' ', '').replace('(', '').replace(')', '')
        feat_2 = re.sub(r'[^\w\s]', '', feat_2).lower().replace(' ', '').replace('(', '').replace(')', '')

        path_to = f'{self.path}/jointplot-{feat_1}-{feat_2}.png'

        plt.savefig(path_to)
        plt.close()

    def _plot_bar(self, data: List[float]) -> None:
        """Plot bar chart of mutual information scores."""
        fig, axes = plt.subplots(figsize=(6, 6))
        sns.barplot(y=self.numerical, x=data, palette='bone_r', ax=axes, hue=self.numerical)
        axes.set_xlabel('Mutual Information Score')
        axes.set_ylabel('Numerical Features')
        axes.set_title('Mutual Information Scores between Numerical Features and Target')
        plt.savefig(f'{self.path}/mi-bar.png')
        plt.close()

    def _plot_bar_violin(self) -> None:
        """Plot bar and violin plots of numerical features vs target."""
        fig, axes = plt.subplots(nrows=len(self.numerical), ncols=2, figsize=(12, 5 * len(self.numerical)))
        for i, num_feature in enumerate(self.numerical):
            sns.boxplot(x=self.df[self.target], y=self.df[num_feature], ax=axes[i, 0], palette='bone_r',
                        hue=self.df[self.target])
            axes[i, 0].set_title(f'Barplot of {num_feature} vs {self.target}')
            sns.violinplot(x=self.df[self.target], y=self.df[num_feature], ax=axes[i, 1], palette='bone_r',
                           hue=self.df[self.target])
            axes[i, 1].set_title(f'Violinplot of {num_feature} vs {self.target}')
        plt.tight_layout()
        plt.savefig(f'{self.path}/distribution-numerical-target.png')
        plt.close()

    def _plot_countplots(self) -> None:
        """Plot count plots of categorical features vs target."""
        fig, axes = plt.subplots(nrows=len(self.categorical), ncols=1, figsize=(10, 5 * len(self.categorical)))
        if (len(self.categorical) > 1):
            for i, cat_feature in enumerate(self.categorical):
                sns.countplot(x=cat_feature, palette='bone_r', hue=self.target, data=self.df, ax=axes[i])
                axes[i].set_title(f'Countplot of {cat_feature} vs {self.target}')
        else:
            for i, cat_feature in enumerate(self.categorical):
                sns.countplot(x=cat_feature, palette='bone_r', hue=self.target, data=self.df, ax=axes)
                axes.set_title(f'Countplot of {cat_feature} vs {self.target}')

        plt.tight_layout()
        plt.savefig(f'{self.path}/distribution-catclass-target.png')
        plt.close()

    def _plot_pairplot(self) -> None:
        """Plot pairplot of numerical features vs target."""
        sns.pairplot(data=self.df[self.numerical + [self.target]], palette='bone_r', hue=self.target)
        plt.tight_layout()
        plt.savefig(f'{self.path}/pairplot.png')
        plt.close()

    def _plot_boxplot(self) -> None:
        """Plot boxplot of categorical features vs target."""
        fig, axes = plt.subplots(nrows=len(self.categorical), ncols=1, figsize=(10, 5 * len(self.categorical)))
        if (len(self.categorical) > 1):
            for i, cat_feature in enumerate(self.categorical):
                category_counts = self.df[cat_feature].value_counts()
                top_categories = category_counts.head(10).index.tolist()
                filtered_df = self.df[self.df[cat_feature].isin(top_categories)]
                sns.boxplot(y=cat_feature, x=self.target, data=filtered_df, ax=axes[i], palette='bone_r', hue=cat_feature)
                axes[i].set_title(f'Boxplot of {cat_feature} vs {self.target}')
                axes[i].tick_params(axis='x', rotation=45)
        else:
            for i, cat_feature in enumerate(self.categorical):
                category_counts = self.df[cat_feature].value_counts()
                top_categories = category_counts.head(10).index.tolist()
                filtered_df = self.df[self.df[cat_feature].isin(top_categories)]
                sns.boxplot(y=cat_feature, x=self.target, data=filtered_df, ax=axes, palette='bone_r', hue=cat_feature)
                axes.set_title(f'Boxplot of {cat_feature} vs {self.target}')
                axes.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.path}/distribution-catregr-target.png')