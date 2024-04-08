from IPython.display import display, HTML, Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from edaeasy.analysis.analysis import Analysis

class TargetAnalysis(Analysis):
    """Analyze target variable in a dataset."""

    def __init__(self, df: pd.DataFrame, problem_type: str, target: str, path: str) -> None:
        """
        Initialize TargetAnalysis instance.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            problem_type (str): Type of the problem, 'classification' or 'regression'.
            target (str): Name of the target variable.
            path (str): Path to save analysis results.
        """
        super().__init__(df=df, problem_type=problem_type, target=target, path=path)

    def generate_report(self) -> None:
        """Generate analysis report for the target variable."""
        display(HTML('<h1>Target Analysis</h1>'))
        if self.problem_type == 'classification':
            sections = [
                ('Unique values count', self.unique),
                ('Distribution', self.bardist)
            ]
        else:
            sections = [
                ('Describe statistics', self.describe),
                ('Boxplot/Violinplot', self.boxviolin),
                ('Histplot', self.histdist),
                ('Outliers', self.outliers)
            ]
        self._display_sections(sections)

    def unique(self) -> None:
        """Display unique values count of the target variable."""
        vc = self.df[self.target].value_counts()
        vc_df = pd.DataFrame({'Value': vc.index, 'Count': vc.values})
        display(vc_df)

    def bardist(self) -> None:
        """Display bar plot of target variable distribution."""
        self._plot_count()
        display(Image(filename=f'{self.path}/target-count.png'))

    def describe(self) -> None:
        """Display descriptive statistics of the target variable."""
        desc = self.df[self.target].describe()
        desc_df = pd.DataFrame({'Stats': desc.index, 'Value': desc.values})
        display(desc_df)
        self._plot_descriptive_statistics(desc)
        display(Image(filename=f'{self.path}/target-descriptive-statistics.png'))

    def boxviolin(self) -> None:
        """Display box and violin plots of the target variable."""
        self._plot_bv()
        display(Image(filename=f'{self.path}/target-box-violin-plot.png'))

    def histdist(self) -> None:
        """Display histogram plot of the target variable."""
        self._plot_histplot()
        display(Image(filename=f'{self.path}/target-histplot.png'))

    def outliers(self) -> None:
        """Detect and display outliers in the target variable."""
        num_outliers, percentage_outliers = self._detect_outliers(self.df[self.target])
        display(HTML(f'<p>Number of outliers in {self.target}: {num_outliers}</p>'))
        display(HTML(f'<p>Percentage of outliers in {self.target}: {round(percentage_outliers * 100, 2)}%</p>'))

    def _plot_bv(self) -> None:
        """Plot box and violin plots of the target variable."""
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.boxplot(x=self.df[self.target], ax=ax[0], color='#8091a0')
        ax[0].set_title(f"Boxplot of {self.target}")
        sns.violinplot(x=self.df[self.target], ax=ax[1], color='#9fbcbf')
        ax[1].set_title(f"Violinplot of {self.target}")
        plt.tight_layout()
        plt.savefig(f'{self.path}/target-box-violin-plot.png')
        plt.close()

    def _plot_histplot(self) -> None:
        """Plot histogram plot of the target variable."""
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.histplot(x=self.df[self.target], ax=ax, kde=True, color='#404059')
        ax.set_title(f"Hisplot of {self.target}")
        plt.tight_layout()
        plt.savefig(f'{self.path}/target-histplot.png')
        plt.close()

    def _plot_count(self) -> None:
        """Plot bar plot of target variable distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        plot = sns.countplot(x=self.target, hue=self.target, palette='bone_r', data=self.df)
        total = len(self.df[self.target])
        for p in plot.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 1,
                    '{:.1f}%'.format((height / total) * 100),
                    ha="center")
        ax.set_xlabel(self.target)
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Value Counts')
        plt.tight_layout()
        plt.savefig(f'{self.path}/target-count.png')
        plt.close()

    def _plot_descriptive_statistics(self, describe_result: pd.Series) -> None:
        """Plot descriptive statistics of the target variable."""
        fig, ax = plt.subplots(figsize=(5, 5))

        mean = describe_result['mean']
        std = describe_result['std']
        minimum = describe_result['min']
        maximum = describe_result['max']
        percentile_25 = describe_result['25%']
        median = describe_result['50%']
        percentile_75 = describe_result['75%']

        ax.bar(self.target, mean, color='#9fbcbf', label='Mean')
        ax.errorbar(self.target, mean, yerr=std, fmt='o', color='black', label='Std')
        ax.scatter(self.target, minimum, color='red', label='Min')
        ax.scatter(self.target, maximum, color='green', label='Max')
        ax.scatter(self.target, percentile_25, color='orange', label='25th percentile')
        ax.scatter(self.target, median, color='purple', label='50th percentile (Median)')
        ax.scatter(self.target, percentile_75, color='brown', label='75th percentile')

        ax.set_title(f'Descriptive Statistics for {self.target}')
        ax.set_ylabel('Values')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.path}/target-descriptive-statistics.png')
        plt.close()

    def _detect_outliers(self, data: pd.Series) -> Tuple[int, float]:
        """Detect outliers in the target variable."""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        return np.sum(outliers), np.mean(outliers)