from IPython.display import display, HTML, Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List
import math

from edaeasy.analysis.analysis import Analysis

class UnivariateAnalysis(Analysis):
    """Perform univariate analysis on a dataset."""

    def __init__(self, df: pd.DataFrame, problem_type: str, target: str, path: str,
                 numerical: Optional[List[str]] = None, categorical: Optional[List[str]] = None) -> None:
        """
        Initialize UnivariateAnalysis instance.

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
        display(HTML('<h1>Univariate Analysis</h1>'))
        sections = [
            ('Descriptive statistics', self.describe),
            ('Numerical features distribution', self.dist_numerical),
            ('Outliers', self.outliers),
            ('Missing in numerical', self.missing_numerical),
            ('Categorical features distribution', self.dist_categorical),
            ('Missing in categorical', self.missing_categorical)
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

    def describe(self) -> None:
        """Display descriptive statistics."""
        desc_st = self.df[self.numerical].describe()
        display(HTML(desc_st.to_html()))
        self._plot_descriptive_statistics(desc_st)
        display(Image(filename=f'{self.path}/descriptive-statistics.png'))

    def dist_numerical(self) -> None:
        """Display distribution plots of numerical features."""
        self._plot_box_violin_hist()
        display(Image(f'{self.path}/box_violin_hist_plots.png'))

    def missing_numerical(self) -> None:
        """Display missing data in numerical features."""
        if self.df[self.numerical].isnull().values.any():
            missing_data = self.df[self.numerical].isnull()
            missing_amount = pd.DataFrame(missing_data.sum(), columns=['Amount']).to_html()
            display(HTML(missing_amount))
            self._plot_heatmap(missing_data, True)
            display(Image(filename=f'{self.path}/miss-num-heatplot.png'))
        else:
            display(HTML('<h3>No missing data in numerical features!</h3>'))

    def dist_categorical(self) -> None:
        """Display distribution plots of categorical features."""
        display(HTML('<h3>If categorical feature contains more than 10 subcategories, plot will contain only 10 with largest amount'))
        self._plot_countplot()
        display(Image(f'{self.path}/countplot.png'))

    def missing_categorical(self) -> None:
        """Display missing data in categorical features."""
        if self.df[self.categorical].isnull().values.any():
            missing_data = self.df[self.categorical].isnull()
            missing_amount = pd.DataFrame(missing_data.sum(), columns=['Amount']).to_html()
            display(HTML(missing_amount))
            self._plot_heatmap(missing_data, False)
            display(Image(filename=f'{self.path}/miss-cat-heatplot.png'))
        else:
            display(HTML('<h3>No missing data in categorical features!</h3>'))

    def outliers(self) -> None:
        """Detect and display outliers."""
        for column in self.numerical:
            num_outliers, percentage_outliers = self.detect_outliers(self.df[column])
            print(f"Number of outliers in {column}:", num_outliers)
            print(f"Percentage of outliers {column}:", percentage_outliers * 100, "%\n")

    def _plot_descriptive_statistics(self, desc_st: pd.DataFrame) -> None:
        """Plot descriptive statistics."""
        desc_st_sorted = desc_st.loc[:, desc_st.loc['mean'].sort_values(ascending=False).index]
        col_amount = math.ceil(len(self.numerical) / 3)
        rows = math.ceil(len(self.numerical) / 9)
        fig, axes = plt.subplots(nrows=rows, ncols=col_amount, figsize=(15, 5*rows))
        columns_per_subplot = 3
        if (rows > 1 or col_amount > 1):
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                start_idx = i * columns_per_subplot
                end_idx = start_idx + columns_per_subplot
                ax.bar(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['mean'][start_idx:end_idx], color='#9fbcbf', label='Mean')
                ax.errorbar(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['mean'][start_idx:end_idx], yerr=desc_st_sorted.loc['std'][start_idx:end_idx], fmt='o', color='black', label='Std')
                ax.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['min'][start_idx:end_idx], color='red', label='Min')
                ax.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['max'][start_idx:end_idx], color='green', label='Max')
                ax.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['25%'][start_idx:end_idx], color='orange', label='25th percentile')
                ax.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['50%'][start_idx:end_idx], color='purple', label='50th percentile (Median)')
                ax.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['75%'][start_idx:end_idx], color='brown', label='75th percentile')
                ax.set_title(f'Descriptive Statistics for Columns {start_idx+1} to {end_idx}')
                ax.set_ylabel('Values')
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
                ax.grid(True)
        else:
            start_idx = 0 * columns_per_subplot
            end_idx = start_idx + columns_per_subplot
            axes.bar(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['mean'][start_idx:end_idx], color='#9fbcbf', label='Mean')
            axes.errorbar(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['mean'][start_idx:end_idx], yerr=desc_st_sorted.loc['std'][start_idx:end_idx], fmt='o', color='black', label='Std')
            axes.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['min'][start_idx:end_idx], color='red', label='Min')
            axes.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['max'][start_idx:end_idx], color='green', label='Max')
            axes.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['25%'][start_idx:end_idx], color='orange', label='25th percentile')
            axes.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['50%'][start_idx:end_idx], color='purple', label='50th percentile (Median)')
            axes.scatter(desc_st_sorted.columns[start_idx:end_idx], desc_st_sorted.loc['75%'][start_idx:end_idx], color='brown', label='75th percentile')
            axes.set_title(f'Descriptive Statistics for Columns {start_idx+1} to {end_idx}')
            axes.set_ylabel('Values')
            axes.tick_params(axis='x', rotation=45)
            axes.legend()
            axes.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.path}/descriptive-statistics.png')
        plt.close()

    def _plot_box_violin_hist(self) -> None:
        """Plot box, violin, and histogram plots."""
        cols = 3
        rows = len(self.numerical)
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 3*rows))
        axes = axes.flatten()
        for i, data in enumerate(self.numerical):
            sns.boxplot(x=self.df[data], ax=axes[3*i], color='#9fbcbf')
            axes[3*i].set_title(f"Boxplot of {data}")
            sns.violinplot(x=self.df[data], ax=axes[3*i+1], color='#9fbcbf')
            axes[3*i+1].set_title(f"Violinplot of {data}")
            sns.histplot(x=self.df[data], ax=axes[3*i+2], kde=True, color='#9fbcbf')
            axes[3*i+2].set_title(f"Histogram of {data}")
        plt.tight_layout()
        plt.savefig(f'{self.path}/box_violin_hist_plots.png')
        plt.close()

    def _plot_heatmap(self, missing: pd.DataFrame, numerical: bool) -> None:
        """Plot heatmap of missing data."""
        if numerical:
            subpath = 'num'
        else:
            subpath = 'cat'
        fig, axes = plt.subplots(figsize=(15, 6))
        sns.heatmap(missing.transpose(), cmap='bone_r', ax=axes)
        axes.set_title('Missing data')
        plt.tight_layout()
        plt.savefig(f'{self.path}/miss-{subpath}-heatplot.png')
        plt.close()

    def _plot_countplot(self) -> None:
        """Plot count plots."""
        if len(self.categorical) < 3:
            cols = len(self.categorical)
        else:
            cols = 2
        rows = math.ceil(len(self.categorical) / cols)
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 8*rows))

        if (rows > 1 or cols > 1):
            axes = axes.flatten()

            for i in range(len(self.categorical), rows*cols):
                fig.delaxes(axes[i])

            for i, data in enumerate(self.categorical):
                category_counts = self.df[data].value_counts()
                if len(category_counts) > 10:
                    category_counts = category_counts.head(10)
                total = category_counts.sum()
                bars = sns.countplot(x=self.df[data], palette='bone_r', ax=axes[i], order=category_counts.index, hue=self.df[data])
                axes[i].set_title(f'Countplot of {data}')
                axes[i].tick_params(axis='x', rotation=45)
                for bar in bars.patches:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width() / 2., height, '{:.1f}%'.format((height / total) * 100), ha='center', va='bottom')
        else:
            for i, data in enumerate(self.categorical):
                category_counts = self.df[data].value_counts()
                if len(category_counts) > 10:
                    category_counts = category_counts.head(10)
                total = category_counts.sum()
                bars = sns.countplot(x=self.df[data], palette='bone_r', ax=axes, order=category_counts.index, hue=self.df[data])
                axes.set_title(f'Countplot of {data}')
                axes.tick_params(axis='x', rotation=45)
                for bar in bars.patches:
                    height = bar.get_height()
                    axes.text(bar.get_x() + bar.get_width() / 2., height, '{:.1f}%'.format((height / total) * 100), ha='center', va='bottom')

        plt.savefig(f'{self.path}/countplot.png')
        plt.close()

    def detect_outliers(self, data: pd.Series) -> Tuple[int, float]:
        """Detect outliers."""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        return np.sum(outliers), np.mean(outliers)