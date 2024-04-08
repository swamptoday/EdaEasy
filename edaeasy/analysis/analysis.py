import pandas as pd
from IPython.display import display, HTML
from typing import Optional, List

class Analysis:
    """Parent class for performing analysis on a df.

    Attributes:
        df (DataFrame, optional): The df to be analyzed.
        target (str, optional): The target column name.
        problem_type (str, optional): The type of problem, e.g., 'classification', 'regression'.
        path (str, optional): The path where images and reports will be stored.
    """

    def __init__(self, df: pd.DataFrame, target: Optional[str] = None,
                 problem_type: Optional[str] = None, path: Optional[str] = None, numerical: list[str] = None, categorical: list[str] = None):
        """Initialize Analysis with df, target column, problem type, and file path.

        Args:
            df (DataFrame, optional): The df to be analyzed.
            target (str, optional): The target column name.
            problem_type (str, optional): The type of problem, e.g., 'classification', 'regression'.
            path (str, optional): The file path of the df.
        """
        self._df = df
        self._target = target
        self._problem_type = problem_type
        self._path = path
        self._numerical = numerical
        self._categorical = categorical

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame):
        self._df = df

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target: str):
        self._target = target

    @property
    def problem_type(self):
        return self._problem_type

    @problem_type.setter
    def problem_type(self, problem_type: str):
        self._problem_type = problem_type

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: str):
        self._path = path

    @property
    def numerical(self):
        return self._numerical

    @numerical.setter
    def numerical(self, numerical: List[str]):
        self._numerical = numerical

    @property
    def categorical(self):
        return self._categorical

    @categorical.setter
    def categorical(self, categorical: List[str]):
        self._categorical = categorical

    def _display_sections(self, sections):
        """Displays a section with the given title by calling the corresponding method.

        Args:
            title (str): The title of the section.
            func (callable): The method to be called to display the content of the section.
        """
        for title, func in sections:
            display(HTML(f'<h2>{title}:</h2>'))
            func()
            display(HTML('</br>'))

    def generate_report(self):
        """Report"""
        pass
