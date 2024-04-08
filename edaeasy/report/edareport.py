from typing import List
import pandas as pd
from IPython.display import display, HTML, Image


from edaeasy.analysis.analysis import Analysis
from edaeasy.analysis.basic import BasicAnalysis
from edaeasy.analysis.target import TargetAnalysis
from edaeasy.analysis.univariate import UnivariateAnalysis
from edaeasy.analysis.multivariate import MultivariateAnalysis

from edaeasy.errors.parameter import AnalysisParameterError

class EDAReport:
    def __init__(self, df: pd.DataFrame, problem_type: str = None, target: str = None, path: str = None, numerical: List[str] = None, categorical: List[str] = None) -> None:
        """
        Initialize the EDAReport object.

        Args:
            df: DataFrame
                The DataFrame for analysis.
            problem_type: str, optional
                The type of problem, e.g., 'classification' or 'regression'.
            target: str, optional
                The target variable.
            path: str, optional
                The path to store reports.
            numerical: list of str, optional
                The list of numerical features.
            categorical: list of str, optional
                The list of categorical features.
        """
        self.analysis = Analysis(df, target, problem_type, path, numerical, categorical)

    def set_df(self, df: pd.DataFrame) -> None:
        """Set the DataFrame."""
        self.analysis.set_df(df)

    def set_target(self, target: str) -> None:
        """Set the target."""
        self.analysis.set_target(target)

    def set_problem_type(self, problem_type: str) -> None:
        """Set the problem type."""
        self.analysis.set_problem_type(problem_type)

    def set_path(self, path: str) -> None:
        """Set the path."""
        self.analysis.set_path(path)

    def set_numerical(self, numerical: List[str]) -> None:
        """Set the numerical."""
        self.analysis.set_numerical(numerical)

    def set_categorical(self, categorical: List[str]) -> None:
        """Set the categorical."""
        self.analysis.set_categorical(categorical)

    def perform_basic(self) -> None:
        """
        Perform basic analysis and generate a report.

        Raises:
            AnalysisParameterError: If required parameters are missing.
        """
        self._check_analysis_parameters(['df'])
        basic_analysis = BasicAnalysis(df=self.analysis.df)
        basic_analysis.generate_report()

    def perform_target(self) -> None:
        """
        Perform target analysis and generate a report.

        Raises:
            AnalysisParameterError: If required parameters are missing.
        """
        self._check_analysis_parameters(['df', 'problem_type', 'target', 'path'])
        target_analysis = TargetAnalysis(df=self.analysis.df, target=self.analysis.target,
                                          path=self.analysis.path, problem_type=self.analysis.problem_type)
        target_analysis.generate_report()

    def perform_univariate(self) -> None:
        """
        Perform univariate analysis and generate a report.

        Raises:
            AnalysisParameterError: If required parameters are missing.
        """
        self._check_analysis_parameters(['df', 'problem_type', 'target', 'path'])
        if self.analysis.numerical is None and self.analysis.categorical is None:
            print("Automatically setting numerical and categorical variables.")
        univariate_analysis = UnivariateAnalysis(df=self.analysis.df, target=self.analysis.target,
                                                  problem_type=self.analysis.problem_type, path=self.analysis.path,
                                                  numerical=self.analysis.numerical, categorical=self.analysis.categorical)
        univariate_analysis.generate_report()

    def perform_multivariate(self) -> None:
        """
        Perform multivariate analysis and generate a report.

        Raises:
            AnalysisParameterError: If required parameters are missing.
        """
        self._check_analysis_parameters(['df', 'problem_type', 'target', 'path'])
        if self.analysis.numerical is None and self.analysis.categorical is None:
            print("Automatically setting numerical and categorical variables.")
        multivariate_analysis = MultivariateAnalysis(df=self.analysis.df, target=self.analysis.target,
                                                      problem_type=self.analysis.problem_type, path=self.analysis.path,
                                                      numerical=self.analysis.numerical, categorical=self.analysis.categorical)
        multivariate_analysis.generate_report()

    def perform(self) -> None:
        """
        Perform all analysis steps and generate reports.

        Raises:
            AnalysisParameterError: If required parameters are missing.
        """
        self._check_analysis_parameters(['df', 'problem_type', 'target', 'path'])
        if self.analysis.numerical is None and self.analysis.categorical is None:
            print("Automatically setting numerical and categorical variables.")
        basic_analysis = BasicAnalysis(df=self.analysis.df)
        target_analysis = TargetAnalysis(df=self.analysis.df, target=self.analysis.target,
                                          path=self.analysis.path, problem_type=self.analysis.problem_type)
        univariate_analysis = UnivariateAnalysis(df=self.analysis.df, target=self.analysis.target,
                                                  problem_type=self.analysis.problem_type, path=self.analysis.path,
                                                  numerical=self.analysis.numerical, categorical=self.analysis.categorical)
        multivariate_analysis = MultivariateAnalysis(df=self.analysis.df, target=self.analysis.target,
                                                      problem_type=self.analysis.problem_type, path=self.analysis.path,
                                                      numerical=self.analysis.numerical, categorical=self.analysis.categorical)

        basic_analysis.generate_report()
        target_analysis.generate_report()
        univariate_analysis.generate_report()
        multivariate_analysis.generate_report()

    def _check_analysis_parameters(self, required_params: List[str]) -> None:
        """
        Check if required analysis parameters are provided.

        Args:
            required_params: list of str
                List of required parameters.

        Raises:
            AnalysisParameterError: If required parameters are missing or if parameters are not valid.
        """
        if (self.analysis.problem_type != 'regression' and self.analysis.problem_type != 'classification'):
            raise AnalysisParameterError(f"Problem type is not compatible. Use 'regression' or 'classification'.")
                
        missing_params = [param for param in required_params if getattr(self.analysis, param) is None]
        if missing_params:
            error_message = "Missing parameters: " + ", ".join(missing_params)
            raise AnalysisParameterError(error_message)
        

        if self.analysis.numerical:
            for feature in self.analysis.numerical:
                if feature not in self.analysis.df.columns:
                    raise AnalysisParameterError(f"Feature '{feature}' not found in DataFrame.")
                if self.analysis.df[feature].dtype not in ['int64', 'float64']:
                    raise AnalysisParameterError(f"Feature '{feature}' is not of type 'int64' or 'float64'. You cannot set it to numerical.")

        if self.analysis.categorical:
            for feature in self.analysis.categorical:
                if feature not in self.analysis.df.columns:
                    raise AnalysisParameterError(f"Feature '{feature}' not found in DataFrame.")