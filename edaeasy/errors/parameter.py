from typing import Optional, List


class AnalysisParameterError(Exception):
    """Exception raised for errors in analysis parameters.

    Attributes:
        message -- explanation of the error
        missing_params -- list of missing parameters
    """

    def __init__(self, message: str, missing_params: Optional[List[str]] = None) -> None:
        self.message: str = message
        self.missing_params: Optional[List[str]] = missing_params

        # Call the base class constructor with the parameters it needs
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.missing_params:
            return f"{self.message}. Missing parameters: {', '.join(self.missing_params)}"
        else:
            return self.message