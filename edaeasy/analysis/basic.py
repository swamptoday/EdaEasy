from IPython.display import display, HTML
import pandas as pd


from edaeasy.analysis.analysis import Analysis


class BasicAnalysis(Analysis):
    """Performs basic analysis on a DataFrame.

    This class provides methods to generate a basic analysis report for a given DataFrame.
    The report includes information about the DataFrame, its shape, types of columns,
    presence of duplicate rows, and counts of unique values in each column.

    Attributes:
        df (DataFrame): The DataFrame to be analyzed.
    """

    def __init__(self, df: pd.DataFrame):
        """Initializes BasicAnalysis with a DataFrame.

        Args:
            df (DataFrame): The DataFrame to be analyzed.
        """
        super().__init__(df=df)


    def generate_report(self):
        """Generates a report with sections for information, head, shape, types, duplicates, and unique."""
        display(HTML(f'<h1>Basic information about DataFrame</h1>'))

        sections = [
            ('Information', self.info),
            ('Head', self.head),
            ('Shape', self.shape),
            ('Types', self.types),
            ('Duplicates', self.duplicates),
            ('Unique', self.unique)
        ]

        self._display_sections(sections)

    def info(self):
        """Displays DataFrame information."""
        self.df.info()

    def head(self):
        """Displays the first few rows of the DataFrame."""
        display(self.df.head())

    def shape(self):
        """Displays the shape of the DataFrame."""
        display(self.df.shape)

    def types(self):
        """Displays the data types of columns in the DataFrame."""
        type_counts = self.df.dtypes.value_counts()
        type_info_df = pd.DataFrame({'Type': type_counts.index, 'Amount': type_counts.values})
        type_info_df['Columns'] = type_info_df['Type'].apply(lambda dtype: ', '.join(self.df.select_dtypes(include=[dtype]).columns.tolist()))
        type_info_df = type_info_df.style.set_properties(**{'text-align': 'right'})
        display(type_info_df)

    def duplicates(self):
        """Displays duplicate rows in the DataFrame, if any."""
        if self.df.duplicated().sum() > 0:
            display(HTML(f'<p>Amount: {self.df.duplicated().sum()}</p>'))
            # display(HTML(self.df[self.df.duplicated()].to_html()))
        else:
            display(HTML('<p>No duplicates!</p>'))

    def unique(self):
        """Displays the count of unique values in each column of the DataFrame."""
        unique_counts = self.df.nunique()
        unique_counts_df = pd.DataFrame(unique_counts, columns=['Count'])
        display(unique_counts_df)