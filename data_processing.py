import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, Any

class DataProcessor:
    def __init__(self):
        self.month_replacements = {
            'janv': 'jan', 'fév': 'feb', 'mars': 'mar', 'avril': 'apr',
            'mai': 'may', 'juin': 'jun', 'juil': 'jul', 'août': 'aug',
            'sept': 'sep', 'oct': 'oct', 'nov': 'nov', 'déc': 'dec'
        }
    
    def detect_data_orientation(self, df: pd.DataFrame) -> str:
        """Detect if dates are in rows or columns"""
        first_row = df.iloc[0, 1:].astype(str)
        first_col = df.iloc[1:, 0].astype(str)
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(janv|fév|mars|avril|mai|juin|juil|août|sept|oct|nov|déc)\s*\d{4}',
            r'\d{4}[mM]\d{1,2}',
            r'\d{4}[-_]\d{2}',
        ]
        
        dates_in_row = sum(1 for cell in first_row if any(re.search(pattern, str(cell), re.IGNORECASE) for pattern in date_patterns))
        dates_in_col = sum(1 for cell in first_col if any(re.search(pattern, str(cell), re.IGNORECASE) for pattern in date_patterns))
        
        return "dates_in_columns" if dates_in_row > dates_in_col else "dates_in_rows"
    
    def standardize_dataframe(self, df: pd.DataFrame, orientation: str = None) -> pd.DataFrame:
        """Standardize dataframe to common format"""
        if orientation is None:
            orientation = self.detect_data_orientation(df)
        
        if orientation == "dates_in_columns":
            first_col_name = df.columns[0]
            df = df.set_index(first_col_name).T.reset_index()
            df.rename(columns={'index': 'Date'}, inplace=True)
        else:
            date_regex = r'^(date|year|annee|année|period|période)$'
            candidates = [c for c in df.columns if re.search(date_regex, str(c), flags=re.IGNORECASE)]
            if candidates:
                date_col = candidates[0]
                cols = list(df.columns)
                cols.insert(0, cols.pop(cols.index(date_col)))
                df = df[cols]
            if df.columns[0] != 'Date':
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        
        return df
    
    def clean_and_convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert date column"""
        if 'Date' not in df.columns:
            raise ValueError("Column 'Date' not found")
        
        df['Date'] = df['Date'].astype(str).str.strip()
        
        for fr_month, en_month in self.month_replacements.items():
            df['Date'] = df['Date'].str.replace(fr_month, en_month, case=False, regex=False)
        
        date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
            '%b %Y', '%B %Y', '%Y%m', '%YM%m'
        ]
        
        converted_dates = []
        for date_str in df['Date']:
            converted = False
            for fmt in date_formats:
                try:
                    converted_date = pd.to_datetime(date_str, format=fmt)
                    converted_dates.append(converted_date)
                    converted = True
                    break
                except:
                    continue
            if not converted:
                try:
                    converted_date = pd.to_datetime(date_str, errors='coerce')
                    converted_dates.append(converted_date)
                except:
                    converted_dates.append(pd.NaT)
        
        df['Date'] = converted_dates
        return df
    
    def validate_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns to proper format"""
        numeric_columns = df.columns[1:]
        for col in numeric_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to process uploaded dataframe"""
        df_processed = self.standardize_dataframe(df.copy())
        df_processed = self.clean_and_convert_dates(df_processed)
        df_processed = self.validate_numeric_columns(df_processed)
        df_processed = df_processed.dropna(subset=['Date'])
        return df_processed