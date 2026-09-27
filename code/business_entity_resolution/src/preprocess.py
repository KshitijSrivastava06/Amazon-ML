import pandas as pd
import re
from anyascii import anyascii
import numpy as np
import os
import gc
try:
    from .config import BUSINESS_ABBREVIATION_MAP, ADDRESS_ABBREVIATION_MAP
except ImportError:
    from config import BUSINESS_ABBREVIATION_MAP, ADDRESS_ABBREVIATION_MAP

# Precompile regexes once for massive speedup
CLEAN_PUNCT_REGEX = re.compile(r'[^a-z0-9\s]')
WHITESPACE_REGEX = re.compile(r'\s+')
COMPILED_BUSINESS_MAP = [(re.compile(abbr, flags=re.IGNORECASE), expansion) for abbr, expansion in BUSINESS_ABBREVIATION_MAP.items()]
COMPILED_ADDRESS_MAP = [(re.compile(abbr, flags=re.IGNORECASE), expansion) for abbr, expansion in ADDRESS_ABBREVIATION_MAP.items()]

def normalize_text(text, type='name'):
    """
    Normalizes a text string (business name or address).
    Applies lowercasing, transliteration (anyascii), punctuation removal, and abbreviation expansion.
    """
    if pd.isna(text) or text is None:
        return ""
    
    # 1. Lowercase
    text = str(text).lower()
    
    # 2. Transliterate to ASCII (handles Devanagari, French accents, etc.)
    text = anyascii(text)
    
    # 3. Replace common separators and punctuation with space (keep alphanumeric)
    text = CLEAN_PUNCT_REGEX.sub(' ', text)
    
    # 4. Expand abbreviations using precompiled regexes
    if type == 'name':
        for pattern, expansion in COMPILED_BUSINESS_MAP:
            text = pattern.sub(expansion, text)
    elif type == 'address':
        for pattern, expansion in COMPILED_ADDRESS_MAP:
            text = pattern.sub(expansion, text)
            
    # 5. Remove extra whitespace
    return WHITESPACE_REGEX.sub(' ', text).strip()

def preprocess_dataframe(df):
    """
    Applies memory-efficient normalization to the business_name and business_address columns.
    Creates 'name_clean' and 'addr_clean' columns and drops raw text to conserve RAM.
    """
    needed_cols = [c for c in ['entity_id', 'country', 'business_name', 'business_address'] if c in df.columns]
    df_clean = df[needed_cols].copy()
    
    # Process with list comprehension (much faster and lower memory overhead than .apply(lambda))
    names = df_clean['business_name'].tolist() if 'business_name' in df_clean.columns else []
    df_clean['name_clean'] = [normalize_text(n, 'name') for n in names]
    if 'business_name' in df_clean.columns:
        df_clean.drop(columns=['business_name'], inplace=True)
        
    addrs = df_clean['business_address'].tolist() if 'business_address' in df_clean.columns else []
    df_clean['addr_clean'] = [normalize_text(a, 'address') for a in addrs]
    if 'business_address' in df_clean.columns:
        df_clean.drop(columns=['business_address'], inplace=True)
        
    if 'country' in df_clean.columns:
        df_clean['country'] = df_clean['country'].astype('category')
        
    gc.collect()
    return df_clean

def load_and_preprocess(filepath, is_ground_truth=False):
    """
    Loads a TSV file and applies preprocessing with progress prints.
    """
    fname = os.path.basename(filepath)
    print(f"  Reading {fname}...")
    df = pd.read_csv(filepath, sep='\t', dtype=str)
    
    if is_ground_truth:
        return df
        
    required = {'entity_id', 'country'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filepath} is missing required columns: {missing}")
        
    print(f"  Preprocessing {len(df)} records in {fname}...")
    cleaned_df = preprocess_dataframe(df)
    del df
    gc.collect()
    print(f"  Finished {fname}.")
    return cleaned_df

if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # Simple test
    test_cases = [
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "name"), # Hindi: Ram Marketing Private Limited
        ("SCI Ptit Àmicale", "name"),              # French: SCI -> societe civile immobiliere, Àmicale -> amicale
        ("175 Blvd du Président", "address"),      # French Address: Blvd -> boulevard, Président -> President
        ("ABC Corp., Ltd.", "name"),               # English: Corp -> corporation, Ltd -> limited
        ("<< Team Ecole", "name")                  # Noise characters
    ]
    
    print("Testing Preprocessing:")
    for text, type_ in test_cases:
        res = normalize_text(text, type_)
        print(f"Original: {text}\nCleaned:  {res}\n")
