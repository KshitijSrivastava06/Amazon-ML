import pandas as pd
import re
from anyascii import anyascii
import numpy as np
try:
    from .config import BUSINESS_ABBREVIATION_MAP, ADDRESS_ABBREVIATION_MAP
except ImportError:
    from config import BUSINESS_ABBREVIATION_MAP, ADDRESS_ABBREVIATION_MAP

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
    # This also helps split words correctly.
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 4. Expand abbreviations based on type
    if type == 'name':
        # Replace abbreviations as whole words
        for abbr, expansion in BUSINESS_ABBREVIATION_MAP.items():
            text = re.sub(abbr, expansion, text)
    elif type == 'address':
        for abbr, expansion in ADDRESS_ABBREVIATION_MAP.items():
            text = re.sub(abbr, expansion, text)
            
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df):
    """
    Applies normalization to the business_name and business_address columns.
    Creates 'name_clean', 'addr_clean', and 'name_addr_clean' columns.
    """
    # Create copies to avoid SettingWithCopyWarning if working on slices
    df_clean = df.copy()
    
    # Apply normalization
    df_clean['name_clean'] = df_clean['business_name'].apply(lambda x: normalize_text(x, 'name'))
    df_clean['addr_clean'] = df_clean['business_address'].apply(lambda x: normalize_text(x, 'address'))
    
    # Handle missing addresses for concatenation (though normalize_text already returns "")
    df_clean['name_addr_clean'] = df_clean['name_clean'] + " " + df_clean['addr_clean']
    df_clean['name_addr_clean'] = df_clean['name_addr_clean'].str.strip()
    
    return df_clean

def load_and_preprocess(filepath, is_ground_truth=False):
    """
    Loads a TSV file and applies preprocessing.
    """
    df = pd.read_csv(filepath, sep='\t')
    
    if is_ground_truth:
        # For ground truth, we don't need text normalization
        return df
        
    return preprocess_dataframe(df)

if __name__ == '__main__':
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
