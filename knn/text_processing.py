import re
import argparse

# Mapping of non-standard characters to their standard equivalents
char_map = {
    '“': '"', '”': '"', '‘': "'", '’': "'", '–': '-', '—': '-', '…': '...',
    'à': 'a', 'á': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a',
    'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
    'ò': 'o', 'ó': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
    'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
    'ç': 'c', 'ñ': 'n'
}

def replace_non_standard_chars(text):
    for non_standard, standard in char_map.items():
        text = text.replace(non_standard, standard)
    return text

def preprocess_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
        
    # Replace line breaks with spaces
    text = text.replace('\n', ' ')
    
    # Replace non-standard characters using the mapping
    text = replace_non_standard_chars(text)
    
    # Remove remaining non-standard characters
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:()\'"-]', '', text)
        
    # Optional: Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)

#I think this is here to solve a problem I avoided by putting the text into the same folder as the script?
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text for machine learning.")
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument("output_file", help="Path to the output text file.")
    
    args = parser.parse_args()
    
    preprocess_text(args.input_file, args.output_file)
