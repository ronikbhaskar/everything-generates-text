#comment
#clean up the text, decide about punctuation and such, delete line breaks and replace w spaces

import re
import argparse

def preprocess_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Replace line breaks with spaces
    text = text.replace('\n', ' ')
    
    # Replace non-standard characters with standard characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Optional: Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text for machine learning.")
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument("output_file", help="Path to the output text file.")
    
    args = parser.parse_args()
    
    preprocess_text(args.input_file, args.output_file)