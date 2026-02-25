import pandas as pd
import os

def preprocess_books_data(input_path: str, output_path: str):
    """
    Loads book data, cleans it, combines relevant text fields,
    and saves the processed data to a new CSV file.

    Args:
        input_path (str): The path to the input CSV file (e.g., 'datasets/books.csv').
        output_path (str): The path to save the cleaned CSV file (e.g., 'datasets/books_cleaned.csv').
    """
    print(f"Loading data from {input_path}...")
    # Add error_bad_lines=False and warn_bad_lines=True to handle potential parsing errors in the large CSV
    try:
        df = pd.read_csv(input_path, on_bad_lines='warn')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # Try with a different parser if default fails
        df = pd.read_csv(input_path, engine='python', on_bad_lines='warn')


    # 1. Drop unnecessary columns
    columns_to_drop = [
        'isbn13', 'isbn10', 'subtitle', 'thumbnail', 'published_year',
        'average_rating', 'num_pages', 'ratings_count'
    ]
    # some columns might not exist, so ignore errors
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # 2. Handle missing values in key text fields
    text_cols = ['title', 'authors', 'categories', 'description']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # 3. Create a combined text column for embeddings
    def create_embedding_text(row):
        title = row.get('title', '')
        authors = row.get('authors', '')
        categories = row.get('categories', '')
        description = row.get('description', '')
        return f"Title: {title}\nAuthors: {authors}\nCategories: {categories}\nDescription: {description}"

    df['text_for_embedding'] = df.apply(create_embedding_text, axis=1)

    # 4. Remove rows where the title is missing as they are not useful
    df.dropna(subset=['title'], inplace=True)
    df = df[df['title'] != '']


    # 5. Remove duplicates based on title and authors
    df.drop_duplicates(subset=['title', 'authors'], inplace=True)

    # 6. Save the cleaned dataframe
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Use a raw string for the path to avoid unicode errors
    full_input_path = os.path.abspath(input_path)
    
    print("Preprocessing complete.")
    try:
        print(f"Original number of rows: {len(pd.read_csv(full_input_path, on_bad_lines='skip'))}")
    except Exception as e:
        print(f"Could not count original rows due to parsing issues: {e}")

    print(f"Cleaned number of rows: {len(df)}")
    print("\nCleaned data sample:")
    print(df.head())

if __name__ == '__main__':
    # This allows the script to be run directly
    INPUT_FILE_PATH = r'datasets/books.csv'
    OUTPUT_FILE_PATH = r'datasets/books_cleaned.csv'
    preprocess_books_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
