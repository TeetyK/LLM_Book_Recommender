from dotenv import load_dotenv
from src.preprocessing import preprocessing
load_dotenv()

def main():
    print("Hello from llm-book-recommander!")


if __name__ == "__main__":
    main()
    preprocessing(".\\datasets\\","books_cleaned.csv")
