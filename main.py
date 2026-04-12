
from RAG.rag_pipeline import run_pipeline

def main():
    print(" Starting AI Meeting Assistant...\n")

    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n Program stopped by user.")
    except Exception as e:
        print(f"\np An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()