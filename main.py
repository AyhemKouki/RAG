from rag_pipeline import ask_question
import os

def main():

    print("\nAvailable PDFs:\n")

    pdfs = [f.replace(".pdf", "") for f in os.listdir("data/pdfs") if f.endswith(".pdf")]

    for p in pdfs:
        print("-", p)

    collection = input("\nSelect PDF: ").strip().lower()

    while True:

        q = input("\nQuestion (exit): ")

        if q == "exit":
            break

        answer = ask_question(q, collection)

        print("\n----------------------")
        print(answer)
        print("----------------------\n")


if __name__ == "__main__":
    main()