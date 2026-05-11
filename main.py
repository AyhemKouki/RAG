from rag_pipeline import ask_question

def main():

    print("\n======================================")
    print("      AI Academic Assistant")
    print("======================================\n")

    while True:

        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            print("\nGoodbye!\n")
            break

        if not question.strip():
            print("Please enter a valid question.\n")
            continue
        else:
            answer = ask_question(question)

            print("\n--------------------------------------")
            print(answer)
            print("--------------------------------------\n")

if __name__ == "__main__":
    main()