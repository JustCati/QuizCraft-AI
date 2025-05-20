<div align="center">
  <img src="assets/QuizCraft-AI LOGO.png" alt="QuizCraft-AI Logo" width="200"/>
</div>

# QuizCraft-AI

QuizCraft-AI is a RAG (Retrieval Augmented Generation) system designed to help users understand university-level topics from PDF slides. It can explain concepts and generate questionnaires based on the provided material.

## Environment Setup

**Create a Conda Environment:**
It is recommended to use Python version 3.11.9.
```bash
conda create -n quizcraft python=3.11.9
conda activate quizcraft
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command in your terminal from the project's root directory:

```bash
chainlit run main.py
```

This will start the Chainlit interface, allowing you to interact with the RAG system. You can upload PDF slides, ask questions, and request questionnaires.

## Dataset Generation for Evaluation

Before running the evaluation scripts, you need to generate a dataset of queries and corresponding ideal answers (goldens). This is done using the `generate_dataset.py` script.

1.  **Prepare your data:** Place the PDF slides you want to use for evaluation into a folder named `dataset` in the project's root directory.
2.  **Set up Environment Variables:** Create a `.env` file in the project's root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```
    The script uses Google's Gemini model via this API key to generate queries and golden answers from the chunks extracted from your slides.
3.  **Run the script:**
    ```bash
    python generate_dataset.py --input_path dataset/ --output_path generated_dataset/ --num_chunks <number_of_chunks_to_process> 
    ```
    Replace `<number_of_chunks_to_process>` with appropriate values. For example:
    ```bash
    python generate_dataset.py --input_path dataset/ --output_path generated_dataset/ --num_chunks 100
    ```
    This script will process the documents, split them into chunks, and generate corresponding questions and ideal answers, saving them for the evaluation phase.

## Evaluation

The project includes notebooks for evaluating different components of the system:

### Vector Store Evaluation

The `evaluation_vector_store.ipynb` notebook focuses on evaluating the performance of the vector store. This includes assessing the relevance and accuracy of retrieved document chunks based on test queries.

### End-to-End System Evaluation

The `evaluation.ipynb` notebook provides a comprehensive evaluation of the entire RAG system. It utilizes standard RAG evaluation metrics such as:
*   **RAG Triad:** Faithfulness, Answer Relevance, and Context Relevance.
*   **Text Similarity Metrics:** BERTScore, BLEU, and ROUGE scores are used to compare the generated answers against the golden answers.

These evaluations help in understanding the system's strengths and weaknesses and guide further improvements.

## License

This project is licensed under the terms of the GNU General Public License v3.0. You can find the full license text in the [LICENSE](LICENSE) file.
