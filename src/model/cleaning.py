import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser




def clean_text(llm, texts):
    with open(os.path.join("src", "model", "prompts", "formatting.txt"), "r") as f:
        formatting_prompt = f.read()

    if isinstance(texts, str):
        texts = [texts]

    prompt = PromptTemplate(input_variables=["text"], template=formatting_prompt)

    llm_chain = (prompt | llm | StrOutputParser())


    formatted_texts = []
    for text in texts:
        blocks = [text[i:i + 5000] for i in range(0, len(text), 5000)]
        print(f"Processing {len(blocks)} blocks of text.")
        complete_text = ""
        for i, block in enumerate(blocks):
            result = llm_chain.invoke(block)
            complete_text += result
            print(f"Processed block {i} of size {len(block)} with result length {len(result)}")
        formatted_texts.append(complete_text)

    print(type(formatted_texts))
    print(len(formatted_texts))
    return "\n\n".join(formatted_texts)
