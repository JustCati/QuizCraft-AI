import os 
from dotenv import load_dotenv

from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.base_model import DeepEvalBaseLLM

from src.model.model import MultiModalEmbeddingModel
from deepeval.synthesizer.config import StylingConfig
from deepeval.synthesizer.config import ContextConstructionConfig

from deepeval.synthesizer import Synthesizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough






class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        self.model = MultiModalEmbeddingModel("nomic-ai/nomic-embed-text-v1.5", "nomic-ai/nomic-embed-vision-v1.5")

    def load_model(self):
        return self.model

    def embed_text(self, text):
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_texts(self, texts):
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)

    async def a_embed_text(self, text):
        return await self.embed_text(text)

    async def a_embed_texts(self, texts):
        return await self.embed_texts(texts)

    def get_model_name(self):
        "Custom Azure Embedding Model"



class SynthetizerModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model_ref = model
        self.model = (
            RunnablePassthrough()
            | self.model_ref
            | StrOutputParser()
        )

    def load_model(self):
        return self.model

    def get_model_name(self):
        return "gemini-2.0-flash"

    def generate(self, prompt):
        model = self.load_model()
        return model.invoke(prompt)

    async def a_generate(self, prompt):
        model = self.load_model()
        return await model.invoke(prompt)



def main():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.25,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        rate_limiter=rate_limiter,
    )
    model = SynthetizerModel(llm)


    styling_config = StylingConfig(
            input_format="University slides written in English or Italian",
            task="Generate a question that a student might ask about the text, in English. Do not cite the text verbatim, paraphrase when possible. Generate the question in English.",
            scenario="A student is preparing for an exam and wants to test their understanding of the material or want an explanation of the content.",
        )
    synthesizer = Synthesizer(model=model, 
                            async_mode=False,
                            max_concurrent=1,
                            styling_config=styling_config,)


    documents_path = []
    root_path = os.path.join(os.getcwd(), "dataset", "slides")
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith(".pdf"):
                    documents_path.append(file_path)


    context_construction_config = ContextConstructionConfig(
        embedder=CustomEmbeddingModel(),
        max_contexts_per_document=4,
        min_contexts_per_document=1,
        min_context_length=0,
    )

    res = synthesizer.generate_goldens_from_docs(
        documents_path,
        include_expected_output=False,
        max_goldens_per_context=1,
        context_construction_config=context_construction_config,
    )

    queries = [str(elem.input) for elem in res]
    with open("queries.txt", "w") as f:
        for query in queries:
            f.write(f"{query}\n")




if __name__ == "__main__":
    main()
