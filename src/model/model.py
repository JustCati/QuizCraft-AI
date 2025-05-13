import torch
import subprocess
import numpy as np
from PIL import Image
import torch.nn.functional as F
from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoImageProcessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


class GenericOllamaModel(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.available_models = self.__retrieve_available_models("available")

    def __retrieve_info(self, operation):
        process = subprocess.run(["ollama", str(operation)], stdout=subprocess.PIPE)
        outFolder = process.stdout.decode("utf-8").split("\n")[1:-1]
        data = [x.split(" ")[0] for x in outFolder]
        data = [x.replace(":latest", "") for x in data]
        return data

    def __retrieve_available_models(self, scope):
        if scope not in ["available", "running"]:
            raise ValueError(f"Invalid scope: {scope}")
        if scope == "available":
            return self.__retrieve_info("list")
        if scope == "running":
            return self.__retrieve_info("ps")

    def check_model(self):
        running_models = self.__retrieve_available_models("running")
        if len(running_models) > 0:
            for model in running_models:
                if model == self.model_name:
                    self.__manage_Model("stop")
        if self.model_name not in self.available_models:
            print(f"Model {self.model_name} not available.")
            print("Retrieving model...")
            process = subprocess.Popen(["ollama", "pull", self.model_name], stdout=subprocess.PIPE, bufsize=0)
            process.wait()

    def __manage_Model(self, action = "run"):
        if action not in ["run", "stop"]:
            raise ValueError(f"Invalid action: {action}")
        subprocess.Popen(['ollama', action, self.model_name], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def get(self):
        self.__manage_Model("run")
        return self

    def stop(self):
        self.__manage_Model("stop")

    def __enter__(self) -> ChatOllama:
        self.get()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()



class OllamaLanguageModel(GenericOllamaModel):
    def __init__(self, model_name, temperature = 0.0, num_predict = -1) -> None:
        super().__init__(model_name)
        super().check_model()
        self.model = ChatOllama(model=model_name, 
                        temperature=temperature,
                        num_predict=num_predict)


# nomic-ai/nomic-embed-text-v1.5
# nomic-ai/nomic-embed-vision-v1.5
class MultiModalEmbeddingModel(Embeddings):
    def __init__(self, text_model_name, visual_model_name) -> None:
        self.model = None
        self.processor = None
        self.query_prefix = "search_query: "
        self.indexing_prefix = "search_document: "

        self.text_model = SentenceTransformer(text_model_name, 
                                              device="cuda",
                                              trust_remote_code=True,)
        self.text_model.eval()

        self.vis_processor = AutoImageProcessor.from_pretrained(visual_model_name)
        self.vis_model = AutoModel.from_pretrained(visual_model_name,
                                                   device_map="cuda",
                                                   trust_remote_code=True)
        self.vis_model.eval()


    def embed_documents(self, texts):
        texts = [texts] if isinstance(texts, str) else texts
        texts = [self.indexing_prefix + text for text in texts]
        embeddings = self.text_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings


    def embed_query(self, text):
        text = self.query_prefix + text
        embedding = self.text_model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return embedding


    def embed_image(self, uris):
        images = uris
        images = [images] if isinstance(images, str) else images

        embeddings = []
        for image in images:
            image = Image.open(image)

            inputs = self.vis_processor(image, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                img_emb = self.vis_model(**inputs).last_hidden_state
            img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
            img_embeddings = img_embeddings.cpu().numpy()
            img_embeddings = img_embeddings.squeeze(0)
            embeddings.append(img_embeddings) 
            
        return embeddings



class LanguageClassifier():
    def __init__(self, model_name="qanastek/51-languages-classifier", device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map=device)
        self.classifier = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer)
        self.model.eval()
        
        self.vocabulary = {
            "it-IT": "it",
            "en-US": "en",
        }


    def classify(self, text):
        if len(text) > 512:
            text = text[:512]

        res = self.classifier(text)
        lang_key = res[0]['label']
        lang = self.vocabulary.get(lang_key, "Unknown")
        return lang.lower()
