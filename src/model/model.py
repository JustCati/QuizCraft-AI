import subprocess
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel


# llava:34b-v1.6-q3_K_M
# qwen2.5:32b-instruct-q3_K_M

# mxbai-embed-large



class GenericOllamaModel:
    def __init__(self, model_name, temperature = 0.0, num_predict = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.num_predict = num_predict

    def __manage_Model(self, action = "run"):
        if action not in ["run", "stop"]:
            raise ValueError(f"Invalid action: {action}")
        subprocess.Popen(['ollama', action, self.model_name], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


    def __enter__(self):
        self.__manage_Model("run")
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__manage_Model("stop")



class LanguageModel(GenericOllamaModel):
    def __init__(self, model_name, temperature = 0.0, num_predict = 512):
        self.model = ChatOllama(model=model_name, 
                          temperature=temperature,
                          num_predict=num_predict)
        super().__init__(model_name, temperature, num_predict)


class EmbeddingModel(GenericOllamaModel):
    def __init__(self, model_name):
        self.model = OllamaEmbeddings(model=model_name)
        super().__init__(model_name)
