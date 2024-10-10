import subprocess
from langchain_ollama import ChatOllama, OllamaEmbeddings




class GenericOllamaModel(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.available_models = self.__retrieve_available_models()

    def __retrieve_available_models(self):
        process = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE)
        outFolder = process.stdout.decode("utf-8").split("\n")[1:-1]
        models = [x.split(" ")[0] for x in outFolder]
        models = [x.replace(":latest", "") for x in models]
        return models

    def check_model(self):
        if self.model_name not in self.available_models:
            print(f"Model {self.model_name} not available.")
            print("Retrieving model...")
            process = subprocess.Popen(["ollama", "pull", self.model_name], stdout=subprocess.PIPE, bufsize=0)
            process.wait()

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
        super().__init__(model_name)
        super().check_model()
        self.model = ChatOllama(model=model_name, 
                        temperature=temperature,
                        num_predict=num_predict)



class EmbeddingModel(GenericOllamaModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        super().check_model()
        self.model = OllamaEmbeddings(model=model_name)
