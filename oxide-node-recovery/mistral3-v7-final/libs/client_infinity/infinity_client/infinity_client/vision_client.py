from io import BytesIO
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np  # Import numpy
from typing import Union, Literal

HAS_IMPORTS = True
try:
    from PIL import Image
    import numpy as np

except ImportError:
    HAS_IMPORTS = False

try:
    import requests
    from requests.adapters import HTTPAdapter, Retry
except ImportError:
    HAS_IMPORTS = False


class InfinityVisionAPI:
    def __init__(
        self,
        url: str = "https://infinity-multimodal.modal.michaelfeil.eu",
        format: Literal["base64", "float"] = "base64",
        model: str = "michaelfeil/colqwen2-v0.1",
    ) -> None:
        """client usage for infinity multimodal api

        Args:
            url (str, optional): url of the deployment. Defaults to "https://infinity-multimodal.modal.michaelfeil.eu".
            format (str, optional): base. Defaults to "base64".
            model (str, optional): served_model_name in the deployment. Defaults to "michaelfeil/colqwen2-v0.1".
        """
        req = requests.post(
            url + "/embeddings",
            json={  # get shape of output by sending a float request
                "model": model,
                "input": ["test"],
                "encoding_format": "float",
                "modality": "text",
            },
        )
        req.raise_for_status()
        self.url = url
        self.hidden_dim = np.array(req.json()["data"][0]["embedding"]).shape[-1]
        self.format = format
        self.tp = ThreadPoolExecutor()
        self.tp.__enter__()

        self.sem = threading.Semaphore(64)
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(total=10, backoff_factor=0.5))
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _image_payload(self, images: list["Image.Image"]) -> list[str]:
        if not HAS_IMPORTS:
            raise ImportError("PIL is required to use this class")
        b64_strs = []
        for image in images:
            buffered = BytesIO()
            if not hasattr(image, "save"):
                raise ValueError("Image must be a PIL Image")
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            b64_strs.append(f"data:image/jpeg;base64,{img_str}")
        return b64_strs

    def _text_payload(self, texts: list[str]) -> list[str]:
        return texts

    def health(self) -> bool:
        req = self.session.get(f"{self.url}/health")
        req.raise_for_status()
        return req.status_code == 200

    def _request(self, model: str, images_or_text: list[Union["Image.Image", str]]) -> dict:
        if all(hasattr(item, "save") for item in images_or_text):
            payload = self._image_payload(images_or_text)
            modality = "image"
        elif all(isinstance(item, str) for item in images_or_text):
            payload = self._text_payload(images_or_text)
            modality = "text"
        else:
            raise ValueError("Images and text cannot be mixed in a single request")

        embeddings_req = self.session.post(
            f"{self.url}/embeddings",
            json={"model": model, "input": payload, "encoding_format": self.format, "modality": modality},
        )
        embeddings_req.raise_for_status()
        embeddings = embeddings_req.json()

        if self.format == "base64":
            embeddings_decoded = [
                np.frombuffer(base64.b64decode(e["embedding"]), dtype=np.float32).reshape(-1, self.hidden_dim)
                for e in embeddings["data"]
            ]
        else:
            embeddings_decoded = [np.array(e["embedding"]) for e in embeddings["data"]]
        return embeddings_decoded, embeddings["usage"]["total_tokens"]

    def embed(self, model: str, sentences: list[str]) -> Future[list]:
        self.health()
        with self.sem:
            return self.tp.submit(self._request, model=model, images_or_text=sentences)

    def image_embed(self, model: str, images: list["Image.Image"]) -> Future[list]:
        self.health()  # Call once instead of per image
        with self.sem:
            return self.tp.submit(self._request, model=model, images_or_text=images)


def test_colpali():
    colpali = InfinityVisionAPI()
    future = colpali.embed("michaelfeil/colqwen2-v0.1", ["test"])
    embeddings, total_tokens = future.result()
    print(embeddings, total_tokens)


if __name__ == "__main__":
    test_colpali()
