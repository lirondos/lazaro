import json
import pickle
from pathlib import Path
from typing import Dict, List, Union

from nerpy.document import Document


def load_pickled_documents(path: Union[str, Path]) -> List[Document]:
    with open(path, "rb") as file:
        return pickle.load(file)


def pickle_documents(docs: List[Document], path: Union[str, Path]) -> None:
    with open(path, "wb") as file:
        return pickle.dump(docs, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(path: Union[str, Path]) -> Dict:
    with open(path, encoding="utf8") as file:
        return json.load(file)
