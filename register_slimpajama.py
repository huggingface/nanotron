from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.index.parser import MetadataParser
from typing import Any, Optional

class SlimPajamaParser(MetadataParser):
    def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
        metadata = payload["meta"]
        self._index.append_entry("redpajama_set_name", metadata["redpajama_set_name"], self.dataset_id, self.file_id, line_number)

def parsing_func(sample):
    import json
    return json.loads(sample)["text"]

if __name__ == "__main__":
    client = MixteraClient.from_remote("localhost", 8888)

    client.register_metadata_parser("SLIMPAJAMA_PARSER", SlimPajamaParser)
    
    # Registering the dataset with the client.
    client.register_dataset(
        "slimpajama_benchmark", "/scratch/maximilian.boether/mixtera/slimpajama", JSONLDataset, parsing_func, "SLIMPAJAMA_PARSER"
    )