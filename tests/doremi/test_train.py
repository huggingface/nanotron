from nanotron.doremi.train import train
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_toy_dataset


def test_train_doremi():
    NUM_DOMAINS = 10

    model = AutoModelForCausalLM.from_pretrained("stas/tiny-random-llama-2").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("stas/tiny-random-llama-2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = generate_toy_dataset(tokenizer, NUM_DOMAINS)

    domain_weights = train(model, dataset)
    assert domain_weights.shape == (NUM_DOMAINS,)
