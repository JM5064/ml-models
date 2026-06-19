import torch
from torch.utils.data.dataloader import DataLoader
from tokenizers import Tokenizer

from ..model.bumblebee.bumblebee import Bumblebee
from .train import validate
from ..model.loss import CrossEntropyLoss
from datasets.Wikitext.wikitext import WikiText
from models.utils import DEVICE
from datasets.Wikitext.special_chars import EOS, UNK


def test_model(model, test_set, batch_size):
    # Create dataloader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    loss_func = CrossEntropyLoss(label_smoothing=0.0)

    print("Testing model...")
    metrics = validate(model, test_loader, loss_func)

    print("Cross-entropy loss:", metrics['average_val_loss'])
    print("Perplexity:", 2.71828183 ** metrics['average_val_loss'])


def greedypredict(model, input_seq, tokenizer_json, max_iters=20, ignore_eos=False, temperature=0):
    # Get vocab
    tokenizer = Tokenizer.from_file(tokenizer_json)

    # Encode input
    tokenized_input = tokenizer.encode(input_seq)
    context = tokenized_input.ids

    # Potentially in-context learning if input length > seq_len ?????

    # Convert to tensor
    context = torch.tensor(context)

    # # Predict for max number of iterations or until EOS
    encoded_output = []
    model.eval()
    with torch.no_grad():
        context = context.to(DEVICE)

        for i in range(max_iters):
            preds = model(context)

            if temperature == 0:
                # greedy
                best = preds[0].argmax(dim=1)
                best = best.squeeze().tolist()
                best_next = best[-1]
            else:
                # Apply softmax
                softmax = torch.softmax(preds / temperature, dim=2)

                # Draw a random token weighted by probability for the next token
                best_next = torch.multinomial(softmax[0][-1], num_samples=1).item()         

            if EOS in tokenizer.decode([best_next]) and not ignore_eos:
                break
            else:
                encoded_output.append(best_next)
                context = context.squeeze().tolist()
                context = context[1:] + [best_next]
                context = torch.tensor(context)
                context = context.to(DEVICE)

    output = ""
    for encoded_token in encoded_output:
        output += tokenizer.decode([encoded_token.item()])
    
    print("Input:", input_seq)
    print()
    print("Output:", output)


def topPpredict(model, input_seq, tokenizer_json, p=0.85, max_iters=20, ignore_eos=False, temperature=0):
    # Get vocab
    tokenizer = Tokenizer.from_file(tokenizer_json)

    # Encode input
    tokenized_input = tokenizer.encode(input_seq)
    context = tokenized_input.ids

    # Convert to tensor
    context = torch.tensor(context)

    # Predict for max number of iterations or until EOS
    encoded_output = []
    model.eval()
    with torch.no_grad():
        context = context.to(DEVICE)

        for i in range(max_iters):
            preds = model(context)

            # Apply softmax
            if temperature == 0:
                softmax = torch.softmax(preds, dim=2)
            else:
                softmax = torch.softmax(preds/temperature, dim=2)

            # Sort and take ones that add up to p
            probs, indices = torch.sort(softmax[0][-1], descending=True)
            total_prob = 0
            for i in range(len(probs)):
                total_prob += probs[i]
                if total_prob > p:
                    break
            probs = probs[:i+1]
            indices = indices[:i+1]

            # Softmax again (???)
            softmax = torch.softmax(probs, dim=-1)

            # Completely random
            best_next = indices[torch.randint(0, len(indices), size=(1,))]

            # Draw a random token weighted by probability for the next token
            # best_next = indices[torch.multinomial(softmax, num_samples=1).item()]

            if EOS in tokenizer.decode([best_next]) and not ignore_eos:
                break
            else:
                encoded_output.append(best_next)
                context = context.squeeze().tolist()
                context = context[1:] + [best_next]
                context = torch.tensor(context)
                context = context.to(DEVICE)

    output = ""
    for encoded_token in encoded_output:
        output += tokenizer.decode([encoded_token.item()])
        
    print("Input:", input_seq)
    print()
    print("Output:", output)


if __name__ == "__main__":
    # Modify model path for desired model
    MODEL_PATH = 'runs/wikitext103/best.pt'

    encoded_text_json = 'data/wikitext103/encoded_text_test.json'
    merge_pairs_json = 'data/wikitext103/merge_pairs.json'
    vocab_json = 'data/wikitext103/vocab.json'

    # Load dataset
    wikitext2_test = WikiText(
        encoded_text_json=encoded_text_json,
        vocab_json=vocab_json
    )

    # Define params
    VOCAB_SIZE = wikitext2_test.get_vocab_size()
    BATCH_SIZE = 32

    # Load model
    d_model = 512
    model = Bumblebee(vocab_size=VOCAB_SIZE, d_model=d_model)
    model = model.to(DEVICE)

    model_state_dict = torch.load(MODEL_PATH, map_location=DEVICE)['state_dict']
    model.load_state_dict(model_state_dict)

    # Uncomment to test model
    # test_model(model, wikitext2_test, BATCH_SIZE)

    text = "The Montreal Canadiens , officially Club de hockey Canadien and colloquially known as the Habs , "
    topPpredict(model, text, "data/wikitext103/hf_data_json.json", max_iters=50, ignore_eos=False, temperature=0.2)
    