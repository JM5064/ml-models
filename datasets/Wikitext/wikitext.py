from datasets.Wikitext.bpe import load_from_file
import torch
from torch.utils.data import Dataset


class WikiText(Dataset):

    def __init__(self, seq_len=128, 
            encoded_text_json='data/wikitext2/encoded_text_train.json',
            vocab_json='data/wikitext2/vocab.json', 
        ):
        """Make dataset from encoded text and vocabulary

        Args:
            seq_len (int, optional): length of "context window" for training. 
                Defaults to 128.
            encoded_text_json (str, optional): file path of encoded text. 
                Defaults to 'data/wikitext2/encoded_text.json'.
            vocab_json (str, optional): file path of vocabulary. 
                Defaults to 'data/wikitext2/vocab.json'.
        """

        super().__init__()

        self.seq_len = seq_len

        # Load data files
        self.encoded_text = load_from_file(encoded_text_json)
        # self.encoded_text = self.encoded_text[:int(len(self.encoded_text) * percent)]

        self.vocab = load_from_file(vocab_json)


    def __getitem__(self, index):
        """Retrives a training + target sample
            
        The training sample is a piece of text start at index `index` of length `seq_len`
        The target is the same piece of text, shifted right by 1

        Args:
            index (int): index of sample to get

        Returns:
            input_tokens (list[int]): tokenized training sample
            target_tokens (list[int]): tokenized target of the training sample
        """
        skip = self.seq_len
        index *= skip

        input_tokens = self.encoded_text[index : index + self.seq_len]
        target_tokens = self.encoded_text[index + 1 : index + self.seq_len + 1]

        return torch.tensor(input_tokens), torch.tensor(target_tokens)
    

    def __len__(self):
        return (len(self.encoded_text) - self.seq_len - 1) // (self.seq_len) + 1
    

    def get_vocab_size(self):
        return len(self.vocab)
    

if __name__ == "__main__":
    train_set = WikiText(encoded_text_json='data/wikitext2/encoded_text_train.json')
    val_set = WikiText(encoded_text_json='data/wikitext2/encoded_text_val.json')
    test_set = WikiText(encoded_text_json='data/wikitext2/encoded_text_test.json')

    print("Train set:", len(train_set), "Val set:", len(val_set))
    
