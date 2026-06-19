import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFD, StripAccents, Strip, Sequence
from tokenizers.pre_tokenizers import Whitespace

import os
from collections import Counter
from datasets.Wikitext.special_chars import UNK, EOS
import datasets.Wikitext.bpe as bpe


class DatasetCreator:

    def __init__(self, split='train', parquet_file='data/original/train-wikitext2.parquet',
        encoded_text_json='data/wikitext2/encoded_text.json',
        vocab_json='data/wikitext2/vocab.json', 
        merge_pairs_json='data/wikitext2/merge_pairs.json',
        hf_data_json='data/wikitext2/hf_data_json.json'
    ):
        super().__init__()

        self.split = split

        self.vocab_json = vocab_json
        self.encoded_text_json = encoded_text_json
        self.merge_pairs_json = merge_pairs_json
        self.hf_data_json = hf_data_json

        # Load parquet file
        self.df = pd.read_parquet(parquet_file)

        # Create tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=UNK))
        self.trainer = BpeTrainer(special_tokens=[UNK, EOS], vocab_size=30000, min_frequency=10)

        # Add normalizer and pre-tokenizer
        self.tokenizer.normalizer = Sequence([Strip(), NFD(), StripAccents()])
        self.tokenizer.pre_tokenizer = Whitespace()

        self.train_tokenizer(min_length=50)


    def train_tokenizer(self, min_length):
        """Runs BPE to create vocab, merge pairs, and encoded text
        Args:
            min_length (int) : threshold for removing empty/short paragraphs

        Returns:
            None
        """

        # Stop if encoded text file already exists
        if self.file_exists(self.encoded_text_json):
            return 

        # Convert to array of paragraphs
        text = self.df.loc[:,"text"].to_list()

        # Remove empty/very short paragraphs, strip text, and add EOS
        self.array_text = [t.strip() + EOS for t in text if len(t) > min_length]

        # Run BPE if tokenizer doesn't exist
        if not self.file_exists(self.hf_data_json):
            print("Running BPE...")
            
            # Run BPE
            self.tokenizer.train_from_iterator(self.array_text, trainer=self.trainer)

            # Save BPE info
            self.tokenizer.save(self.hf_data_json)

        else:
            self.tokenizer = Tokenizer.from_file(self.hf_data_json)

        self.create_files()

    
    def create_files(self):
        """Creates encoded_text_json, and vocab_json and merge_pairs_json if necessary"""

        # Encode dataset
        print("Encoding dataset...")
        encoded_text = []
        vocab_counts = Counter()

        for text in self.array_text:
            encoded_text_sample = self.tokenizer.encode(text)

            encoded_text.extend(encoded_text_sample.ids)
            vocab_counts.update(encoded_text_sample.tokens)
        
        bpe.save_to_file(encoded_text, self.encoded_text_json)

        # Don't create vocab and merge pairs files for val and test
        if self.split != 'train':
            return
        
        # Save vocab
        print("Saving vocab...")
        vocab_dict = self.tokenizer.get_vocab()

        # Sort tokens by id
        ordered_tokens = [token for token, idx in sorted(vocab_dict.items(), key=lambda x: x[1])]

        # Order token counts
        ordered_vocab_counts = {}
        for token in ordered_tokens:
            ordered_vocab_counts[token] = vocab_counts[token]

        bpe.save_to_file(ordered_vocab_counts, self.vocab_json, indent=2)

        # Load hugging face generated file
        print("Saving merge pairs...")
        hf_data = bpe.load_from_file(self.hf_data_json)

        # Save merge pairs
        merge_pairs = hf_data['model']['merges']
        bpe.save_to_file(merge_pairs, self.merge_pairs_json)


    def file_exists(self, file_path):
        """Evaluates whether the given file exists
        
        Args:
            file_path (str): path to check
        
        Returns:
            bool
        """

        return os.path.exists(file_path) and os.path.getsize(file_path) > 0


if __name__ == "__main__":
    train_set = DatasetCreator(
        split='train', 
        parquet_file='data/original/train-wikitext103-ALL.parquet',
        encoded_text_json='data/wikitext103/encoded_text_train.json',
        vocab_json='data/wikitext103/vocab.json', 
        merge_pairs_json='data/wikitext103/merge_pairs.json',
        hf_data_json='data/wikitext103/hf_data_json.json'
    )

    val_set = DatasetCreator(
        split='val', 
        parquet_file='data/original/validation.parquet',
        encoded_text_json='data/wikitext103/encoded_text_val.json',
        hf_data_json='data/wikitext103/hf_data_json.json'
    )

    test_set = DatasetCreator(
        split='test', 
        parquet_file='data/original/test.parquet',
        encoded_text_json='data/wikitext103/encoded_text_test.json',
        hf_data_json='data/wikitext103/hf_data_json.json'
    )
