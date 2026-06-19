import os
import pandas as pd

from datasets.Wikitext.special_chars import UNK
import datasets.Wikitext.bpe as bpe


class DatasetCreator:

    def __init__(self, split='train', parquet_file='data/original/train-wikitext2.parquet',
        encoded_text_json='data/wikitext2/encoded_text.json',
        vocab_json='data/wikitext2/vocab.json', 
        merge_pairs_json='data/wikitext2/merge_pairs.json'
    ):
        super().__init__()

        self.split = split

        self.vocab_json = vocab_json
        self.encoded_text_json = encoded_text_json
        self.merge_pairs_json = merge_pairs_json

        # Load parquet file
        self.df = pd.read_parquet(parquet_file)

        self.create_data(min_length=50, num_merges=10000)


    def create_data(self, min_length, num_merges):
        """Runs BPE to create vocab, merge pairs, and encoded text
        Args:
            min_length (int) : threshold for removing empty/short paragraphs
            num_merges (int) : number of merges to perform

        Returns:
            None
        """

        # Stop if encoded text file already exists
        if self.file_exists(self.encoded_text_json):
            return 

        # Convert to array of paragraphs
        text = self.df.loc[:,"text"].to_list()
        
        # Remove empty/very short paragraphs and strip text
        self.array_text = [t.strip("\n") for t in text if len(t) > min_length and t != ""]

        if self.split == 'train':
            # Run BPE
            tokenized_text, self.vocab, merge_pairs = bpe.bpe(self.array_text, num_merges)

            # Create encoding and decoding maps
            self.encoding, self.decoding = bpe.make_mapping(self.vocab)

            # Encode text
            self.encoded_text = self.encode_text(tokenized_text)
            
            # Save vocab, encoded text, and merge pairs
            bpe.save_to_file(self.vocab, self.vocab_json, indent=2)
            bpe.save_to_file(self.encoded_text, self.encoded_text_json)
            bpe.save_to_file(merge_pairs, self.merge_pairs_json)

        else:
            # Load merge pair file and vocabulary
            # NOTE: these files need to be created by running the split as 'train' first
            merge_pairs = bpe.load_from_file(self.merge_pairs_json)
            self.vocab = bpe.load_from_file(self.vocab_json)

            # Run merges
            tokenized_text = bpe.apply_merge_pairs(self.array_text, merge_pairs)

            # Create encoding and decoding maps
            self.encoding, self.decoding = bpe.make_mapping(self.vocab)

            # Encode text
            self.encoded_text = self.encode_text(tokenized_text)
            bpe.save_to_file(self.encoded_text, self.encoded_text_json)


    def encode_text(self, tokenized_text):
        """Encodes tokenized text
        Args:
            tokenized_text (list[str]) : the dataset text as a list of string tokens
                eg. ['the', 'qu', 'ick', ...]
        
        Returns:
            encoded_text (list[int]) : the dataset text as a list of integer tokens
                eg. [0, 4, 5, ...]
        """

        encoded_text = []
        vocab = bpe.load_from_file(self.vocab_json)
        for token in tokenized_text:
            if token in self.encoding:
                # Encode token
                encoded_text.append(self.encoding[token])
            else:
                # Append unknown character if token not in vocabulary
                encoded_text.append(self.encoding[UNK])

        return encoded_text


    def file_exists(self, file_path):
        """Evaluates whether the given file exists
        
        Args:
            file_path (str): path to check
        
        Returns:
            bool
        """

        return os.path.exists(file_path) and os.path.getsize(file_path) > 0
    

if __name__ == "__main__":
    # train_set = DatasetCreator(
    #     split='train', 
    #     parquet_file='data/origina/train-wikitext2.parquet',
    #     encoded_text_json='data/wikitext2/encoded_text_train.json'
    # )

    val_set = DatasetCreator(
        split='val', 
        parquet_file='data/original/validation.parquet',
        encoded_text_json='data/wikitext2/encoded_text_val.json'
    )

    test_set = DatasetCreator(
        split='test', 
        parquet_file='data/original/test.parquet',
        encoded_text_json='data/wikitext2/encoded_text_test.json'
    )
