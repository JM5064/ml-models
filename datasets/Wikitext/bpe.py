import collections
from datasets.Wikitext.special_chars import UNK, EOS
import json
import time


def get_pairs(chars):
    """Outputs a frequency map of token pairs
    Args:
        chars (list[str]) : the current tokenized text list 
            eg. ['the', 'qu', 'ick', ...]
    
    Returns:
        pairs (dict[tuple[str, str], int]) : a frequency list of all pairs in chars
            eg. { ('the', 'qu') : 4, ... }
    """

    pairs = collections.defaultdict(int)

    for i in range(len(chars) - 1):
        pairs[chars[i], chars[i + 1]] += 1

    return pairs


def merge_pair(pair, chars, vocab, pairs):
    """Merge pair in chars while updating vocab and pairs dict
        Args:
            pair (tuple[str, str]) : the pair to merge
            chars (list[str]) : the current tokenized text list 
            vocab (dict[str, int]) : the current vocabulary
            pairs (dict[tuple[str, str], int]) : the current frequency map of token pairs

        Returns:
            new_chars (list[str]) : an updated tokenized text list
    """

    merged = "".join(pair)
    new_chars = []

    # Remove pair from pairs
    if pair in pairs:
        del pairs[pair]

    i = 0
    while i < len(chars):
        if i+1 < len(chars) and (chars[i], chars[i+1]) == pair:

            new_chars.append(merged)
            
            # update counts for vocabulary
            vocab[chars[i]] -= 1
            vocab[chars[i+1]] -= 1
            vocab[merged] += 1

            # Remove tokens no longer in vocabulary
            if vocab[chars[i]] == 0:
                del vocab[chars[i]]
            if vocab[chars[i+1]] == 0:
                del vocab[chars[i+1]]

            # Update counts for previous and next pairs with merged pair
            # e.g. (a, b) (b, c) (c, d) -> (a, bc) (bc, d)
            if i-1 >= 0:
                prev_pair = (chars[i-1], chars[i])
                new_pair = (chars[i-1], merged)

                pairs[prev_pair] -= 1
                pairs[new_pair] += 1

                if pairs[prev_pair] == 0:
                    del pairs[prev_pair]
            if i + 2 < len(chars):
                prev_pair = (chars[i+1], chars[i+2])
                new_pair = (merged, chars[i+2])

                pairs[prev_pair] -= 1
                pairs[new_pair] += 1

                if pairs[prev_pair] == 0:
                    del pairs[prev_pair]

            i += 1
        else:
            new_chars.append(chars[i])
        i += 1

    return new_chars


def split_chars(text_arr):
    """Filter and split sentences into individual characters

    Args:
        text_arr (list[str]): list of sentences to split

    Returns:
        list[str]: list of characters, pluse <EOS> and <UNK> symbols
    """
    chars = []
    for t in text_arr:
        # Filter for unicode characters up to greek and coptic
        for c in t:
            if ord(c) <= 1023:
                chars.append(c)
            else:
                # Unknown character
                chars.append(UNK)

        # Append end of text character after each sample
        chars.append(EOS)

    # add BOW, EOW markers
    # sym = ".,:@- !?;[]()+=&%$#/\"'"
    # if chars[0] not in sym:
    #     chars[0] = "_" + chars[0]
    # for i in range(1, len(chars) - 1):
    #     if chars[i] in sym:
    #         continue
    #     if chars[i - 1] == " ":
    #         chars[i] = "_" + chars[i]
    #     if chars[i + 1] == " ":
    #         chars[i] = chars[i] + "_"
    # if chars[-1] not in sym:
    #     chars[-1] = chars[-1] + "_"

    return chars


def get_vocab(chars):
    """Make vocabulary from tokenized text

    Args:
        chars (list[str]): list of tokens

    Returns:
        dict[str, int]: dictionary with count of each unique token
    """
    vocab = collections.Counter(chars)

    return vocab


def make_mapping(vocab):
    """Make token-integer mapping as two dictionaries

    Args:
        vocab (dict[str, int]): dictionary with count of each unique token

    Returns:
        dict[str, int]: token to integer mapping
        dict[int, str]: integer to token mapping
    """
    # token to number
    encoding = {
        token: i for i, token in enumerate(vocab)
    }

    # number to token
    decoding = {
        i : token for i, token in enumerate(vocab)
    }

    return encoding, decoding


def save_to_file(data, file_path, indent=None):
    """Save data to json file

    Args:
        data (dict): object to save
        file_path (str): file name to save data to
        indent (int|str, optional): indent character/size for file, default None
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def load_from_file(file_path):
    """Load json data from file

    Args:
        file_path (str): file to read from

    Returns:
        dict: file contents as a dictionary
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def bpe(text, num_merges):
    """Run byte pair encoding (BPE) algorithm on text

    Args:
        text (list[str]): list of sentences to tokenize
        num_merges (int): number of merges to perform

    Returns:
        list[str]: tokenized text, as list of tokens in order
        dict[str, int]: unique tokens and their frequency
        list[tuple[str, str]]: list of token pairs that were merged by the algorithm
    """
    print("Performing BPE")

    chars = split_chars(text)
    vocab = get_vocab(chars)
    pairs = get_pairs(chars)

    merge_pairs = []

    # Perform merges
    for i in range(num_merges):
        s = time.time()

        # Find most common pair and merge
        best_pair = max(pairs, key=pairs.get)
        merge_pairs.append(best_pair)

        chars = merge_pair(best_pair, chars, vocab, pairs)
        
        print("Merge ", i, "took", (time.time() - s) * 1000, "ms")

    return chars, vocab, merge_pairs


def apply_merge_pairs(text, merge_pairs):
    """Returns tokenized text given a list of merge pairs

    Args:
        text (list[str]): list of sentences to tokenize
        merge_pairs (list[tuple[str, str]]): list of token pairs to merge into tokens

    Returns:
        list[str]: tokenized text, as list of tokens in order
    """

    print("Applying merge pairs")

    chars = split_chars(text)
    vocab = get_vocab(chars)
    pairs = get_pairs(chars)

    for i, pair in enumerate(merge_pairs):
        s = time.time()

        # Convert pair from array to tuple and merge
        pair = (pair[0], pair[1])
        chars = merge_pair(pair, chars, vocab, pairs)
        
        # print("Merge ", i, "took", (time.time() - s) * 1000, "ms")

    return chars