import os
import re
import multiprocessing as mp
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def _find_chunk_boundaries(
    file_path: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    with open(file_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


def _pretokenize_chunk(args):
    """
    Pre-tokenize a chunk of text, splitting on special tokens first.
    
    Args:
        args: Tuple of (file_path, start, end, pattern, special_tokens)
    
    Returns:
        List of words where each word is a list of byte values
    """
    file_path, start, end, pattern, special_tokens = args
    
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        text_chunk = chunk_bytes.decode("utf-8", errors="ignore")
    
    # Create GPT-2 byte ordering mapping
    def gpt2_bytes_to_unicode() -> dict[int, str]:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))
    
    gpt2_byte_order = list(gpt2_bytes_to_unicode().keys())
    byte_to_token_id = {byte_val: len(special_tokens) + i for i, byte_val in enumerate(gpt2_byte_order)}

    # Split on special tokens to prevent merging across boundaries
    if special_tokens:
        # Escape special characters in special tokens and join with |
        escaped_special_tokens = [re.escape(token) for token in special_tokens]
        delimiter_pattern = "|".join(escaped_special_tokens)
        # Split the text on special tokens
        segments = re.split(delimiter_pattern, text_chunk)
    else:
        segments = [text_chunk]
    
    # Pre-tokenize each segment separately
    words = []
    for segment in segments:
        if segment.strip():  # Skip empty segments
            # Use re.finditer as specified in the instructions
            for match in re.finditer(pattern, segment):
                pre_token = match.group()
                if pre_token.strip():  # Skip empty tokens
                    token_bytes = pre_token.encode('utf-8')
                    # Convert each byte to its corresponding token ID using GPT-2 ordering
                    word_token_ids = [byte_to_token_id[byte_val] for byte_val in token_bytes]
                    words.append(word_token_ids)
    
    return words


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.
    
    Args:
        input_path (str): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size 
                         (including the initial byte vocabulary, vocabulary items produced 
                         from merging, and any special tokens).
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special 
                                   tokens do not otherwise affect BPE training.
    
    Returns:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the 
                                 vocabulary) to bytes (token bytes).
        merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training. Each 
                                           list item is a tuple of bytes (<token1>, <token2>), 
                                           representing that <token1> was merged with <token2>. 
                                           The merges should be ordered by order of creation.
    """
    # Read the input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize vocabulary with special tokens first (starting at ID 0)
    vocab = {}
    
    # Add special tokens to vocabulary at the beginning
    for i, special_token in enumerate(special_tokens):
        special_token_bytes = special_token.encode('utf-8')
        vocab[i] = special_token_bytes
    
    # Add all possible bytes using GPT-2 ordering
    # Get the GPT-2 byte ordering (printable chars first, then shifted non-printable)
    def gpt2_bytes_to_unicode() -> dict[int, str]:
        # These 188 integers can used as-is, since they are not whitespace or control characters.
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        # now get the representations of the other 68 integers that do need shifting
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))
    
    gpt2_byte_order = list(gpt2_bytes_to_unicode().keys())
    for i, byte_val in enumerate(gpt2_byte_order):
        vocab[len(special_tokens) + i] = bytes([byte_val])
    
    # Create a mapping from byte values to token IDs for efficient lookup
    byte_to_token_id = {byte_val: len(special_tokens) + i for i, byte_val in enumerate(gpt2_byte_order)}

    # Pre-tokenize using regex pattern similar to GPT-2
    # This pattern handles contractions, words, numbers, punctuation, and whitespace
    # Since Python's re doesn't support \p{L} and \p{N}, we use character classes
    pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-ZÀ-ÿĀ-žА-я]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    
    # Parallelize pre-tokenization for better performance
    num_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes to avoid overhead
    
    # Check file size to decide whether to parallelize
    file_size = os.path.getsize(input_path)
    
    if False:  # Temporarily disable parallelization for debugging
        # Use <|endoftext|> as the split token for chunk boundaries
        split_special_token = b"<|endoftext|>"
        
        # Get chunk boundaries that align with special token boundaries
        boundaries = _find_chunk_boundaries(input_path, num_processes, split_special_token)
        
        # Create chunks for parallel processing
        chunks = []
        for i in range(len(boundaries) - 1):
            chunks.append((input_path, boundaries[i], boundaries[i + 1], pattern, special_tokens))
        
        # Process chunks in parallel
        with mp.Pool(num_processes) as pool:
            chunk_results = pool.map(_pretokenize_chunk, chunks)
        
        # Combine results
        words = []
        for chunk_words in chunk_results:
            words.extend(chunk_words)
    else:
        # For small files, use single-threaded approach
        # Split on special tokens to prevent merging across boundaries
        if special_tokens:
            # Escape special characters in special tokens and join with |
            escaped_special_tokens = [re.escape(token) for token in special_tokens]
            delimiter_pattern = "|".join(escaped_special_tokens)
            # Split the text on special tokens
            segments = re.split(delimiter_pattern, text)
        else:
            segments = [text]
        
        # Pre-tokenize each segment separately
        words = []
        for segment in segments:
            if segment.strip():  # Skip empty segments
                # Use re.finditer as specified in the instructions
                for match in re.finditer(pattern, segment):
                    pre_token = match.group()
                    if pre_token.strip():  # Skip empty tokens
                        token_bytes = pre_token.encode('utf-8')
                        # Convert each byte to its corresponding token ID using GPT-2 ordering
                        word_token_ids = [byte_to_token_id[byte_val] for byte_val in token_bytes]
                        words.append(word_token_ids)
    
    # List to store merges in order
    merges = []
    
    # Calculate how many merges we can perform
    # vocab_size = initial_bytes (256) + special_tokens + merges
    num_merges = vocab_size - 256 - len(special_tokens)
    
    if num_merges <= 0:
        return vocab, merges
    
    # Optimize: Pre-compute word frequencies to avoid redundant processing
    word_counts = Counter(tuple(word) for word in words)
    
    # Perform BPE merges
    for merge_idx in range(num_merges):
        # Count all adjacent token pairs across all unique words
        pairs = defaultdict(int)
        # Sort word_counts.items() to ensure deterministic iteration
        for word_tuple, count in sorted(word_counts.items()):
            word = list(word_tuple)
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += count
        
        if not pairs:
            break
            
        # Find the most frequent pair, with lexicographic tie-breaking
        # Single pass: find max count and collect pairs with that count
        max_count = 0
        max_pairs = []
        
        # Sort pairs.items() to ensure deterministic iteration order
        for pair, count in sorted(pairs.items()):
            if count > max_count:
                max_count = count
                max_pairs = [pair]  # Start new list with this pair
            elif count == max_count:
                max_pairs.append(pair)  # Add to existing max pairs
        
        # Use lexicographic tie-breaking: choose the lexicographically greatest pair
        best_pair = max(max_pairs, key=lambda pair: (vocab[pair[0]], vocab[pair[1]]))
        
        # Create new token ID for the merged pair
        # IDs 0 to len(special_tokens)-1 are for special tokens
        # IDs len(special_tokens) to len(special_tokens)+255 are for bytes
        # New merged tokens start after that
        new_token_id = len(special_tokens) + 256 + merge_idx
        
        # Add the merged token to vocabulary
        token1_bytes = vocab[best_pair[0]]
        token2_bytes = vocab[best_pair[1]]
        merged_token_bytes = token1_bytes + token2_bytes
        vocab[new_token_id] = merged_token_bytes
        
        # Add this merge to our list (as bytes)
        merges.append((token1_bytes, token2_bytes))
        
        # Apply the merge to all unique words and update counts
        new_word_counts = Counter()
        # Sort word_counts.items() to ensure deterministic iteration
        for word_tuple, count in sorted(word_counts.items()):
            word = list(word_tuple)
            new_word = []
            i = 0
            while i < len(word):
                # Check if current and next token form the best pair
                if (i < len(word) - 1 and 
                    word[i] == best_pair[0] and 
                    word[i + 1] == best_pair[1]):
                    # Replace the pair with the new token ID
                    new_word.append(new_token_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] += count
        word_counts = new_word_counts
    
    return vocab, merges


