import regex as re
from functools import lru_cache
from typing import Iterable, Iterator, Optional

class Tokenizer:
    """Fast(er) BPE tokenizer with greedy pair merges, caching, and streaming I/O."""

    WORD_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        self.vocab = vocab # token ID to bytes 
        self.vocab_reverse: dict[bytes, int] = {b: i for i, b in vocab.items()}

        # Rank dict: pair -> order
        # (lower rank = higher priority)
        self.rank: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}

        # Precompile regex
        self.word_re = re.compile(self.WORD_PAT)

        self.special_tokens = special_tokens or []
        if self.special_tokens:
            # Sort by length (longest first) for longest-match behavior, refer tests/test_tokenizer.py::test_overlapping_special_tokens
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pat = f"({'|'.join(re.escape(t) for t in sorted_tokens)})"
            self.special_re = re.compile(pat)
            # Precompute ids for special tokens (avoid repeated encode/lookups)
            self.special_to_id = {
                t: self.vocab_reverse[t.encode("utf-8")] for t in self.special_tokens
            }
        else:
            self.special_re = None
            self.special_to_id = {}

    @property
    def eos_token_id(self) -> Optional[int]:
        """Return the ID of the end-of-sequence token, if defined."""
        return self.special_to_id.get("<|endoftext|>") if self.special_tokens else None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None, **kw):
        # Same as your version, kept for compatibility
        vocab = {}
        with open(vocab_filepath, "rb") as f:
            vocab_size = int.from_bytes(f.read(4), "little")
            for _ in range(vocab_size):
                token_id = int.from_bytes(f.read(4), "little")
                token_len = int.from_bytes(f.read(4), "little")
                token = f.read(token_len)
                vocab[token_id] = token

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "rb") as f:
            num_merges = int.from_bytes(f.read(4), "little")
            for _ in range(num_merges):
                a_len = int.from_bytes(f.read(4), "little")
                a = f.read(a_len)
                b_len = int.from_bytes(f.read(4), "little")
                b = f.read(b_len)
                merges.append((a, b))

        return cls(vocab, merges, special_tokens, **kw)

    # ---------- public API ----------

    def encode(self, text: str) -> list[int]:
        """Encode full text into token ids (returns a list)."""
        if not self.special_tokens:
            return self._encode_plain(text)

        out: list[int] = []
        # Split while preserving specials (capturing group)
        for part in self.special_re.split(text):
            if not part:
                continue
            sid = self.special_to_id.get(part)
            if sid is not None:
                out.append(sid)
            else:
                out.extend(self._encode_plain(part))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily yield token ids from an iterable of strings (e.g., file lines)."""
        if not self.special_tokens:
            for text in iterable:
                yield from self._encode_plain_iter(text)
        else:
            for text in iterable:
                for part in self.special_re.split(text):
                    if not part:
                        continue
                    sid = self.special_to_id.get(part)
                    if sid is not None:
                        yield sid
                    else:
                        yield from self._encode_plain_iter(part)

    # ---------- fast paths (no specials) ----------

    def _encode_plain(self, text: str) -> list[int]:
        """Encode text with the word regex and greedy BPE; returns list[int]."""
        out: list[int] = []
        get_id = self.vocab_reverse.get
        for m in self.word_re.finditer(text):
            w_bytes = m.group().encode("utf-8")
            for piece in self._bpe_word(w_bytes):
                tid = get_id(piece)
                if tid is None:
                    tid = 65533  # Unicode replacement character
                out.append(tid)
        return out

    def _encode_plain_iter(self, text: str) -> Iterator[int]:
        """Generator version (avoids building a list)."""
        get_id = self.vocab_reverse.get
        for m in self.word_re.finditer(text):
            w_bytes = m.group().encode("utf-8")
            for piece in self._bpe_word(w_bytes):
                tid = get_id(piece)
                if tid is None:
                    tid = 65533  # Unicode replacement character
                yield tid

    # ---------- BPE core (greedy best-ranked adjacent pair) ----------

    @lru_cache(maxsize=200_000)
    def _bpe_word(self, word_utf8: bytes) -> tuple[bytes, ...]:
        """
        Encode a single pretoken (as UTF-8 bytes) into BPE bytepieces.
        Greedy: repeatedly merge the pair with the lowest rank (earliest merge).
        Caches results per distinct byte sequence.
        """
        # Start from individual bytes (byte-level base vocab)
        toks = [bytes([b]) for b in word_utf8]
        if len(toks) <= 1:
            return tuple(toks)

        rank = self.rank  # local bind for speed

        while True:
            smallest_idx = -1
            smallest_rank = 1 << 60

            # Find smallest-ranked pair
            for i in range(len(toks) - 1):
                r = rank.get((toks[i], toks[i + 1]))
                if r is not None and r < smallest_rank:
                    smallest_rank = r
                    smallest_idx = i

            if smallest_idx < 0:
                break  # no mergeable pair remains

            a = toks[smallest_idx]
            b = toks[smallest_idx + 1]
            ab = a + b

            # Merge **all occurrences** of this best pair in one pass
            new = []
            i = 0
            n = len(toks)
            while i < n:
                if i < n - 1 and toks[i] == a and toks[i + 1] == b:
                    new.append(ab)
                    i += 2
                else:
                    new.append(toks[i])
                    i += 1
            toks = new

        return tuple(toks)


    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        
        Args:
            ids: list[int] - Sequence of token IDs
            
        Returns:
            str - Decoded text
        """
        # Convert IDs to bytes
        byte_tokens = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_tokens.append(self.vocab[token_id])
            else:
                # Handle unknown token IDs - use replacement character
                byte_tokens.append(b'\xef\xbf\xbd')  # Unicode replacement character in UTF-8
        
        # Concatenate all bytes
        all_bytes = b''.join(byte_tokens)
        
        # Decode to string
        try:
            return all_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # 'replace': automatically replace malformed data with the replacement marker.
            return all_bytes.decode('utf-8', errors='replace')