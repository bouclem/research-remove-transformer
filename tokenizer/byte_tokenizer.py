"""
Byte-level tokenizer — maps raw bytes (0-255) to token IDs.

No training needed, no external dependencies.
Supports arbitrary text via UTF-8 encoding.
"""

import torch


class ByteTokenizer:
    """Simple byte-level tokenizer: text <-> byte IDs (0-255)."""

    def __init__(self):
        self.vocab_size = 256
        self.pad_id = 0   # null byte
        self.bos_id = 1   # start of heading
        self.eos_id = 2   # end of text (STX/ETX control chars)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to list of byte-token IDs."""
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Decode byte-token IDs back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # Filter out non-printable control chars for clean output
        byte_vals = bytes([b & 0xFF for b in ids])
        return byte_vals.decode("utf-8", errors="replace")

    def decode_lossless(self, ids: list[int] | torch.Tensor) -> bytes:
        """Decode to raw bytes (no UTF-8 decoding)."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes([b & 0xFF for b in ids])


if __name__ == "__main__":
    tok = ByteTokenizer()
    text = "Hello, world!"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Text:    {text!r}")
    print(f"IDs:     {ids}")
    print(f"Decoded: {decoded!r}")
    print(f"Match:   {text == decoded}")
