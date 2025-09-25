# Initially I have implemented the tokenizer with python from scratch but it was very slow to work with so I have switched with to tiktoken library 

# Tokenizer from scratch (Byte Pair Encoding Algorithm)
#####################################################################################################

# import regex as re
# import requests

# class Tokenizer:
#     def __init__(self):
#         self.merges = {}
#         self.vocab = {}
#         self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

#     def _get_pair_counts(self, tokens):
#         counts = {}
#         for i in range(len(tokens) - 1):
#             pair = (tokens[i], tokens[i+1])
#             counts[pair] = counts.get(pair, 0) + 1
#         return sorted(((v, k) for k, v in counts.items()), reverse=True)

#     def _merge(self, tokens, pair, new_idx):
#         new_tokens = []
#         i = 0
#         while i < len(tokens):
#             if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
#                 new_tokens.append(new_idx)
#                 i += 2
#             else:
#                 new_tokens.append(tokens[i])
#                 i += 1
#         return new_tokens

#     def train(self, text, num_merges, verbose=False):
#         print("Starting training...")
#         text_chunks = self.pattern.findall(text)
#         tokens = []
#         for chunk in text_chunks:
#             tokens.extend(list(chunk.encode("utf-8")))

#         merges = {}
#         for i in range(num_merges):
#             pair_counts = self._get_pair_counts(tokens)
#             if not pair_counts:
#                 break
            
#             top_pair = pair_counts[0][1]
#             new_idx = 256 + i
#             merges[top_pair] = new_idx
#             tokens = self._merge(tokens, top_pair, new_idx)
#             if verbose and (i + 1) % 50 == 0:
#                 print(f"  Merge {i+1}/{num_merges} completed...")
        
#         self.merges = merges
#         self._build_vocab()
#         print("Training finished!")

#     def _build_vocab(self):
#         self.vocab = {idx: bytes([idx]) for idx in range(256)}
#         for (p1, p2), idx in self.merges.items():
#             self.vocab[idx] = self.vocab[p1] + self.vocab[p2]

#     def save(self, filepath="merge_rules.bpe"):
#         print(f"Saving merge rules to {filepath}...")
#         with open(filepath, 'w', encoding="utf-8") as f:
#             for (p1, p2) in self.merges:
#                 f.write(f"{p1} {p2}\n")
#         print("Done.")

#     def load(self, filepath="merge_rules.bpe"):
#         print(f"Loading merge rules from {filepath}...")
#         merges = {}
#         with open(filepath, 'r', encoding="utf-8") as f:
#             for i, line in enumerate(f):
#                 p1, p2 = line.strip().split()
#                 merges[(int(p1), int(p2))] = 256 + i
#         self.merges = merges
#         self._build_vocab()
#         print("Tokenizer loaded.")

#     def encode(self, text):
#         text_chunks = self.pattern.findall(text)
#         tokens = []
#         for chunk in text_chunks:
#             tokens.extend(list(chunk.encode("utf-8")))

#         for pair, new_idx in self.merges.items():
#             tokens = self._merge(tokens, pair, new_idx)
#         return tokens

#     def decode(self, ids):
#         byte_chunk = b"".join(self.vocab[idx] for idx in ids)
#         return byte_chunk.decode("utf-8", errors="replace")
    
# if __name__ == "__main__":
#     print("Dowloading text for training...")
#     url = "https://www.gutenberg.org/files/100/100-0.txt"
#     response = requests.get(url)
#     text = response.text
#     tokenizer = Tokenizer()
#     print("Training tokenizer...")
#     tokenizer.train(text, num_merges=500, verbose=True)
#     tokenizer.save("my_tokenizer.bpe")






