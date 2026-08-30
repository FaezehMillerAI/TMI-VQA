import torch
import torch.nn as nn
from typing import List, Dict, Any

class ModularTextEncoder(nn.Module):
    def __init__(self, encoder_type: str = "pubmedbert", text_dim: int = 768, max_seq_len: int = 128):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.text_dim = text_dim
        self.max_seq_len = max_seq_len
        
        # Simple vocabulary for mock encoding if HuggingFace is offline
        self.vocab = {"<pad>": 0, "<unk>": 1, "yes": 2, "no": 3, "is": 4, "there": 5, "evidence": 6, "of": 7, "pleural": 8, "effusion": 9}
        self.embedding = nn.Embedding(1000, text_dim, padding_idx=0)
        self.lstm = nn.LSTM(text_dim, text_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        
        self.use_hf = False
        if "biomedclip" in self.encoder_type:
            try:
                from transformers import AutoTokenizer, CLIPTextModel
                model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                print(f"Loading real BiomedCLIP Text Encoder: {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.hf_model = CLIPTextModel.from_pretrained(model_name)
                self.use_hf = True
                print("Successfully loaded pre-trained BiomedCLIP Text Encoder!")
            except Exception as e:
                print(f"Failed to load BiomedCLIP Text Model: {e}. Using mock LSTM encoder.")
        else:
            try:
                from transformers import AutoTokenizer, AutoModel
                model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
                print(f"Loading real PubMedBERT Text Encoder: {model_name}...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    self.hf_model = AutoModel.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.hf_model = AutoModel.from_pretrained(model_name)
                self.use_hf = True
                print("Successfully loaded pre-trained PubMedBERT Text Encoder!")
            except Exception as e:
                print(f"Failed to load PubMedBERT: {e}. Using mock LSTM encoder.")

    def tokenize_and_encode_mock(self, texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        """Simple rule-based tokenizer and sequence encoder when transformers are unavailable."""
        batch_ids = []
        for text in texts:
            words = text.lower().replace("?", "").replace(".", "").split()
            ids = []
            for w in words:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab) % 1000
                ids.append(self.vocab[w])
            # Pad
            if len(ids) < self.max_seq_len:
                ids += [0] * (self.max_seq_len - len(ids))
            else:
                ids = ids[:self.max_seq_len]
            batch_ids.append(ids)
            
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        attention_mask = (input_ids != 0).long()
        
        # Embed and LSTM
        embeds = self.embedding(input_ids) # [B, L, D]
        outputs, (hn, _) = self.lstm(embeds) # [B, L, D]
        pooled = hn.transpose(0, 1).flatten(1) # [B, D]
        
        return {
            "last_hidden_state": outputs,
            "pooler_output": pooled,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def forward(self, texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        # Always fallback to mock if use_hf is false or not initialized
        if not self.use_hf:
            return self.tokenize_and_encode_mock(texts, device)
            
        # In case HuggingFace is enabled and initialized
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt").to(device)
            outputs = self.hf_model(**inputs)
            return {
                "last_hidden_state": outputs.last_hidden_state,
                "pooler_output": outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1),
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        except Exception as e:
            # Silent fallback
            return self.tokenize_and_encode_mock(texts, device)

if __name__ == "__main__":
    x = ["Is there evidence of pleural effusion?", "Yes"]
    device = torch.device("cpu")
    encoder = ModularTextEncoder()
    res = encoder(x, device)
    print("last_hidden_state:", res["last_hidden_state"].shape)
    print("pooler_output:", res["pooler_output"].shape)
