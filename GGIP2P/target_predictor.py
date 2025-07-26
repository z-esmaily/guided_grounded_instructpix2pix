import re
import torch
import spacy
from peft import PeftModel, PeftConfig
from transformers import BertTokenizerFast, BertForTokenClassification

nlp = spacy.load("en_core_web_sm")

class TargetPredictor:
    def __init__(self, model_path):
        self.config = PeftConfig.from_pretrained(model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        self.base_model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=2)
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.directional_words = {
            "left", "right", "up", "down", "upper", "lower",
            "top", "under", "below", "underneath", "above", "over",
            "bottom", "front", "back", "center", "middle"
        }

    def extract_noun_phrase(self, sentence, target_indices):
        doc = nlp(sentence)
        token_to_word_map = self.tokenizer(sentence, return_offsets_mapping=True)["offset_mapping"]
        target_char_indices = set()
        for token_idx in target_indices:
            if token_idx < len(token_to_word_map):
                start, end = token_to_word_map[token_idx]
                target_char_indices.update(range(start, end))
        for chunk in doc.noun_chunks:
            phrase_start, phrase_end = chunk.start_char, chunk.end_char
            phrase_indices = set(range(phrase_start, phrase_end))
            if target_char_indices & phrase_indices:
                return chunk.text
        return ""

    def remove_phrase(self, sentence, phrase):
        if phrase:
            sentence = re.sub(rf'\b{re.escape(phrase)}\b', '', sentence).strip()
            sentence = " ".join(sentence.split())
        return sentence

    def extract_directional_phrase(self, sentence, directional_words):
        directional_patterns = {
            'prefixes': {'in', 'at', 'on', 'to', 'from', 'the'},
            'suffixes': {'of', 'side', 'side of', 'hand', 'corner', 'end', 'part'}
        }
        words = sentence.lower().split()
        for i, dir_word in enumerate(words):
            if dir_word in directional_words:
                phrase_indices = {i}
                if i > 1 and words[i - 2] in directional_patterns['prefixes']:
                    phrase_indices.add(i - 2)
                if i > 0 and words[i - 1] in directional_patterns['prefixes']:
                    phrase_indices.add(i - 1)
                if i < len(words) - 1 and words[i + 1] in directional_patterns['suffixes']:
                    phrase_indices.add(i + 1)
                    if i < len(words) - 2 and words[i + 2] in directional_patterns['suffixes']:
                        phrase_indices.add(i + 2)
                phrase_words = [words[idx] for idx in sorted(phrase_indices)]
                extracted_phrase = " ".join(phrase_words)
                return extracted_phrase, dir_word if extracted_phrase in sentence else dir_word
        return "", ""

    def find_target(self, sentence):
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            predictions = torch.argmax(logits, dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        target_spans = []
        current_span = []
        current_indices = []
        for idx, (token, pred) in enumerate(zip(tokens, predictions)):
            if pred == 1:
                current_span.append(token)
                current_indices.append(idx)
            elif current_span:
                target_spans.append({
                    'tokens': current_span.copy(),
                    'indices': current_indices.copy()
                })
                current_span = []
                current_indices = []
        if current_span:
            target_spans.append({'tokens': current_span, 'indices': current_indices})
        if target_spans:
            best_span = target_spans[0]
            target_tokens = []
            for token in best_span['tokens']:
                if token.startswith("##") and target_tokens:
                    target_tokens[-1] += token[2:]
                else:
                    target_tokens.append(token)
            target = self.tokenizer.convert_tokens_to_string(target_tokens)
            return target, best_span['indices']
        return "", []

    def predict(self, sentence):
        target, target_indices = self.find_target(sentence)
        target_phrase = self.extract_noun_phrase(sentence, target_indices)
        removed_target_sentence = self.remove_phrase(sentence, target_phrase)
        directional_phrase, direction_word = self.extract_directional_phrase(sentence, self.directional_words)
        final_sentence = self.remove_phrase(removed_target_sentence, directional_phrase)
        second_target, second_target_indices = self.find_target(final_sentence)
        second_target_phrase = self.extract_noun_phrase(final_sentence, second_target_indices)

        return {
            'sentence': sentence,
            'detected_target': target if target else "No target found",
            'target_phrase': target_phrase,
            'removed_target_sentence': removed_target_sentence,
            'direction_word': direction_word,
            'directional_phrase': directional_phrase,
            'final_sentence': final_sentence,
            'second_target': second_target if second_target else "No second target found",
            'second_target_phrase': second_target_phrase,
        }