from typing import Any, Dict, List, Optional
from konlpy.tag import Okt
import pandas as pd

def coarse_pos_from_okt(tag: Optional[str]) -> str:
    if tag is None: return "UNK"
    if tag == "Josa": return "JOSA"
    if tag == "Noun": return "NOUN"
    if tag in {"Verb", "Adjective"}: return "VERB_OR_ADJ"
    if tag in {"Adverb", "Modifier", "Determiner"}: return "MODIFIER"
    if tag in {"Suffix", "VerbPrefix"}: return "AUX_OR_SUFFIX"
    if tag in {"Punctuation", "KoreanParticle"}: return "PUNCT_OR_SYMBOL"
    if tag in {"Number", "Alpha", "Foreign"}: return "OTHER_LEXICAL"
    return "OTHER"

def char_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))

def clean_slice_for_display(text: str, start: int, end: int) -> str:
    raw = text[start:end].replace(" ", "␠").replace("\n", "↵").replace("\t", "⇥")
    return raw if raw else "∅"

def pretty_special_token(token_str: str) -> str:
    return {"<|begin_of_text|>": "BOS", "<|end_of_text|>": "EOS", "<|eot_id|>": "EOT", "<s>": "BOS", "</s>": "EOS", "<pad>": "PAD"}.get(token_str, token_str)

def okt_morphs_with_spans(text: str, okt: Okt) -> List[Dict[str, Any]]:
    pos_pairs = okt.pos(text, norm=False, stem=False)
    out = []
    cursor = 0
    for morph_idx, (form, tag) in enumerate(pos_pairs):
        if form == "":
            continue
        found = text.find(form, cursor)
        if found < 0:
            cursor2 = cursor
            while cursor2 < len(text) and text[cursor2].isspace():
                cursor2 += 1
            found = text.find(form, cursor2)
        low_conf = False
        if found < 0:
            found = text.find(form)
            low_conf = True
        if found < 0:
            start, end = -1, -1
            low_conf = True
        else:
            start, end = found, found + len(form)
            cursor = end
        out.append({"morph_idx": morph_idx, "form": form, "tag": tag, "coarse_pos": coarse_pos_from_okt(tag), "start": int(start), "end": int(end), "length": int(max(0, end - start)), "is_josa_morph": tag == "Josa", "span_low_confidence": bool(low_conf)})
    return out

def align_sentence(text: str, tokenizer, sentence_id: int, max_length: int, strict_josa_ratio: float, okt: Okt) -> Dict[str, Any]:
    enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length, return_offsets_mapping=True, return_special_tokens_mask=True)
    morphs = okt_morphs_with_spans(text, okt)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets = enc["offset_mapping"]
    special_mask = enc["special_tokens_mask"]
    token_strs = tokenizer.convert_ids_to_tokens(input_ids)
    token_meta = []
    for token_idx, (tok_str, offset, is_special) in enumerate(zip(token_strs, offsets, special_mask)):
        s, e = int(offset[0]), int(offset[1])
        if bool(is_special) or (s == 0 and e == 0):
            token_meta.append({"sentence_id": sentence_id, "token_idx": token_idx, "token_text": tok_str, "token_display": pretty_special_token(tok_str), "char_start": s, "char_end": e, "char_len": max(0, e - s), "rel_pos": token_idx / max(1, len(input_ids) - 1), "is_special": True, "best_morph_idx": None, "best_morph_form": None, "best_pos_tag": None, "major_coarse_pos": "SPECIAL", "coarse_pos": "SPECIAL", "josa_overlap_chars": 0, "josa_overlap_ratio": 0.0, "is_josa": False, "is_josa_strict": False, "alignment_confidence": "SPECIAL"})
            continue
        token_len = max(1, e - s)
        overlaps = []
        josa_overlap = 0
        for m in morphs:
            if m["start"] < 0 or m["end"] < 0:
                continue
            ov = char_overlap(s, e, m["start"], m["end"])
            if ov > 0:
                overlaps.append((ov, m))
                if m["is_josa_morph"]:
                    josa_overlap += ov
        if overlaps:
            overlaps = sorted(overlaps, key=lambda x: x[0], reverse=True)
            best_ov, best_m = overlaps[0]
            best_tag = best_m["tag"]
            best_form = best_m["form"]
            best_idx = best_m["morph_idx"]
            major_coarse = best_m["coarse_pos"]
            align_conf = "OK" if best_ov / token_len >= 0.50 else "MIXED_LOW_OVERLAP"
        else:
            best_tag, best_form, best_idx, major_coarse, align_conf = None, None, None, "UNK_ALIGN", "NO_MORPH_OVERLAP"
        josa_ratio = float(josa_overlap / token_len)
        is_josa_any = josa_overlap > 0
        token_meta.append({"sentence_id": sentence_id, "token_idx": token_idx, "token_text": tok_str, "token_display": clean_slice_for_display(text, s, e), "char_start": s, "char_end": e, "char_len": max(0, e - s), "rel_pos": token_idx / max(1, len(input_ids) - 1), "is_special": False, "best_morph_idx": best_idx, "best_morph_form": best_form, "best_pos_tag": best_tag, "major_coarse_pos": major_coarse, "coarse_pos": "JOSA" if is_josa_any else major_coarse, "josa_overlap_chars": int(josa_overlap), "josa_overlap_ratio": josa_ratio, "is_josa": bool(is_josa_any), "is_josa_strict": bool(josa_ratio >= strict_josa_ratio), "alignment_confidence": align_conf})
    return {"sentence_id": sentence_id, "text": text, "input_ids": input_ids, "attention_mask": attention_mask, "token_meta": token_meta, "morphs": morphs}

def align_corpus(corpus: List[str], tokenizer, max_length: int, strict_josa_ratio: float):
    okt = Okt()
    encoded_samples, morph_rows, token_rows = [], [], []
    for sid, text in enumerate(corpus):
        item = align_sentence(text, tokenizer, sid, max_length, strict_josa_ratio, okt)
        encoded_samples.append(item)
        for m in item["morphs"]:
            morph_rows.append({"sentence_id": sid, "text": text, **m})
        for meta in item["token_meta"]:
            token_rows.append({"text": text, "seq_len": len(item["input_ids"]), **meta})
    return encoded_samples, pd.DataFrame(morph_rows), pd.DataFrame(token_rows)
