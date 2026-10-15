
from pathlib import Path

import torch
import kenlm

from src.text_encoder import CTCTextEncoder


def main():
    lm_path = Path("resources/lm/3-gram.pruned.1e-7.arpa")

    assert lm_path.exists(), f"LM not found: {lm_path}"

    # 1. Проверяем языковую модель
    lm = kenlm.Model(str(lm_path))
    print("the in LM:", "the" in lm)
    print("THE in LM:", "THE" in lm)
    print("cat in LM:", "cat" in lm)
    print("CAT in LM:", "CAT" in lm)

    print("LM order:", lm.order)
    print("the in LM:", "the" in lm)
    print("THE in LM:", "THE" in lm)

    # 2. Создаём CTCTextEncoder
    encoder = CTCTextEncoder()

    print("Vocabulary size:", len(encoder))
    print("Blank index:", encoder.char2ind[encoder.EMPTY_TOK])

    # 3. Загружаем декодер с LM
    encoder.load_lm(
        lm_path=lm_path,
        alpha=0.5,
        beta=1.0,
    )

    print("LM decoder initialized!")

    # 4. Создаём искусственные выходы CTC
    target_text = "the cat"
    blank = encoder.char2ind[encoder.EMPTY_TOK]

    token_ids = [blank]

    for char in target_text:
        token_ids.append(encoder.char2ind[char])
        token_ids.append(blank)

    # Матрица [T, vocab_size]
    logits = torch.full(
        (len(token_ids), len(encoder)),
        -8.0,
    )

    for t, token_id in enumerate(token_ids):
        logits[t, token_id] = 8.0

    log_probs = torch.log_softmax(logits, dim=-1)

    # 5. Сравниваем декодеры
    greedy = encoder.ctc_decode(
        log_probs.argmax(dim=-1).tolist()
    )

    beam = encoder.ctc_beam_search(
        log_probs,
        beam_size=25,
    )

    beam_lm = encoder.ctc_decode_lm(
        log_probs,
        beam_size=25,
    )

    print("\nTarget: ", target_text)
    print("Greedy: ", greedy)
    print("Beam:   ", beam)
    print("Beam+LM:", beam_lm)

    assert greedy == target_text
    assert beam == target_text
    assert beam_lm == target_text

    print("\nAll LM decoder smoke tests passed!")


if __name__ == "__main__":
    main()
