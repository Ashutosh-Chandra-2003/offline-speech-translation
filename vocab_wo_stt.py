import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from wordfreq import zipf_frequency
from difflib import get_close_matches

# -------------------------
# REQUIRED NLTK DATA
# -------------------------
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# -------------------------
# AUX VERBS (DO NOT TOUCH)
# -------------------------
AUX_VERBS = {
    "am", "is", "are", "was", "were",
    "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "will", "would", "shall", "should",
    "may", "might", "must", "can", "could"
}

# -------------------------
# MANUAL SIMPLIFICATION MAP
# -------------------------
SIMPLE_MAP = {
    "lethargic": "lazy",
    "fatigued": "tired",
    "commence": "start",
    "terminate": "end",
    "utilize": "use",
    "assist": "help",
    "purchase": "buy",
    "reside": "live",
}

# -------------------------
# POS CONVERSION
# -------------------------
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wn.ADJ
    if treebank_tag.startswith("V"):
        return wn.VERB
    if treebank_tag.startswith("N"):
        return wn.NOUN
    if treebank_tag.startswith("R"):
        return wn.ADV
    return None

# -------------------------
# OFFLINE AUTOCORRECT
# -------------------------
def autocorrect(word):
    matches = get_close_matches(word, wn.words(), n=1, cutoff=0.85)
    return matches[0] if matches else word

# -------------------------
# FIND SIMPLER WORD
# -------------------------
def get_simpler_word(word, pos=None):
    word = word.lower()

    # Hard override
    if word in SIMPLE_MAP:
        return SIMPLE_MAP[word]

    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return word

    candidates = set()

    for syn in synsets:
        for lemma in syn.lemmas():
            candidates.add(lemma.name().replace("_", " "))
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                candidates.add(lemma.name().replace("_", " "))

    if not candidates:
        return word

    best = max(candidates, key=lambda w: zipf_frequency(w, "en"))

    if zipf_frequency(best, "en") > zipf_frequency(word, "en"):
        return best

    return word

# -------------------------
# TEXT SIMPLIFIER
# -------------------------
def simplify_text(text):
    tokens = text.split()
    tagged = pos_tag(tokens)
    output = []

    for token, tag in tagged:
        clean = token.strip(".,?!").lower()

        # Do not touch auxiliary verbs
        if clean in AUX_VERBS:
            output.append(token)
            continue

        # Fix spelling first
        corrected = autocorrect(clean)

        wn_pos = get_wordnet_pos(tag)
        simple = get_simpler_word(corrected, wn_pos)

        if simple != clean:
            output.append(token.replace(clean, simple))
        else:
            output.append(token)

    return " ".join(output)

# -------------------------
# INTERACTIVE LOOP
# -------------------------
print("\n--- ✅ AUTO VOCAB SIMPLIFIER (STABLE & OFFLINE) ---")

current = input("\nEnter text:\n> ")

while True:
    cmd = input("\nType 'simplify' or 'understood':\n> ").lower()

    if cmd == "simplify":
        current = simplify_text(current)
        print("\n--- SIMPLIFIED ---")
        print(current)

    elif cmd == "understood":
        print("\n✅ Done.")
        break

    else:
        print("\n❌ Unknown command. Type 'simplify' or 'understood'")
