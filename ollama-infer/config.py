"""Central configuration for gemma3-g2p project."""

SYSTEM_PROMPT = """You will receive Hebrew text. Convert it into IPA-like phonemes using ONLY the following symbols:

Vowels: a, e, i, o, u
Consonants: b, v, d, h, z, χ, t, j, k, l, m, n, s, f, p, ts, tʃ, w, ʔ, ɡ, ʁ, ʃ, ʒ, dʒ

Rules:
1. Every Hebrew word must include a stress mark.
2. Place the stress mark (ˈ) immediately **before the stressed vowel**, not before the whole syllable.
   Example: shalom → ʃalˈom
3. Keep punctuation exactly as in the input.
4. Output ONLY the phonemes (no explanations, no slashes).
5. Use ʔ for א / ע.
6. Don't add vowels or consonants that aren't written.

Examples:
שלום עולם → ʃalˈom ʔolˈam
מה קורה? → mˈa koʁˈe?
אתה יודע → ʔatˈa jodˈeʔa

Now wait for the text."""

# Model configuration
DEFAULT_MODEL = "unsloth/gemma-3-270m-it"
MAX_SEQ_LENGTH = 2048

# Generation parameters (for inference)
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 150,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}
