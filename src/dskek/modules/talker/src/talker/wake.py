"""Voice markers: wake on the bot's name, end on a phrase, silence on "stop".

Transcripts are error-prone, so every comparison is fuzzy (``difflib`` ratio) and the
name may be given with aliases for common mishearings.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

from .config import WakeConfig

_FILLERS = frozenset({"hey", "hi", "hello", "ok", "okay", "yo", "oh", "so", "um", "uh", "well"})
_POLITE = frozenset({"please", "thanks", "thank", "you", "ok", "okay", "um", "uh", "so", "well", "and"})
_PUNCT_EDGE = " \t\r\n,.:;!?-—–…\"'"


class WakeAction(StrEnum):
    NONE = "none"
    WAKE = "wake"
    """The bot's name was said (``remainder`` is what followed it)."""
    END = "end"
    """An end-of-conversation phrase."""
    SILENCE = "silence"
    """A bare 'stop' / 'be quiet': silence the bot without asking the LLM."""


@dataclass(frozen=True, slots=True)
class WakeResult:
    action: WakeAction
    remainder: str = ""


@dataclass(frozen=True, slots=True)
class _Word:
    raw: str
    norm: str


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower()).strip("'")


def _tokens(phrase: str) -> tuple[str, ...]:
    return tuple(t for t in (_norm(w) for w in phrase.split()) if t)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class WakeWordMatcher:
    def __init__(self, config: WakeConfig) -> None:
        self._cfg = config
        self._names = [t for t in (_tokens(n) for n in config.names) if t]
        self._end = [t for t in (_tokens(p) for p in config.end_phrases) if t]
        self._silence = [t for t in (_tokens(p) for p in config.silence_phrases) if t]

    # ---- public -----------------------------------------------------------------------
    def classify(self, text: str, *, active: bool) -> WakeResult:
        words = self._split(text)
        if not words:
            return WakeResult(WakeAction.NONE)
        span = self._find_name(words, self._cfg.fuzzy_threshold, self._cfg.max_leading_words)
        core = self._core_tokens(words)

        if core and self._phrase_match(core, self._end, min_ratio=0.85, slack=2):
            return WakeResult(WakeAction.END if active else WakeAction.NONE)
        if core and self._phrase_match(core, self._silence, min_ratio=0.9, slack=1):
            return WakeResult(WakeAction.SILENCE)
        if span is not None:
            return WakeResult(WakeAction.WAKE, self._join(words[span[1] :]))
        return WakeResult(WakeAction.NONE, self._join(words))

    def might_contain_wake(self, text: str) -> bool:
        """Cheap, forgiving check used to skip the expensive second pass on idle chatter."""
        words = self._split(text)
        relaxed = max(0.5, self._cfg.fuzzy_threshold - 0.2)
        return self._find_name(words, relaxed, self._cfg.max_leading_words + 2) is not None

    # ---- internals --------------------------------------------------------------------
    @staticmethod
    def _split(text: str) -> list[_Word]:
        words = [_Word(raw=w, norm=_norm(w)) for w in text.split()]
        return [w for w in words if w.norm]

    @staticmethod
    def _join(words: list[_Word]) -> str:
        return " ".join(w.raw for w in words).lstrip(_PUNCT_EDGE)

    def _name_matches_at(self, words: list[_Word], start: int, threshold: float) -> int | None:
        """Return the end index of a name match starting at ``start`` (or None)."""
        for name in self._names:
            end = start + len(name)
            if end > len(words):
                continue
            candidate = "".join(w.norm for w in words[start:end])
            target = "".join(name)
            if len(target) < 4:  # too short to fuzz: exact only
                if candidate == target:
                    return end
                continue
            # Short names allow substitutions only ("noba" ~ "nova" but not "now"); longer ones may also lose/gain a letter.
            max_len_diff = 0 if len(target) < 5 else 1
            if abs(len(candidate) - len(target)) <= max_len_diff and _ratio(candidate, target) >= threshold:
                return end
        return None

    def _find_name(self, words: list[_Word], threshold: float, leading: int) -> tuple[int, int] | None:
        for start in range(min(len(words), leading + 1)):
            if (end := self._name_matches_at(words, start, threshold)) is not None:
                return start, end
        return None

    def _core_tokens(self, words: list[_Word]) -> list[str]:
        """Words left after removing the name (anywhere), greetings and politeness."""
        drop: set[int] = set()
        for i in range(len(words)):
            if (end := self._name_matches_at(words, i, self._cfg.fuzzy_threshold)) is not None:
                drop.update(range(i, end))
        return [
            w.norm
            for i, w in enumerate(words)
            if i not in drop and w.norm not in _FILLERS and w.norm not in _POLITE
        ]

    @staticmethod
    def _phrase_match(core: list[str], phrases: list[tuple[str, ...]], *, min_ratio: float, slack: int) -> bool:
        text = " ".join(core)
        for phrase in phrases:
            target = " ".join(phrase)
            if _ratio(text, target) >= min_ratio:
                return True
            if len(core) <= len(phrase) + slack and any(
                core[i : i + len(phrase)] == list(phrase) for i in range(len(core) - len(phrase) + 1)
            ):
                return True
        return False
