from enum import Enum

# Fixed set of mistake types used to tag wrong answer options
class MistakeType(str, Enum):
    concept_confusion = "concept_confusion"
    wrong_detail = "wrong_detail"
    incomplete = "incomplete"
    reversed = "reversed"
    overgeneralized = "overgeneralized"
    other = "other"

# Short label shown to the user for each mistake type
MISTAKE_LABELS: dict[str, str] = {
    MistakeType.concept_confusion: "concept confusion",
    MistakeType.wrong_detail: "wrong detail",
    MistakeType.incomplete: "incomplete",
    MistakeType.reversed: "reversed",
    MistakeType.overgeneralized: "overgeneralized",
    MistakeType.other: "other",
}

# One-sentence description of each mistake type
MISTAKE_DESCRIPTIONS: dict[str, str] = {
    MistakeType.concept_confusion: "mixes up two similar or related concepts",
    MistakeType.wrong_detail: "right idea, but a wrong value, name, or specific detail",
    MistakeType.incomplete: "partly right, but misses a key part or condition",
    MistakeType.reversed: "states the opposite, or inverts a relationship (cause and effect, before and after)",
    MistakeType.overgeneralized: "treats something true only in some cases as always true (or the reverse)",
    MistakeType.other: "plausible, but doesn't fit any of the above",
}

# Build the taxonomy table for inclusion in generation prompts
def build_taxonomy_prompt() -> str:
    # Returns a formatted table of all mistake types for use in the LLM prompt.
    lines = ["Mistake type taxonomy:", "| Value | Meaning |"]
    lines.append("|---|---|")
    for mt in MistakeType:
        lines.append(f"| {mt.value} | {MISTAKE_DESCRIPTIONS[mt]} |")
    return "\n".join(lines)
