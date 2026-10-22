from recall_loop.mistakes import MistakeType, MISTAKE_LABELS, MISTAKE_DESCRIPTIONS, build_taxonomy_prompt

def test_all_types_have_labels():
    for mt in MistakeType:
        assert mt in MISTAKE_LABELS, f"Missing label for {mt}"

def test_all_types_have_descriptions():
    for mt in MistakeType:
        assert mt in MISTAKE_DESCRIPTIONS, f"Missing description for {mt}"

def test_taxonomy_prompt_mentions_all_values():
    prompt = build_taxonomy_prompt()
    for mt in MistakeType:
        assert mt.value in prompt, f"Taxonomy prompt missing {mt.value}"

def test_labels_are_strings():
    for mt, label in MISTAKE_LABELS.items():
        assert isinstance(label, str) and len(label) > 0

def test_descriptions_are_strings():
    for mt, desc in MISTAKE_DESCRIPTIONS.items():
        assert isinstance(desc, str) and len(desc) > 0
