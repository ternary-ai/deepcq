from deepcq.schema import MemorySchema


def test_extraction_pattern_finds_section():
    schema = MemorySchema(sections=["errors", "metrics"])
    text = "[errors]\n- OOM at 02:15\n- Timeout\n[metrics]\nCPU: 78%\n"
    m = schema.extraction_pattern("errors").search(text)
    assert m is not None
    assert "OOM" in m.group(0)
    assert "CPU" not in m.group(0)


def test_extraction_pattern_last_section():
    schema = MemorySchema(sections=["errors", "metrics"])
    text = "[errors]\n- OOM\n[metrics]\nCPU: 78%\nMemory: 4GB\n"
    m = schema.extraction_pattern("metrics").search(text)
    assert m is not None
    assert "CPU" in m.group(0)
    assert "Memory" in m.group(0)


def test_custom_header_format():
    schema = MemorySchema(
        sections=["status", "logs"],
        section_header_format="## {name}",
    )
    text = "## status\nOK\n## logs\nERROR: timeout\n"
    m = schema.extraction_pattern("logs").search(text)
    assert m is not None
    assert "timeout" in m.group(0)


def test_schema_id_stable():
    s1 = MemorySchema(sections=["a", "b"])
    s2 = MemorySchema(sections=["a", "b"])
    assert s1.schema_id() == s2.schema_id()


def test_optional_keywords_accepted():
    schema = MemorySchema(
        sections=["db"],
        keywords={"db": ["database", "postgres"]},
    )
    assert "db" in schema.keywords
