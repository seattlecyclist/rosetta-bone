import io


from rosetta_bone.common.pdf import pdf_to_text


def _make_minimal_pdf() -> bytes:
    """Build a 1-page PDF containing the text 'hello dog' using reportlab."""
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        import pytest
        pytest.skip("reportlab not installed")
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, "hello dog")
    c.save()
    return buf.getvalue()


def test_pdf_to_text(tmp_path):
    pdf = _make_minimal_pdf()
    p = tmp_path / "x.pdf"
    p.write_bytes(pdf)
    text = pdf_to_text(p)
    assert "hello dog" in text
