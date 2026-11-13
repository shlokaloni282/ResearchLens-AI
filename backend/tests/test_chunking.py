from backend.app.services.chunking import (
    fixed_size_chunking,
    recursive_chunking,
    semantic_chunking
)


sample_text = """
Artificial intelligence is transforming healthcare.
Machine learning helps doctors analyze medical images.
Deep learning models improve disease detection.

Climate change affects global temperatures.
Rising temperatures contribute to glacier melting.
Carbon emissions accelerate global warming.
"""
def test_recursive_chunking():
    chunks = recursive_chunking(
    sample_text,
    chunk_size=100,
    overlap=20
)

    assert chunks is not None
    assert len(chunks) > 0
    assert all(len(chunk) <= 100 for chunk in chunks)

    print("\nRecursive chunks:")
    for chunk in chunks:
        print(chunk)
        print("---")
def test_fixed_chunking():

    chunks = fixed_size_chunking(
        sample_text,
        chunk_size=100,
        overlap=20
    )

    assert chunks is not None
    assert len(chunks) > 0

    print("\nFixed-size chunks:")
    for chunk in chunks:
        print(chunk)


def test_semantic_chunking():

    chunks = semantic_chunking(sample_text)

    assert chunks is not None
    assert len(chunks) > 0

    print("\nSemantic chunks:")
    for chunk in chunks:
        print(chunk)