from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    # Initialize the splitter with size limits and separator priority order
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # Maximum characters per chunk
        chunk_overlap=chunk_overlap,  # Characters shared between consecutive chunks to preserve context
        separators=["\n\n", "\n", ".", " ", ""],  # Try splitting by paragraphs first, then lines, sentences, words, chars
    )
    # Split the raw text and return the resulting list of string chunks
    return splitter.split_text(text)