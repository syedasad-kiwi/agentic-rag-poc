import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from typing import List, Optional

from llama_index.core import SimpleDirectoryReader
from docling.document_converter import DocumentConverter


class DoclingIngestion:
    """
    A class to handle document ingestion using Docling and LlamaIndex.
    Converts documents to Markdown format with optimized processing.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the ingestion processor.
        
        Args:
            input_dir: Path to directory containing input documents
            output_dir: Path to directory for processed outputs
        """
        self.input_dir = Path(input_dir)
        self.output_root = Path(output_dir)
        self.md_dir = self.output_root / "md"
        
        # Create directories once
        self.md_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize converter
        self.converter = DocumentConverter()
        
        # Path keys for metadata extraction
        self.path_keys = ("file_path", "filepath", "path", "source", "filename", "file_name")
    
    def _extract_source_path(self, doc) -> Optional[Path]:
        """Best-effort: pull original file path from LlamaIndex metadata."""
        meta = getattr(doc, "metadata", {}) or {}
        
        for key in self.path_keys:
            value = meta.get(key)
            if value:
                path_obj = Path(value)
                if not path_obj.is_absolute():
                    path_obj = self.input_dir / value
                if path_obj.exists():
                    return path_obj.resolve()
        
        # Fallback: doc_id as path (rare) - optimized exception handling
        doc_id = getattr(doc, "doc_id", "")
        if doc_id:
            try:
                path_obj = Path(doc_id)
                if path_obj.exists():
                    return path_obj.resolve()
            except (OSError, ValueError):
                pass
        return None

    def _gather_candidate_paths(self, documents: List) -> List[Path]:
        """Gather candidate paths from LlamaIndex docs with fallback to directory listing."""
        # Use list comprehension for efficiency
        candidate_paths = [
            path for doc in documents 
            for path in [self._extract_source_path(doc)]
            if path and path.is_file()
        ]

        # If nothing came from LI metadata, fallback to directory listing
        if not candidate_paths:
            candidate_paths = [path.resolve() for path in self.input_dir.iterdir() if path.is_file()]
        
        return candidate_paths
    
    def _deduplicate_paths(self, candidate_paths: List[Path]) -> List[Path]:
        """De-duplicate while preserving order - optimized with early exit and single loop."""
        seen = set()
        unique_paths = []
        dupe_counts = Counter()

        for path in candidate_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
            else:
                dupe_counts[path.name] += 1

        if dupe_counts:
            total_dupes = sum(dupe_counts.values())
            print(f"ℹ️ De-duplicated {total_dupes} extra references across {len(dupe_counts)} files.")
        
        return unique_paths
    
    def _convert_document(self, src_path: Path) -> bool:
        """Convert a single document to Markdown format."""
        base_name = src_path.stem
        out_md = self.md_dir / f"{base_name}.md"
        
        try:
            # Convert path to string once
            src_path_str = str(src_path)
            conv_res = self.converter.convert(src_path_str)
            dl_doc = conv_res.document

            # Export to markdown
            doc_markdown = dl_doc.export_to_markdown()
            
            # Write markdown file
            out_md.write_text(doc_markdown, encoding="utf-8")
            
            print(f"✅ Saved: {out_md}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error processing {src_path.name}: {e}")
            return False
    
    def process_documents(self) -> None:
        """Main processing method to convert all documents."""
        # Load documents using LlamaIndex
        reader = SimpleDirectoryReader(input_dir=str(self.input_dir))
        documents = reader.load_data()
        
        # Gather and deduplicate paths
        candidate_paths = self._gather_candidate_paths(documents)
        unique_paths = self._deduplicate_paths(candidate_paths)
        
        # Convert each unique file exactly once
        total_files = len(unique_paths)
        successful_conversions = 0
        
        for src_path in tqdm(unique_paths, desc="Converting", unit="file", total=total_files):
            if self._convert_document(src_path):
                successful_conversions += 1
        
        print(f"\n🎉 Processing complete! Successfully converted {successful_conversions}/{total_files} files.")


def main():
    """Main function to run the document ingestion process."""
    # Default paths
    input_dir = "/Users/kiwitech/Documents/agentic-rag-poc/data/raw"
    output_dir = "/Users/kiwitech/Documents/agentic-rag-poc/data/processed"
    
    # Create and run ingestion processor
    processor = DoclingIngestion(input_dir, output_dir)
    processor.process_documents()


if __name__ == "__main__":
    main()