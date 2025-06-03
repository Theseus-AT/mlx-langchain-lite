"""
MLX Document Processor
Intelligente Verarbeitung von PDFs, Markdown und anderen Dokumenten
Optimiert für Brain System Integration und MLX Components
"""

import asyncio
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re
import aiofiles

# Document processing libraries
import fitz  # PyMuPDF für PDF processing
import markdown
from bs4 import BeautifulSoup
import mammoth  # für .docx
import pandas as pd

# MLX Components Integration
from mlx_components.embedding_engine import MLXEmbeddingEngine, EmbeddingConfig
from mlx_components.vector_store import MLXVectorStore, VectorStoreConfig

@dataclass
class ProcessingConfig:
    """Konfiguration für Document Processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 4000
    preserve_structure: bool = True
    extract_metadata: bool = True
    auto_summarize: bool = True
    embedding_model: str = "mlx-community/gte-small"
    supported_formats: List[str] = None
    output_format: str = "markdown"  # markdown, json, structured

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                ".pdf", ".md", ".txt", ".docx", ".doc", 
                ".html", ".rtf", ".csv", ".json"
            ]

@dataclass
class DocumentChunk:
    """Einzelner Document Chunk"""
    id: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None

@dataclass
class ProcessedDocument:
    """Vollständig verarbeitetes Dokument"""
    document_id: str
    title: str
    content: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    processing_time: float
    word_count: int
    chunk_count: int
    file_path: Optional[str] = None
    summary: Optional[str] = None

class MLXDocumentProcessor:
    """
    High-Performance Document Processor für MLX Ecosystem
    
    Features:
    - Multi-Format Support (PDF, Markdown, DOCX, etc.)
    - Intelligent Chunking mit Structure Preservation
    - Automatic Metadata Extraction
    - MLX Integration für Embeddings
    - Brain System Ready
    - Batch Processing
    - Progress Tracking
    - Error Recovery
    """
    
    def __init__(self, 
                 config: ProcessingConfig = None,
                 embedding_engine: MLXEmbeddingEngine = None):
        self.config = config or ProcessingConfig()
        self.embedding_engine = embedding_engine
        
        # Performance Metrics
        self.total_documents = 0
        self.total_chunks = 0
        self.total_processing_time = 0.0
        self.format_stats = {}
        
        # Processing cache
        self.document_cache = {}
        self.chunk_cache = {}
        
        # Format-specific processors
        self.processors = {
            ".pdf": self._process_pdf,
            ".md": self._process_markdown,
            ".txt": self._process_text,
            ".docx": self._process_docx,
            ".doc": self._process_docx,
            ".html": self._process_html,
            ".csv": self._process_csv,
            ".json": self._process_json
        }
    
    async def initialize(self) -> None:
        """
        Initialisiert Embedding Engine falls nicht übergeben
        """
        if self.embedding_engine is None:
            embedding_config = EmbeddingConfig(
                model_path=self.config.embedding_model,
                batch_size=32,
                cache_embeddings=True
            )
            self.embedding_engine = MLXEmbeddingEngine(embedding_config)
            await self.embedding_engine.initialize()
    
    async def process_document(self, 
                             file_path: Union[str, Path],
                             user_id: Optional[str] = None,
                             custom_metadata: Optional[Dict[str, Any]] = None) -> ProcessedDocument:
        """
        Hauptfunktion: Verarbeitet einzelnes Dokument
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        await self.initialize()
        
        # Check if supported format
        if file_path.suffix.lower() not in self.config.supported_formats:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        # Generate document ID
        document_id = self._generate_document_id(file_path)
        
        # Check cache
        if document_id in self.document_cache:
            print(f"📄 Document found in cache: {file_path.name}")
            return self.document_cache[document_id]
        
        print(f"📄 Processing document: {file_path.name}")
        
        try:
            # Extract content using appropriate processor
            processor = self.processors.get(file_path.suffix.lower())
            if not processor:
                raise ValueError(f"No processor for format: {file_path.suffix}")
            
            content, extracted_metadata = await processor(file_path)
            
            # Merge metadata
            metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "file_type": file_path.suffix.lower(),
                "processed_at": datetime.now().isoformat(),
                "user_id": user_id,
                **extracted_metadata
            }
            
            if custom_metadata:
                metadata.update(custom_metadata)
            
            # Create chunks
            chunks = await self._create_chunks(content, document_id, metadata)
            
            # Generate embeddings for chunks
            if chunks:
                await self._generate_chunk_embeddings(chunks)
            
            # Generate document summary
            summary = await self._generate_summary(content, chunks) if self.config.auto_summarize else None
            
            # Create processed document
            processed_doc = ProcessedDocument(
                document_id=document_id,
                title=metadata.get("title", file_path.stem),
                content=content,
                chunks=chunks,
                metadata=metadata,
                processing_time=time.time() - start_time,
                word_count=len(content.split()),
                chunk_count=len(chunks),
                file_path=str(file_path),
                summary=summary
            )
            
            # Update cache and metrics
            self.document_cache[document_id] = processed_doc
            self._update_metrics(processed_doc)
            
            print(f"✅ Processed {file_path.name}: {len(chunks)} chunks, {processed_doc.word_count} words")
            
            return processed_doc
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            raise
    
    async def process_directory(self, 
                              directory_path: Union[str, Path],
                              user_id: Optional[str] = None,
                              recursive: bool = True,
                              pattern: str = "*") -> List[ProcessedDocument]:
        """
        Verarbeitet alle Dokumente in einem Verzeichnis
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        # Find all supported files
        files = []
        
        if recursive:
            for suffix in self.config.supported_formats:
                files.extend(directory_path.rglob(f"*{suffix}"))
        else:
            for suffix in self.config.supported_formats:
                files.extend(directory_path.glob(f"*{suffix}"))
        
        print(f"📁 Found {len(files)} documents in {directory_path}")
        
        # Process files in batches
        processed_docs = []
        batch_size = 5  # Process 5 documents at a time
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.process_document(file_path, user_id)
                for file_path in batch_files
            ]
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        print(f"❌ Failed to process {batch_files[j]}: {result}")
                    else:
                        processed_docs.append(result)
                        
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
        
        print(f"✅ Successfully processed {len(processed_docs)}/{len(files)} documents")
        
        return processed_docs
    
    async def process_text_directly(self, 
                                  text: str,
                                  title: str = "Direct Text",
                                  user_id: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> ProcessedDocument:
        """
        Verarbeitet Text direkt ohne Datei
        """
        start_time = time.time()
        
        await self.initialize()
        
        document_id = hashlib.md5(f"{title}:{text[:100]}".encode()).hexdigest()
        
        # Prepare metadata
        doc_metadata = {
            "title": title,
            "source": "direct_text",
            "processed_at": datetime.now().isoformat(),
            "user_id": user_id,
            "content_length": len(text),
            "word_count": len(text.split())
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        # Create chunks
        chunks = await self._create_chunks(text, document_id, doc_metadata)
        
        # Generate embeddings
        if chunks:
            await self._generate_chunk_embeddings(chunks)
        
        # Generate summary
        summary = await self._generate_summary(text, chunks) if self.config.auto_summarize else None
        
        processed_doc = ProcessedDocument(
            document_id=document_id,
            title=title,
            content=text,
            chunks=chunks,
            metadata=doc_metadata,
            processing_time=time.time() - start_time,
            word_count=len(text.split()),
            chunk_count=len(chunks),
            summary=summary
        )
        
        self._update_metrics(processed_doc)
        
        return processed_doc
    
    async def _process_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        PDF Processing mit PyMuPDF
        """
        try:
            doc = fitz.open(str(file_path))
            
            # Extract text and metadata
            text_content = []
            metadata = {
                "page_count": doc.page_count,
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", "")
            }
            
            # Extract text page by page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():
                    if self.config.preserve_structure:
                        text_content.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                    else:
                        text_content.append(page_text)
            
            doc.close()
            
            full_text = "\n".join(text_content)
            
            return full_text, metadata
            
        except Exception as e:
            raise Exception(f"PDF processing error: {e}")
    
    async def _process_markdown(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Markdown Processing
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse markdown
            md = markdown.Markdown(extensions=['meta', 'toc', 'tables'])
            html = md.convert(content)
            
            # Extract metadata from markdown meta
            metadata = {
                "title": "",
                "headers": []
            }
            
            if hasattr(md, 'Meta') and md.Meta:
                for key, value in md.Meta.items():
                    metadata[key] = value[0] if isinstance(value, list) and len(value) == 1 else value
            
            # Extract headers
            headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            metadata["headers"] = [{"level": len(h[0]), "text": h[1]} for h in headers]
            
            # Use title from first header if not in metadata
            if not metadata.get("title") and headers:
                metadata["title"] = headers[0][1]
            
            return content, metadata
            
        except Exception as e:
            raise Exception(f"Markdown processing error: {e}")
    
    async def _process_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Plain Text Processing
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Basic metadata
            metadata = {
                "title": file_path.stem,
                "encoding": "utf-8",
                "line_count": len(content.split('\n'))
            }
            
            return content, metadata
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'ascii']:
                try:
                    async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                        content = await f.read()
                    
                    metadata = {
                        "title": file_path.stem,
                        "encoding": encoding,
                        "line_count": len(content.split('\n'))
                    }
                    
                    return content, metadata
                except:
                    continue
            
            raise Exception("Could not decode text file with any encoding")
    
    async def _process_docx(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        DOCX Processing mit mammoth
        """
        try:
            with open(file_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                content = result.value
                
                # Extract basic metadata
                metadata = {
                    "title": file_path.stem,
                    "warnings": result.messages,
                    "extraction_method": "mammoth"
                }
                
                return content, metadata
                
        except Exception as e:
            raise Exception(f"DOCX processing error: {e}")
    
    async def _process_html(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        HTML Processing mit BeautifulSoup
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                html_content = await f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            metadata = {
                "title": "",
                "description": "",
                "keywords": []
            }
            
            # Get title
            title_tag = soup.find('title')
            if title_tag:
                metadata["title"] = title_tag.get_text().strip()
            
            # Get meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata["description"] = meta_desc.get('content', '')
            
            # Get meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                metadata["keywords"] = [k.strip() for k in meta_keywords.get('content', '').split(',')]
            
            # Extract text content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text, metadata
            
        except Exception as e:
            raise Exception(f"HTML processing error: {e}")
    
    async def _process_csv(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        CSV Processing
        """
        try:
            df = pd.read_csv(file_path)
            
            # Convert to readable text format
            text_parts = [f"Dataset: {file_path.stem}"]
            text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_parts.append(f"Rows: {len(df)}")
            text_parts.append("\nData Summary:")
            text_parts.append(df.describe(include='all').to_string())
            
            # Add sample data
            text_parts.append(f"\nSample Data (first 5 rows):")
            text_parts.append(df.head().to_string())
            
            content = "\n".join(text_parts)
            
            metadata = {
                "title": file_path.stem,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "file_type": "csv"
            }
            
            return content, metadata
            
        except Exception as e:
            raise Exception(f"CSV processing error: {e}")
    
    async def _process_json(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        JSON Processing
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                json_content = await f.read()
            
            data = json.loads(json_content)
            
            # Convert JSON to readable text
            if isinstance(data, dict):
                text_parts = [f"JSON Document: {file_path.stem}"]
                for key, value in data.items():
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")
                    else:
                        text_parts.append(f"{key}: {type(value).__name__} with {len(value) if hasattr(value, '__len__') else '?'} items")
            else:
                text_parts = [f"JSON Array: {file_path.stem}"]
                text_parts.append(f"Items: {len(data) if hasattr(data, '__len__') else 'Unknown'}")
            
            # Add formatted JSON sample
            text_parts.append("\nFormatted Content:")
            text_parts.append(json.dumps(data, indent=2)[:2000])  # Limit to 2000 chars
            
            content = "\n".join(text_parts)
            
            metadata = {
                "title": file_path.stem,
                "json_type": type(data).__name__,
                "item_count": len(data) if hasattr(data, '__len__') else 0,
                "file_type": "json"
            }
            
            return content, metadata
            
        except Exception as e:
            raise Exception(f"JSON processing error: {e}")
    
    async def _create_chunks(self, 
                           content: str, 
                           document_id: str, 
                           metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Intelligente Content-Chunking
        """
        chunks = []
        
        if len(content) <= self.config.min_chunk_size:
            # Document too small, create single chunk
            chunk = DocumentChunk(
                id=f"{document_id}_chunk_0",
                content=content,
                metadata=metadata.copy(),
                chunk_index=0,
                start_char=0,
                end_char=len(content)
            )
            chunks.append(chunk)
            return chunks
        
        # Split content intelligently
        if self.config.preserve_structure:
            # Structure-aware chunking
            chunks = await self._structure_aware_chunking(content, document_id, metadata)
        else:
            # Simple sliding window chunking
            chunks = await self._sliding_window_chunking(content, document_id, metadata)
        
        return chunks
    
    async def _structure_aware_chunking(self, 
                                      content: str, 
                                      document_id: str, 
                                      metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Structure-aware chunking basierend auf Paragraphen und Sections
        """
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        start_char = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) > self.config.chunk_size and current_chunk:
                # Create chunk with current content
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_index,
                    "chunk_type": "structure_aware"
                })
                
                chunk = DocumentChunk(
                    id=f"{document_id}_chunk_{chunk_index}",
                    content=current_chunk,
                    metadata=chunk_metadata,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.config.chunk_overlap:] if len(current_chunk) > self.config.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + paragraph
                start_char = start_char + len(current_chunk) - len(overlap_text) - len(paragraph) - 2
                chunk_index += 1
                
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": chunk_index,
                "chunk_type": "structure_aware"
            })
            
            chunk = DocumentChunk(
                id=f"{document_id}_chunk_{chunk_index}",
                content=current_chunk,
                metadata=chunk_metadata,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _sliding_window_chunking(self, 
                                     content: str, 
                                     document_id: str, 
                                     metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Sliding window chunking für gleichmäßige Chunk-Größen
        """
        chunks = []
        chunk_index = 0
        
        for start in range(0, len(content), self.config.chunk_size - self.config.chunk_overlap):
            end = min(start + self.config.chunk_size, len(content))
            chunk_content = content[start:end]
            
            if len(chunk_content) < self.config.min_chunk_size and chunk_index > 0:
                # Too small, merge with previous chunk
                if chunks:
                    chunks[-1].content += "\n" + chunk_content
                    chunks[-1].end_char = end
                break
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": chunk_index,
                "chunk_type": "sliding_window"
            })
            
            chunk = DocumentChunk(
                id=f"{document_id}_chunk_{chunk_index}",
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end
            )
            chunks.append(chunk)
            chunk_index += 1
            
            if end >= len(content):
                break
        
        return chunks
    
    async def _generate_chunk_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """
        Generiert Embeddings für alle Chunks
        """
        if not chunks or not self.embedding_engine:
            return
        
        try:
            # Extract content for embedding
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings in batch
            embedding_result = await self.embedding_engine.embed(texts)
            
            # Assign embeddings to chunks
            for i, chunk in enumerate(chunks):
                if i < len(embedding_result.embeddings):
                    chunk.embedding = embedding_result.embeddings[i].tolist()
            
        except Exception as e:
            print(f"⚠️ Error generating chunk embeddings: {e}")
    
    async def _generate_summary(self, 
                              content: str, 
                              chunks: List[DocumentChunk]) -> Optional[str]:
        """
        Generiert automatische Zusammenfassung
        """
        try:
            # Simple extractive summary
            sentences = content.split('.')
            
            # Take first few sentences and some from middle/end
            summary_sentences = []
            
            if len(sentences) > 0:
                summary_sentences.append(sentences[0])
            
            if len(sentences) > 5:
                mid_index = len(sentences) // 2
                summary_sentences.append(sentences[mid_index])
            
            if len(sentences) > 10:
                summary_sentences.append(sentences[-2])
            
            summary = '. '.join(s.strip() for s in summary_sentences if s.strip()) + '.'
            
            # Limit summary length
            if len(summary) > 500:
                summary = summary[:497] + "..."
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Error generating summary: {e}")
            return None
    
    def _generate_document_id(self, file_path: Path) -> str:
        """
        Generiert eindeutige Document ID
        """
        # Use file path and modification time for ID
        try:
            stat = file_path.stat()
            content = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        except:
            content = str(file_path)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _update_metrics(self, processed_doc: ProcessedDocument) -> None:
        """
        Aktualisiert Performance-Metriken
        """
        self.total_documents += 1
        self.total_chunks += processed_doc.chunk_count
        self.total_processing_time += processed_doc.processing_time
        
        file_type = processed_doc.metadata.get("file_type", "unknown")
        if file_type not in self.format_stats:
            self.format_stats[file_type] = {"count": 0, "total_time": 0.0}
        
        self.format_stats[file_type]["count"] += 1
        self.format_stats[file_type]["total_time"] += processed_doc.processing_time
    
    async def save_to_vector_store(self, 
                                 processed_doc: ProcessedDocument,
                                 vector_store: MLXVectorStore,
                                 user_id: str,
                                 model_id: str = "gte-small") -> bool:
        """
        Speichert verarbeitetes Dokument in Vector Store
        """
        try:
            # Prepare vectors and metadata for chunks with embeddings
            vectors = []
            metadata_list = []
            
            for chunk in processed_doc.chunks:
                if chunk.embedding:
                    vectors.append(chunk.embedding)
                    
                    chunk_metadata = chunk.metadata.copy()
                    chunk_metadata.update({
                        "document_id": processed_doc.document_id,
                        "document_title": processed_doc.title,
                        "chunk_id": chunk.id,
                        "chunk_content": chunk.content,
                        "text": chunk.content,  # For compatibility
                        "summary": chunk.summary or processed_doc.summary
                    })
                    metadata_list.append(chunk_metadata)
            
            if not vectors:
                print(f"⚠️ No embeddings found for document {processed_doc.title}")
                return False
            
            # Add to vector store
            success = await vector_store.add_vectors(
                user_id=user_id,
                model_id=model_id,
                vectors=vectors,
                metadata=metadata_list,
                namespace=f"document_{processed_doc.document_id}"
            )
            
            if success:
                print(f"✅ Saved {len(vectors)} chunks to vector store for {processed_doc.title}")
            else:
                print(f"❌ Failed to save chunks to vector store for {processed_doc.title}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error saving to vector store: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Liefert Performance-Statistiken
        """
        avg_processing_time = self.total_processing_time / self.total_documents if self.total_documents > 0 else 0
        avg_chunks_per_doc = self.total_chunks / self.total_documents if self.total_documents > 0 else 0
        
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "average_chunks_per_document": avg_chunks_per_doc,
            "documents_per_second": self.total_documents / self.total_processing_time if self.total_processing_time > 0 else 0,
            "chunks_per_second": self.total_chunks / self.total_processing_time if self.total_processing_time > 0 else 0,
            "format_statistics": self.format_stats,
            "supported_formats": self.config.supported_formats,
            "cache_size": len(self.document_cache)
        }
    
    def clear_cache(self) -> None:
        """
        Leert Document und Chunk Cache
        """
        self.document_cache.clear()
        self.chunk_cache.clear()
    
    async def export_processed_document(self, 
                                      processed_doc: ProcessedDocument,
                                      output_path: Union[str, Path],
                                      format: str = "json") -> bool:
        """
        Exportiert verarbeitetes Dokument
        """
        output_path = Path(output_path)
        
        try:
            if format == "json":
                # Convert to JSON-serializable format
                export_data = {
                    "document_id": processed_doc.document_id,
                    "title": processed_doc.title,
                    "content": processed_doc.content,
                    "metadata": processed_doc.metadata,
                    "summary": processed_doc.summary,
                    "processing_time": processed_doc.processing_time,
                    "word_count": processed_doc.word_count,
                    "chunk_count": processed_doc.chunk_count,
                    "chunks": [
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "chunk_index": chunk.chunk_index,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "has_embedding": chunk.embedding is not None,
                            "summary": chunk.summary
                        } for chunk in processed_doc.chunks
                    ]
                }
                
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(export_data, indent=2, ensure_ascii=False))
                    
            elif format == "markdown":
                # Export as structured markdown
                md_content = [
                    f"# {processed_doc.title}",
                    "",
                    f"**Document ID:** {processed_doc.document_id}",
                    f"**Processing Time:** {processed_doc.processing_time:.3f}s",
                    f"**Word Count:** {processed_doc.word_count}",
                    f"**Chunks:** {processed_doc.chunk_count}",
                    ""
                ]
                
                if processed_doc.summary:
                    md_content.extend([
                        "## Summary",
                        processed_doc.summary,
                        ""
                    ])
                
                md_content.extend([
                    "## Content",
                    processed_doc.content,
                    "",
                    "## Chunks"
                ])
                
                for i, chunk in enumerate(processed_doc.chunks):
                    md_content.extend([
                        f"### Chunk {i+1}",
                        f"**ID:** {chunk.id}",
                        f"**Range:** {chunk.start_char}-{chunk.end_char}",
                        "",
                        chunk.content,
                        ""
                    ])
                
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write('\n'.join(md_content))
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            print(f"✅ Exported document to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            return False
    
    async def benchmark(self, test_files: Optional[List[Path]] = None) -> Dict[str, float]:
        """
        Performance Benchmark für Document Processor
        """
        print("Running Document Processor Benchmark...")
        
        if test_files is None:
            # Create test documents
            test_dir = Path("test_documents")
            test_dir.mkdir(exist_ok=True)
            
            # Create test markdown
            md_content = "# Test Document\n\nThis is a test document for benchmarking.\n\n" + "Lorem ipsum dolor sit amet. " * 100
            md_file = test_dir / "test.md"
            
            async with aiofiles.open(md_file, 'w', encoding='utf-8') as f:
                await f.write(md_content)
            
            test_files = [md_file]
        
        # Clear cache for fair benchmark
        self.clear_cache()
        
        # Benchmark processing
        start_time = time.time()
        processed_docs = []
        
        for file_path in test_files:
            try:
                doc = await self.process_document(file_path, "benchmark_user")
                processed_docs.append(doc)
            except Exception as e:
                print(f"Benchmark error for {file_path}: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        total_words = sum(doc.word_count for doc in processed_docs)
        total_chunks = sum(doc.chunk_count for doc in processed_docs)
        
        return {
            "documents_processed": len(processed_docs),
            "total_time_seconds": total_time,
            "documents_per_second": len(processed_docs) / total_time,
            "total_words": total_words,
            "words_per_second": total_words / total_time,
            "total_chunks": total_chunks,
            "chunks_per_second": total_chunks / total_time,
            "average_processing_time": total_time / len(processed_docs) if processed_docs else 0
        }

# Usage Examples and Integration
async def example_usage():
    """Beispiele für Document Processor Usage"""
    
    # Initialize with custom config
    config = ProcessingConfig(
        chunk_size=800,
        chunk_overlap=150,
        preserve_structure=True,
        auto_summarize=True,
        embedding_model="mlx-community/gte-small"
    )
    
    processor = MLXDocumentProcessor(config)
    
    # Process single document
    try:
        doc = await processor.process_document(
            "example.pdf",
            user_id="user_123",
            custom_metadata={"category": "research", "priority": "high"}
        )
        
        print(f"Processed: {doc.title}")
        print(f"Chunks: {doc.chunk_count}")
        print(f"Words: {doc.word_count}")
        print(f"Time: {doc.processing_time:.3f}s")
        
        # Export document
        await processor.export_processed_document(doc, "processed_document.json", "json")
        
    except Exception as e:
        print(f"Processing error: {e}")
    
    # Process directory
    try:
        docs = await processor.process_directory(
            "documents/",
            user_id="user_123",
            recursive=True
        )
        
        print(f"Processed {len(docs)} documents from directory")
        
    except Exception as e:
        print(f"Directory processing error: {e}")
    
    # Process text directly
    text = """
    This is a sample text that we want to process directly.
    It contains multiple paragraphs and should be chunked appropriately.
    
    The document processor will create embeddings and prepare it for the brain system.
    """
    
    direct_doc = await processor.process_text_directly(
        text,
        title="Direct Text Example",
        user_id="user_123",
        metadata={"source": "api", "type": "note"}
    )
    
    print(f"Direct processing: {direct_doc.chunk_count} chunks")
    
    # Integration with Vector Store
    from mlx_components.vector_store import MLXVectorStore, VectorStoreConfig
    
    vector_config = VectorStoreConfig(base_url="http://localhost:8000")
    vector_store = MLXVectorStore(vector_config)
    
    # Save to vector store
    success = await processor.save_to_vector_store(
        direct_doc,
        vector_store,
        user_id="user_123",
        model_id="gte-small"
    )
    
    if success:
        print("✅ Document saved to vector store")
    
    # Performance stats
    stats = processor.get_performance_stats()
    print(f"Performance: {stats}")
    
    # Benchmark
    benchmark_results = await processor.benchmark()
    print(f"Benchmark: {benchmark_results}")

# Brain System Integration Helper
class DocumentToBrainConverter:
    """
    Helper class für Integration mit Brain System
    """
    
    def __init__(self, processor: MLXDocumentProcessor):
        self.processor = processor
    
    async def document_to_brain_context(self, 
                                      processed_doc: ProcessedDocument,
                                      context_type: str = "knowledge") -> Dict[str, Any]:
        """
        Konvertiert Dokument zu Brain Context Format
        """
        # Combine chunks into structured content
        brain_content = []
        
        # Add document summary as header
        if processed_doc.summary:
            brain_content.append(f"## Summary\n{processed_doc.summary}\n")
        
        # Add structured content
        brain_content.append(f"## Content from {processed_doc.title}\n")
        
        # Group chunks by semantic similarity (simple approach)
        for i, chunk in enumerate(processed_doc.chunks):
            if i % 3 == 0:  # Create sections every 3 chunks
                brain_content.append(f"\n### Section {i//3 + 1}\n")
            
            brain_content.append(chunk.content)
            brain_content.append("")  # Empty line between chunks
        
        # Prepare metadata for brain context
        brain_metadata = {
            "source_document": processed_doc.document_id,
            "document_title": processed_doc.title,
            "chunk_count": processed_doc.chunk_count,
            "word_count": processed_doc.word_count,
            "processing_date": processed_doc.metadata.get("processed_at"),
            "context_type": context_type,
            "file_type": processed_doc.metadata.get("file_type")
        }
        
        return {
            "content": "\n".join(brain_content),
            "metadata": brain_metadata,
            "context_name": f"{context_type}_{processed_doc.title.lower().replace(' ', '_')}",
            "operation": "create"
        }
    
    async def batch_documents_to_brain(self, 
                                     processed_docs: List[ProcessedDocument],
                                     user_id: str) -> List[Dict[str, Any]]:
        """
        Konvertiert mehrere Dokumente zu Brain Updates
        """
        brain_updates = []
        
        for doc in processed_docs:
            try:
                brain_context = await self.document_to_brain_context(doc)
                
                # Add user context
                brain_context.update({
                    "user_id": user_id,
                    "priority": 1  # High priority for new documents
                })
                
                brain_updates.append(brain_context)
                
            except Exception as e:
                print(f"Error converting document {doc.title} to brain context: {e}")
        
        return brain_updates

if __name__ == "__main__":
    asyncio.run(example_usage())