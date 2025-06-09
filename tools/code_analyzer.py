#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

"""
MLX Code Analyzer
Intelligente Analyse von Code Repositories für Brain System Integration
Optimiert für verschiedene Programmiersprachen und MLX Components
"""

import asyncio
import time
import ast
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import aiofiles

# Code analysis libraries
import tree_sitter # --- WAS ALREADY IMPORTED ---
from tree_sitter import Language, Parser # --- WAS ALREADY IMPORTED ---
import lizard  # Code complexity analysis

# MLX Components Integration
from mlx_components.embedding_engine import MLXEmbeddingEngine
from mlx_components.vector_store import MLXVectorStore
# --- Assuming tools.document_processor is available as in the original ---
# from tools.document_processor import MLXDocumentProcessor, ProcessedDocument
# --- Placeholder if the above is not directly available for execution ---
if True: # Conditional import for placeholder
    @dataclass
    class ProcessedDocument:
        content: str

    class MLXDocumentProcessor:
        def __init__(self, config=None, embedding_engine=None):
            pass
        async def process_document(self, path: Path) -> ProcessedDocument:
            async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            return ProcessedDocument(content=content)


@dataclass
class CodeAnalysisConfig:
    """Konfiguration für Code Analysis"""
    supported_languages: List[str] = None
    max_file_size: int = 1024 * 1024  # 1MB max per file
    include_dependencies: bool = True
    extract_documentation: bool = True
    analyze_complexity: bool = True
    generate_embeddings: bool = True
    chunk_functions: bool = True # --- Not directly used in current extractors, but good for config ---
    chunk_classes: bool = True   # --- Not directly used in current extractors, but good for config ---
    include_comments: bool = True # --- Not directly used in current extractors, but good for config ---
    exclude_patterns: List[str] = None
    embedding_model: str = "mlx-community/gte-small"
    # --- ADDED ---
    tree_sitter_grammar_dir: str = "./tree_sitter_grammars" # Directory for .so files

    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = [
                "python", "javascript", "typescript", "java", "cpp", "c",
                "go", "rust", "swift", "kotlin", "ruby", "php", "scala",
                "r", "matlab", "sql", "shell", "dockerfile"
            ]

        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "node_modules", ".git", "__pycache__", ".pytest_cache",
                "venv", "env", ".venv", "dist", "build", ".next",
                "coverage", ".coverage", "*.pyc", "*.pyo", "*.log"
            ]

@dataclass
class CodeElement:
    """Einzelnes Code-Element (Function, Class, etc.)"""
    type: str  # function, class, method, variable, import
    name: str
    content: str
    start_line: int
    end_line: int
    file_path: str
    language: str
    complexity: Optional[int] = None
    docstring: Optional[str] = None
    parameters: Optional[List[str]] = None # --- MODIFIED: Made Optional for consistency ---
    return_type: Optional[str] = None
    decorators: Optional[List[str]] = None # --- MODIFIED: Made Optional for consistency ---
    dependencies: Optional[List[str]] = None # --- MODIFIED: Made Optional for consistency ---
    embedding: Optional[List[float]] = None

@dataclass
class CodeFile:
    """Analysierte Code-Datei"""
    file_path: str
    language: str
    content: str
    elements: List[CodeElement]
    imports: List[str]
    complexity_score: float
    line_count: int
    documentation: Optional[str] = None
    dependencies: List[str] = None
    file_hash: Optional[str] = None

@dataclass
class RepositoryAnalysis:
    """Vollständige Repository-Analyse"""
    repo_path: str
    project_name: str
    files: List[CodeFile]
    language_distribution: Dict[str, int]
    total_lines: int
    total_complexity: float
    dependency_graph: Dict[str, List[str]]
    documentation_coverage: float
    analysis_time: float
    readme_content: Optional[str] = None
    structure_summary: Optional[str] = None

class MLXCodeAnalyzer:
    """
    High-Performance Code Analyzer für MLX Ecosystem

    Features:
    - Multi-Language Support mit Tree-sitter
    - Intelligent Code Chunking (Functions, Classes)
    - Complexity Analysis mit Lizard
    - Dependency Graph Generation
    - Documentation Extraction
    - MLX Embedding Integration
    - Repository Structure Analysis
    - Brain System Ready Output
    """

    def __init__(self,
                 config: CodeAnalysisConfig = None,
                 embedding_engine: MLXEmbeddingEngine = None,
                 document_processor: MLXDocumentProcessor = None):
        self.config = config or CodeAnalysisConfig()
        self.embedding_engine = embedding_engine
        self.document_processor = document_processor

        self.total_files = 0
        self.total_elements = 0
        self.total_analysis_time = 0.0
        self.language_stats = {}

        self.language_extensions = {
            "python": [".py", ".pyx", ".pyi"],
            "javascript": [".js", ".jsx", ".mjs"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h", ".hh"], # --- ADDED .hh ---
            "c": [".c", ".h"],
            "go": [".go"],
            "rust": [".rs"],
            "swift": [".swift"],
            "kotlin": [".kt", ".kts"],
            "ruby": [".rb"],
            "php": [".php"],
            "scala": [".scala", ".sc"],
            "r": [".r", ".R"],
            "sql": [".sql"],
            "shell": [".sh", ".bash", ".zsh"],
            "dockerfile": ["Dockerfile", ".dockerfile"]
            # --- Matlab is listed in supported_languages but no extension here, can be added ---
        }

        # --- MODIFIED: Regex patterns, can be extended or used as fallback ---
        self.language_patterns = {
            "python": {
                "function": r"def\s+(\w+)\s*\(",
                "class": r"class\s+(\w+)\s*[\(:]",
            },
            "javascript": { # Used by _extract_js_elements, keep for now or refactor to tree-sitter
                "function": r"function\s+(\w+)\s*\(|(\w+)\s*:\s*function|const\s+(\w+)\s*=\s*function|let\s+(\w+)\s*=\s*function|var\s+(\w+)\s*=\s*function|(\w+)\s*=\s*\(.*?\)\s*=>",
                "class": r"class\s+(\w+)",
            },
            "java": { # Used by _extract_java_elements, keep for now or refactor to tree-sitter
                "method": r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+(?:\[\])?\s+)*(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*{",
                "class": r"(?:public|private|protected)?\s*(?:abstract\s+|final\s+)?class\s+(\w+)",
            },
            # --- Consider adding patterns for other languages if tree-sitter is not used/fails for them ---
            "c": {
                "function": r"\w+\s+\**\s*(\w+)\s*\([^)]*\)\s*{", # Simplified C function
            },
            "cpp": {
                "function": r"\w+(?:\s*::\s*\w+)*\s+\**\s*(\w+)\s*\([^)]*\)\s*(const)?\s*{", # Simplified C++ func/method
                "class": r"class\s+(\w+)",
                "struct": r"struct\s+(\w+)",
            }
        }

        # --- ADDED: Tree-sitter parser setup ---
        self.tree_sitter_parsers: Dict[str, Dict[str, Any]] = {}
        self._initialize_tree_sitter_parsers()


    def _initialize_tree_sitter_parsers(self): # --- ADDED ---
        """Loads tree-sitter grammars for supported languages."""
        # Format: "language_key": ("language_name_for_grammar_file", "language_name_for_queries")
        # Example: tree-sitter-cpp.so, language_name "cpp"
        # Ensure grammar files are compiled and present in self.config.tree_sitter_grammar_dir
        tree_sitter_langs_config = {
            "cpp": "cpp",
            "c": "c",
            "go": "go",
            "rust": "rust",
            "javascript": "javascript", # Can also use tree-sitter for JS/TS
            "typescript": "typescript", # Can also use tree-sitter for JS/TS
            # Add other languages here if you have the grammars
            # "python": "python", # Python uses AST module by default, but tree-sitter is an option
            # "java": "java",     # Java uses regex by default, tree-sitter is an option
        }

        grammar_base_path = Path(self.config.tree_sitter_grammar_dir)
        if not grammar_base_path.exists():
            print(f"⚠️ Tree-sitter grammar directory not found: {grammar_base_path}. Tree-sitter specific parsing will be limited.")
            return

        for lang_key, lang_name_for_grammar in tree_sitter_langs_config.items():
            # Common naming convention for shared library objects
            # Adjust if your files are named differently (e.g. libtree-sitter-python.so)
            grammar_file = grammar_base_path / f"tree-sitter-{lang_name_for_grammar}.so"
            if not grammar_file.exists():
                # Try .dylib for macOS
                grammar_file = grammar_base_path / f"libtree-sitter-{lang_name_for_grammar}.dylib"
            if not grammar_file.exists():
                # Try .dll for Windows - not typically how tree-sitter grammars are named but as a fallback
                grammar_file = grammar_base_path / f"tree-sitter-{lang_name_for_grammar}.dll"


            if grammar_file.exists():
                try:
                    ts_language_obj = Language(str(grammar_file.resolve()), lang_name_for_grammar)
                    parser = Parser()
                    parser.set_language(ts_language_obj)
                    self.tree_sitter_parsers[lang_key] = {
                        "parser": parser,
                        "language_object": ts_language_obj,
                        "query_language_name": lang_name_for_grammar # Used for finding query files if any
                    }
                    print(f"Loaded tree-sitter grammar for {lang_key} from {grammar_file}")
                except Exception as e:
                    print(f"⚠️ Failed to load tree-sitter grammar for {lang_key} from {grammar_file}: {e}")
            else:
                print(f"⚠️ Grammar file not found for {lang_key}: Expected at or similar to {grammar_file.parent / f'tree-sitter-{lang_name_for_grammar}.so'}")
    
    async def initialize(self) -> None:
        """
        Initialisiert Code Analyzer mit dependencies
        """
        if self.embedding_engine is None and self.config.generate_embeddings: # --- MODIFIED: check generate_embeddings flag ---
            from mlx_components.embedding_engine import MLXEmbeddingEngine, EmbeddingConfig # type: ignore

            embedding_config = EmbeddingConfig(
                model_path=self.config.embedding_model,
                batch_size=16, # --- Consider making batch_size configurable ---
                cache_embeddings=True
            )
            self.embedding_engine = MLXEmbeddingEngine(embedding_config)
            await self.embedding_engine.initialize()

        if self.document_processor is None:
            # from tools.document_processor import MLXDocumentProcessor, ProcessingConfig # type: ignore
            # --- Placeholder if the above is not directly available for execution ---
            class ProcessingConfig:
                def __init__(self, chunk_size, preserve_structure, extract_metadata):
                    pass # dummy
            
            doc_config = ProcessingConfig( # type: ignore
                chunk_size=800, # --- Consider making these configurable ---
                preserve_structure=True,
                extract_metadata=True
            )
            self.document_processor = MLXDocumentProcessor(doc_config, self.embedding_engine)


    async def analyze_repository(self,
                                 repo_path: Union[str, Path],
                                 user_id: Optional[str] = None) -> RepositoryAnalysis:
        start_time = time.time()
        repo_path = Path(repo_path)

        await self.initialize()

        if not repo_path.is_dir():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        print(f"🔍 Analyzing repository: {repo_path.name}")

        code_files_paths = await self._discover_code_files(repo_path)
        print(f"📁 Found {len(code_files_paths)} code files")

        analyzed_files: List[CodeFile] = []
        batch_size = 10 # --- Consider making batch_size configurable ---

        for i in range(0, len(code_files_paths), batch_size):
            batch_paths = code_files_paths[i:i + batch_size]
            tasks = [self._analyze_file(file_path, repo_path) for file_path in batch_paths]
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        print(f"❌ Failed to analyze {batch_paths[j]}: {result}")
                    elif result:
                        analyzed_files.append(result)
            except Exception as e:
                print(f"❌ Batch analysis error: {e}")

        dependency_graph = await self._build_dependency_graph(analyzed_files)
        language_distribution: Dict[str, int] = {}
        total_lines = 0
        total_complexity = 0.0

        for file_obj in analyzed_files:
            lang = file_obj.language
            language_distribution[lang] = language_distribution.get(lang, 0) + 1
            total_lines += file_obj.line_count
            total_complexity += file_obj.complexity_score

        readme_content = await self._extract_readme(repo_path)
        structure_summary = await self._generate_structure_summary(analyzed_files, repo_path)

        documented_elements = sum(
            1 for file_obj in analyzed_files
            for element in file_obj.elements
            if element.docstring
        )
        total_code_elements = sum(len(file_obj.elements) for file_obj in analyzed_files) # --- RENAMED variable for clarity ---
        documentation_coverage = (documented_elements / total_code_elements * 100) if total_code_elements > 0 else 0.0 # --- Ensure float division ---

        analysis_time_taken = time.time() - start_time # --- RENAMED variable for clarity ---

        self.total_files += len(analyzed_files)
        self.total_elements += total_code_elements
        self.total_analysis_time += analysis_time_taken

        analysis = RepositoryAnalysis(
            repo_path=str(repo_path),
            project_name=repo_path.name,
            files=analyzed_files,
            language_distribution=language_distribution,
            total_lines=total_lines,
            total_complexity=total_complexity,
            dependency_graph=dependency_graph,
            documentation_coverage=documentation_coverage,
            analysis_time=analysis_time_taken,
            readme_content=readme_content,
            structure_summary=structure_summary
        )

        print(f"✅ Repository analysis complete:")
        print(f"   📄 Files: {len(analyzed_files)}")
        print(f"   📝 Lines: {total_lines:,}")
        print(f"   🧩 Elements: {total_code_elements}") # --- Use updated variable name ---
        print(f"   📚 Documentation: {documentation_coverage:.1f}%")
        print(f"   ⏱️ Time: {analysis_time_taken:.2f}s")

        return analysis

    async def analyze_file(self,
                           file_path: Union[str, Path],
                           user_id: Optional[str] = None) -> Optional[CodeFile]:
        file_path = Path(file_path)
        await self.initialize()
        if not file_path.is_file():
            # raise ValueError(f"File does not exist: {file_path}")
            print(f"⚠️ File does not exist: {file_path}") # --- Softer error handling ---
            return None
        # --- Determine repo_root: if it's a single file, parent is root. If part of repo, need actual root. ---
        # --- For standalone file analysis, its parent can be the root. ---
        return await self._analyze_file(file_path, file_path.parent)


    async def _discover_code_files(self, repo_path: Path) -> List[Path]:
        code_files = []
        all_extensions = set()
        for ext_list in self.language_extensions.values():
            all_extensions.update(ext_list)

        # Using a try-except block for rglob if permissions are an issue for some subdirs
        try:
            for item_path in repo_path.rglob("*"):
                if not item_path.is_file():
                    continue
                if self._is_excluded(item_path, repo_path):
                    continue

                # Check by extension or specific filenames
                if (item_path.suffix.lower() in all_extensions or
                        item_path.name in self.language_extensions.get("dockerfile", []) or # Example for Dockerfile
                        item_path.name in self.language_extensions.get("shell", [])): # Example for shell script names
                    try:
                        if item_path.stat().st_size <= self.config.max_file_size:
                            code_files.append(item_path)
                        else:
                            print(f"⚠️ Skipping large file: {item_path} (size: {item_path.stat().st_size} bytes)")
                    except FileNotFoundError: # File might be a broken symlink or removed during scan
                        print(f"⚠️ File not found during stat: {item_path}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Error stating file {item_path}: {e}")
                        continue
        except PermissionError:
            print(f"⚠️ Permission denied while scanning directory: {repo_path}")
        except Exception as e:
            print(f"⚠️ Error discovering code files in {repo_path}: {e}")
        return sorted(code_files)

    def _is_excluded(self, file_path: Path, repo_path: Path) -> bool:
        try:
            relative_path_str = str(file_path.relative_to(repo_path))
        except ValueError: # file_path might not be under repo_path (e.g. symlink outside)
            return True # Exclude if not clearly within the repository relative structure

        for pattern in self.config.exclude_patterns:
            # More robust pattern matching (e.g., fnmatch or regex can be used here)
            # For simplicity, keeping "in" but mindful of its limitations (e.g. "src" in "some_source_file.txt")
            # A common way is to check parts of the path:
            if any(part == pattern for part in relative_path_str.split(Path().sep)) or \
               relative_path_str.startswith(pattern + Path().sep) or \
               Path(relative_path_str).name == pattern or \
               Path(relative_path_str).suffix == pattern : # for excluding by extension like '*.log'
                return True
        return False

    async def _analyze_file(self, file_path: Path, repo_root: Path) -> Optional[CodeFile]:
        try:
            language = self._detect_language(file_path)
            if not language:
                # print(f"⚠️ Could not detect language for {file_path}, skipping.")
                return None

            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()

            if not content.strip(): # Skip empty or whitespace-only files
                return None

            file_hash = hashlib.md5(content.encode('utf-8')).hexdigest() # --- Ensure utf-8 for hashing ---
            
            # Use relative path for CodeFile object
            try:
                relative_file_path_str = str(file_path.relative_to(repo_root))
            except ValueError: # If file_path is not under repo_root (e.g. when analyzing a single external file)
                relative_file_path_str = file_path.name


            elements = await self._extract_code_elements(content, Path(relative_file_path_str), language) # --- Pass relative path ---
            imports = await self._extract_imports(content, language)
            complexity_score = await self._calculate_complexity(content, str(file_path), language) # --- Lizard needs absolute path ---
            documentation = await self._extract_file_documentation(content, language)
            dependencies = await self._extract_dependencies(content, language, file_path)

            if self.config.generate_embeddings and elements and self.embedding_engine:
                await self._generate_element_embeddings(elements)

            return CodeFile(
                file_path=relative_file_path_str,
                language=language,
                content=content, # Storing full content can be memory intensive for large repos. Consider if needed.
                elements=elements,
                imports=imports,
                complexity_score=complexity_score,
                line_count=len(content.splitlines()), # Use splitlines for more accurate line count
                documentation=documentation,
                dependencies=dependencies,
                file_hash=file_hash
            )
        except Exception as e:
            print(f"⚠️ Error analyzing file {file_path}: {e}")
            # import traceback
            # traceback.print_exc() # For more detailed debugging
            return None


    def _detect_language(self, file_path: Path) -> Optional[str]:
        extension = file_path.suffix.lower()
        file_name = file_path.name # For files like 'Dockerfile'

        for lang, exts_or_names in self.language_extensions.items():
            if extension in exts_or_names or file_name in exts_or_names:
                return lang
        return None


    # --- MODIFIED: _extract_code_elements to include tree-sitter ---
    async def _extract_code_elements(self,
                                     content: str,
                                     file_path: Path, # Should be relative path
                                     language: str) -> List[CodeElement]:
        elements: List[CodeElement] = []

        if language == "python":
            elements = await self._extract_python_elements(content, file_path)
        elif language in ["javascript", "typescript"]:
            # Option to use tree-sitter for JS/TS if a parser is loaded
            if language in self.tree_sitter_parsers and self.config.supported_languages.count(language+"_ts") > 0 : # custom flag or check
                 print(f"Using Tree-sitter for {language}")
                 elements = await self._extract_tree_sitter_elements(content, file_path, language)
            else: # Fallback to regex or existing method
                elements = await self._extract_js_elements(content, file_path, language)
        elif language == "java":
             if language in self.tree_sitter_parsers and self.config.supported_languages.count(language+"_ts") > 0:
                 print(f"Using Tree-sitter for {language}")
                 elements = await self._extract_tree_sitter_elements(content, file_path, language)
             else: # Fallback to regex or existing method
                elements = await self._extract_java_elements(content, file_path)
        elif language in self.tree_sitter_parsers: # C, CPP, Go, Rust etc.
            print(f"Using Tree-sitter for {language}")
            elements = await self._extract_tree_sitter_elements(content, file_path, language)
        else:
            # Fallback to generic regex-based extraction if patterns are defined
            # Or if no tree-sitter parser is available for this language
            print(f"Using generic regex extraction for {language}")
            elements = await self._extract_generic_elements(content, file_path, language)

        return elements

    # --- ADDED: Tree-sitter based element extraction ---
    async def _extract_tree_sitter_elements(self, content: str, file_path: Path, language: str) -> List[CodeElement]:
        elements: List[CodeElement] = []
        if language not in self.tree_sitter_parsers:
            return elements # Should not happen if called from _extract_code_elements logic

        parser_info = self.tree_sitter_parsers[language]
        parser: Parser = parser_info["parser"]
        ts_lang_obj: Language = parser_info["language_object"]
        # query_lang_name: str = parser_info["query_language_name"] # For loading .scm files

        tree = parser.parse(bytes(content, "utf8"))
        
        # Example: Define queries for functions and classes (these are language-specific)
        # You would typically store these in .scm files and load them.
        # This is a simplified inline example.
        queries = {}
        if language == "cpp":
            queries = {
                "function": """
                    (function_definition
                        declarator: [
                            (function_declarator name: (identifier) @function.name)
                            (function_declarator name: (qualified_identifier scope: _ name: (identifier) @function.name))
                            (function_declarator name: (operator_name) @function.name)
                        ]
                    ) @function.definition
                    (function_definition
                        type: (_) @function.return_type
                        declarator: (function_declarator
                            name: (field_identifier) @function.name
                        )
                    ) @function.definition
                """,
                "class": """
                    (class_specifier name: (type_identifier) @class.name) @class.definition
                    (struct_specifier name: (type_identifier) @class.name) @class.definition
                """
                # Add queries for methods within classes, variables, etc.
            }
        elif language == "c":
             queries = {
                "function": """
                    (function_definition
                        declarator: (function_declarator
                            declarator: (identifier) @function.name
                        )
                    ) @function.definition
                """,
                "struct": """
                    (struct_specifier name: (type_identifier) @struct.name) @struct.definition
                """
             }
        elif language == "go":
            queries = {
                "function": """(function_declaration name: (identifier) @function.name) @function.definition""",
                "method": """(method_declaration name: (field_identifier) @function.name) @function.definition""",
                "type": """(type_alias name: (type_identifier) @type.name) @type.definition""", # type Foo string
                "struct": """(type_spec name: (type_identifier) @struct.name (struct_type) ) @struct.definition""" # type Foo struct { ... }
            }
        elif language == "rust":
            queries = {
                "function": """(function_item name: (identifier) @function.name) @function.definition""",
                "struct": """(struct_item name: (type_identifier) @struct.name) @struct.definition""",
                "enum": """(enum_item name: (type_identifier) @enum.name) @enum.definition""",
                "trait": """(trait_item name: (type_identifier) @trait.name) @trait.definition""",
                "impl": """(impl_item trait: (type_identifier)? @impl.trait type: (type_identifier) @impl.type) @impl.definition"""
            }
        # Add more language queries...

        for element_type_key, query_scm in queries.items():
            try:
                query = ts_lang_obj.query(query_scm)
                captures = query.captures(tree.root_node)

                # Store definitions to avoid duplicate processing if name and definition are separate captures
                processed_nodes = set()

                for node, capture_name in captures:
                    if node.id in processed_nodes:
                        continue

                    if capture_name.endswith(".definition"):
                        element_name_node = None
                        # Try to find the corresponding '.name' capture for this definition
                        for cap_node, cap_name_inner in captures:
                            if cap_name_inner == capture_name.replace(".definition", ".name") and \
                               (node.start_byte <= cap_node.start_byte and node.end_byte >= cap_node.end_byte): # name is child of def
                                element_name_node = cap_node
                                break
                        
                        if not element_name_node: # Fallback if name capture is not found or structured differently
                            # Try to infer name from common patterns if specific name capture fails.
                            # This part needs to be very language-specific.
                            if node.child_by_field_name("name"):
                                element_name_node = node.child_by_field_name("name")
                            elif len(node.children) > 1 and node.children[1].type in ["identifier", "type_identifier", "field_identifier"]:
                                element_name_node = node.children[1]


                        element_name = element_name_node.text.decode('utf-8') if element_name_node else "unknown_ts_element"
                        
                        start_line = node.start_point[0] + 1
                        end_line = node.end_point[0] + 1
                        element_content = self._get_ts_node_content(content, node) # Use helper
                        
                        # TODO: Extract docstrings, parameters, return types using more tree-sitter queries
                        docstring = self._extract_ts_docstring(node, content)
                        params = self._extract_ts_parameters(node, ts_lang_obj)


                        code_el = CodeElement(
                            type=element_type_key, # "function", "class" etc. from our query keys
                            name=element_name,
                            content=element_content,
                            start_line=start_line,
                            end_line=end_line,
                            file_path=str(file_path),
                            language=language,
                            docstring=docstring,
                            parameters=params
                            # complexity, return_type, decorators, dependencies would need more logic
                        )
                        elements.append(code_el)
                        processed_nodes.add(node.id)
            except Exception as e: # Catch errors during querying or processing for one element type
                print(f"⚠️ Error processing tree-sitter query for {element_type_key} in {file_path} ({language}): {e}")
                # import traceback
                # traceback.print_exc()

        return elements

    def _extract_ts_docstring(self, node: tree_sitter.Node, full_content: str) -> Optional[str]: # --- ADDED ---
        """Extracts docstring/comment immediately preceding a tree-sitter node."""
        # This is a simplified example. Real docstring extraction can be complex
        # and language-specific (e.g., Python docstrings vs Javadoc vs C++ Doxygen).
        # It might involve looking at previous_sibling which is a comment node.
        # For now, let's assume comments directly above the node.
        
        # Look for comments attached to or immediately preceding the node.
        # Tree-sitter's AST usually includes comments as nodes.
        # We need to walk upwards or check previous siblings.
        
        # Placeholder: A more robust solution would involve specific queries for comments
        # or analyzing the node's siblings.
        # For example, in Python tree-sitter grammar, a function_definition might have a
        # block whose first child is an expression_statement with a string literal (docstring).
        
        # This is a very basic check for a comment block before the node.
        # It does not parse the comment type.
        comment_text = []
        current_node = node.prev_named_sibling
        while current_node and "comment" in current_node.type.lower():
            comment_lines = current_node.text.decode('utf-8').splitlines()
            # Clean comment markers if necessary (e.g. //, /*, #) - very language specific
            cleaned_lines = [re.sub(r"^\s*(\/\/|\* ?|# ?)", "", line).strip() for line in comment_lines]
            comment_text = cleaned_lines + comment_text # Prepend to keep order
            current_node = current_node.prev_named_sibling
            if not comment_text and current_node: # if current node is not a comment, stop
                 if "comment" not in current_node.type.lower(): break

        if comment_text:
            return "\n".join(comment_text)

        # For languages like Python, docstring might be the first statement in a function/class body
        if len(node.children) > 0:
            body_node = None
            if node.child_by_field_name("body"): # Common field name in many grammars
                body_node = node.child_by_field_name("body")

            if body_node and len(body_node.children) > 0:
                first_child_in_body = body_node.children[0]
                if first_child_in_body.type == "expression_statement":
                    if len(first_child_in_body.children) > 0 and first_child_in_body.children[0].type == "string_literal": # Python-like docstring
                        doc_node = first_child_in_body.children[0]
                        raw_doc = doc_node.text.decode('utf-8')
                        # Remove quotes
                        if (raw_doc.startswith('"""') and raw_doc.endswith('"""')) or \
                           (raw_doc.startswith("'''") and raw_doc.endswith("'''")):
                            return raw_doc[3:-3].strip()
                        if (raw_doc.startswith('"') and raw_doc.endswith('"')) or \
                           (raw_doc.startswith("'") and raw_doc.endswith("'")):
                            return raw_doc[1:-1].strip()
        return None

    def _extract_ts_parameters(self, node: tree_sitter.Node, language_obj: Language) -> Optional[List[str]]: # --- ADDED ---
        """Extracts parameters from a function/method node using tree-sitter queries."""
        # This is highly language-specific. Requires queries for parameter lists.
        # Example for a C-like language (simplified):
        # query_scm_params = "((parameter_declaration declarator: (identifier) @param.name) .)""
        # This function would need the specific query for the language.
        
        param_query_str = ""
        if node.type == "function_definition" or node.type == "function_item" or node.type == "method_declaration": # Add other types
            # Generic query, may need adjustment per language
            # This looks for (parameters ... (identifier) @name ) or similar constructs.
            if language_obj.name == "python": # Python ast.arg.arg is simpler
                return None # Python handled by _extract_python_elements
            elif language_obj.name in ["c", "cpp"]:
                 param_query_str = """
                    (parameter_declaration declarator: (identifier) @param.name)
                    (parameter_declaration name: (identifier) @param.name) ;; For some C++ styles
                    (parameter_list . (parameter_declaration (identifier) @param.name)) ;; More specific for some grammars
                 """
            elif language_obj.name == "go":
                 param_query_str = """
                    (parameter_declaration name: (identifier) @param.name)
                 """
            elif language_obj.name == "rust":
                param_query_str = """
                    (parameter pattern: (identifier) @param.name)
                """
            # Add more specific parameter queries for other languages
            
        if not param_query_str:
            return None

        try:
            query = language_obj.query(param_query_str)
            # We need to query within the context of the current function node (e.g., its parameter_list child)
            parameter_list_node = node.child_by_field_name("parameters") or node.child_by_field_name("parameter_list")
            
            params = []
            if parameter_list_node:
                captures = query.captures(parameter_list_node)
                for param_node, capture_name in captures:
                    if capture_name == "param.name":
                        params.append(param_node.text.decode('utf-8'))
                return list(dict.fromkeys(params)) if params else None # Remove duplicates
            else: # If no specific parameter list node, try on the function node itself
                captures = query.captures(node)
                for param_node, capture_name in captures:
                    if capture_name == "param.name":
                        params.append(param_node.text.decode('utf-8'))
                return list(dict.fromkeys(params)) if params else None

        except Exception as e:
            print(f"⚠️ Error extracting parameters with tree-sitter: {e}")
            return None
        return None


    async def _extract_python_elements(self, content: str, file_path: Path) -> List[CodeElement]:
        elements = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): # --- ADDED AsyncFunctionDef ---
                    element = CodeElement(
                        type="function",
                        name=node.name,
                        content=self._get_node_content(content, node),
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno), # end_lineno is Python 3.8+
                        file_path=str(file_path),
                        language="python",
                        docstring=ast.get_docstring(node, clean=False), # --- Preserve raw docstring ---
                        parameters=[arg.arg for arg in node.args.args],
                        decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                        return_type=ast.unparse(node.returns) if node.returns else None # Python 3.8+ for unparse
                    )
                    elements.append(element)
                elif isinstance(node, ast.ClassDef):
                    element = CodeElement(
                        type="class",
                        name=node.name,
                        content=self._get_node_content(content, node),
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        file_path=str(file_path),
                        language="python",
                        docstring=ast.get_docstring(node, clean=False),
                        decorators=[self._get_decorator_name(d) for d in node.decorator_list]
                    )
                    elements.append(element)
                elif isinstance(node, ast.Assign):
                    # Extract important variables/constants (e.g., all-caps)
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper(): # Constants by convention
                             # Check if it's a module-level constant
                            is_module_level = True # Assume true unless found in func/class
                            parent = getattr(node, '_parent', None) # Requires parent pointers if using custom AST walk
                                                                    # Standard ast.walk doesn't add parent
                                                                    # For simplicity, we'll assume all-caps are module level if not using complex parent tracking
                            
                            # A simple check: if lineno is small and it's not inside a known class/func range, assume module level
                            # This is heuristic. A more robust way needs full scope analysis.
                            
                            # If we want to be more precise about module-level, we'd need to track current scope
                            # or check if node is a direct child of ast.Module.
                            # For now, keeping the original logic.

                            element = CodeElement(
                                type="constant", # Or "variable" if not strictly constant
                                name=target.id,
                                content=self._get_node_content(content, node), # Content of the assignment line
                                start_line=node.lineno,
                                end_line=getattr(node, 'end_lineno', node.lineno),
                                file_path=str(file_path),
                                language="python"
                            )
                            elements.append(element)
        except SyntaxError as e:
            print(f"⚠️ Python syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"⚠️ Error parsing Python file {file_path}: {e}")
            # import traceback
            # traceback.print_exc()
        return elements

    async def _extract_js_elements(self, content: str, file_path: Path, language: str) -> List[CodeElement]:
        # This regex-based method can be kept as a fallback or for speed if tree-sitter is too slow for some use cases.
        # However, tree-sitter is generally more robust for JS/TS.
        elements = []
        lines = content.splitlines() # --- Use splitlines ---

        # Adjusted patterns for common JS/TS function and class declarations
        # Note: Regex for complex JS/TS (especially with frameworks, decorators, complex exports) is very hard.
        function_patterns = [
            r'(?:async\s+)?function\s+(?:\*\s*)?(\w+)\s*\(',  # function foo() / async function foo() / function* foo()
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function(?:\s*\*)?\s*\(', # const foo = function()
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>', # const foo = () => / async () =>
            r'(?:export\s+(?:default\s+)?)(?:async\s+)?function\s+(?:\*\s*)?(\w+)\s*\(', # export function foo / export default function
            r'(\w+)\s*:\s*(?:async\s+)?function(?:\s*\*)?\s*\(', # foo: function() in object literals
            r'(?:static\s+)?(?:async\s+)?(\w+)\s*\([^)]*\)\s*{', # class methods: method() {} / static method() {} / async method() {}
        ]
        class_patterns = [
            r'(?:export\s+(?:default\s+)?)?class\s+(\w+)(?:\s+extends\s+[\w.]+)?', # class Foo / class Foo extends Bar / export class Foo
        ]

        # Attempt to find matching elements. This is simplified.
        # Proper parsing of nested structures and accurate end_line is difficult with regex.
        # _find_block_end is a heuristic.
        
        # Combine patterns for a single pass if performance is critical,
        # but separate loops are clearer for different element types.

        # Find Classes first to help differentiate methods from standalone functions
        class_regions: List[Tuple[int, int, str]] = []
        for line_num, line_text in enumerate(lines):
            for pattern in class_patterns:
                match = re.search(pattern, line_text)
                if match:
                    class_name = match.group(1)
                    start_line_idx = line_num
                    try:
                        end_line_idx = self._find_block_end(lines, start_line_idx)
                        class_content = '\n'.join(lines[start_line_idx:end_line_idx])
                        elements.append(CodeElement(
                            type="class", name=class_name, content=class_content,
                            start_line=start_line_idx + 1, end_line=end_line_idx, # 1-based
                            file_path=str(file_path), language=language
                        ))
                        class_regions.append((start_line_idx + 1, end_line_idx, class_name))
                    except Exception as e_block:
                        print(f"Error finding block end for class {class_name} in {file_path}: {e_block}")
                    break # Found class on this line

        # Find Functions (including methods)
        for line_num, line_text in enumerate(lines):
            for pattern in function_patterns:
                match = re.search(pattern, line_text)
                if match:
                    # Find first non-None group for name (regex patterns have multiple capture groups for name)
                    func_name = next((g for g in match.groups() if g is not None), None)
                    if not func_name: continue

                    start_line_idx = line_num
                    
                    # Heuristic: if func_name starts with uppercase and no class was found, it might be a constructor-like function
                    # More robust: check if inside a class region
                    is_method = False
                    current_class_name = None
                    for class_start, class_end, c_name in class_regions:
                        if class_start <= (start_line_idx + 1) <= class_end:
                            is_method = True
                            current_class_name = c_name
                            break
                    
                    element_type = "method" if is_method else "function"
                    # If it's a method, prepend class name for uniqueness if desired, or store context
                    # element_name_to_store = f"{current_class_name}.{func_name}" if is_method and current_class_name else func_name

                    try:
                        end_line_idx = self._find_block_end(lines, start_line_idx) # 0-based index
                        func_content = '\n'.join(lines[start_line_idx:end_line_idx])
                        elements.append(CodeElement(
                            type=element_type, name=func_name, content=func_content,
                            start_line=start_line_idx + 1, end_line=end_line_idx, # 1-based
                            file_path=str(file_path), language=language
                        ))
                    except Exception as e_block_func:
                         print(f"Error finding block end for function {func_name} in {file_path}: {e_block_func}")
                    break # Found function/method on this line
        return elements


    async def _extract_java_elements(self, content: str, file_path: Path) -> List[CodeElement]:
        # This regex-based method can be kept as a fallback. Tree-sitter is generally more robust for Java.
        elements = []
        lines = content.splitlines()

        # Java patterns (simplified, might need refinement for generics, annotations, etc.)
        # Order matters: find classes/interfaces/enums first.
        class_interface_enum_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|static\s+|final\s+)*?(class|interface|enum)\s+(\w+)'
        # Method pattern (simplified) - needs to avoid constructors matching class names.
        method_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:static\s+|final\s+|abstract\s+)?(?:<\w+(?:\s+extends\s+\w+)?(?:,\s*\w+(?:\s+extends\s+\w+)?)*>\s+)?(?:\w+(?:\[\])?\s+)+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,.]+)?\s*{'
        # Constructor pattern
        constructor_pattern = r'(?:public\s+|private\s+|protected\s+)?(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,.]+)?\s*{'

        found_outer_structures: List[Tuple[int,int,str,str]] = [] # start, end, type, name

        for line_num, line_text in enumerate(lines):
            match = re.search(class_interface_enum_pattern, line_text)
            if match:
                structure_type = match.group(1) # class, interface, enum
                structure_name = match.group(2)
                start_line_idx = line_num
                try:
                    end_line_idx = self._find_block_end(lines, start_line_idx)
                    structure_content = '\n'.join(lines[start_line_idx:end_line_idx])
                    elements.append(CodeElement(
                        type=structure_type, name=structure_name, content=structure_content,
                        start_line=start_line_idx + 1, end_line=end_line_idx,
                        file_path=str(file_path), language="java"
                    ))
                    found_outer_structures.append((start_line_idx + 1, end_line_idx, structure_type, structure_name))
                except Exception as e_block_struct:
                    print(f"Error finding block for {structure_type} {structure_name} in {file_path}: {e_block_struct}")
                continue # Process one structure per line, then move to next line for methods etc.

        for line_num, line_text in enumerate(lines):
            # Check for methods
            method_match = re.search(method_pattern, line_text)
            if method_match:
                method_name = method_match.group(1)
                # Avoid matching class/interface/enum keywords as method names by checking if it's a reserved word for structures
                if method_name in ["class", "interface", "enum"]: 
                    continue
                
                # Heuristic to avoid matching constructor as a regular method initially (if name matches a class)
                # A proper constructor check is done next.
                is_potential_constructor = any(method_name == s_name for _,_,s_type,s_name in found_outer_structures if s_type == "class")
                if is_potential_constructor: # Skip here, let constructor logic handle it
                    pass # It will be checked by constructor pattern

                start_line_idx = line_num
                try:
                    end_line_idx = self._find_block_end(lines, start_line_idx)
                    method_content = '\n'.join(lines[start_line_idx:end_line_idx])
                    elements.append(CodeElement(
                        type="method", name=method_name, content=method_content,
                        start_line=start_line_idx + 1, end_line=end_line_idx,
                        file_path=str(file_path), language="java"
                    ))
                except Exception as e_block_method:
                     print(f"Error finding block for method {method_name} in {file_path}: {e_block_method}")
                continue
            
            # Check for constructors
            constructor_match = re.search(constructor_pattern, line_text)
            if constructor_match:
                constructor_name = constructor_match.group(1)
                # Verify it's a constructor (name matches a known class name)
                is_actual_constructor = any(constructor_name == s_name for _,_,s_type,s_name in found_outer_structures if s_type == "class")
                if is_actual_constructor:
                    start_line_idx = line_num
                    try:
                        end_line_idx = self._find_block_end(lines, start_line_idx)
                        constructor_content = '\n'.join(lines[start_line_idx:end_line_idx])
                        elements.append(CodeElement(
                            type="constructor", name=constructor_name, content=constructor_content,
                            start_line=start_line_idx + 1, end_line=end_line_idx,
                            file_path=str(file_path), language="java"
                        ))
                    except Exception as e_block_constructor:
                        print(f"Error finding block for constructor {constructor_name} in {file_path}: {e_block_constructor}")


        return elements


    async def _extract_generic_elements(self, content: str, file_path: Path, language: str) -> List[CodeElement]:
        elements = []
        lines = content.splitlines() # Use splitlines

        # Generic patterns are hard. This is a very simplified approach.
        # It tries to find function/class-like structures using patterns defined in self.language_patterns.
        # This method is a fallback if no specific parser (AST, Tree-sitter) is used.
        
        lang_specific_patterns = self.language_patterns.get(language)
        if not lang_specific_patterns:
            # If no patterns for this language, try some very generic ones (less reliable)
            # This part could be expanded with truly generic regexes, but their utility is limited.
            # For example, looking for `word (...) {` patterns.
            # print(f"No generic patterns defined for {language}, generic extraction will be limited.")
            return elements

        # Iterate over defined pattern types (e.g., "function", "class")
        for element_type, pattern_str in lang_specific_patterns.items():
            if element_type not in ["function", "class", "method", "struct"]: # Focus on common structures
                continue
            try:
                regex_pattern = re.compile(pattern_str)
                for line_num, line_text in enumerate(lines):
                    match = regex_pattern.search(line_text)
                    if match:
                        # Try to get the first non-None capturing group as the name
                        name = next((g for g in match.groups() if g is not None), None)
                        if not name:
                            name = f"unnamed_{element_type}"

                        start_line_idx = line_num
                        # Heuristic for block end: use existing _find_block_end or simpler logic
                        try:
                            end_line_idx = self._find_block_end(lines, start_line_idx)
                            element_content_lines = lines[start_line_idx:end_line_idx]
                        except Exception: # Fallback if _find_block_end fails badly
                             # Simple fallback: take a fixed number of lines or up to next blank line
                            end_idx_fallback = start_line_idx + 1
                            for i in range(start_line_idx + 1, min(start_line_idx + 30, len(lines))):
                                if not lines[i].strip():
                                    end_idx_fallback = i
                                    break
                                end_idx_fallback = i + 1
                            element_content_lines = lines[start_line_idx:end_idx_fallback]
                            end_line_idx = end_idx_fallback
                        
                        element_content_str = '\n'.join(element_content_lines)

                        elements.append(CodeElement(
                            type=element_type,
                            name=name,
                            content=element_content_str,
                            start_line=start_line_idx + 1, # 1-based
                            end_line=end_line_idx,       # 1-based
                            file_path=str(file_path),
                            language=language
                        ))
                        # Note: This simple generic parser doesn't handle overlapping matches well.
                        # A more advanced approach would consume lines once matched.
            except re.error as re_err:
                print(f"⚠️ Regex error in generic pattern for {language}, type {element_type}: {re_err}")
            except Exception as e:
                print(f"⚠️ Error during generic extraction for {language}, type {element_type}: {e}")

        return elements

    # --- ADDED: Helper to get content for a tree-sitter node ---
    def _get_ts_node_content(self, full_content: str, node: tree_sitter.Node) -> str:
        """Extracts the original text content for a tree-sitter node."""
        # tree_sitter.Node.text gives bytes, so decode
        return node.text.decode('utf-8', 'replace')
        # An alternative if you want to reconstruct from lines using start/end points:
        # lines = full_content.splitlines()
        # start_row, start_col = node.start_point
        # end_row, end_col = node.end_point
        # if start_row == end_row:
        #     return lines[start_row][start_col:end_col]
        # else:
        #     node_lines = [lines[start_row][start_col:]]
        #     node_lines.extend(lines[start_row + 1:end_row])
        #     node_lines.append(lines[end_row][:end_col])
        #     return '\n'.join(node_lines)


    def _get_node_content(self, content: str, node: ast.AST) -> str:
        """Extrahiert Content für AST Node"""
        # This works for Python AST nodes.
        # ast.unparse(node) is available in Python 3.9+ and is generally preferred.
        # The original slice-based method is kept for compatibility if needed.
        if hasattr(ast, 'unparse'): # Python 3.9+
            try:
                return ast.unparse(node)
            except Exception: # Fallback if unparse fails for some specific node types
                pass # Fall through to original method

        lines = content.splitlines() # Using splitlines for consistency
        start_line_idx = node.lineno - 1
        
        # getattr(node, 'end_lineno', node.lineno) can be None for some nodes or Python versions
        # Ensure end_lineno is valid
        end_lineno_attr = getattr(node, 'end_lineno', None)
        if end_lineno_attr is None:
            end_lineno_attr = node.lineno # Fallback to single line
        
        end_line_idx = end_lineno_attr -1


        if end_line_idx >= len(lines):
            end_line_idx = len(lines) - 1
        if start_line_idx > end_line_idx : # Should not happen with valid AST nodes
            start_line_idx = end_line_idx


        # Original slicing logic was: lines[start_line:end_line + 1]
        # With 0-based indexing, it's lines[start_line_idx : end_line_idx + 1]
        
        # More robust way considering column offsets for precise content:
        # This reconstructs from line numbers and column offsets.
        # However, _get_node_content is often used to get the whole block.
        # The original line-based slicing is usually sufficient for functions/classes.
        
        # For nodes that span multiple lines:
        # If start_col_offset and end_col_offset are available (Python 3.8+)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', -1) # -1 means to end of line

        if start_line_idx == end_line_idx: # Single line node
            if end_col == -1: # Node goes to end of line
                 return lines[start_line_idx][start_col:]
            return lines[start_line_idx][start_col:end_col]
        else: # Multi-line node
            # Extract relevant lines and join them
            extracted_lines = []
            # First line: from col_offset to end
            extracted_lines.append(lines[start_line_idx][start_col:])
            # Middle lines (if any): full lines
            for i in range(start_line_idx + 1, end_line_idx):
                extracted_lines.append(lines[i])
            # Last line: from start to end_col_offset
            if end_col == -1:
                extracted_lines.append(lines[end_line_idx])
            else:
                extracted_lines.append(lines[end_line_idx][:end_col])
            return '\n'.join(extracted_lines)


    def _get_decorator_name(self, decorator: ast.AST) -> str:
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            # Recursively get the full attribute path, e.g., "module.submodule.decorator_attr"
            # For simplicity, just returning the final attribute name.
            # To get full path:
            # value = decorator.value
            # parts = [decorator.attr]
            # while isinstance(value, ast.Attribute):
            #     parts.append(value.attr)
            #     value = value.value
            # if isinstance(value, ast.Name):
            #     parts.append(value.id)
            # return '.'.join(reversed(parts))
            return decorator.attr # Original behavior
        elif isinstance(decorator, ast.Call): # Decorator with arguments, e.g. @my_decorator(arg1)
            # Return the name of the function being called as the decorator
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                # Similar to above, could get full path. For now, just attr.
                return decorator.func.attr
        return "unknown_decorator" # --- More descriptive unknown ---

    def _find_block_end(self, lines: List[str], start_line_idx: int) -> int: # --- Param is 0-based index ---
        """Finds the end of a code block (e.g. JS/Java) based on brace counting."""
        brace_count = 0
        in_string_literal = False # --- Renamed for clarity ---
        string_char_type = None # --- Renamed for clarity ---
        in_line_comment = False
        in_block_comment = False

        first_brace_found = False

        for i in range(start_line_idx, len(lines)):
            line_text = lines[i]
            # Reset line comment state for each new line
            in_line_comment = False

            for char_idx, char_code in enumerate(line_text):
                # Handle block comments first
                if not in_string_literal and not in_line_comment:
                    if char_code == '/' and char_idx + 1 < len(line_text) and line_text[char_idx + 1] == '*':
                        in_block_comment = True
                        continue # Move to next character
                    if char_code == '*' and char_idx + 1 < len(line_text) and line_text[char_idx + 1] == '/':
                        if in_block_comment:
                            in_block_comment = False
                        continue # Move to next character
                
                if in_block_comment:
                    continue

                # Handle line comments
                if not in_string_literal and char_code == '/' and char_idx + 1 < len(line_text) and line_text[char_idx + 1] == '/':
                    in_line_comment = True
                    break # Rest of the line is a comment

                if in_line_comment:
                    continue # Skip characters in line comment

                # Handle string literals
                if char_code in ['"', "'", '`'] and (char_idx == 0 or line_text[char_idx - 1] != '\\'): # Basic escape check
                    if not in_string_literal:
                        in_string_literal = True
                        string_char_type = char_code
                    elif string_char_type == char_code:
                        in_string_literal = False
                        string_char_type = None
                    # Not an 'else' here, as a string char could be the one we are looking for inside another string type (e.g. '`' inside "'").
                    # This simple logic doesn't handle nested different string types perfectly but is okay for brace counting.
                    
                if in_string_literal:
                    continue # Skip characters inside strings

                # Brace counting
                if char_code == '{':
                    brace_count += 1
                    if not first_brace_found:
                        first_brace_found = True
                elif char_code == '}':
                    brace_count -= 1
                    if first_brace_found and brace_count == 0:
                        return i + 1 # Return the line number (0-based) *after* the closing brace line.
            
        # Fallback: if block end not found (e.g. syntax error, or very long block)
        # Return a reasonable limit or end of file.
        # The original fallback was min(start_line + 50, len(lines))
        # With 0-based index, it would be min(start_line_idx + 50, len(lines))
        # Returning len(lines) means "up to the end of the file" if no proper end found.
        if first_brace_found: # If we started counting but didn't find the end
            print(f"⚠️ Unmatched braces starting around line {start_line_idx + 1}. Block end determination might be inaccurate.")
        
        return min(start_line_idx + 50, len(lines)) # Returns 0-based index for slicing (exclusive end)


    async def _extract_imports(self, content: str, language: str) -> List[str]:
        imports: Set[str] = set() # --- Use set for automatic deduplication ---

        if language == "python":
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module_name = node.module if node.module else "" # Handle "from . import foo"
                        if node.level > 0: # Relative import
                            module_name = "." * node.level + module_name
                        
                        for alias in node.names:
                            if alias.name == "*":
                                imports.add(f"{module_name}.*")
                            else:
                                imports.add(f"{module_name}.{alias.name}" if module_name else alias.name)
            except SyntaxError: # Fallback for syntax errors preventing AST parsing
                # print(f"Syntax error in Python, falling back to regex for imports.")
                import_patterns = [
                    r'^\s*import\s+([\w.]+)',              # import foo, import foo.bar
                    r'^\s*from\s+([\w.]+)\s+import\s+'     # from foo import ..., from foo.bar import ...
                ]
                for line in content.splitlines():
                    for pattern in import_patterns:
                        match = re.match(pattern, line)
                        if match:
                            # This regex is very basic, doesn't capture specific imported names from "from ... import ..."
                            # For "from X import Y, Z", it would add X.
                            # For "import X as Y", it adds X.
                            # The AST method is much better.
                            imports.add(match.group(1))


        elif language in ["javascript", "typescript"]:
            # More robust regex for JS/TS imports, including type imports and dynamic imports
            patterns = [
                r'import\s+(?:type\s+)?(?:.*?from\s+)?[\'"]([^\'"@][^\'"]*)[\'"]', # import ... from 'module', import type ... from 'module'
                r'import\s*\(?\s*[\'"]([^\'"@][^\'"]*)[\'"]\s*\)?', # import('module')
                r'require\s*\(\s*[\'"]([^\'"@][^\'"]*)[\'"]\s*\)',  # require('module')
                r'export\s+.*?from\s+[\'"]([^\'"@][^\'"]*)[\'"]' # export ... from 'module'
            ]
            for pattern_str in patterns:
                try:
                    # Find all non-overlapping matches
                    matches = re.findall(pattern_str, content)
                    for match_item in matches:
                        # re.findall returns list of strings if one group, list of tuples if multiple
                        # Ensure we handle this correctly based on actual regex structure
                        mod_name = match_item if isinstance(match_item, str) else match_item[0] # Assuming first group is the module name
                        if mod_name and not mod_name.startswith(('.', '/')): # Filter out relative paths for this list, or normalize them
                             imports.add(mod_name)
                except re.error as e:
                    print(f"Regex error for JS/TS import pattern: {e}")


        elif language == "java":
            # Java import: import package.class; or import package.*;
            pattern = r'^\s*import\s+(static\s+)?([\w.]+\*?);'
            for line in content.splitlines():
                match = re.match(pattern, line)
                if match:
                    imports.add(match.group(2)) # Group 2 is the package/class path

        # For other languages, specific regex might be needed or use tree-sitter if available
        elif language in self.tree_sitter_parsers:
            parser_info = self.tree_sitter_parsers[language]
            parser = parser_info["parser"]
            ts_lang_obj = parser_info["language_object"]
            tree = parser.parse(bytes(content, "utf8"))
            
            # Example import queries (highly language-specific)
            import_query_scm = ""
            if language == "go":
                import_query_scm = '(import_spec_list (import_spec path: (string_literal) @import.path) .)'
            elif language == "rust":
                import_query_scm = '(use_declaration (use_path (identifier) @import.path))' # Simplified
            # ... add queries for other tree-sitter supported languages

            if import_query_scm:
                try:
                    query = ts_lang_obj.query(import_query_scm)
                    captures = query.captures(tree.root_node)
                    for node, capture_name in captures:
                        if capture_name == "import.path":
                            import_path = node.text.decode('utf-8').strip('"')
                            imports.add(import_path)
                except Exception as e:
                    print(f"Error extracting imports with tree-sitter for {language}: {e}")

        return list(imports)


    async def _calculate_complexity(self, content: str, file_path_str: str, language: str) -> float:
        # lizard needs a file path, but analyze_source_code can take content directly.
        # However, lizard's primary interface analyze_file expects a path.
        # Let's use analyze_source_code.
        try:
            # Ensure the language is supported by lizard or map it
            lizard_lang_map = {
                "cpp": "cpp", "c": "c", "java": "java",
                "javascript": "javascript", "python": "python",
                "swift": "swift", "objectivec": "objectivec", # example mappings
                "go": "golang", "rust": "rust", "php": "php", "ruby": "ruby",
                "scala": "scala", "csharp": "csharp" # C# is 'cs' in lizard usually
            }
            lizard_language = lizard_lang_map.get(language.lower())

            # analyze_source_code(filename, code)
            # The filename is used for context in lizard's output but content is primary.
            analysis = lizard.analyze_source_code(file_path_str, content) # lizard_language is auto-detected by lizard from filename or content.

            if analysis and analysis.function_list:
                total_ccn = sum(func.cyclomatic_complexity for func in analysis.function_list)
                avg_ccn = total_ccn / len(analysis.function_list)
                # You could also consider NLOC (lines of code without comments/blanks) or token count from lizard
                # For now, returning average cyclomatic complexity.
                return avg_ccn
            else: # No functions found or analysis failed to produce function list
                # Fallback: simple line-based complexity if lizard fails or finds no functions
                # This is the original fallback logic.
                lines = content.splitlines()
                complexity_keywords = ['if', 'else', 'for', 'while', 'try', 'catch', 'switch', 'case', '&&', '||']
                complexity_score = 1.0 # Base complexity
                for line in lines:
                    line_lower = line.lower().strip()
                    if not line_lower or line_lower.startswith(("//", "#", "/*", "*", "'''", '"""')): # Skip comments/empty
                        continue
                    complexity_score += sum(1 for indicator in complexity_keywords if indicator in line_lower)
                
                # Normalize or cap the score. This heuristic is very rough.
                # Original: return max(1.0, complexity / 10)
                return max(1.0, complexity_score / 20 if len(lines) > 0 else 1.0) # Slightly adjusted normalization
        except Exception as e:
            print(f"⚠️ Complexity analysis failed for {file_path_str} ({language}): {e}")
            # Fallback from original code if lizard errors out completely
            lines = content.splitlines()
            complexity_indicators = ['if', 'else', 'for', 'while', 'try', 'catch', 'switch', 'case']
            complexity = sum(1 for line in lines if any(indicator in line.lower() for indicator in complexity_indicators))
            return max(1.0, complexity / 10.0 if len(lines) > 0 else 1.0)


    async def _extract_file_documentation(self, content: str, language: str) -> Optional[str]:
        lines = content.splitlines()
        doc_lines: List[str] = []

        if language == "python":
            try:
                tree = ast.parse(content)
                module_docstring = ast.get_docstring(tree, clean=False) # Get raw docstring
                if module_docstring:
                    return module_docstring
            except SyntaxError: # If AST parsing fails, try regex/comment parsing
                pass
            except Exception as e_ast_doc:
                print(f"Error extracting Python module docstring via AST: {e_ast_doc}")


        # Generic header comment extraction (first 20 lines)
        # This is a common pattern for file-level documentation.
        in_comment_block = False
        comment_block_type: Optional[str] = None # "triple_quote", "slash_star", "hash"

        for line_num, line_text in enumerate(lines[:min(20, len(lines))]): # Check up to first 20 lines
            stripped_line = line_text.strip()

            if not in_comment_block:
                # Python triple quotes
                if language == "python" and (stripped_line.startswith('"""') or stripped_line.startswith("'''")):
                    in_comment_block = True
                    comment_block_type = "triple_quote"
                    # Handle single-line triple-quote docstring
                    if (stripped_line.endswith('"""') and stripped_line.startswith('"""') and len(stripped_line) > 5) or \
                       (stripped_line.endswith("'''") and stripped_line.startswith("'''") and len(stripped_line) > 5):
                        doc_lines.append(stripped_line[3:-3])
                        in_comment_block = False # Closed on the same line
                        break # Assume this is the main file doc
                    else: # Multi-line started
                        doc_lines.append(stripped_line[3:])
                        continue
                # C-style block comments /* ... */
                elif stripped_line.startswith('/*') and language in ["javascript", "typescript", "java", "c", "cpp", "go", "rust", "swift", "kotlin", "php", "scala"]:
                    in_comment_block = True
                    comment_block_type = "slash_star"
                    if stripped_line.endswith('*/') and len(stripped_line) > 3: # Single line /* ... */
                        doc_lines.append(stripped_line[2:-2].strip())
                        in_comment_block = False
                        break
                    else: # Multi-line started
                        doc_lines.append(stripped_line[2:].strip())
                        continue
                # Hash comments (Python, Ruby, Shell, R, Perl)
                elif stripped_line.startswith('#') and not stripped_line.startswith('#!'): # Avoid shebang
                     if language in ["python", "ruby", "shell", "r", "perl", "dockerfile"]: # Dockerfile uses # for comments too
                        # Start a block of hash comments if consecutive
                        in_comment_block = True
                        comment_block_type = "hash"
                        doc_lines.append(stripped_line[1:].strip())
                        continue
                # Line comments // (JS, Java, C++, C#, Go, Rust, Swift, Kotlin, Scala, PHP)
                elif stripped_line.startswith('//') and language in ["javascript", "typescript", "java", "c", "cpp", "go", "rust", "swift", "kotlin", "php", "scala", "csharp"]:
                    in_comment_block = True # Treat consecutive // as a block
                    comment_block_type = "double_slash"
                    doc_lines.append(stripped_line[2:].strip())
                    continue

                # If not starting a new block and not in one, and line is not empty, likely past header comments
                if stripped_line and not in_comment_block:
                    break # Stop if we hit actual code or non-comment content unless already in a block

            elif in_comment_block:
                if comment_block_type == "triple_quote":
                    if (stripped_line.endswith('"""') and lines[line_num-1].strip().startswith('"""') and not (lines[line_num-1].strip().endswith('"""') and len(lines[line_num-1].strip()) > 5) ) or \
                       (stripped_line.endswith("'''") and lines[line_num-1].strip().startswith("'''") and not (lines[line_num-1].strip().endswith("'''") and len(lines[line_num-1].strip()) > 5) ): # End of multi-line
                        doc_lines.append(stripped_line[:-3])
                        in_comment_block = False
                        break
                    else:
                        doc_lines.append(stripped_line)
                elif comment_block_type == "slash_star":
                    if stripped_line.endswith('*/'):
                        doc_lines.append(stripped_line[:-2].strip())
                        in_comment_block = False
                        break
                    else:
                        # Remove leading " * " or " *" if present (common Javadoc/Doxygen style)
                        clean_line = re.sub(r"^\s*\*\s?", "", stripped_line)
                        doc_lines.append(clean_line)
                elif comment_block_type == "hash":
                    if stripped_line.startswith('#'):
                        doc_lines.append(stripped_line[1:].strip())
                    else: # Non-hash line, block ends
                        in_comment_block = False
                        break 
                elif comment_block_type == "double_slash":
                    if stripped_line.startswith('//'):
                        doc_lines.append(stripped_line[2:].strip())
                    else: # Non-// line, block ends
                        in_comment_block = False
                        break
            
            # If we have collected some doc lines and encounter an empty line, assume header block ended.
            if doc_lines and not stripped_line and not in_comment_block: # Check in_comment_block to allow empty lines inside multiline comments
                break

        documentation = '\n'.join(doc_lines).strip()
        return documentation if documentation else None


    async def _extract_dependencies(self, content: str, language: str, file_path: Path) -> List[str]:
        # This method should ideally return external package dependencies, not just local imports.
        # The current _extract_imports is more about local/module imports.
        # For true package dependencies, one would need to parse build files (package.json, requirements.txt, pom.xml, etc.)
        # This function can extend the results of _extract_imports with commented dependencies or simple patterns.
        
        dependencies: Set[str] = set() # Use set for deduplication

        # Include imports as a starting point (these are often module names, sometimes packages)
        imports = await self._extract_imports(content, language)
        for imp_item in imports:
            # Heuristic: if import is not relative and not a stdlib module (hard to check robustly here)
            # it might be an external dependency.
            # For Python, one could try to filter out stdlib names.
            # For JS, non-relative paths are usually packages.
            if not imp_item.startswith(".") and "/" not in imp_item: # Very basic filter
                 # Further filtering could be done against a list of known stdlib modules for the language.
                dependencies.add(imp_item.split('.')[0]) # Take the base package name often


        # Language-specific dependency hints from comments or specific keywords
        if language == "python":
            # Look for "requires" in comments (e.g., # requires: requests, numpy)
            req_pattern = r'#\s*(?:requires?|requirement|dependency|depends? on):\s*([\w\s.,\[\]\-=<>~]+)'
            comment_matches = re.findall(req_pattern, content, re.IGNORECASE)
            for match_str in comment_matches:
                # Split by comma, remove version specifiers for simplicity for now
                potential_deps = [re.split(r'[<>=~\[\]\s]', dep.strip())[0] for dep in match_str.split(',')]
                dependencies.update(d for d in potential_deps if d)
        
        elif language in ["javascript", "typescript"]:
            # package.json is the source of truth, but this looks for inline hints
            # Could look for comments like `// depends: lodash`
            # Or try to infer from known CDN links or script tags (if analyzing HTML context, not here)
            pass # No simple in-code patterns added for now beyond imports

        elif language == "java":
            # Maven/Gradle files are the source of truth.
            # Could look for comments like `// Maven: groupId:artifactId:version`
            pass # No simple in-code patterns added for now beyond imports

        return list(dependencies)


    async def _generate_element_embeddings(self, elements: List[CodeElement]) -> None:
        if not elements or not self.embedding_engine or not self.config.generate_embeddings:
            return

        try:
            texts_to_embed = []
            for element in elements:
                # Prepare text for embedding: combine relevant info
                # Shorter, more focused text can be better than full content sometimes.
                text_parts = [
                    f"Element Type: {element.type}",
                    f"Name: {element.name}",
                    f"Language: {element.language}",
                ]
                if element.docstring:
                    text_parts.append(f"Documentation: {element.docstring[:250]}") # Limit docstring length
                if element.parameters:
                    text_parts.append(f"Parameters: {', '.join(element.parameters)}")
                if element.return_type:
                     text_parts.append(f"Returns: {element.return_type}")
                
                # Add a snippet of the content
                content_snippet = element.content.strip()
                # Take first few lines and last few lines, or a middle chunk, to represent the element.
                # Max 500 chars for content snippet.
                if len(content_snippet) > 500:
                    first_lines = "\n".join(content_snippet.splitlines()[:5])
                    content_snippet = first_lines + "\n..." if len(first_lines) < 450 else first_lines[:450]+"..."

                text_parts.append(f"Code Snippet: {content_snippet}")
                
                texts_to_embed.append('\n---\n'.join(text_parts)) # Use a clear separator

            if not texts_to_embed:
                return

            # Generate embeddings in batches (MLXEmbeddingEngine handles batching internally if configured)
            embedding_result = await self.embedding_engine.embed(texts_to_embed)

            if embedding_result and embedding_result.embeddings:
                for i, element in enumerate(elements):
                    if i < len(embedding_result.embeddings):
                        # Ensure embedding is a flat list of floats
                        raw_embedding = embedding_result.embeddings[i]
                        if hasattr(raw_embedding, 'tolist'): # For numpy arrays or similar
                            element.embedding = raw_embedding.tolist()
                        elif isinstance(raw_embedding, list):
                            element.embedding = raw_embedding
                        else:
                            print(f"⚠️ Unexpected embedding type for {element.name}: {type(raw_embedding)}")
            else:
                print(f"⚠️ Embedding generation returned no results for {len(texts_to_embed)} elements.")

        except Exception as e:
            print(f"⚠️ Error generating element embeddings: {e}")
            # import traceback
            # traceback.print_exc()


    async def _build_dependency_graph(self, files: List[CodeFile]) -> Dict[str, List[str]]:
        dependency_graph: Dict[str, Set[str]] = {} # Use set for unique dependencies
        
        # Create a mapping of all symbols (classes, functions) available in the project by file
        # This helps resolve if an import refers to an internal project file.
        available_project_symbols: Dict[str, Set[str]] = {} # file_path -> set of symbol names
        file_path_to_lang: Dict[str, str] = {} # file_path -> language

        for code_file in files:
            file_path_to_lang[code_file.file_path] = code_file.language
            symbols_in_file: Set[str] = set()
            for element in code_file.elements:
                if element.type in ["class", "function", "method", "struct", "interface", "enum", "constant", "type"]: # Add relevant types
                    symbols_in_file.add(element.name)
            available_project_symbols[code_file.file_path] = symbols_in_file


        for code_file in files:
            file_deps: Set[str] = set()
            current_file_path = code_file.file_path
            current_lang = code_file.language

            # 1. Dependencies from explicit `file.dependencies` (e.g., from comment parsing)
            # These are usually external package names.
            if code_file.dependencies:
                file_deps.update(f"pkg:{dep}" for dep in code_file.dependencies) # Prefix to mark as package

            # 2. Dependencies from `file.imports`
            for import_name_full in code_file.imports:
                # Try to resolve if this import refers to another file in the project
                resolved_internal_dependency = False
                
                # Python: import_name_full can be "module.submodule.ClassName" or "module"
                # Java: "com.example.MyClass" or "com.example.*"
                # JS/TS: "./relative/file" or "package_name"

                # Simple check for relative paths in JS/TS (already handled by _extract_imports if kept as is)
                if current_lang in ["javascript", "typescript"] and (import_name_full.startswith("./") or import_name_full.startswith("../")):
                    try:
                        # Resolve relative path against current file's directory
                        # file_path is relative to repo_root. Path(file_path).parent should work.
                        abs_imported_path = (Path(repo_root) / Path(current_file_path).parent / import_name_full).resolve()
                        # Normalize to relative path from repo_root again
                        relative_imported_path_str = str(abs_imported_path.relative_to(Path(repo_root))) + ( ".js" if not Path(import_name_full).suffix else "" ) # common auto-suffix in JS

                        if relative_imported_path_str in available_project_symbols:
                            file_deps.add(relative_imported_path_str)
                            resolved_internal_dependency = True
                    except Exception: # Path resolution error
                        pass # Might be an alias or non-file path

                if resolved_internal_dependency:
                    continue

                # General check: does the import name match a known file or a symbol within a known file?
                # This is complex due to language-specific module resolution rules.
                # Example: Python's `import my_module` could refer to `my_module.py` or `my_module/__init__.py`
                # Example: Java's `import com.example.Foo`
                
                # Simplified check: if import_name matches a file (without extension or with common ones)
                # Or if a part of the import path matches a file that exports the rest.
                # This needs to be more language specific.
                
                import_parts = import_name_full.split('.') # For Java/Python-like imports
                potential_module_or_class_name = import_parts[-1] # e.g. ClassName from com.example.ClassName

                for other_file_path, symbols_in_other_file in available_project_symbols.items():
                    if other_file_path == current_file_path:
                        continue # Don't depend on self

                    # Check 1: Direct file import (e.g. Python's `import file_b`)
                    # Path(other_file_path).stem == import_name_full
                    if Path(other_file_path).stem == import_name_full: # Compare 'my_module' with 'my_module.py'
                        file_deps.add(other_file_path)
                        resolved_internal_dependency = True
                        break
                    
                    # Check 2: Symbol import (e.g. Python `from my_module import MyClass`)
                    # Here, import_name_full might be "my_module.MyClass" or just "MyClass" (if module context is known)
                    # The `code_file.imports` list should ideally contain fully qualified names where possible.
                    if potential_module_or_class_name in symbols_in_other_file:
                        # Further check if the module path matches `other_file_path`
                        # e.g. if import is "moduleA.ClassB", and other_file_path is "moduleA.py" (or "moduleA/__init__.py")
                        # This part is tricky and needs proper module resolution logic per language.
                        # For now, a simpler association if a symbol matches:
                        if len(import_parts) > 1: # e.g. "moduleA.ClassB"
                            module_path_from_import = ".".join(import_parts[:-1])
                            if Path(other_file_path).stem == module_path_from_import:
                                file_deps.add(other_file_path)
                                resolved_internal_dependency = True
                                break
                        elif len(import_parts) == 1 and language not in ["java"]: # Single name import, e.g. import MyUtility (JS)
                             # If 'MyUtility' is in symbols_in_other_file, assume it's from there.
                             # This can lead to false positives if symbol names are not unique.
                             # Only add if it's a strong convention for the language.
                             # In JS, `import { MyUtility } from './utilityFile'` - utilityFile is the dependency.
                             # The `imports` list should contain './utilityFile' in that case.
                             pass


                if not resolved_internal_dependency and not import_name_full.startswith(".") and "/" not in import_name_full:
                    # If not resolved internally and looks like a package name, add it as a package dependency
                    file_deps.add(f"pkg:{import_name_full.split('.')[0]}") # Add base package name


            dependency_graph[current_file_path] = set(file_deps) # Store as set initially

        # Convert sets to lists for the final graph
        final_graph: Dict[str, List[str]] = {file: sorted(list(deps)) for file, deps in dependency_graph.items()}
        return final_graph


    async def _extract_readme(self, repo_path: Path) -> Optional[str]:
        readme_filenames = ["README.md", "README.rst", "README.txt", "README", "readme.md"] # Common variations
        for readme_name in readme_filenames:
            readme_file_path = repo_path / readme_name
            if readme_file_path.exists() and readme_file_path.is_file():
                try:
                    if self.document_processor and self.config.extract_documentation: # Ensure doc proc is enabled
                        # Use document processor for potentially richer extraction (e.g. parsing Markdown)
                        # The placeholder ProcessedDocument currently just returns content.
                        processed_doc = await self.document_processor.process_document(readme_file_path)
                        return processed_doc.content
                    else:
                        # Simple text extraction
                        async with aiofiles.open(readme_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            return await f.read()
                except Exception as e:
                    print(f"⚠️ Error reading {readme_name} from {repo_path}: {e}")
                    # Fallback to simple read if processor fails
                    try:
                        async with aiofiles.open(readme_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            return await f.read()
                    except Exception as e_fallback:
                        print(f"⚠️ Fallback read error for {readme_name}: {e_fallback}")
        return None


    async def _generate_structure_summary(self, files: List[CodeFile], repo_path: Path) -> str:
        # This creates a Markdown summary.
        summary_parts = [
            f"# Repository Structure Summary: {repo_path.name}",
            "",
            "## Overview",
            f"- **Total Files Analyzed:** {len(files)}",
            f"- **Total Lines of Code:** {sum(f.line_count for f in files):,}",
        ]
        if files: # Avoid division by zero
            avg_complexity = sum(f.complexity_score for f in files) / len(files) if len(files) > 0 else 0
            summary_parts.append(f"- **Average File Complexity (CCN):** {avg_complexity:.2f}")
        summary_parts.append("")


        # Language distribution
        lang_dist: Dict[str, int] = {}
        for code_file in files:
            lang_dist[code_file.language] = lang_dist.get(code_file.language, 0) + 1

        if lang_dist:
            summary_parts.append("## Language Distribution")
            for lang, count in sorted(lang_dist.items(), key=lambda item: item[1], reverse=True):
                percentage = (count / len(files)) * 100 if len(files) > 0 else 0
                summary_parts.append(f"- **{lang.title() if lang else 'Unknown'}:** {count} files ({percentage:.1f}%)")
            summary_parts.append("")

        # Top-level directory structure (summary)
        summary_parts.append("## Key Directories (Top 5 by file count)")
        dir_file_counts: Dict[str, int] = {}
        max_depth = 2 # Limit directory depth for summary (e.g., src/, src/utils/)

        for code_file in files:
            # file.file_path is relative to repo_root
            file_path_obj = Path(code_file.file_path)
            # Get parent directories up to max_depth
            for i in range(min(max_depth, len(file_path_obj.parts) -1 )): # -1 because last part is filename
                # Construct directory path with current depth
                dir_path_parts = file_path_obj.parts[:i+1]
                dir_path_str = str(Path(*dir_path_parts)) + "/" # Ensure trailing slash for dirs
                dir_file_counts[dir_path_str] = dir_file_counts.get(dir_path_str, 0) + 1
        
        # If no subdirectories, list root files or indicate flat structure
        if not dir_file_counts and files:
             summary_parts.append("- Repository has a flat structure or files are primarily in the root.")
        elif dir_file_counts:
            # Sort directories by file count, descending, take top 5
            sorted_dirs = sorted(dir_file_counts.items(), key=lambda item: item[1], reverse=True)[:5]
            for dir_name, count in sorted_dirs:
                summary_parts.append(f"- `{dir_name}` - approx. {count} relevant files recursively")
        summary_parts.append("")


        # Key files identification (e.g., most complex, largest)
        summary_parts.append("## Noteworthy Files")
        # Most complex files (top 3)
        if files:
            complex_files = sorted(files, key=lambda f: f.complexity_score, reverse=True)[:3]
            if complex_files and complex_files[0].complexity_score > 0 : # Only show if meaningful complexity
                summary_parts.append("### Most Complex Files (by average CCN):")
                for code_file in complex_files:
                    if code_file.complexity_score > 0: # Threshold to be "noteworthy"
                        summary_parts.append(f"- `{code_file.file_path}` (Complexity: {code_file.complexity_score:.1f})")

            # Largest files (top 3 by line count)
            large_files = sorted(files, key=lambda f: f.line_count, reverse=True)[:3]
            if large_files and large_files[0].line_count > 50: # Arbitrary threshold for "large"
                summary_parts.append("### Largest Files (by lines):")
                for code_file in large_files:
                     if code_file.line_count > 50:
                        summary_parts.append(f"- `{code_file.file_path}` ({code_file.line_count:,} lines)")
        summary_parts.append("")

        return '\n'.join(summary_parts)


    async def save_to_vector_store(self,
                                   analysis: RepositoryAnalysis,
                                   vector_store: MLXVectorStore, # Type hint if available
                                   user_id: str,
                                   model_id: str = "gte-small") -> bool: # Default was gte-small
        if not self.config.generate_embeddings or not self.embedding_engine:
            print("⚠️ Embeddings not generated or engine not available, skipping save to vector store.")
            return False
        if not vector_store:
             print("⚠️ Vector store not provided, skipping save.")
             return False

        try:
            all_vectors: List[List[float]] = []
            all_metadata: List[Dict[str, Any]] = []

            # 1. Repository-level summary embedding
            repo_summary_text_parts = [
                f"Repository Name: {analysis.project_name}",
                f"Path: {analysis.repo_path}",
                f"Total Files: {len(analysis.files)}",
                f"Total Lines of Code: {analysis.total_lines:,}",
                f"Languages: {', '.join(analysis.language_distribution.keys())}",
                f"Average File Complexity: {analysis.total_complexity / len(analysis.files):.2f}" if analysis.files else "N/A",
                f"Documentation Coverage: {analysis.documentation_coverage:.1f}%",
            ]
            if analysis.readme_content:
                repo_summary_text_parts.append(f"\nREADME Summary:\n{analysis.readme_content[:1000]}...") # Truncate README
            if analysis.structure_summary:
                 repo_summary_text_parts.append(f"\nStructure Summary:\n{analysis.structure_summary[:1000]}")

            repo_full_summary_text = "\n".join(repo_summary_text_parts)

            repo_embedding_result = await self.embedding_engine.embed([repo_full_summary_text])
            if repo_embedding_result and repo_embedding_result.embeddings:
                repo_emb = repo_embedding_result.embeddings[0]
                all_vectors.append(repo_emb.tolist() if hasattr(repo_emb, 'tolist') else repo_emb)
                all_metadata.append({
                    "type": "repository_summary",
                    "repository_name": analysis.project_name,
                    "text_content": repo_full_summary_text, # Store the text used for embedding
                    "total_files": len(analysis.files),
                    "total_lines": analysis.total_lines,
                    "languages": list(analysis.language_distribution.keys()),
                    "avg_complexity": analysis.total_complexity / len(analysis.files) if analysis.files else 0,
                    "doc_coverage": analysis.documentation_coverage
                })

            # 2. Code File Summaries (optional, can be verbose for vector store)
            # Instead of file summaries, focus on CodeElements as they are more granular.

            # 3. Code Element embeddings
            for code_file in analysis.files:
                for element in code_file.elements:
                    if element.embedding: # Check if embedding was generated
                        all_vectors.append(element.embedding) # Already a list of floats

                        element_summary_for_retrieval = f"""
                        Type: {element.type}
                        Name: {element.name}
                        Language: {element.language}
                        File: {element.file_path}
                        Lines: {element.start_line}-{element.end_line}
                        {"Documentation: " + element.docstring[:200] + "..." if element.docstring else "No documentation."}
                        Code Snippet: {element.content[:300]}... 
                        """ # Keep text concise for metadata / retrieval matching

                        all_metadata.append({
                            "type": "code_element",
                            "element_type": element.type,
                            "element_name": element.name,
                            "file_path": element.file_path, # Relative path
                            "language": element.language,
                            "repository_name": analysis.project_name,
                            "text_content": element_summary_for_retrieval, # Textual summary
                            # Keep original content out of metadata unless small or specifically needed for direct display post-retrieval
                            # "full_content": element.content, # Can be large
                            "start_line": element.start_line,
                            "end_line": element.end_line,
                            "complexity": element.complexity if element.complexity is not None else -1, # Use -1 if not applicable
                            "has_docstring": bool(element.docstring)
                        })
            
            if not all_vectors:
                print(f"⚠️ No embeddings generated or found for repository {analysis.project_name}. Nothing to save to vector store.")
                return False

            # Define a namespace for this repository's data in the vector store
            # Namespace should be unique per repository and user if needed
            # Example: code_analysis_user123_my-repo-name
            # Sanitize project name for namespace
            sanitized_project_name = re.sub(r'[^a-zA-Z0-9_-]', '_', analysis.project_name)
            namespace = f"code_analysis_{user_id}_{sanitized_project_name}"
            
            print(f"Attempting to save {len(all_vectors)} vectors to namespace: {namespace}")

            # The vector_store.add_vectors might need to be called in batches if number of vectors is very large.
            # Assuming MLXVectorStore handles internal batching or is robust to large inputs.
            success = await vector_store.add_vectors(
                user_id=user_id,
                model_id=self.config.embedding_model, # Use the model that generated these embeddings
                vectors=all_vectors,
                metadata=all_metadata,
                namespace=namespace
            )

            if success:
                print(f"✅ Successfully saved {len(all_vectors)} code analysis vectors to vector store for {analysis.project_name}.")
            else:
                print(f"❌ Failed to save code analysis to vector store for {analysis.project_name}.")
            return success

        except Exception as e:
            print(f"❌ Error saving code analysis to vector store for {analysis.project_name}: {e}")
            # import traceback
            # traceback.print_exc()
            return False


    def get_performance_stats(self) -> Dict[str, Any]:
        avg_analysis_time_per_file = self.total_analysis_time / self.total_files if self.total_files > 0 else 0
        files_per_sec = self.total_files / self.total_analysis_time if self.total_analysis_time > 0 else 0
        elements_per_sec = self.total_elements / self.total_analysis_time if self.total_analysis_time > 0 else 0

        # Aggregate language stats if collected during analysis (not currently implemented in this version)
        # For example, `self.language_stats` could store { "python": {"files": N, "time": T}, ... }

        return {
            "total_repositories_analyzed": 1 if self.total_files > 0 else 0, # Assuming stats are per instance lifetime, might be reset
            "total_files_analyzed": self.total_files,
            "total_code_elements_extracted": self.total_elements,
            "total_analysis_duration_seconds": round(self.total_analysis_time, 3),
            "average_time_per_file_seconds": round(avg_analysis_time_per_file, 4),
            "files_processed_per_second": round(files_per_sec, 2),
            "elements_extracted_per_second": round(elements_per_sec, 2),
            "language_processing_details": self.language_stats if self.language_stats else "Not collected",
            "configured_supported_languages": self.config.supported_languages,
            "configured_complexity_thresholds": self.complexity_thresholds,
            "embedding_model_configured": self.config.embedding_model if self.config.generate_embeddings else "Embeddings disabled"
        }


    async def export_analysis(self,
                              analysis: RepositoryAnalysis,
                              output_path: Union[str, Path],
                              output_format: str = "json") -> bool: # --- Renamed format to output_format ---
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        try:
            if output_format.lower() == "json":
                # Prepare data for JSON serialization (handle Path objects, complex types)
                # Using asdict for dataclasses is a good start.
                # Need to ensure all fields are JSON serializable.
                
                # Custom encoder for Path objects or other non-serializable types if any
                class CustomEncoder(json.JSONEncoder):
                    def default(self, o):
                        if isinstance(o, Path):
                            return str(o)
                        if isinstance(o, set): # For dependency_graph if it uses sets internally
                            return list(o)
                        # Let the base class default method raise the TypeError
                        return json.JSONEncoder.default(self, o)

                # Convert RepositoryAnalysis and its nested dataclasses to dicts
                export_data = asdict(analysis)
                
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    # Use ensure_ascii=False for proper UTF-8 output of special characters
                    await f.write(json.dumps(export_data, indent=2, cls=CustomEncoder, ensure_ascii=False))

            elif output_format.lower() == "markdown":
                # Generate a Markdown report
                md_content = [
                    f"# Code Analysis Report: {analysis.project_name}",
                    f"**Analyzed on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"**Repository Path:** `{analysis.repo_path}`",
                    f"**Total Analysis Time:** {analysis.analysis_time:.2f} seconds",
                    "",
                    "## I. Overall Summary",
                    f"- **Total Files Analyzed:** {len(analysis.files)}",
                    f"- **Total Lines of Code (LOC):** {analysis.total_lines:,}",
                    f"- **Average File Complexity (CCN):** {analysis.total_complexity / len(analysis.files):.2f}" if analysis.files else "N/A",
                    f"- **Overall Documentation Coverage:** {analysis.documentation_coverage:.1f}%",
                    "",
                    "## II. Language Distribution",
                ]
                for lang, count in sorted(analysis.language_distribution.items(), key=lambda item: item[1], reverse=True):
                    percentage = (count / len(analysis.files)) * 100 if analysis.files else 0
                    md_content.append(f"- **{lang.title()}:** {count} files ({percentage:.1f}%)")
                
                md_content.extend(["", "## III. Repository Structure Summary"])
                if analysis.structure_summary:
                    # Assuming structure_summary is already in Markdown format from _generate_structure_summary
                    md_content.append(analysis.structure_summary)
                else:
                    md_content.append("No detailed structure summary available.")
                
                md_content.extend(["", "## IV. README Digest"])
                if analysis.readme_content:
                    md_content.append("```text") # Block for README content
                    md_content.append(analysis.readme_content[:1500] + ("..." if len(analysis.readme_content) > 1500 else "")) # Truncate long READMEs
                    md_content.append("```")
                else:
                    md_content.append("No README file found or content extracted.")

                md_content.extend(["", "## V. Dependency Overview (Inter-file)"])
                if analysis.dependency_graph:
                    # Summarize dependencies - e.g., top 5 most depended-on files, or files with most outgoing dependencies
                    dep_counts_incoming = {} # file -> number of other files depending on it
                    for target_file, deps in analysis.dependency_graph.items():
                        for dep_target in deps:
                            if not dep_target.startswith("pkg:"): # Focus on internal dependencies
                                dep_counts_incoming[dep_target] = dep_counts_incoming.get(dep_target,0) + 1
                    
                    if dep_counts_incoming:
                        md_content.append("### Most Referenced Internal Files:")
                        for fname, count in sorted(dep_counts_incoming.items(), key=lambda x:x[1], reverse=True)[:5]:
                            md_content.append(f"- `{fname}` (Referenced by {count} other files)")
                    else:
                        md_content.append("No significant internal file dependencies identified or all are external packages.")
                else:
                    md_content.append("Dependency graph not generated or empty.")

                md_content.extend(["", "## VI. Detailed File Analysis (Highlights)"])
                # Highlight a few key files (e.g., most complex, largest, or entry points if identifiable)
                # For brevity, just show top 2 complex and 2 large files.
                highlight_files = sorted(analysis.files, key=lambda f: f.complexity_score, reverse=True)[:2] + \
                                  sorted(analysis.files, key=lambda f: f.line_count, reverse=True)[:2]
                highlight_files = list(dict.fromkeys(highlight_files)) # Remove duplicates if a file is both complex and large

                for code_file in highlight_files[:4]: # Max 4 highlighted files
                    md_content.extend([
                        f"### File: `{code_file.file_path}`",
                        f"- **Language:** {code_file.language}",
                        f"- **Lines:** {code_file.line_count:,}",
                        f"- **Complexity (CCN):** {code_file.complexity_score:.2f}",
                        f"- **Code Elements Found:** {len(code_file.elements)}",
                        f"- **Imports/Dependencies:** {len(code_file.imports)} local, {len(code_file.dependencies)} inferred external",
                        f"- **File Documentation:** {'Present (snippet below)' if code_file.documentation else 'Not found'}",
                    ])
                    if code_file.documentation:
                        md_content.append(f"  > {code_file.documentation[:200].replacechr(10, ' ')}...") # Show a snippet
                    
                    if code_file.elements:
                        md_content.append("- **Key Elements (examples):**")
                        for el in code_file.elements[:min(3, len(code_file.elements))]: # Show up to 3 elements
                            md_content.append(f"  - `{el.type}: {el.name}` (Lines: {el.start_line}-{el.end_line})")
                    md_content.append("")


                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write('\n'.join(md_content))

            else:
                raise ValueError(f"Unsupported export format: {output_format}. Supported formats: 'json', 'markdown'.")

            print(f"✅ Successfully exported code analysis to {output_path} (Format: {output_format})")
            return True

        except Exception as e:
            print(f"❌ Export error to {output_path}: {e}")
            # import traceback
            # traceback.print_exc()
            return False

    async def benchmark(self, test_repo_path: Optional[Path] = None) -> Dict[str, Any]: # --- Return type Any for dict values ---
        print("🚀 Running Code Analyzer Benchmark...")
        benchmark_results: Dict[str, Any] = {}

        # Use a temporary directory for test repo if none provided
        if test_repo_path is None:
            import tempfile
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="mlx_code_analyzer_benchmark_")
            test_repo_path = Path(temp_dir_obj.name)
            created_temp_repo = True
            print(f"Temporary benchmark repository created at: {test_repo_path}")

            # Create some sample files for benchmarking different languages
            sample_files_content = {
                "main.py": """
import os
# This is a test Python file.
class MyClass:
    def __init__(self, name):
        self.name = name
    def greet(self, message="hello"):
        '''Greets the entity.'''
        if len(message) > 0:
            print(f"{self.name} says: {message}")
        return True
MY_CONSTANT = 123
def top_level_func(x, y): return x + y
                """,
                "utils.js": """
// Test JavaScript file
function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        if (item.price > 0) total += item.price;
    }
    return total;
}
class Helper { constructor() { this.id = Math.random(); } process() { /* noop */ }}
export { calculateTotal, Helper };
                """,
                "core.cpp": """
#include <iostream>
#include <vector>
// A C++ test file
namespace Core {
    class Processor {
    public:
        Processor(int id) : _id(id) {}
        void run(const std::vector<int>& data) {
            if (data.empty()) return;
            for(int val : data) { std::cout << val << std::endl; }
        }
    private:
        int _id;
    };
}
int main_func(int argc, char **argv) { return 0; }
                """,
                 "README.md": """
# Benchmark Test Repository
This is a small repository used for benchmarking the MLXCodeAnalyzer.
It contains a few files in different languages.
                 """
            }
            try:
                for fname, fcontent in sample_files_content.items():
                    file_p = test_repo_path / fname
                    file_p.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(file_p, 'w', encoding='utf-8') as af:
                        await af.write(fcontent)
            except Exception as e_create:
                print(f"Error creating benchmark files: {e_create}")
                if created_temp_repo: temp_dir_obj.cleanup()
                return {"error": f"Failed to create benchmark files: {e_create}"}

        else: # User provided a path
            created_temp_repo = False
            if not Path(test_repo_path).is_dir():
                return {"error": f"Provided test_repo_path is not a valid directory: {test_repo_path}"}
            print(f"Using provided repository for benchmark: {test_repo_path}")


        start_wall_time = time.perf_counter() # More precise for benchmarking
        start_process_time = time.process_time()

        try:
            # Use a dummy user_id for benchmark
            analysis_result = await self.analyze_repository(test_repo_path, user_id="benchmark_user")

            end_wall_time = time.perf_counter()
            end_process_time = time.process_time()

            wall_duration = end_wall_time - start_wall_time
            process_duration = end_process_time - start_process_time

            if analysis_result:
                num_files = len(analysis_result.files)
                num_elements = sum(len(f.elements) for f in analysis_result.files)
                total_loc = analysis_result.total_lines

                benchmark_results = {
                    "repository_path_used": str(test_repo_path),
                    "wall_time_seconds": round(wall_duration, 4),
                    "cpu_process_time_seconds": round(process_duration, 4),
                    "files_analyzed_count": num_files,
                    "code_elements_extracted_count": num_elements,
                    "total_lines_of_code": total_loc,
                    "files_per_wall_second": round(num_files / wall_duration, 2) if wall_duration > 0 else 0,
                    "lines_per_wall_second": round(total_loc / wall_duration, 2) if wall_duration > 0 else 0,
                    "elements_per_wall_second": round(num_elements / wall_duration, 2) if wall_duration > 0 else 0,
                    "average_complexity_ccn": round(analysis_result.total_complexity / num_files, 2) if num_files > 0 else 0,
                    "documentation_coverage_percent": round(analysis_result.documentation_coverage, 1),
                    "language_distribution_found": analysis_result.language_distribution
                }
                print("✅ Benchmark completed successfully.")
            else:
                benchmark_results = {"error": "Repository analysis returned no result during benchmark."}
                print("❌ Benchmark failed: Analysis returned None.")

        except Exception as e:
            print(f"❌ Benchmark execution error: {e}")
            # import traceback
            # traceback.print_exc()
            benchmark_results = {"error": f"Exception during benchmark: {e}"}
        finally:
            if created_temp_repo:
                temp_dir_obj.cleanup() # Clean up temporary directory
                print(f"Temporary benchmark repository cleaned up: {test_repo_path}")
        
        print(f"Benchmark Results: {json.dumps(benchmark_results, indent=2)}")
        return benchmark_results


# Brain System Integration Helper (Original code was here)
# No changes made to CodeToBrainConverter, as the request was to complete MLXCodeAnalyzer primarily.
# Keeping it as is from the original prompt.
class CodeToBrainConverter:
    """
    Helper class für Integration mit Brain System
    """
    def __init__(self, analyzer: MLXCodeAnalyzer): # Type hint MLXCodeAnalyzer
        self.analyzer = analyzer

    async def repository_to_brain_context(self,
                                          analysis: RepositoryAnalysis,
                                          context_type: str = "project_code_overview") -> Dict[str, Any]: # More specific type
        brain_content_parts = [
            f"# Code Repository Analysis: {analysis.project_name}",
            "## Project Snapshot:",
            f"- **Total Files:** {len(analysis.files)}",
            f"- **Lines of Code (LOC):** {analysis.total_lines:,}",
            f"- **Primary Languages:** {', '.join(sorted(analysis.language_distribution.keys(), key=lambda k: analysis.language_distribution[k], reverse=True)[:3])}", # Top 3
            f"- **Overall Documentation Coverage:** {analysis.documentation_coverage:.1f}%",
            f"- **Average File Complexity (CCN):** {analysis.total_complexity / len(analysis.files):.2f}" if analysis.files else "N/A",
            ""
        ]

        if analysis.readme_content:
            brain_content_parts.extend([
                "## Project README Summary:",
                analysis.readme_content[:1000] + ("..." if len(analysis.readme_content) > 1000 else ""),
                ""
            ])
        
        if analysis.structure_summary:
            brain_content_parts.extend([
                "## Repository Structure Highlights:",
                # Limit structure summary length if it's too verbose
                analysis.structure_summary[:1500] + ("..." if len(analysis.structure_summary) > 1500 else ""),
                ""
            ])

        # Language-specific highlights (e.g. key files per language)
        brain_content_parts.append("## Language-Specific Insights:")
        for lang, count in sorted(analysis.language_distribution.items(), key=lambda x:x[1], reverse=True)[:3]: # Top 3 langs
            lang_files = [f for f in analysis.files if f.language == lang]
            if not lang_files: continue

            brain_content_parts.append(f"### {lang.title()} ({count} files):")
            # Highlight 1-2 key files (e.g. most complex or largest) for this language
            key_lang_files = sorted(lang_files, key=lambda f: (f.complexity_score, f.line_count), reverse=True)[:2]
            for kf in key_lang_files:
                brain_content_parts.append(f"- **Key File:** `{kf.file_path}` (Lines: {kf.line_count}, CCN: {kf.complexity_score:.1f})")
                if kf.documentation:
                    brain_content_parts.append(f"  - *Doc Snippet:* {kf.documentation[:100].replace(chr(10),' ')}...")
                top_element = kf.elements[0] if kf.elements else None
                if top_element:
                     brain_content_parts.append(f"  - *Example Element:* `{top_element.type}: {top_element.name}`")
            brain_content_parts.append("")


        # Key dependencies (if graph is rich)
        if analysis.dependency_graph:
            external_deps = set()
            for deps in analysis.dependency_graph.values():
                for dep in deps:
                    if dep.startswith("pkg:"):
                        external_deps.add(dep.replace("pkg:", ""))
            if external_deps:
                brain_content_parts.extend([
                    "## Key External Dependencies (Inferred):",
                    f"- {', '.join(sorted(list(external_deps))[:10])}" + ("..." if len(external_deps) > 10 else ""), # Show up to 10
                    ""
                ])


        brain_metadata = {
            "source_type": "code_repository_analysis",
            "repository_name": analysis.project_name,
            "repository_disk_path": analysis.repo_path, # For reference, might not be accessible by brain directly
            "analysis_timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "total_files_analyzed": len(analysis.files),
            "total_loc": analysis.total_lines,
            "primary_languages": list(analysis.language_distribution.keys()),
            "context_generated_by": "MLXCodeAnalyzer_vNext", # Version/ID of this tool
            "original_context_type_request": context_type,
        }
        
        # Ensure context_name is filesystem / URL friendly
        safe_project_name = re.sub(r'[^a-zA-Z0-9_-]', '_', analysis.project_name.lower())
        context_name_for_brain = f"{context_type}_{safe_project_name}"


        return {
            "context_name": context_name_for_brain,
            "operation_type": "upsert_context", # Or "create_context"
            "content_format": "markdown", # The content is formatted as Markdown
            "full_text_content": "\n".join(brain_content_parts),
            "metadata_dict": brain_metadata,
            # "vector_embedding": None, # Optionally, embed the whole summary for semantic search in brain
        }

    async def code_element_to_brain_knowledge_unit(self, # Renamed method
                                               element: CodeElement,
                                               file_info: CodeFile, # Pass CodeFile for more context
                                               repo_name: str,
                                               knowledge_type: str = "code_component") -> Dict[str, Any]:
        
        content_parts = [
            f"# Code Component: {element.type.title()} `{element.name}`",
            f"**Source Repository:** {repo_name}",
            f"**File:** `{element.file_path}` ({element.language})",
            f"**Lines:** {element.start_line} - {element.end_line}",
        ]
        if element.complexity is not None:
            content_parts.append(f"**Cyclomatic Complexity (CCN):** {element.complexity}")

        if element.docstring:
            content_parts.extend(["", "## Documentation:", "```text", element.docstring, "```"])
        else:
            content_parts.append("- *No direct documentation found for this element.*")

        if element.parameters:
            content_parts.extend(["", "## Parameters:", f"`{', '.join(element.parameters)}`"])
        if element.return_type:
            content_parts.append(f"**Returns:** `{element.return_type}`")
        if element.decorators:
            content_parts.append(f"**Decorators:** `{', '.join(element.decorators)}`")
        
        content_parts.extend(["", "## Code Implementation:", f"```{element.language}", element.content, "```"])

        # Add context from the containing file
        content_parts.extend([
            "", "## File Context:",
            f"- The containing file `{file_info.file_path}` has {file_info.line_count} lines and an average complexity of {file_info.complexity_score:.1f}.",
            f"- It includes {len(file_info.elements)} major code elements and imports {len(file_info.imports)} modules/dependencies.",
        ])
        if file_info.documentation:
             content_parts.append(f"- File-level documentation snippet: {file_info.documentation[:150].replace(chr(10),' ')}...")


        metadata = {
            "knowledge_unit_type": knowledge_type,
            "element_type": element.type,
            "element_name": element.name,
            "source_file_path": element.file_path,
            "source_repository": repo_name,
            "programming_language": element.language,
            "code_start_line": element.start_line,
            "code_end_line": element.end_line,
            "ccn_complexity": element.complexity,
            "has_docstring": bool(element.docstring),
            "parameter_count": len(element.parameters) if element.parameters else 0,
            "content_hash_md5": hashlib.md5(element.content.encode('utf-8')).hexdigest(), # Hash of the element's code
            "related_file_hash_md5": file_info.file_hash,
        }
        
        safe_element_name = re.sub(r'[^a-zA-Z0-9_-]', '_', element.name.lower())
        knowledge_unit_id = f"{knowledge_type}_{repo_name}_{element.file_path.replace('/', '_')}_{safe_element_name}_{element.type}"
        knowledge_unit_id = re.sub(r'_+', '_', knowledge_unit_id) # Consolidate underscores

        return {
            "knowledge_unit_id": knowledge_unit_id,
            "operation_type": "upsert_knowledge",
            "content_format": "markdown",
            "full_text_content": "\n".join(content_parts),
            "metadata_dict": metadata,
            "vector_embedding": element.embedding, # Use pre-computed embedding if available
            "embedding_model_id": self.analyzer.config.embedding_model if element.embedding else None
        }

# Usage Examples (Original code was here)
# No changes made to example_usage, as the request was to complete MLXCodeAnalyzer primarily.
# Keeping it as is from the original prompt.
async def example_usage():
    """Beispiele für Code Analyzer Usage"""

    # Initialize with custom config
    # Ensure tree_sitter_grammars directory exists and contains .so files for cpp, go, rust etc.
    # e.g., ./tree_sitter_grammars/tree-sitter-cpp.so
    # You can download/compile them from https://github.com/tree-sitter/tree-sitter-<language>
    config = CodeAnalysisConfig(
        supported_languages=["python", "javascript", "typescript", "java", "cpp", "c", "go", "rust"],
        extract_documentation=True,
        analyze_complexity=True,
        generate_embeddings=True, # Set to False if no embedding engine is available/needed
        embedding_model="mlx-community/gte-small", # Or your preferred model
        tree_sitter_grammar_dir="./tree_sitter_grammars" # IMPORTANT: Set this path correctly
    )

    # --- Mock Embedding Engine if not available for standalone run ---
    class MockEmbeddingEngine:
        async def initialize(self): print("MockEmbeddingEngine initialized.")
        async def embed(self, texts: List[str]):
            print(f"MockEmbeddingEngine: Pretending to embed {len(texts)} texts.")
            # Return dummy embeddings of expected dimension (e.g., GTE-small is 384)
            class EmbeddingItem: pass # Dummy for dot access
            class EmbeddingResult:
                def __init__(self, num_embeddings, dim=384):
                    self.embeddings = []
                    for _ in range(num_embeddings):
                        # Create a dummy list of floats for each embedding
                        dummy_emb_data = [0.0] * dim 
                        # Make it look like the structure MLXEmbeddingEngine might return
                        emb_item = EmbeddingItem()
                        # setattr(emb_item, "tolist", lambda: dummy_emb_data) # if it's a numpy array
                        # If it's already a list in MLXEmbeddingEngine, then:
                        self.embeddings.append(dummy_emb_data)


            return EmbeddingResult(len(texts)) if texts else None


    # analyzer = MLXCodeAnalyzer(config, embedding_engine=MockEmbeddingEngine()) # Use mock for testing without real engine
    # For real usage, provide a real MLXEmbeddingEngine instance
    # embedding_engine_instance = MLXEmbeddingEngine(EmbeddingConfig(model_path=config.embedding_model))
    # await embedding_engine_instance.initialize()
    # analyzer = MLXCodeAnalyzer(config, embedding_engine=embedding_engine_instance)
    analyzer = MLXCodeAnalyzer(config, embedding_engine=MockEmbeddingEngine()) # Using Mock for this example to run

    # Create a dummy repo for testing
    test_repo_dir = Path("temp_test_repo_example")
    test_repo_dir.mkdir(exist_ok=True)
    
    (test_repo_dir / "example.py").write_text("""
def hello_python(name):
    \"\"\"Greets in Python.\"\"\"
    print(f"Hello, {name} from Python!")
class PythonUtil:
    CONSTANT_VAL = 100
    def process(self): pass
    """)
    (test_repo_dir / "example.cpp").write_text("""
#include <string>
// C++ example
void hello_cpp(std::string name) { /* Say hi */ }
class CppUtil { public: void compute(); };
    """)
    (test_repo_dir / "README.md").write_text("# Test Repo\nThis is a test.")


    # Analyze single file
    try:
        single_file_path = test_repo_dir / "example.py"
        if single_file_path.exists():
            code_file_analysis = await analyzer.analyze_file(
                str(single_file_path), # Pass as string or Path
                user_id="user_example_123"
            )
            if code_file_analysis:
                print(f"\n--- Single File Analysis: {code_file_analysis.file_path} ---")
                print(f"Language: {code_file_analysis.language}")
                print(f"Elements Found: {len(code_file_analysis.elements)}")
                for el in code_file_analysis.elements:
                    print(f"  - {el.type}: {el.name} (Lines: {el.start_line}-{el.end_line}, Has Embedding: {el.embedding is not None})")
                print(f"Lines: {code_file_analysis.line_count}")
                print(f"Complexity (CCN): {code_file_analysis.complexity_score:.2f}")
                print(f"File Documentation: {'Yes' if code_file_analysis.documentation else 'No'}")
                print(f"Imports: {code_file_analysis.imports}")
        else:
            print(f"Test file {single_file_path} not found for single file analysis example.")

    except Exception as e_single:
        print(f"Single file analysis example error: {e_single}")

    # Analyze repository
    try:
        repo_analysis_result = await analyzer.analyze_repository(
            str(test_repo_dir), # Pass as string or Path
            user_id="user_example_123"
        )

        print(f"\n--- Repository Analysis: {repo_analysis_result.project_name} ---")
        print(f"Total Files Analyzed: {len(repo_analysis_result.files)}")
        print(f"Languages Detected: {', '.join(repo_analysis_result.language_distribution.keys())}")
        print(f"Total Lines of Code: {repo_analysis_result.total_lines:,}")
        print(f"Overall Documentation Coverage: {repo_analysis_result.documentation_coverage:.1f}%")
        print(f"Total Analysis Time: {repo_analysis_result.analysis_time:.2f} seconds")
        
        if repo_analysis_result.readme_content:
            print(f"README Found: Yes (length {len(repo_analysis_result.readme_content)})")
        if repo_analysis_result.structure_summary:
            print("\nRepository Structure Summary (first 200 chars):")
            print(repo_analysis_result.structure_summary[:200] + "...")


        # Export analysis
        export_json_path = test_repo_dir / "code_analysis_report.json"
        export_md_path = test_repo_dir / "code_analysis_report.md"
        await analyzer.export_analysis(repo_analysis_result, str(export_json_path), "json")
        await analyzer.export_analysis(repo_analysis_result, str(export_md_path), "markdown")
        print(f"\nAnalysis exported to {export_json_path} and {export_md_path}")

        # Integration with Vector Store (Mock example)
        class MockVectorStore:
            async def add_vectors(self, user_id, model_id, vectors, metadata, namespace):
                print(f"MockVectorStore: Received {len(vectors)} vectors for namespace '{namespace}' by user '{user_id}' using model '{model_id}'.")
                # print(f"Sample metadata: {metadata[0] if metadata else 'N/A'}")
                return True
        
        mock_vector_store_instance = MockVectorStore()
        
        if 'repo_analysis_result' in locals() and repo_analysis_result:
            saved_to_vs = await analyzer.save_to_vector_store(
                repo_analysis_result,
                mock_vector_store_instance, # type: ignore
                user_id="user_example_123",
                model_id=config.embedding_model
            )
            if saved_to_vs:
                print("✅ Mock: Code analysis data 'sent' to vector store.")

        # Brain System Integration (using the converter)
        brain_converter = CodeToBrainConverter(analyzer)
        if 'repo_analysis_result' in locals() and repo_analysis_result:
            brain_repo_context = await brain_converter.repository_to_brain_context(
                repo_analysis_result,
                context_type="project_code_overview_test"
            )
            print(f"\n--- Brain Repository Context Generated ---")
            print(f"Context Name: {brain_repo_context['context_name']}")
            # print(f"Content (first 300 chars): {brain_repo_context['full_text_content'][:300]}...")
            print(f"Metadata Keys: {list(brain_repo_context['metadata_dict'].keys())}")

            # Example for a single code element
            if repo_analysis_result.files and repo_analysis_result.files[0].elements:
                first_element = repo_analysis_result.files[0].elements[0]
                first_file_info = repo_analysis_result.files[0]
                brain_element_context = await brain_converter.code_element_to_brain_knowledge_unit(
                    first_element,
                    first_file_info,
                    repo_analysis_result.project_name,
                    knowledge_type="code_function_detail_test"
                )
                print(f"\n--- Brain Code Element Context Generated ---")
                print(f"Knowledge Unit ID: {brain_element_context['knowledge_unit_id']}")
                # print(f"Content (first 300 chars): {brain_element_context['full_text_content'][:300]}...")


    except Exception as e_repo:
        print(f"Repository analysis example error: {e_repo}")
        import traceback
        traceback.print_exc()

    # Performance stats
    current_perf_stats = analyzer.get_performance_stats()
    print(f"\n--- Current Performance Stats from Analyzer Instance ---")
    print(json.dumps(current_perf_stats, indent=2))

    # Benchmark (will create its own temp repo if path not given)
    print("\n--- Running Benchmark (with internal temp repo) ---")
    benchmark_run_results = await analyzer.benchmark()
    print(f"Benchmark Run Results: {json.dumps(benchmark_run_results, indent=2)}")

    # Clean up dummy repo
    import shutil
    try:
        shutil.rmtree(test_repo_dir)
        print(f"\nCleaned up test directory: {test_repo_dir}")
    except Exception as e_clean:
        print(f"Error cleaning up test directory {test_repo_dir}: {e_clean}")


if __name__ == "__main__":
    # For tree-sitter grammars, ensure they are compiled and in the correct path
    # e.g. create a folder "./tree_sitter_grammars" and place .so files there.
    # On Linux: gcc -shared -o tree-sitter-cpp.so -I./src ./src/parser.c ./src/scanner.cc -fPIC
    # On macOS: gcc -shared -o tree-sitter-cpp.dylib -I./src ./src/parser.c ./src/scanner.cc -fPIC -dynamiclib
    # (paths are relative to the cloned tree-sitter-<language> repository)
    asyncio.run(example_usage())