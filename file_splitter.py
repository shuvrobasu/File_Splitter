# !/usr/bin/env python3
"""
Professional Python Code Splitter GUI with Syntax Highlighting
Uses Tkinter and AST to parse and split Python files
Features: Split by tokens/lines/parts, context menu, copy/export, comment removal, syntax highlighting
# Author: CyberWiz, 2025-26, github:shuvrobasu
"""


from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
from tkinter.scrolledtext import ScrolledText
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any, Tuple, Set
import re
import os

#==========================
# GLOBALS
#==========================



# ============================================================================
# SYNTAX HIGHLIGHTING
# ============================================================================

class SyntaxHighlighter:
    """Provides syntax highlighting for Python code in Text widgets."""

    # Color scheme (can be customized)
    COLORS = {
        'keyword': '#0000FF',
        'builtin': '#900090',
        'string': '#008000',
        'comment': '#808080',
        'decorator': '#AA22FF',
        'number': '#FF6600',
        'classname': '#0066BB',
        'funcname': '#00627A',
        'self': '#9D2863',
        'operator': '#555555',
        'default': '#000000',
    }

    KEYWORDS = {
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
        'try', 'while', 'with', 'yield'
    }

    BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr',
        'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
        'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr',
        'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
        'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord',
        'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
        'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str',
        'sum', 'super', 'tuple', 'type', 'vars', 'zip', '__import__',
        'Exception', 'BaseException', 'TypeError', 'ValueError', 'KeyError',
        'IndexError', 'AttributeError', 'RuntimeError', 'StopIteration',
        'NotImplementedError', 'OSError', 'IOError', 'FileNotFoundError'
    }

    def __init__(self, text_widget: tk.Text):
        self.text = text_widget
        self._configure_tags()

    def _configure_tags(self):
        """Configure text tags for highlighting."""
        for tag, color in self.COLORS.items():
            font_weight = 'bold' if tag in ('keyword', 'classname', 'funcname') else 'normal'
            font_slant = 'italic' if tag == 'comment' else 'roman'
            self.text.tag_configure(
                tag,
                foreground=color,
                font=('Consolas', 10, font_weight, font_slant)
            )

        # Ensure proper tag priority (comments and strings override keywords)
        self.text.tag_raise('string')
        self.text.tag_raise('comment')

    def highlight(self):
        """Apply syntax highlighting to the text widget content."""
        # Remove existing tags
        for tag in self.COLORS.keys():
            self.text.tag_remove(tag, "1.0", tk.END)

        content = self.text.get("1.0", tk.END)

        # Order matters: highlight in order of precedence (lowest to highest)
        # Numbers
        self._highlight_pattern(r'\b\d+\.?\d*([eE][+-]?\d+)?\b', 'number')
        self._highlight_pattern(r'\b0[xX][0-9a-fA-F]+\b', 'number')
        self._highlight_pattern(r'\b0[bB][01]+\b', 'number')
        self._highlight_pattern(r'\b0[oO][0-7]+\b', 'number')

        # Keywords
        for kw in self.KEYWORDS:
            self._highlight_word(kw, 'keyword')

        # Builtins
        for bi in self.BUILTINS:
            self._highlight_word(bi, 'builtin')

        # Self/cls
        self._highlight_word('self', 'self')
        self._highlight_word('cls', 'self')

        # Decorators
        self._highlight_pattern(r'@[\w\.]+', 'decorator')

        # Class and function names - using group capture instead of lookbehind
        self._highlight_definition(r'\bclass\s+(\w+)', 'classname')
        self._highlight_definition(r'\bdef\s+(\w+)', 'funcname')
        self._highlight_definition(r'\basync\s+def\s+(\w+)', 'funcname')

        # Strings (override everything inside them)
        # Triple-quoted strings first
        self._highlight_pattern(r'"""[\s\S]*?"""', 'string')
        self._highlight_pattern(r"'''[\s\S]*?'''", 'string')
        # f-strings
        self._highlight_pattern(r'f"[^"\\]*(?:\\.[^"\\]*)*"', 'string')
        self._highlight_pattern(r"f'[^'\\]*(?:\\.[^'\\]*)*'", 'string')
        # r-strings
        self._highlight_pattern(r'r"[^"]*"', 'string')
        self._highlight_pattern(r"r'[^']*'", 'string')
        # Regular strings
        self._highlight_pattern(r'"[^"\\]*(?:\\.[^"\\]*)*"', 'string')
        self._highlight_pattern(r"'[^'\\]*(?:\\.[^'\\]*)*'", 'string')

        # Comments (highest priority - override everything)
        self._highlight_pattern(r'#[^\n]*', 'comment')

    def _highlight_pattern(self, pattern: str, tag: str):
        """Apply tag to all matches of pattern."""
        content = self.text.get("1.0", tk.END)
        for match in re.finditer(pattern, content):
            start_idx = f"1.0+{match.start()}c"
            end_idx = f"1.0+{match.end()}c"
            self.text.tag_add(tag, start_idx, end_idx)

    def _highlight_definition(self, pattern: str, tag: str):
        """Highlight the captured group (name) in class/function definitions."""
        content = self.text.get("1.0", tk.END)
        for match in re.finditer(pattern, content):
            if match.groups():
                # Get the position of group 1 (the name)
                name_start = match.start(1)
                name_end = match.end(1)
                start_idx = f"1.0+{name_start}c"
                end_idx = f"1.0+{name_end}c"
                self.text.tag_add(tag, start_idx, end_idx)

    def _highlight_word(self, word: str, tag: str):
        """Highlight a specific word with word boundaries."""
        pattern = rf'\b{re.escape(word)}\b'
        self._highlight_pattern(pattern, tag)


# ============================================================================
# ORIGINAL SPLITTER CODE (Enhanced)
# ============================================================================

def approx_tokens(text: str) -> int:
    """Conservative cross-provider approximation for token count."""
    if not text:
        return 0
    non_space = len(re.sub(r"\s+", "", text))
    spaces = len(text) - non_space
    est = int(non_space / 3.2 + spaces / 12.0)
    return max(est, 1)


@dataclass
class SplitConfig:
    max_tokens: int = 8000
    target_ratio: float = 0.88
    min_chunk_tokens: int = 200


_TOPLEVEL_RE = re.compile(r"^(class\s+\w+|def\s+\w+)\b", re.MULTILINE)


def split_python_for_paste(
        text: str,
        token_count: Callable[[str], int] = approx_tokens,
        cfg: SplitConfig = SplitConfig()
) -> List[str]:
    """Split Python code by token count."""
    hard_limit = cfg.max_tokens
    soft_limit = int(cfg.max_tokens * cfg.target_ratio)

    if token_count(text) <= soft_limit:
        return [text]

    starts = [m.start() for m in _TOPLEVEL_RE.finditer(text)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    starts = sorted(set(starts))
    starts.append(len(text))

    sections: List[str] = []
    for i in range(len(starts) - 1):
        seg = text[starts[i]:starts[i + 1]]
        if seg.strip():
            sections.append(seg)

    refined: List[str] = []
    for s in sections:
        if token_count(s) <= soft_limit:
            refined.append(s)
            continue
        parts = re.split(r"\n{2,}", s)
        for p in parts:
            p = p.rstrip()
            if p.strip():
                refined.append(p + "\n\n")

    chunks: List[str] = []
    cur = ""
    cur_tok = 0

    def flush():
        nonlocal cur, cur_tok
        if cur.strip():
            chunks.append(cur)
        cur = ""
        cur_tok = 0

    for piece in refined:
        ptok = token_count(piece)

        if ptok > soft_limit:
            lines = piece.splitlines(keepends=True)
            buf = ""
            for ln in lines:
                if not buf or token_count(buf + ln) <= soft_limit:
                    buf += ln
                else:
                    if cur and cur_tok + token_count(buf) > soft_limit:
                        flush()
                    cur += buf
                    cur_tok += token_count(buf)
                    buf = ln
            if buf.strip():
                if cur and cur_tok + token_count(buf) > soft_limit:
                    flush()
                cur += buf
                cur_tok += token_count(buf)
            continue

        if cur and cur_tok + ptok > soft_limit:
            flush()

        cur += piece
        cur_tok += ptok

    flush()

    final_chunks: List[str] = []
    for c in chunks:
        if token_count(c) <= hard_limit:
            final_chunks.append(c)
            continue
        lines = c.splitlines(keepends=True)
        buf = ""
        for ln in lines:
            if not buf or token_count(buf + ln) <= hard_limit:
                buf += ln
            else:
                if buf.strip():
                    final_chunks.append(buf)
                buf = ln
        if buf.strip():
            final_chunks.append(buf)

    merged: List[str] = []
    for c in final_chunks:
        if merged and token_count(merged[-1] + c) <= soft_limit and token_count(c) < cfg.min_chunk_tokens:
            merged[-1] += c
        else:
            merged.append(c)

    return merged


def split_by_lines(text: str, max_lines: int = 200) -> List[str]:
    """Split text by maximum number of lines per chunk."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return [text]

    chunks = []
    for i in range(0, len(lines), max_lines):
        chunk = ''.join(lines[i:i + max_lines])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ============================================================================
# INTELLIGENT N-PARTS SPLITTER
# ============================================================================

@dataclass
class AtomicBlock:
    """Represents a block of code that should not be split."""
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    code: str
    tokens: int
    block_type: str = "unknown"  # 'function', 'method', 'class_header', 'statement'


def get_atomic_blocks(text: str) -> List[AtomicBlock]:
    """
    Extract atomic code blocks that shouldn't be split internally.
    Functions and methods are atomic; we can split between class members.
    """
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    if not text.strip():
        return []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Fall back: treat each line as atomic
        blocks = []
        for i, line in enumerate(lines, 1):
            if line.strip():
                blocks.append(AtomicBlock(i, i, line, approx_tokens(line), 'line'))
        return blocks

    blocks: List[AtomicBlock] = []
    covered_lines: Set[int] = set()

    def get_code_for_lines(start: int, end: int) -> str:
        """Get code for line range (1-indexed, inclusive)."""
        return ''.join(lines[start - 1:end])

    def process_node(node: ast.AST, parent_is_class: bool = False):
        """Recursively process AST nodes to find atomic blocks."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Functions and methods are atomic - cannot be split
                start = child.lineno
                end = child.end_lineno or child.lineno

                # Include decorators
                if child.decorator_list:
                    start = min(d.lineno for d in child.decorator_list)

                code = get_code_for_lines(start, end)
                block_type = 'method' if parent_is_class else 'function'
                blocks.append(AtomicBlock(start, end, code, approx_tokens(code), block_type))

                for ln in range(start, end + 1):
                    covered_lines.add(ln)

            elif isinstance(child, ast.ClassDef):
                # Class definition - we CAN split between its members
                # Add class header (decorators + class line + docstring if any)
                start = child.lineno

                # Include decorators
                if child.decorator_list:
                    start = min(d.lineno for d in child.decorator_list)

                # Find where class body starts
                if child.body:
                    first_body = child.body[0]
                    header_end = first_body.lineno - 1

                    # If first body item is a docstring, include it in header
                    if (isinstance(first_body, ast.Expr) and
                            isinstance(first_body.value, (ast.Str, ast.Constant))):
                        header_end = first_body.end_lineno or first_body.lineno
                        if len(child.body) > 1:
                            # There's more after docstring
                            pass
                        else:
                            header_end = child.end_lineno or child.lineno
                else:
                    header_end = child.end_lineno or child.lineno

                # Ensure header_end is at least start
                header_end = max(header_end, start)

                # Add class header as atomic block
                if header_end >= start:
                    code = get_code_for_lines(start, header_end)
                    blocks.append(AtomicBlock(start, header_end, code, approx_tokens(code), 'class_header'))
                    for ln in range(start, header_end + 1):
                        covered_lines.add(ln)

                # Process class body (methods will be added as separate atomic blocks)
                process_node(child, parent_is_class=True)

            else:
                # Other statements (imports, assignments, etc.)
                if hasattr(child, 'lineno') and hasattr(child, 'end_lineno'):
                    start = child.lineno
                    end = child.end_lineno or child.lineno

                    # Skip if already covered
                    if any(ln in covered_lines for ln in range(start, end + 1)):
                        continue

                    code = get_code_for_lines(start, end)
                    if code.strip():
                        blocks.append(AtomicBlock(start, end, code, approx_tokens(code), 'statement'))
                        for ln in range(start, end + 1):
                            covered_lines.add(ln)

    process_node(tree)

    # Handle uncovered lines (comments, blank lines at module level)
    i = 1
    while i <= total_lines:
        if i not in covered_lines:
            # Find contiguous uncovered block
            start = i
            while i <= total_lines and i not in covered_lines:
                i += 1
            end = i - 1

            code = get_code_for_lines(start, end)
            if code.strip():
                blocks.append(AtomicBlock(start, end, code, approx_tokens(code), 'other'))
        else:
            i += 1

    # Sort by start line
    blocks.sort(key=lambda b: b.start_line)

    return blocks


def split_into_n_parts(text: str, n_parts: int) -> List[str]:
    """
    Split code into exactly N parts without breaking function/method bodies.
    Uses intelligent boundary detection to ensure clean splits.
    """
    if n_parts <= 0:
        return [text] if text.strip() else []
    if n_parts == 1:
        return [text] if text.strip() else []

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    if total_lines == 0:
        return []

    # Get atomic blocks
    blocks = get_atomic_blocks(text)

    if not blocks:
        return [text] if text.strip() else []

    if len(blocks) < n_parts:
        # Fewer blocks than requested parts - return blocks as parts
        return [b.code for b in blocks if b.code.strip()]

    # Calculate total tokens and target per part
    total_tokens = sum(b.tokens for b in blocks)
    target_tokens_per_part = total_tokens / n_parts

    # Greedy algorithm: assign blocks to parts, trying to balance token count
    parts: List[List[AtomicBlock]] = [[] for _ in range(n_parts)]
    current_part = 0
    current_tokens = 0

    for i, block in enumerate(blocks):
        remaining_blocks = len(blocks) - i
        remaining_parts = n_parts - current_part

        # If we need to save blocks for remaining parts, move to next part
        if remaining_blocks <= remaining_parts and current_part < n_parts - 1:
            if parts[current_part]:  # Only move if current part has content
                current_part += 1
                current_tokens = 0

        # Add block to current part
        parts[current_part].append(block)
        current_tokens += block.tokens

        # Check if we should move to next part
        if current_part < n_parts - 1:
            # Move to next part if we've exceeded target and there are more blocks
            if current_tokens >= target_tokens_per_part and i + 1 < len(blocks):
                # Ensure remaining blocks can fill remaining parts
                remaining = len(blocks) - i - 1
                remaining_p = n_parts - current_part - 1
                if remaining >= remaining_p:
                    current_part += 1
                    current_tokens = 0

    # Combine blocks within each part
    result = []
    for part_blocks in parts:
        if part_blocks:
            # Sort by line number to maintain order
            part_blocks.sort(key=lambda b: b.start_line)

            # Combine code from all blocks in this part
            combined_lines: List[str] = []
            last_end = 0

            for block in part_blocks:
                # Add any gap lines (might be blank lines between blocks)
                if last_end > 0 and block.start_line > last_end + 1:
                    gap_start = last_end
                    gap_end = block.start_line - 1
                    for ln in range(gap_start, gap_end):
                        if ln < len(lines):
                            combined_lines.append(lines[ln])

                combined_lines.append(block.code)
                last_end = block.end_line

            part_code = ''.join(combined_lines) if combined_lines else ''

            # Clean up: ensure proper line endings
            if part_code.strip():
                if not part_code.endswith('\n'):
                    part_code += '\n'
                result.append(part_code)

    # If we got fewer parts than requested, try to rebalance
    while len(result) < n_parts and len(result) > 1:
        # Find the largest part and try to split it
        largest_idx = max(range(len(result)), key=lambda i: approx_tokens(result[i]))
        largest = result[largest_idx]

        # Try to find a safe split point in the largest part
        sub_blocks = get_atomic_blocks(largest)
        if len(sub_blocks) >= 2:
            mid = len(sub_blocks) // 2
            first_half = ''.join(b.code for b in sub_blocks[:mid])
            second_half = ''.join(b.code for b in sub_blocks[mid:])

            if first_half.strip() and second_half.strip():
                result[largest_idx] = first_half
                result.insert(largest_idx + 1, second_half)
            else:
                break
        else:
            break

    return result if result else [text]


def remove_consecutive_comments(text: str, min_consecutive: int = 3) -> str:
    """Remove blocks of 3 or more consecutive comment lines."""
    lines = text.splitlines(keepends=True)
    result = []
    comment_block = []

    for line in lines:
        stripped = line.strip()
        is_comment = stripped.startswith('#')

        if is_comment:
            comment_block.append(line)
        else:
            if len(comment_block) < min_consecutive:
                result.extend(comment_block)
            comment_block = []
            result.append(line)

    # Handle trailing comments
    if len(comment_block) < min_consecutive:
        result.extend(comment_block)

    return ''.join(result)


# ============================================================================
# AST CODE STRUCTURE PARSER
# ============================================================================

@dataclass
class CodeElement:
    """Represents a parsed code element (class, function, method)."""
    type: str
    name: str
    lineno: int
    end_lineno: int
    col_offset: int
    code: str
    parent_name: Optional[str] = None
    children: Optional[List['CodeElement']] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class CodeStructureParser:
    """Parse Python source code and extract structural elements."""

    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.elements: List[CodeElement] = []
        self.tree: Optional[ast.AST] = None
        self._parse()

    def _parse(self):
        """Parse the source code using AST."""
        try:
            self.tree = ast.parse(self.source)
            self._extract_elements(self.tree)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {e}")

    def _get_code_for_node(self, node: ast.AST) -> str:
        """Extract source code for a given AST node."""
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            return ''.join(self.lines[node.lineno - 1:node.end_lineno])
        return ""

    def _extract_elements(self, node: ast.AST, parent: Optional[CodeElement] = None):
        """Recursively extract code elements from AST."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                code = self._get_code_for_node(child)
                elem = CodeElement(
                    type='class',
                    name=child.name,
                    lineno=child.lineno,
                    end_lineno=child.end_lineno or child.lineno,
                    col_offset=child.col_offset,
                    code=code,
                    parent_name=parent.name if parent else None
                )
                self.elements.append(elem)
                if parent:
                    parent.children.append(elem)
                self._extract_elements(child, elem)

            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                elem_type = 'method' if parent and parent.type == 'class' else 'function'
                code = self._get_code_for_node(child)
                elem = CodeElement(
                    type=elem_type,
                    name=child.name,
                    lineno=child.lineno,
                    end_lineno=child.end_lineno or child.lineno,
                    col_offset=child.col_offset,
                    code=code,
                    parent_name=parent.name if parent else None
                )
                self.elements.append(elem)
                if parent:
                    parent.children.append(elem)
                self._extract_elements(child, elem)

    def get_top_level_elements(self) -> List[CodeElement]:
        """Get only top-level classes and functions."""
        return [e for e in self.elements if e.parent_name is None]

    def get_element_by_name(self, name: str, parent_name: Optional[str] = None) -> Optional[CodeElement]:
        """Find an element by name and optional parent."""
        for elem in self.elements:
            if elem.name == name and elem.parent_name == parent_name:
                return elem
        return None


# ============================================================================
# GUI APPLICATION
# ============================================================================

class CodeSplitterApp:
    """Main GUI Application for Python Code Splitter."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Python Code Splitter - Professional Edition")
        self.root.geometry("1700x1000")
        self.root.minsize(1200, 700)
        self.SETTINGS_FILE = "splitter.settings"
        # State variables
        self.current_file: Optional[Path] = None
        self.source_code: str = ""
        self.parser: Optional[CodeStructureParser] = None
        self.split_parts: List[str] = []
        self.selected_part_index: int = -1

        # Syntax highlighters
        self.source_highlighter: Optional[SyntaxHighlighter] = None
        self.part_highlighter: Optional[SyntaxHighlighter] = None

        # Configuration variables
        self.remove_comments_var = tk.BooleanVar(value=False)
        self.max_tokens_var = tk.IntVar(value=8000)
        self.split_mode_var = tk.StringVar(value="tokens")
        self.max_lines_var = tk.IntVar(value=200)
        self.num_parts_var = tk.IntVar(value=5)
        self.target_ratio_var = tk.DoubleVar(value=0.88)
        self.syntax_highlight_var = tk.BooleanVar(value=True)


        self._load_settings()

        # Setup UI
        self._setup_style()
        self._create_menu()
        self._create_toolbar()
        self._create_status_bar()
        self._create_main_layout()

        self._create_context_menus()
        self.root.update_idletasks()

        # Keyboard shortcuts
        self._bind_shortcuts()

    def _load_settings(self):
        """Load settings from ini file."""
        try:
            with open(self.SETTINGS_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key, value = key.strip(), value.strip()
                        if key == 'max_tokens':
                            self.max_tokens_var.set(int(value))
                        elif key == 'max_lines':
                            self.max_lines_var.set(int(value))
                        elif key == 'num_parts':
                            self.num_parts_var.set(int(value))
                        elif key == 'target_ratio':
                            self.target_ratio_var.set(float(value))
                        elif key == 'split_mode':
                            self.split_mode_var.set(value)
                        elif key == 'remove_comments':
                            self.remove_comments_var.set(value.lower() == 'true')
                        elif key == 'syntax_highlight':
                            self.syntax_highlight_var.set(value.lower() == 'true')
        except FileNotFoundError:
            self._save_settings()

    def _save_settings(self):
        """Save settings to ini file."""
        with open(self.SETTINGS_FILE, 'w') as f:
            f.write("# Python Code Splitter Settings\n")
            f.write(f"max_tokens={self.max_tokens_var.get()}\n")
            f.write(f"max_lines={self.max_lines_var.get()}\n")
            f.write(f"num_parts={self.num_parts_var.get()}\n")
            f.write(f"target_ratio={self.target_ratio_var.get()}\n")
            f.write(f"split_mode={self.split_mode_var.get()}\n")
            f.write(f"remove_comments={self.remove_comments_var.get()}\n")
            f.write(f"syntax_highlight={self.syntax_highlight_var.get()}\n")

    def _setup_style(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        # style.theme_use('winnative')
        # style.theme_use('vista')



        style.configure("Treeview",
                        rowheight=28,
                        font=('Consolas', 10))
        style.configure("Treeview.Heading",
                        font=('Segoe UI', 10, 'bold'))
        style.configure("TButton",
                        padding=(10, 5),
                        font=('Segoe UI', 9))
        style.configure("TLabel",
                        font=('Segoe UI', 9))
        style.configure("Header.TLabel",
                        font=('Segoe UI', 11, 'bold'))
        style.configure("Status.TLabel",
                        font=('Consolas', 9))

        style.map('Treeview',
                  background=[('selected', '#0078d4')],
                  foreground=[('selected', 'white')])

    def _create_menu(self):
        """Create the main menu bar."""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Python File...",
                              command=self.load_file,
                              accelerator="Ctrl+O")
        file_menu.add_command(label="Reload Current File",
                              command=self.reload_file,
                              accelerator="F5")
        file_menu.add_separator()

        # Export submenu
        export_menu = Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(label="Export All Parts as .py...",
                                command=lambda: self.export_all_parts('py'))
        export_menu.add_command(label="Export All Parts as .txt...",
                                command=lambda: self.export_all_parts('txt'))
        export_menu.add_separator()
        export_menu.add_command(label="Export Selected Part as .py...",
                                command=lambda: self.export_selected_part('py'))
        export_menu.add_command(label="Export Selected Part as .txt...",
                                command=lambda: self.export_selected_part('txt'))

        file_menu.add_separator()
        file_menu.add_command(label="Exit",
                              command=self.root.quit,
                              accelerator="Alt+F4")

        # Edit menu
        edit_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy Selected Part",
                              command=self.copy_selected_part,
                              accelerator="Ctrl+C")
        edit_menu.add_command(label="Copy All Parts (Separated)",
                              command=self.copy_all_parts)
        edit_menu.add_separator()
        edit_menu.add_command(label="Copy Without Comments",
                              command=self.copy_part_no_comments)

        # Split menu
        split_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Split", menu=split_menu)
        split_menu.add_command(label="Split Entire File",
                               command=self.split_entire_file,
                               accelerator="Ctrl+S")
        split_menu.add_separator()
        split_menu.add_radiobutton(label="Split by Tokens",
                                   variable=self.split_mode_var,
                                   value="tokens",
                                   command=self.on_split_mode_change)
        split_menu.add_radiobutton(label="Split by Lines",
                                   variable=self.split_mode_var,
                                   value="lines",
                                   command=self.on_split_mode_change)
        split_menu.add_radiobutton(label="Split by Parts (N)",
                                   variable=self.split_mode_var,
                                   value="parts",
                                   command=self.on_split_mode_change)

        # View menu
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Syntax Highlighting",
                                  variable=self.syntax_highlight_var,
                                  command=self.toggle_syntax_highlighting)

        # Options menu
        options_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=options_menu)
        options_menu.add_checkbutton(
            label="Remove 3+ Consecutive Comment Lines Before Copy",
            variable=self.remove_comments_var
        )
        options_menu.add_separator()
        options_menu.add_command(label="Settings...",
                                 command=self.show_settings_dialog)

        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Usage Guide", command=self.show_usage_guide)

    def _create_toolbar(self):
        """Create the toolbar."""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=5, pady=5)

        # Open button
        self.btn_open = ttk.Button(toolbar, text="📂 Open", width=8,command=self.load_file)
        self.btn_open.pack(side=tk.LEFT, padx=2)

        # Reload button
        self.btn_reload = ttk.Button(toolbar, text="🔄Reload", width=10, command=self.reload_file)
        self.btn_reload.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Split button
        self.btn_split = ttk.Button(toolbar, text="✂️Split File", width=13, command=self.split_entire_file)
        self.btn_split.pack(side=tk.LEFT, padx=2)

        # Split mode selection
        ttk.Label(toolbar, text="  Mode:").pack(side=tk.LEFT)
        self.mode_combo = ttk.Combobox(toolbar, textvariable=self.split_mode_var,
                                       values=["tokens", "lines", "parts"],
                                       width=8, state="readonly")
        self.mode_combo.pack(side=tk.LEFT, padx=5)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.on_split_mode_change())

        # Limit label and entry (changes based on mode)
        self.limit_label = ttk.Label(toolbar, text="  Max Tokens:")
        self.limit_label.pack(side=tk.LEFT)

        self.limit_entry = ttk.Entry(toolbar, textvariable=self.max_tokens_var, width=8)
        self.limit_entry.pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Remove comments checkbox
        ttk.Checkbutton(toolbar, text="Remove 3+ Comment Lines",
                        variable=self.remove_comments_var).pack(side=tk.LEFT, padx=5)

        # Syntax highlighting checkbox
        ttk.Checkbutton(toolbar, text="Syntax Highlight",
                        variable=self.syntax_highlight_var,
                        command=self.toggle_syntax_highlighting).pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Copy and Export buttons
        self.btn_copy = ttk.Button(toolbar, text="📋Cpy Part",width=10, command=self.copy_selected_part)
        self.btn_copy.pack(side=tk.LEFT, padx=2)

        self.btn_export = ttk.Button(toolbar, text="💾 Xprt .py",width=10,
                                     command=lambda: self.export_selected_part('py'))
        self.btn_export.pack(side=tk.LEFT, padx=2)

        self.btn_export_txt = ttk.Button(toolbar, text="📄 Xprt .txt",width=10,
                                         command=lambda: self.export_selected_part('txt'))
        self.btn_export_txt.pack(side=tk.LEFT, padx=2)

        self.btn_export_all = ttk.Button(toolbar, text="📦 Xprt All", width=12,
                                         command=lambda: self.export_all_parts('py'))
        self.btn_export_all.pack(side=tk.LEFT, padx=2)

    def _create_main_layout(self):
        """Create the main three-panel layout."""
        # Main paned window
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Code Structure Tree
        left_frame = ttk.LabelFrame(self.main_paned, text="Code Structure")
        self.main_paned.add(left_frame, weight=1)

        # Tree view for code structure
        tree_container = ttk.Frame(left_frame)
        tree_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.structure_tree = ttk.Treeview(tree_container,
                                           columns=("type", "lines", "tokens"),
                                           show="tree headings",
                                           selectmode="browse")
        self.structure_tree.heading("#0", text="Name", anchor=tk.W)
        self.structure_tree.heading("type", text="Type", anchor=tk.W)
        self.structure_tree.heading("lines", text="Lines", anchor=tk.CENTER)
        self.structure_tree.heading("tokens", text="Tokens", anchor=tk.CENTER)

        self.structure_tree.column("#0", width=200, minwidth=150)
        self.structure_tree.column("type", width=80, minwidth=60)
        self.structure_tree.column("lines", width=60, minwidth=50)
        self.structure_tree.column("tokens", width=70, minwidth=50)

        tree_scroll_y = ttk.Scrollbar(tree_container, orient=tk.VERTICAL,
                                      command=self.structure_tree.yview)
        tree_scroll_x = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL,
                                      command=self.structure_tree.xview)
        self.structure_tree.configure(yscrollcommand=tree_scroll_y.set,
                                      xscrollcommand=tree_scroll_x.set)

        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.structure_tree.pack(fill=tk.BOTH, expand=True)

        self.structure_tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.structure_tree.bind("<Double-1>", self.on_tree_double_click)

        # Center panel - Source Code Preview
        center_frame = ttk.LabelFrame(self.main_paned, text="Source Code Preview")
        self.main_paned.add(center_frame, weight=2)

        # Create text widget with scrollbars
        source_container = ttk.Frame(center_frame)
        source_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.source_text = tk.Text(source_container, wrap=tk.NONE,
                                   font=('Consolas', 10),
                                   state=tk.DISABLED,
                                   bg='#FAFAFA')

        source_scroll_y = ttk.Scrollbar(source_container, orient=tk.VERTICAL,
                                        command=self.source_text.yview)
        source_scroll_x = ttk.Scrollbar(source_container, orient=tk.HORIZONTAL,
                                        command=self.source_text.xview)
        self.source_text.configure(yscrollcommand=source_scroll_y.set,
                                   xscrollcommand=source_scroll_x.set)

        source_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        source_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.source_text.pack(fill=tk.BOTH, expand=True)

        # Initialize syntax highlighter for source
        self.source_highlighter = SyntaxHighlighter(self.source_text)

        # Right panel - Split Parts
        right_frame = ttk.LabelFrame(self.main_paned, text="Split Parts")
        self.main_paned.add(right_frame, weight=2)

        # Parts list with buttons
        parts_toolbar = ttk.Frame(right_frame)
        parts_toolbar.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(parts_toolbar, text="Parts:", style="Header.TLabel").pack(side=tk.LEFT)
        self.parts_count_label = ttk.Label(parts_toolbar, text="0 parts")
        self.parts_count_label.pack(side=tk.LEFT, padx=10)

        ttk.Button(parts_toolbar, text="◀ Prev",
                   command=self.prev_part).pack(side=tk.RIGHT, padx=2)
        ttk.Button(parts_toolbar, text="Next ▶",
                   command=self.next_part).pack(side=tk.RIGHT, padx=2)

        # Parts listbox
        parts_list_frame = ttk.Frame(right_frame)
        parts_list_frame.pack(fill=tk.X, padx=5)

        self.parts_listbox = tk.Listbox(parts_list_frame, height=6,
                                        font=('Consolas', 9),
                                        selectmode=tk.BROWSE)
        parts_scroll = ttk.Scrollbar(parts_list_frame, orient=tk.VERTICAL,
                                     command=self.parts_listbox.yview)
        self.parts_listbox.configure(yscrollcommand=parts_scroll.set)

        parts_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.parts_listbox.pack(fill=tk.X, expand=True)

        self.parts_listbox.bind("<<ListboxSelect>>", self.on_part_select)

        # Part info
        self.part_info_label = ttk.Label(right_frame, text="", style="Status.TLabel")
        self.part_info_label.pack(fill=tk.X, padx=5, pady=5)

        # Part content preview
        ttk.Label(right_frame, text="Part Content:",
                  style="Header.TLabel").pack(anchor=tk.W, padx=5)

        # Create text widget with scrollbars for part preview
        part_container = ttk.Frame(right_frame)
        part_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.part_text = tk.Text(part_container, wrap=tk.NONE,
                                 font=('Consolas', 10),
                                 state=tk.DISABLED,
                                 bg='#FAFAFA')

        part_scroll_y = ttk.Scrollbar(part_container, orient=tk.VERTICAL,
                                      command=self.part_text.yview)
        part_scroll_x = ttk.Scrollbar(part_container, orient=tk.HORIZONTAL,
                                      command=self.part_text.xview)
        self.part_text.configure(yscrollcommand=part_scroll_y.set,
                                 xscrollcommand=part_scroll_x.set)

        part_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        part_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.part_text.pack(fill=tk.BOTH, expand=False)

        # Initialize syntax highlighter for parts
        self.part_highlighter = SyntaxHighlighter(self.part_text)

        # Part action buttons
        part_actions = ttk.Frame(right_frame)
        part_actions.pack(fill=tk.X, padx=2, pady=2)

        ttk.Button(part_actions, text="📋Cpy|Part", width=10,
                   command=self.copy_selected_part).pack(side=tk.LEFT, padx=2)
        ttk.Button(part_actions, text="💾Xprt .py",width=10,
                   command=lambda: self.export_selected_part('py')).pack(side=tk.LEFT, padx=2)
        ttk.Button(part_actions, text="📄Xprt .txt",width=10,
                   command=lambda: self.export_selected_part('txt')).pack(side=tk.LEFT, padx=2)
        ttk.Button(part_actions, text="📋Cpy (w/o cmnts)",width=10,
                   command=self.copy_part_no_comments).pack(side=tk.LEFT, padx=2)

    def _create_context_menus(self):
        """Create context menus for tree and text widgets."""
        # Context menu for structure tree
        self.tree_context_menu = Menu(self.root, tearoff=0)

        # Split submenu
        split_submenu = Menu(self.tree_context_menu, tearoff=0)
        split_submenu.add_command(label="Split by Tokens",
                                  command=lambda: self.split_selected_element("tokens"))
        split_submenu.add_command(label="Split by Lines",
                                  command=lambda: self.split_selected_element("lines"))
        split_submenu.add_command(label="Split by Parts (N)",
                                  command=lambda: self.split_selected_element("parts"))

        self.tree_context_menu.add_cascade(label="Split Selected", menu=split_submenu)
        self.tree_context_menu.add_separator()
        self.tree_context_menu.add_command(label="Copy Code",
                                           command=self.copy_selected_tree_item)
        self.tree_context_menu.add_command(label="Copy Code (No Comments)",
                                           command=self.copy_selected_tree_item_no_comments)
        self.tree_context_menu.add_separator()

        # Export submenu in context menu
        export_submenu = Menu(self.tree_context_menu, tearoff=0)
        export_submenu.add_command(label="Export as .py...",
                                   command=lambda: self.export_selected_tree_item('py'))
        export_submenu.add_command(label="Export as .txt (with indents)...",
                                   command=lambda: self.export_selected_tree_item('txt'))
        self.tree_context_menu.add_cascade(label="Export", menu=export_submenu)

        self.tree_context_menu.add_separator()
        self.tree_context_menu.add_command(label="Show in Preview",
                                           command=self.show_selected_in_preview)

        self.structure_tree.bind("<Button-3>", self.show_tree_context_menu)

        # Context menu for parts list
        self.parts_context_menu = Menu(self.root, tearoff=0)
        self.parts_context_menu.add_command(label="Copy Part",
                                            command=self.copy_selected_part)
        self.parts_context_menu.add_command(label="Copy Part (No Comments)",
                                            command=self.copy_part_no_comments)
        self.parts_context_menu.add_separator()

        # Export submenu for parts
        parts_export_submenu = Menu(self.parts_context_menu, tearoff=0)
        parts_export_submenu.add_command(label="Export as .py...",
                                         command=lambda: self.export_selected_part('py'))
        parts_export_submenu.add_command(label="Export as .txt (with indents)...",
                                         command=lambda: self.export_selected_part('txt'))
        self.parts_context_menu.add_cascade(label="Export", menu=parts_export_submenu)

        self.parts_listbox.bind("<Button-3>", self.show_parts_context_menu)

    def _create_status_bar(self):
        """Create the status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(status_frame, text="Ready",
                                      style="Status.TLabel", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)

        self.file_label = ttk.Label(status_frame, text="No file loaded",
                                    style="Status.TLabel", anchor=tk.E)
        self.file_label.pack(side=tk.RIGHT, padx=10, pady=5)

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind("<Control-o>", lambda e: self.load_file())
        self.root.bind("<Control-s>", lambda e: self.split_entire_file())
        self.root.bind("<Control-e>", lambda e: self.export_selected_part('py'))
        self.root.bind("<Control-Shift-E>", lambda e: self.export_all_parts('py'))
        self.root.bind("<F5>", lambda e: self.reload_file())

    # ========================================================================
    # SYNTAX HIGHLIGHTING
    # ========================================================================

    def toggle_syntax_highlighting(self):
        """Toggle syntax highlighting on/off."""
        if self.source_code:
            self._show_source_code(self.source_code)
        if self.split_parts and self.selected_part_index >= 0:
            self._show_part_content(self.selected_part_index)

    def _apply_highlighting(self, text_widget: tk.Text, highlighter: SyntaxHighlighter):
        """Apply syntax highlighting if enabled."""
        if self.syntax_highlight_var.get():
            highlighter.highlight()

    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================

    def load_file(self):
        """Load a Python file."""
        file_path = filedialog.askopenfilename(
            title="Select Python File",
            filetypes=[
                ("Python Files", "*.py"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            self._load_file_content(file_path)

    def _load_file_content(self, file_path: str):
        """Load and parse file content."""
        try:
            path = Path(file_path)
            self.source_code = path.read_text(encoding="utf-8")
            self.current_file = path

            # Parse code structure
            self.parser = CodeStructureParser(self.source_code)

            # Update UI
            self._populate_structure_tree()
            self._show_source_code(self.source_code)
            self._clear_parts()

            self.file_label.config(text=f"📄 {path.name}")
            self.status_label.config(text=f"Loaded: {path.name} | "
                                          f"{len(self.source_code.splitlines())} lines | "
                                          f"~{approx_tokens(self.source_code)} tokens")

        except ValueError as e:
            messagebox.showerror("Parse Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    def reload_file(self):
        """Reload the current file."""
        if self.current_file:
            self._load_file_content(str(self.current_file))

    # ========================================================================
    # STRUCTURE TREE
    # ========================================================================

    def _populate_structure_tree(self):
        """Populate the structure tree with parsed elements."""
        self.structure_tree.delete(*self.structure_tree.get_children())

        if not self.parser:
            return

        # Add file root
        file_name = self.current_file.name if self.current_file else "code"
        root_id = self.structure_tree.insert(
            "", "end",
            text=f"📄 {file_name}",
            values=("file",
                    len(self.source_code.splitlines()),
                    approx_tokens(self.source_code)),
            open=True
        )

        # Track parent items
        parent_map: Dict[str, str] = {None: root_id}

        # Icons for different types
        icons = {
            'class': '📦',
            'function': '🔧',
            'method': '⚙️'
        }

        for elem in self.parser.elements:
            parent_id = parent_map.get(elem.parent_name, root_id)

            icon = icons.get(elem.type, '•')
            lines = elem.end_lineno - elem.lineno + 1
            tokens = approx_tokens(elem.code)

            item_id = self.structure_tree.insert(
                parent_id, "end",
                text=f"{icon} {elem.name}",
                values=(elem.type, lines, tokens),
                tags=(elem.type,),
                open=True
            )

            parent_map[elem.name] = item_id

        # Configure tags for colors
        self.structure_tree.tag_configure('class', foreground='#0066cc')
        self.structure_tree.tag_configure('function', foreground='#008800')
        self.structure_tree.tag_configure('method', foreground='#884400')

    def on_tree_select(self, event):
        """Handle tree selection."""
        selection = self.structure_tree.selection()
        if selection:
            item = selection[0]
            values = self.structure_tree.item(item, 'values')
            if values and values[0] != 'file':
                self.show_selected_in_preview()

    def on_tree_double_click(self, event):
        """Handle double-click on tree item."""
        self.split_selected_element(self.split_mode_var.get())

    def _get_selected_element(self) -> Optional[CodeElement]:
        """Get the currently selected code element."""
        selection = self.structure_tree.selection()
        if not selection or not self.parser:
            return None

        item = selection[0]
        item_text = self.structure_tree.item(item, 'text')
        values = self.structure_tree.item(item, 'values')

        if values[0] == 'file':
            return None

        # Extract name (remove icon)
        name = item_text.split(' ', 1)[-1] if ' ' in item_text else item_text

        # Find parent
        parent_item = self.structure_tree.parent(item)
        parent_text = self.structure_tree.item(parent_item, 'text')
        parent_values = self.structure_tree.item(parent_item, 'values')

        parent_name = None
        if parent_values and parent_values[0] != 'file':
            parent_name = parent_text.split(' ', 1)[-1] if ' ' in parent_text else parent_text

        return self.parser.get_element_by_name(name, parent_name)

    def show_tree_context_menu(self, event):
        """Show context menu for tree."""
        item = self.structure_tree.identify_row(event.y)
        if item:
            self.structure_tree.selection_set(item)
            self.tree_context_menu.post(event.x_root, event.y_root)

    # ========================================================================
    # SPLITTING
    # ========================================================================

    def split_entire_file(self):
        """Split the entire file."""
        if not self.source_code:
            messagebox.showwarning("No File", "Please load a file first.")
            return

        self._split_code(self.source_code)

    def split_selected_element(self, mode: str):
        """Split the selected tree element."""
        if not self.parser:
            return

        selection = self.structure_tree.selection()
        if not selection:
            messagebox.showinfo("Select Item", "Please select an item to split.")
            return

        item = selection[0]
        values = self.structure_tree.item(item, 'values')

        if values[0] == 'file':
            code = self.source_code
        else:
            elem = self._get_selected_element()
            if elem:
                code = elem.code
            else:
                return

        self.split_mode_var.set(mode)
        self.on_split_mode_change()  # Update UI
        self._split_code(code)

    def _split_code(self, code: str):
        """Perform the actual split operation."""
        mode = self.split_mode_var.get()

        if mode == "tokens":
            max_tokens = self.max_tokens_var.get()
            cfg = SplitConfig(max_tokens=max_tokens,
                              target_ratio=self.target_ratio_var.get())
            self.split_parts = split_python_for_paste(code, approx_tokens, cfg)
        elif mode == "lines":
            max_lines = self.max_lines_var.get()
            self.split_parts = split_by_lines(code, max_lines)
        else:  # parts
            n_parts = self.num_parts_var.get()
            self.split_parts = split_into_n_parts(code, n_parts)

        self._update_parts_display()

        limit_str = ""
        if mode == "tokens":
            limit_str = f"Max: {self.max_tokens_var.get()} tokens"
        elif mode == "lines":
            limit_str = f"Max: {self.max_lines_var.get()} lines"
        else:
            limit_str = f"Target: {self.num_parts_var.get()} parts"

        self.status_label.config(
            text=f"Split into {len(self.split_parts)} parts | Mode: {mode} | {limit_str}"
        )

    def _update_parts_display(self):
        """Update the parts list display."""
        self.parts_listbox.delete(0, tk.END)

        for i, part in enumerate(self.split_parts, 1):
            lines = len(part.splitlines())
            tokens = approx_tokens(part)
            self.parts_listbox.insert(tk.END,
                                      f"Part {i}: {lines} lines, ~{tokens} tokens")

        self.parts_count_label.config(text=f"{len(self.split_parts)} parts")

        if self.split_parts:
            self.parts_listbox.selection_set(0)
            self.selected_part_index = 0
            self._show_part_content(0)

    def _clear_parts(self):
        """Clear the parts display."""
        self.split_parts = []
        self.parts_listbox.delete(0, tk.END)
        self.parts_count_label.config(text="0 parts")
        self.part_info_label.config(text="")
        self.part_text.config(state=tk.NORMAL)
        self.part_text.delete("1.0", tk.END)
        self.part_text.config(state=tk.DISABLED)

    # ========================================================================
    # PARTS DISPLAY
    # ========================================================================

    def on_part_select(self, event):
        """Handle part selection."""
        selection = self.parts_listbox.curselection()
        if selection:
            self.selected_part_index = selection[0]
            self._show_part_content(self.selected_part_index)

    def _show_part_content(self, index: int):
        """Show content of selected part."""
        if 0 <= index < len(self.split_parts):
            part = self.split_parts[index]

            self.part_text.config(state=tk.NORMAL)
            self.part_text.delete("1.0", tk.END)
            self.part_text.insert("1.0", part)

            # Apply syntax highlighting
            if self.syntax_highlight_var.get():
                self.part_highlighter.highlight()

            self.part_text.config(state=tk.DISABLED)

            lines = len(part.splitlines())
            tokens = approx_tokens(part)
            self.part_info_label.config(
                text=f"Part {index + 1}/{len(self.split_parts)} | "
                     f"{lines} lines | ~{tokens} tokens | "
                     f"{len(part)} characters"
            )

    def prev_part(self):
        """Navigate to previous part."""
        if self.split_parts and self.selected_part_index > 0:
            self.selected_part_index -= 1
            self.parts_listbox.selection_clear(0, tk.END)
            self.parts_listbox.selection_set(self.selected_part_index)
            self.parts_listbox.see(self.selected_part_index)
            self._show_part_content(self.selected_part_index)

    def next_part(self):
        """Navigate to next part."""
        if self.split_parts and self.selected_part_index < len(self.split_parts) - 1:
            self.selected_part_index += 1
            self.parts_listbox.selection_clear(0, tk.END)
            self.parts_listbox.selection_set(self.selected_part_index)
            self.parts_listbox.see(self.selected_part_index)
            self._show_part_content(self.selected_part_index)

    # ========================================================================
    # SOURCE CODE PREVIEW
    # ========================================================================

    def _show_source_code(self, code: str):
        """Display source code in the preview with optional highlighting."""
        self.source_text.config(state=tk.NORMAL)
        self.source_text.delete("1.0", tk.END)
        self.source_text.insert("1.0", code)

        # Apply syntax highlighting
        if self.syntax_highlight_var.get():
            self.source_highlighter.highlight()

        self.source_text.config(state=tk.DISABLED)

    def show_selected_in_preview(self):
        """Show selected tree item's code in preview."""
        elem = self._get_selected_element()
        if elem:
            self._show_source_code(elem.code)
            self.status_label.config(
                text=f"Showing: {elem.type} '{elem.name}' | "
                     f"Lines {elem.lineno}-{elem.end_lineno} | "
                     f"~{approx_tokens(elem.code)} tokens"
            )

    # ========================================================================
    # COPY OPERATIONS
    # ========================================================================

    def _process_for_copy(self, text: str) -> str:
        """Process text for copying (apply comment removal if enabled)."""
        if self.remove_comments_var.get():
            return remove_consecutive_comments(text)
        return text

    def copy_selected_part(self):
        """Copy the selected part to clipboard."""
        if not self.split_parts or self.selected_part_index < 0:
            messagebox.showinfo("No Part", "No part selected to copy.")
            return

        part = self.split_parts[self.selected_part_index]
        text = self._process_for_copy(part)

        self.root.clipboard_clear()
        self.root.clipboard_append(text)

        self.status_label.config(
            text=f"Copied Part {self.selected_part_index + 1} to clipboard "
                 f"({len(text)} chars, ~{approx_tokens(text)} tokens)"
        )

    def copy_part_no_comments(self):
        """Copy selected part without consecutive comments."""
        if not self.split_parts or self.selected_part_index < 0:
            messagebox.showinfo("No Part", "No part selected to copy.")
            return

        part = self.split_parts[self.selected_part_index]
        text = remove_consecutive_comments(part)

        self.root.clipboard_clear()
        self.root.clipboard_append(text)

        self.status_label.config(
            text=f"Copied Part {self.selected_part_index + 1} (no comments) to clipboard"
        )

    def copy_all_parts(self):
        """Copy all parts to clipboard with separators."""
        if not self.split_parts:
            messagebox.showinfo("No Parts", "No parts to copy.")
            return

        separator = "\n\n" + "=" * 60 + "\n\n"
        all_text = separator.join([
            f"# PART {i + 1}/{len(self.split_parts)}\n\n{self._process_for_copy(p)}"
            for i, p in enumerate(self.split_parts)
        ])

        self.root.clipboard_clear()
        self.root.clipboard_append(all_text)

        self.status_label.config(
            text=f"Copied all {len(self.split_parts)} parts to clipboard"
        )

    def copy_selected_tree_item(self):
        """Copy code from selected tree item."""
        elem = self._get_selected_element()
        if elem:
            text = self._process_for_copy(elem.code)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text=f"Copied {elem.type} '{elem.name}' to clipboard")
        else:
            # Try to copy entire file if file root selected
            selection = self.structure_tree.selection()
            if selection:
                values = self.structure_tree.item(selection[0], 'values')
                if values and values[0] == 'file':
                    text = self._process_for_copy(self.source_code)
                    self.root.clipboard_clear()
                    self.root.clipboard_append(text)
                    self.status_label.config(text="Copied entire file to clipboard")

    def copy_selected_tree_item_no_comments(self):
        """Copy tree item code without consecutive comments."""
        elem = self._get_selected_element()
        if elem:
            text = remove_consecutive_comments(elem.code)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(
                text=f"Copied {elem.type} '{elem.name}' (no comments) to clipboard"
            )

    # ========================================================================
    # EXPORT OPERATIONS
    # ========================================================================

    def _format_for_txt_export(self, text: str) -> str:
        """Format code for .txt export, preserving indentation."""
        lines = text.splitlines()

        # Add header with file info
        header = [
            "=" * 70,
            f"  Exported from Python Code Splitter",
            f"  Original file: {self.current_file.name if self.current_file else 'Unknown'}",
            f"  Lines: {len(lines)} | Tokens: ~{approx_tokens(text)}",
            "=" * 70,
            ""
        ]

        return '\n'.join(header + lines)

    def export_selected_part(self, file_type: str = 'py'):
        """Export selected part to a file."""
        if not self.split_parts or self.selected_part_index < 0:
            messagebox.showinfo("No Part", "No part selected to export.")
            return

        base_name = self.current_file.stem if self.current_file else "code"
        ext = f".{file_type}"
        default_name = f"{base_name}_part{self.selected_part_index + 1}{ext}"

        filetypes = [("Python Files", "*.py"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        if file_type == 'txt':
            filetypes = [("Text Files", "*.txt"), ("Python Files", "*.py"), ("All Files", "*.*")]

        file_path = filedialog.asksaveasfilename(
            title=f"Export Part as .{file_type}",
            defaultextension=ext,
            initialfile=default_name,
            filetypes=filetypes
        )

        if file_path:
            part = self.split_parts[self.selected_part_index]
            text = self._process_for_copy(part)

            if file_type == 'txt':
                text = self._format_for_txt_export(text)

            Path(file_path).write_text(text, encoding="utf-8")
            self.status_label.config(text=f"Exported Part {self.selected_part_index + 1} to {file_path}")

    def export_all_parts(self, file_type: str = 'py'):
        """Export all parts to separate files."""
        if not self.split_parts:
            messagebox.showinfo("No Parts", "No parts to export.")
            return

        folder = filedialog.askdirectory(title="Select Export Folder")
        if not folder:
            return

        base_name = self.current_file.stem if self.current_file else "code"
        ext = f".{file_type}"

        for i, part in enumerate(self.split_parts, 1):
            file_path = Path(folder) / f"{base_name}_part{i}{ext}"
            text = self._process_for_copy(part)

            if file_type == 'txt':
                # Add part-specific header
                lines = text.splitlines()
                header = [
                    "=" * 70,
                    f"  Part {i} of {len(self.split_parts)}",
                    f"  Original file: {self.current_file.name if self.current_file else 'Unknown'}",
                    f"  Lines: {len(lines)} | Tokens: ~{approx_tokens(text)}",
                    "=" * 70,
                    ""
                ]
                text = '\n'.join(header + lines)

            file_path.write_text(text, encoding="utf-8")

        self.status_label.config(
            text=f"Exported {len(self.split_parts)} parts as .{file_type} to {folder}"
        )
        messagebox.showinfo("Export Complete",
                            f"Exported {len(self.split_parts)} parts to:\n{folder}")

    def export_selected_tree_item(self, file_type: str = 'py'):
        """Export selected tree item to file."""
        elem = self._get_selected_element()
        if not elem:
            # Check if file root selected
            selection = self.structure_tree.selection()
            if selection:
                values = self.structure_tree.item(selection[0], 'values')
                if values and values[0] == 'file':
                    code = self.source_code
                    name = self.current_file.stem if self.current_file else "code"
                else:
                    return
            else:
                return
        else:
            code = elem.code
            name = elem.name

        ext = f".{file_type}"
        filetypes = [("Python Files", "*.py"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        if file_type == 'txt':
            filetypes = [("Text Files", "*.txt"), ("Python Files", "*.py"), ("All Files", "*.*")]

        file_path = filedialog.asksaveasfilename(
            title=f"Export Code as .{file_type}",
            defaultextension=ext,
            initialfile=f"{name}{ext}",
            filetypes=filetypes
        )

        if file_path:
            text = self._process_for_copy(code)

            if file_type == 'txt':
                text = self._format_for_txt_export(text)

            Path(file_path).write_text(text, encoding="utf-8")
            self.status_label.config(text=f"Exported to {file_path}")

    def show_parts_context_menu(self, event):
        """Show context menu for parts list."""
        if self.parts_listbox.size() > 0:
            # Select item under cursor
            index = self.parts_listbox.nearest(event.y)
            self.parts_listbox.selection_clear(0, tk.END)
            self.parts_listbox.selection_set(index)
            self.selected_part_index = index
            self._show_part_content(index)
            self.parts_context_menu.post(event.x_root, event.y_root)

    # ========================================================================
    # DIALOGS
    # ========================================================================

    def on_split_mode_change(self):
        """Handle split mode change."""
        mode = self.split_mode_var.get()
        if mode == "tokens":
            self.limit_label.config(text="  Max Tokens:")
            self.limit_entry.config(textvariable=self.max_tokens_var)
        elif mode == "lines":
            self.limit_label.config(text="  Max Lines:")
            self.limit_entry.config(textvariable=self.max_lines_var)
        else:  # parts
            self.limit_label.config(text="  # of Parts:")
            self.limit_entry.config(textvariable=self.num_parts_var)

    def show_settings_dialog(self):
        """Show settings dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("450x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0

        # Token settings
        ttk.Label(frame, text="Token Splitting Settings",
                  font=('Segoe UI', 11, 'bold')).grid(row=row, column=0, columnspan=2,
                                                      sticky=tk.W, pady=(0, 10))
        row += 1

        ttk.Label(frame, text="Max Tokens per Part:").grid(row=row, column=0, sticky=tk.W, pady=5)
        token_entry = ttk.Entry(frame, textvariable=self.max_tokens_var, width=15)
        token_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(frame, text="Target Ratio (0.0-1.0):").grid(row=row, column=0, sticky=tk.W, pady=5)
        ratio_entry = ttk.Entry(frame, textvariable=self.target_ratio_var, width=15)
        ratio_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Line settings
        ttk.Label(frame, text="Line Splitting Settings",
                  font=('Segoe UI', 11, 'bold')).grid(row=row, column=0, columnspan=2,
                                                      sticky=tk.W, pady=(20, 10))
        row += 1

        ttk.Label(frame, text="Max Lines per Part:").grid(row=row, column=0, sticky=tk.W, pady=5)
        lines_entry = ttk.Entry(frame, textvariable=self.max_lines_var, width=15)
        lines_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Parts settings
        ttk.Label(frame, text="N-Parts Splitting Settings",
                  font=('Segoe UI', 11, 'bold')).grid(row=row, column=0, columnspan=2,
                                                      sticky=tk.W, pady=(20, 10))
        row += 1

        ttk.Label(frame, text="Number of Parts:").grid(row=row, column=0, sticky=tk.W, pady=5)
        parts_entry = ttk.Entry(frame, textvariable=self.num_parts_var, width=15)
        parts_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(frame, text="(Intelligently splits without breaking\n function/method bodies)",
                  font=('Segoe UI', 8, 'italic'), foreground='gray').grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        row += 1

        # Comment removal
        ttk.Label(frame, text="Comment Removal",
                  font=('Segoe UI', 11, 'bold')).grid(row=row, column=0, columnspan=2,
                                                      sticky=tk.W, pady=(20, 10))
        row += 1

        ttk.Checkbutton(frame, text="Remove 3+ consecutive comment lines",
                        variable=self.remove_comments_var).grid(row=row, column=0,
                                                                columnspan=2, sticky=tk.W)
        row += 1

        # Syntax highlighting
        ttk.Checkbutton(frame, text="Enable syntax highlighting",
                        variable=self.syntax_highlight_var).grid(row=row, column=0,
                                                                 columnspan=2, sticky=tk.W, pady=(10, 0))
        row += 1

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=30)

        ttk.Button(btn_frame, text="OK", command=lambda: [self._save_settings(), dialog.destroy()]).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset Defaults",
                   command=self._reset_settings).pack(side=tk.LEFT, padx=5)

    def _reset_settings(self):
        """Reset settings to defaults."""
        self.max_tokens_var.set(8000)
        self.max_lines_var.set(200)
        self.num_parts_var.set(5)
        self.target_ratio_var.set(0.88)
        self.remove_comments_var.set(False)
        self.syntax_highlight_var.set(True)
        self._save_settings()

    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Python Code Splitter",
            "Python Code Splitter - Professional Edition\n\n"
            "Version 2.0\n\n"
            "A professional tool for splitting Python code into\n"
            "manageable parts for web UI pasting.\n\n"
            "Features:\n"
            "• AST-based code structure parsing\n"
            "• Split by tokens, lines, or N parts\n"
            "• Intelligent splitting (preserves functions/methods)\n"
            "• Syntax highlighting\n"
            "• Context menu for quick actions\n"
            "• Export as .py or .txt with indentation\n"
            "• Comment removal option"
        )

    def show_usage_guide(self):
        """Show usage guide dialog."""
        guide = tk.Toplevel(self.root)
        guide.title("Usage Guide")
        guide.geometry("650x600")
        guide.transient(self.root)

        text = ScrolledText(guide, wrap=tk.WORD, font=('Segoe UI', 10), padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)

        guide_text = """
PYTHON CODE SPLITTER - USAGE GUIDE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOADING FILES
• Click "Open" or use Ctrl+O to load a Python file
• The code structure (classes, functions, methods) appears in the left panel
• The source code is shown in the center panel with syntax highlighting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPLIT MODES

1. BY TOKENS (Default)
   • Splits based on approximate token count
   • Best for LLM context limits (ChatGPT, Claude, etc.)
   • Set "Max Tokens" to your target limit (e.g., 8000)

2. BY LINES
   • Simple split based on number of lines
   • Set "Max Lines" per part (e.g., 200)

3. BY PARTS (N) - INTELLIGENT SPLITTING ⭐
   • Specify exact number of parts you want (e.g., 5)
   • Automatically finds optimal split points
   • NEVER breaks inside a function or method body
   • Can split between class members
   • Creates balanced parts by token count

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPLITTING CODE
• Click "Split File" to split the entire file
• Double-click any item in the structure tree to split just that element
• Right-click for context menu with all split options

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPORT OPTIONS

• Export as .py: Standard Python file
• Export as .txt: Text file with header info and preserved indentation

Both options available for:
• Single parts
• All parts (to separate files)
• Individual classes/functions from the tree

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONTEXT MENUS (Right-Click)

Structure Tree:
• Split by tokens/lines/parts
• Copy code (with or without comments)
• Export as .py or .txt

Parts List:
• Copy part
• Copy without comments
• Export as .py or .txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMENT REMOVAL
Enable "Remove 3+ Comment Lines" to automatically strip blocks of 
consecutive comments (3 or more lines) when copying or exporting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYNTAX HIGHLIGHTING
• Toggle via View menu or toolbar checkbox
• Highlights keywords, strings, comments, decorators, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEYBOARD SHORTCUTS
• Ctrl+O: Open file
• Ctrl+S: Split file
• Ctrl+C: Copy selected part
• Ctrl+E: Export selected part
• Ctrl+Shift+E: Export all parts
• F5: Reload current file
"""
        text.insert("1.0", guide_text)
        text.config(state=tk.DISABLED)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    root = tk.Tk()

    app = CodeSplitterApp(root)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
