<img width="1920" height="1128" alt="image" src="https://github.com/user-attachments/assets/08c76a68-e580-4648-9e53-f853e4b34c47" />
# Python Code Splitter

A professional GUI tool for splitting Python source files into manageable parts for pasting into web UIs (ChatGPT, Claude, etc.) without exceeding token limits.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

### 🔍 Code Structure Parsing
- AST-based parsing of Python source files
- Tree view displaying classes, functions, and methods
- Line count and token estimation for each element

### ✂️ Three Split Modes

| Mode | Description |
|------|-------------|
| **Tokens** | Split based on approximate token count (best for LLM context limits) |
| **Lines** | Split based on maximum lines per part |
| **Parts (N)** | Intelligently split into exactly N parts without breaking function/method bodies |

### 🎨 Syntax Highlighting
- Keywords, builtins, strings, comments
- Decorators, numbers, class/function names
- Toggle on/off for performance

### 📋 Copy & Export Options
- Copy individual parts or all parts to clipboard
- Export as `.py` (Python) or `.txt` (with header and indentation)
- Option to remove 3+ consecutive comment lines

### 🖱️ Context Menus
- Right-click on structure tree for quick actions
- Right-click on parts list for copy/export options

### ⚙️ Persistent Settings
- Settings saved to `splitter.settings` file
- Automatically loads on startup

## Installation

### Requirements
- Python 3.8 or higher
- Tkinter (included with Python on Windows/macOS)

### Linux (if Tkinter not installed)
```bash
# Debian/Ubuntu
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
###

python code_splitter.py
```

Usage

Basic Workflow
Open a Python file

Click 📂 Open or press Ctrl+O
Select a .py file
View code structure

Left panel shows classes, functions, methods
Click items to preview their code
Choose split mode

Select tokens, lines, or parts from dropdown
Set the limit value (max tokens, max lines, or number of parts)
Split the code

Click ✂️ Split File or press Ctrl+S
Parts appear in the right panel
Copy or export parts

Select a part and click 📋 Copy Part
Or use 💾 Export .py / 📄 Export .txt
Keyboard Shortcuts
Shortcut	Action
Ctrl+O	Open file
Ctrl+S	Split file
Ctrl+C	Copy selected part
Ctrl+E	Export selected part
Ctrl+Shift+E	Export all parts
F5	Reload current file
Split Mode Details
Token Mode (Default)

Best for LLM context limits. Uses a conservative approximation that slightly overestimates to prevent overflow.

## Recommended limits:
- GPT-3.5: 4000 tokens
- GPT-4: 8000 tokens
- Claude: 8000-10000 tokens
Line Mode
Simple split by line count. May break code at arbitrary points.

Parts Mode (Intelligent) ⭐
Specify exact number of parts. The algorithm:

Never breaks inside function/method bodies
Can split between class members
Balances parts by token count
Falls back gracefully if fewer splits possible
Comment Removal
Enable Remove 3+ Comment Lines to automatically strip blocks of consecutive comments when copying or exporting. Useful for reducing token count.

Before:

Python
```
# ============================================
# This is a header comment block
# Author: John Doe
# Date: 2024-01-01
# ============================================
def my_function():
    pass
After:

Python

def my_function():
    pass
Configuration
Settings are stored in splitter.settings:
```

# Python Code Splitter Settings
<code>
max_tokens=8000
max_lines=200
num_parts=5
target_ratio=0.88
split_mode=tokens
remove_comments=False
syntax_highlight=True

    Settings Description
</code>
Setting	Default	Description

max_tokens	| 8000	| Maximum tokens per part (token mode)<p>
max_lines	|200	| Maximum lines per part (line mode)<p>
num_parts	|5	|Target number of parts (parts mode)<p>
target_ratio	|0.88	| Headroom ratio for token mode (0.88 = 88% of max)<p>
split_mode	|tokens	| Default split mode<p>
remove_comments	|False	| Auto-remove comment blocks<p>
syntax_highlight	|True	| Enable syntax highlighting<p>

Export Formats
Python (.py)
Raw Python code, ready to paste or execute.

Text (.txt)
Includes header with metadata:


```
======================================================================
  Exported from Python Code Splitter
  Original file: my_script.py
  Lines: 150 | Tokens: ~2500
======================================================================

def my_function():
    ...
```

This tends to slightly overestimate, providing safety margin for different tokenizers.

Troubleshooting

Emojis not showing in color
This is a Tkinter limitation on some systems. Functionality is unaffected.

File won't parse
Ensure the file has valid Python syntax. The AST parser will show an error for syntax issues.

Large files are slow
Disable syntax highlighting for better performance on files >5000 lines.

Project Structure

<text>

code_splitter/
├── code_splitter.py    # Main application
├── splitter.settings   # User settings (auto-generated)
└── README.md           # This file
</text>

Dependencies
Standard Library Only - No external packages required!

tkinter - GUI framework
ast - Python code parsing
re - Regular expressions
pathlib - File path handling
dataclasses - Data structures
Contributing
Fork the repository
Create a feature branch
Make your changes
Submit a pull request
License
MIT License - See LICENSE file for details.

Acknowledgments
Built with Python and Tkinter
AST module for intelligent code parsing
Inspired by the need to share code with LLMs efficiently
Made with ❤️ for developers who paste code into AI assistants

