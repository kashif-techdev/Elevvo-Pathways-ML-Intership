"""
Fix Unicode characters in training scripts
"""

import re

def fix_unicode_in_file(file_path):
    """Fix Unicode characters in a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace common Unicode characters
    replacements = {
        '🎵': '[MUSIC]',
        '📊': '[CHART]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '💾': '[SAVED]',
        '🎯': '[TARGET]',
        '🎼': '[SCORE]',
        '🎸': '[GUITAR]',
        '🔧': '[TOOL]',
        '📈': '[GRAPH]',
        '🎨': '[ART]'
    }
    
    for unicode_char, replacement in replacements.items():
        content = content.replace(unicode_char, replacement)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed Unicode characters in {file_path}")

if __name__ == "__main__":
    files_to_fix = [
        "scripts/train_tabular_models.py",
        "scripts/train_cnn_models.py",
        "scripts/transfer_learning.py",
        "scripts/evaluate_models.py",
        "scripts/compare_approaches.py"
    ]
    
    for file_path in files_to_fix:
        try:
            fix_unicode_in_file(file_path)
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")

