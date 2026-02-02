import os
import difflib
from pathlib import Path

def get_python_files(folder):
    """Reads all .py files in a folder into a dictionary {filename: content}"""
    files = {}
    for r, d, f in os.walk(folder):
        for file in f:
            if file.endswith(".py") and "__init__" not in file:
                # Store full relative path as key to avoid name collisions
                path = Path(r) / file
                try:
                    files[file] = path.read_text(encoding='utf-8')
                except Exception as e:
                    print(f"Skipping {file}: {e}")
    return files

def audit():
    print("🔍 SCANNING FOR REDUNDANCY: MoonWire (Crypto) vs AlphaEngine (Equities)...")
    print("-" * 65)
    
    # 1. Load the Code
    crypto_files = get_python_files("moonwire")
    equity_files = get_python_files("alpha_engine")
    
    if not crypto_files or not equity_files:
        print("⚠️  Error: Could not find files. Are you in the root 'IteraDynamics_Mono' folder?")
        return

    # 2. Find common filenames
    common_names = set(crypto_files.keys()) & set(equity_files.keys())
    
    print(f"{'File Name':<30} | {'Similarity':<10} | {'Action'}")
    print("-" * 65)
    
    # 3. Compare them
    found_issues = False
    for fname in sorted(common_names):
        code_a = crypto_files[fname]
        code_b = equity_files[fname]
        
        # Calculate similarity ratio (0.0 to 1.0)
        seq = difflib.SequenceMatcher(None, code_a, code_b)
        ratio = seq.ratio() * 100
        
        status = ""
        if ratio > 85:
            status = "🔴 MOVE TO CORE (Identical)"
            found_issues = True
        elif ratio > 40:
            status = "🟡 ABSTRACT (Similar Logic)"
            found_issues = True
        else:
            status = "🟢 OK (Different)"
            
        print(f"{fname:<30} | {ratio:>5.1f}%     | {status}")

    print("-" * 65)
    if not found_issues:
        print("✅ Clean! No obvious redundancy found.")
    else:
        print("🔧 Recommendation: Move '🔴' items to apex_core immediately.")

if __name__ == "__main__":
    audit()