import os
from chardet import detect

def fix_all_encodings(folder="transcripts"):
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            
            with open(filepath, "rb") as f:
                raw = f.read()
                result = detect(raw)
                encoding = result["encoding"]
            
            # لو مش UTF-8 صلحه
            if not encoding or encoding.lower() == "ascii":
                encoding = "windows-1256"
            
            if encoding.lower() != "utf-8":
                try:
                    content = raw.decode(encoding)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"✅ Fixed: {filename} ({encoding} -> utf-8)")
                except:
                    try:
                        content = raw.decode("windows-1256")
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(content)
                        print(f"✅ Fixed with windows-1256: {filename}")
                    except:
                        print(f"❌ Failed: {filename}")
            else:
                print(f"✓ Already UTF-8: {filename}")

fix_all_encodings()