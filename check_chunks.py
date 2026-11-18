import json, re, sys
ch = json.load(open(sys.argv[1], encoding="utf-8"))["chunks"]
t = [c["text"].strip() for c in ch]
print("chunks:", len(t))
print("end with Fig.:", sum(bool(re.search(r"\bFig\.$", x)) for x in t))
print("end with III./IV.:", sum(bool(re.search(r"(?<!Table )\b[IVX]{2,}\.$", x)) for x in t))
print("footer junk:", sum("IEEE Asia" in x or "978-" in x for x in t))