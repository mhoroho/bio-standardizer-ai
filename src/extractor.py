import json, argparse
from pathlib import Path
from parser import parse_bio_from_file

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Heuristic parser CLI")
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    bio = parse_bio_from_file(Path(args.file))
    print(json.dumps(bio.model_dump(mode="json"), indent=2, ensure_ascii=False))