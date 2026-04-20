import argparse
import featuremind as fm

def main():
    parser = argparse.ArgumentParser(
        description="🧠 featuremind CLI"
    )

    parser.add_argument("file", help="CSV file path")
    parser.add_argument("--target", default=None)
    parser.add_argument("--mode", default="analyze",
                        choices=["analyze", "train", "leakage", "full"])

    args = parser.parse_args()

    # ─────────────────────────────────────────────
    if args.mode == "analyze":
        fm.analyze(args.file, target=args.target)

    elif args.mode == "train":
        pipeline = fm.train(args.file, target=args.target)
        pipeline.save("featuremind_pipeline")

    elif args.mode == "leakage":
        fm.check_leakage(args.file, target=args.target)

    # 🔥 NEW: FULL MODE (like test.py)
    elif args.mode == "full":
        print("\n🚀 Running FULL pipeline...\n")

        fm.analyze(args.file, target=args.target)

        print("\n🛡️ Running Leakage Check...\n")
        fm.check_leakage(args.file, target=args.target)

        print("\n🏗️ Training Pipeline...\n")
        pipeline = fm.train(args.file, target=args.target)
        pipeline.save("featuremind_pipeline")

        print("\n✅ Done!")

    else:
        print("❌ Invalid mode")