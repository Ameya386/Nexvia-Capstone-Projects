"""Run the full pipeline:  python run_all.py --dataset small [--spark] [--synthetic]"""
import argparse, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="small", choices=["small", "25m"])
ap.add_argument("--synthetic", action="store_true", help="offline fake data (smoke test only)")
ap.add_argument("--spark", action="store_true", help="use Apache Spark for the feature tables")
a = ap.parse_args()

steps = [["src.data_prep", "--dataset", a.dataset] + (["--synthetic"] if a.synthetic else []),
         ["src.spark_features"] if a.spark else ["src.features"],
         ["src.train_retrieval"], ["src.build_index"], ["src.train_ranker"], ["src.evaluate"], ["src.ab_test"]]
for s in steps:
    print("\n>>>", " ".join(s))
    subprocess.run([sys.executable, "-m", *s], check=True)
print("\nDone. Demo:  streamlit run app/streamlit_app.py")
