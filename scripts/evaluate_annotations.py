"""Evalúa instancias solo sobre anotaciones manuales ya completadas."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from segmentacion.evaluation import read_annotation,evaluate_instances,semantic_annotation_summary
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--annotations",type=Path,required=True);p.add_argument("--labels",type=Path,required=True,help="labels.npz con original_labels");p.add_argument("--output",type=Path);a=p.parse_args();ann=read_annotation(a.annotations)
 with np.load(a.labels) as data:
  if "original_labels" not in data:raise ValueError("labels.npz requiere original_labels")
  labels=data["original_labels"]
 report={"instances":evaluate_instances(ann,labels),"semantic_interface":semantic_annotation_summary(ann)};text=json.dumps(report,indent=2,ensure_ascii=False)
 if a.output:(a.output if a.output.is_absolute() else ROOT/a.output).write_text(text,encoding="utf-8")
 print(text)
if __name__=="__main__":main()
