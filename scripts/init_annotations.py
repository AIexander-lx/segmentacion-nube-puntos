"""Inicializa plantilla vacía de anotación manual a partir de un ROI."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from segmentacion.evaluation import create_annotation
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--roi",type=Path,required=True);p.add_argument("--region",required=True);p.add_argument("--categories",nargs="*",default=[]);p.add_argument("--output",type=Path,required=True);p.add_argument("--max-points",type=int,default=500);p.add_argument("--seed",type=int,default=42);a=p.parse_args()
 if a.max_points<1:raise ValueError("--max-points debe ser positivo")
 with np.load(a.roi) as data:
  if "original_indices" in data:ids=data["original_indices"]
  elif "resolution_indices" in data and data["resolution"].tolist()==["original"]:ids=data["resolution_indices"]
  else:raise ValueError("El ROI reducido no contiene original_indices")
 if len(ids)>a.max_points:ids=np.sort(np.random.default_rng(a.seed).choice(ids,a.max_points,replace=False))
 out=a.output if a.output.is_absolute() else ROOT/a.output
 print(create_annotation(out,ids,region=a.region,categories=a.categories,provenance=f"empty manual template; deterministic subset seed={a.seed}"))
if __name__=="__main__":main()
