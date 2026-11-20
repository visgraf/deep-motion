"""K19 (#17) on R2, run by Code: Chat's command, `python -m deep stage7a --b2 line=runs/r2-line arc=runs/r2-arc --rung r2
--out docs/runs/issue-17`, stops at argparse (deep/__main__.py at e27ca80: --rung accepts only r0, r1; exit 2). This calls
the CLI's own stage7a() with that command's arguments, rung r2 included; deep/ is unchanged. The record it writes,
stage7a_b2_r2.json, is the one the command would write.
Usage: PYTHONPATH=. .venv/bin/python docs/runs/issue-17/run_k19.py"""
import argparse
import sys

from deep.__main__ import stage7a

a = argparse.Namespace(b2=["line=runs/r2-line", "arc=runs/r2-arc"], rung="r2", max_anchors=20000,
                       out="docs/runs/issue-17")
sys.exit(stage7a(a))
