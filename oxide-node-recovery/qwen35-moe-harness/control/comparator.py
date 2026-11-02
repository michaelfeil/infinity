#!/usr/bin/env python3
"""Strict bit comparator for two harness reports. No tolerance by default."""
import argparse, json, math, struct, sys


def keyed(case):
    return case.get("name")


def canonical_bits(entry):
    values = entry["values"]
    bits = entry["bits_u32"]
    if len(values) != len(bits):
        raise ValueError("values/bits length mismatch")
    for value, bit in zip(values, bits):
        if not math.isfinite(value):
            raise ValueError("non-finite value")
        expected = struct.unpack("<I", struct.pack("<f", value))[0]
        if expected != bit:
            raise ValueError(f"bits do not encode value: {value} {bit} != {expected}")
    return tuple(bits)


def compare(actual, expected):
    failures = []
    a_cases = {keyed(c): c for c in actual.get("quality", {}).get("cases", [])}
    e_cases = {keyed(c): c for c in expected.get("quality", {}).get("cases", [])}
    if set(a_cases) != set(e_cases):
        failures.append(f"case set mismatch actual={sorted(a_cases)} expected={sorted(e_cases)}")
    for name in sorted(set(a_cases) & set(e_cases)):
        a, e = a_cases[name], e_cases[name]
        if a.get("status") != "ok":
            failures.append(f"{name}: non-ok status {a.get('status')}")
            continue
        if a.get("valid") and e.get("valid"):
            ao = a.get("output", {}).get("outputs", [])
            eo = e.get("output", {}).get("outputs", [])
            if len(ao) != len(eo):
                failures.append(f"{name}: output count")
                continue
            for x, y in zip(ao, eo):
                if (x.get("index"), x.get("kind"), x.get("shape")) != (y.get("index"), y.get("kind"), y.get("shape")):
                    failures.append(f"{name}: entry identity")
                    continue
                if canonical_bits(x) != canonical_bits(y):
                    failures.append(f"{name}: F32 bit mismatch at index {x.get('index')}")
        elif a.get("valid") != e.get("valid"):
            failures.append(f"{name}: validity mismatch")
        else:
            if a.get("actual_error_variant") != e.get("actual_error_variant"):
                failures.append(f"{name}: error variant mismatch")
    return failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actual", required=True)
    p.add_argument("--expected", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    actual = json.load(open(args.actual))
    expected = json.load(open(args.expected))
    failures = compare(actual, expected)
    result = {"schema": "oxide-q35moe-unified-comparator/v1", "strict_bits": True,
              "tolerance": 0, "ok": not failures, "failures": failures,
              "actual": args.actual, "expected": args.expected}
    if args.out:
        open(args.out, "w").write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
