from pathlib import Path

REAL_PATH = Path("../dataset/real")
FAKE_PATH = Path("../dataset/fake")

real = list(REAL_PATH.glob("*.jpg"))
fake = list(FAKE_PATH.glob("*.jpg"))

print("=" * 40)
print("DATASET ANALYSIS")
print("=" * 40)

print(f"Real Images : {len(real)}")
print(f"Fake Images : {len(fake)}")

total = len(real) + len(fake)

print(f"Total Images : {total}")

print(f"Real % : {len(real)/total*100:.2f}")

print(f"Fake % : {len(fake)/total*100:.2f}")