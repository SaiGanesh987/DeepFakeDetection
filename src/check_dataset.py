from pathlib import Path

train_real = Path("rvf10k/train/real")
train_fake = Path("rvf10k/train/fake")

valid_real = Path("rvf10k/valid/real")
valid_fake = Path("rvf10k/valid/fake")

print("=" * 40)
print("TRAIN SET")
print("=" * 40)

print("Real :", len(list(train_real.glob("*"))))
print("Fake :", len(list(train_fake.glob("*"))))

print()

print("=" * 40)
print("VALIDATION SET")
print("=" * 40)

print("Real :", len(list(valid_real.glob("*"))))
print("Fake :", len(list(valid_fake.glob("*"))))