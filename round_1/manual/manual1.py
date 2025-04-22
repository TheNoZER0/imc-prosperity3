import itertools

# List of instruments (sn, p, si, se)
instruments = ["sn", "p", "si", "se"]

# Trade prices matrix [from][to]
prices = {
    "sn": {"sn": 1,   "p": 1.45, "si": 0.52, "se": 0.72},
    "p":  {"sn": 0.7, "p": 1,    "si": 0.31, "se": 0.48},
    "si": {"sn": 1.95, "p": 3.1, "si": 1,   "se": 1.49},
    "se": {"sn": 1.34, "p": 1.98, "si": 0.64, "se": 1},
}

def simulate_trade(path, start_amount=2000):
    amount = start_amount
    for i in range(len(path) - 1):
        frm = path[i]
        to = path[i + 1]
        amount *= prices[frm][to]
    return amount

best_path = None
best_amount = 0

start = "se"
end = "se"
middle_steps = 4  # total path length = 6 (start + 4 trades + end)

for middle in itertools.product(instruments, repeat=middle_steps):
    full_path = [start] + list(middle) + [end]
    amount = simulate_trade(full_path)
    if amount > best_amount:
        best_amount = amount
        best_path = full_path

print("Best path:", " -> ".join(best_path))
print("Final amount:", best_amount)
