from itertools import combinations, chain


def powerset(iterable):
    """Return all subsets of an iterable as tuples."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def is_shattered(F, X):
    """Check if subset X is shattered by family F."""
    X = set(X)
    needed = {frozenset(sub) for sub in powerset(X)}
    realized = set()
    for subset in F:
        realized.add(frozenset(set(subset) & X))
    return needed <= realized


def vc_dimension(F, n, print_shattered=True):
    """Compute VC dimension of family F on [n]."""
    max_dim = 0
    ground = list(range(1, n + 1))
    shattered_sets = []
    for r in range(1, n + 1):
        for X in combinations(ground, r):
            if is_shattered(F, X):
                max_dim = max(max_dim, r)
                shattered_sets.append(set(X))
    if print_shattered:
        print("Shattered sets:", shattered_sets)
    return max_dim


def sensitivity(F, n):
    """Compute sensitivity of family F on [n]."""
    F_set = {frozenset(s) for s in F}  # normalize
    ground = list(range(1, n + 1))
    max_sens = 0
    max_sens_set = set()
    for x in powerset(ground):
        S = set(x)
        val = frozenset(S) in F_set
        local_sens = 0
        for i in ground:
            T = frozenset(S ^ {i})  # flip element i
            if (T in F_set) != val:
                local_sens += 1
        if local_sens > max_sens:
            max_sens_set = S
        max_sens = max(max_sens, local_sens)
    return max_sens, max_sens_set


def complement_family(F, n):
    """Return the complement set family of F over [n]."""
    ground = list(range(1, n + 1))
    all_subsets = {frozenset(s) for s in powerset(ground)}
    F_set = {frozenset(s) for s in F}
    complement = all_subsets - F_set
    return [set(s) for s in complement]


from itertools import product


def bits_to_set(bits):
    """Convert n-bit binary vector to subset of [1..n]."""
    return {i + 1 for i, b in enumerate(bits) if b == 1}


def generate_C1():
    suffix = (0, 0, 0, 0, 0, 0)
    return [bits_to_set(prefix + suffix) for prefix in product([0, 1], repeat=3)]


def generate_C2():
    prefix = (0, 0, 0)
    suffix = (1, 1, 1)
    return [
        bits_to_set(prefix + middle + suffix) for middle in product([0, 1], repeat=3)
    ]


def generate_C3():
    prefix = (1, 1, 1, 1, 1, 1)
    return [bits_to_set(prefix + suffix) for suffix in product([0, 1], repeat=3)]


def generate_union_family():
    C1 = generate_C1()
    C2 = generate_C2()
    C3 = generate_C3()
    union_of_cubes = C1 + C2 + C3
    all_sets = complement_family(union_of_cubes, 9)
    return [set(s) for s in all_sets]


# Check the specific family
F = generate_union_family()
print("Size of family F:", len(F))
print("VC dimension:", vc_dimension(F, 9, print_shattered=False))
print("Sensitivity:", sensitivity(F, 15)[0])
print("Point of max sensitivity:", sensitivity(F, 15)[1])
