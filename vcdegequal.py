import itertools
import tqdm


def popcount(n: int) -> int:
    """Calculates the population count (Hamming weight) of an integer."""
    return bin(n).count("1")


def calculate_degree(truth_table: list[int], n: int) -> int:
    """
    Calculates the degree of the unique real multilinear polynomial for f.

    This is also known as the Fourier degree. It's computed by finding the
    largest monomial with a non-zero coefficient in the polynomial that
    interpolates the function on {0, 1}^n.

    Args:
        truth_table: A list of 2^n integers (0 or 1).
        n: The number of boolean variables.

    Returns:
        The real degree of the function.
    """
    if not any(truth_table):  # Handle constant zero function
        return 0

    max_deg = 0
    # Iterate through all 2^n possible monomials (subsets of variables)
    for i in range(1 << n):
        # Calculate the coefficient for the monomial corresponding to mask 'i'
        # using the Mobius inversion formula on the boolean lattice.
        coeff = 0
        for j in range(1 << n):
            # Check if j is a submask of i (i.e., T is a subset of S)
            if (j & i) == j:
                f_val = truth_table[j]
                # Add (-1)^(|S|-|T|) * f(T) to the coefficient
                if (popcount(i) - popcount(j)) % 2 == 1:
                    coeff -= f_val
                else:
                    coeff += f_val

        if coeff != 0:
            max_deg = max(max_deg, popcount(i))

    return max_deg


def calculate_f2_degree(truth_table: list[int], n: int) -> int:
    """
    Calculates the F2-degree of a boolean function using its truth table.

    This is done by converting the function to its Algebraic Normal Form (ANF)
    using a Fast Walsh-Hadamard Transform. The degree is the size of the
    largest monomial in the ANF, which corresponds to the maximum popcount
    of an index with a non-zero coefficient.

    Args:
        truth_table: A list of 2^n integers (0 or 1) representing the
                     function's output for inputs 0 to 2^n - 1.
        n: The number of boolean variables.

    Returns:
        The F2-degree of the function.
    """
    if not any(truth_table):  # Handle constant zero function
        return 0

    # The coefficients of the ANF are calculated via a transform on the truth table
    coeffs = list(truth_table)
    for i in range(n):
        for j in range(len(coeffs)):
            # Apply the transform based on the subset property
            if (j >> i) & 1:
                coeffs[j] ^= coeffs[j ^ (1 << i)]

    max_deg = 0
    for i, coeff in enumerate(coeffs):
        if coeff == 1:
            # The degree of the monomial corresponding to this coefficient
            # is the number of variables in it (popcount of the index).
            max_deg = max(max_deg, popcount(i))

    return max_deg


def calculate_vc_dimension(truth_table: list[int], n: int) -> int:
    """
    Calculates the VC-dimension of the support of a boolean function.

    The support of f, S_f, is the set of inputs x for which f(x)=1. This
    function checks for the largest set of coordinates I that can be
    "shattered" by S_f. A set I is shattered if for every possible
    sub-pattern on those coordinates, there is an input in S_f that
    matches that sub-pattern.

    Args:
        truth_table: A list of 2^n integers (0 or 1).
        n: The number of boolean variables.

    Returns:
        The VC-dimension of the function's support.
    """
    # S_f is the set of integer inputs for which f is 1
    support_set = {i for i, val in enumerate(truth_table) if val == 1}

    if not support_set:
        return 0

    # We check for the largest shattered set, starting from size n
    for d in range(n, 0, -1):
        # Iterate through all possible subsets of coordinates of size d
        for coord_indices in itertools.combinations(range(n), d):

            projections = set()
            # For each input in the support set...
            for x in support_set:
                projection = 0
                # ...create the specific sub-pattern on the chosen coordinates
                for bit_pos, coord_idx in enumerate(coord_indices):
                    if (x >> coord_idx) & 1:
                        projection |= 1 << bit_pos
                projections.add(projection)

            # If we found 2^d unique patterns, the set is shattered
            if len(projections) == (1 << d):
                return d

    # If no set of coordinates of size > 0 is shattered
    return 0


def get_anf_representation(truth_table: list[int], n: int) -> str:
    """
    Computes the Algebraic Normal Form (ANF) representation of a boolean function.

    The representation is a polynomial over GF(2) using XOR (^) and AND (*).

    Args:
        truth_table: A list of 2^n integers (0 or 1).
        n: The number of boolean variables.

    Returns:
        A string representing the function in ANF (e.g., "x1 ^ x2*x3 ^ 1").
    """
    coeffs = list(truth_table)
    # Fast Walsh-Hadamard Transform to get ANF coefficients
    for i in range(n):
        for j in range(len(coeffs)):
            if (j >> i) & 1:
                coeffs[j] ^= coeffs[j ^ (1 << i)]

    monomials = []
    # i=0 is the constant term
    if coeffs[0] == 1:
        monomials.append("1")

    # Iterate through other coefficients to build monomials
    for i in range(1, len(coeffs)):
        if coeffs[i] == 1:
            term_vars = []
            for k in range(n):
                if (i >> k) & 1:
                    # Using 1-based indexing for variables x1, x2, ...
                    term_vars.append(f"x{n-k}")
            term_vars.reverse()
            monomials.append("*".join(term_vars))

    if not monomials:
        return "0"  # Constant zero function

    return " ^ ".join(monomials)


def get_polynomial_representation(truth_table: list[int], n: int) -> str:
    """
    Computes the real multilinear polynomial representation of a boolean function.

    Args:
        truth_table: A list of 2^n integers (0 or 1).
        n: The number of boolean variables.

    Returns:
        A string representing the function as a real polynomial (e.g., "x1 + x2 - 2*x1*x2").
    """
    terms = []
    # Iterate through all 2^n possible monomials (subsets of variables)
    for i in range(1 << n):
        # Calculate the coefficient for the monomial corresponding to mask 'i'
        # using the Mobius inversion formula on the boolean lattice.
        coeff = 0
        for j in range(1 << n):
            # Check if j is a submask of i (i.e., T is a subset of S)
            if (j & i) == j:
                f_val = truth_table[j]
                # Add (-1)^(|S|-|T|) * f(T) to the coefficient
                if (popcount(i) - popcount(j)) % 2 == 1:
                    coeff -= f_val
                else:
                    coeff += f_val

        if coeff == 0:
            continue

        # Build the monomial string
        if i == 0:  # Constant term
            terms.append(str(coeff))
            continue

        term_vars = []
        for k in range(n):
            if (i >> k) & 1:
                term_vars.append(f"x{n-k}")

        term_vars.reverse()  # To maintain order x1, x2, ...
        monomial = "*".join(term_vars)

        # Format the term string
        if coeff == 1:
            terms.append(f"+ {monomial}")
        elif coeff == -1:
            terms.append(f"- {monomial}")
        elif coeff > 0:
            terms.append(f"+ {coeff}*{monomial}")
        else:  # coeff < 0
            terms.append(f"- {-coeff}*{monomial}")

    if not terms:
        return "0"

    # Clean up the final string
    result = " ".join(terms)
    if result.startswith("+ "):
        result = result[2:]

    return result.strip()


def is_power_of_two(x: int) -> bool:
    """Checks if x is a power of two."""
    return (x > 0) and (x & (x - 1)) == 0


def run_checker(n: int, is_f2_degree: bool = False, verbose: bool = True) -> None:
    """
    Iterates through all 2^(2^n) boolean functions for n variables,
    checking if deg(f) + VC(f) = n.

    Args:
        n: The number of variables (e.g., 2, 3).
    """
    print(f"--- Checking for n = {n} ---\n")

    num_inputs = 2**n
    num_functions = 1 << num_inputs
    found_count = 0

    # Each 'i' represents a unique boolean function via its truth table
    if verbose:
        for i in range(num_functions):
            truth_table = [(i >> j) & 1 for j in range(num_inputs)]

            # Skip constant functions as per the problem statement
            # if all(v == 0 for v in truth_table) or all(v == 1 for v in truth_table):
            #    continue
            if is_f2_degree:
                deg = calculate_f2_degree(truth_table, n)
            else:
                deg = calculate_degree(truth_table, n)
            # deg = calculate_f2_degree(truth_table, n)
            vc_dim = calculate_vc_dimension(truth_table, n)

            if deg + vc_dim == n:
                found_count += 1
                anf_representation = get_anf_representation(truth_table, n)
                poly_representation = get_polynomial_representation(truth_table, n)
                # Format the truth table for readability
                inputs_str = [f"{bin(x)[2:]:>0{n}}" for x in range(num_inputs)]
                tt_str = (
                    f"  f({', '.join(f'x{k+1}' for k in range(n))}) | Truth Table\n"
                    f"  {'-' * (n*3 + 5)}|{'-' * 14}\n"
                )
                for j in range(num_inputs):
                    tt_str += f"  f({', '.join(inputs_str[j])}) = {truth_table[j]}\n"

                print(f"Found function #{found_count}:")
                print(f"  ANF Representation: f = {anf_representation}")
                print(f"  Polynomial Rep:     f = {poly_representation}")
                print(f"  deg(f) = {deg}, VC(f) = {vc_dim} => {deg} + {vc_dim} = {n}")
                print(tt_str)

    else:
        for i in tqdm.tqdm(range(num_functions), desc="Checking functions"):
            truth_table = [(i >> j) & 1 for j in range(num_inputs)]

            # Skip constant functions as per the problem statement
            # if all(v == 0 for v in truth_table) or all(v == 1 for v in truth_table):
            #    continue
            if is_f2_degree:
                deg = calculate_f2_degree(truth_table, n)
            else:
                deg = calculate_degree(truth_table, n)
            vc_dim = calculate_vc_dimension(truth_table, n)

            if deg + vc_dim == n:
                found_count += 1

    print(f"\n--- Total functions found for n={n}: {found_count} ---\n")


if __name__ == "__main__":
    # N is the number of variables
    N = 4
    # is_f2_degree = True to use F2-degree, False for real degree
    # verbose = True to print details, False for summary only
    run_checker(N, is_f2_degree=False, verbose=False)
