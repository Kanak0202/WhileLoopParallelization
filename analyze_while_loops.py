import os
import re

SOURCE_DIR = "gpu-rodinia/openmp"

while_pattern = re.compile(r'\bwhile\s*\((.*?)\)')
increment_pattern = re.compile(r'(\+\+|--|\+=|-=)')
array_access_pattern = re.compile(r'\w+\s*\[.*?\]')
pointer_pattern = re.compile(r'->|\*')
function_call_pattern = re.compile(r'\w+\s*\(')


class LoopInfo:
    def __init__(self, depth):
        self.depth = depth
        self.canonical = False
        self.type = "unknown"


def classify_loop(body):

    if pointer_pattern.search(body):
        return "pointer_traversal"

    if "queue" in body or "stack" in body:
        return "data_structure"

    if "error" in body or "residual" in body or "converge" in body:
        return "iterative_solver"

    if array_access_pattern.search(body):
        return "array_computation"

    if function_call_pattern.search(body):
        return "function_driven"

    return "other"


def canonicalizable(condition, body):

    if increment_pattern.search(body):
        if re.search(r'[<>!=]', condition):
            return True

    return False


def analyze_file(filepath):

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    loops = []
    stack = []

    for i, line in enumerate(lines):

        match = while_pattern.search(line)

        if match:

            condition = match.group(1)
            depth = len(stack)

            stack.append(i)

            body = ""
            brace = 0

            for j in range(i, len(lines)):

                body += lines[j]

                brace += lines[j].count("{")
                brace -= lines[j].count("}")

                if brace == 0 and j > i:
                    break

            loop = LoopInfo(depth)

            if canonicalizable(condition, body):
                loop.canonical = True

            loop.type = classify_loop(body)

            loops.append(loop)

        if "}" in line and stack:
            stack.pop()

    return loops


def analyze_directory():

    results = []

    for root, dirs, files in os.walk(SOURCE_DIR):

        for file in files:

            if file.endswith(".c") or file.endswith(".cpp"):

                path = os.path.join(root, file)
                loops = analyze_file(path)

                results.extend(loops)

    return results


def summarize(loops):

    external = 0
    nested = 0
    max_depth = 0
    canonical = 0

    types = {}

    for loop in loops:

        if loop.depth == 0:
            external += 1
        else:
            nested += 1

        max_depth = max(max_depth, loop.depth)

        if loop.canonical:
            canonical += 1

        types[loop.type] = types.get(loop.type, 0) + 1

    print("===== WHILE LOOP ANALYSIS =====\n")

    print("Total while loops:", len(loops))
    print("External while loops:", external)
    print("Nested while loops:", nested)
    print("Maximum depth:", max_depth)
    print("Canonicalizable loops:", canonical)

    print("\nLoop type classification:")
    for t, c in types.items():
        print(f"{t}: {c}")


if __name__ == "__main__":

    loops = analyze_directory()
    summarize(loops)