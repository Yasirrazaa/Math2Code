import json
import os

from e2b_code_interpreter import CodeInterpreter
from tqdm import tqdm


def load_raw_data(filepath: str = "data/raw/synthetic_raw.json") -> list:
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []
    with open(filepath) as f:
        return json.load(f)


def curate_with_e2b(data: list) -> list:
    """
    Executes each Python code snippet in a secure E2B Code Interpreter sandbox.
    Keeps only the samples that execute successfully without raising exceptions.
    """
    curated_data = []

    # Ensure E2B_API_KEY is set in environment
    if not os.environ.get("E2B_API_KEY"):
        print(
            "Warning: E2B_API_KEY environment variable not set. Please set it before running in production."
        )
        # We'll continue anyway; the SDK will raise its own error if it truly needs it and can't find it.

    print("Starting E2B Code Interpreter Sandbox...")
    # Start the sandbox
    with CodeInterpreter() as sandbox:
        print("Sandbox started. Curating data...")

        # We need sympy available in the sandbox. The default CodeInterpreter has common ML libs,
        # but let's make sure sympy is installed.
        sandbox.notebook.exec_cell("!pip install sympy")

        for item in tqdm(data, desc="Curating samples"):
            code = item.get("python_code", "")
            if not code:
                continue

            # Execute the code snippet in the sandbox
            execution = sandbox.notebook.exec_cell(code)

            # If there's an error, we discard the sample
            if execution.error:
                # Execution failed
                continue

            # If successful, keep the sample
            curated_data.append(item)

    return curated_data


def main() -> None:
    raw_data = load_raw_data()
    if not raw_data:
        return

    print(f"Loaded {len(raw_data)} raw samples.")

    curated_data = curate_with_e2b(raw_data)

    print(
        f"Curation complete. Kept {len(curated_data)} out of {len(raw_data)} samples."
    )

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/synthetic_curated.json"
    with open(output_path, "w") as f:
        json.dump(curated_data, f, indent=2)

    print(f"Saved curated data to {output_path}")


if __name__ == "__main__":
    main()
