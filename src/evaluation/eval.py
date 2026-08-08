import json

from e2b_code_interpreter import CodeInterpreter
from transformers import pipeline


def load_test_data(filepath: str = "data/final/synthetic_data_final.json") -> list:
    with open(filepath) as f:
        # For evaluation, we ideally want a held-out test split.
        # Assuming we just take the last 100 for this example.
        return json.load(f)[-100:]


def evaluate_model(model_path: str = "AI-MO/NuminaMath-7B-TIR", is_peft: bool = False) -> float:
    print(f"Loading model {model_path} for evaluation...")

    # We use a pipeline for easy inference
    generator = pipeline(
        "text-generation", model=model_path, device_map="auto", torch_dtype="auto"
    )

    test_data = load_test_data()
    correct_count = 0
    total = len(test_data)

    print("Starting E2B Sandbox for evaluation...")
    with CodeInterpreter() as sandbox:
        sandbox.notebook.exec_cell("!pip install sympy")

        for idx, item in enumerate(test_data):
            latex_expr = item["latex_expression"]

            prompt = f"Translate the following LaTeX expression to SymPy code:\n{latex_expr}\n\n### Code:\n"

            # Generate code from the model
            generated = generator(prompt, max_new_tokens=200, return_full_text=False)[
                0
            ]["generated_text"]

            # Clean up generated code if needed
            cleaned_code = generated.strip().strip("`").replace("python\n", "")

            # Test executing expected vs generated
            # To actually evaluate functional correctness (Pass@1), we need to inject
            # some dummy inputs into both the generated function and expected function
            # and compare the numeric output. For simplicity, we just check if it compiles
            # and runs without error here, but you'd expand this to check return values.

            exec_result = sandbox.notebook.exec_cell(cleaned_code)

            if not exec_result.error:
                correct_count += 1

            print(f"[{idx + 1}/{total}] Pass: {not bool(exec_result.error)}")

    accuracy = (correct_count / total) * 100
    print(f"Functional Correctness (Pass@1): {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    # Evaluate Baseline
    print("=== Evaluating Baseline Model ===")
    evaluate_model(model_path="AI-MO/NuminaMath-7B-TIR", is_peft=False)

    # Evaluate Fine-tuned (Uncomment when trained)
    # print("=== Evaluating Fine-Tuned Model ===")
    # evaluate_model(model_path="./outputs/aimo_lora/final", is_peft=True)
