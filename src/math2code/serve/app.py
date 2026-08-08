import gradio as gr
import requests

API_URL = "http://localhost:8000/generate"


def process_latex(latex_input: str) -> str:
    try:
        response = requests.post(API_URL, json={"latex_expression": latex_input})
        if response.status_code == 200:
            data = response.json()
            code = data.get("python_code", "")
            result = data.get("execution_result", "")
            error = data.get("error", "")

            output_text = f"### Generated Code:\n```python\n{code}\n```\n\n### Execution Output:\n{result}"
            if error:
                output_text += f"\n\n### Execution Error:\n```\n{error}\n```"

            return output_text
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error connecting to API: {str(e)}"


with gr.Blocks(title="Math2Code") as demo:
    gr.Markdown("# LaTeX to Executable Python Code")
    gr.Markdown(
        "Translate mathematical expressions into executable SymPy code using our fine-tuned NuminaMath-7B-TIR model."
    )

    with gr.Row():
        with gr.Column():
            latex_input = gr.Textbox(
                lines=5,
                placeholder="\\frac{x^{2} + 3y^{2}}{2x + 5y}",
                label="Input LaTeX Expression",
            )
            submit_btn = gr.Button("Generate & Execute")

        with gr.Column():
            output_display = gr.Markdown(label="Output")

    submit_btn.click(fn=process_latex, inputs=latex_input, outputs=output_display)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501)
