from transformers import AutoTokenizer
import argparse

def main(args):
    model_dir = args.model_dir

    chat_template = """{%- if messages[0]['role'] == 'system' %}
                            {%- if messages[0]['content'] is string %}
                                {%- set system_message = messages[0]['content']|trim %}
                            {%- else %}
                                {%- set system_message = messages[0]['content'][0]['text']|trim %}
                            {%- endif %}
                            {%- set messages = messages[1:] %}
                        {%- else %}
                            {%- if tools is not none %}
                                {%- set system_message = "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question." %}
                            {%- else %}
                                {%- set system_message = "" %}
                            {%- endif %}
                        {%- endif %}"""

    # Cargar el tokenizador del modelo base
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Definir el Chat Template
    tokenizer.chat_template = chat_template

    # Guardar el tokenizador con el nuevo chat template
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_dir", type=str, help="Path of model", default="")
    args = parser.parse_args()
    main(args)