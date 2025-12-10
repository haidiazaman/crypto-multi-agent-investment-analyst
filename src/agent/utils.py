import ast


def convert_dict_string_to_dict(string: str):
    try:
        return ast.literal_eval(string)
    except (SyntaxError, ValueError) as e:
        print(f"Failed to parse dict string: {e}")
        raise