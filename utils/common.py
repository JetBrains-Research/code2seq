from typing import Dict, List


def print_table(data: Dict[str, List[str]]):
    row_lens = [max(len(header), max([len(s) for s in values])) for header, values in data.items()]
    row_template = "".join(["| {:<" + str(i) + "} " for i in row_lens]) + "|"
    headers = [key for key in data.keys()]
    max_data_per_col = max([len(v) for v in data.values()])
    row_data = []
    for i in range(max_data_per_col):
        row_data.append([v[i] if len(v) > i else "" for k, v in data.items()])

    header_line = row_template.format(*headers)
    delimiter_line = "".join("|" + "-" * (i + 2) for i in row_lens) + "|"
    row_lines = [row_template.format(*row) for row in row_data]
    print(header_line, delimiter_line, *row_lines, delimiter_line, sep="\n")
