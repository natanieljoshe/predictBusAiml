import json

file_path = 'notebooks/03_modeling_and_validation.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "'is_holiday'" in source:
            new_source = []
            for line in cell['source']:
                if "features = ['route_encoded', 'hour', 'day_of_week', 'month', 'is_holiday']" in line:
                    new_source.append("features = ['route_encoded', 'hour', 'day_of_week', 'month']\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook features updated successfully!")
