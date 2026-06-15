import json

file_path = 'notebooks/03_modeling_and_validation.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell to replace
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "pd.read_csv('../data/raw/mta_bus_all_routes_weekly.csv')" in source:
            # Replace the source
            new_source = [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error\n",
                "\n",
                "plt.style.use('seaborn-v0_8-darkgrid')\n",
                "\n",
                "# 1. LOAD DATA 2023 & 2024\n",
                "train_df = pd.read_csv('../data/raw/mta_bus_all_routes_weekly.csv')\n",
                "test_df = pd.read_csv('../data/raw/mta_bus_all_routes_weekly_2024.csv')\n",
                "train_df['transit_timestamp'] = pd.to_datetime(train_df['transit_timestamp'])\n",
                "test_df['transit_timestamp'] = pd.to_datetime(test_df['transit_timestamp'])\n",
                "\n",
                "# 2. TARGET ENCODING (Mencegah Kebocoran Masa Depan)\n",
                "route_target_mean = train_df.groupby('bus_route')['total_ridership'].mean().to_dict()\n",
                "train_df['route_encoded'] = train_df['bus_route'].map(route_target_mean)\n",
                "test_df['route_encoded']  = test_df['bus_route'].map(route_target_mean).fillna(0)\n",
                "\n",
                "features = ['route_encoded', 'hour', 'day_of_week', 'month', 'is_holiday']\n",
                "X_train, y_train = train_df[features], train_df['total_ridership']\n",
                "X_test, y_test   = test_df[features], test_df['total_ridership']\n",
                "\n",
                "print(f\"Data Pembelajaran (2023) : {train_df.shape[0]:,} baris\")\n",
                "print(f\"Data Pengujian (2024)    : {test_df.shape[0]:,} baris\")"
            ]
            cell['source'] = new_source
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully!")
