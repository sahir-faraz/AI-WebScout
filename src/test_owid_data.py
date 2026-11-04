
from owid_data import fetch_owid_chart

url = (
    "https://ourworldindata.org/grapher/"
    "annual-co2-emissions-per-country"
)

df, metadata, error = fetch_owid_chart(url)

if error:
    print("ERROR:", error)
else:
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMetadata:")
    print(metadata)

    print("\nYear range:", df["Year"].min(), "-", df["Year"].max())
    print("Number of rows:", len(df))
    country = "India"
    year = 2020
    value_column = "Annual CO₂ emissions"

    result = df.loc[
        (df["Entity"] == country) & (df["Year"] == year),
        value_column
    ]

    if result.empty:
        print(f"No data found for {country} in {year}")
    else:
        print(f"{country} CO₂ emissions in {year}: {result.iloc[0]:,.0f} tonnes")