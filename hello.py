# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from io import StringIO
import folium
from folium import plugins
from branca.colormap import linear
from folium.plugins import MarkerCluster
import requests


# %%
import os
os.chdir('C:/Users/dshab/Downloads')  # Dit maakt de Downloads-map de werkdirectory


# %%
# Als het weer nodig is, check je huidige working directory
import os
print(os.getcwd())

# Pas je working directory aan indien gewenst
path = "C:\\Users\\dshab\\downloads" # zet hier wat voor jou goed is
os.chdir(path)
print(os.getcwd())

# %%
london_tube_lines = pd.read_csv("London_tube_lines.csv")
print(london_tube_lines.head())


# %%
london_stations_csv = pd.read_csv("London_stations.csv")
print(london_stations_csv.head())

# %%
# Inladen van de JSON-bestanden
london_stations = pd.read_json("London_stations.json")
print(london_stations.head())


# %%
london_train_lines = pd.read_json("London_train_lines.json")
print(london_train_lines.head())

# %%
tfl_stations = pd.read_csv(r"C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Data\TfL_stations.csv")
print(tfl_stations.head())

# %%

# Laad de dataset
tfl_stations = pd.read_csv(r"C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Data\TfL_stations.csv")

# Bekijk de eerste paar rijen van de dataset
print("Dataset voor verwijderen van NaN-waarden:")
print(tfl_stations.head())

# Verwijder NaN-waarden uit de dataset
tfl_stations = tfl_stations.dropna()

# Bekijk de dataset na het verwijderen van NaN-waarden
print("\nDataset na verwijderen van NaN-waarden:")
print(tfl_stations.head())


# %%


# Laad de dataset
tfl_stations = pd.read_csv(r"C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Data\TfL_stations.csv")

# Vervang lege waarden of lege strings met NaN
tfl_stations.replace("", np.nan, inplace=True)

# Verwijder alle rijen met NaN-waarden
tfl_stations = tfl_stations.dropna()

# Bekijk de dataset na het verwijderen van NaN-waarden
print("Dataset na verwijderen van NaN-waarden:")
print(tfl_stations.head())


# %%
import pandas as pd

# Laad de dataset
tfl_stations = pd.read_csv(r"C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Data\TfL_stations.csv")

# Vervang lege waarden of lege strings met NaN (indien nodig)
tfl_stations.replace("", np.nan, inplace=True)

# Verwijder NaN-waarden
tfl_stations = tfl_stations.dropna()

# Controleer op duplicaten
tfl_stations = tfl_stations.duplicated()

# Verwijder duplicaten
tfl_stations = tfl_stations.drop_duplicates()

# Bekijk de dataset na het verwijderen van duplicaten
print("Dataset na verwijderen van duplicaten:")
print(tfl_stations.head())


# %%
stations_20220221 = pd.read_csv(r'C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Geodata\Stations_20220221.csv')
print(stations_20220221.head())

# %%
tfl_lines = pd.read_json(r'C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Geodata\tfl_lines.json')
print(tfl_lines.head())


# %%
night_tube = gpd.read_file(r'C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Geodata\night_tube.geojson')
print(night_tube.head())


# %%
lu_lines = gpd.read_file(r'C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Geodata\lu_lines.geojson')
print(lu_lines.head())

# %% [markdown]
# Algemene inspectie van de datasets
# 
# 1. Hoeveel rijen en kolommen elke dataset heeft.
# 2. Of er missende waarden (NA) zijn.
# 3. Of er duplicaten zijn.
# 4. Of er vreemde waarden (outliers) zijn.

# %%
# Algemene info per dataset
datasets = {
    "london_tube_lines": london_tube_lines,
    "london_stations_csv": london_stations_csv,
    "london_stations": london_stations,
    "london_train_lines": london_train_lines,
    "tfl_stations": tfl_stations,
    "stations_20220221": stations_20220221,
}

for name, df in datasets.items():
    print(f"\nDataset: {name}")
    print(df.info())  # Basisinformatie over de kolommen en missende waarden
    print(df.describe())  # Statistische samenvatting (voor numerieke kolommen)
    print(df.head())  # Eerste paar rijen om te begrijpen hoe de data eruitziet


# %% [markdown]
# Controleren op duplicaten

# %%
# Controleren op exacte duplicaten
duplicates = london_tube_lines.duplicated()
print(f"Aantal exacte duplicaten: {duplicates.sum()}")

# Weergeven van enkele duplicaten (als ze er zijn)
if duplicates.sum() > 0:
    print(london_tube_lines[duplicates].head())


# %% [markdown]
# Controleren op inconsistenties in stationnamen
# 
# Aangezien er meer unieke namen in To Station dan in From Station zijn (647 vs. 615), kunnen er naamvariaties of spelfouten zijn.

# %%
# Unieke namen in 'From Station' en 'To Station'
from_stations = set(london_tube_lines["From Station"].unique())
to_stations = set(london_tube_lines["To Station"].unique())

# Stations die in 'To Station' staan, maar niet in 'From Station'
diff_stations = to_stations - from_stations

print(f"Aantal afwijkende stationnamen in 'To Station': {len(diff_stations)}")
print("Voorbeeld van afwijkende namen:", list(diff_stations)[:10])  # Toon er 10


# %% [markdown]
# Mogelijke naamcorrecties vinden

# %%
# Haal unieke stationnamen op uit 'From Station'
from_stations = set(london_tube_lines['From Station'].unique())

# Controleer of een afwijkende naam gedeeltelijk voorkomt in bestaande namen
possible_corrections = {}
for station in diff_stations:  
    similar = [s for s in from_stations if station.lower() in s.lower() or s.lower() in station.lower()]
    if similar:
        possible_corrections[station] = similar

# Resultaten tonen
for wrong, suggestions in possible_corrections.items():
    print(f"Mogelijke correctie voor '{wrong}': {suggestions}")


# %% [markdown]
# automatische correctie

# %%
from difflib import get_close_matches

# Unieke stationsnamen in de dataset
unieke_stations = set(london_tube_lines['To Station'].unique())

# Functie om stationnamen automatisch te corrigeren
def corrigeer_stationnaam(station):
    match = get_close_matches(station, unieke_stations, n=1, cutoff=0.8)  # cutoff bepaalt hoe strikt de match moet zijn
    return match[0] if match else station  # Gebruik de match als die er is, anders behoud de originele naam

# Pas de functie toe op de 'To Station' kolom
london_tube_lines['To Station'] = london_tube_lines['To Station'].apply(corrigeer_stationnaam)

# Controleer of de wijzigingen correct zijn
print(london_tube_lines.head(10))


# %% [markdown]
# Data-inspectie voor alle datasets

# %%
# Functie om snel een overzicht te krijgen van missing values en duplicaten
def inspecteer_data(df, naam):
    print(f"\nDataset: {naam}")
    print(df.info())  # Geeft een overzicht van de dataset
    print("\nAantal NA-waarden per kolom:\n", df.isna().sum())  # Check NA-waarden
    print("\nAantal exacte duplicaten:", df.duplicated().sum())  # Check duplicaten
    print("\nVoorbeeld van data:\n", df.head())

# Inspecteer de datasets
datasets = {
    "london_stations_csv": london_stations_csv,
    "tfl_stations": tfl_stations,
    "stations_20220221": stations_20220221
}

for naam, df in datasets.items():
    inspecteer_data(df, naam)


# %% [markdown]
# output:  Er zijn geen duplicaten of outliers gevonden. wel NA waarden.

# %% [markdown]
#  Behandeling van NA-waarden
# 
#  we willen de integriteit van de dataset behouden zonder waardevolle gegevens (stations) te verliezen. Dus verwijderen is geen optie.  De meest voorkomende waarde (mode) in de Zone kolom is waarschijnlijk een veelvoorkomende zone in de dataset, dus het vervangt de ontbrekende waarden met iets dat logisch is binnen de context.
# 
# 

# %%
# Vervang NA-waarden met de meest voorkomende waarde (mode) in de 'Zone' kolom
mode_value = london_stations_csv['Zone'].mode()[0]
london_stations_csv_cleaned = london_stations_csv.copy()
london_stations_csv_cleaned['Zone'] = london_stations_csv_cleaned['Zone'].fillna(mode_value)

# Output voor bevestiging
print(f"NA-waarden vervangen door de mode waarde ('{mode_value}').")
print(f"Voorbeeld van de aangepaste dataset:\n{london_stations_csv_cleaned.head()}")


# %% [markdown]
# uitleg mode: 
# 
# In veel gevallen willen we de dataset opschonen door ontbrekende gegevens (NA-waarden) te vervangen. De mode van een kolom is de waarde die het meest voorkomt in die kolom. Dit is een gangbare manier om ontbrekende waarden in een kolom te vervangen, omdat we ervan uitgaan dat het invullen met de meest voorkomende waarde een redelijke benadering is voor de ontbrekende gegevens. Het zorgt ervoor dat we geen gegevens verliezen door rijen met NA-waarden en maakt onze dataset compleet.

# %% [markdown]
# outliers verwijderen

# %%
import pandas as pd

# Laad de dataset

# 1. Inspecteer de numerieke kolommen
numerical_cols = ['Latitude', 'Longitude']

# 2. Bereken de IQR voor de numerieke kolommen
Q1 = london_stations_csv[numerical_cols].quantile(0.25)
Q3 = london_stations_csv[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# 3. Bepaal de outlier-drempels
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 4. Verwijder de outliers
df_no_outliers = london_stations_csv[~((london_stations_csv[numerical_cols] < lower_bound) | (london_stations_csv[numerical_cols] > upper_bound)).any(axis=1)]

# Bekijk de aangepaste dataset
print(df_no_outliers)


# %% [markdown]
# NA-waarden controleren

# %%
# Controleer op NA-waarden per kolom
na_per_column = tfl_stations.isna().sum()
print("Aantal NA-waarden per kolom:\n", na_per_column)

# Kijk naar de percentage van NA-waarden in de kolommen
na_percentage = tfl_stations.isna().mean() * 100
print("Percentage NA-waarden per kolom:\n", na_percentage)


# %% [markdown]
# Plan:
# 
# 1. Verwijderen van rijen met te veel NA-waarden in de Elizabeth Line, DLR, en London Overground kolommen.
# 2. Imputatie van NA-waarden in de NETWORK en LINES kolommen met de mode.
# 3. Imputatie van NA-waarden in de En/Ex kolommen met de mediaan.

# %% [markdown]
# duplicaten controleren

# %%
# Controleren op volledige duplicaten
duplicates = tfl_stations.duplicated()

# Aantal exacte duplicaten
print(f"Aantal exacte duplicaten: {duplicates.sum()}")


# %% [markdown]
# NA-waarden behandelen

# %%
# Aantal NA-waarden per kolom in stations_20220221
na_values = stations_20220221.isna().sum()
print("Aantal NA-waarden per kolom in stations_20220221:")
print(na_values)


# %% [markdown]
# plan: 
# 
# 1. Verwijderen van rijen met te veel NA-waarden in de Elizabeth Line, DLR, en London Overground kolommen.
# 2. Imputatie van NA-waarden in de NETWORK en LINES kolommen met de mode.
# 3. Imputatie van NA-waarden in de En/Ex kolommen met de mediaan.

# %%
# Verwijderen van rijen waarin 'Elizabeth Line', 'DLR', en 'London Overground' allemaal NA zijn
stations_20220221_cleaned = stations_20220221.dropna(subset=['Elizabeth Line', 'DLR', 'London Overground'], how='all')


# %%
# Imputeren van NA-waarden in de 'NETWORK' en 'LINES' kolommen met de mode
stations_20220221_cleaned['NETWORK'] = stations_20220221_cleaned['NETWORK'].fillna(stations_20220221_cleaned['NETWORK'].mode()[0])
stations_20220221_cleaned['LINES'] = stations_20220221_cleaned['LINES'].fillna(stations_20220221_cleaned['LINES'].mode()[0])


# %%
# Voor de En/Ex kolommen gaan we de mediaan gebruiken
entry_exit_columns_20220221 = [col for col in stations_20220221_cleaned.columns if 'En/Ex' in col]

# Imputeren van de NA-waarden in de En/Ex kolommen met de mediaan
for col in entry_exit_columns_20220221:
    stations_20220221_cleaned[col] = stations_20220221_cleaned[col].fillna(stations_20220221_cleaned[col].median())


# %% [markdown]
# Controleren op duplicaten

# %%
# Controleren op exacte duplicaten (alle kolommen samen)
duplicates_all_columns = stations_20220221_cleaned.duplicated().sum()

# Controleren op duplicaten op basis van de 'NAME' kolom (aangezien stations vaak uniek zijn)
duplicates_name_column = stations_20220221_cleaned['NAME'].duplicated().sum()

# Resultaten tonen
print(f"Aantal exacte duplicaten (alle kolommen samen): {duplicates_all_columns}")
print(f"Aantal duplicaten op basis van de 'NAME' kolom: {duplicates_name_column}")


# %%
import pandas as pd
import folium
import streamlit as st
import streamlit.components.v1 as components

st.title(" London Underground Stations per Zone")

# 📥 Data inladen met caching
@st.cache_data
def load_station_data():
    df = pd.read_csv("London_stations.csv")
    df['Zone'] = pd.to_numeric(df['Zone'], errors='coerce')
    return df

london_stations = load_station_data()

# 🏷️ Toggle om de kaart visualisatie in/uit te schakelen
toon_kaart = st.sidebar.checkbox("Stations per Zone", value=True)

if toon_kaart:
    # 🎯 Zone selectie (boven de plot, niet in de zijbalk)
    zones = sorted(london_stations['Zone'].dropna().unique())
    selected_zone = st.selectbox("Selecteer een Zone:", zones, format_func=lambda x: f"Zone {int(x)}")

    # 🎨 Zone kleuren
    zone_colors = {
        1: 'green', 2: 'orange', 3: 'yellow', 4: 'blue',
        5: 'purple', 6: 'red', 7: 'pink', 8: 'brown', 9: 'cadetblue'
    }

    # 📌 Data filteren op geselecteerde zone
    zone_data = london_stations[london_stations['Zone'] == selected_zone]

    # 📍 Kaart setup
    zoom_level = 10 if not zone_data.empty else 8
    map_center = [zone_data['Latitude'].mean(), zone_data['Longitude'].mean()] if not zone_data.empty else [51.5074, -0.1278]

    london_map = folium.Map(location=map_center, zoom_start=zoom_level, tiles="CartoDB positron")

    # 📍 Markers toevoegen zonder clustering
    for _, row in zone_data.iterrows():
        color = zone_colors.get(int(row['Zone']), 'gray')
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>Station:</b> {row['Station']}<br><b>Zone:</b> Zone {int(row['Zone'])}",
            tooltip=f"{row['Station']} (Zone {int(row['Zone'])})"
        ).add_to(london_map)

    # 🗺️ Legenda
    legend_html = """
    <div style='position: fixed; bottom: 30px; left: 30px; width: 240px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px; border-radius: 8px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3); opacity: 0.9;'>
    <b>Legenda - Zones</b><br>
    <div style="margin-top: 5px;">
    <span style="color:green; font-weight:bold;">●</span> Zone 1<br>
    <span style="color:orange; font-weight:bold;">●</span> Zone 2<br>
    <span style="color:yellow; font-weight:bold;">●</span> Zone 3<br>
    <span style="color:blue; font-weight:bold;">●</span> Zone 4<br>
    <span style="color:purple; font-weight:bold;">●</span> Zone 5<br>
    <span style="color:red; font-weight:bold;">●</span> Zone 6<br>
    <span style="color:pink; font-weight:bold;">●</span> Zone 7<br>
    <span style="color:brown; font-weight:bold;">●</span> Zone 8<br>
    <span style="color:cadetblue; font-weight:bold;">●</span> Zone 9<br>
    </div>
    </div>
    """
    london_map.get_root().html.add_child(folium.Element(legend_html))

    # 📍 Kaart renderen
    st.write(f"### Stations in Zone {int(selected_zone)}")
    components.html(london_map._repr_html_(), height=700, width=900)


# %%
import pandas as pd
import numpy as np

# Laad de dataset
tfl_stations = pd.read_csv(r"C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Data\TfL_stations.csv")

# Vervang lege waarden of lege strings met NaN
tfl_stations.replace("", np.nan, inplace=True)

# Verwijder alle rijen waar alle waarden in de opgegeven kolommen NaN zijn
tfl_stations = tfl_stations.dropna(subset=["Elizabeth Line", "DLR", "London Overground"], how="all")

# Verwijder alle rijen met NaN-waarden in de dataset
tfl_stations = tfl_stations.dropna()

# Bekijk de dataset na het verwijderen van NaN-waarden
print("Dataset na verwijderen van NaN-waarden:")
print(tfl_stations.head())



# %%
import pandas as pd
import numpy as np

# Laad de dataset
tfl_stations = pd.read_csv(r"C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Data\TfL_stations.csv")

# Vervang lege waarden of lege strings met NaN
tfl_stations.replace("", np.nan, inplace=True)

# Verwijder rijen waar alle waarden in de opgegeven kolommen NaN zijn
tfl_stations = tfl_stations.dropna(subset=["Elizabeth Line", "DLR", "London Overground"], how="all")

# Verwijder alle rijen met NaN-waarden in de dataset
tfl_stations = tfl_stations.dropna()

# Bekijk de dataset na het verwijderen van NaN-waarden
print("Dataset na verwijderen van NaN-waarden:")
print(tfl_stations.head())


# %%
# Laad de dataset
tfl_stations = pd.read_csv(r"C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Data\TfL_stations.csv")

# Vervang lege strings met NaN
tfl_stations.replace("", np.nan, inplace=True)

# Maak een lijst van de kolommen met jaartallen
year_columns = [col for col in tfl_stations.columns if 'En/Ex' in col]

# Verwijder alleen rijen die NaN bevatten in de jaartallen
tfl_stations = tfl_stations.dropna(subset=year_columns, how='any')

# Check hoeveel stations je nu hebt
print(tfl_stations['Station'].unique())
print(tfl_stations.shape)


# %%
print(tfl_stations.isna().sum().sum())  # Totaal aantal NaN-waardes


# %%
# Alle kolommen met 'En/Ex' (dat zijn je instap/uitstap cijfers)
entry_exit_cols = [col for col in tfl_stations.columns if 'En/Ex' in col]

# Vervang NaN's in deze kolommen met de mediaan van de kolom
for col in entry_exit_cols:
    median = tfl_stations[col].median()
    tfl_stations[col].fillna(median, inplace=True)

categorical_cols = ['LINES', 'NETWORK', 'London Underground', 'Elizabeth Line', 'London Overground', 'DLR', 'Night Tube?']

for col in categorical_cols:
    if tfl_stations[col].isna().sum() > 0:
        mode = tfl_stations[col].mode()[0]
        tfl_stations[col].fillna(mode, inplace=True)

print("Aantal NaN na vullen:", tfl_stations.isna().sum().sum())



# %%
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🚉 TfL Stations: Instap/Uitstap Analyse per Jaar")

# 📥 Data inladen & voorbereiden met caching
@st.cache_data
def load_data():
    tfl_stations = pd.read_csv(r"C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Data\TfL_stations.csv")
    entry_exit_cols = [col for col in tfl_stations.columns if 'En/Ex' in col]
    years = [col.replace('En/Ex ', '') for col in entry_exit_cols]
    tfl_stations['Totaal'] = tfl_stations[entry_exit_cols].sum(axis=1)
    return tfl_stations, entry_exit_cols, years

tfl_stations, entry_exit_cols, years = load_data()

# 🏷️ Toggle om de tabs te tonen of verbergen
toon_tabs = st.sidebar.checkbox("Instap/Uitstap per Jaar", value=True)

if toon_tabs:
    # 📊 Tabs aanmaken
    tab1, tab2, tab3, tab4 = st.tabs([" Lijngrafiek per Station", " Top 3 Drukste", " Top 3 Minst Druk", " Data Tabel"])

    with tab1:
        st.header("🔎 Lijngrafiek per Station")
        station_selection = st.multiselect(
            "Selecteer stations:",
            options=tfl_stations['Station'].unique(),
            default=[tfl_stations['Station'].iloc[0]]
        )

        if station_selection:
            filtered_data = tfl_stations[tfl_stations['Station'].isin(station_selection)]
            melted = filtered_data.melt(
                id_vars=['Station'],
                value_vars=entry_exit_cols,
                var_name='Jaar',
                value_name='Aantal Instap/Uitstap'
            )
            melted['Jaar'] = melted['Jaar'].str.replace('En/Ex ', '')

            fig = px.line(
                melted,
                x='Jaar',
                y='Aantal Instap/Uitstap',
                color='Station',
                markers=True,
                title='Aantal Instap/Uitstap per Jaar'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecteer minimaal één station om de grafiek te bekijken.")

    with tab2:
        st.header(" Top 3 Drukste Stations (Totaal)")
        # Automatisch top 3 drukste stations berekenen en tonen
        top3 = tfl_stations.nlargest(3, 'Totaal')
        top3_melted = top3.melt(
            id_vars=['Station'],
            value_vars=entry_exit_cols,
            var_name='Jaar',
            value_name='Aantal Instap/Uitstap'
        )
        top3_melted['Jaar'] = top3_melted['Jaar'].str.replace('En/Ex ', '')

        fig_top3 = px.line(
            top3_melted,
            x='Jaar',
            y='Aantal Instap/Uitstap',
            color='Station',
            markers=True,
            title='Top 3 Drukste Stations'
        )
        st.plotly_chart(fig_top3, use_container_width=True)

    with tab3:
        st.header(" Top 3 Minst Drukke Stations (Totaal)")
        # Automatisch top 3 minst drukke stations berekenen en tonen
        bottom3 = tfl_stations.nsmallest(3, 'Totaal')
        bottom3_melted = bottom3.melt(
            id_vars=['Station'],
            value_vars=entry_exit_cols,
            var_name='Jaar',
            value_name='Aantal Instap/Uitstap'
        )
        bottom3_melted['Jaar'] = bottom3_melted['Jaar'].str.replace('En/Ex ', '')

        fig_bottom3 = px.line(
            bottom3_melted,
            x='Jaar',
            y='Aantal Instap/Uitstap',
            color='Station',
            markers=True,
            title='Top 3 Minst Drukke Stations'
        )
        st.plotly_chart(fig_bottom3, use_container_width=True)

    with tab4:
        st.header(" Bekijk de Data Tabel")
        st.dataframe(tfl_stations[['Station', 'Totaal'] + entry_exit_cols].sort_values(by='Totaal', ascending=False).reset_index(drop=True))



# %%
import pandas as pd
import numpy as np

# Laad de nieuwe dataset
stations_20220221 = pd.read_csv(r'C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Geodata\Stations_20220221.csv')

# Bekijk de eerste paar rijen
print("Eerste paar rijen van de dataset:")
print(stations_20220221.head())

# Check op duplicaten
print(f"\nAantal duplicaten: {stations_20220221.duplicated().sum()}")
stations_20220221 = stations_20220221.drop_duplicates()
print(f"Aantal rijen na verwijderen duplicaten: {stations_20220221.shape[0]}")

# Bekijk het aantal NaN-waarden
print("\nAantal NaN-waarden per kolom:")
print(stations_20220221.isna().sum())

# Vul numerieke NaN-waarden met de mediaan
for col in stations_20220221.select_dtypes(include=[np.number]).columns:
    median = stations_20220221[col].median()
    stations_20220221[col] = stations_20220221[col].fillna(median)

# Vul categorische NaN-waarden met de mode
for col in stations_20220221.select_dtypes(include=['object']).columns:
    mode = stations_20220221[col].mode()[0]
    stations_20220221[col] = stations_20220221[col].fillna(mode)

# Controleer nogmaals op NaN-waarden
print("\nAantal NaN-waarden NA opschonen:")
print(stations_20220221.isna().sum())


# %%
import pandas as pd
import numpy as np

# Laad de dataset
stations_20220221 = pd.read_csv(r'C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Geodata\Stations_20220221.csv')

# Verwijder duplicaten
stations_20220221.drop_duplicates(inplace=True)

# Toon NaN check vóór vullen
print("NaN per kolom vóór vullen:\n", stations_20220221.isna().sum())

# Vul numerieke kolommen met de mediaan
numeric_cols = stations_20220221.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    stations_20220221[col] = stations_20220221[col].fillna(stations_20220221[col].median())

# Vul categorische kolommen met de mode
cat_cols = stations_20220221.select_dtypes(include=['object']).columns
for col in cat_cols:
    if stations_20220221[col].isna().sum() > 0:
        mode_val = stations_20220221[col].mode()[0]
        stations_20220221[col] = stations_20220221[col].fillna(mode_val)

# NaN check na vullen
print("\nNaN per kolom NA vullen:\n", stations_20220221.isna().sum())


# %%
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Data inladen
tfl_stations = pd.read_csv(r"C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Data\TfL_stations.csv")
stations_geo = pd.read_csv(r'C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Geodata\Stations_20220221.csv')

# Opschonen
tfl_stations.replace("", np.nan, inplace=True)
tfl_stations.fillna(tfl_stations.median(numeric_only=True), inplace=True)
stations_geo.replace("", np.nan, inplace=True)
stations_geo.fillna(stations_geo.mode().iloc[0], inplace=True)

# Merge datasets
merged_df = pd.merge(tfl_stations, stations_geo, left_on='Station', right_on='NAME', how='inner')

st.title(" Voorspelling Reizigersgroei TfL Stations (2021 vs. 2030)")

# Jaren kolommen en getallen extraheren
jaar_kolommen = [col for col in tfl_stations.columns if 'En/Ex' in col]
jaren = np.array([int(col.split()[-1]) for col in jaar_kolommen]).reshape(-1, 1)

growth_results = []

for index, row in merged_df.iterrows():
    y_values = row[jaar_kolommen].values.astype(float)
    
    # Filter: alleen stations met genoeg historische data
    if np.count_nonzero(y_values) < 2 or np.sum(y_values) < 100000:
        continue  # Sla dit station over

    y = y_values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(jaren, y)

    voorspel_jaren = np.array([[2030]])
    voorspelling_2030 = model.predict(voorspel_jaren)[0][0]

    # Geen negatieve voorspelling toestaan
    voorspelling_2030 = max(voorspelling_2030, 0)

    growth_results.append({
        'Station': row['Station'],
        'Growth_Slope': model.coef_[0][0],
        'En/Ex 2021': row['En/Ex 2021'],
        'Pred_2030': voorspelling_2030,
        'Verschil': voorspelling_2030 - row['En/Ex 2021']
    })

# Resultaat dataframe
growth_df = pd.DataFrame(growth_results)

# Top 2 groei en top 2 krimp
top_2_groei = growth_df.sort_values(by='Verschil', ascending=False).head(2)
top_2_krimp = growth_df.sort_values(by='Verschil').head(2)

# Formatter voor miljoenen
def miljoenen(x):
    return f"{x/1e6:.1f}M"

# 🏷️ Toggle om de analyse te tonen of verbergen
toon_analyse = st.sidebar.checkbox("Voorspelling Reizigersgroei", value=True)

if toon_analyse:
    # Selectie dropdown
    st.subheader("Bekijk top 2 groei of krimp (2021 -> 2030 voorspelling)")
    keuze = st.selectbox("Keuze:", ["Top 2 groei", "Top 2 krimp"])

    if keuze == "Top 2 groei":
        selectie = top_2_groei
    else:
        selectie = top_2_krimp

    # Tabel met exacte cijfers (geen afronding)
    st.subheader("Exacte cijfers per station (niet in miljoenen)")
    numeriek = selectie[['En/Ex 2021', 'Pred_2030', 'Verschil']].astype(int)
    exacte_df = pd.concat([selectie[['Station']].reset_index(drop=True), numeriek.reset_index(drop=True)], axis=1)
    st.dataframe(exacte_df)

    # Grafiek station kiezen
    station_keuze = st.selectbox("Kies een station voor de groeigrafiek", selectie['Station'].tolist())

    # Plot
    if station_keuze:
        gekozen_row = merged_df[merged_df['Station'] == station_keuze].iloc[0]
        y = gekozen_row[jaar_kolommen].values.astype(float).reshape(-1, 1)
        model = LinearRegression()
        model.fit(jaren, y)

        future_years = np.append(jaren.flatten(), 2030)
        y_future = model.predict(future_years.reshape(-1, 1))
        y_future = np.maximum(y_future, 0)  # Geen negatieve voorspellingen in de plot

        fig, ax = plt.subplots()
        ax.plot(jaren.flatten(), y.flatten() / 1e6, 'o-', label="Historisch (miljoen)")
        ax.plot(future_years, y_future.flatten() / 1e6, 'r--', label="Voorspelling 2030 (miljoen)")

        ax.set_xlabel("Jaar")
        ax.set_ylabel("Aantal reizigers (miljoen)")
        ax.legend()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}M'))
        st.pyplot(fig)




# %%
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium


# Cachen alleen de dataload
@st.cache_data
def laad_data_en_verwerken():
    # Data inladen
    night_tube = gpd.read_file(r'C:\Users\dshab\Downloads\archive\TFL Entry and Exit Data\Geodata\night_tube.geojson')
    london_tube_lines = pd.read_csv("London_tube_lines.csv")

    # Lijnnamen uit night_tube GeoJSON
    night_lines = night_tube['id'].unique()

    # Voeg kolom toe aan reguliere data: heeft nachtdienst (ja/nee)
    london_tube_lines['Night_Service'] = london_tube_lines['Tube Line'].apply(lambda x: 'Ja' if x in night_lines else 'Nee')

    # Groeperen voor barplot: aantal verbindingen per lijn mét night service
    night_count = london_tube_lines.groupby(['Tube Line', 'Night_Service']).size().reset_index(name='Aantal verbindingen')
    night_services = night_count[night_count['Night_Service'] == 'Ja']

    return night_tube, night_services

# Laad de data
night_tube, night_services = laad_data_en_verwerken()

# Titel
st.title("🌙 Night Tube Analyse: Kaart & Barplot")

# 🏷️ Toggle om de analyse te tonen of verbergen
toon_analyse = st.sidebar.checkbox("Night Tube", value=True)

if toon_analyse:
    # Tabs voor kaart en grafiek
    tab1, tab2 = st.tabs(["🌍 Kaartweergave", "📊 Barplot Night Tube lijnen"])

    with tab1:
        st.subheader("Night Tube Lijnen op de Kaart")
        m = folium.Map(location=[51.5074, -0.1278], zoom_start=11, tiles='CartoDB positron')

        # Voeg night_tube GeoJSON toe aan de kaart (in paars)
        folium.GeoJson(
            night_tube,
            style_function=lambda x: {'color': 'purple', 'weight': 5, 'opacity': 0.7},
            tooltip=folium.GeoJsonTooltip(fields=['id'], aliases=['Night Line:'])
        ).add_to(m)

        st_folium(m, width=900, height=600)

    with tab2:
        st.subheader("Aantal Night Tube verbindingen per Lijn")
        
        # Bouw de plot pas hier (om trage rendering te vermijden)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=night_services, x='Aantal verbindingen', y='Tube Line', palette='plasma', ax=ax)
        ax.set_xlabel("Aantal verbindingen (Night Tube)")
        ax.set_ylabel("Tube Lijn")
        ax.set_title("Aantal Night Tube verbindingen per lijn")
        st.pyplot(fig)

        # Bonus tabel
        st.subheader("Detailoverzicht lijnen met Night Tube service")
        st.dataframe(night_services.sort_values(by='Aantal verbindingen', ascending=False).reset_index(drop=True))
else:
    st.info("Schakel de analyse in via de zijbalk om de kaart en grafiek te bekijken.")





# %%
import pandas as pd
import glob
import os
import plotly.graph_objects as go
import streamlit as st

# %%
import os
import glob
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Basispad waar de mappen staan
basis_pad = r"C:\Users\dshab\Downloads"

# Lijst met submappen
mappen = ["fiets2019zomer", "fiets2019winter", "fiets2020zomer", "fiets2020winter"]

# Lege lijst om alle DataFrames op te slaan
dataframes = []

# Door elke map gaan en de CSV-bestanden inlezen
for mapnaam in mappen:
    map_pad = os.path.join(basis_pad, mapnaam)
    csv_bestanden = glob.glob(os.path.join(map_pad, "*.csv"))

    for bestand in csv_bestanden:
        df = pd.read_csv(bestand, parse_dates=["Start Date"], dayfirst=True)
        df["Datum"] = df["Start Date"].dt.date  # Alleen de datum (zonder tijd)
        df["Jaar"] = df["Start Date"].dt.year   # Jaar toevoegen
        df["Seizoen"] = "zomer" if "zomer" in mapnaam else "winter"  # Seizoen afleiden
        dataframes.append(df)

# Alle ingelezen datasets samenvoegen in één DataFrame
fiets_data = pd.concat(dataframes, ignore_index=True)

# Vul NA's op met median of mode
numeric_cols = fiets_data.select_dtypes(include='number').columns
for col in numeric_cols:
    fiets_data[col] = fiets_data[col].fillna(fiets_data[col].median())
non_numeric_cols = fiets_data.select_dtypes(exclude='number').columns
for col in non_numeric_cols:
    fiets_data[col] = fiets_data[col].fillna(fiets_data[col].mode()[0] if not fiets_data[col].mode().empty else "Onbekend")

# Duplicaten verwijderen
fiets_data = fiets_data.drop_duplicates()

# Fietsdata per dag aggregeren
fiets_per_dag = fiets_data.groupby(["Jaar", "Datum", "Seizoen"]).size().reset_index(name="Aantal ritten")
fiets_per_dag["Datum"] = pd.to_datetime(fiets_per_dag["Datum"])

# Weerdata inlezen
weer = pd.read_csv("weather_london.csv", parse_dates=["Unnamed: 0"])
weer.rename(columns={"Unnamed: 0": "Datum"}, inplace=True)
weer["Datum"] = pd.to_datetime(weer["Datum"])

# Vul NA's in weerdata
numeric_cols = weer.select_dtypes(include='number').columns
for col in numeric_cols:
    weer[col] = weer[col].fillna(weer[col].median())
non_numeric_cols = weer.select_dtypes(exclude='number').columns
for col in non_numeric_cols:
    weer[col] = weer[col].fillna(weer[col].mode()[0] if not weer[col].mode().empty else "Onbekend")

# Duplicaten verwijderen
weer = weer.drop_duplicates()

# Merge fietsdata met weerdata
fiets_weer = pd.merge(fiets_per_dag, weer, on="Datum", how="left")

# Streamlit App
st.title("Fietsritten vs. Temperatuur in Londen (september en december)")

# Sidebar met checkbox
show_plot = st.sidebar.checkbox("Fietsritten vs. Temperatuur", value=True)

def laad_en_verwerk_data():
    fiets_weer['Datum'] = pd.to_datetime(fiets_weer['Datum'])
    # Pas seizoen aan op basis van maand (voor plot)
    fiets_weer['Seizoen_plot'] = fiets_weer['Datum'].dt.month.map(lambda x: 'Zomer' if 6 <= x <= 9 else 'Winter')
    return fiets_weer

# Data laden
fiets_weer = laad_en_verwerk_data()

# Definieer kleuren
seizoens_kleuren = {"Winter": "blue", "Zomer": "red"}

# Haal unieke jaren op
jaren = sorted(fiets_weer["Jaar"].dropna().unique())

# Jaarselectie
jaar_selectie = st.selectbox("Selecteer een jaar:", options=jaren)

# Alleen plotten als checkbox aan staat
if show_plot:
    # Filter alleen data voor het geselecteerde jaar
    data_jaar = fiets_weer[fiets_weer["Jaar"] == jaar_selectie]

    # Plot maken
    fig = go.Figure()

    for seizoen in ["Zomer", "Winter"]:
        data_seizoen = data_jaar[data_jaar["Seizoen_plot"] == seizoen]
        fig.add_trace(go.Scatter(
            x=data_seizoen["tavg"],
            y=data_seizoen["Aantal ritten"],
            mode="markers",
            name=f"{jaar_selectie} - {seizoen}",
            marker=dict(color=seizoens_kleuren[seizoen])
        ))

    # Aslimieten vooraf bepalen
    x_min, x_max = fiets_weer["tavg"].min(), fiets_weer["tavg"].max()
    y_min, y_max = fiets_weer["Aantal ritten"].min(), fiets_weer["Aantal ritten"].max()

    fig.update_layout(
        title="Aantal fietsritten vs. Gemiddelde temperatuur",
        xaxis_title="Gemiddelde temperatuur (°C)",
        yaxis_title="Aantal ritten",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        legend_title="Seizoen",
        showlegend=True
    )

    st.plotly_chart(fig)
else:
    st.info("Vink 'Toon de grafiek' aan in de sidebar om de visualisatie te zien.")


# %%
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static

# Cache alleen het inladen van de data
@st.cache_data
def laad_data():
    bike_data_path = r"C:\Users\dshab\Downloads\cycle_stations.csv"
    bike_df = pd.read_csv(bike_data_path)

    # Verwerk locatie kolom
    bike_df["location"] = bike_df["name"].apply(lambda x: x.split(",")[-1].strip() if "," in x else "Onbekend")

    # Vul NA's op met median of mode
    numeric_cols = bike_df.select_dtypes(include='number').columns
    for col in numeric_cols:
        bike_df[col] = bike_df[col].fillna(bike_df[col].median())
    non_numeric_cols = bike_df.select_dtypes(exclude='number').columns
    for col in non_numeric_cols:
        bike_df[col] = bike_df[col].fillna(bike_df[col].mode()[0] if not bike_df[col].mode().empty else "Onbekend")

    # Verwijder duplicaten
    bike_df = bike_df.drop_duplicates()

    return bike_df

bike_df = laad_data()

# Sidebar
show_map = st.sidebar.checkbox("Toon de kaart met fietsstations", value=True)

# Hoofdtitel
st.title("Fietsstations in Londen")

bike_type = st.radio("Kies welk type fietsen je wilt bekijken:", ["Alle fietsen", "Standaard fietsen", "E-bikes"])
min_bikes = st.slider("Toon stations met minimaal aantal beschikbare fietsen", 0, int(bike_df['nbBikes'].max()), 0)

st.subheader("Filter op locatie")
locaties = sorted(bike_df["location"].unique())
gekozen_locatie = st.selectbox("Kies een locatie", ["Alle locaties"] + locaties)

top_most = st.checkbox("Toon top 3 stations met meeste beschikbare fietsen")
top_least = st.checkbox("Toon top 3 stations met minste beschikbare fietsen")

# Fietskeuze
if bike_type == "Alle fietsen":
    bike_column = "nbBikes"
elif bike_type == "Standaard fietsen":
    bike_column = "nbStandardBikes"
else:
    bike_column = "nbEBikes"

# Filteren
filtered_df = bike_df.copy()
if gekozen_locatie != "Alle locaties":
    filtered_df = filtered_df[filtered_df["location"] == gekozen_locatie]
filtered_df = filtered_df[filtered_df[bike_column] >= min_bikes]

if top_most:
    filtered_df = filtered_df.nlargest(3, bike_column)
if top_least:
    filtered_df = filtered_df.nsmallest(3, bike_column)

# Kaart tekenen als de checkbox aanstaat
if show_map:
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=12)
    for _, row in filtered_df.iterrows():
        bike_count = row[bike_column]
        if bike_count == 0:
            color = "gray"
        elif bike_count < 5:
            color = "red"
        elif bike_count < 10:
            color = "orange"
        else:
            color = "green"
        popup_text = f"Station: {row['name']}<br>Aantal fietsen: {bike_count}<br>Locatie: {row['location']}"
        folium.Marker(
            location=[row['lat'], row['long']],
            popup=popup_text,
            icon=folium.Icon(color=color, icon="bicycle", prefix="fa")
        ).add_to(m)

    folium_static(m)
else:
    st.info("Vink 'Toon de kaart met fietsstations' aan in de sidebar om de kaart te bekijken.")


# %%
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static

st.title("🚲 Fietsstations & 🚇 Metro Stations in Londen")

# 📥 Data laden
@st.cache_data
def load_bike_data():
    df = pd.read_csv(r"C:\Users\dshab\Downloads\cycle_stations.csv")
    df["location"] = df["name"].apply(lambda x: x.split(",")[-1].strip() if "," in x else "Onbekend")
    return df

@st.cache_data
def load_metro_data():
    df = pd.read_csv("London_stations.csv")
    df['Zone'] = pd.to_numeric(df['Zone'], errors='coerce')
    return df

bike_df = load_bike_data()
metro_df = load_metro_data()

# 📍 Dropdownmenu om metrozone te kiezen
zones = metro_df['Zone'].dropna().unique().tolist()  # Unieke zones ophalen
zones = sorted(zones)  # Sorteer zones op volgorde
selected_zone = st.selectbox("Selecteer een metrozone:", zones)

# 📍 Kaart aanmaken
m = folium.Map(location=[51.5074, -0.1278], zoom_start=12, tiles="CartoDB positron")

# 🚇 Metrostations plotten voor de geselecteerde zone
filtered_metro_df = metro_df[metro_df['Zone'] == selected_zone]  # Filter metrostations op geselecteerde zone

for _, row in filtered_metro_df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.Icon(color='red', icon='subway', prefix='fa'),
        popup=f"<b>Metrostation:</b> {row['Station']}<br><b>Zone:</b> {int(row['Zone']) if not pd.isna(row['Zone']) else 'Onbekend'}"
    ).add_to(m)

# 🚲 Fietsstations plotten voor de geselecteerde zone
# Veronderstellen dat de fietsdata een 'Zone' kolom bevat die de zone van het fietsstation vertegenwoordigt
# Als de fietsstations geen zonekolom bevatten, moeten we de afstand van het metrostation gebruiken om fietsen in de geselecteerde zone te vinden.
filtered_bike_df = bike_df  # Hier zouden we filteren op basis van de zone van de fietsen, indien aanwezig.

for _, row in filtered_bike_df.iterrows():
    # Dit filtert fietsen in de geselecteerde zone, als er een zone kolom is in de fietsdata.
    # Als de fietsstations geen 'Zone' kolom bevatten, kun je een andere manier van filteren gebruiken, zoals afstand.
    folium.Marker(
        location=[row['lat'], row['long']],
        icon=folium.Icon(color='blue', icon='bicycle', prefix='fa'),
        popup=f"<b>Fietsstation:</b> {row['name']}<br><b>Beschikbare fietsen:</b> {row['nbBikes']}"
    ).add_to(m)

# 📍 Kaart weergeven
folium_static(m)



# %%
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static

# 📥 Data laden
@st.cache_data
def load_bike_data():
    df = pd.read_csv(r"C:\Users\dshab\Downloads\cycle_stations.csv")
    df["location"] = df["name"].apply(lambda x: x.split(",")[-1].strip() if "," in x else "Onbekend")
    return df

@st.cache_data
def load_metro_data():
    df = pd.read_csv("London_stations.csv")
    df['Zone'] = pd.to_numeric(df['Zone'], errors='coerce')
    df['passenger_volume'] = df['Zone'].apply(lambda x: 1000 if x == 1 else 500)  # Simulatie drukte
    return df

bike_df = load_bike_data()
metro_df = load_metro_data()

st.title("🚲 Correlatie: Fiets vs Drukke Metro in Londen")

afstand_drempel = 0.3  # km
fiets_drempel = 5      # fietsen
toon_resultaat = st.sidebar.checkbox("Toon alleen stations met weinig fietsen bij drukke metro", True)

# 👉 Gebiedselectie
gebieden = sorted(bike_df["location"].unique())
gekozen_gebied = st.selectbox("📍 Kies een gebied / district:", ["Alle gebieden"] + gebieden)

# 🧠 Haversine functie (pure Python, geen geopy nodig)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * (2 * np.arcsin(np.sqrt(a)))

# 🚇 Pak de drukke metro stations
drukke_metro = metro_df[metro_df['passenger_volume'] > 800]

correlaties = []
for _, metro_row in drukke_metro.iterrows():
    metro_lat, metro_lon = metro_row['Latitude'], metro_row['Longitude']
    
    # Filter bike_df als een gebied is gekozen
    if gekozen_gebied != "Alle gebieden":
        subset_bike_df = bike_df[bike_df['location'] == gekozen_gebied].copy()
    else:
        subset_bike_df = bike_df.copy()

    subset_bike_df['afstand'] = haversine(metro_lat, metro_lon, subset_bike_df['lat'], subset_bike_df['long'])
    dichtbij_bikes = subset_bike_df[(subset_bike_df['afstand'] <= afstand_drempel) & (subset_bike_df['nbBikes'] < fiets_drempel)]

    if not dichtbij_bikes.empty:
        correlaties.append({
            'metro_station': metro_row['Station'],
            'metro_lat': metro_lat,
            'metro_lon': metro_lon,
            'bikes': dichtbij_bikes
        })

# 🗺️ Kaart maken
m = folium.Map(location=[51.5074, -0.1278], zoom_start=12, tiles="CartoDB positron")

for corr in correlaties:
    # Rode marker = druk metrostation
    folium.Marker(
        location=[corr['metro_lat'], corr['metro_lon']],
        icon=folium.Icon(color='red', icon='subway', prefix='fa'),
        popup=f"<b>Metro:</b> {corr['metro_station']}"
    ).add_to(m)

    # Blauwe markers = fietsstations met weinig fietsen
    for _, bike in corr['bikes'].iterrows():
        folium.Marker(
            location=[bike['lat'], bike['long']],
            icon=folium.Icon(color='blue', icon='bicycle', prefix='fa'),
            popup=(f"<b>Fietsstation:</b> {bike['name']}<br>"
                   f"<b>Beschikbare fietsen:</b> {bike['nbBikes']}<br>"
                   f"<b>Afstand:</b> {round(bike['afstand'] * 1000)} m")
        ).add_to(m)

# 🖌️ Simpele legenda toevoegen
legend_html = """
<div style='position: fixed; bottom: 30px; left: 30px; width: 220px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px; border-radius: 8px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3); opacity: 0.9;'>
<b>Legenda</b><br>
<span style='color:red; font-weight:bold;'>■</span> Druk Metrostation<br>
<span style='color:blue; font-weight:bold;'>■</span> Fietsstation met weinig fietsen
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ✅ Kaart tonen
if toon_resultaat and correlaties:
    folium_static(m)
elif not correlaties:
    st.warning("Geen matches gevonden binnen de ingestelde afstand en fietsgrens.")
else:
    st.info("Gebruik de checkbox om de correlatiekaart te tonen.")







