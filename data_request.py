# import os
# import requests
# from PIL import Image
# from io import BytesIO
# from bs4 import BeautifulSoup
# import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# # List of birds with common and scientific names
# # birds = [
# #     {"common_name": "Red Junglefowl", "scientific_name": "Gallus gallus"},
# #     # {"common_name": "Large-tailed Nightjar", "scientific_name": "Caprimulgus macrurus"},
# #     # {"common_name": "Savanna Nightjar", "scientific_name": "Caprimulgus affinis"},
# #     # {"common_name": "Black-nest Swiftlet", "scientific_name": "Aerodramus maximus"},
# #     # {"common_name": "Asian Koel", "scientific_name": "Eudynamys scolopaceus"},
# #     # {"common_name": "Asian Emerald Cuckoo", "scientific_name": "Chrysococcyx maculatus"},
# #     # {"common_name": "Horsfield's Bronze Cuckoo", "scientific_name": "Chrysococcyx basalis"},
# #     # {"common_name": "Banded Bay Cuckoo", "scientific_name": "Cacomantis sonneratii"},
# #     # {"common_name": "Plaintive Cuckoo", "scientific_name": "Cacomantis merulinus"},
# #     # {"common_name": "Rusty-breasted Cuckoo", "scientific_name": "Cacomantis sepulcralis"},
# #     # {"common_name": "Cinnamon-headed Green Pigeon", "scientific_name": "Treron fulvicollis"},
# #     # {"common_name": "Pink-necked Green Pigeon", "scientific_name": "Treron vernans"},
# #     # {"common_name": "Slaty-breasted Rail", "scientific_name": "Lewinia striata"},
# #     # {"common_name": "Common Moorhen", "scientific_name": "Gallinula chloropus"},
# #     # {"common_name": "Ruddy-breasted Crake", "scientific_name": "Zapornia fusca"},
# #     # {"common_name": "Red-legged Crake", "scientific_name": "Rallina fasciata"},
# #     # {"common_name": "White-breasted Waterhen", "scientific_name": "Amaurornis phoenicurus"},
# #     # {"common_name": "Pacific Golden Plover", "scientific_name": "Pluvialis fulva"},
# #     # {"common_name": "Ruddy Turnstone", "scientific_name": "Arenaria interpres"},
# #     # {"common_name": "Grey Heron", "scientific_name": "Ardea cinerea"},
# #     # {"common_name": "Purple Heron", "scientific_name": "Ardea purpurea"},
# #     # {"common_name": "Great Egret", "scientific_name": "Ardea alba"}
# # ]

# # Function to download and save image
# def download_image(common_name, scientific_name):
#     search_query = f"{common_name} {scientific_name} bird"
#     url = f"https://images.search.yahoo.com/search/images?p={search_query}"

#     response = requests.get(url, verify=False)  # Disabling SSL verification
#     soup = BeautifulSoup(response.text, 'html.parser')
#     img_tags = soup.find_all('img', class_='process')

#     if img_tags:
#         image_url = img_tags[0]['data-src']

#         # Download the image
#         img_response = requests.get(image_url)
#         img = Image.open(BytesIO(img_response.content))

#         # Save the image
#         img_path = f"images/{common_name}.jpg"
#         img.save(img_path)
#         print(f"Saved image for {common_name} as {img_path}")
#     else:
#         print(f"No image found for {common_name}")

# # Get scientific names
# # scientific_names = [bird["scientific_name"] for bird in birds]
# # print("Scientific Names:", scientific_names)

# # Download and save images for each bird
# # Iterate over DataFrame rows and download images
# birds_df=pd.read_csv("Birds_full_data.csv")
# for index, row in birds_df.iterrows():
#     try:
#         download_image(row['common_name'], row['scientific_name'])
#     except:
#         continue

import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

# List of wanted bird species scientific names
wanted_scientific_names = [
    'Psittacula longicauda', 'Ardea alba', 'Dicrurus leucophaeus', 'Dicrurus paradiseus', 'Cinnyris jugularis',
    'Pycnonotus goiavier', 'Aplonis panayensis', 'Pernis ptilorhynchus', 'Orthotomus sutorius', 'Orthotomus atrogularis',
    'Phylloscopus borealis', 'Hemixos cinereus', 'Acrocephalus orientalis', 'Psittacula krameri', 'Elanus caeruleus',
    'Falco peregrinus', 'Acrocephalus bistrigiceps', 'Gallinula chloropus', 'Dicaeum cruentatum', 'Pycnonotus plumosus',
    'Eudynamys scolopaceus', 'Dicrurus annectens', 'Rallina fasciata', 'Passer domesticus', 'Phylloscopus coronatus',
    'Cacomantis merulinus', 'Phylloscopus borealoides', 'Dinopium javanense', 'Ficedula zanthopygia', 'Anthracoceros albirostris',
    'Todiramphus chloris', 'Garrulax leucolophus', 'Arenaria interpres', 'Caprimulgus macrurus', 'Pitta megarhyncha',
    'Pycnonotus jocosus', 'Amaurornis phoenicurus', 'Ardea cinerea', 'Ardea purpurea', 'Psilopogon rafflesii', 'Gallus gallus',
    'Corvus macrorhynchos', 'Strix seloputo', 'Phylloscopus inornatus', 'Pitta moluccensis', 'Aegithina tiphia', 'Motacilla cinerea',
    'Ninox scutulata', 'Pluvialis fulva', 'Helopsaltes certhiola', 'Amandava amandava', 'Pelargopsis capensis', 'Tyto alba',
    'Anthreptes malacensis', 'Orthotomus ruficeps'
]

# # Example DataFrame with bird data
# data = {
#     'common_name': ['Red Junglefowl', 'Asian Koel', 'Cinnyris jugularis', 'Psittacula longicauda'],
#     'scientific_name': ['Gallus gallus', 'Eudynamys scolopaceus', 'Cinnyris jugularis', 'Psittacula longicauda']
# }
# birds_df = pd.DataFrame(data)



# Function to download and save image
def download_image(common_name, scientific_name):
    search_query = f"{common_name} {scientific_name} bird"
    url = f"https://images.search.yahoo.com/search/images?p={search_query}"

    response = requests.get(url, verify=False)  # Disabling SSL verification
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img', class_='process')

    if img_tags:
        image_url = img_tags[0]['data-src']

        # Download the image
        img_response = requests.get(image_url, verify=False)  # Disabling SSL verification
        img = Image.open(BytesIO(img_response.content))

        # Save the image
        img_path = f"images2/{common_name}.jpg"
        img.save(img_path)
        print(f"Saved image for {common_name} as {img_path}")
    else:
        print(f"No image found for {common_name}")

birds_df=pd.read_csv("Birds_full_data.csv")
# Filter DataFrame for wanted scientific names
filtered_df = birds_df[birds_df['scientific_name'].str.lower().isin([name.lower() for name in wanted_scientific_names])]
filtered_df.to_csv("available_birds.csv",index=False)
# Iterate over filtered DataFrame rows and download images
for index, row in filtered_df.iterrows():
    try:
        download_image(row['common_name'], row['scientific_name'])
    except:
        print(row['common_name'], row['scientific_name'])
        continue
