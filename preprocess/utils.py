import time

import numpy as np
import pandas as pd
import pubchempy as pcp
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def get_smiles_from_nsc(nsc_code, retries=3):
    url = f"https://cactus.nci.nih.gov/chemical/structure/NSC{nsc_code}/smiles"
    session = requests.Session()

    # Retry configuration
    retry = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    while True:
        try:
            response = session.get(url)

            # Handle 404 Not Found
            if response.status_code == 404:
                print(f"SMILES for NSC{nsc_code} not found.")
                return None

            response.raise_for_status()  # Raise exception for other HTTP errors
            return response.text.strip()

        except requests.exceptions.RequestException as e:
            error_message = str(e)
            print(f"Error: {error_message}")

            if "Connection reset by peer" in error_message:
                print("Waiting 5 minutes before retrying...")
                time.sleep(300)  # Wait for 5 minutes (300 seconds)
            else:
                return None


def get_smiles_from_cid(cid):
    max_retries = 5
    retry_delay = 5  # Initial retry delay in seconds
    for attempt in range(max_retries):
        try:
            # Search for compound by CID in PubChem
            compound = pcp.Compound.from_cid(cid)
            # Get SMILES notation
            smiles = compound.isomeric_smiles
            return smiles
        except Exception as e:
            if "PUGREST.ServerBusy" in str(e):
                print(f"Server is busy. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Double the delay time
            else:
                error_message = str(e)
                return error_message  # Return error message for other exceptions

    return (
        "Maximum retry attempts reached."  # Error message when retry limit is exceeded
    )


def get_pubchem_info(nsc_number):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # First, we need to get the PubChem CID
    nsc_query = f"{base_url}/compound/name/NSC{nsc_number}/cids/JSON"
    response = requests.get(nsc_query)

    if response.status_code != 200:
        return None, None

    data = response.json()
    if "IdentifierList" not in data or "CID" not in data["IdentifierList"]:
        return None, None

    cid = data["IdentifierList"]["CID"][0]

    # Now we can get the SMILES
    smiles_query = f"{base_url}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    response = requests.get(smiles_query)

    if response.status_code != 200:
        return cid, None

    data = response.json()
    if "PropertyTable" not in data or "Properties" not in data["PropertyTable"]:
        return cid, None

    smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]

    return cid, smiles


def get_canonical_smiles(drug_name):
    # Base URL for PubChem PUG REST API
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            # Search for compound by drug name
            search_url = f"{base_url}/compound/name/{drug_name}/cids/JSON"
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()

            # Get the first CID
            cid = data["IdentifierList"]["CID"][0]

            # Get Canonical SMILES using the CID
            smiles_url = f"{base_url}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
            response = requests.get(smiles_url)
            response.raise_for_status()
            data = response.json()

            # Return Canonical SMILES
            return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"An error occurred: {e}")
                return None
            elif e.response.status_code == 503:
                if attempt < max_retries - 1:
                    print(f"Server is busy. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Maximum retry attempts reached: {e}")
                    return None
            else:
                print(f"An error occurred: {e}")
                return None
        except (KeyError, IndexError) as e:
            print(f"An error occurred while retrieving or parsing data: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the request: {e}")
            return None
