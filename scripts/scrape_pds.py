import os
import requests

# Direct links to standard Australian Home & Contents PDS documents
PDS_URLS = {
    "NAB": "https://www.nab.com.au/content/dam/nab/documents/policy/insurance/nab-home-insurance-pds.pdf",
    "RAA": "https://www.raa.com.au/globalassets/documents/insurance/home/raa-home-and-contents-insurance-pds.pdf",
    "Bankwest": "https://www.bankwest.com.au/content/dam/bankwest/documents/insurance/home-insurance/home-insurance-pds.pdf",
    "Australia_Post": "https://auspost.com.au/content/dam/auspost_corp/media/documents/australia-post-home-and-contents-insurance-pds.pdf",
    "Virgin_Money": "https://virginmoney.com.au/content/dam/virginmoney/documents/insurance/home-and-contents-insurance-pds.pdf",
    "RACQ": "https://www.racq.com.au/-/media/pdf/insurance/pds/household-insurance-pds.pdf",
    "ING": "https://www.ing.com.au/pdf/home_contents_insurance_pds.pdf",
    "Westpac": "https://www.westpac.com.au/content/dam/public/wbc/documents/pdf/personal/insurance/WBC_Home_and_Contents_Insurance_PDS.pdf",
    "Allianz": "https://www.allianz.com.au/content/dam/onemarketing/azap/allianz-au/documents/home/pds/Allianz-Home-and-Contents-Insurance-PDS.pdf",
    "QBE": "https://www.qbe.com/au/-/media/au/files/home-insurance/home-insurance-pds-qm8283.pdf"
}

OUTPUT_DIR = "data/policies"

def download_pds():
    print("Starting PDS Download Script...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
        
    for carrier, url in PDS_URLS.items():
        print(f"Downloading {carrier} PDS...")
        try:
            # Using a browser user-agent to prevent 403 blocks from simple bots
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, timeout=20, headers=headers)
            response.raise_for_status()
            
            filepath = os.path.join(OUTPUT_DIR, f"{carrier}_PDS.pdf")
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Success: Saved to {filepath}")
        except Exception as e:
            print(f"Failed: Could not download {carrier} - {e}")

    print("\nDownload complete. Check the data/policies/ directory.")

if __name__ == "__main__":
    download_pds()
