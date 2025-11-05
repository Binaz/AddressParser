# Address Parser

### Extract structured address information using NLP & Regex

**Author:** Binaz Pardiwala (https://github.com/Binaz)  
**Notebook:** [ View Full Notebook on nbviewer](https://nbviewer.org/github/Binaz/AddressParser/blob/main/address_parser_main.ipynb)  

---

## Overview
Addresses often appear in messy, unstructured formats, making them difficult to process for databases or mapping APIs.  
This project demonstrates how to parse and standardize free-text addresses into structured fields such as:

- Unit number
- Street name and number  
- City  
- State or region  
- Postal code  
- Country  

---

## Tech Stack
- Python  
- Regex  
- pandas  
- Google Colab  

---

## Example
| Input | Parsed Output |
|--------|----------------|
| `"1600 Amphitheatre Pkwy, Mountain View, CA 94043"` | `{ "street number":"1600", "street": "Amphitheatre Pkwy", "city": "Mountain View", "state": "CA", "postal_code": "94043" }` |
| `"221B-667 Baker Street, London NW1 6XE, UK"` | `{ "unit number": "221B", ""street number":"667"," ,"street": "Baker Street", "city": "London", "postal_code": "NW1 6XE", "country": "UK" }` |

---

## Resources
- [Hugging Face] (https://huggingface.co/)
- [OpenAddresses Project](https://openaddresses.io/)  
- [USPS Addressing Standards](https://pe.usps.com/text/pub28/welcome.htm)  
- [Regex101](https://regex101.com/)  

