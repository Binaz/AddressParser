# Address Parser using Named Entity Recognition (NER)

### Fine-tuning a Transformer-based model to extract structured address components

**Author:** [Binaz Pardiwala](https://github.com/Binaz)  
**Notebook:** [View on nbviewer](https://nbviewer.org/github/Binaz/AddressParser/blob/main/Address_Parser_Main.ipynb)  
**Date:** October 2024  

---

## Overview
This project builds a **custom address parser** using **Hugging Face Transformers** and **Named Entity Recognition (NER)** to extract structured information such as unit, street, city, state, postal code, and country from unstructured address text.  

The model is fine-tuned using a synthetic labeled dataset of addresses. It leverages the `bert-base-NER-uncased` transformer model as a base, retrained to recognize address-specific entities.

---

## Key Features
- Fine-tunes a **BERT-based model** for address parsing  
- Splits dataset into train, validation, and test sets  
- Implements **token alignment** between tokens and labels  
- Uses **seqeval** for NER evaluation (precision, recall, F1, accuracy)  
- Supports model saving and reloading for inference using the Hugging Face `pipeline`  

---

## Tech Stack
| Component | Tool / Library |
|------------|----------------|
| Model |  Transformers (BERT) |
| Dataset Handling | `datasets` |
| Tokenization | `AutoTokenizer` |
| Evaluation | `seqeval`, `evaluate` |
| Training | `Trainer`, `TrainingArguments` |
| Platform | Google Colab |

---

## Dataset
The synthetic dataset was created which includes variety of addresses, in different formats and different abbreviations. The dataset is stored on Google Drive and loaded using the `datasets` library.  
It contains tokens and NER tags for each address component, e.g.:

### Example 1
**Input:**  
`TT Accident-I-81 NB MM191.1-Lexington-VA--Rockbridge County`

**Parsed Output:**
| Token | Tag |
|-------|-----|
| TT | O |
| Accident | O |
| I-81 | B-STREET |
| MM191.1 | B-STREET_NUM |
| NB | O |
| Lexington | B-CITY |
| VA | B-STATE |
| Rockbridge | B-COUNTY |
| County | I-COUNTY |

---

### Example 2
**Input:**  
`Interstate 70 westbound at mile marker 163.7, Williamsburg, MO, 63388, Callaway`

**Parsed Output:**
| Token | Tag |
|-------|-----|
| Interstate | B-STREET |
| 70 | I-STREET |
| westbound | O |
| mile | O |
| marker | O |
| 163.7 | B-STREET_NUM |
| Williamsburg | B-CITY |
| MO | B-STATE |
| 63388 | B-POSTAL |
| Callaway | B-COUNTY |

---

### Example 3
**Input:**  
`I-95 & Namaans Road, Tri-State Mall, Claymont, DE 19703 US`

**Parsed Output:**
| Token | Tag |
|-------|-----|
| I-95 | B-STREET |
| Namaans | I-STREET |
| Road | I-STREET |
| Tri-State | B-PLACE |
| Mall | I-PLACE |
| Claymont | B-CITY |
| DE | B-STATE |
| 19703 | B-POSTAL |
| US | I-COUNTRY |

---

### Example 4
**Input:**  
`3224 Phila Pike, Claymont, DE 19703 US`

**Parsed Output:**
| Token | Tag |
|-------|-----|
| 3224 | B-STREET_NUM |
| Phila | B-STREET |
| Pike | I-STREET |
| Claymont | B-CITY |
| DE | B-STATE |
| 19703 | B-POSTAL |
| US | I-COUNTRY |

---

##  Model Workflow
1. **Data Loading & Cleaning**  
   Load dataset from CSV, convert stringified lists, and verify structure.  
2. **Splitting Data**  
   Create training, validation, and test datasets (80/10/10).  
3. **Label Encoding**  
   Define entity label mappings (e.g., `B-STREET`, `I-CITY`).  
4. **Tokenization & Label Alignment**  
   Align labels to tokens for subword tokenization.  
5. **Model Fine-tuning**  
   Train `bert-base-NER-uncased` using Hugging Face `Trainer`.  
6. **Evaluation**  
   Use `seqeval` for NER metrics.  
7. **Deployment**  
   Save model and load with `pipeline` for live predictions.

---

## Evaluation Metrics

| Metric | Description |
|------------|----------------|
| Precision |  Correct entities out of predicted entities |
| Recall | Correct entities out of all true entities |
| F1-score | Harmonic mean of precision & recall |
| Accuracy | Overall label match rate |

---

## Future Enhancements

- Add more address formats for international coverage
- Add more address for french like street names and postal code
- Integrate with Google Maps API for validation
- Publish the fine-tuned model on Hugging Face Hub

## Example Inference
```python
from transformers import pipeline

nerTrainer = pipeline(
    'token-classification',
    '/path/to/saved/model',
    tokenizer=tokenizer,
    grouped_entities=True,
    aggregation_strategy='simple'
)

nerTrainer("221B Baker Street, London NW1 6XE, UK")


## Example Address Parsing Results

Below are sample address strings and the structured address components extracted by the fine-tuned **Address Parser** model.

Each address is converted into a set of label and respective value pairs for easy downstream use.

---
# Address Parsing Results

## 1. Address: 754-782 BROADWAY, 23, CHULA VISTA CA 919105372

- **STREET_NUM**: `754 - 782`, Score: `1.0`
- **STREET**: `BROADWAY`, Score: `1.0`
- **CITY**: `CHULA VISTA`, Score: `1.0`
- **STATE**: `CA`, Score: `1.0`
- **POSTAL**: `919105372`, Score: `1.0`

---

## 2. Address: Kashyap Property-1284 Leland Road-Manassas-VA-20111-Prince William County

- **NAME**: `kashyap property`, Score: `1.0`
- **STREET_NUM**: `1284`, Score: `1.0`
- **STREET**: `LELAND ROAD`, Score: `1.0`
- **CITY**: `MANASSAS`, Score: `1.0`
- **STATE**: `VA`, Score: `1.0`
- **POSTAL**: `20111`, Score: `1.0`

---

## 3. Address: The address of the suspected auto service station is 108 Harrison Street Southeast, Leesburg, VA 20198.

- **NAME**: `suspected auto service station`, Score: `0.9999`
- **STREET_NUM**: `108`, Score: `1.0`
- **STREET**: `HARRISON STREET SOUTHEAST`, Score: `1.0`
- **CITY**: `LEESBURG`, Score: `1.0`
- **STATE**: `VA`, Score: `1.0`
- **POSTAL**: `20198`, Score: `1.0`

---

## 4. Address: 7890 Old Mill Road, Richmond, VA 23225

- **STREET_NUM**: `7890`, Score: `1.0`
- **STREET**: `OLD MILL ROAD`, Score: `1.0`
- **CITY**: `RICHMOND`, Score: `1.0`
- **STATE**: `VA`, Score: `1.0`
- **POSTAL**: `23225`, Score: `1.0`

---

## 5. Address: Exit 54 on I-95 South near Fayetteville, NC

- **STREET**: `EXIT 54 ON I-95 SOUTH`, Score: `0.8667`
- **CITY**: `FAYETTEVILLE`, Score: `1.0`
- **STATE**: `NC`, Score: `0.9998`

---

## 6. Address: Various locations throughout the Upper Peninsula, corporate address located at 920 10th Avenue North, varies, MI, 95855-0000, US

- **STREET_NUM**: `920`, Score: `1.0`
- **STREET**: `10TH AVENUE NORTH`, Score: `1.0`
- **CITY**: `VARIES`, Score: `1.0`
- **STATE**: `MI`, Score: `1.0`
- **POSTAL**: `95855-0000`, Score: `1.0`

---

## 7. Address: 1228-1290 Middletown & Warwick Road, Middletown, DE 19709 US

- **STREET_NUM**: `1228 - 1290`, Score: `1.0`
- **STREET**: `MIDDLETOWN & WARWICK ROAD`, Score: `1.0`
- **CITY**: `MIDDLETOWN`, Score: `1.0`
- **STATE**: `DE`, Score: `1.0`
- **POSTAL**: `19709`, Score: `1.0`

---

## 8. Address: Express Trucking Co-700 1st St-Harrison-VA-07029-Frederick County

- **NAME**: `TRUCKING CO`, Score: `0.9005`
- **STREET_NUM**: `700`, Score: `1.0`
- **STREET**: `1ST ST`, Score: `1.0`
- **CITY**: `HARRISON`, Score: `1.0`
- **STATE**: `VA`, Score: `1.0`
- **POSTAL**: `07029`, Score: `1.0`
- **CITY**: `FREDERICK COUNTY`, Score: `1.0`

---

## 9. Address: 4200 Summit Bridge Road, Summit Airport, Middletown, DE 19709 US

- **STREET_NUM**: `4200`, Score: `1.0`
- **STREET**: `SUMMIT BRIDGE ROAD`, Score: `1.0`
- **CITY**: `SUMMIT AIRPORT`, Score: `0.9954`
- **CITY**: `MIDDLETOWN`, Score: `1.0`
- **STATE**: `DE`, Score: `1.0`
- **POSTAL**: `19709`, Score: `1.0`

---

## 10. Address: The actual compost site is located across the street from a community member home: 17397 Count Turf Place.  However, the name of the business, Clairvoux LLC and address is: 40730 Farm Market Road, Leesburg 20176. The compost site is located on land they own in the community.  Route 7 West to Farm Market Road, turn right onto Alysheba Drive, left onto Count Turf.  Compost site is on right about 100 feet.

- **NAME**: `ACTUAL COMPOST SITE`, Score: `0.9638`
- **STREET_NUM**: `17397`, Score: `1.0`
- **STREET**: `COUNT TURF PLACE`, Score: `1.0`
- **NAME**: `CLAIRVOUX LLC`, Score: `0.9998`
- **STREET_NUM**: `40730`, Score: `1.0`
- **STREET**: `FARM MARKET ROAD`, Score: `1.0`
- **CITY**: `LEESBURG`, Score: `1.0`
- **POSTAL**: `20176`, Score: `1.0`

---

## 11. Address: Capitol Fiber, Inc - Recycling Center-6610 Electronic Drive-Springfield-VA-22151-Fairfax County

- **NAME**: `CAPITOL FIBER, INC`, Score: `0.9999`
- **STREET_NUM**: `6610`, Score: `1.0`
- **STREET**: `ELECTRONIC DRIVE`, Score: `1.0`
- **CITY**: `SPRINGFIELD`, Score: `1.0`
- **STATE**: `VA`, Score: `1.0`
- **POSTAL**: `22151`, Score: `1.0`
- **CITY**: `FAIRFAX COUNTY`, Score: `1.0`

---

## 12. Address: This incident took place at 980 Bayshore rd. Cape Charles, VA 23318

- **STREET_NUM**: `980`, Score: `0.9999`
- **STREET**: `BAYSHORE RD.`, Score: `0.9999`
- **CITY**: `CAPE CHARLES`, Score: `1.0`
- **STATE**: `VA`, Score: `1.0`
- **POSTAL**: `23318`, Score: `1.0`

---

## 13. Address: Oyster Farm at Kings Creek 500 Marina Village Cir Cape Charles, VA 23310

- **NAME**: `OYSTER FARM AT KINGS`, Score: `0.9892`
- **STREET_NUM**: `500`, Score: `1.0`
- **STREET**: `MARINA VILLAGE CIR`, Score: `1.0`
- **CITY**: `CAPE CHARLES`, Score: `1.0`
- **STATE**: `VA`, Score: `1.0`
- **POSTAL**: `23310`, Score: `1.0`

---

## 14. Address: On North Curry St. between County St and Sewell Ave. Directly across the Street from 33 N.Curry St, Hampton,Va 23663-5858

- **STREET**: `COUNTY ST AND SEWELL AVE`, Score: `0.9999`
- **STREET_NUM**: `33`, Score: `1.0`
- **STREET**: `N. CURRY ST`, Score: `1.0`
- **CITY**: `HAMPTON`, Score: `1.0`
- **STATE**: `VA`, Score: `1.0`
- **POSTAL**: `23663-5858`, Score: `1.0`

---
