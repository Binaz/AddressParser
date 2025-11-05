# 🏠 Address Parser using Named Entity Recognition (NER)

### Fine-tuning a Transformer-based model to extract structured address components

**Author:** [Binaz Pardiwala](https://github.com/YOUR_GITHUB_USERNAME)  
**Notebook:** [View on nbviewer](https://nbviewer.org/github/Binaz/AddressParser/blob/main/Address_Parser_Main.ipynb)  
**Date:** November 2025  

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

Each address is converted into a set of `(value, label)` pairs for easy downstream use.

---

#### Example 1  
**Address string →**  
`I 70 East Bound Mile Marker 112 Greenfield, IN`  
**Parsed address →**  
`[('112', 'STREET_NUM'), ('I 70 East Bound Mile Marker', 'STREET'), ('Greenfield', 'CITY'), ('IN', 'STATE')]`

#### Example 2  
**Address string →**  
`TT Accident-I-81 MM297 NB-Strasburg-VA--Shenandoah County`  
**Parsed address →**  
`[('I-81', 'STREET'), ('MM297', 'STREET_NUM'), ('Strasburg', 'CITY'), ('VA', 'STATE'), ('Shenandoah County', 'COUNTY')]`

#### Example 3  
**Address string →**  
`TT Accident-I-81 NB MM191.1-Lexington-VA--Rockbridge County`  
**Parsed address →**  
`[('I-81', 'STREET'), ('MM191.1', 'STREET_NUM'), ('Lexington', 'CITY'), ('VA', 'STATE'), ('Rockbridge County', 'COUNTY')]`

####  Example 4  
**Address string →**  
`Interstate 70 westbound at mile marker 163.7, Williamsburg, MO, 63388, Callaway`  
**Parsed address →**  
`[('Interstate 70', 'STREET'), ('163.7', 'STREET_NUM'), ('Williamsburg', 'CITY'), ('MO', 'STATE'), ('63388', 'POSTAL'), ('Callaway', 'COUNTY')]`

#### Example 5  
**Address string →**  
`I-95 & Namaans Road, Tri-State Mall, Claymont, DE 19703 US`  
**Parsed address →**  
`[('I-95', 'STREET'), ('Namaans Road', 'STREET'), ('Tri-State Mall', 'PLACE'), ('Claymont', 'CITY'), ('DE', 'STATE'), ('19703', 'POSTAL'), ('US', 'COUNTRY')]`

#### Example 6  
**Address string →**  
`3224 Phila Pike, Claymont, DE 19703 US`  
**Parsed address →**  
`[('3224', 'STREET_NUM'), ('Phila Pike', 'STREET'), ('Claymont', 'CITY'), ('DE', 'STATE'), ('19703', 'POSTAL'), ('US', 'COUNTRY')]`

#### Example 7  
**Address string →**  
`6000 Philadelphia Pike, Claymont, DE 19703 US`  
**Parsed address →**  
`[('6000', 'STREET_NUM'), ('Philadelphia Pike', 'STREET'), ('Claymont', 'CITY'), ('DE', 'STATE'), ('19703', 'POSTAL'), ('US', 'COUNTRY')]`

#### Example 8  
**Address string →**  
`35-A Salem Church Road, Newark, DE 19713 US`  
**Parsed address →**  
`[('35-A', 'STREET_NUM'), ('Salem Church Road', 'STREET'), ('Newark', 'CITY'), ('DE', 'STATE'), ('19713', 'POSTAL'), ('US', 'COUNTRY')]`

#### Example 9  
**Address string →**  
`28560 Landfill Lane, Jones Crossroads, Georgetown, DE 19947-6060 US`  
**Parsed address →**  
`[('28560', 'STREET_NUM'), ('Landfill Lane', 'STREET'), ('Jones Crossroads', 'PLACE'), ('Georgetown', 'CITY'), ('DE', 'STATE'), ('19947-6060', 'POSTAL'), ('US', 'COUNTRY')]`

#### Example 10  
**Address string →**  
`Kashyap Property-1284 Leland Road-Manassas-VA-20111-Prince William County`  
**Parsed address →**  
`[('1284', 'STREET_NUM'), ('Leland Road', 'STREET'), ('Manassas', 'CITY'), ('VA', 'STATE'), ('20111', 'POSTAL'), ('Prince William County', 'COUNTY')]`

#### Example 11  
**Address string →**  
`The construction site is at the corner of Signal Hill Road and Moore Drive in the 20111 area of Prince William County.`  
**Parsed address →**  
`[('Signal Hill Road', 'STREET'), ('Moore Drive', 'STREET'), ('20111', 'POSTAL'), ('Prince William County', 'COUNTY')]`

#### Example 12  
**Address string →**  
`The address of the suspected auto service station is 108 Harrison Street Southeast, Leesburg, VA 20198.`  
**Parsed address →**  
`[('108', 'STREET_NUM'), ('Harrison Street Southeast', 'STREET'), ('Leesburg', 'CITY'), ('VA', 'STATE'), ('20198', 'POSTAL')]`

#### Example 13  
**Address string →**  
`7890 Old Mill Road, Richmond, VA 23225`  
**Parsed address →**  
`[('7890', 'STREET_NUM'), ('Old Mill Road', 'STREET'), ('Richmond', 'CITY'), ('VA', 'STATE'), ('23225', 'POSTAL')]`

#### Example 14  
**Address string →**  
`Exit 54 on I-95 South near Fayetteville, NC`  
**Parsed address →**  
`[('I-95', 'STREET'), ('54', 'STREET_NUM'), ('Fayetteville', 'CITY'), ('NC', 'STATE')]`

#### Example 15  
**Address string →**  
`123 Main Street, Apartment 5B, Wilmington, DE 19801`  
**Parsed address →**  
`[('123', 'STREET_NUM'), ('Main Street', 'STREET'), ('Apartment 5B', 'UNIT'), ('Wilmington', 'CITY'), ('DE', 'STATE'), ('19801', 'POSTAL')]`
