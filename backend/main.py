from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from typing import Annotated
from fastapi.responses import JSONResponse
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED, rdMolDescriptors, Lipinski
from rdkit import DataStructs
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import uvicorn  


app = FastAPI(title="Molecular Predictor API")



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

solubility_model = joblib.load(os.path.join(MODEL_DIR, "model_solubility.pkl"))
drug_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model_drug.pkl"))
drug_scaler = joblib.load(os.path.join(MODEL_DIR, "drug_scaler_1.pkl"))

tox_model = None
TOX_MODEL_PATH = os.path.join(MODEL_DIR, "final_tox21_model_state.pt")



class ToxicityNet(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2000, output_dim=12, dropout=0.18, stddev=0.025):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        nn.init.normal_(self.fc1.weight, mean=0, std=stddev)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.fc2.weight, mean=0, std=stddev)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


@app.on_event("startup")
def load_toxicity_model():
    """Load PyTorch toxicity model on startup"""
    global tox_model
    state_dict = torch.load(TOX_MODEL_PATH, map_location="cpu")
    model = ToxicityNet()
    model.load_state_dict(state_dict)
    model.eval()
    tox_model = model
    print("✅ Toxicity model loaded successfully!")



def morgan_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def report_toxicity_pytorch(smiles, model, threshold=0.75):
    fp = morgan_fingerprint(smiles)
    if fp is None:
        return "Invalid SMILES string."

    x = torch.tensor([fp], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()[0]
        preds = (probs >= threshold).astype(int)

    tasks = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    assay_info = {
        "NR-AR": "Possible interference with androgen receptors (hormone disruption).",
        "NR-AR-LBD": "Binding to androgen receptor ligand domain (affects hormone signaling).",
        "NR-AhR": "Activation of aryl hydrocarbon receptor (potential carcinogen response).",
        "NR-Aromatase": "Affects estrogen synthesis (endocrine disruption).",
        "NR-ER": "Interaction with estrogen receptor (hormone disruption).",
        "NR-ER-LBD": "Binding to estrogen receptor ligand domain.",
        "NR-PPAR-gamma": "Affects metabolism and inflammation pathways.",
        "SR-ARE": "Activates antioxidant response (oxidative stress).",
        "SR-ATAD5": "DNA damage repair pathway activation (genotoxic stress).",
        "SR-HSE": "Heat shock protein activation (cellular stress response).",
        "SR-MMP": "Mitochondrial toxicity (energy metabolism disruption).",
        "SR-p53": "p53 protein activation (DNA damage and apoptosis)."
    }

    report_lines = [f"Toxicity Report for molecule: {smiles}", "-" * 40]
    any_toxic = False

    for i, task in enumerate(tasks):
        if preds[i] == 1:
            any_toxic = True
            report_lines.append(f"[{task}] Toxicity Likely (Probability: {probs[i]:.2f})")
            report_lines.append(f"   ↳ {assay_info.get(task, 'No description available.')}")
    if not any_toxic:
        report_lines.append("No significant toxicity predicted.")

    return "\n".join(report_lines)


def get_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RingCount": Descriptors.RingCount(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "Fsp3": Descriptors.FractionCSP3(mol),
    }


def feature_extract(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid Smiles"
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "Formal_Charge": Chem.GetFormalCharge(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "armotic_ring": rdMolDescriptors.CalcNumAromaticRings(mol),
        "stero_centre": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "rot_bond": Lipinski.NumRotatableBonds(mol),
        "qed_score": QED.qed(mol),
    }



class Molecular_var(BaseModel):
    smiles: Annotated[str, Field(..., description="SMILES of the molecule", example="CCO")]

    @validator("smiles")
    def valid_smiles(cls, v):
        mol = Chem.MolFromSmiles(v)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        return v



@app.get("/")
def root():
    return {"message": "Molecular predictor API is running 🚀"}


@app.post("/predictSolubility")
def predict_solubility(smiles: Molecular_var):
    details = get_descriptors(smiles.smiles)
    if details is None:
        return JSONResponse(status_code=400, content={"error": "Invalid SMILES"})

    params = [
        details["MolWt"], details["LogP"], details["NumHDonors"],
        details["NumHAcceptors"], details["TPSA"], details["RingCount"],
        details["NumRotatableBonds"], details["Fsp3"]
    ]

    value = solubility_model.predict([params])
    return {"smiles": smiles.smiles, "predicted_solubility": round(float(value[0]), 2), "descriptors": details}


@app.post("/predictTox")
def predict_toxicity(smiles: Molecular_var):
    if tox_model is None:
        return JSONResponse(status_code=500, content={"error": "Toxicity model not loaded"})

    report = report_toxicity_pytorch(smiles.smiles, tox_model)
    return {"toxicity_report": report}


@app.post("/predictDrug")
def predict_drug(smiles: Molecular_var):
    desc = feature_extract(smiles.smiles)
    if desc == "Invalid Smiles":
        return JSONResponse(status_code=400, content={"error": "Invalid SMILES"})

    param = [
        desc["MolWt"], desc["LogP"], desc["Formal_Charge"], desc["NumHDonors"],
        desc["NumHAcceptors"], desc["armotic_ring"], desc["stero_centre"], desc["rot_bond"]
    ]

    scaled = drug_scaler.transform([param])
    prediction = drug_model.predict_proba(scaled)

    return {
        "Drug Class": int(np.argmax(prediction)),
        "Drug Probability": float(np.max(prediction)),
        "qed_Score": desc["qed_score"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
