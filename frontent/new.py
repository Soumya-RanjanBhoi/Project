import os
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, QED, rdMolDescriptors, Lipinski
from rdkit import DataStructs
import pandas as pd
from io import BytesIO
import py3Dmol
import joblib
import torch
import torch.nn as nn
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

solubility_model = joblib.load(os.path.join(MODEL_DIR, "model_solubility.pkl"))
drug_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model_drug.pkl"))
drug_scaler = joblib.load(os.path.join(MODEL_DIR, "drug_scaler_1.pkl"))

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


tox_model = ToxicityNet()
state_dict = torch.load(TOX_MODEL_PATH, map_location="cpu")
tox_model.load_state_dict(state_dict)
tox_model.eval()


def optimize_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, "Invalid SMILES"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    AllChem.EmbedMolecule(mol, params)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
        return mol, "Optimization successful"
    except:
        return mol, "Optimization failed, but 3D coords generated"

def render_3d(mol):
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    return viewer

def download_sdf(mol):
    sdf_block = Chem.MolToMolBlock(mol)
    buf = BytesIO()
    buf.write(sdf_block.encode("utf-8"))
    buf.seek(0)
    return buf

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
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

def feature_extract(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "Formal_Charge": Chem.GetFormalCharge(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "aromatic_ring": rdMolDescriptors.CalcNumAromaticRings(mol),
        "stereo_centre": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "rot_bond": Lipinski.NumRotatableBonds(mol),
        "qed_score": QED.qed(mol),
    }

def morgan_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def report_toxicity_pytorch(smiles, model, threshold=0.75):
    fp = morgan_fingerprint(smiles)
    x = torch.tensor([fp], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()[0]
        preds = (probs >= threshold).astype(int)

    tasks = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    report_lines = [f"Toxicity Report for molecule: {smiles}", "-" * 40]
    for i, task in enumerate(tasks):
        if preds[i] == 1:
            report_lines.append(f"[{task}] Toxicity Likely (p={probs[i]:.2f})")
    if len(report_lines) == 2:
        report_lines.append("No significant toxicity predicted.")
    return "\n".join(report_lines)


st.set_page_config(page_title="Molecule Analyzer", layout="wide")
st.title("🧪 Molecular Property Predictor")

smiles = st.text_input("Enter SMILES string:", value="CCO")

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.image(Draw.MolToImage(mol, size=(300, 300)), caption="2D Structure")

        tab1, tab2 = st.tabs(["🔄 Structure Optimization", "📊 Predictions"])

        with tab1:
            if st.button("Optimize Structure"):
                optimized_mol, msg = optimize_structure(smiles)
                if optimized_mol:
                    st.success(msg)
                    st.image(Draw.MolToImage(optimized_mol, size=(300, 300)), caption="Optimized Structure")

                    st.subheader("🌀 Interactive 3D View")
                    viewer = render_3d(optimized_mol)
                    viewer_html = viewer._make_html()
                    st.components.v1.html(viewer_html, height=450)

                    st.download_button(
                        "📥 Download Optimized Structure (SDF)",
                        data=download_sdf(optimized_mol),
                        file_name="optimized_structure.sdf",
                        mime="chemical/x-mdl-sdfile"
                    )

        with tab2:
            if st.button("Run Predictions"):
                desc = get_descriptors(smiles)
                sol_pred = solubility_model.predict([list(desc.values())])[0]

                feat = feature_extract(smiles)
                param = [
                    feat["MolWt"], feat["LogP"], feat["Formal_Charge"], feat["NumHDonors"],
                    feat["NumHAcceptors"], feat["aromatic_ring"], feat["stereo_centre"], feat["rot_bond"]
                ]
                scaled = drug_scaler.transform([param])
                drug_pred = drug_model.predict_proba(scaled)
                qed_score = feat["qed_score"]

                tox_report = report_toxicity_pytorch(smiles, tox_model)

                st.subheader("🔹 Predicted Solubility")
                st.write(f"**Solubility:** {sol_pred:.2f}")
                st.dataframe(pd.DataFrame(desc.items(), columns=["Feature", "Value"]))

                st.subheader("🔹 Drug-likeness")
                st.write(f"**Class:** {'Likely Drug' if qed_score >= 0.5 else 'Non-drug'}")
                st.write(f"**QED SCORE :** {qed_score:.3f}")

                st.subheader("🔹 Toxicity Report")
                st.code(tox_report)

                report_text = f"""SMILES: {smiles}\n----------------------------------------\nSolubility: {sol_pred:.2f}\n\nDrug-likeness:\n - Class: {'Likely Drug' if qed_score >= 0.5 else 'Non-drug'}\n - QED Score: {qed_score:.3f}\n\nToxicity Report:\n{tox_report}\n"""
                buf = BytesIO()
                buf.write(report_text.encode())
                buf.seek(0)
                st.download_button("📄 Download TXT Report", data=buf, file_name="molecule_report.txt", mime="text/plain")
    else:
        st.error("Invalid SMILES string. Please check your input.")
