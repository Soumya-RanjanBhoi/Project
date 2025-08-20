import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, QED, rdMolDescriptors, Lipinski
from rdkit import DataStructs
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from io import StringIO, BytesIO
import py3Dmol
import xgboost as xgb


with open('model/model_solubility.pkl', 'rb') as f:
    solubility_model = pickle.load(f)

with open('model/xgb_model_drug.pkl', 'rb') as f:
    drug_model = pickle.load(f)

with open('model/drug_scaler_1.pkl', 'rb') as f:
    drug_scaler = pickle.load(f)



class ToxicityNet(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2000, output_dim=12, dropout=0.18, stddev=0.025):
        super(ToxicityNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        nn.init.normal_(self.fc1.weight, mean=0, std=stddev)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.fc2.weight, mean=0, std=stddev)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


tox_model = ToxicityNet()
state_dict = torch.load("model/final_tox21_model_state.pt", map_location="cpu")
tox_model.load_state_dict(state_dict)
tox_model.eval()

# ------------------ Helper Functions ------------------
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

def get_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RingCount': Descriptors.RingCount(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'Fsp3': Descriptors.FractionCSP3(mol)
    }

def feature_extract(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "Formal_Charge": Chem.GetFormalCharge(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "armotic_ring": rdMolDescriptors.CalcNumAromaticRings(mol),
        "stero_centre": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "rot_bond": Lipinski.NumRotatableBonds(mol),
        "qed_score": QED.qed(mol)
    }

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
        prob = probs[i]
        pred = preds[i]
        if pred == 1:
            any_toxic = True
            report_lines.append(f"[{task}] Toxicity Likely (Probability: {prob:.2f})")
            report_lines.append(f"   ↳ {assay_info.get(task, 'No description available.')}")
    if not any_toxic:
        report_lines.append("No significant toxicity predicted.")

    return "\n".join(report_lines)

# ------------------ Streamlit UI ------------------
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
                    st.components.v1.html(viewer._make_html(), height=450)

                    sdf_buffer = StringIO()
                    writer = Chem.SDWriter(sdf_buffer)
                    writer.write(optimized_mol)
                    writer.close()

                    sdf_bytes = sdf_buffer.getvalue().encode("utf-8")
                    st.download_button("📥 Download Optimized Structure (SDF)",
                                       data=sdf_bytes,
                                       file_name="optimized_structure.sdf",
                                       mime="chemical/x-mdl-sdfile")

        with tab2:
            if st.button("Run Predictions"):
                with st.spinner("Predicting..."):
                    # Solubility
                    details = get_descriptors(smiles)
                    params = [
                        details['MolWt'], details['LogP'], details['NumHDonors'],
                        details['NumHAcceptors'], details['TPSA'], details['RingCount'],
                        details['NumRotatableBonds'], details['Fsp3']
                    ]
                    solubility = solubility_model.predict([params])[0]

                    # Drug-likeness
                    desc = feature_extract(smiles)
                    param = [
                        desc['MolWt'], desc['LogP'], desc['Formal_Charge'], desc['NumHDonors'],
                        desc['NumHAcceptors'], desc['armotic_ring'], desc['stero_centre'], desc['rot_bond']
                    ]
                    scaled = drug_scaler.transform([param])
                    prediction = drug_model.predict_proba(scaled)

                    drug_class = int(np.argmax(prediction))
                    drug_prob = float(np.max(prediction))
                    qed_score = desc['qed_score']

                    # Toxicity
                    tox_report = report_toxicity_pytorch(smiles, tox_model)

                # Show Results
                st.subheader("🔹 Predicted Solubility")
                st.write(f"**Solubility:** {solubility:.2f}")
                st.dataframe(pd.DataFrame(details.items(), columns=["Feature", "Value"]))

                st.subheader("🔹 Drug-likeness")
                st.write(f"**Class:** {'Likely Drug' if drug_class == 1 else 'Non-drug'}")
                st.write(f"**Probability:** {drug_prob:.3f}")
                st.write(f"**QED SCORE:** {qed_score:.3f}")

                st.subheader("🔹 Toxicity Report")
                st.code(tox_report)

                # Download Report
                report_text = f"""SMILES: {smiles}
----------------------------------------
Solubility: {solubility:.2f}

Drug-likeness:
 - Class: {'Likely Drug' if drug_class == 1 else 'Non-drug'}
 - Probability: {drug_prob:.3f}
 - QED Score: {qed_score:.3f}

Toxicity Report:
{tox_report}
"""
                buf = BytesIO()
                buf.write(report_text.encode())
                buf.seek(0)
                st.download_button("📄 Download TXT Report", data=buf,
                                   file_name="molecule_report.txt", mime="text/plain")
    else:
        st.error("Invalid SMILES string. Please check your input.")
