import os
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import requests
import pandas as pd
from io import BytesIO
import py3Dmol


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Molecule Analyzer", layout="wide")

st.title("🧪 Molecular Property Predictor")


smiles = st.text_input(
    "Enter SMILES string:",
    value="CCO",
    help="Type a valid SMILES (e.g., CCO for ethanol)"
)

def optimize_structure(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, "❌ Invalid SMILES"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    AllChem.EmbedMolecule(mol, params)

    try:
        AllChem.MMFFOptimizeMolecule(mol)
        return mol, "✅ Optimization successful"
    except:
        return mol, "⚠ Optimization failed, but 3D coords generated"

def render_3d(mol):
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mblock, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    return viewer

def download_sdf(mol):
    buf = BytesIO()
    writer = Chem.SDWriter(buf)
    writer.write(mol)
    writer.close()
    return buf.getvalue()


if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.image(Draw.MolToImage(mol, size=(300, 300)), caption="2D Structure")

        tab1, tab2 = st.tabs(["🔄 Structure Optimization", "📊 Predictions"])

        
        with tab1:
            if st.button("Optimize Structure"):
                with st.spinner("Optimizing structure..."):
                    optimized_mol, msg = optimize_structure(smiles)
                    if optimized_mol:
                        st.success(msg)
                        st.image(Draw.MolToImage(optimized_mol, size=(300, 300)),
                                 caption="Optimized Structure")

                        st.subheader("🌀 Interactive 3D View")
                        viewer = render_3d(optimized_mol)
                        st.components.v1.html(viewer._make_html(), height=450)

                        st.download_button(
                            "📥 Download Optimized Structure (SDF)",
                            data=download_sdf(optimized_mol),
                            file_name="optimized_structure.sdf",
                            mime="chemical/x-mdl-sdfile"
                        )

        with tab2:
            if st.button("Run Predictions"):
                data = {"smiles": smiles}
                try:
                    with st.spinner("Predicting..."):
                        sol = requests.post(f"{API_URL}/predictSolubility", json=data).json()
                        drug = requests.post(f"{API_URL}/predictdrug", json=data).json()
                        tox = requests.post(f"{API_URL}/predictTox", json=data).json()

                    st.subheader("🔹 Predicted Solubility")
                    st.write(f"**Solubility:** {sol.get('predicted_solubility', 'N/A')}")
                    if "descriptors" in sol:
                        st.write("**Descriptors:**")
                        st.dataframe(pd.DataFrame(sol["descriptors"].items(),
                                                  columns=["Feature", "Value"]))

                    st.subheader("🔹 Drug-likeness")
                    qed = drug.get("qed_Score", 0.0)
                    st.write(f"**Class:** {'Likely Drug' if qed >= 0.5 else 'Non-drug'}")
                    st.write(f"**QED SCORE:** {qed:.3f}")

                    st.subheader("🔹 Toxicity Report")
                    st.code(tox.get("toxicity_report", "No report available"))

                    report_text = f"""
SMILES: {smiles}
----------------------------------------
Solubility: {sol.get('predicted_solubility', 'N/A')}

Drug-likeness:
 - Class: {'Likely Drug' if qed >= 0.5 else 'Non-drug'}
 - QED Score: {qed:.3f}

Toxicity Report:
{tox.get('toxicity_report', 'N/A')}
"""
                    buf = BytesIO(report_text.encode())
                    st.download_button(
                        "📄 Download TXT Report",
                        data=buf,
                        file_name="molecule_report.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Failed to connect to API at {API_URL}. Error: {e}")
    else:
        st.error("❌ Invalid SMILES string. Please check your input.")
