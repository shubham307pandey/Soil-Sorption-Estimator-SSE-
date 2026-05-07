import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from rasar import calculate_descriptor
import joblib




def std_ad(df: pd.DataFrame, avg, stdev):
    """
    Standard AD calculation using mean and std of training data.
    """

    df = df.copy()
    #mean = pd.Series(avg)
    #std = pd.Series(stdev)
    mean = avg
    std = stdev

    std_df = np.abs(df - mean) / std
    print(std_df)

    sk_score = std_df.mean(axis=1) + (1.28 * std_df.std(axis=1))
    max_std = std_df.max(axis=1)
    min_std = std_df.min(axis=1)

    ad_status = np.full(len(df), "Inside AD", dtype=object)

    # Conditions
    cond1 = (max_std > 3) & (min_std > 3)
    cond2 = ((max_std > 3) & (min_std < 3)) & (sk_score > 3)
    cond3 = ((max_std < 3) & (min_std > 3)) & (sk_score > 3)

    ad_status[cond1 | cond2 | cond3] = "Outside AD"

    df["AD Status"] = ad_status

    return df


#datasets

qsar_tr = pd.read_excel("Training set QSPR.xlsx", index_col=0)
rasar_tr = pd.read_excel("Training Set q-RASPR.xlsx", index_col=0)

#load_model
qsar_model = joblib.load("qsar_pls.joblib")
rasar_model = joblib.load("rasar_mlr.joblib")


##Custom Font
st.set_page_config(page_title="SEE- A LogKoc Prediction Tool", layout="wide", page_icon="🧪")

custom_css = """
<style>

    .stApp {
        font-family: 'Times New Roman', serif;
    }

    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        font-family: 'Times New Roman', serif !important;
        font-style: italic !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        font-family: 'Times New Roman', serif !important;
        font-style: italic !important;
        font-weight: bold;
    }

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Function to calculate required descriptors from SMILES
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        descriptors = {
            'MolLogP': Descriptors.MolLogP(mol),
            'RingCount': rdMolDescriptors.CalcNumRings(mol),
            'fr_nitro_arom': Fragments.fr_nitro_arom(mol),
            'MaxAbsEStateIndex': Descriptors.MaxAbsEStateIndex(mol)
        }
        return descriptors
    except:
        return None
    



st.sidebar.image("SSE.png", width=400)
Model_selection= st.sidebar.selectbox("Select Regression Model", ["PLS-QSPR Model", "Univariate q-RASPR Model"])

#Model Training
def prediction(data, type):
    if type == "PLS-QSPR Model":
        try:
            pred=qsar_model.predict(data)

        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None


    elif type == "Univariate q-RASPR Model":
        try:
            pred=rasar_model.predict(data)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None
    else:
        st.error(f"Invalid model selection: {type}")
        return None

    return pred

def main():
    st.title("Soil Sorption Estimator (SSE)")
    st.write("*Predict LogKoc using PLS QSPR or Univariate q-RASPR models from SMILES.")


    # Input method
    input_method = st.radio("Input Method", ["Single SMILES", "Upload Excel"])

    
    if input_method == "Single SMILES":
        smiles = st.text_input("Enter SMILES string:")
        smiles_list = [smiles]

    
    else:
        uploaded_file = st.file_uploader("Upload Excel with 'SMILES' column", type=['xlsx'])
        if uploaded_file:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                pass
            if 'SMILES' in df.columns:
                smiles_list = df['SMILES'].dropna().tolist()
                st.success(f"Loaded {len(smiles_list)} compounds")
            else:
                st.error("File must have a 'SMILES' column")

    
    if st.button("Predict LogKoc"):
        calc_desc = []
        for smi in smiles_list:
            desc = calculate_descriptors(smi)
            if desc is None:
                calc_desc.append({
                                'MolLogP': None,
                                'RingCount': None,
                                'fr_nitro_arom': None,
                                'MaxAbsEStateIndex': None})
                continue
            else:
                calc_desc.append(desc)
       
    
        if Model_selection == "PLS-QSPR Model":
            desc_df = pd.DataFrame(calc_desc)
    
            pred = prediction(desc_df, Model_selection)
            ad_status= std_ad(desc_df, avg=qsar_tr.iloc[:,:-1].mean(), stdev=qsar_tr.iloc[:,:-1].std())
        if Model_selection == "Univariate q-RASPR Model":
            desc_df = pd.DataFrame(calc_desc)
            __, rasar_ds = calculate_descriptor(df1=qsar_tr, df2=desc_df, method="Laplacian Kernel", ctc=10, gamma=1)
            test_rasar = rasar_ds["RA_function"]
            test_rasar.name = "RA function"
            test_rasar_df = pd.DataFrame(test_rasar)
            pred = prediction(test_rasar_df, Model_selection)
            ad_status= std_ad(test_rasar_df, avg=rasar_tr.iloc[:,:-1].mean(), stdev= rasar_tr.iloc[:,:-1].std())

            
        result_df = pd.DataFrame(pred, columns=["Predicted LogKoc"], index=smiles_list)
        result_df["AD Status"] = ad_status["AD Status"].values
        result_df.index.name = "SMILES"
        st.success("Predictions Complete!")
        st.dataframe(result_df)

        # Download
        csv = result_df.to_csv(index=True)
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

    st.sidebar.header("About")
    st.sidebar.info("""
    - **PLS QSPR Model**: Uses MolLogP, RingCount, fr_nitro_arom, MaxAbsEStateIndex
    - **q-RASPR Model**: Uses RA function(LK)
    - Structural & Physiochemical descriptor are calculated using RDKit
    """)

    st.sidebar.markdown("### Requirements")
    st.sidebar.code("""
pip install streamlit rdkit scikit-learn pandas openpyxl rasar
    """)

if __name__ == "__main__":
    main()



if "show_info" not in st.session_state:
    st.session_state.show_info = False
if "show_manual" not in st.session_state:
    st.session_state.show_manual = False

st.divider()
col1, col2, col3 = st.columns([1.5, 6, 1.5])

with col1:
    if st.button("ℹ️ Contact Info", use_container_width=True):
        st.session_state.show_info = not st.session_state.show_info
        st.session_state.show_manual = False

with col3:
    if st.button("📖 User Manual", use_container_width=True):
        st.session_state.show_manual = not st.session_state.show_manual
        st.session_state.show_info = False


if st.session_state.show_info:
    st.markdown("""
    <div class="panel-box contact">
 
      <p class="panel-title contact"> Contact Information</p>
 
      <p class="sec-heading contact"> Contact </p>
      <div class="contact-card" style='font-family: "Times New Roman", serif; font-size: 16px; font-style: italic;'>
        <strong>Prof. (Dr.) Kunal Roy</strong><br>
        <a href="mailto:kunal.roy@jadavpuruniversity.in">kunal.roy@jadavpuruniversity.in</a><br>
        Drug Theoretics and Cheminformatics Laboratory<br>
        Jadavpur University, Kolkata, IN
      </div>
      </p>
    """, unsafe_allow_html=True)

if st.session_state.show_manual:
      with open("User_Manual.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

        st.markdown("""
    <div style='padding: 10px; background-color: #e6f3ff;
                border-radius: 8px; margin-top: 10px;'>
        <h4>User Manual</h4>
        <p>Click below to download the user manual.</p>
    </div>
    """, unsafe_allow_html=True)

        st.download_button(
        label="📖 Download User Manual",
        data=PDFbyte,
        file_name="User_Manual.pdf",
        mime="application/pdf"
    )
