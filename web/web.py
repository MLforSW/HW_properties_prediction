import warnings
from rdkit import RDLogger
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

import io
import base64
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# --- 1. 加载模型 & scaler 路径 ---
model_paths = {
    "Toxicity":        "toxicity_xgboost_model.joblib",
    "Reactivity":      "reactivity_xgboost_model.joblib",
    "Flammability":    "flammability_xgb_model.joblib",
    "Water Reactivity":"reactivitywater_xgb_model.joblib"
}
scaler_paths = {
    "Toxicity":        "scaler_toxicity.joblib",
    "Reactivity":      "scaler_reactivity.joblib",
    "Flammability":    "scaler_flammability.joblib",
    "Water Reactivity":"scaler_water_reactivity.joblib"
}
models = {p: joblib.load(m) for p,m in model_paths.items()}

# --- 2. 载入各数据集 ---
df_map = {
    "Toxicity":        pd.read_csv("./Dataset/imputed_selected_features_Toxcity.csv"),
    "Reactivity":      pd.read_csv("./Dataset/imputed_selected_features_Reactivity.csv"),
    "Flammability":    pd.read_csv("./Dataset/imputed_selected_features_Flam.csv"),
    "Water Reactivity":pd.read_csv("./Dataset/imputed_selected_features_W.csv")
}

# --- 3. SMILES -> 特征 ---
def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return np.concatenate([desc, arr])

def mol_to_base64(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(300,300))
    buf = io.BytesIO()
    img.save(buf,'PNG'); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# --- 4. 网页模板 ---
tpl = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SMILES Hazard Prediction</title>
<style>
  body { font-family: Arial, sans-serif; margin:2rem; }
  .container { max-width:700px; margin:auto; }
  input, select, button { width:100%; padding:8px; margin:8px 0; }
  .result { font-size:1.2em; margin:1em 0; }
  img { max-width:100%; display:block; margin:1em auto; }
  .analysis { display:none; margin-top:1em; }
  .btn-group button { margin-right:8px; }
</style>
<script>
function toggleSection(id) {
  const sec = document.getElementById(id);
  sec.style.display = sec.style.display === 'block' ? 'none' : 'block';
}
</script>
</head>
<body>
<div class="container">
  <h1>SMILES Hazard Prediction</h1>
  <form method="post" action="/predict">
    <input name="smiles" placeholder="Enter SMILES string" required />
    <select name="prop">
      {% for p in props %}
      <option>{{p}}</option>
      {% endfor %}
    </select>
    <button type="submit">Predict</button>
  </form>

  {% if result %}
  <div class="result">{{ result }}</div>
  {% endif %}

  {% if img %}
  <h3>Molecule</h3>
  <img src="data:image/png;base64,{{img}}" alt="Molecule" />
  {% endif %}

  {% if shap_png or ice_png or pdp_png %}
  <div class="btn-group">
    {% if shap_png %}<button type="button" onclick="toggleSection('shap')">SHAP Analysis</button>{% endif %}
    {% if ice_png %}<button type="button" onclick="toggleSection('ice')">ICE Analysis</button>{% endif %}
    {% if pdp_png %}<button type="button" onclick="toggleSection('pdp')">PDP Analysis</button>{% endif %}
  </div>

  {% if shap_png %}
  <div id="shap" class="analysis">
    <h3>SHAP Waterfall</h3>
    <img src="data:image/png;base64,{{ shap_png }}" alt="SHAP" />
  </div>
  {% endif %}

  {% if ice_png %}
  <div id="ice" class="analysis">
    <h3>ICE Top9 Features</h3>
    <img src="data:image/png;base64,{{ ice_png }}" alt="ICE" />
  </div>
  {% endif %}

  {% if pdp_png %}
  <div id="pdp" class="analysis">
    <h3>PDP Top9 Features</h3>
    <img src="data:image/png;base64,{{ pdp_png }}" alt="PDP" />
  </div>
  {% endif %}
  {% endif %}
</div>
</body>
</html>
'''

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template_string(tpl, props=list(models), result=None,
                                  img=None, shap_png=None, ice_png=None, pdp_png=None)

@app.route('/predict', methods=['POST'])
def predict():
    smiles = request.form['smiles'].strip()
    prop    = request.form['prop']
    df      = df_map[prop]
    img     = mol_to_base64(smiles)

    # 查找SMILES
    row = df[df.SMILES == smiles]
    if row.empty:
        return render_template_string(tpl, props=list(models), result="❌ SMILES not found",
                                      img=img, shap_png=None, ice_png=None, pdp_png=None)

    extra = row.iloc[:,1:-2].values.ravel()
    feats = featurize_smiles(smiles)
    if feats is None:
        return render_template_string(tpl, props=list(models), result="❌ Invalid SMILES",
                                      img=img, shap_png=None, ice_png=None, pdp_png=None)

    # 标准化
    X_raw = np.concatenate([feats, extra])
    scaler = joblib.load(scaler_paths[prop])
    X      = scaler.transform(X_raw.reshape(1,-1))

    # 预测
    model = models[prop]
    proba = model.predict_proba(X)[0,1]
    label = model.predict(X)[0]
    result= f"{'✔️ Has' if label==1 else '❌ Does not have'} “{prop}” (prob={proba:.3f})"

    # SHAP
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv,list): sv, base = sv[1][0], explainer.expected_value[1]
    else:                sv, base = (sv[0] if sv.ndim==2 else sv), explainer.expected_value

    # 特征名
    desc_cols = ["MolWt","MolLogP","HDonor","HAcceptor"]
    fp_cols   = [f"Bit_{i}" for i in range(2048)]
    extra_cols= df.columns[1:-2].tolist()
    fnames = desc_cols + fp_cols + extra_cols

    shap_exp = shap.Explanation(values=sv, base_values=base, data=X[0], feature_names=fnames)
    plt.figure(figsize=(8,4)); shap.plots.waterfall(shap_exp, show=False)
    buf=io.BytesIO(); plt.savefig(buf,format='png',bbox_inches='tight'); buf.seek(0)
    shap_png=base64.b64encode(buf.read()).decode(); plt.clf()

    # Top9
    top9 = np.argsort(np.abs(sv))[-9:][::-1]

    # 重建全样本X
    all_feats = np.vstack([featurize_smiles(s) for s in df.SMILES])
    extras    = df.iloc[:,1:-2].values
    Xs = scaler.transform(np.hstack([all_feats, extras]))

    # ICE
    fig,axes=plt.subplots(3,3,figsize=(12,12))
    for ax,idx in zip(axes.flat, top9):
        PartialDependenceDisplay.from_estimator(model, Xs, [idx], kind='individual', ax=ax,
                                                line_kw={'alpha':0.3,'color':'blue'})
        ax.set_title(fnames[idx])
    fig.tight_layout()
    buf=io.BytesIO(); plt.savefig(buf,format='png',bbox_inches='tight'); buf.seek(0)
    ice_png=base64.b64encode(buf.read()).decode(); plt.clf()

    # PDP
    fig,axes=plt.subplots(3,3,figsize=(12,12))
    for ax,idx in zip(axes.flat, top9):
        PartialDependenceDisplay.from_estimator(model, Xs, [idx], kind='average', ax=ax,
                                                line_kw={'color':'green','linewidth':2})
        ax.set_title(fnames[idx])
    fig.tight_layout()
    buf=io.BytesIO(); plt.savefig(buf,format='png',bbox_inches='tight'); buf.seek(0)
    pdp_png=base64.b64encode(buf.read()).decode(); plt.clf()

    return render_template_string(tpl, props=list(models), result=result,
                                  img=img, shap_png=shap_png,
                                  ice_png=ice_png, pdp_png=pdp_png)

if __name__ == '__main__':
    app.run(debug=True)
