"""
HTTP Header Vulnerability Detection — Model Training & Evaluation Pipeline
Trains baseline ML models (Random Forest, SVM, XGBoost, Gradient Boosting)
and Deep Learning architectures (CNN-LSTM, BiLSTM+Attention, Transformer Fusion),
evaluates performance, saves artifacts, and tests live URL scanning.
"""

import os
import sys

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import random
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from tabulate import tabulate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    'DATASET_FILE': 'training_dataset.csv',
    'AUGMENT_ONLINE': False,
    'ONLINE_URLS': 50,
    'FETCH_WORKERS': 10,
    'FETCH_TIMEOUT': 5,
    'MAX_SEQ_LEN': 256,
    'VOCAB_SIZE': 8000,
    'EMBED_DIM': 128,
    'LSTM_UNITS': 128,
    'ATTN_HEADS': 4,
    'TRANSFORMER_DIM': 128,
    'FF_DIM': 256,
    'DROPOUT': 0.3,
    'BATCH_SIZE': 64,
    'EPOCHS': 25,
    'PATIENCE': 6,
    'TEST_SIZE': 0.2,
    'MODEL_DIR': 'saved_models',
}
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)

CLASS_NAMES = {0: 'Secure', 1: 'Low Risk', 2: 'Medium Risk', 3: 'High Risk'}
PALETTE = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

print(f'✅ TensorFlow {tf.__version__}', flush=True)
print(f'✅ NumPy {np.__version__} | Pandas {pd.__version__}', flush=True)
gpus = tf.config.list_physical_devices('GPU')
print(f'✅ GPU available: {len(gpus) > 0} ({[g.name for g in gpus] if gpus else "CPU vectorized mode"})', flush=True)

# ── Security & Leak Headers ──────────────────────────────────────────────────
SECURITY_HEADERS = [
    'strict-transport-security', 'content-security-policy',
    'x-frame-options', 'x-content-type-options', 'x-xss-protection',
    'referrer-policy', 'permissions-policy', 'cache-control', 'expect-ct',
    'cross-origin-embedder-policy', 'cross-origin-opener-policy',
    'cross-origin-resource-policy', 'access-control-allow-origin',
    'access-control-allow-credentials',
]
LEAK_HEADERS = [
    'server', 'x-powered-by', 'x-aspnet-version', 'x-aspnetmvc-version',
    'x-generator', 'x-drupal-cache', 'x-wordpress-cache', 'x-runtime',
    'x-version', 'x-backend-server', 'x-php-version', 'x-framework',
    'via', 'x-forwarded-server',
]

def vulnerability_score(row) -> int:
    s = 0
    if not row.get('https', 1):                          s += 3
    if not row.get('has_strict_transport_security', 0):  s += 3
    elif row.get('hsts_max_age', 0) < 31536000:          s += 2
    if not row.get('csp_present', 0):                    s += 3
    else:
        if row.get('csp_unsafe_inline', 0): s += 2
        if row.get('csp_unsafe_eval',   0): s += 2
        if row.get('csp_wildcard',      0): s += 1

    if not row.get('has_x_frame_options', 0):            s += 2
    elif row.get('xfo_allowfrom', 0):                    s += 1
    if not row.get('xcto_nosniff', 0):                   s += 2
    if not row.get('has_referrer_policy', 0):            s += 2
    elif row.get('rp_unsafe_url', 0):                    s += 2
    if row.get('cookie_present', 0):
        if not row.get('cookie_secure',   0): s += 2
        if not row.get('cookie_httponly', 0): s += 2
        if not row.get('cookie_samesite', 0): s += 1

    if row.get('cors_cred_wildcard', 0):                 s += 4
    elif row.get('cors_wildcard', 0):                    s += 1

    s += min(row.get('info_leak_count', 0), 4)
    if row.get('server_version_exposed', 0):             s += 2
    if row.get('x_powered_by_present',   0):             s += 1

    if row.get('has_permissions_policy', 0):             s -= 1
    if row.get('has_coep', 0):                           s -= 1
    if row.get('has_coop', 0):                           s -= 1
    if row.get('hsts_include_subdomains', 0):            s -= 1
    if row.get('hsts_preload', 0):                       s -= 1

    s = max(s, 0)
    if   s <= 2:  return 0
    elif s <= 6:  return 1
    elif s <= 12: return 2
    else:         return 3

def extract_features(h: dict) -> dict:
    f = {}
    f['url']          = h.get('_url', '')
    f['status']       = h.get('_status', 0)
    f['https']        = int(h.get('_url', '').startswith('https'))
    f['header_count'] = len([k for k in h if not k.startswith('_')])

    hsts = h.get('strict-transport-security', '')
    f['has_strict_transport_security'] = int(bool(hsts))
    m = re.search(r'max-age=(\d+)', hsts, re.I)
    f['hsts_max_age']          = int(m.group(1)) if m else 0
    f['hsts_include_subdomains']= int('includesubdomains' in hsts.lower())
    f['hsts_preload']           = int('preload' in hsts.lower())
    f['hsts_valid']             = int(f['hsts_max_age'] >= 31536000)

    csp = h.get('content-security-policy', '')
    f['has_content_security_policy'] = int(bool(csp))
    f['csp_present']           = int(bool(csp))
    cv = csp.lower()
    f['csp_unsafe_inline']     = int("'unsafe-inline'" in cv)
    f['csp_unsafe_eval']       = int("'unsafe-eval'"   in cv)
    f['csp_wildcard']          = int(' * ' in cv or cv.startswith('*'))
    f['csp_allow_data']        = int('data:'           in cv)
    f['csp_directive_count']   = len([d for d in csp.split(';') if d.strip()])

    xfo = h.get('x-frame-options', '').upper()
    f['has_x_frame_options']   = int(bool(xfo))
    f['xfo_deny']              = int('DENY'       in xfo)
    f['xfo_sameorigin']        = int('SAMEORIGIN' in xfo)
    f['xfo_allowfrom']         = int('ALLOW-FROM' in xfo)

    xcto = h.get('x-content-type-options', '')
    f['has_x_content_type_options'] = int(bool(xcto))
    f['xcto_nosniff']          = int('nosniff' in xcto.lower())

    xxss = h.get('x-xss-protection', '')
    f['has_x_xss_protection']  = int(bool(xxss))
    f['xxss_enabled']          = int('1' in xxss)
    f['xxss_block']            = int('mode=block' in xxss.lower())
    f['xxss_report_uri']       = int('report=' in xxss.lower())

    rp = h.get('referrer-policy', '').lower()
    f['has_referrer_policy']   = int(bool(rp))
    f['rp_no_referrer']        = int('no-referrer' in rp and 'when' not in rp)
    f['rp_same_origin']        = int('same-origin'    in rp)
    f['rp_strict_origin']      = int('strict-origin'  in rp)
    f['rp_unsafe_url']         = int('unsafe-url'     in rp)
    f['rp_no_restriction']     = int(not rp or rp == 'no-referrer-when-downgrade')

    f['has_permissions_policy']= int('permissions-policy' in h)
    cc = h.get('cache-control', '').lower()
    f['has_cache_control']     = int(bool(cc))
    f['cc_no_store']           = int('no-store' in cc)
    f['cc_no_cache']           = int('no-cache' in cc)
    f['cc_public_sensitive']   = int('public'   in cc)
    f['has_expect_ct']         = int('expect-ct' in h)

    f['has_cross_origin_embedder_policy'] = int('cross-origin-embedder-policy' in h)
    f['has_coep']              = f['has_cross_origin_embedder_policy']
    f['has_cross_origin_opener_policy']   = int('cross-origin-opener-policy'   in h)
    f['has_coop']              = f['has_cross_origin_opener_policy']
    f['has_cross_origin_resource_policy'] = int('cross-origin-resource-policy' in h)
    f['has_corp']              = f['has_cross_origin_resource_policy']

    acao = h.get('access-control-allow-origin', '')
    f['has_access_control_allow_origin']      = int(bool(acao))
    f['has_access_control_allow_credentials'] = int('access-control-allow-credentials' in h)
    f['cors_present']          = int(bool(acao))
    f['cors_wildcard']         = int(acao == '*')
    f['cors_cred_wildcard']    = int(acao == '*' and
        h.get('access-control-allow-credentials', '').lower() == 'true')

    sc = h.get('set-cookie', '').lower()
    f['cookie_present']        = int('set-cookie' in h)
    f['cookie_secure']         = int('secure'   in sc)
    f['cookie_httponly']       = int('httponly' in sc)
    f['cookie_samesite']       = int('samesite' in sc)

    f['info_leak_count']       = sum(1 for lh in LEAK_HEADERS if lh in h)
    srv = h.get('server', '')
    f['server_version_exposed']= int(bool(re.search(r'[0-9]+\.[0-9]+', srv)))
    f['x_powered_by_present']  = int('x-powered-by' in h)

    f['header_text'] = ' | '.join(f'{k}: {v}' for k, v in h.items() if not k.startswith('_'))
    return f

# ── Load Dataset ─────────────────────────────────────────────────────────────
DATASET_FILE = CONFIG['DATASET_FILE']
if not os.path.exists(DATASET_FILE):
    try:
        import generate_dataset
        print("Generating 5,000-record dataset using generate_dataset module...", flush=True)
        generate_dataset.main(DATASET_FILE)
    except Exception:
        if os.path.exists('generate_dataset.py'):
            import subprocess
            print("Running generate_dataset.py to create training dataset...", flush=True)
            subprocess.run([sys.executable, 'generate_dataset.py'], check=True)
        else:
            raise FileNotFoundError(
                f"'{DATASET_FILE}' not found! Please upload 'training_dataset.csv' or run 'generate_dataset.py'."
            )

df = pd.read_csv(DATASET_FILE)
print(f'✅ Loaded {DATASET_FILE}: {len(df):,} rows × {len(df.columns)} columns', flush=True)

# ── EDA Chart ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle('HTTP Header Vulnerability Dataset — EDA', fontsize=18, fontweight='bold')

counts = df['vuln_class'].value_counts().sort_index()
bars = axes[0, 0].bar([CLASS_NAMES[c] for c in counts.index], counts.values,
                      color=PALETTE, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{val:,}', ha='center', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Vulnerability Class Distribution', fontweight='bold')
axes[0, 0].set_ylabel('Count')

sh_map = {
    'HSTS': 'has_strict_transport_security',
    'CSP': 'has_content_security_policy',
    'X-Frame-Opt': 'has_x_frame_options',
    'X-Content-Type': 'has_x_content_type_options',
    'Referrer-Pol': 'has_referrer_policy',
    'Permissions': 'has_permissions_policy',
    'Cache-Ctrl': 'has_cache_control',
    'COEP': 'has_coep',
    'COOP': 'has_coop',
    'Expect-CT': 'has_expect_ct',
}
adoption = {k: df[v].mean()*100 for k, v in sh_map.items() if v in df.columns}
sorted_a = sorted(adoption.items(), key=lambda x: x[1])
labels, vals = zip(*sorted_a)
colors = ['#e74c3c' if v < 50 else '#f39c12' if v < 80 else '#2ecc71' for v in vals]
axes[0, 1].barh(labels, vals, color=colors, edgecolor='white')
axes[0, 1].set_title('Security Header Adoption (%)', fontweight='bold')
axes[0, 1].set_xlabel('% of Records')

vuln_types = {
    'No CSP': (df['csp_present']==0).sum(),
    'No HSTS': (df['has_strict_transport_security']==0).sum(),
    'No X-Frame-Opt': (df['has_x_frame_options']==0).sum(),
    'Server leak': (df['server_version_exposed']==1).sum(),
    'X-Powered-By': (df['x_powered_by_present']==1).sum(),
    'CORS wildcard': (df['cors_wildcard']==1).sum(),
    'No Cookie Sec': ((df['cookie_present']==1)&(df['cookie_secure']==0)).sum(),
    'CSP unsafe-inline': (df['csp_unsafe_inline']==1).sum(),
    'No Referrer-Pol': (df['has_referrer_policy']==0).sum(),
    'No Permissions': (df['has_permissions_policy']==0).sum(),
    'CORS+Cred': (df['cors_cred_wildcard']==1).sum(),
    'No HTTPS': (df['https']==0).sum(),
}
sorted_vt = sorted(vuln_types.items(), key=lambda x: x[1], reverse=True)
vt_labels, vt_vals = zip(*sorted_vt)
axes[0, 2].barh(list(reversed(vt_labels)), list(reversed(vt_vals)), color='#e74c3c', alpha=0.8, edgecolor='white')
axes[0, 2].set_title('Records by Vulnerability Type', fontweight='bold')
axes[0, 2].set_xlabel('Count')

csp_df = df[df['csp_present']==1]
csp_issues = {
    "'unsafe-inline'": csp_df['csp_unsafe_inline'].sum(),
    "'unsafe-eval'": csp_df['csp_unsafe_eval'].sum(),
    'Wildcard (*)': csp_df['csp_wildcard'].sum(),
    'Allows data:': csp_df['csp_allow_data'].sum(),
}
axes[1, 0].bar(list(csp_issues.keys()), list(csp_issues.values()),
               color=['#e74c3c','#e67e22','#f39c12','#f1c40f'], edgecolor='white')
axes[1, 0].set_title(f'CSP Misconfigs (among {len(csp_df):,} with CSP)', fontweight='bold')
axes[1, 0].set_ylabel('Count')

ck = df[df['cookie_present']==1]
ck_data = {
    'Secure\nflag': ck['cookie_secure'].mean()*100,
    'HttpOnly\nflag': ck['cookie_httponly'].mean()*100,
    'SameSite\nattr': ck['cookie_samesite'].mean()*100,
}
axes[1, 1].bar(list(ck_data.keys()), list(ck_data.values()), color=['#2ecc71','#27ae60','#16a085'], edgecolor='white')
axes[1, 1].set_title(f'Cookie Security Attributes\n(among {len(ck):,} with Set-Cookie)', fontweight='bold')
axes[1, 1].set_ylabel('% Correct')

https_c = df['https'].value_counts()
axes[1, 2].pie(https_c, labels=['HTTPS' if i==1 else 'HTTP' for i in https_c.index],
               colors=['#2ecc71' if i==1 else '#e74c3c' for i in https_c.index],
               autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
axes[1, 2].set_title('HTTPS vs HTTP Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ EDA chart saved → eda_analysis.png', flush=True)

# ── Preprocessing ────────────────────────────────────────────────────────────
EXCLUDE = {'url','vuln_class','vuln_label','is_vulnerable','header_text','_url','_status','_final_url','status'}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]

X     = df[FEATURE_COLS].fillna(0).values
y     = df['vuln_class'].values
y_bin = df['is_vulnerable'].values
texts = df['header_text'].fillna('').values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(CONFIG['MODEL_DIR'], 'scaler.pkl'))

tokenizer = Tokenizer(num_words=CONFIG['VOCAB_SIZE'], oov_token='<OOV>', lower=True)
tokenizer.fit_on_texts(texts)
seqs   = tokenizer.texts_to_sequences(texts)
X_text = pad_sequences(seqs, maxlen=CONFIG['MAX_SEQ_LEN'], padding='post', truncating='post')

with open(os.path.join(CONFIG['MODEL_DIR'], 'tokenizer.json'), 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())

(X_tr, X_te, X_txt_tr, X_txt_te,
 y_tr, y_te, yb_tr, yb_te) = train_test_split(
    X_scaled, X_text, y, y_bin,
    test_size=CONFIG['TEST_SIZE'], stratify=y, random_state=42
)

NUM_CLASSES = len(CLASS_NAMES)
y_tr_cat = keras.utils.to_categorical(y_tr, NUM_CLASSES)
y_te_cat = keras.utils.to_categorical(y_te, NUM_CLASSES)

cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
CLASS_WEIGHT = {int(c): float(w) for c, w in zip(np.unique(y_tr), cw)}

print(f'\nTrain: {len(X_tr):,} | Test: {len(X_te):,}', flush=True)
print(f'Numeric features: {X_tr.shape[1]} | Text tokens: {X_txt_tr.shape[1]}', flush=True)

# ── Model Evaluation Helper ──────────────────────────────────────────────────
RESULTS = {}

def evaluate(name, preds_or_model, X_test=None, y_test=None, proba=None):
    if hasattr(preds_or_model, 'predict') and X_test is not None:
        preds = preds_or_model.predict(X_test)
        if proba is None and hasattr(preds_or_model, 'predict_proba'):
            proba = preds_or_model.predict_proba(X_test)
    else:
        preds = preds_or_model

    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds, average='weighted', zero_division=0)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec  = recall_score(y_test, preds, average='weighted', zero_division=0)

    auc = 0.0
    if proba is not None:
        try:
            yb = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
            auc = roc_auc_score(yb, proba, multi_class='ovr', average='weighted')
        except Exception:
            pass

    RESULTS[name] = {'Accuracy': acc, 'F1-Score': f1, 'Precision': prec,
                     'Recall': rec, 'AUC-ROC': auc}
    tag = '🔥' if acc >= 0.95 else '✅' if acc >= 0.90 else '⚠️'
    print(f'  {tag} {name:<32s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}', flush=True)
    return preds

print('\n' + '='*72, flush=True)
print('  BASELINE ML MODELS', flush=True)
print('='*72, flush=True)

# 1. Random Forest
print('\n[1/4] Training Random Forest…', flush=True)
rf = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1,
                            random_state=42, class_weight='balanced')
rf.fit(X_tr, y_tr)
joblib.dump(rf, os.path.join(CONFIG['MODEL_DIR'], 'random_forest.pkl'))
evaluate('Random Forest', rf, X_te, y_te)

# 2. SVM
print('\n[2/4] Training SVM (RBF)…', flush=True)
svm = SVC(C=10, kernel='rbf', gamma='scale', probability=True,
          class_weight='balanced', random_state=42)
svm.fit(X_tr, y_tr)
joblib.dump(svm, os.path.join(CONFIG['MODEL_DIR'], 'svm.pkl'))
evaluate('SVM (RBF)', svm, X_te, y_te)

# 3. XGBoost
print('\n[3/4] Training XGBoost…', flush=True)
xgb = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric='mlogloss', n_jobs=-1, random_state=42)
xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
xgb.save_model(os.path.join(CONFIG['MODEL_DIR'], 'xgboost.json'))
evaluate('XGBoost', xgb, X_te, y_te)

# 4. Gradient Boosting
print('\n[4/4] Training Gradient Boosting…', flush=True)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=6,
                                learning_rate=0.05, random_state=42)
gb.fit(X_tr, y_tr)
joblib.dump(gb, os.path.join(CONFIG['MODEL_DIR'], 'gradient_boosting.pkl'))
evaluate('Gradient Boosting', gb, X_te, y_te)

# ── Deep Learning Model 1: CNN-LSTM ──────────────────────────────────────────
print('\n' + '='*72, flush=True)
print('  DEEP LEARNING ARCHITECTURES', flush=True)
print('='*72, flush=True)

def build_cnn_lstm():
    text_in = layers.Input(shape=(CONFIG['MAX_SEQ_LEN'],), name='text_input')
    emb     = layers.Embedding(CONFIG['VOCAB_SIZE'], CONFIG['EMBED_DIM'], mask_zero=False)(text_in)
    c1      = layers.Conv1D(128, 3, activation='relu', padding='same')(emb)
    c2      = layers.Conv1D(128, 5, activation='relu', padding='same')(emb)
    merged  = layers.Concatenate()([c1, c2])
    pool    = layers.MaxPooling1D(2)(merged)
    d1      = layers.Dropout(CONFIG['DROPOUT'])(pool)
    lstm    = layers.LSTM(CONFIG['LSTM_UNITS'])(d1)
    d2      = layers.Dropout(CONFIG['DROPOUT'])(lstm)

    num_in  = layers.Input(shape=(X_tr.shape[1],), name='num_input')
    n1      = layers.Dense(128, activation='relu')(num_in)
    n1      = layers.BatchNormalization()(n1)
    n1      = layers.Dropout(CONFIG['DROPOUT'])(n1)
    n2      = layers.Dense(64, activation='relu')(n1)

    fused   = layers.Concatenate()([d2, n2])
    h       = layers.Dense(128, activation='relu')(fused)
    h       = layers.BatchNormalization()(h)
    h       = layers.Dropout(CONFIG['DROPOUT'])(h)
    out     = layers.Dense(NUM_CLASSES, activation='softmax')(h)

    m = Model(inputs=[text_in, num_in], outputs=out, name='CNN_LSTM')
    m.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return m

print('\n[1/3] Training CNN-LSTM Model…', flush=True)
cnn_lstm_model = build_cnn_lstm()
cbs = [
    EarlyStopping(monitor='val_accuracy', patience=CONFIG['PATIENCE'],
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0),
    ModelCheckpoint(os.path.join(CONFIG['MODEL_DIR'], 'cnn_lstm_best.keras'),
                    monitor='val_accuracy', save_best_only=True, verbose=0),
]
history_cnn = cnn_lstm_model.fit(
    [X_txt_tr, X_tr], y_tr_cat,
    validation_data=([X_txt_te, X_te], y_te_cat),
    epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
    class_weight=CLASS_WEIGHT, callbacks=cbs, verbose=1
)
prob_cnn  = cnn_lstm_model.predict([X_txt_te, X_te], verbose=0)
preds_cnn = np.argmax(prob_cnn, axis=1)
evaluate('CNN-LSTM', preds_cnn, None, y_te, proba=prob_cnn)

# ── Deep Learning Model 2: BiLSTM + Multi-Head Self-Attention ────────────────
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kw):
        super().__init__(**kw)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads, dropout=dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        out = self.attn(x, x, training=training)
        return self.norm(x + self.drop(out, training=training))

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate,
        })
        return config

def build_bilstm_attention():
    text_in  = layers.Input(shape=(CONFIG['MAX_SEQ_LEN'],), name='text_input')
    emb      = layers.Embedding(CONFIG['VOCAB_SIZE'], CONFIG['EMBED_DIM'])(text_in)
    emb      = layers.Dropout(0.2)(emb)
    bi1      = layers.Bidirectional(layers.LSTM(CONFIG['LSTM_UNITS'], return_sequences=True))(emb)
    bi1      = layers.Dropout(CONFIG['DROPOUT'])(bi1)
    bi2      = layers.Bidirectional(layers.LSTM(CONFIG['LSTM_UNITS']//2, return_sequences=True))(bi1)
    attn     = MultiHeadSelfAttention(CONFIG['LSTM_UNITS'], CONFIG['ATTN_HEADS'],
                                      dropout=0.1, name='mhsa')(bi2)
    avg_p    = layers.GlobalAveragePooling1D()(attn)
    max_p    = layers.GlobalMaxPooling1D()(attn)
    txt_vec  = layers.Concatenate()([avg_p, max_p])
    txt_vec  = layers.Dense(256, activation='gelu')(txt_vec)
    txt_vec  = layers.Dropout(CONFIG['DROPOUT'])(txt_vec)

    num_in   = layers.Input(shape=(X_tr.shape[1],), name='num_input')
    n1       = layers.Dense(256, activation='gelu')(num_in)
    n1       = layers.BatchNormalization()(n1)
    n1       = layers.Dropout(CONFIG['DROPOUT'])(n1)
    n2       = layers.Dense(128, activation='gelu')(n1)
    n2       = layers.BatchNormalization()(n2)

    fused    = layers.Concatenate()([txt_vec, n2])
    gate     = layers.Dense(fused.shape[-1], activation='sigmoid', name='gate')(fused)
    fused_g  = layers.Multiply()([fused, gate])

    h        = layers.Dense(256, activation='gelu')(fused_g)
    h        = layers.BatchNormalization()(h)
    h        = layers.Dropout(CONFIG['DROPOUT'])(h)
    h        = layers.Dense(128, activation='gelu')(h)
    h        = layers.Dropout(CONFIG['DROPOUT']*0.7)(h)
    out      = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(h)

    m = Model(inputs=[text_in, num_in], outputs=out, name='BiLSTM_Attention')
    m.compile(optimizer=Adam(5e-4, clipnorm=1.0),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m

print('\n[2/3] Training BiLSTM + Self-Attention…', flush=True)
bilstm_model = build_bilstm_attention()
cbs_bi = [
    EarlyStopping(monitor='val_accuracy', patience=CONFIG['PATIENCE']+2,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0),
    ModelCheckpoint(os.path.join(CONFIG['MODEL_DIR'], 'bilstm_best.keras'),
                    monitor='val_accuracy', save_best_only=True, verbose=0),
]
history_bi = bilstm_model.fit(
    [X_txt_tr, X_tr], y_tr_cat,
    validation_data=([X_txt_te, X_te], y_te_cat),
    epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
    class_weight=CLASS_WEIGHT, callbacks=cbs_bi, verbose=1
)
prob_bi  = bilstm_model.predict([X_txt_te, X_te], verbose=0)
preds_bi = np.argmax(prob_bi, axis=1)
evaluate('BiLSTM + Self-Attention', preds_bi, None, y_te, proba=prob_bi)

# ── Deep Learning Model 3: Transformer Encoder ───────────────────────────────
class PositionalEncoding(layers.Layer):
    def __init__(self, maxlen, embed_dim, **kw):
        super().__init__(**kw)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        pos  = np.arange(maxlen)[:, np.newaxis]
        dims = np.arange(embed_dim)[np.newaxis, :]
        ang  = pos / np.power(10000, (2*(dims//2)) / embed_dim)
        ang[:, 0::2] = np.sin(ang[:, 0::2])
        ang[:, 1::2] = np.cos(ang[:, 1::2])
        self.pos_enc = tf.cast(ang[np.newaxis], tf.float32)

    def call(self, x):
        return x + self.pos_enc[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({'maxlen': self.maxlen, 'embed_dim': self.embed_dim})
        return config

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kw):
        super().__init__(**kw)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.attn  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads, dropout=dropout)
        self.ffn1  = layers.Dense(ff_dim, activation='gelu')
        self.ffn2  = layers.Dense(embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        x2 = self.norm1(x)
        x  = x + self.drop1(self.attn(x2, x2, training=training), training=training)
        x2 = self.norm2(x)
        x  = x + self.drop2(self.ffn2(self.ffn1(x2)), training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate,
        })
        return config

def build_transformer():
    D = CONFIG['TRANSFORMER_DIM']

    text_in  = layers.Input(shape=(CONFIG['MAX_SEQ_LEN'],), name='text_input')
    tok_emb  = layers.Embedding(CONFIG['VOCAB_SIZE'], D)(text_in)
    tok_emb  = layers.Dropout(0.1)(tok_emb)
    enc      = PositionalEncoding(CONFIG['MAX_SEQ_LEN'], D)(tok_emb)

    for i in range(4):
        enc = TransformerEncoderBlock(
            D, CONFIG['ATTN_HEADS'], CONFIG['FF_DIM'],
            dropout=CONFIG['DROPOUT'], name=f'enc_{i}'
        )(enc)

    avg_p    = layers.GlobalAveragePooling1D()(enc)
    max_p    = layers.GlobalMaxPooling1D()(enc)
    txt_vec  = layers.Concatenate()([avg_p, max_p])
    txt_vec  = layers.Dense(D*2, activation='gelu')(txt_vec)
    txt_vec  = layers.Dropout(CONFIG['DROPOUT'])(txt_vec)

    num_in   = layers.Input(shape=(X_tr.shape[1],), name='num_input')
    n1       = layers.Dense(256, activation='gelu')(num_in)
    n1       = layers.BatchNormalization()(n1)
    n1       = layers.Dropout(CONFIG['DROPOUT'])(n1)
    n2       = layers.Dense(256, activation='gelu')(n1)
    n2       = layers.BatchNormalization()(n2)
    n_out    = layers.Add()([n1, n2])
    n_out    = layers.Dense(D*2, activation='gelu')(n_out)

    txt_seq  = layers.Reshape((1, D*2))(txt_vec)
    num_seq  = layers.Reshape((1, D*2))(n_out)
    cross    = layers.MultiHeadAttention(
        num_heads=CONFIG['ATTN_HEADS'], key_dim=D//2, name='cross_attn'
    )(txt_seq, num_seq)
    cross    = layers.Flatten()(cross)

    fused    = layers.Concatenate()([txt_vec, n_out, cross])

    h        = layers.Dense(512, activation='gelu')(fused)
    h        = layers.BatchNormalization()(h)
    h        = layers.Dropout(CONFIG['DROPOUT'])(h)
    h        = layers.Dense(256, activation='gelu')(h)
    h        = layers.Dropout(CONFIG['DROPOUT']*0.7)(h)
    out      = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(h)

    m = Model(inputs=[text_in, num_in], outputs=out, name='Transformer_Encoder')
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=5e-4, first_decay_steps=500, t_mul=2.0
    )
    m.compile(optimizer=Adam(lr_sched, clipnorm=1.0),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m

print('\n[3/3] Training Transformer Encoder (BERT-style 4-block fusion)…', flush=True)
transformer_model = build_transformer()
cbs_tr = [
    EarlyStopping(monitor='val_accuracy', patience=CONFIG['PATIENCE']+3,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(CONFIG['MODEL_DIR'], 'transformer_best.keras'),
                    monitor='val_accuracy', save_best_only=True, verbose=0),
]
history_tr = transformer_model.fit(
    [X_txt_tr, X_tr], y_tr_cat,
    validation_data=([X_txt_te, X_te], y_te_cat),
    epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
    class_weight=CLASS_WEIGHT, callbacks=cbs_tr, verbose=1
)
prob_tr  = transformer_model.predict([X_txt_te, X_te], verbose=0)
preds_tr = np.argmax(prob_tr, axis=1)
evaluate('Transformer (BERT-style)', preds_tr, None, y_te, proba=prob_tr)

# ── Summary & Charts ─────────────────────────────────────────────────────────
res_df = pd.DataFrame(RESULTS).T.round(4).sort_values('Accuracy', ascending=False)
print('\n' + '='*75, flush=True)
print('  PERFORMANCE COMPARISON — ALL MODELS', flush=True)
print('='*75, flush=True)
print(tabulate(res_df, headers='keys', tablefmt='grid', floatfmt='.4f'), flush=True)

winner = res_df.index[0]
second = res_df.index[1] if len(res_df) > 1 else winner
improvement = (res_df.loc[winner,'Accuracy'] - res_df.loc[second,'Accuracy'])*100
print(f'\n🏆 Best: {winner} (Acc={res_df.loc[winner,"Accuracy"]:.4f}, F1={res_df.loc[winner,"F1-Score"]:.4f}) +{improvement:.2f}% over {second}', flush=True)

# Model comparison plot
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC-ROC']
model_names = res_df.index.tolist()
x, w = np.arange(len(model_names)), 0.15
pal6 = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6']

fig, ax = plt.subplots(figsize=(18, 7))
for i, m in enumerate(metrics):
    vals = res_df[m].values
    bars = ax.bar(x + i*w - w*2, vals, w, label=m, color=pal6[i], alpha=0.87, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{v:.3f}', ha='center', fontsize=7, rotation=50)

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=22, ha='right', fontsize=10)
ax.set_ylim(0.4, 1.05)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('HTTP Header Vulnerability Detection — Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Confusion Matrices — Advanced DL Models', fontsize=14, fontweight='bold')
for ax, (preds, title) in zip(axes, [(preds_bi, 'BiLSTM + Self-Attention'), (preds_tr, 'Transformer (BERT-style)')]):
    cm = confusion_matrix(y_te, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(CLASS_NAMES.values()),
                yticklabels=list(CLASS_NAMES.values()), ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# Training curves
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Training Curves — Deep Learning Models', fontsize=15, fontweight='bold')
histories = [
    (history_cnn, 'CNN-LSTM', '#3498db'),
    (history_bi, 'BiLSTM + Self-Attention', '#2ecc71'),
    (history_tr, 'Transformer', '#e74c3c'),
]
for col, (hist, name, color) in enumerate(histories):
    axes[0, col].plot(hist.history['accuracy'], label='Train', color=color)
    axes[0, col].plot(hist.history['val_accuracy'], label='Val', color=color, linestyle='--')
    axes[0, col].set_title(f'{name} — Accuracy')
    axes[0, col].legend()
    axes[0, col].grid(alpha=0.3)
    axes[0, col].set_ylim(0, 1.05)

    axes[1, col].plot(hist.history['loss'], label='Train', color=color)
    axes[1, col].plot(hist.history['val_loss'], label='Val', color=color, linestyle='--')
    axes[1, col].set_title(f'{name} — Loss')
    axes[1, col].legend()
    axes[1, col].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(importances.index, importances.values,
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances))), edgecolor='white')
ax.set_title('Top-20 Feature Importances (Random Forest)', fontweight='bold', fontsize=13)
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Classification report
for preds, name in [(preds_bi, 'BiLSTM + Self-Attention'), (preds_tr, 'Transformer (BERT-style)')]:
    print(f'\n📋 {name}', flush=True)
    print('─'*60, flush=True)
    print(classification_report(y_te, preds, target_names=list(CLASS_NAMES.values()), zero_division=0), flush=True)

# ── Live URL Analyser ────────────────────────────────────────────────────────
best_model = transformer_model
best_name = 'Transformer (BERT-style)'

VULN_RULES = [
    ('https', lambda v: v==0, 'CRITICAL', 'No HTTPS', 'All traffic is unencrypted and interceptable.', 'Enable HTTPS immediately. Get a free cert from Let\'s Encrypt.'),
    ('has_strict_transport_security', lambda v: v==0, 'HIGH', 'Missing HSTS', 'Site is vulnerable to SSL stripping and MITM attacks.', 'Add: Strict-Transport-Security: max-age=63072000; includeSubDomains; preload'),
    ('hsts_max_age', lambda v: 0<v<31536000, 'MEDIUM', 'HSTS max-age Too Short', 'HSTS max-age under 1 year gives a short enforcement window.', 'Set max-age ≥ 31536000 (1 year), ideally 63072000 (2 years).'),
    ('csp_present', lambda v: v==0, 'HIGH', 'Missing Content-Security-Policy', 'No CSP — site is unprotected against XSS and injection attacks.', "Add CSP. Start with: Content-Security-Policy: default-src 'self'"),
    ('csp_unsafe_inline', lambda v: v==1, 'HIGH', "CSP 'unsafe-inline'", "'unsafe-inline' negates XSS protection entirely.", "Remove 'unsafe-inline'. Use nonces or hashes for inline scripts."),
    ('csp_unsafe_eval', lambda v: v==1, 'HIGH', "CSP 'unsafe-eval'", "'unsafe-eval' allows eval() and similar — major XSS surface.", "Remove 'unsafe-eval'. Refactor away from eval/Function/setTimeout(string)."),
    ('csp_wildcard', lambda v: v==1, 'MEDIUM', 'CSP Wildcard (*)', 'Wildcard allows scripts from any origin.', 'Replace * with specific trusted domains.'),
    ('has_x_frame_options', lambda v: v==0, 'MEDIUM', 'Missing X-Frame-Options', 'Site can be embedded in iframes — clickjacking risk.', 'Add: X-Frame-Options: DENY'),
    ('xfo_allowfrom', lambda v: v==1, 'LOW', 'X-Frame-Options ALLOW-FROM (Deprecated)', 'ALLOW-FROM is deprecated and ignored by modern browsers.', 'Use DENY or SAMEORIGIN; control framing via CSP frame-ancestors instead.'),
    ('xcto_nosniff', lambda v: v==0, 'MEDIUM', 'Missing X-Content-Type-Options', 'Browsers may MIME-sniff responses, executing malicious content.', 'Add: X-Content-Type-Options: nosniff'),
    ('has_referrer_policy', lambda v: v==0, 'LOW', 'Missing Referrer-Policy', 'Full URL is leaked in Referer header to third parties.', 'Add: Referrer-Policy: strict-origin-when-cross-origin'),
    ('rp_unsafe_url', lambda v: v==1, 'MEDIUM', 'Referrer-Policy: unsafe-url', 'Full URL including query strings sent in cross-origin requests.', 'Change to: Referrer-Policy: strict-origin-when-cross-origin'),
    ('has_permissions_policy', lambda v: v==0, 'LOW', 'Missing Permissions-Policy', 'Browser features (camera, mic, geolocation) are unrestricted.', 'Add: Permissions-Policy: camera=(), microphone=(), geolocation=()'),
    ('cors_cred_wildcard', lambda v: v==1, 'CRITICAL', 'CORS Wildcard + Credentials', 'Any origin can make credentialed requests — critical misconfiguration.', 'Never combine Access-Control-Allow-Origin: * with credentials: true.'),
    ('cors_wildcard', lambda v: v==1, 'MEDIUM', 'CORS Wildcard Origin', 'Any website can read API responses.', 'Restrict to specific trusted origins.'),
    ('cookie_secure', lambda v: v==0, 'HIGH', 'Cookie Missing Secure Flag', 'Session cookie can be sent over HTTP — interception risk.', 'Add Secure attribute: Set-Cookie: name=value; Secure; HttpOnly; SameSite=Strict'),
    ('cookie_httponly', lambda v: v==0, 'HIGH', 'Cookie Missing HttpOnly Flag', 'Cookie is accessible via JavaScript — XSS theft risk.', 'Add HttpOnly attribute to all session cookies.'),
    ('cookie_samesite', lambda v: v==0, 'MEDIUM', 'Cookie Missing SameSite', 'Cookie sent in cross-site requests — CSRF risk.', 'Add SameSite=Strict or SameSite=Lax to all cookies.'),
    ('server_version_exposed', lambda v: v==1, 'MEDIUM', 'Server Version Disclosed', 'Server header reveals exact version — aids CVE targeting.', 'Configure server to suppress version from the Server header.'),
    ('x_powered_by_present', lambda v: v==1, 'LOW', 'X-Powered-By Header', 'Backend technology stack is disclosed.', 'Remove X-Powered-By from server/framework configuration.'),
    ('has_coep', lambda v: v==0, 'LOW', 'Missing COEP', 'Without Cross-Origin-Embedder-Policy, site cannot use cross-origin isolation.', 'Add: Cross-Origin-Embedder-Policy: require-corp'),
    ('has_coop', lambda v: v==0, 'LOW', 'Missing COOP', 'Without Cross-Origin-Opener-Policy, browsing context may be shared.', 'Add: Cross-Origin-Opener-Policy: same-origin'),
    ('cc_public_sensitive', lambda v: v==1, 'MEDIUM', 'Cache-Control: public on Sensitive Page', 'Proxies may cache responses with sensitive data.', 'Use Cache-Control: no-store for authenticated/private pages.'),
]

SEV_ORDER = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}
SEV_EMOJI = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🔵', 'OK': '🟢'}

def fetch_one(url, timeout=CONFIG['FETCH_TIMEOUT']):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True, verify=False,
                          headers={'User-Agent': 'Mozilla/5.0 SecurityResearchBot/1.0'})
        hdrs = {k.lower(): v for k, v in r.headers.items()}
        hdrs.update({'_url': url, '_status': r.status_code})
        return hdrs
    except Exception:
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True, verify=False,
                             stream=True,
                             headers={'User-Agent': 'Mozilla/5.0 SecurityResearchBot/1.0'})
            r.close()
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            hdrs.update({'_url': url, '_status': r.status_code})
            return hdrs
        except Exception:
            return None

def analyze_url(url: str) -> dict:
    print('\n' + '━'*70, flush=True)
    print('  🔍  HTTP HEADER VULNERABILITY ANALYSER', flush=True)
    print('━'*70, flush=True)
    print(f'  URL   : {url}', flush=True)
    print(f'  Model : {best_name}', flush=True)

    print('\n  📡 Fetching headers…', end=' ', flush=True)
    import urllib3; urllib3.disable_warnings()
    raw = fetch_one(url)
    if raw is None:
        print('FAILED — could not reach URL', flush=True)
        return {'url': url, 'error': 'Could not reach URL'}
    print(f'OK (HTTP {raw.get("_status","?")})', flush=True)

    feat = extract_features(raw)
    feat_df = pd.DataFrame([feat])
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0
    X_num = scaler.transform(feat_df[FEATURE_COLS].fillna(0).values)

    seq = tokenizer.texts_to_sequences([feat.get('header_text', '')])
    X_txt = pad_sequences(seq, maxlen=CONFIG['MAX_SEQ_LEN'], padding='post', truncating='post')
    proba = best_model.predict([X_txt, X_num], verbose=0)[0]
    pred_class = int(np.argmax(proba))
    pred_label = CLASS_NAMES[pred_class]
    confidence = float(proba[pred_class])
    class_probs = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}

    cookie_ok = feat.get('cookie_present', 0)
    issues = []
    for key, cond, sev, title, desc, rec in VULN_RULES:
        if key in ('cookie_secure','cookie_httponly','cookie_samesite') and not cookie_ok:
            continue
        if cond(feat.get(key, 0)):
            issues.append({'severity': sev, 'title': title, 'description': desc, 'recommendation': rec})
    issues.sort(key=lambda x: SEV_ORDER[x['severity']])
    rule_score = vulnerability_score(feat)

    print('\n  📦 Headers Received:', flush=True)
    for k, v in raw.items():
        if k.startswith('_'): continue
        mark = '✅' if k in SECURITY_HEADERS else '  '
        print(f'    {mark}  {k}: {str(v)[:90]}', flush=True)

    print('\n  🛡️  Security Header Checklist:', flush=True)
    for sh in SECURITY_HEADERS:
        sym = '✅' if sh in raw else '❌'
        print(f'    {sym}  {sh}', flush=True)

    print(f'\n  🤖 AI Model — {best_name}:', flush=True)
    cls_emoji = SEV_EMOJI.get({'Secure':'OK','Low Risk':'LOW','Medium Risk':'MEDIUM','High Risk':'HIGH'}.get(pred_label,'LOW'), '🔵')
    print(f'    Prediction  : {cls_emoji} {pred_label}  (confidence {confidence:.1%})', flush=True)
    print('    Class probabilities:', flush=True)
    for cn, p in class_probs.items():
        bar = '█' * int(p * 30)
        print(f'      {cn:<14s} {p:5.1%}  {bar}', flush=True)

    print(f'\n  ⚠️  Vulnerabilities Detected ({len(issues)}):', flush=True)
    if not issues:
        print('    🎉 No vulnerabilities found — site is well-configured!', flush=True)
    else:
        for i, iss in enumerate(issues, 1):
            e = SEV_EMOJI.get(iss['severity'], '🔵')
            print(f'\n  [{i}] {e} [{iss["severity"]}] {iss["title"]}', flush=True)
            print(f'       📌 {iss["description"]}', flush=True)
            print(f'       ✔  Fix: {iss["recommendation"]}', flush=True)

    sev_c = Counter(i['severity'] for i in issues)
    print(f'\n  📊 Overall Risk:', flush=True)
    print(f'     Rule-based : {CLASS_NAMES[rule_score]} (score {rule_score}/3)', flush=True)
    print(f'     AI model   : {pred_label}', flush=True)
    print(f'     Issues     : 🔴 Critical={sev_c["CRITICAL"]}  🟠 High={sev_c["HIGH"]}  🟡 Medium={sev_c["MEDIUM"]}  🔵 Low={sev_c["LOW"]}', flush=True)
    print('\n' + '━'*70 + '\n', flush=True)

    return {
        'url': url, 'status': raw.get('_status'),
        'features': feat, 'vulnerabilities': issues,
        'prediction': {'class': pred_class, 'label': pred_label,
                       'confidence': confidence, 'probs': class_probs},
        'rule_score': rule_score, 'rule_label': CLASS_NAMES[rule_score],
    }

# Test sample URLs
TEST_URLS = [
    'https://github.com',
    'https://owasp.org',
    'http://example.com',
]
scan_results = []
for u in TEST_URLS:
    res = analyze_url(u)
    scan_results.append(res)

summary = []
for r in scan_results:
    if 'error' in r: continue
    sev_c = Counter(i['severity'] for i in r.get('vulnerabilities', []))
    pred  = r.get('prediction', {})
    summary.append({
        'URL': r['url'][:45],
        'Status': r.get('status', '?'),
        'AI Prediction': pred.get('label', 'N/A'),
        'Confidence': f"{pred.get('confidence', 0):.0%}",
        'Rule Label': r.get('rule_label', 'N/A'),
        '🔴': sev_c['CRITICAL'], '🟠': sev_c['HIGH'],
        '🟡': sev_c['MEDIUM'],   '🔵': sev_c['LOW'],
    })

print('\n📊 SCAN SUMMARY', flush=True)
print(tabulate(summary, headers='keys', tablefmt='grid'), flush=True)

print('\n' + '█'*72, flush=True)
print('  FINAL PERFORMANCE SUMMARY', flush=True)
print('█'*72, flush=True)
print(tabulate(res_df, headers='keys', tablefmt='fancy_grid', floatfmt='.4f'), flush=True)

print('\n📁 Saved files:', flush=True)
for root, dirs, files in os.walk(CONFIG['MODEL_DIR']):
    for fn in files:
        fp = os.path.join(root, fn)
        print(f'   {fp}  ({os.path.getsize(fp)//1024:,} KB)', flush=True)
for fn in ['training_dataset.csv', 'eda_analysis.png', 'model_comparison.png',
           'confusion_matrices.png', 'training_curves.png', 'feature_importance.png']:
    if os.path.exists(fn):
        print(f'   {fn}  ({os.path.getsize(fn)//1024:,} KB)', flush=True)

print('\n✅ All pipeline steps completed successfully!', flush=True)
