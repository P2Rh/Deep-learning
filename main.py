# ===============================================================
# PARTIE 2 : ARCHITECTURE DEVA STRICTE & ENTRAÎNEMENT
# ===============================================================

import warnings
import numpy as np
import torch
import torch.nn as nn
import librosa
import math
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr

# Suppress warnings
warnings.filterwarnings("ignore")

# Vérification des données de la Partie 1
if 'texts' not in locals():
    raise ValueError("Veuillez d'abord lancer la Partie 1 pour charger 'texts', 'audios', 'images'.")

print(f"✅ Données récupérées : {len(texts)} échantillons.")

# ===============================================================
# A. EXTRACTION DES FEATURES (AED & VED - ÉTAPES 2 & 3)
# ===============================================================

# 1. AUDIO (AED)
def extract_audio_features(y, sr=16000):
    """ Extrait Pitch, Loudness, Jitter, Shimmer """
    if isinstance(y, torch.Tensor): y = y.cpu().numpy()
    if len(y) < 2048: return {"pitch": 0, "loudness": 0, "jitter": 0, "shimmer": 0} # Sécurité

    rms = librosa.feature.rms(y=y)[0]
    loudness = float(np.mean(rms))

    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr, frame_length=2048)
        f0 = f0[~np.isnan(f0)]
        pitch = float(np.mean(f0)) if len(f0) > 0 else 0.0
    except: pitch = 0.0

    jitter = float(np.mean(np.abs(np.diff(f0)) / (np.abs(f0[:-1]) + 1e-6))) if len(f0) > 1 else 0.0
    shimmer = float(np.mean(np.abs(np.diff(rms)) / (np.abs(rms[:-1]) + 1e-6))) if len(rms) > 1 else 0.0

    return {"pitch": pitch, "loudness": loudness, "jitter": jitter, "shimmer": shimmer}

def audio_description(feats):
    """ Convertit les features en phrase descriptive """
    # Seuils simplifiés pour la démo
    p = "high" if feats["pitch"] > 120 else "low"
    l = "loud" if feats["loudness"] > 0.05 else "quiet"
    return f"The speaker used {p} pitch and {l} loudness with {feats['jitter']:.2f} jitter."

# 2. VISION (VED)
def visual_description(img):
    """ Simule des Action Units (AUs) basées sur l'intensité de l'image """
    intensity = img.mean()
    if intensity < 50: aus = "frown, lowered brow"
    elif intensity > 150: aus = "smile, raised cheeks"
    else: aus = "neutral expression"
    return f"The person shows signs of: {aus}."

# --- GÉNÉRATION DES DESCRIPTIONS ---
print("Génération des descriptions AED/VED...")
aed_sentences = [audio_description(extract_audio_features(a)) for a in audios]
ved_sentences = [visual_description(img) for img in images]

# --- PRÉPARATION DES TENSEURS NUMÉRIQUES (MFCC & CNN) ---
# Audio (MFCC)
def get_mfcc(y, sr=16000):
    if len(y) < 2048: y = np.pad(y, (0, 2048-len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return torch.tensor(mfcc.mean(axis=1)).float().to(device)

audio_feats_tensor = torch.stack([get_mfcc(a) for a in audios])

# Vision (CNN Simple)
class SimpleVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(4), # 64->16
            nn.Conv2d(16, 8, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x): return self.net(x).flatten(1)

# Préparation images
imgs_np = np.array([cv2.resize(img, (64, 64)) for img in images])
imgs_tensor = torch.tensor(imgs_np).float().permute(0, 3, 1, 2).to(device) / 255.0
vis_enc = SimpleVisualEncoder().to(device)
with torch.no_grad(): visual_feats_tensor = vis_enc(imgs_tensor)

# ===============================================================
# B. ARCHITECTURE DEVA (ÉTAPE 1, 4, 5)
# ===============================================================

D_MODEL = 64
T_TOKENS = 8  # Séquence cible fixe

# --- ÉTAPE 1 : TEXT ENCODER AMÉLIORÉ ---
class TextEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, T=T_TOKENS):
        super().__init__()
        self.d_model = d_model
        self.T = T

        # BERT figé
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters(): p.requires_grad = False

        self.proj = nn.Linear(768, d_model)

        # Token Em (Learnable)
        self.Em = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer Encoder Layer
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, texts):
        # 1. BERT
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
        with torch.no_grad():
            bert_out = self.bert(**inputs).last_hidden_state # [B, Seq, 768]

        # 2. Projection
        x = self.proj(bert_out) # [B, Seq, D]

        # 3. Ajout Token Em en tête
        B = x.shape[0]
        Em_batch = self.Em.expand(B, 1, -1)
        x = torch.cat([Em_batch, x], dim=1) # [B, Seq+1, D]

        # 4. Transformer Encoder
        x = self.transformer(x)

        # 5. Sélection des T premiers tokens (Xt)
        if x.shape[1] < self.T:
            # Padding si trop court
            pad = torch.zeros(B, self.T - x.shape[1], self.d_model).to(device)
            x = torch.cat([x, pad], dim=1)
        return x[:, :self.T, :] # [B, T, D]

# --- ÉTAPE 4 : MFU & ATTENTION CROISÉE ---

class AudioVisualFeatureProjector(nn.Module):
    """ Transforme [B, Dim] en [B, T, D_model] """
    def __init__(self, in_dim, d_model=D_MODEL, T=T_TOKENS):
        super().__init__()
        self.T = T
        self.d_model = d_model
        self.fc = nn.Linear(in_dim, T * d_model)

    def forward(self, x):
        B = x.shape[0]
        return self.fc(x).view(B, self.T, self.d_model)

class CrossModalAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key_value):
        # query (Xt) guides interaction with key_value (Xa ou Xv)
        Q = self.W_q(query)      # [B, T, D]
        K = self.W_k(key_value)  # [B, T, D]
        V = self.W_v(key_value)  # [B, T, D]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn = self.softmax(scores)
        return torch.matmul(attn, V)

class MFU(nn.Module):
    """ Minor Fusion Unit : Fusion résiduelle guidée par le texte """
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.att_t_a = CrossModalAttention(d_model) # T -> A
        self.att_t_v = CrossModalAttention(d_model) # T -> V

        # Paramètres scalaires learnables alpha et beta
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, prev_fusion, Xt, Xa, Xv):
        # Attention Text-Guided : Le texte (Query) va chercher l'info dans Audio/Vision (Key/Value)
        # Note: L'article peut varier sur qui est Q et K, ici on suit "Le texte guide la fusion" => Texte est Q
        att_a = self.att_t_a(query=Xt, key_value=Xa)
        att_v = self.att_t_v(query=Xt, key_value=Xv)

        # Formule résiduelle
        output = prev_fusion + (self.alpha * att_a) + (self.beta * att_v)
        return self.norm(output)

# --- ÉTAPE 5 : DEVANET FINAL ---

class DEVANet(nn.Module):
    def __init__(self, dim_audio, dim_visual, d_model=D_MODEL):
        super().__init__()
        # Encoders
        self.text_encoder = TextEncoder(d_model)
        self.audio_proj = AudioVisualFeatureProjector(dim_audio, d_model)
        self.visual_proj = AudioVisualFeatureProjector(dim_visual, d_model)

        # MFU
        self.mfu = MFU(d_model)

        # Feature Enhancement (Fusion initiale simple pour prev_fusion)
        self.fc_init = nn.Linear(d_model * 3, d_model) # Concat(Xt, Da, Dv) -> Init

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Score de sentiment
        )

    def forward(self, texts, aud_vec, vis_vec, aed, ved):
        # 1. Encodage Séquentiel [B, T, D]
        Xt = self.text_encoder(texts)
        Da = self.text_encoder(aed)
        Dv = self.text_encoder(ved)

        Xa = self.audio_proj(aud_vec)
        Xv = self.visual_proj(vis_vec)

        # 2. Initialisation de la fusion (basée sur le texte enrichi)
        # On utilise Xt + descriptions comme base
        init_feat = torch.cat([Xt, Da, Dv], dim=-1) # [B, T, 3D]
        prev_fusion = self.fc_init(init_feat)       # [B, T, D]

        # 3. MFU (Fusion guidée)
        refined = self.mfu(prev_fusion, Xt, Xa, Xv) # [B, T, D]

        # 4. Pooling Temporel (Moyenne sur T)
        final_repr = refined.mean(dim=1) # [B, D]

        # 5. Prédiction
        return self.classifier(final_repr).squeeze(-1)

# ===============================================================
# C. ENTRAÎNEMENT & MÉTRIQUES (ÉTAPE 5.2)
# ===============================================================

def evaluate_metrics(y_true, y_pred):
    """ Calcul des 4 métriques académiques """
    # Binarisation pour Acc-2 / F1 (Seuil 0 pour MOSI)
    y_true_bin = (y_true >= 0).astype(int)
    y_pred_bin = (y_pred >= 0).astype(int)

    acc2 = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, average='weighted')
    mae = mean_absolute_error(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0,0)

    return {"Acc-2": acc2, "F1": f1, "MAE": mae, "Corr": corr}

# --- SETUP ---
dim_a = audio_feats_tensor.shape[1]
dim_v = visual_feats_tensor.shape[1]

model = DEVANet(dim_a, dim_v).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

EPOCHS = 15
BATCH_SIZE = 16

# Normalisation des labels (-3, 3) -> (-1, 1) pour l'entrainement
labels_norm = labels / 3.0

print(f"\n🚀 Démarrage de l'entraînement DEVA sur {len(texts)} exemples...")

# --- BOUCLE D'ENTRAÎNEMENT ---
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    # Batch loop
    for i in range(0, len(texts), BATCH_SIZE):
        # Slicing
        b_txt = texts[i:i+BATCH_SIZE]
        b_aud = audio_feats_tensor[i:i+BATCH_SIZE]
        b_vis = visual_feats_tensor[i:i+BATCH_SIZE]
        b_aed = aed_sentences[i:i+BATCH_SIZE]
        b_ved = ved_sentences[i:i+BATCH_SIZE]
        b_lbl = labels_norm[i:i+BATCH_SIZE]

        if len(b_txt) == 0: continue

        # Forward
        preds = model(b_txt, b_aud, b_vis, b_aed, b_ved)
        loss = loss_fn(preds, b_lbl)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Store for metrics (denormalized)
        all_preds.extend((preds * 3.0).detach().cpu().numpy())
        all_labels.extend((b_lbl * 3.0).detach().cpu().numpy())

    # Calcul métriques fin d'époque
    metrics = evaluate_metrics(np.array(all_labels), np.array(all_preds))
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc-2: {metrics['Acc-2']:.2f} | MAE: {metrics['MAE']:.2f}")

# --- RÉSULTAT FINAL ---
print("\n📊 RÉSULTATS FINAUX (Standards Académiques) :")
print(metrics)

# Test unitaire sur un exemple
idx = 0
model.eval()
with torch.no_grad():
    p = model([texts[idx]], audio_feats_tensor[idx].unsqueeze(0), visual_feats_tensor[idx].unsqueeze(0), [aed_sentences[idx]], [ved_sentences[idx]])
    print(f"\nTest Exemple : '{texts[idx]}'")
    print(f"Label Réel : {labels[idx]:.2f}")
    print(f"Prédiction DEVA : {p.item() * 3.0:.2f}")