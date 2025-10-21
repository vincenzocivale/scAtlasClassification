import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, EvalPrediction
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import wandb
import os

# --- 1. CONFIGURAZIONE ---
# Modifica queste variabili per il tuo progetto
config = {
    "dataset_repo_id": "Yuto2007/lung-tumor-embeddings", # OBBLIGATORIO: Sostituisci con il tuo repo
    "model_name": "MLP-CellType-Classifier",
    "learning_rate": 2e-5,
    "batch_size": 512,
    "num_epochs": 50,
    "weight_decay": 0.01,
    "early_stopping_patience": 5,
    "input_dim": 3072, # Dimensione degli embedding (3072 per scFoundation)
    "hidden_dims": [ 1024, 512], # Dimensione dello strato nascosto del MLP
    "dropout_rate": 0.2
}

hidden_dims_str = "_".join(map(str, config["hidden_dims"]))  # "1024_512"
config["model_name"] = f"MLP-CellType-Classifier_hiddendims_{hidden_dims_str}"

# Imposta il nome del progetto W&B
os.environ["WANDB_PROJECT"] = "scTumorClassification"

# --- 2. CARICAMENTO E PREPARAZIONE DEL DATASET ---
print(f"Caricamento dataset da: {config['dataset_repo_id']}")
dataset = load_dataset(config['dataset_repo_id'])

# Rinomina la colonna della label in 'labels' (richiesto dal Trainer)
dataset = dataset.rename_column("cell_type", "labels")

# Estrai le informazioni sulle classi
features = dataset['train'].features['labels']
id2label = {i: label for i, label in enumerate(features.names)}
label2id = {label: i for i, label in enumerate(features.names)}
config["num_labels"] = len(id2label)

print(f"Trovate {config['num_labels']} classi di tipi di cellula.")

# --- 3. DEFINIZIONE DEL MODELLO MLP ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_labels, dropout_rate, class_weights=None):
        """
        hidden_dims: lista delle dimensioni dei layer nascosti, es. [2048, 1024, 512]
        """
        super().__init__()

        layers = []
        last_dim = input_dim

        # Layer nascosti con BatchNorm
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))   # ✅ BatchNorm aggiunto
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = h_dim

        # Layer finale di classificazione
        layers.append(nn.Linear(last_dim, num_labels))

        self.layers = nn.Sequential(*layers)

        # Registra i pesi per classi sbilanciate
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, embedding, labels=None):
        logits = self.layers(embedding)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.layers[-1].out_features), labels.view(-1))
        return {"loss": loss, "logits": logits}


# --- Bilanciamento classi ---
print("\nCalcolo dei pesi per classi sbilanciate...")

# Conta le occorrenze per classe nel training set
label_counts = np.bincount(dataset["train"]['labels'])
class_freq = label_counts / label_counts.sum()

# Peso inversamente proporzionale alla frequenza
weights = 1.0 / np.maximum(label_counts, 1)
weights = weights / weights.sum() * len(weights)  # normalizzazione opzionale

# Converti in tensore
class_weights = torch.tensor(weights, dtype=torch.float32)
print("Pesi di classe calcolati:", class_weights)


# --- 4. FUNZIONE PER LE METRICHE ---
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": accuracy, "f1_weighted": f1}

# --- 5. ADDESTRAMENTO ---
# Inizializza il run di W&B
wandb.init(config=config, name=config["model_name"])

# Definisci gli argomenti per l'addestramento
training_args = TrainingArguments(
    output_dir=f"./results/{config['model_name']}",
    run_name=config["model_name"],
    num_train_epochs=config["num_epochs"],
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    weight_decay=config["weight_decay"],
    
    # Valutazione e salvataggio
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True, # Fondamentale per EarlyStopping
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    
    # Logging
    logging_dir=f'./logs/{config["model_name"]}',
    logging_strategy="epoch",
    report_to="wandb",
    
    # Performance
    fp16=torch.cuda.is_available(), # Usa mixed precision se hai una GPU
)

# Inizializza il modello
model = MLPClassifier(
    input_dim=config["input_dim"],
    hidden_dims=config["hidden_dims"],
    num_labels=config["num_labels"],
    dropout_rate=config["dropout_rate"],
    class_weights=class_weights
)


# Inizializza il Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])]
)

# Avvia l'addestramento
print("🚀 Inizio dell'addestramento...")
trainer.train()

# --- 6. VALUTAZIONE FINALE E MATRICE DI CONFUSIONE ---
print("📊 Valutazione sul test set...")
test_results = trainer.predict(dataset["test"])

# Estrai le predizioni e le label vere
y_pred = np.argmax(test_results.predictions, axis=1)
y_true = test_results.label_ids
class_names = features.names

# Logga la matrice di confusione su W&B con le label testuali
print("Logging della matrice di confusione su W&B...")
wandb.log({
    "conf_mat": wandb.plot.confusion_matrix(
        preds=y_pred,
        y_true=y_true,
        class_names=class_names
    )
})

# Logga le metriche finali del test set
test_metrics = compute_metrics(test_results)
print(f"Metriche sul test set: {test_metrics}")
wandb.log({"test_accuracy": test_metrics["accuracy"], "test_f1_weighted": test_metrics["f1_weighted"]})


# Fine del run
wandb.finish()
print("✅ Addestramento e valutazione completati.")