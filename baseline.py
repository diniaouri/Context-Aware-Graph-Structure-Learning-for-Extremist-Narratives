from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from preprocessing import GraphDataset

basic_data = GraphDataset().data
basic_data = basic_data[["Text", "Out-group"]]
basic_data = basic_data.rename(columns={"Text": "text", "Out-group": "label"})
# Convert to HF dataset
dataset = Dataset.from_pandas(basic_data)

# Create train-test split
splits = dataset.train_test_split(test_size=0.6, seed=42)

final = {
    'train': splits['train'],
    'test': splits['test']
}
train_ds = final["train"].shuffle(seed=42).select(range(8 * 2))
test_ds = final["test"]


model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    metric="confusion_matrix",
    num_iterations=20,  # Number of text pairs to generate for contrastive learning
    num_epochs=1  # Number of epochs to use for contrastive learning
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)
