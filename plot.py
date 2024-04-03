import matplotlib.pyplot as plt
import json

with open('loss.json', 'r') as file:
    data = json.load(file)
train_loss = list(map(float, data['train']))
val_loss = list(map(float, data['val']))
epochs = data['train_epochs']
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.show()
