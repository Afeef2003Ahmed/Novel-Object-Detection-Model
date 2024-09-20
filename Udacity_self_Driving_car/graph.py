from Dataset import load_checkpoint 
import matplotlib.pyplot as plt

loaded_checkpoint = load_checkpoint()

if loaded_checkpoint is not None:
    train_loss_values = loaded_checkpoint["epoch_losses"]
    
epochs = [i for i in range(len(train_loss_values))]


plt.plot(epochs,train_loss_values, linestyle='-', color='b', label='SDC Dataset')

plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')

plt.show()

plt.savefig('/raid/cs21resch15003/Afeef_Intern/CustomModel_Self_DrivingCars/loss_curve')

